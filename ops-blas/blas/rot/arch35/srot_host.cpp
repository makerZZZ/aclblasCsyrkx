/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file srot_host.cpp
 * \brief Host-side API for BLAS Level-1 Givens plane rotation: aclblasSrot.
 *        Arch35 (ascend950) implementation.
 *
 *        Applies the Givens rotation to vectors x and y in place:
 *            x[i] = c*x[i] + s*y[i]
 *            y[i] = c*y[i] - s*x[i]
 *
 *        Two compute paths selected by contiguity of the strides:
 *          - incx==1 && incy==1 : contiguous path, SIMD membase kernel
 *          - otherwise          : strided path, SIMT kernel (grid-stride loop)
 *
 *        netlib srot.f boundary alignment: n<=0 short-circuits to SUCCESS;
 *        zero / negative strides and the identity rotation (c==1 && s==0) are
 *        intentionally not intercepted, matching the reference behavior.
 */

#include <cstdint>
#include <climits>
#include <algorithm>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "srot_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"

static aclblasStatus_t ValidateSrotParams(const float* x, const float* y, const float* c, const float* s)
{
    if (x == nullptr || y == nullptr) {
        OP_LOGE("aclblasSrot", "x/y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (c == nullptr || s == nullptr) {
        OP_LOGE("aclblasSrot", "c/s must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Pointer location query: determine whether a c/s scalar pointer lives on the
// host or on the device (NPU HBM). Used to select the host-dereference path vs
// the GM-forwarding path. Mirrors the aclrtPointerGetAttributes usage in rotg
// and scalex. On query failure the call is rejected (consistent with scalex).
// ==========================================================================
static aclblasStatus_t SrotCheckPtrLocation(const void* ptr, bool* isDevice)
{
    aclrtPtrAttributes ptrAttr{};
    aclError aclRet = aclrtPointerGetAttributes(ptr, &ptrAttr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSrot", "aclrtPointerGetAttributes failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *isDevice = (ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Tiling computation for the contiguous path (SIMD membase).
// The contiguous kernel uses a TQue three-stage pipeline with five UB tiles:
//   inQueueX / inQueueY (VECIN, MTE2 input)
//   outNewX / outNewY   (VECOUT, Vector output)
//   workBuf             (VECCALC, pure-Vector scratch for round(c*y))
// Each tile is tileSize elements, so UB must hold bufferCount * tileSize floats.
// ==========================================================================
static void CalSrotTilingContiguous(uint32_t n, uint32_t numBlocks, SrotTilingData& tiling)
{
    // Even split (no alignment rounding): perCoreN = n/numBlocks base elements per
    // core, remainder = n%numBlocks front cores each get one extra element. Unlike
    // the old align-down scheme, no core starves when n/numBlocks < ELEMENTS_PER_BLOCK:
    // the load stays balanced down to n < numBlocks. The contiguous kernel copies with
    // pure DataCopyPad, whose GM side requires only 1-byte alignment, so the
    // non-block-aligned per-core start offset and element count are both safe.
    tiling.perCoreN = n / numBlocks;
    tiling.remainder = n % numBlocks;

    // Five UB tiles: inQueueX + inQueueY + outNewX + outNewY + workBuf.
    // tileSize is the per-copy UB cap; kept block-aligned for UB buffer allocation.
    constexpr uint32_t alignUnit = SROT_ELEMENTS_PER_BLOCK;
    constexpr uint32_t bufferCount = 5;
    uint32_t maxElements = UB_SIZE / (bufferCount * sizeof(float)); // UB_SIZE from kernel_constant.h
    tiling.tileSize = (maxElements / alignUnit) * alignUnit;
}

// ==========================================================================
// Tiling computation for the strided path (SIMT).
// The kernel handles negative strides internally (negX/negY flags + (n-1-i)*absInc),
// so it consumes only incx/incy from the tiling. The (n-1)*stride product is still
// computed here in int64_t and range-checked before narrowing: if it exceeds the
// int32_t range, the kernel-side int32_t offset would overflow (risk R5), so the
// call is rejected with an explicit error instead of launching a broken kernel.
// The narrowed last offsets are returned via outLastX/outLastY for host-side
// diagnostics (OP_LOGD) only; they are NOT consumed by the kernel.
//
// nthreads: SIMT threads per block. The grid-stride loop only needs as many
// threads as the per-core element count justifies, so this is sized from
// ceilDiv(n,numBlocks) — rounded UP to SIMT_MIN_THREAD_NUM (the hardware's thread
// scheduling granularity) and capped at SIMT_MAX_THREAD_NUM — instead of always
// launching SIMT_MAX_THREAD_NUM threads. A small n (e.g. n=10, 1 block) thus
// launches 128 threads rather than 2048, eliminating the wasted-thread startup
// cost while large n still saturates the full 2048 per block.
// ==========================================================================
static aclblasStatus_t CalSrotTilingStrided(uint32_t n, int incx, int incy, uint32_t numBlocks,
                                            SrotTilingData& tiling, int32_t* outLastX, int32_t* outLastY)
{
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.nthreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(n, numBlocks), SIMT_MIN_THREAD_NUM),
        SIMT_MAX_THREAD_NUM);
    int64_t lastX = static_cast<int64_t>(n - 1) * static_cast<int64_t>(incx);
    int64_t lastY = static_cast<int64_t>(n - 1) * static_cast<int64_t>(incy);
    if (lastX > INT32_MAX || lastX < INT32_MIN || lastY > INT32_MAX || lastY < INT32_MIN) {
        OP_LOGE("aclblasSrot", "strided offset overflow: n=%u incx=%d incy=%d lastX=%lld lastY=%lld",
                static_cast<unsigned>(n), incx, incy, static_cast<long long>(lastX), static_cast<long long>(lastY));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *outLastX = static_cast<int32_t>(lastX);
    *outLastY = static_cast<int32_t>(lastY);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Resolve the c/s source: query each pointer's location, fill the corresponding
// TilingData fields, and produce the float* pointers to forward to the kernel.
//   host pointer  -> dereference once, store the scalar, forward nullptr
//   device pointer-> store 0.0f placeholder, forward the pointer verbatim
// The two pointers are independent (c on host + s on device is legal). Device
// pointers are forwarded as-is with no dereference, no D2H copy, no stream sync.
// ==========================================================================
static aclblasStatus_t PrepareSrotCsSource(const float* c, const float* s, SrotTilingData& tiling, float** cPtr,
                                           float** sPtr)
{
    bool cIsDevice = false;
    bool sIsDevice = false;
    aclblasStatus_t cLocSt = SrotCheckPtrLocation(c, &cIsDevice);
    if (cLocSt != ACLBLAS_STATUS_SUCCESS) {
        return cLocSt;
    }
    aclblasStatus_t sLocSt = SrotCheckPtrLocation(s, &sIsDevice);
    if (sLocSt != ACLBLAS_STATUS_SUCCESS) {
        return sLocSt;
    }

    tiling.cIsDevice = cIsDevice ? 1u : 0u;
    tiling.sIsDevice = sIsDevice ? 1u : 0u;
    tiling.cosValue = cIsDevice ? 0.0f : (*c);
    tiling.sinValue = sIsDevice ? 0.0f : (*s);
    // Device path forwards the pointer verbatim; host path forwards nullptr (kernel
    // reads the tiling scalar instead).
    *cPtr = cIsDevice ? const_cast<float*>(c) : nullptr;
    *sPtr = sIsDevice ? const_cast<float*>(s) : nullptr;
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Launch — tiling computation + asynchronous kernel launch
// ==========================================================================
static aclblasStatus_t LaunchSrotKernel(aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy,
                                        const float* c, const float* s)
{
    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSrot", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t totalN = static_cast<uint32_t>(n);
    bool contiguous = (incx == 1 && incy == 1);

    SrotTilingData tiling{};
    tiling.totalN = totalN;

    // Resolve c/s source (host scalar vs device GM pointer); fills tiling.cosValue/
    // sinValue/cIsDevice/sIsDevice and yields the pointers to forward.
    float* cPtr = nullptr;
    float* sPtr = nullptr;
    aclblasStatus_t csSt = PrepareSrotCsSource(c, s, tiling, &cPtr, &sPtr);
    if (csSt != ACLBLAS_STATUS_SUCCESS) {
        return csSt;
    }

    // lastX/lastY: strided-path tail offset ((n-1)*stride), host-side diagnostic only.
    // Contiguous path leaves them at 0 (no stride offset); the kernel never reads them.
    int32_t lastX = 0;
    int32_t lastY = 0;
    uint32_t numBlocks;
    if (contiguous) {
        tiling.tilingKey = 0;
        numBlocks = (totalN < aivCoreNum) ? totalN : aivCoreNum;
        CalSrotTilingContiguous(totalN, numBlocks, tiling);
    } else {
        tiling.tilingKey = 1;
        uint32_t blocksByThreads = CeilDiv<uint32_t>(totalN, SIMT_MIN_THREAD_NUM);
        numBlocks = (blocksByThreads < aivCoreNum) ? blocksByThreads : aivCoreNum;
        if (numBlocks == 0) {
            numBlocks = 1;
        }
        aclblasStatus_t stridedSt = CalSrotTilingStrided(totalN, incx, incy, numBlocks, tiling, &lastX, &lastY);
        if (stridedSt != ACLBLAS_STATUS_SUCCESS) {
            return stridedSt;
        }
    }

    OP_LOGD("aclblasSrot",
            "tiling: key=%u totalN=%u incx=%d incy=%d lastX=%d lastY=%d c=%f s=%f cIsDevice=%u sIsDevice=%u "
            "perCoreN=%u remainder=%u tileSize=%u numBlocks=%u nthreads=%u",
            tiling.tilingKey, tiling.totalN, incx, incy, lastX, lastY, tiling.cosValue,
            tiling.sinValue, tiling.cIsDevice, tiling.sIsDevice, tiling.perCoreN, tiling.remainder, tiling.tileSize,
            numBlocks, tiling.nthreads);
    OP_LOGI("aclblasSrot", "launching kernel: key=%u blocks=%u cIsDevice=%u sIsDevice=%u", tiling.tilingKey,
            numBlocks, tiling.cIsDevice, tiling.sIsDevice);

    // In-place operation: no workspace needed. Pass nullptr; no aclrtMalloc here.
    srot_kernel_do(x, y, cPtr, sPtr, numBlocks, tiling, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Public API entry — dispatch only (validate then launch)
// ==========================================================================
extern "C" aclblasStatus_t aclblasSrot(aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy,
                                       const float* c, const float* s)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSrot", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateSrotParams(x, y, c, s);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    return LaunchSrotKernel(handle, n, x, incx, y, incy, c, s);
}