/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSYMM_NPU_H
#define CSYMM_NPU_H

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "acl/acl.h"
#include "cann_ops_blas.h"

static inline bool CsymmTimingEnabled()
{
    static const bool on = (std::getenv("CSYMM_TIMING") != nullptr);
    return on;
}

static inline void CsymmTimingMark(const char* tag)
{
    if (!CsymmTimingEnabled()) return;
    static double last = 0.0;
    const double now = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    printf("[TIMING] %-14s +%-9.1fus\n", tag, now - last);
    fflush(stdout);
    last = now;
}

static inline aclError CsymmAllocCopyH2D(void*& dPtr, const void* hPtr, size_t bytes)
{
    dPtr = nullptr;
    if (hPtr == nullptr) {
        return ACL_SUCCESS;
    }
    CsymmTimingMark("h2d_alloc");
    const aclError ret = aclrtMalloc(&dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    const aclError cpRet = aclrtMemcpy(dPtr, bytes, hPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (cpRet != ACL_SUCCESS) {
        (void)aclrtFree(dPtr);
        dPtr = nullptr;
    }
    CsymmTimingMark("h2d_copy");
    return cpRet;
}

static inline void CsymmFreeDev(void* dPtr)
{
    if (dPtr != nullptr) {
        (void)aclrtFree(dPtr);
    }
}

static inline void CsymmCopyResultD2H(
    aclblasHandle handle, void* hPtr, void* dPtr, size_t bytes, aclblasStatus_t& ret, bool copyBack)
{
    if (ret != ACLBLAS_STATUS_SUCCESS || hPtr == nullptr || dPtr == nullptr) {
        return;
    }
    aclrtStream stream = nullptr;
    aclblasGetStream(handle, &stream);
    if (stream != nullptr) {
        (void)aclrtSynchronizeStream(stream);
    }
    CsymmTimingMark("kernel_sync");
    // The stream drain is unconditional so the reported GTest wall time always
    // includes the kernels. The D2H copy is skippable: TC_PF_* perf rows never
    // verify C, so copying 8MB back to host only inflates their wall time.
    if (!copyBack) {
        return;
    }
    if (aclrtMemcpy(hPtr, bytes, dPtr, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        ret = ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    CsymmTimingMark("d2h_copy");
}

// Copy one operand H2D when its host source is non-null; returns the device
// status (SUCCESS or ALLOC_FAILED) without early-returning the caller.
static inline aclblasStatus_t CsymmCopyCachedOne(void* d, const void* src, size_t bytes)
{
    if (src == nullptr) return ACLBLAS_STATUS_SUCCESS;
    return (aclrtMemcpy(d, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS)
               ? ACLBLAS_STATUS_SUCCESS
               : ACLBLAS_STATUS_ALLOC_FAILED;
}

// Alloc/copy the A/B/C operands to device tensors (with an optional per-process
// cache, CSYMM_CACHE_BUFS=1, used only as a diagnostic). Returns
// ACLBLAS_STATUS_SUCCESS on success or ACLBLAS_STATUS_ALLOC_FAILED.
static inline aclblasStatus_t CsymmPrepareDeviceTensors(
    void*& dA, void*& dB, void*& dC,
    const void* A, const void* B, const void* C,
    size_t aBytes, size_t bBytes, size_t cBytes, bool cacheBufs)
{
    static void* cachedA = nullptr;
    static void* cachedB = nullptr;
    static void* cachedC = nullptr;
    static size_t cachedSize = 0;
    if (cacheBufs) {
        if (cachedSize < aBytes) {
            if (cachedA != nullptr) aclrtFree(cachedA);
            aclrtMalloc(&cachedA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (cachedSize < bBytes) {
            if (cachedB != nullptr) aclrtFree(cachedB);
            aclrtMalloc(&cachedB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (cachedSize < cBytes) {
            if (cachedC != nullptr) aclrtFree(cachedC);
            aclrtMalloc(&cachedC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        cachedSize = std::max(cachedSize, std::max(aBytes, std::max(bBytes, cBytes)));
        dA = cachedA;
        dB = cachedB;
        dC = cachedC;
        aclblasStatus_t s = CsymmCopyCachedOne(dA, A, aBytes);
        if (s != ACLBLAS_STATUS_SUCCESS) return s;
        s = CsymmCopyCachedOne(dB, B, bBytes);
        if (s != ACLBLAS_STATUS_SUCCESS) return s;
        s = CsymmCopyCachedOne(dC, C, cBytes);
        if (s != ACLBLAS_STATUS_SUCCESS) return s;
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError aclRet = CsymmAllocCopyH2D(dA, A, aBytes);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = CsymmAllocCopyH2D(dB, B, bBytes);
    if (aclRet != ACL_SUCCESS) {
        CsymmFreeDev(dA);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (C != nullptr) {
        aclRet = CsymmAllocCopyH2D(dC, C, cBytes);
        if (aclRet != ACL_SUCCESS) {
            CsymmFreeDev(dA);
            CsymmFreeDev(dB);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

/*!
 * \brief Run aclblasCsymm on the device with host-resident operands.
 *
 * Cases that only exercise parameter validation (null handle / null pointers / zero
 * dimensions) are forwarded to the operator directly so the host-side checks are covered
 * without any device traffic.
 */
inline aclblasStatus_t aclblasCsymm_npu(
    aclblasHandle handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    int m,
    int n,
    const aclblasComplex* alpha,
    const aclblasComplex* A,
    int lda,
    const aclblasComplex* B,
    int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C,
    int ldc,
    bool copyBack = true)
{
    if (handle == nullptr || m <= 0 || n <= 0 || alpha == nullptr || A == nullptr || B == nullptr ||
        beta == nullptr) {
        const aclblasStatus_t ret = aclblasCsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
        // alpha == 0 still launches an asynchronous C-scaling kernel, so the caller (which
        // inspects the host buffer right away) needs the stream drained.
        aclrtStream stream = nullptr;
        if (handle != nullptr && aclblasGetStream(handle, &stream) == ACLBLAS_STATUS_SUCCESS && stream != nullptr) {
            (void)aclrtSynchronizeStream(stream);
        }
        return ret;
    }

    const int aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    const size_t aBytes = static_cast<size_t>(aDim) * static_cast<size_t>(lda) * sizeof(aclblasComplex);
    const size_t bBytes = static_cast<size_t>(ldb) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    const size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    const bool cacheBufs = (std::getenv("CSYMM_CACHE_BUFS") != nullptr);
    aclblasStatus_t ret = CsymmPrepareDeviceTensors(dA, dB, dC, A, B, C, aBytes, bBytes, cBytes, cacheBufs);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    if (std::getenv("CSYMM_DUMP_WANY") != nullptr) {
        printf("[DBG] test bufs: dA=%p dB=%p dC=%p (aBytes=%zu bBytes=%zu cBytes=%zu)\n",
            dA, dB, dC, aBytes, bBytes, cBytes);
    }

    ret = aclblasCsymm(handle, side, uplo, m, n,
        alpha, static_cast<const aclblasComplex*>(dA), lda,
        static_cast<const aclblasComplex*>(dB), ldb,
        beta, static_cast<aclblasComplex*>(dC), ldc);

    CsymmCopyResultD2H(handle, C, dC, cBytes, ret, copyBack);

    if (!cacheBufs) {
        CsymmFreeDev(dA);
        CsymmFreeDev(dB);
        CsymmFreeDev(dC);
    }
    return ret;
}

#endif // CSYMM_NPU_H
