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
 * \file axpy_ex_host.cpp
 * \brief Host-side implementation for aclblasAxpyEx: y = alpha * x + y.
 *        Dual-path dispatch: SIMD (incx==1 && incy==1) or SIMT (strided).
 *        Supports FP32/FP16/BF16 xTypes with FP32 compute precision.
 *        Alpha can be in Host or Device memory (detected via aclrtPointerGetAttributes).
 */

#include <algorithm>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "axpy_ex_tiling_data.h"
#include "axpy_ex_kernel.h"

static uint32_t GetBytePerElement(uint32_t xType)
{
    constexpr uint32_t kFp16Bf16Size = 2;
    constexpr uint32_t kFloatSize = sizeof(float);

    if (xType == static_cast<uint32_t>(ACL_FLOAT)) {
        return 2 * kFloatSize;
    }
    return 2 * kFp16Bf16Size;
}

static uint32_t GetAlignUnit(uint32_t xType)
{
    if (xType == static_cast<uint32_t>(ACL_FLOAT)) {
        return 32 / sizeof(float); // 8
    }
    return 32 / 2;
}

static aclblasStatus_t ValidateAxpyExParams(
    const void* alpha, aclDataType alphaType, const void* x, aclDataType xType, int incx, void* y, aclDataType yType,
    int incy, aclDataType executionType)
{
    if (alphaType != ACL_FLOAT) {
        OP_LOGE("aclblasAxpyEx", "alphaType must be ACL_FLOAT(0), got %d", static_cast<int>(alphaType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (executionType != ACL_FLOAT) {
        OP_LOGE("aclblasAxpyEx", "executionType must be ACL_FLOAT(0), got %d", static_cast<int>(executionType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (xType != ACL_FLOAT16 && xType != ACL_BF16 && xType != ACL_FLOAT) {
        OP_LOGE(
            "aclblasAxpyEx", "xType must be ACL_FLOAT16(1), ACL_BF16(27) or ACL_FLOAT(0), got %d",
            static_cast<int>(xType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (yType != ACL_FLOAT16 && yType != ACL_BF16 && yType != ACL_FLOAT) {
        OP_LOGE(
            "aclblasAxpyEx", "yType must be ACL_FLOAT16(1), ACL_BF16(27) or ACL_FLOAT(0), got %d",
            static_cast<int>(yType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (xType != yType) {
        OP_LOGE("aclblasAxpyEx", "xType(%d) must equal yType(%d)", static_cast<int>(xType), static_cast<int>(yType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclblasAxpyEx", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasAxpyEx", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (y == nullptr) {
        OP_LOGE("aclblasAxpyEx", "y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasAxpyEx", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0) {
        OP_LOGE("aclblasAxpyEx", "incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static AxpyExTilingData CalAxpyExTilingDataContiguous(uint32_t totalN, uint32_t coreNum, uint32_t xType)
{
    AxpyExTilingData tiling{};
    tiling.totalN = totalN;
    tiling.incx = 1;
    tiling.incy = 1;
    tiling.startOffsetX = 0;
    tiling.startOffsetY = 0;
    tiling.xType = xType;

    uint32_t alignUnit = GetAlignUnit(xType);
    if (coreNum == 0 || alignUnit == 0) {
        OP_LOGE("aclblasAxpyEx", "invalid coreNum=%u or alignUnit=%u", coreNum, alignUnit);
        return tiling;
    }

    uint32_t rawPerCore = totalN / coreNum;
    tiling.perCoreN = (rawPerCore / alignUnit) * alignUnit;
    tiling.remainder = totalN - tiling.perCoreN * coreNum;

    uint32_t bytesPerElement = GetBytePerElement(xType);
    uint32_t maxElements = UB_SIZE / bytesPerElement;
    uint32_t alignedMax = (maxElements / alignUnit) * alignUnit;
    tiling.tileSize = (alignedMax > alignUnit) ? (alignedMax - alignUnit) : alignedMax;

    if (xType != static_cast<uint32_t>(ACL_FLOAT)) {
        constexpr uint32_t VF_VL = 64;
        tiling.tileSize = (tiling.tileSize / VF_VL) * VF_VL;
    }

    return tiling;
}

static AxpyExTilingData CalAxpyExTilingDataStrided(
    int64_t n, int64_t incx, int64_t incy, uint32_t numBlocks, uint32_t xType)
{
    AxpyExTilingData tiling{};
    tiling.totalN = static_cast<uint32_t>(n);
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.xType = xType;
    tiling.numBlocks = numBlocks;

    tiling.startOffsetX = (incx >= 0) ? 0 : (1 - n) * incx;
    tiling.startOffsetY = (incy >= 0) ? 0 : (1 - n) * incy;

    if (numBlocks == 0) {
        OP_LOGE("aclblasAxpyEx", "numBlocks is 0");
        return tiling;
    }
    uint32_t totalN = static_cast<uint32_t>(n);
    uint32_t avgElements = totalN / numBlocks;
    uint32_t alignedDown = (avgElements / SIMT_MIN_THREAD_NUM) * SIMT_MIN_THREAD_NUM;
    tiling.nthreads = std::max(alignedDown, SIMT_MIN_THREAD_NUM);
    tiling.nthreads = std::min(tiling.nthreads, SIMT_MAX_THREAD_NUM);

    return tiling;
}

static aclblasStatus_t LaunchAxpyExKernel(
    int n, int incx, int incy, uint32_t aivCoreNum, const void* alpha, bool alphaIsDevice, uint32_t xType, void* x,
    void* y, aclrtStream stream)
{
    AxpyExTilingData tiling{};
    uint32_t numBlocks;
    bool isSimdPath = (incx == 1 && incy == 1);

    if (isSimdPath) {
        uint32_t totalN = static_cast<uint32_t>(n);
        constexpr uint32_t MIN_ELEMS_PER_CORE = 256;
        numBlocks = std::min(CeilDiv(totalN, MIN_ELEMS_PER_CORE), aivCoreNum);
        numBlocks = std::max(numBlocks, 1u);
        tiling = CalAxpyExTilingDataContiguous(totalN, numBlocks, xType);
        if (tiling.tileSize == 0) {
            OP_LOGE("aclblasAxpyEx", "SIMD tiling tileSize is 0, totalN=%u numBlocks=%u", totalN, numBlocks);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    } else {
        numBlocks = std::min(CeilDiv<uint32_t>(static_cast<uint32_t>(n), SIMT_MIN_THREAD_NUM), aivCoreNum);
        tiling = CalAxpyExTilingDataStrided(n, incx, incy, numBlocks, xType);
        if (tiling.nthreads == 0) {
            OP_LOGE("aclblasAxpyEx", "SIMT tiling nthreads is 0, n=%d numBlocks=%u", n, numBlocks);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
    }

    // Alpha Host/Device handling
    tiling.alphaIsDevice = alphaIsDevice ? 1u : 0u;
    if (!alphaIsDevice) {
        tiling.alpha = *reinterpret_cast<const float*>(alpha);
    } else {
        tiling.alpha = 0.0f; // Kernel reads from GM when alphaIsDevice==1
    }

    OP_LOGD(
        "aclblasAxpyEx",
        "tiling: totalN=%u perCoreN=%u remainder=%u tileSize=%u incx=%lld incy=%lld "
        "startOffsetX=%lld startOffsetY=%lld nthreads=%u numBlocks=%u alpha=%.7g "
        "alphaIsDevice=%u xType=%u",
        tiling.totalN, tiling.perCoreN, tiling.remainder, tiling.tileSize, static_cast<long long>(tiling.incx),
        static_cast<long long>(tiling.incy), static_cast<long long>(tiling.startOffsetX),
        static_cast<long long>(tiling.startOffsetY), tiling.nthreads, tiling.numBlocks, tiling.alpha,
        tiling.alphaIsDevice, tiling.xType);

    OP_LOGI(
        "aclblasAxpyEx",
        "launching %s kernel: blocks=%u, cores=%u, incx=%d, incy=%d, "
        "alphaIsDevice=%u, xType=%u",
        isSimdPath ? "SIMD" : "SIMT", numBlocks, aivCoreNum, incx, incy, tiling.alphaIsDevice, xType);

    void* alphaPtr = alphaIsDevice ? const_cast<void*>(alpha) : nullptr;

    axpy_ex_kernel_do(
        reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(y), alphaPtr, tiling, numBlocks, stream);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasAxpyEx(
    aclblasHandle_t handle, int n, const void* alpha, aclDataType alphaType, const void* x, aclDataType xType, int incx,
    void* y, aclDataType yType, int incy, aclDataType executionType)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasAxpyEx", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    if (n < 0) {
        OP_LOGE("aclblasAxpyEx", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t status = ValidateAxpyExParams(alpha, alphaType, x, xType, incx, y, yType, incy, executionType);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasAxpyEx", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    auto* h = handle;
    aclrtPtrAttributes ptrAttr{};
    aclError aclRet = aclrtPointerGetAttributes(alpha, &ptrAttr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasAxpyEx", "aclrtPointerGetAttributes for alpha failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    bool alphaIsDevice = (ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);

    return LaunchAxpyExKernel(
        n, incx, incy, aivCoreNum, alpha, alphaIsDevice, static_cast<uint32_t>(xType), const_cast<void*>(x), y,
        h->stream);
}
