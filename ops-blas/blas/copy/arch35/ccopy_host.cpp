/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "cann_ops_blas.h"
#include "ccopy_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "log/log.h"

namespace {

// The TPipe/DataCopy path has excellent bandwidth once transfers are large, but its
// queue initialization dominates the small and medium contiguous cases.  A direct
// coalesced SIMT copy avoids that fixed cost.  Keep large shapes on the MTE path.
constexpr uint32_t CCOPY_SIMT_CONTIGUOUS_MAX_N = 3 * 1024 * 1024;

aclblasStatus_t ValidateCcopyParams(const aclblasComplex* x, const aclblasComplex* y, int incx, int incy)
{
    if (x == nullptr || y == nullptr) {
        OP_LOGE("aclblasCcopy", "x/y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasCcopy", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0) {
        OP_LOGE("aclblasCcopy", "incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

bool CalculateCcopyTilingData(uint32_t n, uint32_t coreNum, int incx, int incy, CcopyTilingData& tiling)
{
    if (coreNum == 0) {
        return false;
    }

    tiling = {};
    tiling.totalN = n;
    tiling.incx = incx;
    tiling.incy = incy;

    constexpr uint32_t alignUnit = CCOPY_COMPLEX_PER_BLOCK;
    uint32_t rawPerCore = n / coreNum;
    tiling.perCoreN = (rawPerCore / alignUnit) * alignUnit;
    uint32_t leftover = n - tiling.perCoreN * coreNum;
    tiling.extraBlockCores = leftover / alignUnit;
    tiling.tailElements = leftover % alignUnit;

    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 248 * 1024; // fallback for 950PR
    if (platform != nullptr) {
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }

    constexpr uint32_t ubSafetyMargin = 256;
    uint64_t usableUb = ubSize > ubSafetyMargin ? ubSize - ubSafetyMargin : 0;
    uint32_t availableUb = static_cast<uint32_t>(std::min<uint64_t>(usableUb, std::numeric_limits<uint32_t>::max()));
    if (incx == 1 && incy == 1) {
        uint32_t largestCoreN = tiling.perCoreN;
        if (tiling.extraBlockCores > 0) {
            largestCoreN += alignUnit;
        }
        largestCoreN = std::max(largestCoreN, tiling.perCoreN + tiling.tailElements);

        uint32_t singleBufferMax = (availableUb / CCOPY_BYTES_PER_COMPLEX / alignUnit) * alignUnit;
        if (largestCoreN <= singleBufferMax) {
            // If every core fits in one UB tile, a second queue slot only shrinks the tile and can force
            // an otherwise unnecessary second MTE2/MTE3 transaction.
            tiling.queueBufferCount = 1;
            tiling.tileSize = ((largestCoreN + alignUnit - 1) / alignUnit) * alignUnit;
        } else {
            tiling.queueBufferCount = 2;
            uint32_t maxElements = availableUb / (tiling.queueBufferCount * CCOPY_BYTES_PER_COMPLEX);
            tiling.tileSize = (maxElements / alignUnit) * alignUnit;
        }
    } else {
        // Two queue slots plus pair-preserving output, Gather offsets, and offset-generation scratch.
        tiling.queueBufferCount = 2;
        uint32_t bytesPerElement = (tiling.queueBufferCount + 3) * CCOPY_BYTES_PER_COMPLEX;
        uint32_t maxElements = availableUb / bytesPerElement;
        maxElements = std::min(maxElements, CCOPY_MAX_COMPACT_BLOCKS);
        tiling.tileSize = (maxElements / CCOPY_LANES_PER_BLOCK) * CCOPY_LANES_PER_BLOCK;
    }
    if (tiling.tileSize == 0) {
        tiling.tileSize = incx == 1 && incy == 1 ? alignUnit : CCOPY_LANES_PER_BLOCK;
    }
    return true;
}

aclblasStatus_t LaunchContiguousCcopy(
    aclblasHandle_t handle, uint32_t totalN, const aclblasComplex* x, aclblasComplex* y, uint32_t aivCoreNum)
{
    uint32_t numBlocks = std::min(CeilDiv<uint32_t>(totalN, SIMT_MIN_THREAD_NUM), aivCoreNum);
    numBlocks = std::max(numBlocks, 1U);
    uint32_t threadCount =
        std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(totalN, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    OP_LOGD(
        "aclblasCcopy", "launching contiguous SIMT path: n=%u blocks=%u threads=%u", totalN, numBlocks, threadCount);
    ccopy_contiguous_simt_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(y), totalN, threadCount,
        numBlocks, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchGeneralCcopy(
    aclblasHandle_t handle, uint32_t totalN, const aclblasComplex* x, int incx, aclblasComplex* y, int incy,
    uint32_t aivCoreNum)
{
    uint32_t numBlocks = std::min(totalN, aivCoreNum);
    CcopyTilingData tiling{};
    if (!CalculateCcopyTilingData(totalN, numBlocks, incx, incy, tiling)) {
        OP_LOGE("aclblasCcopy", "cannot calculate tiling with zero blocks");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    OP_LOGD(
        "aclblasCcopy",
        "tiling: totalN=%u perCoreN=%u extra=%u tail=%u tileSize=%u queueBuffers=%u blocks=%u incx=%d incy=%d",
        tiling.totalN, tiling.perCoreN, tiling.extraBlockCores, tiling.tailElements, tiling.tileSize,
        tiling.queueBufferCount, numBlocks, tiling.incx, tiling.incy);
    ccopy_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(y), tiling, numBlocks,
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" aclblasStatus_t aclblasCcopy(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, aclblasComplex* y, int incy)
{
    if (n < 0) {
        OP_LOGE("aclblasCcopy", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (handle == nullptr) {
        OP_LOGE("aclblasCcopy", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t status = ValidateCcopyParams(x, y, incx, incy);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCcopy", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    uint32_t totalN = static_cast<uint32_t>(n);
    if (incx == 1 && incy == 1 && totalN <= CCOPY_SIMT_CONTIGUOUS_MAX_N) {
        return LaunchContiguousCcopy(handle, totalN, x, y, aivCoreNum);
    }
    return LaunchGeneralCcopy(handle, totalN, x, incx, y, incy, aivCoreNum);
}
