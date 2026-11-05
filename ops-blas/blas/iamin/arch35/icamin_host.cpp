/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <climits>
#include "acl/acl.h"
#include <algorithm>
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "common/helper/kernel_constant.h"
#include "icamin_kernel.h"
#include "icamin_tiling_data.h"

namespace {

static aclblasStatus_t ValidateIcaminParams(
    aclblasHandle_t handle, int n, int incx, const aclblasComplex* x, int* result)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasIcamin", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasIcamin", "n must not be negative, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasIcamin", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || incx < 1) {
        return ACLBLAS_STATUS_SUCCESS; // caller writes device result via WriteQuickReturn
    }
    if (x == nullptr) {
        OP_LOGE("aclblasIcamin", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Reject strides whose last complex index (n-1)*incx exceeds int32 range so
    // host-side and SIMT address math cannot wrap (complex float needs *2).
    const int64_t lastComplex = static_cast<int64_t>(n - 1) * static_cast<int64_t>(incx);
    if (lastComplex < 0 || lastComplex > static_cast<int64_t>(INT32_MAX) ||
        lastComplex > (static_cast<int64_t>(INT64_MAX) / 2)) {
        OP_LOGE(
            "aclblasIcamin", "strided offset overflow: n=%d incx=%d lastComplex=%lld", n, incx,
            static_cast<long long>(lastComplex));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static IcaminTilingData CalcIcaminTiling(int n, uint32_t numBlocks, int incx)
{
    IcaminTilingData tiling{};
    uint32_t totalN = static_cast<uint32_t>(n);
    tiling.totalN = totalN;
    tiling.incx = static_cast<uint32_t>(incx);

    // Callers guarantee numBlocks >= 1 (validation returned early for n == 0 and
    // numBlocks = min(totalN, aivCoreNum) with aivCoreNum >= 1). Clamp the local
    // copy defensively so the divisors below can never be zero.
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    if (incx == 1) {
        // Vector path: every per-core slice must start at a 32B-aligned GM address
        // and have a 32B-multiple length so the bulk DataCopy of a full tile is
        // always aligned (4 complex = 8 float = 32B). Ceil-aligning the slice can
        // only shrink the block count, never grow it.
        constexpr uint32_t ALIGN_COMPLEX = 4;
        uint32_t perCoreN = CeilAlign<uint32_t>((totalN + numBlocks - 1) / numBlocks, ALIGN_COMPLEX);
        uint32_t useCoreNum = (totalN + perCoreN - 1) / perCoreN;
        tiling.perCoreN = perCoreN;
        tiling.useCoreNum = useCoreNum;
        tiling.lastCoreN = totalN - (useCoreNum - 1) * perCoreN;
    } else {
        // General path: SIMT scan reads GM with scalar loads, any alignment is fine.
        tiling.perCoreN = totalN / numBlocks;
        tiling.lastCoreN = tiling.perCoreN + (totalN % numBlocks);
        tiling.useCoreNum = numBlocks;
    }

    tiling.tileSize = ICAMIN_TILE_COMPLEX;

    // The scalar SIMT kernel runs this many threads. It serves the strided scan
    // and the unit-stride NaN fallback, so it is needed for every stride: aim for
    // one complex element per thread (128-aligned), capped at SIMT_MAX_THREAD_NUM.
    tiling.nthreads = std::min(CeilAlign<uint32_t>(tiling.perCoreN, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);

    return tiling;
}

static aclblasStatus_t WriteQuickReturn(aclblasHandle_t handle, int* result)
{
    const aclError status = aclrtMemsetAsync(result, sizeof(int), 0, sizeof(int), handle->stream);
    if (status != ACL_SUCCESS) {
        OP_LOGE("aclblasIcamin", "failed to write quick-return result, ret=%d", status);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchIcaminKernel(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)
{
    aclblasStatus_t vStatus = ValidateIcaminParams(handle, n, incx, x, result);
    if (vStatus != ACLBLAS_STATUS_SUCCESS) {
        return vStatus;
    }
    if (n <= 0 || incx < 1) {
        return WriteQuickReturn(handle, result);
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasIcamin", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t numBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);

    IcaminTilingData tiling = CalcIcaminTiling(n, numBlocks, incx);

    OP_LOGD(
        "aclblasIcamin", "tiling: totalN=%u perCoreN=%u lastCoreN=%u useCoreNum=%u tileSize=%u nthreads=%u incx=%d",
        tiling.totalN, tiling.perCoreN, tiling.lastCoreN, tiling.useCoreNum, tiling.tileSize, tiling.nthreads, incx);

    void* workSpace = GetEffectiveWorkspace(handle);
    size_t workspaceBytes = GetEffectiveWorkspaceSize(handle);
    constexpr uint32_t ALIGN_FLOATS = 64;
    uint32_t totalFloats = numBlocks * 2;
    uint32_t alignedFloats = ((totalFloats + ALIGN_FLOATS - 1) / ALIGN_FLOATS) * ALIGN_FLOATS;
    size_t requiredBytes = static_cast<size_t>(alignedFloats) * sizeof(float);
    if (workspaceBytes < requiredBytes) {
        OP_LOGE("aclblasIcamin", "workspace too small: need %zu, got %zu", requiredBytes, workspaceBytes);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    OP_LOGI(
        "aclblasIcamin", "launching kernel: blocks=%u, useCoreNum=%u, nthreads=%u, incx=%d", numBlocks,
        tiling.useCoreNum, tiling.nthreads, incx);

    icamin_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(x)), reinterpret_cast<GM_ADDR>(result),
        reinterpret_cast<GM_ADDR>(workSpace), tiling, numBlocks, reinterpret_cast<void*>(handle->stream));

    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasIcamin(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)
{
    return LaunchIcaminKernel(handle, n, x, incx, result);
}
