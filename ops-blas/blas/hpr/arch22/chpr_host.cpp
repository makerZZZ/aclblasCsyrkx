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
#include <climits>
#include <cstdint>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "chpr_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

static aclblasStatus_t ValidateChprParams(
    aclblasFillMode_t uplo, int incx, const float* alpha, const aclblasComplex* x, const aclblasComplex* ap)
{
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasChpr", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasChpr", "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasChpr", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != INT_MIN, OP_LOGE("aclblasChpr", "incx must not be INT_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(x != nullptr, OP_LOGE("aclblasChpr", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ap != nullptr, OP_LOGE("aclblasChpr", "ap must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchChprKernel(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* ap)
{
    auto* h = handle;
    float alphaVal = *alpha;
    if (alphaVal == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasChpr", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    uint32_t nU32 = static_cast<uint32_t>(n);
    uint32_t useCoreNum = std::min(nU32, aivCoreNum);
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }

    ChprTilingData tiling{};
    tiling.n = nU32;
    tiling.useCoreNum = useCoreNum;
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.incx = static_cast<int64_t>(incx);
    tiling.alpha = alphaVal;

    OP_LOGD(
        "aclblasChpr", "tiling: n=%u, uplo=%u, incx=%ld, cores=%u, alpha=%f", tiling.n, tiling.uplo,
        static_cast<long>(tiling.incx), tiling.useCoreNum, tiling.alpha);
    OP_LOGI("aclblasChpr", "launching kernel: blocks=%u, cores=%u", useCoreNum, aivCoreNum);

    chpr_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(ap), tiling, useCoreNum,
        h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasChpr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* ap)
{
    CHECK_RET(handle != nullptr, OP_LOGE("aclblasChpr", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(n >= 0, OP_LOGE("aclblasChpr", "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t st = ValidateChprParams(uplo, incx, alpha, x, ap);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return LaunchChprKernel(handle, uplo, n, alpha, x, incx, ap);
}
