/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline aclblasStatus_t ValidateSprParamsCpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const float* alpha, const float* x, int incx, float* ap)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return ACLBLAS_STATUS_SUCCESS;
    if (alpha == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == INT_MIN) return ACLBLAS_STATUS_INVALID_VALUE;
    if (x == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (ap == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasSspr_cpu(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    int n,
    const float* alpha,
    const float* x,
    int incx,
    float* ap)
{
    aclblasStatus_t st = ValidateSprParamsCpu(handle, uplo, n, alpha, x, incx, ap);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    if (n == 0) return ACLBLAS_STATUS_SUCCESS;

    float alphaVal = *alpha;
    if (alphaVal == 0.0f) return ACLBLAS_STATUS_SUCCESS;

    cblas_sspr(CblasColMajor, ToCblasUplo(uplo), n, alphaVal, x, incx, ap);

    return ACLBLAS_STATUS_SUCCESS;
}
