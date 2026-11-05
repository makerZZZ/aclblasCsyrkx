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
#include <cmath>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline aclblasStatus_t ValidateTrsvCpuParams(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int64_t n,
    int64_t lda, int64_t incx, const float* A, const float* x)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || lda < std::max(int64_t(1), n))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (A == nullptr || x == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasStrsv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int64_t n,
    const float* A, int64_t lda, float* x, int64_t incx)
{
    aclblasStatus_t st = ValidateTrsvCpuParams(handle, uplo, trans, diag, n, lda, incx, A, x);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    if (n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    cblas_strsv(CblasColMajor, ToCblasUplo(uplo), ToCblasOp(trans), ToCblasDiag(diag),
                static_cast<int>(n), A, static_cast<int>(lda), x, static_cast<int>(incx));
    return ACLBLAS_STATUS_SUCCESS;
}

