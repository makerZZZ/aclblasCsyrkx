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

inline aclblasStatus_t aclblasSsbmv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (k < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (lda < k + 1)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0 || incy == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    cblas_ssbmv(CblasColMajor, ToCblasUplo(uplo), n, k, *alpha, a, lda, x, incx, *beta, y, incy);
    return ACLBLAS_STATUS_SUCCESS;
}
