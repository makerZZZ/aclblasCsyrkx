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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline aclblasStatus_t aclblasStrmv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* a, int lda, float* x, int incx)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) return ACLBLAS_STATUS_INVALID_VALUE;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) return ACLBLAS_STATUS_INVALID_VALUE;
    if (diag != ACLBLAS_UNIT && diag != ACLBLAS_NON_UNIT) return ACLBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, n)) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (a == nullptr || x == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return ACLBLAS_STATUS_SUCCESS;

    cblas_strmv(CblasColMajor, ToCblasUplo(uplo), ToCblasOp(trans), ToCblasDiag(diag), n, a, lda, x, incx);
    return ACLBLAS_STATUS_SUCCESS;
}
