/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STBSV_GOLDEN_H
#define STBSV_GOLDEN_H

#include <climits>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline aclblasStatus_t aclblasStbsv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int k, const float* A, int lda, float* x, int incx)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || k < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda <= k) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incx == INT_MIN) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr || x == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (diag != ACLBLAS_NON_UNIT && diag != ACLBLAS_UNIT) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    cblas_stbsv(CblasColMajor, ToCblasUplo(uplo), ToCblasOp(trans), ToCblasDiag(diag),
                n, k, A, lda, x, incx);

    return ACLBLAS_STATUS_SUCCESS;
}

#endif // STBSV_GOLDEN_H
