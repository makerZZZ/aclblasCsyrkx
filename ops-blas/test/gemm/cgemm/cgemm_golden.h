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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline bool IsValidCgemmOperation(aclblasOperation_t operation)
{
    return operation == ACLBLAS_OP_N || operation == ACLBLAS_OP_T || operation == ACLBLAS_OP_C;
}

inline aclblasStatus_t ValidateCgemmCpuArguments(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasComplex* alpha, int lda, int ldb, const aclblasComplex* beta, int ldc)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!IsValidCgemmOperation(transA)) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (!IsValidCgemmOperation(transB)) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int minLda = transA == ACLBLAS_OP_N ? std::max(1, m) : std::max(1, k);
    const int minLdb = transB == ACLBLAS_OP_N ? std::max(1, k) : std::max(1, n);
    if (lda < minLda || ldb < minLdb || ldc < std::max(1, m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasCgemmCpu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb,
    const aclblasComplex* beta, aclblasComplex* c, int ldc)
{
    const aclblasStatus_t status =
        ValidateCgemmCpuArguments(handle, transA, transB, m, n, k, alpha, lda, ldb, beta, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (c == nullptr || (k > 0 && (a == nullptr || b == nullptr))) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    cblas_cgemm(CblasColMajor, ToCblasOp(transA), ToCblasOp(transB), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return ACLBLAS_STATUS_SUCCESS;
}
