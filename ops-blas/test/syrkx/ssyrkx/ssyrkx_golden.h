/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSYRKX_GOLDEN_H
#define SSYRKX_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

static inline aclblasStatus_t ValidateSsyrkxCpuPointers(
    int k, const float* alpha, const float* beta,
    const float* A, const float* B,
    float* goldenC, const float* originalC)
{
    if (alpha == nullptr || beta == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (goldenC == nullptr || originalC == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (k > 0 && (A == nullptr || B == nullptr)) return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

static inline aclblasStatus_t ValidateSsyrkxCpuLeadingDims(
    aclblasOperation_t trans, int n, int k, int lda, int ldb, int ldc)
{
    if (trans == ACLBLAS_OP_N) {
        if (lda < std::max(1, n) || ldb < std::max(1, n)) return ACLBLAS_STATUS_INVALID_VALUE;
    } else {
        if (lda < std::max(1, k) || ldb < std::max(1, k)) return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, n)) return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

static inline aclblasStatus_t ValidateSsyrkxCpuParams(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta,
    float* goldenC, const float* originalC, int ldc)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n == 0) return ACLBLAS_STATUS_SUCCESS;
    if (n < 0 || k < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) return ACLBLAS_STATUS_INVALID_VALUE;
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) return ACLBLAS_STATUS_INVALID_VALUE;
    aclblasStatus_t st = ValidateSsyrkxCpuPointers(k, alpha, beta, A, B, goldenC, originalC);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    return ValidateSsyrkxCpuLeadingDims(trans, n, k, lda, ldb, ldc);
}

static inline void RestoreNonUploTriangle(
    aclblasFillMode_t uplo, int n, int ldc, float* goldenC, const float* originalC)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            bool isUplo = (uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (!isUplo) {
                size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(ldc);
                goldenC[idx] = originalC[idx];
            }
        }
    }
}

inline aclblasStatus_t aclblasSsyrkx_cpu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const float* alpha,
    const float* A, int lda, const float* B, int ldb,
    const float* beta, float* goldenC, int ldc,
    const float* originalC)
{
    aclblasStatus_t st = ValidateSsyrkxCpuParams(handle, uplo, trans, n, k,
        alpha, A, lda, B, ldb, beta, goldenC, originalC, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }

    if (trans == ACLBLAS_OP_C) {
        trans = ACLBLAS_OP_T;
    }

    if (k == 0) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                bool isUplo = (uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
                if (isUplo) {
                    size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(ldc);
                    goldenC[idx] = (*beta) * originalC[idx];
                }
            }
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    CBLAS_TRANSPOSE opA = (trans == ACLBLAS_OP_T) ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE opB = (trans == ACLBLAS_OP_T) ? CblasNoTrans : CblasTrans;

    cblas_sgemm(CblasColMajor, opA, opB, n, n, k,
                *alpha, A, lda, B, ldb, *beta, goldenC, ldc);

    RestoreNonUploTriangle(uplo, n, ldc, goldenC, originalC);

    return ACLBLAS_STATUS_SUCCESS;
}

#endif // SSYRKX_GOLDEN_H
