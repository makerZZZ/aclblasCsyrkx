/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGEMM3M_GOLDEN_H
#define SGEMM3M_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "sgemm3m_param.h"

// ═══════════════════════════════════════════════════════════════════════════════
// CPU golden: C = alpha * (op(A1)*op(B1) + op(A2)*op(B2) + op(A3)*op(B3)) + beta * C
//
// A contains A1/A2/A3 merged along K; B contains B1/B2/B3 merged along K.
// Sub-matrix pointers are computed inside from the merged A and B.
//
// Implements the 3M method real decomposition using 3 chained cblas_sgemm calls:
//   Step 1: C = alpha * op(A1)*op(B1) + beta * C_init
//   Step 2: C = alpha * op(A2)*op(B2) + 1.0 * C   (accumulate)
//   Step 3: C = alpha * op(A3)*op(B3) + 1.0 * C   (accumulate)
//
// Short-circuit paths (matching NPU operator semantics):
//   M=0 or N=0: return SUCCESS, C unchanged (Kernel not launched)
//   K=0 or alpha=0: C = beta * C (scalar scaling, Kernel not launched)
//
// Parameter validation mirrors the NPU operator (requirement doc §3.1).
// Column-major storage throughout, consistent with BLAS standard.
// ═══════════════════════════════════════════════════════════════════════════════


// Parameter validation: basic checks (handle, transA/transB, m/n/k, alpha/beta).
inline aclblasStatus_t ValidateSgemm3mBasicParams(
    aclblasHandle handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const float* alpha, const float* beta)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (transA != ACLBLAS_OP_N && transA != ACLBLAS_OP_T && transA != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (transB != ACLBLAS_OP_N && transB != ACLBLAS_OP_T && transB != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

// Parameter validation: dimension and pointer checks (lda/ldb/ldc, A/B/C).
inline aclblasStatus_t ValidateSgemm3mDimAndPtrParams(
    aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, float alphaVal,
    const float* A, const float* B, float* C, int lda, int ldb, int ldc)
{
    int physRowsA = gemm3mPhysRowsA(m, k, transA);
    int physRowsB = gemm3mPhysRowsB(k, n, transB);
    if (lda < std::max(1, physRowsA))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (ldb < std::max(1, physRowsB))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (ldc < std::max(1, m))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m > 0 && n > 0 && k > 0 && alphaVal != 0.0f) {
        if (A == nullptr || B == nullptr)
            return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0 && C == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

// Short-circuit: scale C by beta (K=0 or alpha=0 path).
inline void ScaleMatrixC(float* C, int m, int n, int ldc, float betaVal)
{
    if (betaVal == 0.0f) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                C[static_cast<size_t>(j) * ldc + i] = 0.0f;
    } else {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                C[static_cast<size_t>(j) * ldc + i] *= betaVal;
    }
}

// Compute sub-matrix pointers from merged A and B (3 pairs each).
inline void ComputeSgemm3mSubMatrixPtrs(
    const float* A, const float* B, int k, int lda, int ldb,
    bool isTransA, bool isTransB,
    const float*& A1, const float*& A2, const float*& A3,
    const float*& B1, const float*& B2, const float*& B3)
{
    if (isTransA) {
        A1 = A;
        A2 = A + static_cast<size_t>(k);
        A3 = A + 2 * static_cast<size_t>(k);
    } else {
        A1 = A;
        A2 = A + static_cast<size_t>(k) * lda;
        A3 = A + 2 * static_cast<size_t>(k) * lda;
    }
    if (isTransB) {
        B1 = B;
        B2 = B + static_cast<size_t>(k) * ldb;
        B3 = B + 2 * static_cast<size_t>(k) * ldb;
    } else {
        B1 = B;
        B2 = B + static_cast<size_t>(k);
        B3 = B + 2 * static_cast<size_t>(k);
    }
}

inline aclblasStatus_t aclblasSgemm3m_cpu(
    aclblasHandle handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta, float* C, int ldc)
{
    aclblasStatus_t status = ValidateSgemm3mBasicParams(handle, transA, transB, m, n, k, alpha, beta);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    float alphaVal = *alpha;
    float betaVal = *beta;

    status = ValidateSgemm3mDimAndPtrParams(transA, transB, m, n, k, alphaVal,
                                             A, B, C, lda, ldb, ldc);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    if (m == 0 || n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    if (k == 0 || alphaVal == 0.0f) {
        ScaleMatrixC(C, m, n, ldc, betaVal);
        return ACLBLAS_STATUS_SUCCESS;
    }

    bool isTransA = (transA != ACLBLAS_OP_N);
    bool isTransB = (transB != ACLBLAS_OP_N);
    const float *A1, *A2, *A3, *B1, *B2, *B3;
    ComputeSgemm3mSubMatrixPtrs(A, B, k, lda, ldb, isTransA, isTransB,
                                 A1, A2, A3, B1, B2, B3);

    CBLAS_TRANSPOSE opA = ToCblasOp(transA);
    CBLAS_TRANSPOSE opB = ToCblasOp(transB);

    cblas_sgemm(CblasColMajor, opA, opB, m, n, k, alphaVal, A1, lda, B1, ldb, betaVal, C, ldc);
    cblas_sgemm(CblasColMajor, opA, opB, m, n, k, alphaVal, A2, lda, B2, ldb, 1.0f, C, ldc);
    cblas_sgemm(CblasColMajor, opA, opB, m, n, k, alphaVal, A3, lda, B3, ldb, 1.0f, C, ldc);

    return ACLBLAS_STATUS_SUCCESS;
}

#endif // SGEMM3M_GOLDEN_H
