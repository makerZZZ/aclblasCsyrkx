/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_STRIDED_BATCHED_GOLDEN_H
#define GEMM_STRIDED_BATCHED_GOLDEN_H

#include <algorithm>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Parameter validation — mirrors NPU operator (requirement doc §3.1).
// CBLAS crashes on null pointers, so validation must run before any cblas call.
// ═══════════════════════════════════════════════════════════════════════════════
inline aclblasStatus_t ValidateGemmSbBasicParams(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* beta, int batchCount)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (transA != ACLBLAS_OP_N && transA != ACLBLAS_OP_T && transA != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (transB != ACLBLAS_OP_N && transB != ACLBLAS_OP_T && transB != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (m < 0 || n < 0 || k < 0 || batchCount < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ValidateGemmSbDimAndPtrParams(
    aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, const float* A, int lda, const float* B,
    int ldb, const float* beta, float* C, int ldc)
{
    const int physRowsA = (transA == ACLBLAS_OP_N) ? m : k;
    const int physRowsB = (transB == ACLBLAS_OP_N) ? k : n;
    if (lda < std::max(1, physRowsA))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (ldb < std::max(1, physRowsB))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (ldc < std::max(1, m))
        return ACLBLAS_STATUS_INVALID_VALUE;

    if (m > 0 && n > 0) {
        if (k > 0 && (A == nullptr || B == nullptr))
            return ACLBLAS_STATUS_INVALID_VALUE;
        if (*beta != 0.0f && C == nullptr)
            return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ValidateGemmStridedBatchedParams(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc,
    int batchCount)
{
    aclblasStatus_t status = ValidateGemmSbBasicParams(handle, transA, transB, m, n, k, alpha, beta, batchCount);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    return ValidateGemmSbDimAndPtrParams(transA, transB, m, n, k, A, lda, B, ldb, beta, C, ldc);
}

// ── C_i = beta * C_i (used for alpha==0 / k==0 scaling paths) ──
// beta==0 → memset zero (matches operator's aclrtMemsetAsync, avoids 0*NaN=NaN).
inline void ScaleCGolden(float* c, int m, int n, int ldc, float beta)
{
    if (beta == 0.0f) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                c[static_cast<size_t>(j) * ldc + i] = 0.0f;
            }
        }
        return;
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            c[static_cast<size_t>(j) * ldc + i] = beta * c[static_cast<size_t>(j) * ldc + i];
        }
    }
}

// Copies transposed B into contiguous K columns before the Netlib call.
// This changes only memory access: the same values and K-order reach SGEMM.
inline void PackTransposedB(const float* Bi, int k, int n, int ldb, float* packedB)
{
    for (int j = 0; j < n; ++j)
        for (int l = 0; l < k; ++l)
            packedB[static_cast<size_t>(j) * k + l] = Bi[static_cast<size_t>(l) * ldb + j];
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU golden: C_i = alpha * op(A_i) * op(B_i) + beta * C_i, for i in [0, batchCount).
//   A_i = A + i*strideA, B_i = B + i*strideB, C_i = C + i*strideC (element offsets).
//   strideA/strideB == 0 → broadcast (all batches reuse the same base matrix).
//   Signature matches the BLAS API; column-major storage via CblasColMajor.
// ═══════════════════════════════════════════════════════════════════════════════
inline aclblasStatus_t aclblasSgemmStridedBatched_cpu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda, int64_t strideA, const float* B, int ldb, int64_t strideB,
    const float* beta, float* C, int ldc, int64_t strideC, int batchCount)
{
    aclblasStatus_t status = ValidateGemmStridedBatchedParams(
        handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    // No-computation fast paths.
    if (m == 0 || n == 0 || batchCount == 0)
        return ACLBLAS_STATUS_SUCCESS;

    const float alphaVal = *alpha;
    const float betaVal = *beta;

    // Copy transposed B into contiguous K columns before the Netlib call.
    // This changes only memory access: the same values and K-order reach SGEMM.
    const bool packB = transB != ACLBLAS_OP_N && m >= 128 && n >= 128 && k >= 128 && alphaVal != 0.0f;
    std::vector<float> packedB(packB ? static_cast<size_t>(k) * n : 0);

    for (int i = 0; i < batchCount; i++) {
        const float* Ai = (A != nullptr) ? A + i * strideA : nullptr;
        const float* Bi = (B != nullptr) ? B + i * strideB : nullptr;
        float* Ci = (C != nullptr) ? C + i * strideC : nullptr;
        if (Ci == nullptr)
            continue;

        // alpha==0 or k==0 → only scale C by beta (matches operator; dodges 0*inf).
        if (alphaVal == 0.0f || k == 0) {
            ScaleCGolden(Ci, m, n, ldc, betaVal);
            continue;
        }
        if (packB) {
            if (i == 0 || strideB != 0)
                PackTransposedB(Bi, k, n, ldb, packedB.data());
            Bi = packedB.data();
        }
        cblas_sgemm(
            CblasColMajor, ToCblasOp(transA), packB ? CblasNoTrans : ToCblasOp(transB), m, n, k, alphaVal, Ai, lda, Bi,
            packB ? k : ldb, betaVal, Ci, ldc);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // GEMM_STRIDED_BATCHED_GOLDEN_H
