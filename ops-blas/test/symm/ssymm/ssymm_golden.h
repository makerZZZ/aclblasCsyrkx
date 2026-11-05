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
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#ifdef SSYMM_ARCH35
// ===== arch35: column-major =====
#include "cblas_compat.h"

static inline CBLAS_SIDE ToCblasSide(aclblasSideMode_t side)
{
    switch (side) {
        case ACLBLAS_SIDE_LEFT:  return CblasLeft;
        case ACLBLAS_SIDE_RIGHT: return CblasRight;
        default: assert(false && "Invalid aclblasSideMode_t"); return CblasLeft;
    }
}

static const int SSYMM_CPU_VALID_OK = 0x7FFFFFFF;

static inline int SsymmCpuValidateParams(aclblasSideMode_t side,
    int m, int n, const float* alpha, const float* beta,
    const float* A, const float* B, const float* C,
    int lda, int ldb, int ldc)
{
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT)
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    if (m < 0 || n < 0)
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    if (m == 0 || n == 0)
        return static_cast<int>(ACLBLAS_STATUS_SUCCESS);
    if (alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr)
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    int aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (lda < std::max(1, aDim) || ldb < std::max(1, m) || ldc < std::max(1, m))
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    if (*beta != 0.0f && C == nullptr)
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    return SSYMM_CPU_VALID_OK;
}

inline aclblasStatus_t aclblasSsymm_cpu(aclblasHandle_t handle, aclblasSideMode_t side,
    aclblasFillMode_t uplo, int m, int n, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER) return ACLBLAS_STATUS_INVALID_ENUM;

    int st = SsymmCpuValidateParams(side, m, n, alpha, beta, A, B, C, lda, ldb, ldc);
    if (st != SSYMM_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }

    if (m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;

    float alphaVal = *alpha;
    float betaVal = *beta;

    // BLAS: beta==0 and C==nullptr → no valid output, return success.
    if (betaVal == 0.0f && C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    cblas_ssymm(
        CblasColMajor,
        ToCblasSide(side),
        ToCblasUplo(uplo),
        m, n,
        alphaVal, A, lda,
        B, ldb,
        betaVal, C, ldc);

    return ACLBLAS_STATUS_SUCCESS;
}
#else
// ===== arch22: row-major (original) =====
// Read a value from symmetric matrix A (row-major, only one triangle stored).
// The missing triangle is reconstructed by transposing indices.
static inline float SsymmGetSymValue(const float* a, int lda,
    aclblasFillMode_t uplo, int row, int col)
{
    if (uplo == ACLBLAS_LOWER) {
        return (row >= col) ? a[row * lda + col] : a[col * lda + row];
    }
    return (row <= col) ? a[row * lda + col] : a[col * lda + row];
}

// CPU reference implementation — signature matches aclblasSsymm exactly.
// Reproduces ValidateSsymmArgs parameter validation followed by the
// three-loop matrix multiply.
inline aclblasStatus_t aclblasSsymm_cpu(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    int m,
    int n,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc)
{
    // --- parameter validation (mirrors ValidateSsymmArgs) ---
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr || C == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (m < 0 || n < 0) return ACLBLAS_STATUS_INVALID_VALUE;

    // quick return
    if (m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;

    const int aDim   = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    if (lda < aDim || ldb < n || ldc < n)
        return ACLBLAS_STATUS_INVALID_VALUE;

    // --- computation ---
    // scale C by beta first
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * ldc + j] *= (*beta);
        }
    }

    // accumulate alpha * A_sym * B  (LEFT) or alpha * B * A_sym  (RIGHT)
    if (side == ACLBLAS_SIDE_LEFT) {
        // C[i][j] += alpha * sum_k( A_sym[i][k] * B[k][j] )
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double acc = 0.0;
                for (int k = 0; k < m; ++k) {
                    acc += static_cast<double>(SsymmGetSymValue(A, lda, uplo, i, k))
                         * static_cast<double>(B[k * ldb + j]);
                }
                C[i * ldc + j] += static_cast<float>((*alpha) * static_cast<float>(acc));
            }
        }
    } else {
        // C[i][j] += alpha * sum_k( B[i][k] * A_sym[k][j] )
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                double acc = 0.0;
                for (int k = 0; k < n; ++k) {
                    acc += static_cast<double>(B[i * ldb + k])
                         * static_cast<double>(SsymmGetSymValue(A, lda, uplo, k, j));
                }
                C[i * ldc + j] += static_cast<float>((*alpha) * static_cast<float>(acc));
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
#endif
