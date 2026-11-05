/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSYRKX_GOLDEN_H
#define CSYRKX_GOLDEN_H

#include <algorithm>
#include <cstddef>

#include "cann_ops_blas.h"
#include "cblas_compat.h"

static const int CSYRKX_CPU_VALID_OK = 0x7FFFFFFF;

// Parameter validation mirrors aclblasCsyrkx (task doc §2.4/§2.5):
//   handle -> enum (INVALID_VALUE) -> n/k >= 0 -> leading dimensions
//   -> scalar pointers -> n==0 quick return -> A/B (alpha != 0 && k > 0)
//   -> C (n > 0). Parameter checks precede the no-op decision.
static int ValidateCsyrkxEnumsAndDims(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, int lda, int ldb, int ldc)
{
    if (handle == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (n < 0 || k < 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    const int minLdab = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    if (lda < minLdab || ldb < minLdab || ldc < std::max(1, n)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return CSYRKX_CPU_VALID_OK;
}

static bool CsyrkxIsComplexZero(const aclblasComplex* v) { return v->real == 0.0f && v->imag == 0.0f; }

static int ValidateCsyrkxScalarsAndPtrs(
    int n, int k, const aclblasComplex* alpha, const aclblasComplex* beta, const aclblasComplex* A,
    const aclblasComplex* B, aclblasComplex* C)
{
    if (alpha == nullptr || beta == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    // n == 0: legal no-op after every parameter check above.
    if (n == 0) {
        return static_cast<int>(ACLBLAS_STATUS_SUCCESS);
    }
    // A/B are only referenced when alpha != 0 and k > 0.
    const bool alphaNonZero = !CsyrkxIsComplexZero(alpha);
    if (alphaNonZero && k > 0 && (A == nullptr || B == nullptr)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (C == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return CSYRKX_CPU_VALID_OK;
}

static inline bool CsyrkxIsUplo(aclblasFillMode_t uplo, int i, int j)
{
    return (uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
}

// C_uplo = beta * C_old with complex beta (beta == 0 zeroes the triangle
// instead of propagating NaN/Inf through 0*x).
static inline void ApplyBetaOnlyUploComplex(aclblasFillMode_t uplo, int n, int ldc, aclblasComplex* C,
    const aclblasComplex& beta)
{
    const bool betaZero = (beta.real == 0.0f && beta.imag == 0.0f);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (!CsyrkxIsUplo(uplo, i, j)) {
                continue;
            }
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(ldc);
            if (betaZero) {
                C[idx] = aclblasComplex{0.0f, 0.0f};
            } else {
                const float re = C[idx].real;
                const float im = C[idx].imag;
                C[idx] = aclblasComplex{
                    beta.real * re - beta.imag * im,
                    beta.real * im + beta.imag * re};
            }
        }
    }
}

// Restore the non-uplo triangle to the input-old values: aclblasCsyrkx only
// writes the uplo triangle, while cblas_cgemm updates the full n x n matrix.
static inline void RestoreNonUploTriangleComplex(
    aclblasFillMode_t uplo, int n, int ldc, aclblasComplex* goldenC, const aclblasComplex* originalC)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (!CsyrkxIsUplo(uplo, i, j)) {
                size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(ldc);
                goldenC[idx] = originalC[idx];
            }
        }
    }
}

// CPU golden built on cblas_cgemm (task doc §3.2):
//   trans=N   : C = alpha * A   * B^T   (opA=NoTrans, opB=Trans)
//   trans=T/C : C = alpha * A^T * B     (opA=Trans,   opB=NoTrans)
// OP_C is mapped to Trans on purpose: syrkx is symmetric, not Hermitian.
// `originalC` is a read-only snapshot used to restore the non-uplo triangle.
inline aclblasStatus_t aclblasCsyrkx_cpu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc, const aclblasComplex* originalC)
{
    int st = ValidateCsyrkxEnumsAndDims(handle, uplo, trans, n, k, lda, ldb, ldc);
    if (st != CSYRKX_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }
    st = ValidateCsyrkxScalarsAndPtrs(n, k, alpha, beta, A, B, C);
    if (st != CSYRKX_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // alpha == 0 or k == 0: A/B terms vanish, C_uplo = beta*C_old.
    if (CsyrkxIsComplexZero(alpha) || k == 0) {
        ApplyBetaOnlyUploComplex(uplo, n, ldc, C, *beta);
        return ACLBLAS_STATUS_SUCCESS;
    }

    const CBLAS_TRANSPOSE cblasTransA = (trans == ACLBLAS_OP_N) ? CblasNoTrans : CblasTrans;
    const CBLAS_TRANSPOSE cblasTransB = (trans == ACLBLAS_OP_N) ? CblasTrans : CblasNoTrans;

    // beta == 0: the uplo result must not depend on C_old (0 * NaN / 0 * Inf
    // would otherwise leak into the result). Clear the active region first so
    // the reference BLAS sees a neutral input; the non-uplo triangle is
    // restored from originalC below.
    if (CsyrkxIsComplexZero(beta)) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                C[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(ldc)] =
                    aclblasComplex{0.0f, 0.0f};
            }
        }
    }

    cblas_cgemm(
        CblasColMajor, cblasTransA, cblasTransB, n, n, k, static_cast<const void*>(alpha),
        static_cast<const void*>(A), lda, static_cast<const void*>(B), ldb, static_cast<const void*>(beta),
        static_cast<void*>(C), ldc);

    RestoreNonUploTriangleComplex(uplo, n, ldc, C, originalC);
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // CSYRKX_GOLDEN_H
