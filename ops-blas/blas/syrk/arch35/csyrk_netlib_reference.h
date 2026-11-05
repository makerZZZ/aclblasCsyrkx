/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file csyrk_netlib_reference.h
 * \brief Netlib reference BLAS `csyrk` (LAPACK csyrk.f), complex single precision.
 *
 *        Shared by the host-side special-value fallback (csyrk_host.cpp) and the
 *        test golden (test/syrk/csyrk/csyrk_reference.h) so the two stay bit
 *        identical. Mirrors the Fortran reference loop order and fp32 rounding
 *        exactly, with no third-party BLAS dependency.
 *
 *        C := alpha * A * A**T + beta * C           (trans == ACLBLAS_OP_N)
 *        C := alpha * A**T * A + beta * C           (trans == ACLBLAS_OP_T / OP_C, no conj)
 *        Only the uplo triangle of C is updated; the opposite triangle is untouched.
 */

#pragma once

#include <algorithm>

#include "cann_ops_blas.h"

// Computes the columns [j0, j1) of the result. Each column j only reads A and
// writes C(:,j), so any partition of [0, n) over j0/j1 produces the same
// per-column accumulation order as the full-range call -- the test golden
// parallelises on this, while the host fallback keeps the default full range.
// j1 < 0 means n.
inline void CsyrkNetlibReferenceCore(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* beta, aclblasComplex* C, int ldc, int j0 = 0,
    int j1 = -1)
{
    auto cIsZero = [](const aclblasComplex& z) { return z.real == 0.0f && z.imag == 0.0f; };
    auto cMul = [](const aclblasComplex& a, const aclblasComplex& b) {
        return aclblasComplex{a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
    };
    auto cAdd = [](const aclblasComplex& a, const aclblasComplex& b) {
        return aclblasComplex{a.real + b.real, a.imag + b.imag};
    };

    const bool upper = (uplo == ACLBLAS_UPPER);
    const bool noTrans = (trans == ACLBLAS_OP_N);

    const int colBegin = std::max(0, j0);
    const int colEnd = (j1 < 0) ? n : std::min(j1, n);
    for (int j = colBegin; j < colEnd; ++j) {
        // Scale the active part of column j by beta (Netlib: BETA*C(I,J) first).
        if (cIsZero(*beta)) {
            for (int i = upper ? 0 : j; i < (upper ? j + 1 : n); ++i) {
                C[i + j * ldc] = aclblasComplex{0.0f, 0.0f};
            }
        } else if (beta->real != 1.0f || beta->imag != 0.0f) {
            for (int i = upper ? 0 : j; i < (upper ? j + 1 : n); ++i) {
                C[i + j * ldc] = cMul(*beta, C[i + j * ldc]);
            }
        }

        for (int l = 0; l < k; ++l) {
            // A(J,L) for noTrans (A is N x K); A(L,J) for trans (A is K x N).
            const aclblasComplex* aOuter = noTrans ? &A[j + l * lda] : &A[l + j * lda];
            if (cIsZero(*aOuter)) {
                continue;
            }
            const aclblasComplex temp = cMul(*alpha, *aOuter);
            const int iBegin = upper ? 0 : j;
            const int iEnd = upper ? j + 1 : n;
            for (int i = iBegin; i < iEnd; ++i) {
                const aclblasComplex* aInner = noTrans ? &A[i + l * lda] : &A[l + i * lda];
                C[i + j * ldc] = cAdd(C[i + j * ldc], cMul(temp, *aInner));
            }
        }
    }
}
