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

#include "cann_ops_blas.h"
#include "csyrk_reference.h"

// Sentinel: all validations passed, proceed to compute.
static const int CSYRK_CPU_VALID_OK = 0x7FFFFFFF;

static bool CsyrkIsValidUplo(aclblasFillMode_t uplo) { return uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER; }

// CSYRK forms A*A^T with a *plain* transpose; OP_C maps to OP_T (no conjugation).
// OP_C is legal and equivalent to OP_T (symmetric, not Hermitian: no conjugation),
// per task doc 2.1/2.4. csyrk has no conjugate-transpose variant.
static bool CsyrkIsValidTrans(aclblasOperation_t trans)
{
    return trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C;
}

// Rejects the same inputs the NPU operator rejects (task doc §2.5): enum
// legality and leading dimensions are checked before the pointers, and every
// check precedes the n == 0 quick return, so a null scalar still reports
// INVALID_VALUE on a zero-size call.
static int ValidateCsyrkCpuParams(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, int lda, int ldc,
    const aclblasComplex* alpha, const aclblasComplex* beta, const aclblasComplex* A, const aclblasComplex* C)
{
    if (handle == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
    if (!CsyrkIsValidUplo(uplo)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    if (!CsyrkIsValidTrans(trans)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM);
    }
    if (n < 0 || k < 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    bool isTrans = (trans != ACLBLAS_OP_N);
    int minLda = isTrans ? std::max(1, k) : std::max(1, n);
    if (lda < minLda) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (ldc < std::max(1, n)) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (alpha == nullptr || beta == nullptr) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (A == nullptr && n > 0 && k > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    if (C == nullptr && n > 0) {
        return static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE);
    }
    return CSYRK_CPU_VALID_OK;
}

// CPU golden for aclblasCsyrk. Signature mirrors the NPU API exactly so the test
// harness can swap NPU/CPU calls one-for-one. Validation precedes the reference
// call because the reference must not be entered with null pointers.
//
// The reference is the Netlib complex `csyrk` (LAPACK csyrk.f) shipped with the
// test project — no third-party BLAS dependency, per task doc §3.1/§3.2. It is
// the same loop the operator's host-side special-value fallback uses, so the two
// stay bit-identical (see blas/syrk/arch35/csyrk_netlib_reference.h).
inline aclblasStatus_t aclblasCsyrk_cpu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* beta, aclblasComplex* C, int ldc)
{
    int st = ValidateCsyrkCpuParams(handle, uplo, trans, n, k, lda, ldc, alpha, beta, A, C);
    if (st != CSYRK_CPU_VALID_OK) {
        return static_cast<aclblasStatus_t>(st);
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // OP_C is equivalent to OP_T for csyrk (plain transpose, no conjugation).
    CsyrkNetlibReference(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);

    return ACLBLAS_STATUS_SUCCESS;
}

