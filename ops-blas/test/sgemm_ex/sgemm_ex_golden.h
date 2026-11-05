/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGEMM_EX_GOLDEN_H
#define SGEMM_EX_GOLDEN_H

#include <algorithm>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "sgemm_ex_param.h"

// ═══════════════════════════════════════════════════════════════════════════════
// CPU golden: C = alpha * op(A) * op(B) + beta * C  (column-major, FP32)
//
// Signature matches aclblasSgemmEx API.
// Parameter validation mirrors the NPU host code exactly.
// Calls CBLAS cblas_sgemm for the main computation.
// ═══════════════════════════════════════════════════════════════════════════════

inline aclblasStatus_t aclblasSgemmEx_cpu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const float* alpha,
    const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc, aclblasGemmAlgo_t algo)
{
    // 1. Quick return for zero-dimension matrices (matches host: before validation)
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // 2. Parameter validation (matches host ValidateSgemmExParams)
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (m < 0 || n < 0 || k < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (transA != ACLBLAS_OP_N && transA != ACLBLAS_OP_T && transA != ACLBLAS_OP_C) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (transB != ACLBLAS_OP_N && transB != ACLBLAS_OP_T && transB != ACLBLAS_OP_C) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    bool isTransA = (transA != ACLBLAS_OP_N);
    bool isTransB = (transB != ACLBLAS_OP_N);
    int expectedLda = isTransA ? std::max(1, k) : std::max(1, m);
    int expectedLdb = isTransB ? std::max(1, n) : std::max(1, k);
    if (lda < expectedLda) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldb < expectedLdb) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (alpha == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (algo != ACLBLAS_GEMM_DEFAULT) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }

    float alphaVal = *alpha;
    float betaVal = *beta;

    if (k > 0) {
        if (A == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (B == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (betaVal != 0.0f && C == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    // 3. Edge case: C is null → no output to write
    if (C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // 4. Edge case: k==0 or alpha==0 → C = beta * C (matches host HandleBetaOnly)
    //    Do not call cblas_sgemm because A/B may be nullptr when k==0.
    if (k == 0 || alphaVal == 0.0f) {
        if (betaVal == 0.0f) {
            // Zero C (matches host aclrtMemset)
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    C[static_cast<size_t>(j) * ldc + i] = 0.0f;
                }
            }
        } else {
            // C = beta * C
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    size_t idx = static_cast<size_t>(j) * ldc + i;
                    C[idx] = betaVal * C[idx];
                }
            }
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    // 5. Main computation: call CBLAS cblas_sgemm (column-major)
    cblas_sgemm(
        CblasColMajor, ToCblasOp(transA), ToCblasOp(transB),
        m, n, k, alphaVal, A, lda, B, ldb, betaVal, C, ldc);
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // SGEMM_EX_GOLDEN_H
