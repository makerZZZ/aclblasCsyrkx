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

#include "cann_ops_blas_common.h"
#include "op_common/log/log.h"

#include <string>

inline bool IsGeamScalarZero(const float* p) { return p == nullptr || *p == 0.0f; }

inline bool IsGeamScalarZero(const aclblasComplex* p) { return p == nullptr || (p->real == 0.0f && p->imag == 0.0f); }

inline aclblasStatus_t ValidateGeamTranspose(
    aclblasOperation_t transa, aclblasOperation_t transb, const std::string& op_name)
{
    if (transa != ACLBLAS_OP_N && transa != ACLBLAS_OP_T && transa != ACLBLAS_OP_C) {
        OP_LOGE(op_name, "Invalid transa=%d (must be N/T/C)", transa);
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (transb != ACLBLAS_OP_N && transb != ACLBLAS_OP_T && transb != ACLBLAS_OP_C) {
        OP_LOGE(op_name, "Invalid transb=%d (must be N/T/C)", transb);
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t ValidateGeamLd(
    int m, int n, int lda, int ldb, int ldc, aclblasOperation_t transa, aclblasOperation_t transb,
    const std::string& op_name)
{
    if (transa == ACLBLAS_OP_N) {
        if (lda < std::max(1, m)) {
            OP_LOGE(op_name, "lda=%d must be >= max(1, m=%d) when transa=N", lda, m);
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    } else {
        if (lda < std::max(1, n)) {
            OP_LOGE(op_name, "lda=%d must be >= max(1, n=%d) when transa=T/C", lda, n);
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (transb == ACLBLAS_OP_N) {
        if (ldb < std::max(1, m)) {
            OP_LOGE(op_name, "ldb=%d must be >= max(1, m=%d) when transb=N", ldb, m);
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    } else {
        if (ldb < std::max(1, n)) {
            OP_LOGE(op_name, "ldb=%d must be >= max(1, n=%d) when transb=T/C", ldb, n);
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE(op_name, "ldc=%d must be >= max(1, m=%d)", ldc, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <class T>
inline aclblasStatus_t ValidateGeamPointers(
    const T* alpha, const T* A, const T* beta, const T* B, T* C, const std::string& op_name)
{
    if (alpha == nullptr) {
        OP_LOGE(op_name, "alpha pointer is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE(op_name, "beta pointer is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (C == nullptr) {
        OP_LOGE(op_name, "C pointer is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!IsGeamScalarZero(alpha) && A == nullptr) {
        OP_LOGE(op_name, "A pointer is nullptr (alpha != 0)");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!IsGeamScalarZero(beta) && B == nullptr) {
        OP_LOGE(op_name, "B pointer is nullptr (beta != 0)");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <class T>
inline aclblasStatus_t ValidateGeamInplace(
    const T* A, const T* B, T* C, int lda, int ldb, int ldc, aclblasOperation_t transa, aclblasOperation_t transb,
    const std::string& op_name)
{
    if (A != nullptr && C == A) {
        if (transa != ACLBLAS_OP_N) {
            OP_LOGE(op_name, "In-place C==A requires transa==N (got %d)", static_cast<int>(transa));
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (lda != ldc) {
            OP_LOGE(op_name, "In-place C==A requires lda==ldc (lda=%d, ldc=%d)", lda, ldc);
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (B != nullptr && C == B) {
        if (transb != ACLBLAS_OP_N) {
            OP_LOGE(op_name, "In-place C==B requires transb==N (got %d)", static_cast<int>(transb));
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (ldb != ldc) {
            OP_LOGE(op_name, "In-place C==B requires ldb==ldc (ldb=%d, ldc=%d)", ldb, ldc);
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <class T>
inline aclblasStatus_t ValidateGeamParams(
    const std::string& op_name, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, const T* alpha,
    const T* A, int lda, const T* beta, const T* B, int ldb, T* C, int ldc)
{
    aclblasStatus_t status;
    status = ValidateGeamTranspose(transa, transb, op_name);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (m < 0) {
        OP_LOGE(op_name, "m=%d must be >= 0", m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE(op_name, "n=%d must be >= 0", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    status = ValidateGeamPointers<T>(alpha, A, beta, B, C, op_name);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = ValidateGeamLd(m, n, lda, ldb, ldc, transa, transb, op_name);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    return ValidateGeamInplace<T>(A, B, C, lda, ldb, ldc, transa, transb, op_name);
}
