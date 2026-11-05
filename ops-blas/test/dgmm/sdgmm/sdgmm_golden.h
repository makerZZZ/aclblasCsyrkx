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

#include "cann_ops_blas.h"

static inline aclblasStatus_t SdgmmValidateParams(
    aclblasHandle_t handle, aclblasSideMode_t mode,
    int m, int n, const float* A, int lda,
    const float* x, int incx, float* C, int ldc)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (mode != ACLBLAS_SIDE_LEFT && mode != ACLBLAS_SIDE_RIGHT) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, m) || ldc < std::max(1, m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0 && (x == nullptr || A == nullptr || C == nullptr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// DGMM (Diagonal General Matrix Multiply) — MAGMA extension, not in Netlib BLAS.
//   mode = LEFT:  C[i,j] = x[i] * A[i,j]   (C = diag(x) * A)
//   mode = RIGHT: C[i,j] = A[i,j] * x[j]   (C = A * diag(x))
// Column-major storage. incx stride: x_k = x[(incx<0) ? (len-1-k)*|incx| : k*incx]
inline aclblasStatus_t aclblasSdgmm_cpu(
    aclblasHandle_t handle,
    aclblasSideMode_t mode,
    int m, int n,
    const float* A, int lda,
    const float* x, int incx,
    float* C, int ldc)
{
    aclblasStatus_t st = SdgmmValidateParams(handle, mode, m, n, A, lda, x, incx, C, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const int xLen = (mode == ACLBLAS_SIDE_LEFT) ? m : n;
    const int64_t absIncx = (incx >= 0) ? static_cast<int64_t>(incx)
                                        : -static_cast<int64_t>(incx);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            int k = (mode == ACLBLAS_SIDE_LEFT) ? i : j;
            int64_t xIdx = (incx < 0)
                ? static_cast<int64_t>(xLen - 1 - k) * absIncx
                : static_cast<int64_t>(k) * incx;
            float xVal = x[xIdx];
            float aVal = A[i + static_cast<int64_t>(j) * lda];
            C[i + static_cast<int64_t>(j) * ldc] = xVal * aVal;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
