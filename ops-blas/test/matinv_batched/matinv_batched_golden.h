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
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

extern "C" {
void sgetrf_(const int* m, const int* n, float* a, const int* lda, int* ipiv, int* info);
void sgetri_(const int* n, float* a, const int* lda, int* ipiv, float* work, const int* lwork, int* info);
}

// CPU golden: compute matrix inverse using LAPACK sgetrf_ + sgetri_
inline aclblasStatus_t aclblasSmatinvBatched_cpu(
    aclblasHandle_t handle,
    int n,
    const float* const A[],
    int lda,
    float* const Ainv[],
    int lda_inv,
    int* info,
    int batchSize)
{
    // Parameter validation (mirrors NPU host-side checks)
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n < 0 || n > 32)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (batchSize < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, n) || lda_inv < std::max(1, n))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0 || batchSize == 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (A == nullptr || Ainv == nullptr || info == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    // Process each batch
    for (int b = 0; b < batchSize; b++) {
        // Copy A[b] to Ainv[b] (working buffer)
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                Ainv[b][j * lda_inv + i] = A[b][j * lda + i];
            }
        }

        // LU decomposition on Ainv (in-place), including n==1 for numerical consistency with NPU path
        std::vector<int> ipiv(n);
        int infoVal = 0;
        sgetrf_(&n, &n, Ainv[b], &lda_inv, ipiv.data(), &infoVal);

        if (infoVal > 0) {
            // U(infoVal, infoVal) is exactly zero -> singular
            info[b] = infoVal;
            continue;
        }

        // Compute inverse from LU factored form
        int lwork = -1;
        float work_query;
        sgetri_(&n, Ainv[b], &lda_inv, ipiv.data(), &work_query, &lwork, &infoVal);

        if (infoVal != 0) {
            info[b] = infoVal;
            continue;
        }

        lwork = static_cast<int>(work_query);
        std::vector<float> work(lwork);
        sgetri_(&n, Ainv[b], &lda_inv, ipiv.data(), work.data(), &lwork, &infoVal);

        info[b] = infoVal;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
