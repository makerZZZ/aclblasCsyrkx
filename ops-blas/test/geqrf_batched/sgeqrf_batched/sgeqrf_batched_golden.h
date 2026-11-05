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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

extern "C" {
void sgeqrf_(
    const int* m, const int* n, float* a, const int* lda, float* tau, float* work, const int* lwork, int* info);
}

inline void sgeqrf_single_batch(int m, int n, float* A, int lda, float* tau)
{
    int lwork = -1;
    float work_query;
    int info = 0;
    sgeqrf_(&m, &n, A, &lda, tau, &work_query, &lwork, &info);
    lwork = static_cast<int>(work_query);
    std::vector<float> work(lwork);
    sgeqrf_(&m, &n, A, &lda, tau, work.data(), &lwork, &info);
}

inline aclblasStatus_t aclblasSgeqrfBatched_cpu(
    aclblasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info,
    int batchSize)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (m < 0) {
        if (info != nullptr)
            *info = ACLBLAS_LAPACK_INFO_ARG_1;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        if (info != nullptr)
            *info = ACLBLAS_LAPACK_INFO_ARG_2;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, m)) {
        if (info != nullptr)
            *info = ACLBLAS_LAPACK_INFO_ARG_4;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize < 0) {
        if (info != nullptr)
            *info = ACLBLAS_LAPACK_INFO_ARG_6;
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize == 0 || m == 0 || n == 0) {
        if (info != nullptr)
            *info = ACLBLAS_LAPACK_INFO_OK;
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (Aarray == nullptr || TauArray == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    for (int b = 0; b < batchSize; b++) {
        if (Aarray[b] != nullptr && TauArray[b] != nullptr) {
            sgeqrf_single_batch(m, n, Aarray[b], lda, TauArray[b]);
        }
    }
    if (info != nullptr)
        *info = ACLBLAS_LAPACK_INFO_OK;
    return ACLBLAS_STATUS_SUCCESS;
}
