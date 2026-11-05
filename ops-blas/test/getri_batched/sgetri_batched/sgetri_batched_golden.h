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
void sgetrf_(const int* m, const int* n, float* a, const int* lda, int* ipiv, int* info);
void sgetri_(const int* n, float* a, const int* lda, int* ipiv, float* work, const int* lwork, int* info);
}

inline void ValidateGetriBatchedParams(
    aclblasHandle_t handle, int n, const float* const Aarray[], int lda, float* const Carray[], int ldc, int* infoArray,
    int batchSize, aclblasStatus_t& status)
{
    if (handle == nullptr) {
        status = ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
        return;
    }
    if (n < 0 || batchSize < 0 || lda < std::max(1, n) || ldc < std::max(1, n)) {
        status = ACLBLAS_STATUS_INVALID_VALUE;
        return;
    }
    if (n == 0 || batchSize == 0) {
        status = ACLBLAS_STATUS_SUCCESS;
        return;
    }
    if (Aarray == nullptr || Carray == nullptr || infoArray == nullptr) {
        status = ACLBLAS_STATUS_INVALID_VALUE;
        return;
    }
    status = ACLBLAS_STATUS_SUCCESS;
}

inline void GetriGoldenGetrfNoPivot(float* A, int n, int lda, int& info)
{
    info = 0;
    for (int k = 0; k < n; k++) {
        float diagVal = A[k + k * lda];
        if (diagVal == 0.0f) {
            info = k + 1;
            return;
        }
        for (int i = k + 1; i < n; i++) {
            A[i + k * lda] /= diagVal;
        }
        for (int j = k + 1; j < n; j++) {
            float uKJ = A[k + j * lda];
            for (int i = k + 1; i < n; i++) {
                A[i + j * lda] -= A[i + k * lda] * uKJ;
            }
        }
    }
}

inline int GetriInvertSingle(const float* LU, int n, int lda, const int* piv, float* C, int ldc)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            C[j * ldc + i] = LU[j * lda + i];
        }
    }

    std::vector<int> ipiv(n);
    if (piv != nullptr) {
        for (int i = 0; i < n; i++) {
            ipiv[i] = piv[i];
        }
    } else {
        for (int i = 0; i < n; i++) {
            ipiv[i] = i + 1;
        }
    }

    int lwork = -1;
    float work_query;
    int info = 0;
    sgetri_(&n, C, &ldc, ipiv.data(), &work_query, &lwork, &info);
    if (info != 0) {
        return info;
    }

    lwork = static_cast<int>(work_query);
    std::vector<float> work(lwork);
    sgetri_(&n, C, &ldc, ipiv.data(), work.data(), &lwork, &info);

    return info;
}

inline aclblasStatus_t aclblasSgetriBatched_cpu(
    aclblasHandle_t handle, int n, const float* const Aarray[], int lda, const int* PivotArray, float* const Carray[],
    int ldc, int* infoArray, int batchSize)
{
    aclblasStatus_t status;
    ValidateGetriBatchedParams(handle, n, Aarray, lda, Carray, ldc, infoArray, batchSize, status);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    if (n == 0 || batchSize == 0)
        return ACLBLAS_STATUS_SUCCESS;

    for (int b = 0; b < batchSize; b++) {
        const int* piv = (PivotArray != nullptr) ? (PivotArray + b * n) : nullptr;
        int info = GetriInvertSingle(Aarray[b], n, lda, piv, Carray[b], ldc);
        if (infoArray != nullptr) {
            infoArray[b] = info;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline void aclblasSgetrfBatched_cpu_for_getri(
    int n, float** Aarray, int lda, int* PivotArray, int* infoArray, int batchSize, bool usePivot)
{
    for (int b = 0; b < batchSize; b++) {
        int info = 0;
        if (usePivot) {
            int* ipiv = (PivotArray != nullptr) ? (PivotArray + b * n) : nullptr;
            sgetrf_(&n, &n, Aarray[b], &lda, ipiv, &info);
        } else {
            GetriGoldenGetrfNoPivot(Aarray[b], n, lda, info);
        }
        if (infoArray != nullptr) {
            infoArray[b] = info;
        }
    }
}

