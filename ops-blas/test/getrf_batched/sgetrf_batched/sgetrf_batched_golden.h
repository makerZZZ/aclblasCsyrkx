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
}

inline aclblasStatus_t ValidateGetrfParams(
    aclblasHandle_t handle, int n, float* const Aarray[], int lda, int* PivotArray, int* infoArray, int batchSize)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n < 0 || batchSize < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, n))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (Aarray == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (PivotArray != nullptr && infoArray == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline void FactorizeSingleMatrixNoPivot(float* A, int n, int lda, int& info)
{
    info = 0;
    for (int k = 0; k < n; k++) {
        float diagVal = A[k + k * lda];
        if (diagVal == 0.0f) {
            info = k + 1;
            break;
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

inline aclblasStatus_t aclblasSgetrfBatched_cpu(
    aclblasHandle_t handle, int n, float* const Aarray[], int lda, int* PivotArray, int* infoArray, int batchSize)
{
    aclblasStatus_t validRet = ValidateGetrfParams(handle, n, Aarray, lda, PivotArray, infoArray, batchSize);
    if (validRet != ACLBLAS_STATUS_SUCCESS)
        return validRet;
    if (n == 0 || batchSize == 0)
        return ACLBLAS_STATUS_SUCCESS;

    const bool usePivot = (PivotArray != nullptr);
    for (int b = 0; b < batchSize; b++) {
        int info = 0;
        if (usePivot) {
            int* piv = PivotArray + b * n;
            sgetrf_(&n, &n, Aarray[b], &lda, piv, &info);
            if (info > 0) {
                for (int k = info; k < n; k++) {
                    piv[k] = 0;
                }
            }
        } else {
            FactorizeSingleMatrixNoPivot(Aarray[b], n, lda, info);
        }
        if (infoArray != nullptr) {
            infoArray[b] = info;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

