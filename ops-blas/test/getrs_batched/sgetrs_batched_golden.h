/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGETRS_BATCHED_GOLDEN_H
#define SGETRS_BATCHED_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

extern "C" {
void sgetrf_(const int* m, const int* n, float* a, const int* lda, int* ipiv, int* info);
void sgetrs_(const char* trans, const int* n, const int* nrhs, const float* a, const int* lda, const int* ipiv,
             float* b, const int* ldb, int* info);
}

// ── Parameter validation (mirrors NPU host-side checks) ──

inline void ValidateGetrsBatchedParams(
    aclblasHandle_t handle, aclblasOperation_t trans, int n, int nrhs,
    const float* const Aarray[], int lda, float* const Barray[], int ldb,
    int batchCount, aclblasStatus_t& status)
{
    if (handle == nullptr) {
        status = ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
        return;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        status = ACLBLAS_STATUS_INVALID_ENUM;
        return;
    }
    if (n < 0 || nrhs < 0 || nrhs > 256 || batchCount < 0) {
        status = ACLBLAS_STATUS_INVALID_VALUE;
        return;
    }
    if (lda < std::max(1, n) || ldb < std::max(1, n)) {
        status = ACLBLAS_STATUS_INVALID_VALUE;
        return;
    }
    if (n == 0 || nrhs == 0 || batchCount == 0) {
        status = ACLBLAS_STATUS_SUCCESS;
        return;
    }
    if (Aarray == nullptr || Barray == nullptr) {
        status = ACLBLAS_STATUS_INVALID_VALUE;
        return;
    }
    status = ACLBLAS_STATUS_SUCCESS;
}

// ── No-pivot LU factorization (manual, for NO_PIVOT mode) ──

inline void GetrsGetrfNoPivot(float* A, int n, int lda, int& info)
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

// ── Single batch solve using LAPACK sgetrs_ ──

inline int GetrsSolveSingle(
    const float* LU, int n, int nrhs, int lda, const int* ipiv, float* B, int ldb, char transChar)
{
    int info = 0;
    sgetrs_(&transChar, &n, &nrhs, LU, &lda, ipiv, B, &ldb, &info);
    return info;
}

// ── CPU golden entry point ──

inline aclblasStatus_t aclblasSgetrsBatched_cpu(
    aclblasHandle_t handle, aclblasOperation_t trans, int n, int nrhs,
    const float* const Aarray[], int lda, const int* devIpiv,
    float* const Barray[], int ldb, int* info, int batchCount)
{
    aclblasStatus_t status;
    ValidateGetrsBatchedParams(handle, trans, n, nrhs, Aarray, lda, Barray, ldb, batchCount, status);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    if (n == 0 || nrhs == 0 || batchCount == 0)
        return ACLBLAS_STATUS_SUCCESS;

    char transChar;
    switch (trans) {
        case ACLBLAS_OP_N: transChar = 'N'; break;
        case ACLBLAS_OP_T: transChar = 'T'; break;
        case ACLBLAS_OP_C: transChar = 'C'; break;
        default: transChar = 'N'; break;
    }

    int aggInfo = 0;
    for (int b = 0; b < batchCount; b++) {
        std::vector<int> ipiv(n);
        if (devIpiv != nullptr) {
            for (int i = 0; i < n; i++) {
                ipiv[i] = devIpiv[b * n + i];
            }
        } else {
            for (int i = 0; i < n; i++) {
                ipiv[i] = i + 1;
            }
        }

        int solveInfo = GetrsSolveSingle(Aarray[b], n, nrhs, lda, ipiv.data(), Barray[b], ldb, transChar);
        if (aggInfo == 0 && solveInfo != 0) {
            aggInfo = solveInfo;
        }
    }
    if (info != nullptr) {
        *info = aggInfo;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ── CPU LU factorization for test data preparation ──

inline void aclblasSgetrfBatched_cpu_for_getrs(
    int n, float** Aarray, int lda, int* PivotArray, int* infoArray, int batchCount, bool usePivot)
{
    for (int b = 0; b < batchCount; b++) {
        int infoVal = 0;
        if (usePivot) {
            int* ipiv = (PivotArray != nullptr) ? (PivotArray + b * n) : nullptr;
            sgetrf_(&n, &n, Aarray[b], &lda, ipiv, &infoVal);
        } else {
            GetrsGetrfNoPivot(Aarray[b], n, lda, infoVal);
        }
        if (infoArray != nullptr) {
            infoArray[b] = infoVal;
        }
    }
}

#endif // SGETRS_BATCHED_GOLDEN_H
