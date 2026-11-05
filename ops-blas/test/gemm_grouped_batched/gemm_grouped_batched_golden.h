/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_GROUPED_BATCHED_GOLDEN_H
#define GEMM_GROUPED_BATCHED_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cblas.h>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// ── CBLAS enum mapping ──

inline CBLAS_TRANSPOSE aclblasToCblasTrans(aclblasOperation_t op)
{
    switch (op) {
    case ACLBLAS_OP_N: return CblasNoTrans;
    case ACLBLAS_OP_T: return CblasTrans;
    default:           return CblasNoTrans;  // fallback for invalid values
    }
}

// ── Parameter validation (matches requirement doc §3.1 28 rules) ──

inline aclblasStatus_t validateGemmGroupedBatchedGroupDimensions(
    int m, int n, int k, int groupSize)
{
    if (m < 0 || n < 0 || k < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (groupSize < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t validateGemmGroupedBatchedGroupTranspose(
    aclblasOperation_t transa, aclblasOperation_t transb)
{
    // S supports only N and T.
    if (transa != ACLBLAS_OP_N && transa != ACLBLAS_OP_T) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (transb != ACLBLAS_OP_N && transb != ACLBLAS_OP_T) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t validateGemmGroupedBatchedGroupLeadingDims(
    aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, int lda, int ldb, int ldc)
{
    if (transa == ACLBLAS_OP_N && lda < std::max(1, m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (transa == ACLBLAS_OP_T && lda < std::max(1, k)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (transb == ACLBLAS_OP_N && ldb < std::max(1, k)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (transb == ACLBLAS_OP_T && ldb < std::max(1, n)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t validateGemmGroupedBatchedParams(
    aclblasHandle_t handle, int groupCount,
    const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const int* ldaArray, const int* ldbArray, const int* ldcArray,
    const int* groupSizeArray)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (groupCount < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (groupCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (transaArray == nullptr || transbArray == nullptr ||
        mArray == nullptr || nArray == nullptr || kArray == nullptr ||
        ldaArray == nullptr || ldbArray == nullptr || ldcArray == nullptr ||
        groupSizeArray == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    for (int g = 0; g < groupCount; g++) {
        aclblasStatus_t ret = validateGemmGroupedBatchedGroupDimensions(
            mArray[g], nArray[g], kArray[g], groupSizeArray[g]);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
        ret = validateGemmGroupedBatchedGroupTranspose(transaArray[g], transbArray[g]);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
        ret = validateGemmGroupedBatchedGroupLeadingDims(
            transaArray[g], transbArray[g],
            mArray[g], nArray[g], kArray[g],
            ldaArray[g], ldbArray[g], ldcArray[g]);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
    }

    return ACLBLAS_STATUS_SUCCESS;
}

// ── Compute matrix sizes ──

inline int getACols(aclblasOperation_t transa, int m, int k)
{
    return (transa == ACLBLAS_OP_N) ? k : m;
}

inline int getBCols(aclblasOperation_t transb, int k, int n)
{
    return (transb == ACLBLAS_OP_N) ? n : k;
}

// ── S type golden: cblas_sgemm ──

inline void applyAlphaZeroGolden(float* c, int m, int n, int ldc, float beta)
{
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            c[col * ldc + row] = beta * c[col * ldc + row];
        }
    }
}

inline void computeGroupedBatchGemmItem(
    int batchIdx, int g, int m, int n, int k, float alpha,
    CBLAS_TRANSPOSE cTransa, CBLAS_TRANSPOSE cTransb,
    const float* alphaArray, const float* betaArray,
    const float* const* Aarray, const float* const* Barray, float* const* Carray,
    const int* ldaArray, const int* ldbArray, const int* ldcArray)
{
    // BLAS standard: when alpha=0, C = beta × C_original (no dependency on A/B).
    // cblas_sgemm may compute 0 × inf = nan; handle alpha=0 explicitly.
    if (alpha == 0.0f) {
        applyAlphaZeroGolden(Carray[batchIdx], m, n, ldcArray[g], betaArray[g]);
        return;
    }
    cblas_sgemm(CblasColMajor, cTransa, cTransb,
                m, n, k, alphaArray[g],
                Aarray[batchIdx], ldaArray[g],
                Barray[batchIdx], ldbArray[g],
                betaArray[g],
                Carray[batchIdx], ldcArray[g]);
}

inline aclblasStatus_t aclblasSgemmGroupedBatched_cpu(
    aclblasHandle_t handle, int groupCount,
    const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray,
    const float* const* Aarray, const int* ldaArray,
    const float* const* Barray, const int* ldbArray,
    const float* betaArray,
    float* const* Carray, const int* ldcArray,
    const int* groupSizeArray)
{
    if (alphaArray == nullptr || Aarray == nullptr || Barray == nullptr || Carray == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclblasStatus_t v = validateGemmGroupedBatchedParams(
        handle, groupCount, transaArray, transbArray, mArray, nArray, kArray,
        ldaArray, ldbArray, ldcArray, groupSizeArray);
    if (v != ACLBLAS_STATUS_SUCCESS) {
        return v;
    }

    int batchIdx = 0;
    for (int g = 0; g < groupCount; g++) {
        int m = mArray[g];
        int n = nArray[g];
        int k = kArray[g];
        if (m == 0 || n == 0 || groupSizeArray[g] == 0) {
            batchIdx += groupSizeArray[g];
            continue;
        }

        CBLAS_TRANSPOSE cTransa = aclblasToCblasTrans(transaArray[g]);
        CBLAS_TRANSPOSE cTransb = aclblasToCblasTrans(transbArray[g]);
        float alpha = alphaArray[g];

        for (int i = 0; i < groupSizeArray[g]; i++) {
            computeGroupedBatchGemmItem(
                batchIdx, g, m, n, k, alpha, cTransa, cTransb,
                alphaArray, betaArray, Aarray, Barray, Carray,
                ldaArray, ldbArray, ldcArray);
            batchIdx++;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // GEMM_GROUPED_BATCHED_GOLDEN_H