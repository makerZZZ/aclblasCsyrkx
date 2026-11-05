/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BATCHED_EX_GOLDEN_H
#define GEMM_BATCHED_EX_GOLDEN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <securec.h>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "dtype_cast.h"
#include "gemm_batched_ex_param.h"
// ═══════════════════════════════════════════════════════════════════════════════
// CPU golden helpers
// ═══════════════════════════════════════════════════════════════════════════════

// Validate batched GEMM parameters
inline aclblasStatus_t ValidateBatchedGoldenParams(
    aclblasHandle_t handle, const void* alpha, const void* beta,
    int m, int n, int k, int batchCount,
    aclblasOperation_t transA, aclblasOperation_t transB,
    int lda, int ldb, int ldc,
    const void* const Aarray[], const void* const Barray[], void* const Carray[])
{
    if (handle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (alpha == nullptr || beta == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (batchCount < 0) return ACLBLAS_STATUS_INVALID_VALUE;

    int physRowsA = (transA == ACLBLAS_OP_N) ? m : k;
    int physRowsB = (transB == ACLBLAS_OP_N) ? k : n;
    if (lda < std::max(1, physRowsA)) return ACLBLAS_STATUS_INVALID_VALUE;
    if (ldb < std::max(1, physRowsB)) return ACLBLAS_STATUS_INVALID_VALUE;
    if (ldc < std::max(1, m)) return ACLBLAS_STATUS_INVALID_VALUE;

    if (batchCount > 0) {
        if (Aarray == nullptr || Barray == nullptr || Carray == nullptr)
            return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Compute dot product for one element of C: sum(A[i,p] * B[p,j]) over p
// Accumulation precision matches NPU: FP32 for all compute types (NPU uses FP32 internal accumulator).
inline double ComputeBatchedDotProduct(
    const float* aData, const float* bData,
    int i, int j, int k, int lda, int ldb,
    aclblasOperation_t transA, aclblasOperation_t transB)
{
    if (k <= 0 || aData == nullptr || bData == nullptr) return 0.0;
    float sum = 0.0f;
    for (int p = 0; p < k; p++) {
        float aVal = (transA == ACLBLAS_OP_N)
            ? aData[static_cast<size_t>(p) * lda + i]
            : aData[static_cast<size_t>(i) * lda + p];
        float bVal = (transB == ACLBLAS_OP_N)
            ? bData[static_cast<size_t>(j) * ldb + p]
            : bData[static_cast<size_t>(p) * ldb + j];
        sum += aVal * bVal;
    }
    return static_cast<double>(sum);
}

// Apply alpha/beta and quantize output through Ctype
inline float ApplyBatchedAlphaBetaAndQuantize(
    double sum, float cOrig, float alphaVal, float betaVal,
    aclDataType Atype, aclDataType Btype, aclDataType Ctype)
{
    bool isFp8 = (Atype == ACL_FLOAT8_E4M3FN || Atype == ACL_FLOAT8_E5M2 ||
                  Btype == ACL_FLOAT8_E4M3FN || Btype == ACL_FLOAT8_E5M2);
    double result;
    if (isFp8) {
        float resF32 = (betaVal == 0.0f)
            ? alphaVal * static_cast<float>(sum)
            : alphaVal * static_cast<float>(sum) + betaVal * cOrig;
        result = static_cast<double>(resF32);
    } else {
        if (betaVal == 0.0) {
            result = static_cast<double>(alphaVal) * sum;
        } else {
            result = static_cast<double>(alphaVal) * sum + static_cast<double>(betaVal) * static_cast<double>(cOrig);
        }
    }
    float resFloat = static_cast<float>(result);
    if (isFp8 && !std::isfinite(resFloat)) {
        resFloat = NAN;
    }
    switch (Ctype) {
        case ACL_FLOAT16:
            resFloat = blas_common::HalfToFloat(blas_common::FloatToHalf(resFloat)); break;
        case ACL_BF16:
            resFloat = blas_common::Bf16ToFloat(blas_common::FloatToBf16(resFloat)); break;
        case ACL_FLOAT8_E4M3FN:
            resFloat = fp8E4m3ToFloat(floatToFp8E4m3(resFloat)); break;
        case ACL_FLOAT8_E5M2:
            resFloat = fp8E5m2ToFloat(floatToFp8E5m2(resFloat)); break;
        default: break;
    }
    return resFloat;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU golden: C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
//   for i in [0, batchCount)
//
// Signature matches the BLAS API (column-major storage).
// A, B, C are passed as float arrays (pre-quantized through the target dtype).
// FP16/BF16: uses FP64 accumulation (per test design doc §4.2)
// FP8: uses FP32 accumulation (per test design doc §4.2)
// ═══════════════════════════════════════════════════════════════════════════════

inline aclblasStatus_t aclblasGemmBatchedEx_cpu(
    aclblasHandle_t handle,
    aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k,
    const void* alpha,
    const void* const Aarray[], aclDataType Atype, int lda,
    const void* const Barray[], aclDataType Btype, int ldb,
    const void* beta,
    void* const Carray[], aclDataType Ctype, int ldc,
    int batchCount,
    aclblasComputeType_t computeType)
{
    (void)computeType;

    aclblasStatus_t status = ValidateBatchedGoldenParams(
        handle, alpha, beta, m, n, k, batchCount,
        transA, transB, lda, ldb, ldc, Aarray, Barray, Carray);
    if (status != ACLBLAS_STATUS_SUCCESS) return status;
    if (batchCount == 0 || m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;

    const float alphaVal = *static_cast<const float*>(alpha);
    const float betaVal  = *static_cast<const float*>(beta);

    for (int batch = 0; batch < batchCount; batch++) {
        const float* aData = static_cast<const float*>(Aarray[batch]);
        const float* bData = static_cast<const float*>(Barray[batch]);
        float* cData = static_cast<float*>(Carray[batch]);
        if (cData == nullptr) continue;

        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                double sum = ComputeBatchedDotProduct(
                    aData, bData, i, j, k, lda, ldb, transA, transB);
                float cOrig = cData[static_cast<size_t>(j) * ldc + i];
                cData[static_cast<size_t>(j) * ldc + i] = ApplyBatchedAlphaBetaAndQuantize(
                    sum, cOrig, alphaVal, betaVal, Atype, Btype, Ctype);
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // GEMM_BATCHED_EX_GOLDEN_H
