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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline aclblasStatus_t validateGemvStridedBatchedCpuParams(
    aclblasOperation_t trans, int m, int n, int lda, int incx, int incy, int batchCount,
    int64_t strideA, int64_t stridex, int64_t stridey,
    const float* a, const float* x, float* y)
{
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C)
        return ACLBLAS_STATUS_INVALID_ENUM;
    if (m < 0 || n < 0 || batchCount < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, m))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0 || incx == INT32_MIN || incy == 0 || incy == INT32_MIN)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (strideA <= 0 || stridex <= 0 || stridey <= 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (m > 0 && n > 0 && (a == nullptr || x == nullptr || y == nullptr))
        return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

inline void aclblasGemvStridedBatched_cpu(
    aclblasOperation_t trans, int m, int n, float alpha, const float* a, int lda, int64_t strideA,
    const float* x, int incx, int64_t stridex, float beta, float* y, int incy, int64_t stridey, int batchCount)
{
    if (m == 0 || n == 0 || batchCount == 0)
        return;

    aclblasStatus_t validRet = validateGemvStridedBatchedCpuParams(
        trans, m, n, lda, incx, incy, batchCount, strideA, stridex, stridey, a, x, y);
    if (validRet != ACLBLAS_STATUS_SUCCESS)
        return;

    CBLAS_TRANSPOSE cblasTrans = ToCblasOp(trans);
    for (int b = 0; b < batchCount; b++) {
        const float* aBatch = a + static_cast<int64_t>(b) * strideA;
        const float* xBatch = x + static_cast<int64_t>(b) * stridex;
        float* yBatch = y + static_cast<int64_t>(b) * stridey;
        cblas_sgemv(CblasColMajor, cblasTrans, m, n, alpha, aBatch, lda, xBatch, incx, beta, yBatch, incy);
    }
}
