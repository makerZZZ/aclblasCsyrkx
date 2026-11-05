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

#include "gemm_batched_ex_golden.h"
#include "gemm_strided_batched_ex_param.h"
#include "cblas_compat.h"

inline void GemmStridedBatchedExGolden(const GemmStridedBatchedExParam& p, const float* a, const float* b, float* c)
{
    for (int batch = 0; batch < p.batchCount; ++batch) {
        const float* aBatch = a + static_cast<int64_t>(batch) * p.strideA;
        const float* bBatch = b + static_cast<int64_t>(batch) * p.strideB;
        float* cBatch = c + static_cast<int64_t>(batch) * p.strideC;
        if (p.k == 0 || p.alpha == 0.0f) {
            for (int col = 0; col < p.n; ++col) {
                for (int row = 0; row < p.m; ++row) {
                    const size_t offset = static_cast<size_t>(col) * p.ldc + row;
                    cBatch[offset] = p.beta == 0.0f ? 0.0f : p.beta * cBatch[offset];
                }
            }
            continue;
        }
        cblas_sgemm(
            CblasColMajor, ToCblasOp(p.transA), ToCblasOp(p.transB), p.m, p.n, p.k, p.alpha, aBatch, p.lda, bBatch,
            p.ldb, p.beta, cBatch, p.ldc);
        if (p.Ctype != ACL_FLOAT) {
            for (int col = 0; col < p.n; ++col) {
                for (int row = 0; row < p.m; ++row) {
                    const size_t offset = static_cast<size_t>(col) * p.ldc + row;
                    cBatch[offset] = ApplyBatchedAlphaBetaAndQuantize(
                        static_cast<double>(cBatch[offset]), 0.0f, 1.0f, 0.0f, p.Atype, p.Btype, p.Ctype);
                }
            }
        }
    }
}
