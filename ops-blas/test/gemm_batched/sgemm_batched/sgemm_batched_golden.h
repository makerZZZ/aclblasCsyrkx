/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGEMM_BATCHED_GOLDEN_H
#define SGEMM_BATCHED_GOLDEN_H

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "sgemm_batched_param.h"

inline void ApplyAlphaZeroGolden(float* c, int m, int n, int ldc, float beta)
{
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            c[col * ldc + row] = beta * c[col * ldc + row];
        }
    }
}

inline SGEMM_BATCHED_SIGNATURE(aclblasSgemmBatched_cpu)
{
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (alpha == nullptr || beta == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0 || batchCount < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (batchCount == 0 || m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;

    float alphaVal = *alpha;
    float betaVal = *beta;

    CBLAS_TRANSPOSE cTransA = ToCblasOp(transA);
    CBLAS_TRANSPOSE cTransB = ToCblasOp(transB);

    for (int batch = 0; batch < batchCount; batch++) {
        const float* aData = Aarray ? Aarray[batch] : nullptr;
        const float* bData = Barray ? Barray[batch] : nullptr;
        float* cData = Carray ? Carray[batch] : nullptr;
        if (cData == nullptr) continue;

        if (alphaVal == 0.0f) {
            ApplyAlphaZeroGolden(cData, m, n, ldc, betaVal);
            continue;
        }

        cblas_sgemm(CblasColMajor, cTransA, cTransB,
                    m, n, k, alphaVal,
                    aData, lda, bData, ldb,
                    betaVal, cData, ldc);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // SGEMM_BATCHED_GOLDEN_H
