/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STRMM_GOLDEN_H
#define STRMM_GOLDEN_H

#include <securec.h>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline CBLAS_SIDE ToCblasSide(aclblasSideMode_t side)
{
    return (side == ACLBLAS_SIDE_LEFT) ? CblasLeft : CblasRight;
}

inline aclblasStatus_t aclblasStrmm_cpu(
    aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const float* alpha, const float* A,
    int lda, const float* B, int ldb, float* C, int ldc)
{
    if (m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (alpha == nullptr || C == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    // alpha==0 fast path: C = 0 regardless of A/B. Allows null A/B when alpha==0
    // and zeros padding to match the NPU fast path (bit-reproducible).
    if (*alpha == 0.0f) {
        size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(float);
        if (memset_s(C, cBytes, 0, cBytes) != EOK) return ACLBLAS_STATUS_INTERNAL_ERROR;
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (A == nullptr || B == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    // Column-major: C is ldc×n (ldc = column stride, n columns, m rows).
    // B→C copy is column-by-column; when ldb==ldc the whole ldb*n block can be
    // copied in one shot.
    size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(float);
    size_t colBytes = static_cast<size_t>(m) * sizeof(float);
    if (ldc == ldb) {
        if (memcpy_s(C, cBytes, B, cBytes) != EOK) return ACLBLAS_STATUS_INTERNAL_ERROR;
    } else {
        for (int j = 0; j < n; ++j) {
            size_t dstOffset = static_cast<size_t>(j) * static_cast<size_t>(ldc);
            size_t dstRemain = cBytes - dstOffset * sizeof(float);
            if (memcpy_s(C + dstOffset, dstRemain, B + static_cast<size_t>(j) * ldb, colBytes) != EOK) {
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    cblas_strmm(CblasColMajor, ToCblasSide(side), ToCblasUplo(uplo),
        ToCblasOp(trans), ToCblasDiag(diag),
        m, n, *alpha, A, lda, C, ldc);
    return ACLBLAS_STATUS_SUCCESS;
}

#endif // STRMM_GOLDEN_H
