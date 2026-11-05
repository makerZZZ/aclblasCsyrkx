/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file strmv_kernel.cpp
 * \brief single-precision triangular matrix-vector multiply kernels for ascend950
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#include "strmv_common.h"

__simt_callee__ __aicore__ __inline__ int64_t StrmvVectorIndex(uint32_t idx, uint32_t n, int64_t incx)
{
    return (incx >= 0) ? static_cast<int64_t>(idx) * incx : static_cast<int64_t>(n - 1 - idx) * (-incx);
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StrmvCompute(
    uint32_t n, uint32_t lda, int64_t incx, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* midOutGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        uint32_t colStart = 0;
        uint32_t colEnd = n;

        if constexpr (TRANS_IS_N) {
            if constexpr (UPLO_IS_UPPER) {
                colStart = row;
            } else {
                colEnd = row + 1;
            }
        } else {
            if constexpr (UPLO_IS_UPPER) {
                colEnd = row + 1;
            } else {
                colStart = row;
            }
        }

        for (uint32_t col = colStart; col < colEnd; ++col) {
            float aVal = 1.0f;
            if constexpr (!DIAG_IS_UNIT) {
                aVal = TRANS_IS_N ? aGm[row + static_cast<int64_t>(lda) * col] :
                                    aGm[col + static_cast<int64_t>(lda) * row];
            } else {
                if (col != row) {
                    aVal = TRANS_IS_N ? aGm[row + static_cast<int64_t>(lda) * col] :
                                        aGm[col + static_cast<int64_t>(lda) * row];
                }
            }
            acc += aVal * xGm[StrmvVectorIndex(col, n, incx)];
        }
        midOutGm[row] = acc;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StrmvCopyBack(
    uint32_t n, int64_t incx, __gm__ const float* midOutGm, __gm__ float* xGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        xGm[StrmvVectorIndex(row, n, incx)] = midOutGm[row];
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N>
__aicore__ __inline__ void DispatchDiag(
    uint32_t diag, uint32_t numThreads, uint32_t n, uint32_t lda, int64_t incx, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* midOutGm)
{
    if (diag == ACLBLAS_UNIT) {
        asc_vf_call<StrmvCompute<UPLO_IS_UPPER, TRANS_IS_N, true>>(
            dim3{numThreads, 1, 1}, n, lda, incx, aGm, xGm, midOutGm);
    } else {
        asc_vf_call<StrmvCompute<UPLO_IS_UPPER, TRANS_IS_N, false>>(
            dim3{numThreads, 1, 1}, n, lda, incx, aGm, xGm, midOutGm);
    }
}

__global__ __aicore__ void strmv_compute_kernel(
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ uint8_t* workspaceGm, StrmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ float* midOutGm = reinterpret_cast<__gm__ float*>(workspaceGm);
    bool transIsN = (tdata.trans == ACLBLAS_OP_N);
    if (tdata.uplo == ACLBLAS_UPPER) {
        if (transIsN) {
            DispatchDiag<true, true>(tdata.diag, tdata.numThreads, tdata.n, tdata.lda, tdata.incx, aGm, xGm, midOutGm);
        } else {
            DispatchDiag<true, false>(tdata.diag, tdata.numThreads, tdata.n, tdata.lda, tdata.incx, aGm, xGm, midOutGm);
        }
    } else {
        if (transIsN) {
            DispatchDiag<false, true>(tdata.diag, tdata.numThreads, tdata.n, tdata.lda, tdata.incx, aGm, xGm, midOutGm);
        } else {
            DispatchDiag<false, false>(
                tdata.diag, tdata.numThreads, tdata.n, tdata.lda, tdata.incx, aGm, xGm, midOutGm);
        }
    }
}

__global__ __aicore__ void strmv_copy_kernel(
    __gm__ float* xGm, __gm__ const uint8_t* workspaceGm, StrmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ const float* midOutGm = reinterpret_cast<__gm__ const float*>(workspaceGm);
    asc_vf_call<StrmvCopyBack>(dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.incx, midOutGm, xGm);
}

void strmv_arch35_kernel_do(
    const float* a, float* x, uint8_t* workspace, const StrmvTilingData& tilingData, uint32_t numBlocks, void* stream)
{
    strmv_compute_kernel<<<numBlocks, nullptr, stream>>>(a, x, workspace, tilingData);
    strmv_copy_kernel<<<numBlocks, nullptr, stream>>>(x, workspace, tilingData);
}
