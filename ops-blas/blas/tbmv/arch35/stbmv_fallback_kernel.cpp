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
 * \file stbmv_fallback_kernel.cpp
 * \brief single-precision triangular band matrix-vector multiply SIMT fallback kernels for ascend950
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "stbmv_common.h"

__simt_callee__ __aicore__ __inline__ int64_t StbmvVectorIndex(uint32_t idx, uint32_t n, int64_t incx)
{
    int64_t stride = incx;
    return (stride > 0) ? static_cast<int64_t>(idx) * stride :
                          static_cast<int64_t>(n - 1U - idx) * (-stride);
}

template <bool UPLO_IS_UPPER>
__simt_callee__ __aicore__ __inline__ int64_t StbmvBandIndex(uint32_t row, uint32_t col, uint32_t k, uint32_t lda)
{
    int64_t rowValue = static_cast<int64_t>(row);
    int64_t colValue = static_cast<int64_t>(col);
    int64_t ldaValue = static_cast<int64_t>(lda);
    if constexpr (UPLO_IS_UPPER) {
        return static_cast<int64_t>(k) + rowValue - colValue + colValue * ldaValue;
    } else {
        return rowValue - colValue + colValue * ldaValue;
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N>
__simt_callee__ __aicore__ __inline__ void StbmvColumnRange(
    uint32_t row, uint32_t n, uint32_t k, uint32_t& colStart, uint32_t& colEnd)
{
    if constexpr (TRANS_IS_N) {
        if constexpr (UPLO_IS_UPPER) {
            colStart = row;
            uint64_t end = static_cast<uint64_t>(row) + k + 1U;
            colEnd = (end < n) ? static_cast<uint32_t>(end) : n;
        } else {
            colStart = (row >= k) ? (row - k) : 0U;
            colEnd = row + 1U;
        }
    } else {
        if constexpr (UPLO_IS_UPPER) {
            colStart = (row >= k) ? (row - k) : 0U;
            colEnd = row + 1U;
        } else {
            colStart = row;
            uint64_t end = static_cast<uint64_t>(row) + k + 1U;
            colEnd = (end < n) ? static_cast<uint32_t>(end) : n;
        }
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool DIAG_IS_UNIT>
__simt_callee__ __aicore__ __inline__ float StbmvLoadA(
    uint32_t row, uint32_t col, uint32_t k, uint32_t lda, __gm__ const float* aGm)
{
    if constexpr (DIAG_IS_UNIT) {
        if (row == col) {
            return 1.0f;
        }
    }
    uint32_t aRow = TRANS_IS_N ? row : col;
    uint32_t aCol = TRANS_IS_N ? col : row;
    return aGm[StbmvBandIndex<UPLO_IS_UPPER>(aRow, aCol, k, lda)];
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StbmvComputeGm(uint32_t n, uint32_t k,
    uint32_t lda, int64_t incx, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* midOutGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        uint32_t colStart = 0;
        uint32_t colEnd = 0;
        StbmvColumnRange<UPLO_IS_UPPER, TRANS_IS_N>(row, n, k, colStart, colEnd);

        float acc = 0.0f;
        for (uint32_t col = colStart; col < colEnd; ++col) {
            float aVal = StbmvLoadA<UPLO_IS_UPPER, TRANS_IS_N, DIAG_IS_UNIT>(row, col, k, lda, aGm);
            acc += aVal * xGm[StbmvVectorIndex(col, n, incx)];
        }
        midOutGm[row] = acc;
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StbmvComputeUb(uint32_t n, uint32_t k,
    uint32_t lda, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* midOutGm, uint32_t rowStart,
    uint32_t rowEnd, uint32_t xBase, uint32_t xLen)
{
    __ubuf__ float xUb[STBMV_UB_X_FLOATS];

    for (uint32_t i = threadIdx.x; i < xLen; i += blockDim.x) {
        xUb[i] = xGm[static_cast<int64_t>(xBase) + i];
    }
    asc_syncthreads();

    for (uint32_t row = rowStart + threadIdx.x; row < rowEnd; row += blockDim.x) {
        uint32_t colStart = 0;
        uint32_t colEnd = 0;
        StbmvColumnRange<UPLO_IS_UPPER, TRANS_IS_N>(row, n, k, colStart, colEnd);

        float acc = 0.0f;
        for (uint32_t col = colStart; col < colEnd; ++col) {
            float aVal = StbmvLoadA<UPLO_IS_UPPER, TRANS_IS_N, DIAG_IS_UNIT>(row, col, k, lda, aGm);
            acc += aVal * xUb[col - xBase];
        }
        midOutGm[row] = acc;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StbmvCopyBack(
    uint32_t n, int64_t incx, __gm__ const float* midOutGm, __gm__ float* xGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        xGm[StbmvVectorIndex(row, n, incx)] = midOutGm[row];
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StbmvDiagScale(
    uint32_t n, uint32_t lda, int64_t incx, __gm__ const float* aGm, __gm__ float* xGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        int64_t xIdx = StbmvVectorIndex(row, n, incx);
        int64_t aIdx = static_cast<int64_t>(row) * static_cast<int64_t>(lda);
        xGm[xIdx] = aGm[aIdx] * xGm[xIdx];
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N>
__aicore__ __inline__ void DispatchDiagGm(uint32_t diag, const StbmvTilingData& tdata, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* midOutGm)
{
    if (diag == ACLBLAS_UNIT) {
        asc_vf_call<StbmvComputeGm<UPLO_IS_UPPER, TRANS_IS_N, true>>(
            dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.k, tdata.lda, tdata.incx, aGm, xGm, midOutGm);
    } else {
        asc_vf_call<StbmvComputeGm<UPLO_IS_UPPER, TRANS_IS_N, false>>(
            dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.k, tdata.lda, tdata.incx, aGm, xGm, midOutGm);
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N>
__aicore__ __inline__ void DispatchDiagUb(uint32_t diag, const StbmvTilingData& tdata, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* midOutGm, uint32_t rowStart, uint32_t rowEnd, uint32_t xBase,
    uint32_t xLen)
{
    if (diag == ACLBLAS_UNIT) {
        asc_vf_call<StbmvComputeUb<UPLO_IS_UPPER, TRANS_IS_N, true>>(dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.k,
            tdata.lda, aGm, xGm, midOutGm, rowStart, rowEnd, xBase, xLen);
    } else {
        asc_vf_call<StbmvComputeUb<UPLO_IS_UPPER, TRANS_IS_N, false>>(dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.k,
            tdata.lda, aGm, xGm, midOutGm, rowStart, rowEnd, xBase, xLen);
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N>
__aicore__ __inline__ void DispatchComputeGm(
    const StbmvTilingData& tdata, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* midOutGm)
{
    DispatchDiagGm<UPLO_IS_UPPER, TRANS_IS_N>(tdata.diag, tdata, aGm, xGm, midOutGm);
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_N>
__aicore__ __inline__ void DispatchComputeUb(
    const StbmvTilingData& tdata, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* midOutGm)
{
    uint32_t blkIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
    uint32_t rowStart = blkIdx * tdata.rowsPerBlock;
    uint32_t rowEnd = rowStart + tdata.rowsPerBlock;
    rowEnd = (rowEnd < tdata.n) ? rowEnd : tdata.n;
    if (rowStart >= rowEnd) {
        return;
    }

    uint32_t xBase = 0;
    uint32_t xEnd = 0;
    if constexpr ((TRANS_IS_N && UPLO_IS_UPPER) || (!TRANS_IS_N && !UPLO_IS_UPPER)) {
        xBase = rowStart;
        uint64_t end = static_cast<uint64_t>(rowEnd) + tdata.k;
        xEnd = (end < tdata.n) ? static_cast<uint32_t>(end) : tdata.n;
    } else {
        xBase = (rowStart >= tdata.k) ? (rowStart - tdata.k) : 0U;
        xEnd = rowEnd;
    }
    uint32_t xLen = xEnd - xBase;
    if (xLen == 0 || xLen > STBMV_UB_X_FLOATS) {
        DispatchComputeGm<UPLO_IS_UPPER, TRANS_IS_N>(tdata, aGm, xGm, midOutGm);
        return;
    }
    DispatchDiagUb<UPLO_IS_UPPER, TRANS_IS_N>(tdata.diag, tdata, aGm, xGm, midOutGm, rowStart, rowEnd, xBase, xLen);
}

template <bool UPLO_IS_UPPER>
__aicore__ __inline__ void DispatchTrans(
    const StbmvTilingData& tdata, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* midOutGm)
{
    bool transIsN = (tdata.trans == ACLBLAS_OP_N);
    if (tdata.useUb != 0U) {
        if (transIsN) {
            DispatchComputeUb<UPLO_IS_UPPER, true>(tdata, aGm, xGm, midOutGm);
        } else {
            DispatchComputeUb<UPLO_IS_UPPER, false>(tdata, aGm, xGm, midOutGm);
        }
        return;
    }

    if (transIsN) {
        DispatchComputeGm<UPLO_IS_UPPER, true>(tdata, aGm, xGm, midOutGm);
    } else {
        DispatchComputeGm<UPLO_IS_UPPER, false>(tdata, aGm, xGm, midOutGm);
    }
}

__global__ __aicore__ void stbmv_compute_kernel(
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ uint8_t* workspaceGm, StbmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ float* midOutGm = reinterpret_cast<__gm__ float*>(workspaceGm);
    if (tdata.uplo == ACLBLAS_UPPER) {
        DispatchTrans<true>(tdata, aGm, xGm, midOutGm);
    } else {
        DispatchTrans<false>(tdata, aGm, xGm, midOutGm);
    }
}

__global__ __aicore__ void stbmv_copy_kernel(
    __gm__ float* xGm, __gm__ const uint8_t* workspaceGm, StbmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ const float* midOutGm = reinterpret_cast<__gm__ const float*>(workspaceGm);
    asc_vf_call<StbmvCopyBack>(dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.incx, midOutGm, xGm);
}

__global__ __aicore__ void stbmv_diag_kernel(__gm__ const float* aGm, __gm__ float* xGm, StbmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    asc_vf_call<StbmvDiagScale>(dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.lda, tdata.incx, aGm, xGm);
}

void stbmv_arch35_kernel_do(
    const float* a, float* x, uint8_t* workspace, const StbmvTilingData& tilingData, uint32_t numBlocks, void* stream)
{
    if (tilingData.k == 0U) {
        stbmv_diag_kernel<<<numBlocks, nullptr, stream>>>(a, x, tilingData);
        return;
    }
    stbmv_compute_kernel<<<numBlocks, nullptr, stream>>>(a, x, workspace, tilingData);
    stbmv_copy_kernel<<<numBlocks, nullptr, stream>>>(x, workspace, tilingData);
}
