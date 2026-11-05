/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "csyr2_tiling_data.h"

// Keep the scalar FP32 operation sequence aligned with the CPU reference,
// including overflow/cancellation behavior in the RANDOM_EXTREME cases.
#pragma clang fp contract(off)

namespace {

// K1 keeps both contiguous complex input vectors in UB.  The capacity is in
// FP32 lanes, so it covers n <= 4096 complex elements without an A tile or
// double buffer.  All other layouts keep the K0 GM path below.
constexpr uint32_t K1_UB_FLOAT_LANES = 8192;

__simt_callee__ inline uint64_t VectorOffset(uint32_t index, uint32_t n, int64_t increment)
{
    if (increment >= 0) {
        return static_cast<uint64_t>(index) * static_cast<uint64_t>(increment);
    }
    return static_cast<uint64_t>(n - 1U - index) * static_cast<uint64_t>(-increment);
}

__simt_callee__ __aicore__ inline float2 ComplexMultiply(
    float leftReal, float leftImag, float rightReal, float rightImag)
{
    float realLeft = leftReal * rightReal;
    float realRight = leftImag * rightImag;
    float imagLeft = leftReal * rightImag;
    float imagRight = leftImag * rightReal;
    return {realLeft - realRight, imagLeft + imagRight};
}

__simt_callee__ __aicore__ inline void UpdateElement(float2 xCol, float2 yCol, float2 xRow, float2 yRow,
    float alphaReal, float alphaImag, uint32_t row, uint32_t col, uint32_t lda, __gm__ float* aGm)
{
    const float2 alphaY = ComplexMultiply(alphaReal, alphaImag, yRow.x, yRow.y);
    const float2 alphaX = ComplexMultiply(alphaReal, alphaImag, xRow.x, xRow.y);
    const float2 xy = ComplexMultiply(xCol.x, xCol.y, alphaY.x, alphaY.y);
    const float2 yx = ComplexMultiply(yCol.x, yCol.y, alphaX.x, alphaX.y);
    const uint64_t aOffset =
        2ULL * (static_cast<uint64_t>(col) * static_cast<uint64_t>(lda) + static_cast<uint64_t>(row));
    aGm[aOffset] = (aGm[aOffset] + xy.x) + yx.x;
    aGm[aOffset + 1ULL] = (aGm[aOffset + 1ULL] + xy.y) + yx.y;
}

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void Csyr2Gm(
    uint32_t n, uint32_t lda, float alphaReal, float alphaImag, int64_t incx, int64_t incy,
    __gm__ const float* xGm, __gm__ const float* yGm, __gm__ float* aGm)
{
    for (uint32_t col = blockIdx.x; col < n; col += gridDim.x) {
        const uint64_t xColOffset = 2ULL * VectorOffset(col, n, incx);
        const uint64_t yColOffset = 2ULL * VectorOffset(col, n, incy);
        const float2 xCol{xGm[xColOffset], xGm[xColOffset + 1ULL]};
        const float2 yCol{yGm[yColOffset], yGm[yColOffset + 1ULL]};
        const uint32_t rowBegin = UPLO_IS_UPPER ? 0U : col;
        const uint32_t rowEnd = UPLO_IS_UPPER ? col + 1U : n;

        for (uint32_t row = rowBegin + threadIdx.x; row < rowEnd; row += blockDim.x) {
            const uint64_t xRowOffset = 2ULL * VectorOffset(row, n, incx);
            const uint64_t yRowOffset = 2ULL * VectorOffset(row, n, incy);
            const float2 xRow{xGm[xRowOffset], xGm[xRowOffset + 1ULL]};
            const float2 yRow{yGm[yRowOffset], yGm[yRowOffset + 1ULL]};
            UpdateElement(xCol, yCol, xRow, yRow, alphaReal, alphaImag, row, col, lda, aGm);
        }
    }
}

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void Csyr2Ub(
    uint32_t n, uint32_t lda, float alphaReal, float alphaImag, __gm__ const float* xGm, __gm__ const float* yGm,
    __gm__ float* aGm)
{
    __ubuf__ float xUb[K1_UB_FLOAT_LANES];
    __ubuf__ float yUb[K1_UB_FLOAT_LANES];
    const uint32_t laneCount = 2U * n;
    for (uint32_t lane = threadIdx.x; lane < laneCount; lane += blockDim.x) {
        xUb[lane] = xGm[lane];
        yUb[lane] = yGm[lane];
    }
    asc_syncthreads();

    // Each core owns a disjoint group of columns.  Its selected elements form
    // a triangular column tile (UPPER) or trapezoidal lower tile (LOWER), so
    // every A coordinate still has one writer exactly as in K0.
    for (uint32_t col = blockIdx.x; col < n; col += gridDim.x) {
        const uint32_t colOffset = 2U * col;
        const float2 xCol{xUb[colOffset], xUb[colOffset + 1U]};
        const float2 yCol{yUb[colOffset], yUb[colOffset + 1U]};
        const uint32_t rowBegin = UPLO_IS_UPPER ? 0U : col;
        const uint32_t rowEnd = UPLO_IS_UPPER ? col + 1U : n;

        for (uint32_t row = rowBegin + threadIdx.x; row < rowEnd; row += blockDim.x) {
            const uint32_t rowOffset = 2U * row;
            const float2 xRow{xUb[rowOffset], xUb[rowOffset + 1U]};
            const float2 yRow{yUb[rowOffset], yUb[rowOffset + 1U]};
            UpdateElement(xCol, yCol, xRow, yRow, alphaReal, alphaImag, row, col, lda, aGm);
        }
    }
}

} // namespace
__global__ __aicore__ void csyr2_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR A, const Csyr2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ const float*>(y);
    auto* aGm = reinterpret_cast<__gm__ float*>(A);

    if (tiling.incx == 1 && tiling.incy == 1 && tiling.n >= 32U && tiling.n <= K1_UB_FLOAT_LANES / 2U) {
        if (tiling.uplo == ACLBLAS_UPPER) {
            asc_vf_call<Csyr2Ub<true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alphaReal, tiling.alphaImag, xGm, yGm,
                aGm);
        } else {
            asc_vf_call<Csyr2Ub<false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alphaReal, tiling.alphaImag, xGm, yGm,
                aGm);
        }
        return;
    }

    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<Csyr2Gm<true>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alphaReal, tiling.alphaImag, tiling.incx,
            tiling.incy, xGm, yGm, aGm);
    } else {
        asc_vf_call<Csyr2Gm<false>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.lda, tiling.alphaReal, tiling.alphaImag, tiling.incx,
            tiling.incy, xGm, yGm, aGm);
    }
}

void csyr2_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR A, const Csyr2TilingData& tiling, uint32_t numBlocks, void* stream)
{
    csyr2_kernel<<<numBlocks, nullptr, stream>>>(x, y, A, tiling);
}
