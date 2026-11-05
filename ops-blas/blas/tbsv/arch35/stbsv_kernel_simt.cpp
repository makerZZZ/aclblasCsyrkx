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
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "stbsv_tiling_data.h"
#include "common/helper/kernel_constant.h"

using namespace AscendC;

// partialSums 按 threadIdx.x 索引，大小须 >= SIMT_MAX_THREAD_NUM。
constexpr uint32_t UB_PARTIALS_FLOATS = SIMT_MAX_THREAD_NUM;

// Profiling 实测 UB_X_FLOATS=4096/8192/12288 性能差异 <5%，
// 保持 4096(16KB) 给 DCache 留最大空间(~224KB)。
constexpr uint32_t UB_X_FLOATS = 4096;

__simt_callee__ inline int64_t TbsvSimtBandIdxUpper(uint32_t row, uint32_t col, uint32_t k, uint32_t lda)
{
    return static_cast<int64_t>(k) + static_cast<int64_t>(row) - static_cast<int64_t>(col)
           + static_cast<int64_t>(col) * static_cast<int64_t>(lda);
}

__simt_callee__ inline int64_t TbsvSimtBandIdxLower(uint32_t row, uint32_t col, uint32_t lda)
{
    return static_cast<int64_t>(row) - static_cast<int64_t>(col)
           + static_cast<int64_t>(col) * static_cast<int64_t>(lda);
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_TRANS>
__simt_callee__ inline int64_t TbsvSimtBandIdx(uint32_t row, uint32_t col, uint32_t k, uint32_t lda)
{
    if constexpr (UPLO_IS_UPPER) {
        if constexpr (TRANS_IS_TRANS) {
            return TbsvSimtBandIdxUpper(col, row, k, lda);
        } else {
            return TbsvSimtBandIdxUpper(row, col, k, lda);
        }
    } else {
        if constexpr (TRANS_IS_TRANS) {
            return TbsvSimtBandIdxLower(col, row, lda);
        } else {
            return TbsvSimtBandIdxLower(row, col, lda);
        }
    }
}

__simt_callee__ inline int64_t TbsvSimtXOffset(uint32_t idx, uint32_t n, int32_t incx)
{
    if (incx >= 0) {
        return static_cast<int64_t>(idx) * static_cast<int64_t>(incx);
    } else {
        return static_cast<int64_t>(n - 1 - idx) * static_cast<int64_t>(-incx);
    }
}

__simt_callee__ inline float TbsvSimtReduce(float partial, __ubuf__ float* partialSums)
{
    partialSums[threadIdx.x] = partial;
    asc_syncthreads();

    uint32_t n = blockDim.x;
    uint32_t pow2 = 1;
    while (pow2 * 2 <= n) {
        pow2 *= 2;
    }
    uint32_t tail = n - pow2;
    if (tail > 0) {
        if (threadIdx.x < tail) {
            partialSums[threadIdx.x] += partialSums[threadIdx.x + pow2];
        }
        asc_syncthreads();
    }
    for (uint32_t stride = pow2 / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
        }
        asc_syncthreads();
    }

    return partialSums[0];
}

__simt_callee__ inline void TbsvSimtLoadX(uint32_t n, int32_t incx, __gm__ float* xGm, __ubuf__ float* xUb)
{
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
        xUb[j] = xGm[TbsvSimtXOffset(j, n, incx)];
    }
    asc_syncthreads();
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_TRANS>
__simt_callee__ inline float TbsvSimtDotPartialGm(
    uint32_t row, uint32_t jStart, uint32_t jEnd, uint32_t k, uint32_t lda, uint32_t n, int32_t incx,
    __gm__ const float* aGm, __gm__ float* xGm)
{
    float partial = 0.0f;
    for (uint32_t j = jStart + threadIdx.x; j < jEnd; j += blockDim.x) {
        float aVal = aGm[TbsvSimtBandIdx<UPLO_IS_UPPER, TRANS_IS_TRANS>(row, j, k, lda)];
        partial += aVal * xGm[TbsvSimtXOffset(j, n, incx)];
    }
    return partial;
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_TRANS>
__simt_callee__ inline float TbsvSimtDotPartialUb(
    uint32_t row, uint32_t jStart, uint32_t jEnd, uint32_t k, uint32_t lda, __gm__ const float* aGm,
    __ubuf__ float* xUb)
{
    float partial = 0.0f;
    for (uint32_t j = jStart + threadIdx.x; j < jEnd; j += blockDim.x) {
        float aVal = aGm[TbsvSimtBandIdx<UPLO_IS_UPPER, TRANS_IS_TRANS>(row, j, k, lda)];
        partial += aVal * xUb[j];
    }
    return partial;
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool TRANS_IS_TRANS, bool TILED>
__simt_callee__ inline void TbsvSimtProcessRow(
    uint32_t row, uint32_t n, uint32_t k, uint32_t lda, int32_t incx, __gm__ const float* aGm, __gm__ float* xGm,
    __ubuf__ float* partialSums, __ubuf__ float* xUb, uint32_t jStart, uint32_t jEnd)
{
    int64_t xOff = TbsvSimtXOffset(row, n, incx);
    float sum = TILED ? xGm[xOff] : xUb[row];

    float partial;
    if constexpr (TILED) {
        partial = TbsvSimtDotPartialGm<UPLO_IS_UPPER, TRANS_IS_TRANS>(row, jStart, jEnd, k, lda, n, incx, aGm, xGm);
    } else {
        partial = TbsvSimtDotPartialUb<UPLO_IS_UPPER, TRANS_IS_TRANS>(row, jStart, jEnd, k, lda, aGm, xUb);
    }

    float total = TbsvSimtReduce(partial, partialSums);
    if (threadIdx.x == 0) {
        sum -= total;
        if constexpr (!DIAG_IS_UNIT) {
            float diagVal;
            if constexpr (UPLO_IS_UPPER) {
                diagVal = aGm[static_cast<int64_t>(k) + static_cast<int64_t>(row) * static_cast<int64_t>(lda)];
            } else {
                diagVal = aGm[static_cast<int64_t>(row) * static_cast<int64_t>(lda)];
            }
            sum = sum / diagVal;
        }
        xGm[xOff] = sum;
        if constexpr (!TILED) {
            xUb[row] = sum;
        }
    }
    asc_syncthreads();
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool TRANS_IS_TRANS, bool FORWARD, bool TILED>
__simt_callee__ inline void TbsvSimtProcess(
    uint32_t n, uint32_t k, uint32_t lda, int32_t incx, __gm__ const float* aGm, __gm__ float* xGm,
    __ubuf__ float* partialSums, __ubuf__ float* xUb)
{
    if constexpr (!TILED) {
        TbsvSimtLoadX(n, incx, xGm, xUb);
    }

    if constexpr (FORWARD) {
        for (uint32_t row = 0; row < n; ++row) {
            uint32_t jStart = (row >= k) ? (row - k) : 0;
            TbsvSimtProcessRow<UPLO_IS_UPPER, DIAG_IS_UNIT, TRANS_IS_TRANS, TILED>(
                row, n, k, lda, incx, aGm, xGm, partialSums, xUb, jStart, row);
        }
    } else {
        for (uint32_t row = n; row-- > 0;) {
            uint32_t jEnd = (row + k + 1 < n) ? (row + k + 1) : n;
            TbsvSimtProcessRow<UPLO_IS_UPPER, DIAG_IS_UNIT, TRANS_IS_TRANS, TILED>(
                row, n, k, lda, incx, aGm, xGm, partialSums, xUb, row + 1, jEnd);
        }
    }
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool TRANS_IS_TRANS>
__simt_callee__ inline void StbsvSimtTiled(
    uint32_t n, uint32_t k, uint32_t lda, int32_t incx, __gm__ const float* aGm, __gm__ float* xGm)
{
    __ubuf__ float partialSums[UB_PARTIALS_FLOATS];
    __ubuf__ float dummyXUb[1];
    constexpr bool kForward = (!UPLO_IS_UPPER && !TRANS_IS_TRANS) || (UPLO_IS_UPPER && TRANS_IS_TRANS);

    if constexpr (kForward) {
        TbsvSimtProcess<UPLO_IS_UPPER, DIAG_IS_UNIT, TRANS_IS_TRANS, true, true>(
            n, k, lda, incx, aGm, xGm, partialSums, dummyXUb);
    } else {
        TbsvSimtProcess<UPLO_IS_UPPER, DIAG_IS_UNIT, TRANS_IS_TRANS, false, true>(
            n, k, lda, incx, aGm, xGm, partialSums, dummyXUb);
    }
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool TRANS_IS_TRANS>
__simt_callee__ inline void StbsvSimtUb(
    uint32_t n, uint32_t k, uint32_t lda, int32_t incx, __gm__ const float* aGm, __gm__ float* xGm)
{
    __ubuf__ float partialSums[UB_PARTIALS_FLOATS];
    __ubuf__ float xUb[UB_X_FLOATS];
    constexpr bool kForward = (!UPLO_IS_UPPER && !TRANS_IS_TRANS) || (UPLO_IS_UPPER && TRANS_IS_TRANS);

    if constexpr (kForward) {
        TbsvSimtProcess<UPLO_IS_UPPER, DIAG_IS_UNIT, TRANS_IS_TRANS, true, false>(
            n, k, lda, incx, aGm, xGm, partialSums, xUb);
    } else {
        TbsvSimtProcess<UPLO_IS_UPPER, DIAG_IS_UNIT, TRANS_IS_TRANS, false, false>(
            n, k, lda, incx, aGm, xGm, partialSums, xUb);
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_TRANS, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StbsvSimt(
    uint32_t n, uint32_t k, uint32_t lda, int32_t incx, __gm__ const float* aGm, __gm__ float* xGm)
{
    if (n <= UB_X_FLOATS) {
        StbsvSimtUb<UPLO_IS_UPPER, DIAG_IS_UNIT, TRANS_IS_TRANS>(n, k, lda, incx, aGm, xGm);
    } else {
        StbsvSimtTiled<UPLO_IS_UPPER, DIAG_IS_UNIT, TRANS_IS_TRANS>(n, k, lda, incx, aGm, xGm);
    }
}

__global__ __aicore__ void stbsv_simt_kernel(StbsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* aGm = reinterpret_cast<__gm__ float*>(tiling.a);
    auto* xGm = reinterpret_cast<__gm__ float*>(tiling.x);
    uint32_t ldaU32 = tiling.lda;

    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StbsvSimt<false, false, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, ldaU32, tiling.incx, aGm, xGm);
            } else {
                asc_vf_call<StbsvSimt<false, false, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, ldaU32, tiling.incx, aGm, xGm);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StbsvSimt<false, true, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, ldaU32, tiling.incx, aGm, xGm);
            } else {
                asc_vf_call<StbsvSimt<false, true, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, ldaU32, tiling.incx, aGm, xGm);
            }
        }
    } else {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StbsvSimt<true, false, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, ldaU32, tiling.incx, aGm, xGm);
            } else {
                asc_vf_call<StbsvSimt<true, false, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, ldaU32, tiling.incx, aGm, xGm);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StbsvSimt<true, true, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, ldaU32, tiling.incx, aGm, xGm);
            } else {
                asc_vf_call<StbsvSimt<true, true, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, ldaU32, tiling.incx, aGm, xGm);
            }
        }
    }
}

void stbsv_simt_kernel_do(const StbsvTilingData& tiling, void* stream)
{
    stbsv_simt_kernel<<<1, nullptr, stream>>>(tiling);
}
