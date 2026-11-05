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
 * \file stpmv_kernel_simt.cpp
 * \brief SIMT VF kernel for single-precision triangular packed matrix-vector multiply.
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "stpmv_tiling_data.h"
#include "common/helper/kernel_constant.h"

using namespace AscendC;

// Disable FMA contraction to match CPU BLAS bit-exact results (separate mul + add).
#pragma clang fp contract(off)

// ==========================================================================
//  Packed index functions (uint64_t to avoid overflow)
// ==========================================================================

__simt_callee__ inline uint64_t StpmvPackedUpperIdxSimt(uint32_t i, uint32_t j)
{
    uint64_t j64 = static_cast<uint64_t>(j);
    return static_cast<uint64_t>(i) + j64 * (j64 + 1ULL) / 2ULL;
}

__simt_callee__ inline uint64_t StpmvPackedLowerIdxSimt(uint32_t i, uint32_t j, uint32_t n)
{
    uint64_t n64 = static_cast<uint64_t>(n);
    uint64_t j64 = static_cast<uint64_t>(j);
    return static_cast<uint64_t>(i) + (2ULL * n64 - j64 - 1ULL) * j64 / 2ULL;
}

// ==========================================================================
//  GetA — read matrix element with transpose handling
// ==========================================================================

template <bool UPPER, bool TRANS>
__simt_callee__ inline float StpmvGetASimt(uint32_t row, uint32_t col, uint32_t n, __gm__ const float* apGm)
{
    if constexpr (!TRANS) {
        if constexpr (UPPER) {
            return apGm[StpmvPackedUpperIdxSimt(row, col)];
        } else {
            return apGm[StpmvPackedLowerIdxSimt(row, col, n)];
        }
    } else {
        if constexpr (UPPER) {
            return apGm[StpmvPackedUpperIdxSimt(col, row)];
        } else {
            return apGm[StpmvPackedLowerIdxSimt(col, row, n)];
        }
    }
}

// ==========================================================================
//  Interleaved row mapping — balances work across threads for triangular shape
// ==========================================================================

__simt_callee__ inline uint32_t StpmvInterleavedRow(uint32_t k, uint32_t n)
{
    uint32_t half = k >> 1;
    if ((k & 1u) == 0u) {
        return half;
    }
    return n - 1u - half;
}

// ==========================================================================
//  SIMT VF kernel
// ==========================================================================

template <bool UPPER, bool TRANS, bool UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StpmvSimt(
    uint32_t n, int64_t incx, __gm__ const float* apGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    uint64_t absIncx = (incx >= 0) ? static_cast<uint64_t>(incx) : (static_cast<uint64_t>(~incx) + 1ULL);

    for (uint32_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) {
        uint32_t row = StpmvInterleavedRow(k, n);
        float sum = 0.0f;
        uint32_t colStart = 0;
        uint32_t colEnd = n;

        if constexpr (!TRANS) {
            if constexpr (UPPER) {
                colStart = row;
            } else {
                colEnd = row + 1;
            }
        } else {
            if constexpr (UPPER) {
                colEnd = row + 1;
            } else {
                colStart = row;
            }
        }

        for (uint32_t col = colStart; col < colEnd; ++col) {
            float aVal = 1.0f;
            if (col == row) {
                if constexpr (!UNIT) {
                    aVal = StpmvGetASimt<UPPER, TRANS>(row, col, n, apGm);
                }
            } else {
                aVal = StpmvGetASimt<UPPER, TRANS>(row, col, n, apGm);
            }
            uint64_t xOff =
                (incx >= 0) ? (static_cast<uint64_t>(col) * absIncx) : (static_cast<uint64_t>(n - 1 - col) * absIncx);
            float xVal = xGm[xOff];
            volatile float product = aVal * xVal;
            sum = sum + product;
        }

        yGm[row] = sum;
    }
}

// ==========================================================================
//  SIMT kernel dispatcher — single entry, dispatches via asc_vf_call
// ==========================================================================

__global__ __aicore__ void stpmv_simt_kernel(GM_ADDR aP, GM_ADDR x, GM_ADDR y, StpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* apGm = reinterpret_cast<__gm__ const float*>(aP);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);

    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StpmvSimt<false, false, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
            } else {
                asc_vf_call<StpmvSimt<false, false, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StpmvSimt<false, true, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
            } else {
                asc_vf_call<StpmvSimt<false, true, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
            }
        }
    } else {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StpmvSimt<true, false, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
            } else {
                asc_vf_call<StpmvSimt<true, false, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                asc_vf_call<StpmvSimt<true, true, false>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
            } else {
                asc_vf_call<StpmvSimt<true, true, true>>(
                    dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, apGm, xGm, yGm);
            }
        }
    }
}

// ==========================================================================
//  Scatter kernel — copy contiguous workspace back to strided x
// ==========================================================================

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StpmvScatterSimt(
    uint32_t n, int64_t incx, __gm__ float* dstGm, __gm__ const float* srcGm)
{
    uint64_t absIncx = (incx >= 0) ? static_cast<uint64_t>(incx) : (static_cast<uint64_t>(~incx) + 1ULL);
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        uint64_t dstOff =
            (incx >= 0) ? (static_cast<uint64_t>(i) * absIncx) : (static_cast<uint64_t>(n - 1 - i) * absIncx);
        dstGm[dstOff] = srcGm[i];
    }
}

__global__ __aicore__ void stpmv_scatter_kernel(GM_ADDR dst, GM_ADDR src, StpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* dstGm = reinterpret_cast<__gm__ float*>(dst);
    auto* srcGm = reinterpret_cast<__gm__ const float*>(src);
    asc_vf_call<StpmvScatterSimt>(dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.incx, dstGm, srcGm);
}

// ==========================================================================
//  SIMT kernel launch wrapper — called from host
// ==========================================================================

void stpmv_arch35_kernel_do(
    GM_ADDR aP, GM_ADDR x, GM_ADDR y, const StpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    stpmv_simt_kernel<<<numBlocks, nullptr, stream>>>(aP, x, y, tiling);
}

void stpmv_arch35_scatter_do(GM_ADDR dst, GM_ADDR src, const StpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    stpmv_scatter_kernel<<<numBlocks, nullptr, stream>>>(dst, src, tiling);
}
