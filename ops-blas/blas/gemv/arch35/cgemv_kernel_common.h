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
 * \file cgemv_kernel_common.h
 * \brief Shared CGEMV device arithmetic and constants.
 */

#pragma once

#include "cgemv_kernel.h"
#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/vector_functions.h"
#include "cgemv_tiling_data.h"
#include "common/helper/kernel_constant.h"

// Disable implicit FMA contraction in CGEMV kernels and their shared helpers.
#pragma clang fp contract(off)

static constexpr uint32_t CGEMV_UB_X_FLOATS = 16384; // 64 KB __ubuf__ for x vector (8192 complex)
static constexpr uint32_t CGEMV_UB_MIN_OUT = 32;     // generic path: below this many outputs UB staging does not pay
static constexpr uint32_t CGEMV_TC_FIXED_THREADS = 2048;
static constexpr uint32_t CGEMV_TC_WARP_WIDTH = 32;
static constexpr uint32_t CGEMV_TC_WARPS = CGEMV_TC_FIXED_THREADS / CGEMV_TC_WARP_WIDTH;
static constexpr uint32_t CGEMV_TC_MTE_UB_BYTES = CGEMV_UB_X_FLOATS * sizeof(float);
static constexpr uint32_t CGEMV_TC_SEG_MTE_MIN_COLS = 16;
static constexpr uint32_t CGEMV_TC_SEG_PART_UB_BYTES = CGEMV_T_MAX_ITEMS * 2U * sizeof(float);
static_assert((CGEMV_TC_SEG_PART_UB_BYTES & 31U) == 0U);
// The 56-AIV acceptance target reaches at most 74 columns per slab across the
// supplied PF manifest. Keep the residual specialization bounded to that
// audited range; larger slabs use the general T/C implementation.
static constexpr uint32_t CGEMV_TC_WIDE_MAX_COLS = 74;
// A wide slab carries more columns than the block has warps, so a one-warp-per-column
// mapping always runs ceil(chunkLen/64)=2 rounds and the second round leaves most warps
// idle (chunkLen 67 fills 67 of 128 warp slots, 52%).  Splitting each column's row range
// in two doubles the independent items and refills those slots: 2*chunkLen items over 64
// warps is 3 rounds of half-length work, i.e. 1.5 column-passes instead of 2.  The gate is
// structural rather than shape-specific - this kernel is selected only when
// chunkLen > CGEMV_TC_WARPS, which is exactly the regime where the 2-round penalty applies.
static constexpr uint32_t CGEMV_TC_WIDE_SEGS = 2;
static constexpr uint32_t CGEMV_TC_WIDE_MAX_ITEMS = CGEMV_TC_WIDE_MAX_COLS * CGEMV_TC_WIDE_SEGS;
static_assert(CGEMV_TC_WIDE_MAX_COLS > CGEMV_TC_WARPS);

// N512 has an order-preserving SIMD specialization.  A SIMD lane owns one
// complete row and commits columns 0..511 to one accumulator in order.  The
// two 256-column MTE tiles are only a transport boundary: the rounded FP32
// accumulator is stored and reloaded without forming independent partials.
static constexpr uint32_t CGEMV_N512_DIM = 512;
static constexpr uint32_t CGEMV_N512_ROWS_PER_BLOCK = 64;
static constexpr uint32_t CGEMV_N512_ROW_BLOCKS = CGEMV_N512_DIM / CGEMV_N512_ROWS_PER_BLOCK;
static constexpr uint32_t CGEMV_N512_TILE_COLS = 256;
static constexpr uint32_t CGEMV_N512_COMPLEX_FLOATS = 2;
static constexpr uint32_t CGEMV_N512_COMPLEX_BYTES = CGEMV_N512_COMPLEX_FLOATS * sizeof(float);
static constexpr uint32_t CGEMV_N512_ROW_SLOT_BYTES = CGEMV_N512_ROWS_PER_BLOCK * CGEMV_N512_COMPLEX_BYTES;
static constexpr uint32_t CGEMV_N512_ROW_SLOT_FLOATS = CGEMV_N512_ROW_SLOT_BYTES / sizeof(float);
static constexpr uint32_t CGEMV_N512_A_TILE_BYTES = CGEMV_N512_TILE_COLS * CGEMV_N512_ROW_SLOT_BYTES;
static constexpr uint32_t CGEMV_N512_X_TILE_BYTES = CGEMV_N512_TILE_COLS * CGEMV_N512_COMPLEX_BYTES;
static constexpr uint32_t CGEMV_N512_ACC_BYTES = CGEMV_N512_ROW_SLOT_BYTES;

static_assert(CGEMV_TC_WARPS == 64);
static_assert(CGEMV_TC_MTE_UB_BYTES == 64 * 1024);
static_assert(CGEMV_N512_ROW_BLOCKS == 8);
static_assert(CGEMV_N512_ROW_SLOT_BYTES == 512);
static_assert(CGEMV_N512_A_TILE_BYTES + CGEMV_N512_X_TILE_BYTES + CGEMV_N512_ACC_BYTES == 133632);

// MTE2 stages N slabs with unaligned matrix columns, or short column slabs
// in tall matrices. Two matrix buffers overlap transport and SIMD compute.
// The existing slab boundaries and increasing-column accumulation stay intact.
static constexpr uint32_t CGEMV_N_MTE_ROWS = 512;
static constexpr uint32_t CGEMV_N_MTE_COLS = 16;
static constexpr uint32_t CGEMV_N_MTE_MAX_SLAB_COLS = 128;
static constexpr uint32_t CGEMV_N_MTE_ROW_FLOATS = CGEMV_N_MTE_ROWS * 2U;
static constexpr uint32_t CGEMV_N_MTE_ROW_BYTES = CGEMV_N_MTE_ROW_FLOATS * sizeof(float);
static constexpr uint32_t CGEMV_N_MTE_A_BYTES = CGEMV_N_MTE_COLS * CGEMV_N_MTE_ROW_BYTES;
// Keep a full vector-read suffix after the last scalar broadcast address.
static constexpr uint32_t CGEMV_N_MTE_X_BYTES = CGEMV_N_MTE_MAX_SLAB_COLS * 2U * sizeof(float) + 512U;
static_assert(2U * CGEMV_N_MTE_A_BYTES + CGEMV_N_MTE_X_BYTES + CGEMV_N_MTE_ROW_BYTES <= UB_SIZE);

static constexpr uint32_t CGEMV_TC_STAGED_MIN_ROWS = 128U;
static constexpr uint32_t CGEMV_TC_STAGED_MAX_ROWS = 4096U;
static constexpr uint32_t CGEMV_TC_STAGED_MAX_TILE_ROWS = 1024U;
static constexpr uint32_t CGEMV_TC_STAGED_A_BYTES = CGEMV_TC_WARPS * CGEMV_TC_STAGED_MIN_ROWS * 2U * sizeof(float);
static constexpr uint32_t CGEMV_TC_STAGED_X_BYTES = CGEMV_TC_STAGED_MAX_ROWS * 2U * sizeof(float);
static constexpr uint32_t CGEMV_TC_STAGED_ACC_BYTES = CGEMV_TC_FIXED_THREADS * 2U * sizeof(float);
static constexpr uint32_t CGEMV_TC_STAGED_PART_BYTES = CGEMV_TC_WARPS * 2U * sizeof(float);
static constexpr uint32_t CGEMV_TC_STAGED_UB_BYTES =
    2U * CGEMV_TC_STAGED_A_BYTES + CGEMV_TC_STAGED_X_BYTES + CGEMV_TC_STAGED_ACC_BYTES + CGEMV_TC_STAGED_PART_BYTES;
static_assert(CGEMV_TC_STAGED_UB_BYTES + 64U * 1024U <= UB_SIZE);

// Strided index of logical element i in a vector of logical length len (complex units).
// Negative strides walk backwards from the far end (Netlib reference semantics).
__simt_callee__ __aicore__ inline int64_t CgemvStridedIdx(int64_t i, int64_t len, int64_t inc)
{
    return (inc >= 0) ? (i * inc) : ((len - 1 - i) * (-inc));
}

// An explicit fma with an exact -0 addend is one correctly-rounded binary32
// multiplication.  Unlike an ordinary temporary (including volatile on this
// compiler), the public SIMT intrinsic keeps each complex product rounded
// before the following Sub/Add without introducing stack-backed product slots.
__simt_callee__ __aicore__ inline float CgemvRoundedMul(float lhs, float rhs) { return fmaf(lhs, rhs, -0.0F); }

// The public load intrinsic takes a non-const pointer but never writes it.
__simt_callee__ __aicore__ inline float2 CgemvLoadStreamedMatrix(__gm__ const float2* address)
{
    return asc_ldcg(const_cast<__gm__ float2*>(address));
}

// Complex multiplication with an explicit binary32 boundary after each of
// the four real products. This mirrors the scalar reference implementation
// even when the device compiler would otherwise contract Mul+Add/Sub.
__simt_callee__ __aicore__ inline void CgemvComplexMul(
    float lhsR, float lhsI, float rhsR, float rhsI, float& outR, float& outI)
{
    outR = CgemvRoundedMul(lhsR, rhsR);
    outR -= CgemvRoundedMul(lhsI, rhsI);
    outI = CgemvRoundedMul(lhsR, rhsI);
    outI += CgemvRoundedMul(lhsI, rhsR);
}

// Netlib trans=N first scales y by beta, then accumulates one alpha*x column
// update at a time. Keeping that order is materially different in binary32
// from forming a dot product and applying alpha/beta only at write-back.
__simt_callee__ __aicore__ inline void CgemvNInitialValue(
    __gm__ const float* yGm, int64_t yIdx, float betaR, float betaI, uint32_t betaIsZero, float& accR, float& accI)
{
    if (betaIsZero != 0) {
        accR = 0.0F;
        accI = 0.0F;
        return;
    }

    float yR = yGm[yIdx];
    float yI = yGm[yIdx + 1];
    if (betaR == 1.0F && betaI == 0.0F) {
        accR = yR;
        accI = yI;
    } else {
        CgemvComplexMul(betaR, betaI, yR, yI, accR, accI);
    }
}

__simt_callee__ __aicore__ inline float2 CgemvNScaledX(float2 x, float alphaR, float alphaI)
{
    if (alphaR == 1.0F && alphaI == 0.0F) {
        return x;
    }
    float outR;
    float outI;
    CgemvComplexMul(alphaR, alphaI, x.x, x.y, outR, outI);
    return make_float2(outR, outI);
}

// Complex multiply-accumulate: (accR, accI) += a * x, with optional conj(a).
template <bool IS_CONJ>
__simt_callee__ __aicore__ inline void CgemvCmla(float2 a, float2 x, float& accR, float& accI)
{
    float aI = IS_CONJ ? -a.y : a.y;
    float term = CgemvRoundedMul(a.x, x.x);
    term -= CgemvRoundedMul(aI, x.y);
    accR += term;

    term = CgemvRoundedMul(a.x, x.y);
    term += CgemvRoundedMul(aI, x.x);
    accI += term;
}

// Write-back: y[yIdx] = alpha * acc + beta * y[yIdx].
// Exact-one scalars skip the complex multiply (Netlib semantics): keeps
// Inf/NaN payloads in acc/y intact instead of degrading via 0*Inf = NaN.
__simt_callee__ __aicore__ inline void CgemvWriteBack(
    __gm__ float* yGm, int64_t yIdx, float accR, float accI, float alphaR, float alphaI, float betaR, float betaI,
    uint32_t betaIsZero)
{
    bool alphaIsOne = (alphaR == 1.0f) && (alphaI == 0.0f);
    bool betaIsOne = (betaR == 1.0f) && (betaI == 0.0f);
    float outR;
    float outI;
    if (alphaIsOne) {
        outR = accR;
        outI = accI;
    } else {
        outR = CgemvRoundedMul(alphaR, accR);
        outR -= CgemvRoundedMul(alphaI, accI);

        outI = CgemvRoundedMul(alphaR, accI);
        outI += CgemvRoundedMul(alphaI, accR);
    }
    if (betaIsZero == 0) {
        float yR = yGm[yIdx];
        float yI = yGm[yIdx + 1];
        if (betaIsOne) {
            outR += yR;
            outI += yI;
        } else {
            float betaTermR = CgemvRoundedMul(betaR, yR);
            betaTermR -= CgemvRoundedMul(betaI, yI);

            float betaTermI = CgemvRoundedMul(betaR, yI);
            betaTermI += CgemvRoundedMul(betaI, yR);

            outR += betaTermR;
            outI += betaTermI;
        }
    }
    yGm[yIdx] = outR;
    yGm[yIdx + 1] = outI;
}
