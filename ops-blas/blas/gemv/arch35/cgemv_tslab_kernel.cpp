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
 * \file cgemv_tslab_kernel.cpp
 * \brief Transpose and conjugate-transpose CGEMV slab paths.
 */

#include "cgemv_kernel_common.h"

using namespace AscendC;

// Keep the original binary32 recurrence and real/imaginary shuffle order.
template <bool IS_CONJ>
__simt_callee__ __aicore__ inline void CgemvTAccumulate(float2 av, float2 xv, float& pR, float& pI)
{
    float aIv = IS_CONJ ? -av.y : av.y;
    pR += av.x * xv.x - aIv * xv.y;
    pI += av.x * xv.y + aIv * xv.x;
}

__simt_callee__ __aicore__ inline void CgemvTReduce(float& pR, float& pI, int32_t firstOffset)
{
    for (int32_t offset = firstOffset; offset > 0; offset >>= 1) {
        pR += asc_shfl_down(pR, static_cast<uint32_t>(offset));
        pI += asc_shfl_down(pI, static_cast<uint32_t>(offset));
    }
}

__simt_callee__ __aicore__ inline uint32_t CgemvTColumnEnd(uint32_t c0, uint32_t chunkLen, uint32_t n)
{
    uint32_t c1 = c0 + chunkLen;
    if (c1 > n) {
        c1 = n;
    }
    return c1;
}

struct CgemvTColumnRange {
    uint32_t begin;
    uint32_t end;
};

__simt_callee__ __aicore__ inline CgemvTColumnRange CgemvTGetColumnRange(uint32_t b, uint32_t chunkLen, uint32_t n)
{
    uint32_t c0 = b * chunkLen;
    uint32_t c1 = CgemvTColumnEnd(c0, chunkLen, n);
    return {c0, c1};
}

struct CgemvTGeneralContext {
    uint32_t bdim, t, lane, wid, numWarps;
    __ubuf__ const float2* xUb2;
    __gm__ const float2* xGm2;
};

template <bool X_FROM_GM>
__simt_callee__ __aicore__ inline CgemvTGeneralContext CgemvTPrepareGeneral(
    uint32_t m, __gm__ const float* xGm, __ubuf__ float* xUb)
{
    uint32_t bdim = blockDim.x;
    uint32_t t = threadIdx.x;
    uint32_t lane = t % warpSize;
    uint32_t wid = t / warpSize;
    uint32_t numWarps = bdim / warpSize;
    if constexpr (!X_FROM_GM) {
        uint32_t xFloats = m * 2;
        for (uint32_t i = t; i < xFloats; i += bdim) {
            xUb[i] = xGm[i];
        }
        asc_syncthreads();
    }
    __ubuf__ const float2* xUb2 = reinterpret_cast<__ubuf__ const float2*>(xUb);
    __gm__ const float2* xGm2 = reinterpret_cast<__gm__ const float2*>(xGm);
    return {bdim, t, lane, wid, numWarps, xUb2, xGm2};
}

template <bool IS_CONJ, bool X_FROM_GM>
__simt_callee__ __aicore__ inline float2 CgemvTGeneralDot(
    uint32_t m, uint32_t lane, uint32_t colBase, __gm__ const float2* aGm2, __gm__ const float2* xGm2,
    __ubuf__ const float2* xUb2)
{
    float pR = 0.0f;
    float pI = 0.0f;
    for (uint32_t row = lane; row < m; row += warpSize) {
        float2 av = aGm2[colBase + row];
        float2 xv = X_FROM_GM ? xGm2[row] : xUb2[row];
        CgemvTAccumulate<IS_CONJ>(av, xv, pR, pI);
    }
    CgemvTReduce(pR, pI, warpSize / 2);
    return make_float2(pR, pI);
}

template <bool IS_CONJ, bool X_FROM_GM>
__simt_callee__ __aicore__ inline float2 CgemvTSegmentedDot(
    uint32_t len, uint32_t lane, uint32_t r0, __gm__ const float2* aCol, __gm__ const float2* xGm2,
    __ubuf__ const float2* xUb2)
{
    float pR = 0.0f;
    float pI = 0.0f;
    if constexpr (X_FROM_GM) {
        __gm__ const float2* xCol = xGm2 + r0;
        for (uint32_t row = lane; row < len; row += warpSize) {
            float2 av = aCol[row];
            float2 xv = xCol[row];
            CgemvTAccumulate<IS_CONJ>(av, xv, pR, pI);
        }
    } else {
        __ubuf__ const float2* xCol = xUb2 + r0;
        for (uint32_t row = lane; row < len; row += warpSize) {
            float2 av = aCol[row];
            float2 xv = xCol[row];
            CgemvTAccumulate<IS_CONJ>(av, xv, pR, pI);
        }
    }
    CgemvTReduce(pR, pI, warpSize / 2);
    return make_float2(pR, pI);
}

template <bool IS_CONJ, bool STREAMED>
__simt_callee__ __aicore__ inline float2 CgemvTFixedDot(
    uint32_t rowStart, uint32_t rowEnd, __gm__ const float2* aCol, __ubuf__ const float2* xCol)
{
    float pR = 0.0F;
    float pI = 0.0F;
    for (uint32_t row = rowStart; row < rowEnd; row += CGEMV_TC_WARP_WIDTH) {
        float2 av = STREAMED ? CgemvLoadStreamedMatrix(aCol + row) : aCol[row];
        float2 xv = xCol[row];
        CgemvTAccumulate<IS_CONJ>(av, xv, pR, pI);
    }
    CgemvTReduce(pR, pI, 16);
    return make_float2(pR, pI);
}

template <bool IS_CONJ, bool STREAMED>
__simt_callee__ __aicore__ inline void CgemvTDirectBlock(
    uint32_t m, uint32_t nCols, uint32_t c0, uint32_t segShift, uint32_t t, uint32_t lane, uint32_t wid,
    __gm__ const float2* aGm2, __ubuf__ const float2* xUb, __ubuf__ float2* part, __gm__ float* yGm)
{
    uint32_t segs = 1U << segShift;
    uint32_t segRows = (m + segs - 1U) >> segShift;
    uint32_t items = nCols << segShift;

    uint32_t it = wid;
    if (it < items) {
        uint32_t cRel = it >> segShift;
        uint32_t seg = it & (segs - 1U);
        uint32_t r0 = seg * segRows;
        uint32_t len = (r0 < m) ? ((m - r0 < segRows) ? (m - r0) : segRows) : 0U;
        __gm__ const float2* aCol = aGm2 + (c0 + cRel) * m + r0;
        __ubuf__ const float2* xCol = xUb + r0;
        float2 partial = CgemvTFixedDot<IS_CONJ, STREAMED>(lane, len, aCol, xCol);
        if (lane == 0U) {
            part[it] = make_float2(partial.x, partial.y);
        }
    }
    asc_syncthreads();

    if (t < nCols) {
        float sR = 0.0F;
        float sI = 0.0F;
        uint32_t base = t << segShift;
        for (uint32_t sg = 0; sg < segs; ++sg) {
            float2 p = part[base + sg];
            sR += p.x;
            sI += p.y;
        }
        reinterpret_cast<__gm__ float2*>(yGm)[c0 + t] = make_float2(sR, sI);
    }
}

// ==========================================================================
//  Slab fast path — trans=T/C (incx == 1)
//  Block b streams columns [b*chunkLen, ...) front-to-back and owns those
//  outputs outright. Warp-per-column: lanes stride the rows, one shuffle
//  reduction per column per warp (block-wide reductions/syncs per column are
//  far too expensive). x (logical length m) is cached in UB when it fits
//  (X_FROM_GM=false, measured fastest for lane-varying reads); larger x is read
//  straight from GM (X_FROM_GM=true).
// ==========================================================================
template <bool IS_CONJ, bool X_FROM_GM>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgemvTSlab(
    uint32_t m, uint32_t n, uint32_t lda, float alphaR, float alphaI, float betaR, float betaI, uint32_t betaIsZero,
    int64_t incy, uint32_t chunkLen, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    __ubuf__ float xUb[X_FROM_GM ? 1 : CGEMV_UB_X_FLOATS]; // UB cache only when x is staged

    uint32_t b = blockIdx.x;
    CgemvTGeneralContext context = CgemvTPrepareGeneral<X_FROM_GM>(m, xGm, xUb);

    CgemvTColumnRange columns = CgemvTGetColumnRange(b, chunkLen, n);
    int64_t n64 = static_cast<int64_t>(n);
    // Pure 32-bit complex-unit indexing (host guarantees lda*n fits in uint32).
    __gm__ const float2* aGm2 = reinterpret_cast<__gm__ const float2*>(aGm);
    for (uint32_t c = columns.begin + context.wid; c < columns.end; c += context.numWarps) {
        uint32_t colBase = c * lda;
        float2 partial =
            CgemvTGeneralDot<IS_CONJ, X_FROM_GM>(m, context.lane, colBase, aGm2, context.xGm2, context.xUb2);
        if (context.lane == 0) {
            int64_t yIdx = CgemvStridedIdx(static_cast<int64_t>(c), n64, incy) * 2;
            CgemvWriteBack(yGm, yIdx, partial.x, partial.y, alphaR, alphaI, betaR, betaI, betaIsZero);
        }
    }
}

// Fixed-geometry T/C path for the acceptance-performance scalar combination.
// x is already in UB when the VF starts, so this avoids the scalar copy,
// block-wide barrier, full tiling ABI, and general alpha/beta write-back.
template <bool IS_CONJ>
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMV_TC_FIXED_THREADS) inline void CgemvTFixedMte(
    uint32_t m, uint32_t n, uint32_t chunkLen, __gm__ const float* aGm, __ubuf__ const float* xUb, __gm__ float* yGm)
{
    uint32_t b = blockIdx.x;
    uint32_t t = threadIdx.x;
    uint32_t lane = t & (CGEMV_TC_WARP_WIDTH - 1U);
    uint32_t wid = t >> 5U;
    __ubuf__ const float2* xUb2 = reinterpret_cast<__ubuf__ const float2*>(xUb);
    __gm__ const float2* aGm2 = reinterpret_cast<__gm__ const float2*>(aGm);

    uint32_t c0 = b * chunkLen;
    uint32_t c1 = CgemvTColumnEnd(c0, chunkLen, n);
    uint32_t c = c0 + wid;
    if (c < c1) {
        uint32_t colBase = c * m;
        float2 partial = CgemvTFixedDot<IS_CONJ, true>(lane, m, aGm2 + colBase, xUb2);
        if (lane == 0U) {
            reinterpret_cast<__gm__ float2*>(yGm)[c] = make_float2(partial.x, partial.y);
        }
    }
}

// PR exposes 56 AIV cores, so a few wide T/C fixtures produce 65--74 columns
// per slab.  Keep the proven single-column VF untouched for chunkLen<=64 and
// use this separate instantiation only for the residual columns.  A warp still
// owns a whole output and retains the same lane recurrence and shuffle tree.
template <bool IS_CONJ>
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMV_TC_FIXED_THREADS) inline void CgemvTFixedMteWide(
    uint32_t m, uint32_t n, uint32_t chunkLen, __gm__ const float* aGm, __ubuf__ const float* xUb, __gm__ float* yGm)
{
    uint32_t b = blockIdx.x;
    __ubuf__ float2 part[CGEMV_TC_WIDE_MAX_ITEMS];
    uint32_t t = threadIdx.x;
    uint32_t lane = t & (CGEMV_TC_WARP_WIDTH - 1U);
    uint32_t wid = t >> 5U;
    __ubuf__ const float2* xUb2 = reinterpret_cast<__ubuf__ const float2*>(xUb);
    __gm__ const float2* aGm2 = reinterpret_cast<__gm__ const float2*>(aGm);
    __gm__ float2* yGm2 = reinterpret_cast<__gm__ float2*>(yGm);

    uint32_t c0 = b * chunkLen;
    uint32_t c1 = CgemvTColumnEnd(c0, chunkLen, n);
    uint32_t nCols = c1 - c0;
    uint32_t items = nCols * CGEMV_TC_WIDE_SEGS; // host: nCols <= CGEMV_TC_WIDE_MAX_COLS
    uint32_t segRows = (m + CGEMV_TC_WIDE_SEGS - 1U) / CGEMV_TC_WIDE_SEGS;
    for (uint32_t it = wid; it < items; it += CGEMV_TC_WARPS) {
        uint32_t cRel = it / CGEMV_TC_WIDE_SEGS;
        uint32_t seg = it - cRel * CGEMV_TC_WIDE_SEGS;
        uint32_t r0 = seg * segRows;
        uint32_t rEnd = r0 + segRows;
        if (rEnd > m) {
            rEnd = m; // trailing segment may be short or empty
        }
        uint32_t colBase = (c0 + cRel) * m;
        float2 partial = CgemvTFixedDot<IS_CONJ, true>(r0 + lane, rEnd, aGm2 + colBase, xUb2);
        if (lane == 0U) {
            part[it] = make_float2(partial.x, partial.y);
        }
    }
    asc_syncthreads();
    // Segment order is fixed low-to-high so the summation order stays deterministic.
    for (uint32_t j = t; j < nCols; j += CGEMV_TC_FIXED_THREADS) {
        uint32_t base = j * CGEMV_TC_WIDE_SEGS;
        float sR = 0.0F;
        float sI = 0.0F;
        for (uint32_t sg = 0; sg < CGEMV_TC_WIDE_SEGS; ++sg) {
            float2 pv = part[base + sg];
            sR += pv.x;
            sI += pv.y;
        }
        yGm2[c0 + j] = make_float2(sR, sI);
    }
}

// ==========================================================================
//  Slab fast path — trans=T/C (incx == 1), segmented variant for slabs with
//  fewer columns than warps: every column is cut into S = 2^segShift row
//  segments and the (column, segment) items are dealt round-robin to warps,
//  so all warps stream and the longest warp holds ceil(items/warps)/S of a
//  column. A warp streams one item exactly like the plain kernel streams a
//  column (lane-strided rows from a per-item base pointer); per-item partials
//  meet in UB (one syncthreads per block) and threads then combine the S
//  partials of each column. x is cached in UB when it fits, else read from GM.
// ==========================================================================
template <bool IS_CONJ, bool X_FROM_GM>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgemvTSlabSeg(
    uint32_t m, uint32_t n, uint32_t lda, float alphaR, float alphaI, float betaR, float betaI, uint32_t betaIsZero,
    int64_t incy, uint32_t chunkLen, uint32_t segShift, __gm__ const float* aGm, __gm__ const float* xGm,
    __gm__ float* yGm)
{
    __ubuf__ float xUb[X_FROM_GM ? 1 : CGEMV_UB_X_FLOATS]; // UB cache only when x is staged
    __ubuf__ float2 part[CGEMV_T_MAX_ITEMS];

    uint32_t b = blockIdx.x;
    CgemvTGeneralContext context = CgemvTPrepareGeneral<X_FROM_GM>(m, xGm, xUb);

    CgemvTColumnRange columns = CgemvTGetColumnRange(b, chunkLen, n);
    uint32_t nCols = columns.end - columns.begin;
    int64_t n64 = static_cast<int64_t>(n);
    __gm__ const float2* aGm2 = reinterpret_cast<__gm__ const float2*>(aGm);
    uint32_t segs = 1U << segShift;
    uint32_t segRows = (m + segs - 1) >> segShift;
    uint32_t items = nCols << segShift; // host: items <= CGEMV_T_MAX_ITEMS

    for (uint32_t it = context.wid; it < items; it += context.numWarps) {
        uint32_t cRel = it >> segShift;
        uint32_t seg = it & (segs - 1);
        uint32_t r0 = seg * segRows;
        uint32_t len = (r0 < m) ? ((m - r0 < segRows) ? (m - r0) : segRows) : 0;
        // per-item base pointers (host guarantees lda*n fits in uint32; keep the walk 32-bit)
        __gm__ const float2* aCol = aGm2 + (columns.begin + cRel) * lda + r0;
        float2 partial =
            CgemvTSegmentedDot<IS_CONJ, X_FROM_GM>(len, context.lane, r0, aCol, context.xGm2, context.xUb2);
        if (context.lane == 0) {
            part[it] = make_float2(partial.x, partial.y);
        }
    }
    asc_syncthreads();

    for (uint32_t j = context.t; j < nCols; j += context.bdim) {
        float sR = 0.0f;
        float sI = 0.0f;
        uint32_t base = j << segShift;
        for (uint32_t sg = 0; sg < segs; ++sg) {
            float2 p = part[base + sg];
            sR += p.x;
            sI += p.y;
        }
        int64_t yIdx = CgemvStridedIdx(static_cast<int64_t>(columns.begin + j), n64, incy) * 2;
        CgemvWriteBack(yGm, yIdx, sR, sI, alphaR, alphaI, betaR, betaI, betaIsZero);
    }
}

// Acceptance-performance specialization for segmented T/C slabs. The host
// and dispatcher guarantee lda=m, alpha=1, beta=0, unit strides, at most 64
// (column,segment) items, and the fixed 2048-thread/64-warp geometry. Compared
// with the general segmented VF this uses float2 x staging, a right-sized
// partial array, one guarded item per warp, and a direct float2 result store.
// The lane recurrence, shuffle tree, segment boundaries, and final segment
// order are otherwise identical to CgemvTSlabSeg.
template <bool IS_CONJ>
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMV_TC_FIXED_THREADS) inline void CgemvTSlabSegDirect(
    uint32_t m, uint32_t n, uint32_t chunkLen, uint32_t segShift, __gm__ const float* aGm, __gm__ const float* xGm,
    __gm__ float* yGm)
{
    __ubuf__ float2 xUb[CGEMV_UB_X_FLOATS / 2U];
    __ubuf__ float2 part[CGEMV_T_MAX_ITEMS];

    uint32_t b = blockIdx.x;
    uint32_t t = threadIdx.x;
    uint32_t lane = t & (CGEMV_TC_WARP_WIDTH - 1U);
    uint32_t wid = t >> 5U;
    __gm__ const float2* aGm2 = reinterpret_cast<__gm__ const float2*>(aGm);
    __gm__ const float2* xGm2 = reinterpret_cast<__gm__ const float2*>(xGm);

    for (uint32_t i = t; i < m; i += CGEMV_TC_FIXED_THREADS) {
        xUb[i] = xGm2[i];
    }
    asc_syncthreads();

    uint32_t c0 = b * chunkLen;
    uint32_t c1 = CgemvTColumnEnd(c0, chunkLen, n);
    uint32_t nCols = c1 - c0;
    CgemvTDirectBlock<IS_CONJ, false>(m, nCols, c0, segShift, t, lane, wid, aGm2, xUb, part, yGm);
}

// Transport-only variant of CgemvTSlabSegDirect for wide segmented slabs.
// x and the partial array are allocated by the outer global; only x transport
// changes from cooperative SIMT loads to one MTE2 copy.  Item assignment,
// per-lane recurrence, shuffle order, segment combination, and write-back are
// intentionally kept identical to CgemvTSlabSegDirect.
template <bool IS_CONJ>
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMV_TC_FIXED_THREADS) inline void CgemvTSlabSegDirectMte(
    uint32_t m, uint32_t n, uint32_t chunkLen, uint32_t segShift, __gm__ const float* aGm, __ubuf__ const float* xUbRaw,
    __ubuf__ float* partRaw, __gm__ float* yGm)
{
    uint32_t b = blockIdx.x;
    uint32_t t = threadIdx.x;
    uint32_t lane = t & (CGEMV_TC_WARP_WIDTH - 1U);
    uint32_t wid = t >> 5U;
    __gm__ const float2* aGm2 = reinterpret_cast<__gm__ const float2*>(aGm);
    __ubuf__ const float2* xUb = reinterpret_cast<__ubuf__ const float2*>(xUbRaw);
    __ubuf__ float2* part = reinterpret_cast<__ubuf__ float2*>(partRaw);

    uint32_t c0 = b * chunkLen;
    uint32_t c1 = CgemvTColumnEnd(c0, chunkLen, n);
    uint32_t nCols = c1 - c0;
    CgemvTDirectBlock<IS_CONJ, true>(m, nCols, c0, segShift, t, lane, wid, aGm2, xUb, part, yGm);
}

struct CgemvTCStagedShape {
    uint32_t m;
    uint32_t col0;
    uint32_t cols;
    uint32_t segShift;
    uint32_t tileRows;
};

struct CgemvTCStagedBuffers {
    GlobalTensor<float> a;
    GlobalTensor<float> x;
    LocalTensor<float> aLocal;
    LocalTensor<float> xLocal;
    LocalTensor<float> accLocal;
    LocalTensor<float> partLocal;
};

// Each logical lane keeps its original lane,lane+32,... recurrence within
// its original row segment. MTE row tiles change transport only.
template <bool IS_CONJ>
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMV_TC_FIXED_THREADS) inline void CgemvTCStagedTileVf(
    uint32_t m, uint32_t row0, uint32_t cols, uint32_t segShift, uint32_t tileRows, __ubuf__ const float* aRaw,
    __ubuf__ const float* xRaw, __ubuf__ float* accRaw)
{
    uint32_t lane = threadIdx.x & (CGEMV_TC_WARP_WIDTH - 1U);
    uint32_t item = threadIdx.x >> 5U;
    if (item >= (cols << segShift)) {
        return;
    }
    uint32_t segs = 1U << segShift;
    uint32_t segRows = (m + segs - 1U) >> segShift;
    uint32_t start = (item & (segs - 1U)) * segRows;
    uint32_t end = start + segRows;
    if (end > m) {
        end = m;
    }
    uint32_t tileEnd = row0 + tileRows;
    if (tileEnd > m) {
        tileEnd = m;
    }
    if (start >= tileEnd || end <= row0) {
        return;
    }
    if (end > tileEnd) {
        end = tileEnd;
    }
    auto* partial = reinterpret_cast<__ubuf__ float2*>(accRaw) + threadIdx.x;
    float2 acc = row0 <= start ? make_float2(0.0F, 0.0F) : *partial;
    uint32_t row = start + lane;
    if (row < row0) {
        row += ((row0 - row + CGEMV_TC_WARP_WIDTH - 1U) / CGEMV_TC_WARP_WIDTH) * CGEMV_TC_WARP_WIDTH;
    }
    auto* aCol = reinterpret_cast<__ubuf__ const float2*>(aRaw) + (item >> segShift) * tileRows;
    auto* x = reinterpret_cast<__ubuf__ const float2*>(xRaw);
    for (; row < end; row += CGEMV_TC_WARP_WIDTH) {
        float2 av = aCol[row - row0];
        float2 xv = x[row];
        float ai = IS_CONJ ? -av.y : av.y;
        acc.x += av.x * xv.x - ai * xv.y;
        acc.y += av.x * xv.y + ai * xv.x;
    }
    *partial = acc;
    asc_threadfence_block();
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMV_TC_FIXED_THREADS) inline void CgemvTCStagedFinishVf(
    uint32_t col0, uint32_t cols, uint32_t segShift, __ubuf__ const float* accRaw, __ubuf__ float* partRaw,
    __gm__ float* yRaw)
{
    uint32_t t = threadIdx.x;
    uint32_t lane = t & (CGEMV_TC_WARP_WIDTH - 1U);
    uint32_t item = t >> 5U;
    auto* part = reinterpret_cast<__ubuf__ float2*>(partRaw);
    if (item < (cols << segShift)) {
        float2 acc = reinterpret_cast<__ubuf__ const float2*>(accRaw)[t];
        for (int32_t offset = 16; offset > 0; offset >>= 1) {
            acc.x += asc_shfl_down(acc.x, static_cast<uint32_t>(offset));
            acc.y += asc_shfl_down(acc.y, static_cast<uint32_t>(offset));
        }
        if (lane == 0U) {
            part[item] = acc;
        }
    }
    asc_syncthreads();
    if (t < cols) {
        float2 acc = part[t << segShift];
        if (segShift != 0U) {
            acc = make_float2(0.0F, 0.0F);
            for (uint32_t seg = 0; seg < (1U << segShift); ++seg) {
                float2 value = part[(t << segShift) + seg];
                acc.x += value.x;
                acc.y += value.y;
            }
        }
        reinterpret_cast<__gm__ float2*>(yRaw)[col0 + t] = acc;
    }
}

__aicore__ inline void CgemvTCStagedCopy(
    LocalTensor<float> dst, GlobalTensor<float> src, const CgemvTCStagedShape& shape, uint32_t row0)
{
    uint32_t rows = shape.m - row0;
    if (rows > shape.tileRows) {
        rows = shape.tileRows;
    }
    uint32_t bytes = rows * 2U * sizeof(float);
    uint32_t alignedBytes = (bytes + 31U) & ~31U;
    uint32_t slotBytes = shape.tileRows * 2U * sizeof(float);
    uint32_t srcStride = (shape.m - rows) * 2U * sizeof(float);
    uint8_t pad = static_cast<uint8_t>((alignedBytes - bytes) / sizeof(float));
    DataCopyPad(
        dst, src[(static_cast<uint64_t>(shape.col0) * shape.m + row0) * 2U],
        DataCopyExtParams{static_cast<uint16_t>(shape.cols), bytes, srcStride, (slotBytes - alignedBytes) / 32U, 0},
        DataCopyPadExtParams<float>{true, 0, pad, 0.0F});
}

template <bool IS_CONJ>
__aicore__ inline void CgemvTCStagedCompute(CgemvTCStagedBuffers& buffers, const CgemvTCStagedShape& shape)
{
    uint32_t xBytes = shape.m * 2U * sizeof(float);
    DataCopyPad(
        buffers.xLocal, buffers.x, DataCopyExtParams{1, xBytes, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0});
    CgemvTCStagedCopy(buffers.aLocal, buffers.a, shape, 0U);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    uint32_t tiles = (shape.m + shape.tileRows - 1U) / shape.tileRows;
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        uint32_t slot = tile & 1U;
        if (tile + 1U < tiles) {
            uint32_t next = slot ^ 1U;
            if (tile != 0U) {
                WaitFlag<HardEvent::V_MTE2>(next);
            }
            CgemvTCStagedCopy(
                buffers.aLocal[next * CGEMV_TC_STAGED_A_BYTES / sizeof(float)], buffers.a, shape,
                (tile + 1U) * shape.tileRows);
            SetFlag<HardEvent::MTE2_V>(next);
        }
        WaitFlag<HardEvent::MTE2_V>(slot);
        asc_vf_call<CgemvTCStagedTileVf<IS_CONJ>>(
            dim3{CGEMV_TC_FIXED_THREADS, 1, 1}, shape.m, tile * shape.tileRows, shape.cols, shape.segShift,
            shape.tileRows,
            reinterpret_cast<__ubuf__ const float*>(buffers.aLocal.GetPhyAddr()) +
                slot * CGEMV_TC_STAGED_A_BYTES / sizeof(float),
            reinterpret_cast<__ubuf__ const float*>(buffers.xLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(buffers.accLocal.GetPhyAddr()));
        if (tile + 2U < tiles) {
            SetFlag<HardEvent::V_MTE2>(slot);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <bool IS_CONJ>
__aicore__ inline void CgemvTCStagedRun(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen, uint32_t segShift, TPipe& pipe)
{
    CgemvTCStagedBuffers buffers;
    buffers.a.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    buffers.x.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    TBuf<QuePosition::VECCALC> aBuf, xBuf, accBuf, partBuf;
    pipe.InitBuffer(aBuf, 2U * CGEMV_TC_STAGED_A_BYTES);
    pipe.InitBuffer(xBuf, CGEMV_TC_STAGED_X_BYTES);
    pipe.InitBuffer(accBuf, CGEMV_TC_STAGED_ACC_BYTES);
    pipe.InitBuffer(partBuf, CGEMV_TC_STAGED_PART_BYTES);
    buffers.aLocal = aBuf.Get<float>();
    buffers.xLocal = xBuf.Get<float>();
    buffers.accLocal = accBuf.Get<float>();
    buffers.partLocal = partBuf.Get<float>();

    uint32_t col0 = GetBlockIdx() * chunkLen;
    if (col0 >= n) {
        return;
    }
    uint32_t cols = n - col0;
    if (cols > chunkLen) {
        cols = chunkLen;
    }
    // Use each matrix slot fully while retaining a 32-row-aligned pitch.
    uint32_t tileRows = (CGEMV_TC_STAGED_A_BYTES / (cols * 2U * sizeof(float))) & ~31U;
    if (tileRows > CGEMV_TC_STAGED_MAX_TILE_ROWS) {
        tileRows = CGEMV_TC_STAGED_MAX_TILE_ROWS;
    }
    CgemvTCStagedShape shape{m, col0, cols, segShift, tileRows};
    CgemvTCStagedCompute<IS_CONJ>(buffers, shape);
    asc_vf_call<CgemvTCStagedFinishVf>(
        dim3{CGEMV_TC_FIXED_THREADS, 1, 1}, col0, cols, segShift,
        reinterpret_cast<__ubuf__ const float*>(buffers.accLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(buffers.partLocal.GetPhyAddr()), reinterpret_cast<__gm__ float*>(y));
}

// T/C slab (one column per warp)
template <bool IS_CONJ, bool X_FROM_GM>
__aicore__ inline void CgemvTSlabLaunch(GM_ADDR a, GM_ADDR x, GM_ADDR y, const CgemvTilingData& tiling)
{
    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    asc_vf_call<CgemvTSlab<IS_CONJ, X_FROM_GM>>(
        dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alphaR, tiling.alphaI, tiling.betaR,
        tiling.betaI, tiling.betaIsZero, tiling.incy, tiling.chunkLen, aGm, xGm, yGm);
}

__global__ __aicore__ void cgemv_tslab_t_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, CgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgemvTSlabLaunch<false, false>(a, x, y, tiling);
}

__global__ __aicore__ void cgemv_tslab_c_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, CgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgemvTSlabLaunch<true, false>(a, x, y, tiling);
}

__aicore__ inline LocalTensor<float> CgemvTCopyFixedX(
    GlobalTensor<float>& xGm, uint32_t m, TPipe& pipe, TBuf<QuePosition::VECCALC>& xBuf)
{
    uint32_t xBytes = (m * 2U * sizeof(float) + 31U) & ~31U;
    pipe.InitBuffer(xBuf, xBytes);
    LocalTensor<float> xLocal = xBuf.Get<float>();
    DataCopyPad(
        xLocal, xGm, DataCopyExtParams{1, static_cast<uint32_t>(m * 2U * sizeof(float)), 0, 0, 0},
        DataCopyPadExtParams<float>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    return xLocal;
}

template <bool IS_CONJ>
__aicore__ inline void CgemvTFixedMteLaunch(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen, TPipe& pipe)
{
    GlobalTensor<float> xGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    TBuf<QuePosition::VECCALC> xBuf;
    LocalTensor<float> xLocal = CgemvTCopyFixedX(xGm, m, pipe, xBuf);
    auto* xUb = reinterpret_cast<__ubuf__ const float*>(xLocal.GetPhyAddr());
    asc_vf_call<CgemvTFixedMte<IS_CONJ>>(
        dim3{CGEMV_TC_FIXED_THREADS, 1, 1}, m, n, chunkLen, reinterpret_cast<__gm__ const float*>(a), xUb,
        reinterpret_cast<__gm__ float*>(y));
}

template <bool IS_CONJ>
__aicore__ inline void CgemvTFixedMteWideLaunch(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen, TPipe& pipe)
{
    GlobalTensor<float> xGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    TBuf<QuePosition::VECCALC> xBuf;
    LocalTensor<float> xLocal = CgemvTCopyFixedX(xGm, m, pipe, xBuf);
    auto* xUb = reinterpret_cast<__ubuf__ const float*>(xLocal.GetPhyAddr());
    asc_vf_call<CgemvTFixedMteWide<IS_CONJ>>(
        dim3{CGEMV_TC_FIXED_THREADS, 1, 1}, m, n, chunkLen, reinterpret_cast<__gm__ const float*>(a), xUb,
        reinterpret_cast<__gm__ float*>(y));
}

__global__ __aicore__ void cgemv_t_staged_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen, uint32_t segShift)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgemvTCStagedRun<false>(a, x, y, m, n, chunkLen, segShift, pipe);
}

__global__ __aicore__ void cgemv_c_staged_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen, uint32_t segShift)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgemvTCStagedRun<true>(a, x, y, m, n, chunkLen, segShift, pipe);
}

__global__ __aicore__ void cgemv_tslab_t_fixed_mte_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgemvTFixedMteLaunch<false>(a, x, y, m, n, chunkLen, pipe);
}

__global__ __aicore__ void cgemv_tslab_c_fixed_mte_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgemvTFixedMteLaunch<true>(a, x, y, m, n, chunkLen, pipe);
}

__global__ __aicore__ void cgemv_tslab_t_fixed_mte_wide_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgemvTFixedMteWideLaunch<false>(a, x, y, m, n, chunkLen, pipe);
}

__global__ __aicore__ void cgemv_tslab_c_fixed_mte_wide_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgemvTFixedMteWideLaunch<true>(a, x, y, m, n, chunkLen, pipe);
}

// x larger than the UB cache (m > CGEMV_UB_X_FLOATS / 2): read x from GM
__global__ __aicore__ void cgemv_tslab_t_xgm_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, CgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgemvTSlabLaunch<false, true>(a, x, y, tiling);
}

__global__ __aicore__ void cgemv_tslab_c_xgm_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, CgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgemvTSlabLaunch<true, true>(a, x, y, tiling);
}

// T/C slab, segmented (tall-skinny slabs)
template <bool IS_CONJ, bool X_FROM_GM>
__aicore__ inline void CgemvTSlabSegLaunch(GM_ADDR a, GM_ADDR x, GM_ADDR y, const CgemvTilingData& tiling)
{
    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    asc_vf_call<CgemvTSlabSeg<IS_CONJ, X_FROM_GM>>(
        dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alphaR, tiling.alphaI, tiling.betaR,
        tiling.betaI, tiling.betaIsZero, tiling.incy, tiling.chunkLen, tiling.segShift, aGm, xGm, yGm);
}

__global__ __aicore__ void cgemv_tslabseg_t_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, CgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgemvTSlabSegLaunch<false, false>(a, x, y, tiling);
}

__global__ __aicore__ void cgemv_tslabseg_c_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, CgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgemvTSlabSegLaunch<true, false>(a, x, y, tiling);
}

__global__ __aicore__ void cgemv_tslabseg_t_xgm_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, CgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgemvTSlabSegLaunch<false, true>(a, x, y, tiling);
}

__global__ __aicore__ void cgemv_tslabseg_c_xgm_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, CgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgemvTSlabSegLaunch<true, true>(a, x, y, tiling);
}

template <bool IS_CONJ>
__aicore__ inline void CgemvTSlabSegDirectLaunch(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen, uint32_t segShift)
{
    asc_vf_call<CgemvTSlabSegDirect<IS_CONJ>>(
        dim3{CGEMV_TC_FIXED_THREADS, 1, 1}, m, n, chunkLen, segShift, reinterpret_cast<__gm__ const float*>(a),
        reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ float*>(y));
}

__global__ __aicore__ void cgemv_tslabseg_t_direct_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen, uint32_t segShift)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgemvTSlabSegDirectLaunch<false>(a, x, y, m, n, chunkLen, segShift);
}

__global__ __aicore__ void cgemv_tslabseg_c_direct_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen, uint32_t segShift)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgemvTSlabSegDirectLaunch<true>(a, x, y, m, n, chunkLen, segShift);
}

template <bool IS_CONJ>
__aicore__ inline void CgemvTSlabSegDirectMteLaunch(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen, uint32_t segShift, TPipe& pipe)
{
    GlobalTensor<float> xGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));

    uint32_t xBytes = (m * 2U * sizeof(float) + 31U) & ~31U;
    TBuf<QuePosition::VECCALC> xBuf;
    TBuf<QuePosition::VECCALC> partBuf;
    pipe.InitBuffer(xBuf, xBytes);
    pipe.InitBuffer(partBuf, CGEMV_TC_SEG_PART_UB_BYTES);
    LocalTensor<float> xLocal = xBuf.Get<float>();
    LocalTensor<float> partLocal = partBuf.Get<float>();

    DataCopyPad(
        xLocal, xGm, DataCopyExtParams{1, static_cast<uint32_t>(m * 2U * sizeof(float)), 0, 0, 0},
        DataCopyPadExtParams<float>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    asc_vf_call<CgemvTSlabSegDirectMte<IS_CONJ>>(
        dim3{CGEMV_TC_FIXED_THREADS, 1, 1}, m, n, chunkLen, segShift, reinterpret_cast<__gm__ const float*>(a),
        reinterpret_cast<__ubuf__ const float*>(xLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(partLocal.GetPhyAddr()), reinterpret_cast<__gm__ float*>(y));
}

__global__ __aicore__ void cgemv_tslabseg_t_direct_mte_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen, uint32_t segShift)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgemvTSlabSegDirectMteLaunch<false>(a, x, y, m, n, chunkLen, segShift, pipe);
}

__global__ __aicore__ void cgemv_tslabseg_c_direct_mte_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t chunkLen, uint32_t segShift)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgemvTSlabSegDirectMteLaunch<true>(a, x, y, m, n, chunkLen, segShift, pipe);
}

static void CgemvLaunchTStaged(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    if (tiling.trans == 2) {
        cgemv_c_staged_kernel<<<(numBlocks), CGEMV_TC_STAGED_UB_BYTES, stream>>>(
            a, x, y, tiling.m, tiling.n, tiling.chunkLen, tiling.segShift);
    } else {
        cgemv_t_staged_kernel<<<(numBlocks), CGEMV_TC_STAGED_UB_BYTES, stream>>>(
            a, x, y, tiling.m, tiling.n, tiling.chunkLen, tiling.segShift);
    }
}

static void CgemvLaunchTFixed(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    uint32_t xBytes = (tiling.m * 2U * sizeof(float) + 31U) & ~31U;
    if (tiling.trans == 2) {
        cgemv_tslab_c_fixed_mte_kernel<<<(numBlocks), xBytes, stream>>>(a, x, y, tiling.m, tiling.n, tiling.chunkLen);
    } else {
        cgemv_tslab_t_fixed_mte_kernel<<<(numBlocks), xBytes, stream>>>(a, x, y, tiling.m, tiling.n, tiling.chunkLen);
    }
}

static void CgemvLaunchTFixedWide(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    uint32_t xBytes = (tiling.m * 2U * sizeof(float) + 31U) & ~31U;
    if (tiling.trans == 2) {
        cgemv_tslab_c_fixed_mte_wide_kernel<<<(numBlocks), xBytes, stream>>>(
            a, x, y, tiling.m, tiling.n, tiling.chunkLen);
    } else {
        cgemv_tslab_t_fixed_mte_wide_kernel<<<(numBlocks), xBytes, stream>>>(
            a, x, y, tiling.m, tiling.n, tiling.chunkLen);
    }
}

static void CgemvLaunchTDirectMte(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    uint32_t xBytes = (tiling.m * 2U * sizeof(float) + 31U) & ~31U;
    uint32_t ubBytes = xBytes + CGEMV_TC_SEG_PART_UB_BYTES;
    if (tiling.trans == 2) {
        cgemv_tslabseg_c_direct_mte_kernel<<<(numBlocks), ubBytes, stream>>>(
            a, x, y, tiling.m, tiling.n, tiling.chunkLen, tiling.segShift);
    } else {
        cgemv_tslabseg_t_direct_mte_kernel<<<(numBlocks), ubBytes, stream>>>(
            a, x, y, tiling.m, tiling.n, tiling.chunkLen, tiling.segShift);
    }
}

static void CgemvLaunchTDirect(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    if (tiling.trans == 2) {
        cgemv_tslabseg_c_direct_kernel<<<(numBlocks), nullptr, stream>>>(
            a, x, y, tiling.m, tiling.n, tiling.chunkLen, tiling.segShift);
    } else {
        cgemv_tslabseg_t_direct_kernel<<<(numBlocks), nullptr, stream>>>(
            a, x, y, tiling.m, tiling.n, tiling.chunkLen, tiling.segShift);
    }
}

static void CgemvLaunchTGeneral(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    bool xFitsUb = (tiling.m * 2 <= CGEMV_UB_X_FLOATS);
    if (xFitsUb) {
        if (tiling.trans == 2) {
            cgemv_tslab_c_kernel<<<(numBlocks), nullptr, stream>>>(a, x, y, tiling);
        } else {
            cgemv_tslab_t_kernel<<<(numBlocks), nullptr, stream>>>(a, x, y, tiling);
        }
    } else if (tiling.trans == 2) {
        cgemv_tslab_c_xgm_kernel<<<(numBlocks), nullptr, stream>>>(a, x, y, tiling);
    } else {
        cgemv_tslab_t_xgm_kernel<<<(numBlocks), nullptr, stream>>>(a, x, y, tiling);
    }
}

static void CgemvLaunchTGeneralSegmented(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    bool xFitsUb = (tiling.m * 2 <= CGEMV_UB_X_FLOATS);
    if (xFitsUb) {
        if (tiling.trans == 2) {
            cgemv_tslabseg_c_kernel<<<(numBlocks), nullptr, stream>>>(a, x, y, tiling);
        } else {
            cgemv_tslabseg_t_kernel<<<(numBlocks), nullptr, stream>>>(a, x, y, tiling);
        }
    } else if (tiling.trans == 2) {
        cgemv_tslabseg_c_xgm_kernel<<<(numBlocks), nullptr, stream>>>(a, x, y, tiling);
    } else {
        cgemv_tslabseg_t_xgm_kernel<<<(numBlocks), nullptr, stream>>>(a, x, y, tiling);
    }
}

static bool CgemvUsesTStaged(const CgemvTilingData& tiling)
{
    return tiling.m > SIMT_MAX_THREAD_NUM && tiling.m <= CGEMV_TC_STAGED_MAX_ROWS && tiling.lda == tiling.m &&
           tiling.chunkLen >= CGEMV_TC_SEG_MTE_MIN_COLS && tiling.chunkLen <= CGEMV_TC_WARPS &&
           tiling.segShift <= CGEMV_T_MAX_SEG_SHIFT && (tiling.chunkLen << tiling.segShift) <= CGEMV_TC_WARPS;
}

static bool CgemvUsesTFixedBase(const CgemvTilingData& tiling)
{
    bool xFitsUb = (tiling.m * 2 <= CGEMV_UB_X_FLOATS);
    return xFitsUb && tiling.segShift == 0 && tiling.lda == tiling.m && tiling.alphaR == 1.0F &&
           tiling.alphaI == 0.0F && tiling.betaIsZero != 0 && tiling.incy == 1 &&
           tiling.numThreads == CGEMV_TC_FIXED_THREADS;
}

static bool CgemvUsesTDirect(const CgemvTilingData& tiling)
{
    bool xFitsUb = (tiling.m * 2 <= CGEMV_UB_X_FLOATS);
    return xFitsUb && tiling.segShift != 0 && tiling.lda == tiling.m && tiling.alphaR == 1.0F &&
           tiling.alphaI == 0.0F && tiling.betaIsZero != 0 && tiling.incy == 1 &&
           (tiling.chunkLen << tiling.segShift) <= CGEMV_T_MAX_ITEMS && tiling.numThreads == CGEMV_TC_FIXED_THREADS;
}

void CgemvLaunchTSlab(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    if (CgemvUsesTStaged(tiling)) {
        CgemvLaunchTStaged(a, x, y, workSpace, tiling, numBlocks, stream);
        return;
    }
    bool fixedMteBase = CgemvUsesTFixedBase(tiling);
    if (fixedMteBase && tiling.chunkLen <= CGEMV_TC_WARPS) {
        CgemvLaunchTFixed(a, x, y, workSpace, tiling, numBlocks, stream);
    } else if (fixedMteBase && tiling.chunkLen > CGEMV_TC_WARPS && tiling.chunkLen <= CGEMV_TC_WIDE_MAX_COLS) {
        CgemvLaunchTFixedWide(a, x, y, workSpace, tiling, numBlocks, stream);
    } else if (CgemvUsesTDirect(tiling)) {
        if (tiling.segShift == 1U && tiling.chunkLen >= CGEMV_TC_SEG_MTE_MIN_COLS) {
            CgemvLaunchTDirectMte(a, x, y, workSpace, tiling, numBlocks, stream);
        } else {
            CgemvLaunchTDirect(a, x, y, workSpace, tiling, numBlocks, stream);
        }
    } else if (tiling.segShift == 0) {
        CgemvLaunchTGeneral(a, x, y, workSpace, tiling, numBlocks, stream);
    } else {
        CgemvLaunchTGeneralSegmented(a, x, y, workSpace, tiling, numBlocks, stream);
    }
}
