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
 * \file complex_blas3_expand_arch22.h
 * \brief Phase 0b (chemm / csymm only): triangle -> full square expansion.
 *
 *        Split out of complex_blas3_arch22.h so that file stays under the
 *        repository's per-header line-count guideline; this half is only
 *        pulled in by the two side/uplo operators (chemm, csymm), not by the
 *        three rank-K operators (cherk, csyrk, cher2k).
 *
 *        Phase 0a must already have split the whole d x d array into Ar / Ai,
 *        and the packed buffers must be padded to a multiple of
 *        CBLAS3_EXPAND_TILE in both dimensions (see complex_blas3_host_utils.h).
 */

#ifndef COMPLEX_BLAS3_EXPAND_ARCH22_H
#define COMPLEX_BLAS3_EXPAND_ARCH22_H

#include "complex_blas3_arch22.h"

namespace cblas3 {
namespace expand {
constexpr uint32_t TILE = CBLAS3_EXPAND_TILE;
constexpr uint32_t TILE_ELEMS = TILE * TILE;

// InitTileMaps divides by TILE with a right shift, so the exponent has to track
// TILE instead of being spelled out at the call site. The first assert carries
// the guarantee -- any CBLAS3_EXPAND_TILE that is not exactly 2^TILE_LOG2, a
// non-power-of-two included, fails the build instead of silently shifting by the
// wrong amount. The second states the power-of-two requirement on its own so the
// intent survives a future edit to TILE_LOG2.
constexpr uint32_t TILE_LOG2 = 6U;
static_assert(TILE == (1U << TILE_LOG2), "TILE_LOG2 must satisfy TILE == 2^TILE_LOG2");
static_assert((TILE & (TILE - 1U)) == 0U, "CBLAS3_EXPAND_TILE must be a power of two");

// A tile is held in UB as ub[c * TILE + r], which is what a DataCopyPad of TILE
// columns x TILE floats produces (blockCount = columns, blockLen = bytes per
// column, srcStride = the GM gap in bytes, UB side packed).
//
// Every tile is a full TILE x TILE block, never a partial one. That is why the
// caller pads the packed buffer of the triangular operand to a multiple of TILE in
// both dimensions: a partial tile would land in UB with a row stride of its own
// height instead of TILE, which would invalidate the single precomputed transpose
// index below. Padding costs at most (TILE-1) rows and columns of workspace and
// keeps this stage to one code path.
struct TileCtx {
    GlobalTensor<float> reGm;
    GlobalTensor<float> imGm;
    LocalTensor<float> srcR;
    LocalTensor<float> srcI;
    LocalTensor<float> dstR;
    LocalTensor<float> dstI;
    LocalTensor<uint32_t> idxU;   // byte offsets realising the TILE x TILE transpose
    LocalTensor<uint8_t> keep;    // 1 where the stored triangle owns the element
    LocalTensor<uint8_t> offDiag; // 0 exactly on the diagonal
    uint32_t ld;
    int32_t imagSign;
};

// Build the transpose index and the two diagonal masks.
//
// For a tile held as ub[c * TILE + r], the transpose is dst[c * TILE + r] =
// src[r * TILE + c], so position t = c * TILE + r must read source element
// (t % TILE) * TILE + (t / TILE). Gather consumes byte offsets, hence the scaling.
//
// Both masks come from f = c - r: it is >= 0 exactly on the upper side, <= 0 on the
// lower side and == 0 exactly on the diagonal.
__aicore__ inline void InitTileMaps(
    TileCtx& ctx, const LocalTensor<int32_t>& work0, const LocalTensor<int32_t>& work1, const LocalTensor<int32_t>& idx,
    uint32_t uploMode)
{
    CreateVecIndex(work0, 0, TILE_ELEMS);                                  // work0 = t
    PipeBarrier<PIPE_V>();
    ShiftRight(work1, work0, static_cast<int32_t>(TILE_LOG2), TILE_ELEMS); // work1 = c = t / TILE
    PipeBarrier<PIPE_V>();
    Muls(idx, work1, static_cast<int32_t>(TILE), TILE_ELEMS);
    PipeBarrier<PIPE_V>();
    Sub(work0, work0, idx, TILE_ELEMS); // work0 = r = t % TILE
    PipeBarrier<PIPE_V>();

    // f = c - r, staged through idx and then dstR, which holds no tile data yet.
    Sub(idx, work1, work0, TILE_ELEMS);
    PipeBarrier<PIPE_V>();
    Cast(ctx.dstR, idx, RoundMode::CAST_NONE, TILE_ELEMS);
    PipeBarrier<PIPE_V>();
    const CMPMODE keepMode = (uploMode == CBLAS3_UPLO_UPPER) ? CMPMODE::GE : CMPMODE::LE;
    Compares(ctx.keep, ctx.dstR, 0.0F, keepMode, TILE_ELEMS);
    Compares(ctx.offDiag, ctx.dstR, 0.0F, CMPMODE::NE, TILE_ELEMS);
    PipeBarrier<PIPE_V>();

    // idx = r * TILE * sizeof(float) + c * sizeof(float)
    Muls(idx, work0, static_cast<int32_t>(TILE * sizeof(float)), TILE_ELEMS);
    Muls(work1, work1, static_cast<int32_t>(sizeof(float)), TILE_ELEMS);
    PipeBarrier<PIPE_V>();
    Add(idx, idx, work1, TILE_ELEMS);
    PipeBarrier<PIPE_V>();
    ctx.idxU = idx.ReinterpretCast<uint32_t>();
}

__aicore__ inline void LoadTile(
    const TileCtx& ctx, const LocalTensor<float>& dst, const GlobalTensor<float>& gm, uint32_t rowBase,
    uint32_t colBase)
{
    const uint32_t gap = (ctx.ld - TILE) * static_cast<uint32_t>(sizeof(float));
    DataCopyExtParams params{static_cast<uint16_t>(TILE), TILE * static_cast<uint32_t>(sizeof(float)), gap, 0U, 0U};
    DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
    DataCopyPad(dst, gm[static_cast<uint64_t>(colBase) * ctx.ld + rowBase], params, pad);
}

__aicore__ inline void StoreTile(
    const TileCtx& ctx, const LocalTensor<float>& src, const GlobalTensor<float>& gm, uint32_t rowBase,
    uint32_t colBase)
{
    const uint32_t gap = (ctx.ld - TILE) * static_cast<uint32_t>(sizeof(float));
    DataCopyExtParams params{static_cast<uint16_t>(TILE), TILE * static_cast<uint32_t>(sizeof(float)), 0U, gap, 0U};
    DataCopyPad(gm[static_cast<uint64_t>(colBase) * ctx.ld + rowBase], src, params);
}

// Fill the tile at (rowBase, colBase) from its mirror at (colBase, rowBase).
// `onDiagonal` is true only when the two coincide, which is also the only case in
// which part of the tile is already correct and must be preserved.
__aicore__ inline void MirrorOneTile(TileCtx& ctx, uint32_t rowBase, uint32_t colBase, bool onDiagonal)
{
    LoadTile(ctx, ctx.srcR, ctx.reGm, colBase, rowBase);
    LoadTile(ctx, ctx.srcI, ctx.imGm, colBase, rowBase);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    Gather(ctx.dstR, ctx.srcR, ctx.idxU, 0U, TILE_ELEMS);
    Gather(ctx.dstI, ctx.srcI, ctx.idxU, 0U, TILE_ELEMS);
    PipeBarrier<PIPE_V>();
    if (ctx.imagSign < 0) {
        Muls(ctx.dstI, ctx.dstI, -1.0F, TILE_ELEMS);
        PipeBarrier<PIPE_V>();
    }

    if (onDiagonal) {
        // For a diagonal tile the mirror source is the tile itself, so srcR / srcI
        // still hold the stored side untransposed: keep that side and take the other
        // from the transpose.
        Select(ctx.dstR, ctx.keep, ctx.srcR, ctx.dstR, SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_ELEMS);
        Select(ctx.dstI, ctx.keep, ctx.srcI, ctx.dstI, SELMODE::VSEL_TENSOR_TENSOR_MODE, TILE_ELEMS);
        PipeBarrier<PIPE_V>();
        if (ctx.imagSign < 0) {
            // Hermitian: BLAS does not reference the stored diagonal imaginary part
            // and its mathematical value is zero.
            Select(ctx.dstI, ctx.offDiag, ctx.dstI, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, TILE_ELEMS);
            PipeBarrier<PIPE_V>();
        }
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    StoreTile(ctx, ctx.dstR, ctx.reGm, rowBase, colBase);
    StoreTile(ctx, ctx.dstI, ctx.imGm, rowBase, colBase);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}
} // namespace expand

// Mirror a triangular-stored square matrix into a full one. See the namespace
// comment above for the Phase 0a precondition on the packed buffers.
//
// Tiles are dealt out over the cores only after the ones needing no work have been
// filtered, for the same load-balancing reason as GemmBody: the tiles that need
// work form a triangle, so striding over the raw grid would leave cores idle.
static __aicore__ inline bool MirrorTriBodySetup(const CBlas3ExpandTilingData& t, uint32_t& coreNum, uint32_t& coreIdx)
{
    coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U || t.tileCount == 0U) {
        return false;
    }
    coreIdx = static_cast<uint32_t>(GetBlockIdx());
    return true;
}

// Bundles the nine UB scratch buffers that back a TileCtx. Kept alive in the
// caller's frame (TBuf lifetime must span every MirrorOneTile call).
struct MirrorTriBuffers {
    TBuf<TPosition::VECCALC> srcR;
    TBuf<TPosition::VECCALC> srcI;
    TBuf<TPosition::VECCALC> dstR;
    TBuf<TPosition::VECCALC> dstI;
    TBuf<TPosition::VECCALC> idx;
    TBuf<TPosition::VECCALC> w0;
    TBuf<TPosition::VECCALC> w1;
    TBuf<TPosition::VECCALC> keep;
    TBuf<TPosition::VECCALC> diag;
};

__aicore__ inline void InitMirrorTriTileCtx(
    TPipe& pipe, MirrorTriBuffers& bufs, expand::TileCtx& ctx, GM_ADDR re, GM_ADDR im, const CBlas3ExpandTilingData& t)
{
    pipe.InitBuffer(bufs.srcR, expand::TILE_ELEMS * sizeof(float));
    pipe.InitBuffer(bufs.srcI, expand::TILE_ELEMS * sizeof(float));
    pipe.InitBuffer(bufs.dstR, expand::TILE_ELEMS * sizeof(float));
    pipe.InitBuffer(bufs.dstI, expand::TILE_ELEMS * sizeof(float));
    pipe.InitBuffer(bufs.idx, expand::TILE_ELEMS * sizeof(int32_t));
    pipe.InitBuffer(bufs.w0, expand::TILE_ELEMS * sizeof(int32_t));
    pipe.InitBuffer(bufs.w1, expand::TILE_ELEMS * sizeof(int32_t));
    pipe.InitBuffer(bufs.keep, expand::TILE_ELEMS / BITS_PER_BYTE + BYTENUM_REPEAT);
    pipe.InitBuffer(bufs.diag, expand::TILE_ELEMS / BITS_PER_BYTE + BYTENUM_REPEAT);

    ctx.reGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(re));
    ctx.imGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(im));
    ctx.srcR = bufs.srcR.Get<float>();
    ctx.srcI = bufs.srcI.Get<float>();
    ctx.dstR = bufs.dstR.Get<float>();
    ctx.dstI = bufs.dstI.Get<float>();
    ctx.keep = bufs.keep.Get<uint8_t>();
    ctx.offDiag = bufs.diag.Get<uint8_t>();
    ctx.ld = t.packedLd;
    ctx.imagSign = t.imagSign;
    expand::InitTileMaps(ctx, bufs.w0.Get<int32_t>(), bufs.w1.Get<int32_t>(), bufs.idx.Get<int32_t>(), t.uploMode);
}

__aicore__ inline void MirrorTriBody(GM_ADDR re, GM_ADDR im, const CBlas3ExpandTilingData& t)
{
    uint32_t coreNum = 0U;
    uint32_t coreIdx = 0U;
    if (!MirrorTriBodySetup(t, coreNum, coreIdx)) {
        return;
    }
    const bool isUpper = (t.uploMode == CBLAS3_UPLO_UPPER);

    TPipe pipe;
    MirrorTriBuffers bufs;
    expand::TileCtx ctx;
    InitMirrorTriTileCtx(pipe, bufs, ctx, re, im, t);

    uint32_t needed = 0;
    for (uint32_t ti = 0; ti < t.tileCount; ++ti) {
        for (uint32_t tj = 0; tj < t.tileCount; ++tj) {
            // Tiles wholly inside the stored triangle are already correct. Because
            // the grid is aligned, that is exactly ti < tj for UPPER (ti > tj for
            // LOWER), and ti == tj is the only straddling case.
            const bool alreadyCorrect = isUpper ? (ti < tj) : (ti > tj);
            if (alreadyCorrect) {
                continue;
            }
            if ((needed++ % coreNum) != coreIdx) {
                continue;
            }
            expand::MirrorOneTile(ctx, ti * expand::TILE, tj * expand::TILE, ti == tj);
        }
    }
}
} // namespace cblas3

#endif // COMPLEX_BLAS3_EXPAND_ARCH22_H
