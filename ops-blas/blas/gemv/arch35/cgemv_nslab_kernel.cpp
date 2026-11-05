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
 * \file cgemv_nslab_kernel.cpp
 * \brief N slab computation and workspace finalization.
 */

#include "cgemv_kernel_common.h"

using namespace AscendC;

__simd_vf__ inline void CgemvNSlabMteTileVf(
    uint32_t cols, uint32_t activeRows, uint32_t reset, __ubuf__ float* a, __ubuf__ float* x, __ubuf__ float* acc)
{
    for (uint16_t row = 0; row < static_cast<uint16_t>(activeRows); row += 64U) {
        Reg::RegTensor<float> ar, ai, xr, xi, tmp;
        Reg::RegTensor<float> accR, accI, productR, productI;
        uint32_t maskCount = activeRows - static_cast<uint32_t>(row);
        Reg::MaskReg mask = Reg::UpdateMask<float>(maskCount);
        uint32_t rowOffset = static_cast<uint32_t>(row) * 2U;
        if (reset != 0U) {
            Reg::Duplicate(accR, 0.0F, mask);
            Reg::Duplicate(accI, 0.0F, mask);
        } else {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(accR, accI, acc + rowOffset);
        }
        for (uint16_t col = 0; col < static_cast<uint16_t>(cols); ++col) {
            uint32_t aOffset = static_cast<uint32_t>(col) * CGEMV_N_MTE_ROW_FLOATS + rowOffset;
            uint32_t xOffset = static_cast<uint32_t>(col) * 2U;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(ar, ai, a + aOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(xr, x + xOffset);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(xi, x + xOffset + 1U);
            Reg::Mul(productR, ar, xr, mask);
            Reg::Mul(tmp, ai, xi, mask);
            Reg::Sub(productR, productR, tmp, mask);
            Reg::Add(accR, accR, productR, mask);
            Reg::Mul(productI, ar, xi, mask);
            Reg::Mul(tmp, ai, xr, mask);
            Reg::Add(productI, productI, tmp, mask);
            Reg::Add(accI, accI, productI, mask);
        }
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(acc + rowOffset, accR, accI, mask);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__aicore__ inline void CgemvNSlabMteCopy(
    LocalTensor<float> dst, GlobalTensor<float> src, uint32_t col, uint32_t cols, uint32_t row0, uint32_t activeRows,
    uint32_t lda)
{
    uint32_t rowBytes = activeRows * 2U * sizeof(float);
    uint32_t alignedRowBytes = (rowBytes + 31U) & ~31U;
    uint32_t rightPad = (alignedRowBytes - rowBytes) / sizeof(float);
    uint32_t dstStrideBlocks = (CGEMV_N_MTE_ROW_BYTES - alignedRowBytes) / 32U;
    uint32_t srcStrideBytes = (lda - activeRows) * 2U * sizeof(float);
    uint64_t offset = (static_cast<uint64_t>(col) * lda + row0) * 2U;
    DataCopyPad(
        dst, src[offset], DataCopyExtParams{static_cast<uint16_t>(cols), rowBytes, srcStrideBytes, dstStrideBlocks, 0},
        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(rightPad), 0.0F});
}

struct CgemvNSlabMteTile {
    uint32_t row0;
    uint32_t rows;
    uint32_t col0;
    uint32_t cols;
    uint32_t lda;
};

__aicore__ inline void CgemvNSlabMtePrefetch(
    LocalTensor<float> aLocal, GlobalTensor<float> aGm, const CgemvNSlabMteTile& shape, uint32_t tile)
{
    uint32_t nextBuffer = (tile & 1U) ^ 1U;
    if (tile != 0U) {
        // Reusing a matrix slot must follow its previous vector reads.
        WaitFlag<HardEvent::V_MTE2>(nextBuffer);
    }
    uint32_t nextOffset = (tile + 1U) * CGEMV_N_MTE_COLS;
    uint32_t nextCols = shape.cols - nextOffset;
    if (nextCols > CGEMV_N_MTE_COLS) {
        nextCols = CGEMV_N_MTE_COLS;
    }
    CgemvNSlabMteCopy(
        aLocal[nextBuffer * CGEMV_N_MTE_A_BYTES / sizeof(float)], aGm, shape.col0 + nextOffset, nextCols, shape.row0,
        shape.rows, shape.lda);
    SetFlag<HardEvent::MTE2_V>(nextBuffer);
}

__aicore__ inline void CgemvNSlabMteComputeTiles(
    GlobalTensor<float> aGm, GlobalTensor<float> xGm, LocalTensor<float> aLocal, LocalTensor<float> xLocal,
    LocalTensor<float> accLocal, const CgemvNSlabMteTile& shape)
{
    uint32_t firstCols = shape.cols < CGEMV_N_MTE_COLS ? shape.cols : CGEMV_N_MTE_COLS;
    uint32_t xBytes = shape.cols * 2U * sizeof(float);
    DataCopyPad(
        xLocal, xGm[static_cast<uint64_t>(shape.col0) * 2U], DataCopyExtParams{1, xBytes, 0, 0, 0},
        DataCopyPadExtParams<float>{false, 0, 0, 0});
    CgemvNSlabMteCopy(aLocal, aGm, shape.col0, firstCols, shape.row0, shape.rows, shape.lda);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    uint32_t tiles = (shape.cols + CGEMV_N_MTE_COLS - 1U) / CGEMV_N_MTE_COLS;
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        uint32_t buffer = tile & 1U;
        uint32_t offset = tile * CGEMV_N_MTE_COLS;
        uint32_t tileCols = shape.cols - offset;
        if (tileCols > CGEMV_N_MTE_COLS) {
            tileCols = CGEMV_N_MTE_COLS;
        }
        if (tile + 1U < tiles) {
            CgemvNSlabMtePrefetch(aLocal, aGm, shape, tile);
        }
        WaitFlag<HardEvent::MTE2_V>(buffer);
        asc_vf_call<CgemvNSlabMteTileVf>(
            tileCols, shape.rows, tile == 0U ? 1U : 0U,
            reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr()) + buffer * CGEMV_N_MTE_A_BYTES / sizeof(float),
            reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr()) + offset * 2U,
            reinterpret_cast<__ubuf__ float*>(accLocal.GetPhyAddr()));
        if (tile + 2U < tiles) {
            SetFlag<HardEvent::V_MTE2>(buffer);
        }
    }
}

__aicore__ inline void CgemvNSlabMteStore(
    GlobalTensor<float> wsGm, LocalTensor<float> accLocal, uint32_t chunk, uint32_t m, const CgemvNSlabMteTile& shape)
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    uint32_t outputBytes = shape.rows * 2U * sizeof(float);
    DataCopyPad(
        wsGm[(static_cast<uint64_t>(chunk) * m + shape.row0) * 2U], accLocal,
        DataCopyExtParams{1, outputBytes, 0, 0, 0});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

__aicore__ inline void CgemvNSlabMteRun(
    GM_ADDR a, GM_ADDR x, GM_ADDR workSpace, uint32_t m, uint32_t n, uint32_t lda, uint32_t chunkLen,
    uint32_t colChunks, TPipe& pipe)
{
    if (colChunks == 0U) {
        return;
    }
    uint32_t block = GetBlockIdx();
    uint32_t chunk = block % colChunks;
    uint32_t row0 = (block / colChunks) * CGEMV_N_MTE_ROWS;
    if (row0 >= m) {
        return;
    }
    GlobalTensor<float> aGm, xGm, wsGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace));
    TBuf<QuePosition::VECCALC> aBuf, xBuf, accBuf;
    pipe.InitBuffer(aBuf, 2U * CGEMV_N_MTE_A_BYTES);
    pipe.InitBuffer(xBuf, CGEMV_N_MTE_X_BYTES);
    pipe.InitBuffer(accBuf, CGEMV_N_MTE_ROW_BYTES);
    LocalTensor<float> aLocal = aBuf.Get<float>();
    LocalTensor<float> xLocal = xBuf.Get<float>();
    LocalTensor<float> accLocal = accBuf.Get<float>();

    CgemvNSlabMteTile shape{row0, m - row0, chunk * chunkLen, 0, lda};
    if (shape.rows > CGEMV_N_MTE_ROWS) {
        shape.rows = CGEMV_N_MTE_ROWS;
    }
    shape.cols = n - shape.col0;
    if (shape.cols > chunkLen) {
        shape.cols = chunkLen;
    }
    CgemvNSlabMteComputeTiles(aGm, xGm, aLocal, xLocal, accLocal, shape);
    CgemvNSlabMteStore(wsGm, accLocal, chunk, m, shape);
}

// ==========================================================================
//  Slab fast path — trans=N (incx == 1), phase 1: compute
//  Grid = colChunks column slabs x row tiles of blockDim rows. Block b streams
//  columns [chunk*chunkLen, ...) of its row tile front-to-back; thread t keeps
//  one register accumulator pair for row tile*blockDim + t and stores this
//  slab's partial y value to the workspace at ws[chunk*m + row]. The simple
//  one-load-then-consume loop is the fastest form measured on this arch (the
//  vector pipe hides latency across warps; per-thread unrolling and wider
//  loads were measured slower). The cross-slab reduction runs as a second,
//  stream-ordered launch (no cross-block synchronization).
// ==========================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgemvNSlabCompute(
    uint32_t m, uint32_t n, uint32_t lda, uint32_t chunkLen, uint32_t colChunks, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* wsGm)
{
    uint32_t b = blockIdx.x;
    uint32_t bdim = blockDim.x;
    uint32_t t = threadIdx.x;
    uint32_t chunk = b % colChunks; // column slab
    uint32_t tile = b / colChunks;  // row tile

    uint32_t c0 = chunk * chunkLen;
    uint32_t c1 = c0 + chunkLen;
    if (c1 > n) {
        c1 = n;
    }

    // x is read straight from GM (one uniform float2 per column; it stays cache-resident).
    // UB staging was measured slower: per-thread UB reads cost more than cached GM reads here.
    __gm__ const float2* xGm2 = reinterpret_cast<__gm__ const float2*>(xGm);

    // Pure 32-bit complex-unit indexing (host guarantees lda*n fits in uint32)
    __gm__ const float2* aGm2 = reinterpret_cast<__gm__ const float2*>(aGm);
    uint32_t row = tile * bdim + t;
    if (row >= m) {
        return; // no block-level synchronization follows: idle threads may leave
    }
    float accR = 0.0f;
    float accI = 0.0f;
    uint32_t colBase = c0 * lda + row;
    for (uint32_t c = c0; c < c1; ++c) {
        float2 xv = xGm2[c];
        float2 av = aGm2[colBase];
        accR += av.x * xv.x - av.y * xv.y;
        accI += av.x * xv.y + av.y * xv.x;
        colBase += lda;
    }

    // Store this slab's partial y value to the workspace
    __gm__ float2* myWs = reinterpret_cast<__gm__ float2*>(wsGm) + static_cast<uint64_t>(chunk) * m;
    myWs[row] = make_float2(accR, accI);
}

// ==========================================================================
//  Slab fast path — trans=N, phase 2: cross-slab reduction + write-back
//  Stream-ordered second launch. Block b owns rows [b*FIN_ROWS, ...): thread t
//  handles row (t % FIN_ROWS) and every FIN_GROUPS-th slab partial starting at
//  (t / FIN_ROWS); the FIN_GROUPS partial sums per row meet in UB. Many warps
//  per row group hide the GM load latency of the (short, strided) chunk walk.
// ==========================================================================
static constexpr uint32_t CGEMV_FIN_ROWS = 128; // rows per finalize block (== SIMT_MIN_THREAD_NUM)
static constexpr uint32_t CGEMV_FIN_GROUPS = 8; // chunk groups per row => 1024 threads per block
static constexpr uint32_t CGEMV_FIN_SMALL_ROWS = 64;
static constexpr uint32_t CGEMV_FIN_SMALL_GROUPS = 4;
// MTE2+SIMD finalize candidate.  A block owns exactly one SIMD register of
// complex rows.  At most 64 chunk-major workspace slices are packed into
// fixed 512-byte UB slots; the first eight slots are reused for group sums.
static constexpr uint32_t CGEMV_FIN_VEC_ROWS = 64;
static constexpr uint32_t CGEMV_FIN_VEC_MAX_CHUNKS = 64;
static constexpr uint32_t CGEMV_FIN_VEC_COMPLEX_FLOATS = 2;
static constexpr uint32_t CGEMV_FIN_VEC_SLOT_FLOATS = CGEMV_FIN_VEC_ROWS * CGEMV_FIN_VEC_COMPLEX_FLOATS;
static constexpr uint32_t CGEMV_FIN_VEC_SLOT_BYTES = CGEMV_FIN_VEC_SLOT_FLOATS * sizeof(float);
static constexpr uint32_t CGEMV_FIN_VEC_WS_BYTES = CGEMV_FIN_VEC_MAX_CHUNKS * CGEMV_FIN_VEC_SLOT_BYTES;
static constexpr uint32_t CGEMV_FIN_VEC_OUT_BYTES = CGEMV_FIN_VEC_SLOT_BYTES;
static constexpr uint32_t CGEMV_FIN_VEC_UB_BYTES = CGEMV_FIN_VEC_WS_BYTES + CGEMV_FIN_VEC_OUT_BYTES;
static_assert(CGEMV_FIN_VEC_SLOT_BYTES == 512U);
static_assert(CGEMV_FIN_VEC_UB_BYTES == 33280U);

// Fixed two-warp compute for the smallest slab output. This is the same
// column-chunk recurrence and workspace layout as CgemvNSlabCompute, with the
// row-tile division/modulo and inactive-row path removed.
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMV_NSMALL_ROWS* CGEMV_NSMALL_MAX_GROUPS) inline void CgemvNSlabComputeSmall(
    uint32_t n, uint32_t lda, uint32_t chunkLen, uint32_t groups, __gm__ const float* aGm, __gm__ const float* xGm,
    __gm__ float* wsGm)
{
    __ubuf__ float2 grp[CGEMV_NSMALL_ROWS * CGEMV_NSMALL_MAX_GROUPS];

    uint32_t chunk = blockIdx.x;
    uint32_t t = threadIdx.x;
    uint32_t row = t & (CGEMV_FIN_SMALL_ROWS - 1U);
    uint32_t g = t / CGEMV_FIN_SMALL_ROWS;
    uint32_t c0 = chunk * chunkLen;
    uint32_t cEnd = c0 + chunkLen;
    if (cEnd > n) {
        cEnd = n;
    }
    // Split this slab's columns across the groups; the recurrence inside a group keeps the
    // public low-to-high column order, and the group partials are summed in group order.
    uint32_t nCols = (cEnd > c0) ? (cEnd - c0) : 0U;
    uint32_t perGroup = (nCols + groups - 1U) / groups;
    uint32_t gStart = c0 + g * perGroup;
    uint32_t gEnd = gStart + perGroup;
    if (gEnd > cEnd) {
        gEnd = cEnd;
    }

    __gm__ const float2* aGm2 = reinterpret_cast<__gm__ const float2*>(aGm);
    __gm__ const float2* xGm2 = reinterpret_cast<__gm__ const float2*>(xGm);
    float accR = 0.0F;
    float accI = 0.0F;
    if (gStart < gEnd) {
        uint32_t aIdx = gStart * lda + row;
        for (uint32_t c = gStart; c < gEnd; ++c) {
            float2 xv = xGm2[c];
            float2 av = aGm2[aIdx];
            accR += av.x * xv.x - av.y * xv.y;
            accI += av.x * xv.y + av.y * xv.x;
            aIdx += lda;
        }
    }
    // Reduce the group partials inside the block and publish ONE slab per chunk. Writing a
    // slab per group instead was measured far slower: it multiplies workspace traffic and
    // makes the 256-thread finalize walk groups x colChunks partials per row.
    grp[g * CGEMV_FIN_SMALL_ROWS + row] = make_float2(accR, accI);
    asc_syncthreads();
    if (g == 0U) {
        float sR = 0.0F;
        float sI = 0.0F;
        for (uint32_t k = 0; k < groups; ++k) { // group order == ascending column order
            float2 pv = grp[k * CGEMV_FIN_SMALL_ROWS + row];
            sR += pv.x;
            sI += pv.y;
        }
        reinterpret_cast<__gm__ float2*>(wsGm)[chunk * CGEMV_FIN_SMALL_ROWS + row] = make_float2(sR, sI);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgemvNSlabFinalize(
    uint32_t m, uint32_t numChunks, float alphaR, float alphaI, float betaR, float betaI, uint32_t betaIsZero,
    int64_t incy, __gm__ const float* wsGm, __gm__ float* yGm)
{
    __ubuf__ float2 red[CGEMV_FIN_GROUPS * CGEMV_FIN_ROWS];
    int64_t m64 = static_cast<int64_t>(m);
    __gm__ const float2* ws2 = reinterpret_cast<__gm__ const float2*>(wsGm);
    uint32_t t = threadIdx.x;
    uint32_t rowLocal = t % CGEMV_FIN_ROWS;
    uint32_t grp = t / CGEMV_FIN_ROWS; // < CGEMV_FIN_GROUPS (blockDim.x == FIN_ROWS * FIN_GROUPS)

    uint32_t row = blockIdx.x * CGEMV_FIN_ROWS + rowLocal;

    float sR = 0.0f;
    float sI = 0.0f;
    if (row < m) {
        // Partial j of this row lives at ws2[row + j*m]; this thread sums j = grp, grp+G, ...
        uint32_t idx = row + grp * m;
        for (uint32_t j = grp; j < numChunks; j += CGEMV_FIN_GROUPS) {
            float2 p = ws2[idx];
            sR += p.x;
            sI += p.y;
            idx += CGEMV_FIN_GROUPS * m;
        }
    }
    red[t] = make_float2(sR, sI);
    asc_syncthreads();

    if (grp == 0 && row < m) {
        for (uint32_t g = 1; g < CGEMV_FIN_GROUPS; ++g) {
            float2 p = red[g * CGEMV_FIN_ROWS + rowLocal];
            sR += p.x;
            sI += p.y;
        }
        int64_t yIdx = CgemvStridedIdx(static_cast<int64_t>(row), m64, incy) * 2;
        CgemvWriteBack(yGm, yIdx, sR, sI, alphaR, alphaI, betaR, betaI, betaIsZero);
    }
}

// The common finalize geometry reserves 128 rows per block. For m<=64 that
// leaves half of every group idle. Four chunk groups keep eight warps busy
// while reducing both the UB footprint and launch geometry to 256 threads.
__simt_vf__ __aicore__ LAUNCH_BOUND(CGEMV_FIN_SMALL_ROWS* CGEMV_FIN_SMALL_GROUPS) inline void CgemvNSlabFinalizeSmall(
    uint32_t numChunks, __gm__ const float* wsGm, __gm__ float* yGm)
{
    __ubuf__ float2 red[CGEMV_FIN_SMALL_GROUPS * CGEMV_FIN_SMALL_ROWS];
    __gm__ const float2* ws2 = reinterpret_cast<__gm__ const float2*>(wsGm);
    uint32_t t = threadIdx.x;
    uint32_t row = t & (CGEMV_FIN_SMALL_ROWS - 1U);
    uint32_t grp = t >> 6U;

    float sR = 0.0F;
    float sI = 0.0F;
    uint32_t idx = row + grp * CGEMV_FIN_SMALL_ROWS;
    for (uint32_t j = grp; j < numChunks; j += CGEMV_FIN_SMALL_GROUPS) {
        float2 p = ws2[idx];
        sR += p.x;
        sI += p.y;
        idx += CGEMV_FIN_SMALL_GROUPS * CGEMV_FIN_SMALL_ROWS;
    }
    red[t] = make_float2(sR, sI);
    asc_syncthreads();

    if (grp == 0U) {
        for (uint32_t g = 1; g < CGEMV_FIN_SMALL_GROUPS; ++g) {
            float2 p = red[g * CGEMV_FIN_SMALL_ROWS + row];
            sR += p.x;
            sI += p.y;
        }
        reinterpret_cast<__gm__ float2*>(yGm)[row] = make_float2(sR, sI);
    }
}

// Reduce one 64-row workspace tile with SIMD register operations.  The chunk
// association is intentionally identical to CgemvNSlabFinalize: each group g
// accumulates g,g+8,... in increasing order, then group 0 adds group 1..7 in
// increasing order.  Stores to the first eight workspace slots are exact FP32
// transport and do not introduce another arithmetic operation.
__simd_vf__ inline void CgemvNSlabFinalizeMteSimd64Vf(
    uint32_t numChunks, uint32_t activeRows, __ubuf__ float* ws, __ubuf__ float* out)
{
    Reg::RegTensor<float> partialR, partialI, sumR, sumI;
    uint32_t maskCount = activeRows;
    Reg::MaskReg mask = Reg::UpdateMask<float>(maskCount);

    for (uint16_t group = 0; group < CGEMV_FIN_GROUPS; ++group) {
        Reg::Duplicate(sumR, 0.0F, mask);
        Reg::Duplicate(sumI, 0.0F, mask);
        for (uint16_t chunk = group; chunk < static_cast<uint16_t>(numChunks); chunk += CGEMV_FIN_GROUPS) {
            __ubuf__ float* partial = ws + static_cast<uint32_t>(chunk) * CGEMV_FIN_VEC_SLOT_FLOATS;
            Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(partialR, partialI, partial);
            Reg::Add(sumR, sumR, partialR, mask);
            Reg::Add(sumI, sumI, partialI, mask);
        }
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(
            ws + static_cast<uint32_t>(group) * CGEMV_FIN_VEC_SLOT_FLOATS, sumR, sumI, mask);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

    Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(sumR, sumI, ws);
    for (uint16_t group = 1; group < CGEMV_FIN_GROUPS; ++group) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(
            partialR, partialI, ws + static_cast<uint32_t>(group) * CGEMV_FIN_VEC_SLOT_FLOATS);
        Reg::Add(sumR, sumR, partialR, mask);
        Reg::Add(sumI, sumI, partialI, mask);
    }
    Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(out, sumR, sumI, mask);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__aicore__ inline void CgemvNSlabFinalizeMteSimd64Row(
    uint32_t m, uint32_t numChunks, uint32_t row0, uint32_t activeRows, GlobalTensor<float>& wsGm,
    GlobalTensor<float>& yGm, LocalTensor<float>& wsLocal, LocalTensor<float>& outLocal)
{
    uint32_t rowBytes = activeRows * CGEMV_FIN_VEC_COMPLEX_FLOATS * sizeof(float);
    uint32_t alignedRowBytes = (rowBytes + 31U) & ~31U;
    uint32_t rightPad = (alignedRowBytes - rowBytes) / sizeof(float);
    uint32_t dstStrideBlocks = (CGEMV_FIN_VEC_SLOT_BYTES - alignedRowBytes) / 32U;
    uint32_t srcStrideBytes = (m - activeRows) * CGEMV_FIN_VEC_COMPLEX_FLOATS * sizeof(float);
    DataCopyPad(
        wsLocal, wsGm[static_cast<uint64_t>(row0) * CGEMV_FIN_VEC_COMPLEX_FLOATS],
        DataCopyExtParams{static_cast<uint16_t>(numChunks), rowBytes, srcStrideBytes, dstStrideBlocks, 0},
        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(rightPad), 0.0F});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    asc_vf_call<CgemvNSlabFinalizeMteSimd64Vf>(
        numChunks, activeRows, reinterpret_cast<__ubuf__ float*>(wsLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(outLocal.GetPhyAddr()));

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(
        yGm[static_cast<uint64_t>(row0) * CGEMV_FIN_VEC_COMPLEX_FLOATS], outLocal,
        DataCopyExtParams{1, rowBytes, 0, 0, 0});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

__aicore__ inline void CgemvNSlabFinalizeMteSimd64Run(
    GM_ADDR y, GM_ADDR workSpace, uint32_t m, uint32_t numChunks, TPipe& pipe)
{
    uint32_t row0 = GetBlockIdx() * CGEMV_FIN_VEC_ROWS;
    if (row0 >= m) {
        return;
    }
    uint32_t activeRows = m - row0;
    if (activeRows > CGEMV_FIN_VEC_ROWS) {
        activeRows = CGEMV_FIN_VEC_ROWS;
    }
    GlobalTensor<float> wsGm;
    GlobalTensor<float> yGm;
    wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    TBuf<QuePosition::VECCALC> wsBuf;
    TBuf<QuePosition::VECCALC> outBuf;
    pipe.InitBuffer(wsBuf, CGEMV_FIN_VEC_WS_BYTES);
    pipe.InitBuffer(outBuf, CGEMV_FIN_VEC_OUT_BYTES);
    LocalTensor<float> wsLocal = wsBuf.Get<float>();
    LocalTensor<float> outLocal = outBuf.Get<float>();

    CgemvNSlabFinalizeMteSimd64Row(m, numChunks, row0, activeRows, wsGm, yGm, wsLocal, outLocal);
}

// N-slab phase 1
__global__ __aicore__ void cgemv_nslab_mte_compute_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR workSpace, uint32_t m, uint32_t n, uint32_t lda, uint32_t chunkLen,
    uint32_t colChunks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgemvNSlabMteRun(a, x, workSpace, m, n, lda, chunkLen, colChunks, pipe);
}

__global__ __aicore__ void cgemv_nslab_compute_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR workSpace, uint32_t numThreads, uint32_t m, uint32_t n, uint32_t lda,
    uint32_t chunkLen, uint32_t colChunks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* wsGm = reinterpret_cast<__gm__ float*>(workSpace);
    asc_vf_call<CgemvNSlabCompute>(dim3{numThreads, 1, 1}, m, n, lda, chunkLen, colChunks, aGm, xGm, wsGm);
}

// The 64-row slab is launch-latency dominated. Active blocks publish one
// partial slab per column chunk. Idle blocks still join the device barrier
// before block 0 finalizes in the same launch.
__global__ __aicore__ void cgemv_nslab_small_fused_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, uint32_t n, uint32_t lda, uint32_t chunkLen, uint32_t colChunks,
    uint32_t groups)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (GetBlockIdx() < colChunks) {
        asc_vf_call<CgemvNSlabComputeSmall>(
            dim3{CGEMV_FIN_SMALL_ROWS * groups, 1, 1}, n, lda, chunkLen, groups,
            reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(x),
            reinterpret_cast<__gm__ float*>(workSpace));
    }
    SyncAll<true>();
    if (GetBlockIdx() == 0U) {
        asc_vf_call<CgemvNSlabFinalizeSmall>(
            dim3{CGEMV_FIN_SMALL_ROWS * CGEMV_FIN_SMALL_GROUPS, 1, 1}, colChunks,
            reinterpret_cast<__gm__ const float*>(workSpace), reinterpret_cast<__gm__ float*>(y));
    }
}

// N-slab phase 2 (stream-ordered reduction + write-back)
__global__ __aicore__ void cgemv_nslab_finalize_kernel(GM_ADDR y, GM_ADDR workSpace, CgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* wsGm = reinterpret_cast<__gm__ const float*>(workSpace);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    asc_vf_call<CgemvNSlabFinalize>(
        dim3{CGEMV_FIN_ROWS * CGEMV_FIN_GROUPS, 1, 1}, tiling.m, tiling.colChunks, tiling.alphaR, tiling.alphaI,
        tiling.betaR, tiling.betaI, tiling.betaIsZero, tiling.incy, wsGm, yGm);
}

__global__ __aicore__ void cgemv_nslab_finalize_mte_simd64_kernel(
    GM_ADDR y, GM_ADDR workSpace, uint32_t m, uint32_t numChunks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgemvNSlabFinalizeMteSimd64Run(y, workSpace, m, numChunks, pipe);
}

static bool CgemvUsesSmallSlab(const CgemvTilingData& tiling, uint32_t smallGroups)
{
    return tiling.m == CGEMV_FIN_SMALL_ROWS && smallGroups >= 1U && smallGroups <= CGEMV_NSMALL_MAX_GROUPS &&
           tiling.numThreads == CGEMV_FIN_SMALL_ROWS * smallGroups && tiling.alphaR == 1.0F && tiling.alphaI == 0.0F &&
           tiling.betaIsZero != 0 && tiling.incx == 1 && tiling.incy == 1;
}

static void CgemvLaunchNSmall(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    uint32_t smallGroups = tiling.numThreads / CGEMV_FIN_SMALL_ROWS;
    cgemv_nslab_small_fused_kernel<<<numBlocks, nullptr, stream>>>(
        a, x, y, workSpace, tiling.n, tiling.lda, tiling.chunkLen, tiling.colChunks, smallGroups);
}

static void CgemvLaunchNCompute(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    // Short slabs in tall matrices can replace repeated SIMT GM loads with
    // one or two MTE tiles while retaining the existing column recurrence.
    bool useTallShortSlab =
        tiling.m >= SIMT_MAX_THREAD_NUM && tiling.m > tiling.n && tiling.chunkLen <= 2U * CGEMV_N_MTE_COLS;
    bool useMteTransport = tiling.lda % 4U != 0U || useTallShortSlab;
    if (tiling.m >= CGEMV_N_MTE_ROWS && useMteTransport && tiling.chunkLen >= CGEMV_N_MTE_COLS &&
        tiling.chunkLen <= CGEMV_N_MTE_MAX_SLAB_COLS) {
        uint32_t mteBlocks = tiling.colChunks * ((tiling.m + CGEMV_N_MTE_ROWS - 1U) / CGEMV_N_MTE_ROWS);
        cgemv_nslab_mte_compute_kernel<<<(mteBlocks), nullptr, stream>>>(
            a, x, workSpace, tiling.m, tiling.n, tiling.lda, tiling.chunkLen, tiling.colChunks);
    } else {
        cgemv_nslab_compute_kernel<<<(numBlocks), nullptr, stream>>>(
            a, x, workSpace, tiling.numThreads, tiling.m, tiling.n, tiling.lda, tiling.chunkLen, tiling.colChunks);
    }
}

static void CgemvLaunchNFinalize(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    bool useMteSimd64Finalize = tiling.m > SIMT_MAX_THREAD_NUM && tiling.alphaR == 1.0F && tiling.alphaI == 0.0F &&
                                tiling.betaIsZero != 0 && tiling.incx == 1 && tiling.incy == 1 &&
                                tiling.colChunks != 0 && tiling.colChunks <= CGEMV_FIN_VEC_MAX_CHUNKS;
    if (useMteSimd64Finalize) {
        uint32_t finBlocks = (tiling.m + CGEMV_FIN_VEC_ROWS - 1U) / CGEMV_FIN_VEC_ROWS;
        cgemv_nslab_finalize_mte_simd64_kernel<<<(finBlocks), nullptr, stream>>>(
            y, workSpace, tiling.m, tiling.colChunks);
    } else {
        uint32_t finBlocks = (tiling.m + CGEMV_FIN_ROWS - 1) / CGEMV_FIN_ROWS;
        cgemv_nslab_finalize_kernel<<<(finBlocks), nullptr, stream>>>(y, workSpace, tiling);
    }
}

void CgemvLaunchNSlab(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    uint32_t smallGroups = (tiling.numThreads != 0U) ? (tiling.numThreads / CGEMV_FIN_SMALL_ROWS) : 0U;
    if (CgemvUsesSmallSlab(tiling, smallGroups)) {
        CgemvLaunchNSmall(a, x, y, workSpace, tiling, numBlocks, stream);
        return;
    }
    CgemvLaunchNCompute(a, x, y, workSpace, tiling, numBlocks, stream);
    CgemvLaunchNFinalize(a, x, y, workSpace, tiling, numBlocks, stream);
}
