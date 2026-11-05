/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "gemm_strided_batched_tiling.h"
#include "gemm_strided_batched_fp32.h"

using namespace AscendC;
using namespace matmul;

namespace {
constexpr uint32_t TILE = 128;
constexpr uint32_t VECTOR_TILE = 256;
// Largest finite IEEE-754 binary32 value; larger magnitudes are Inf/NaN and must not be used as scale factors.
constexpr float FP32_MAX_FINITE = 3.402823466e38f;
// 1e30 leaves over eight decimal orders of headroom below FP32 max and routes near-limit values to the safe fallback.
constexpr float EXTREME_VALUE_THRESHOLD = 1.0e30f;
constexpr MatmulShapeParams SHAPE{128, 128, 4096, 128, 128, 64};
constexpr MatmulFuncParams FUNC{false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT,
                                true,  true};
constexpr MatmulBiasParams BIAS{false};
constexpr MatmulConfig CONFIG = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(SHAPE, FUNC, BIAS);

template <bool TRANS>
using Input = MatmulType<TPosition::GM, CubeFormat::ND, float, TRANS>;
using Output = MatmulType<TPosition::GM, CubeFormat::ND, float>;
template <bool TA, bool TB>
constexpr MatmulApiStaticTiling TILING = GetMatmulApiTiling<Input<TB>, Input<TA>, Output, Output>(CONFIG);

template <uint32_t S>
constexpr MatmulShapeParams SMALL_SHAPE{S, S, S, S, S, S};
template <uint32_t S>
__aicore__ constexpr MatmulConfig MakeSmallConfig()
{
    auto c = GetMMConfig<MatmulConfigMode::CONFIG_MDL>(SMALL_SHAPE<S>, FUNC, BIAS);
    c.enUnitFlag = true;
    return c;
}
template <uint32_t S>
constexpr MatmulConfig SMALL_CONFIG = MakeSmallConfig<S>();
template <uint32_t S>
constexpr MatmulApiStaticTiling SMALL_TILING =
    GetMatmulApiTiling<Input<false>, Input<false>, Output, Output>(SMALL_CONFIG<S>);

template <uint32_t S>
__aicore__ inline void ComputeSmallCube(GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmSb22Tiling& t, TPipe& pipe)
{
    MatmulImpl<Input<false>, Input<false>, Output, Output, SMALL_TILING<S>> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    mm.SetOrgShape(S, S, S, S, t.tempLd);
    GlobalTensor<float> ag, bg, cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    for (uint64_t batch = GetBlockIdx(); batch < t.batches; batch += GetBlockNum()) {
        mm.SetSingleShape(S, S, S);
        mm.SetTensorA(bg[batch * t.strideB]);
        mm.SetTensorB(ag[batch * t.strideA]);
        mm.IterateAll(cg[batch * t.strideC]);
        mm.End();
    }
}

template <bool TA, bool TB>
__aicore__ inline void ComputeCube(GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmSb22Tiling& t, TPipe& pipe)
{
    MatmulImpl<Input<TB>, Input<TA>, Output, Output, TILING<TA, TB>> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    GlobalTensor<float> ag, bg, cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    const uint64_t rows = (t.n + TILE - 1) / TILE;
    const uint64_t cols = (t.m + TILE - 1) / TILE;
    // Column-major GEMM is row-major C^T = op(B)^T * op(A)^T.
    for (uint64_t task = GetBlockIdx(); task < rows * cols * t.batches; task += GetBlockNum()) {
        const uint64_t batch = task / (rows * cols);
        const uint64_t row = (task / cols % rows) * TILE;
        const uint64_t col = (task % cols) * TILE;
        mm.SetOrgShape(TB ? t.ldb : t.n, TA ? t.m : t.lda, TB ? t.k : t.ldb, TA ? t.lda : t.k, t.tempLd);
        for (int64_t startK = 0; startK < t.k; startK += 4096) {
            mm.SetSingleShape(
                t.n - row < TILE ? t.n - row : TILE, t.m - col < TILE ? t.m - col : TILE,
                t.k - startK < 4096 ? t.k - startK : 4096);
            mm.SetTensorA(bg[batch * t.strideB + (TB ? startK * t.ldb + row : row * t.ldb + startK)], TB);
            mm.SetTensorB(ag[batch * t.strideA + (TA ? col * t.lda + startK : startK * t.lda + col)], TA);
            mm.IterateAll(cg[batch * t.strideC + row * t.tempLd + col], startK != 0);
            mm.End();
        }
    }
}

// Netlib's non-transposed A path scales B before accumulating.
// With large alpha, moving that rounding after the reduction can
// exceed the ULP bound near cancellation. Preserve its FP32 order.
__aicore__ inline bool NeedsLargeAlphaPath(const GemmSb22Tiling& t)
{
    const float alphaMagnitude = t.alpha < 0.0f ? -t.alpha : t.alpha;
    return !t.transA && t.k > 0 && alphaMagnitude > 65536.0f;
}

// The host gates the vector path on m*n*k <= 4096. Padded two-dimensional DMA
// stores each UB row at a 32-byte boundary, so staging needs limited alignment
// headroom beyond the compact plane size.
constexpr uint32_t A_STAGE_MAX = 5760;
constexpr uint32_t B_STAGE_MAX = 8192;

__aicore__ inline uint32_t AlignFloatRow(uint32_t count)
{
    return (count + 7U) & ~7U;
}

// Stages op(A) rows [row, row+count) of a transposed A into aBlk.
// With transA the physical A is k×m column-major, so op(A) row (row+r) is
// A column (row+r): k consecutive GM elements starting at (row+r)*lda.
__aicore__ inline void StageATransposed(const GlobalTensor<float>& ag, const LocalTensor<float>& aBlk,
    const GemmSb22Tiling& t, uint64_t batch, uint64_t row, uint32_t count)
{
    const DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    const uint64_t base = batch * t.strideA;
    if (t.lda == t.k) {
        const DataCopyExtParams whole{1, count * static_cast<uint32_t>(t.k) * 4, 0, 0, 0};
        DataCopyPad(aBlk, ag[base + row * t.lda], whole, pad);
        return;
    }
    const uint32_t srcStride = static_cast<uint32_t>(t.lda - t.k) * sizeof(float);
    const DataCopyExtParams columns{
        static_cast<uint16_t>(count), static_cast<uint32_t>(t.k) * 4U, srcStride, 0, 0};
    DataCopyPad(aBlk, ag[base + row * t.lda], columns, pad);
}

// Stages the op(B) data needed by one output-column job into bBlk.
// transB=N: B is k×n column-major and op(B) column col is k consecutive elements.
// transB=T: B is n×k column-major and op(B)(p, col) = B(col, p) strides by ldb,
// so all k columns (each n consecutive elements) are staged into bBlk, using
// a 32-byte UB row pitch when the GM leading dimension is padded.
// For n < 8 the staging is skipped: k may reach 4096 there and 4-byte DMA
// transfers would outnumber the bounded scalar loads they replace.
__aicore__ inline void StageBColumn(const GlobalTensor<float>& bg, const LocalTensor<float>& bBlk,
    const GemmSb22Tiling& t, uint64_t batch, uint64_t col)
{
    const DataCopyPadExtParams<float> pad{false, 0, 0, 0};
    const uint64_t base = batch * t.strideB;
    if (!t.transB) {
        const DataCopyExtParams column{1, static_cast<uint32_t>(t.k) * 4, 0, 0, 0};
        DataCopyPad(bBlk, bg[base + col * t.ldb], column, pad);
        return;
    }
    if (t.n < 8)
        return;
    if (t.ldb == t.n) {
        const DataCopyExtParams whole{1, static_cast<uint32_t>(t.n) * t.k * 4, 0, 0, 0};
        DataCopyPad(bBlk, bg[base], whole, pad);
        return;
    }
    const uint32_t srcStride = static_cast<uint32_t>(t.ldb - t.n) * sizeof(float);
    const DataCopyExtParams columns{
        static_cast<uint16_t>(t.k), static_cast<uint32_t>(t.n) * 4U, srcStride, 0, 0};
    DataCopyPad(bBlk, bg[base], columns, pad);
}

// Reads op(B)(p, col); GM is touched only in the un-staged n < 8 case.
__aicore__ inline float LoadBElement(
    const GlobalTensor<float>& bg, const LocalTensor<float>& bBlk, const GemmSb22Tiling& t, uint64_t batch,
    uint64_t col, int p)
{
    if (!t.transB)
        return bBlk.GetValue(p);
    if (t.n < 8)
        return bg.GetValue(batch * t.strideB + static_cast<uint64_t>(p) * t.ldb + col);
    const uint32_t pitch = t.ldb == t.n ? static_cast<uint32_t>(t.n) : AlignFloatRow(t.n);
    return bBlk.GetValue(static_cast<uint32_t>(p) * pitch + static_cast<uint32_t>(col));
}

__aicore__ inline void VectorLargeAlphaRow(const GlobalTensor<float>& ag, const GlobalTensor<float>& bg,
    const GlobalTensor<float>& cg, const LocalTensor<float>& input, const LocalTensor<float>& old,
    const LocalTensor<float>& bBlk, LocalTensor<float>& acc, const GemmSb22Tiling& t, uint64_t batch,
    uint64_t col, uint64_t row, uint32_t count, uint64_t offset, const DataCopyExtParams& copy,
    const DataCopyPadExtParams<float>& pad)
{
    if (t.beta == 0.0f) {
        Duplicate(acc, 0.0f, count);
    } else {
        DataCopyPad(old, cg[offset], copy, pad);
        PipeBarrier<PIPE_ALL>();
        Muls(acc, old, t.beta, count);
    }
    PipeBarrier<PIPE_ALL>();
    // p stays the outer accumulation order, so each element's FMA chain and
    // rounding sequence are unchanged; only the data staging is vectorized.
    for (int p = 0; p < t.k; ++p) {
        const float bv = LoadBElement(bg, bBlk, t, batch, col, p);
        if (bv == 0.0f)
            continue;
        DataCopyPad(input, ag[batch * t.strideA + static_cast<uint64_t>(p) * t.lda + row], copy, pad);
        PipeBarrier<PIPE_ALL>();
        const float scaledB = t.alpha * bv;
        for (uint32_t r = 0; r < count; ++r)
            acc.SetValue(r, GemmSbFp32::Fma(input.GetValue(r), scaledB, acc.GetValue(r)));
    }
}

__aicore__ inline void VectorAccumulateLoop(const GlobalTensor<float>& ag, const GlobalTensor<float>& bg,
    const LocalTensor<float>& input, const LocalTensor<float>& old, LocalTensor<float>& acc,
    const LocalTensor<float>& aBlk, const LocalTensor<float>& bBlk, LocalTensor<int32_t> idx,
    const GemmSb22Tiling& t, uint64_t batch, uint64_t col, uint64_t row, uint32_t count,
    const DataCopyExtParams& copy, const DataCopyPadExtParams<float>& pad)
{
    for (int p = 0; p < t.k; ++p) {
        if (!t.transA) {
            DataCopyPad(input, ag[batch * t.strideA + static_cast<uint64_t>(p) * t.lda + row], copy, pad);
        } else {
            // idx[r] holds the staged UB row's byte offset; the p*4 byte shift
            // selects column p (Gather adds it to every index).
            Gather(input, aBlk, idx.ReinterpretCast<uint32_t>(), p * 4, count);
        }
        const float bv = LoadBElement(bg, bBlk, t, batch, col, p);
        PipeBarrier<PIPE_ALL>();
        // Scale finite products to avoid premature overflow before accumulation.
        // A separate multiply/add otherwise turns finite * finite + infinity into NaN.
        const float magnitude = bv < 0.0f ? -bv : bv;
        const float scale = magnitude > 1.0f && magnitude <= FP32_MAX_FINITE ?
                                (magnitude <= 8.0f ? 8.0f : magnitude) : 1.0f;
        Muls(acc, acc, 1.0f / scale, count);
        Duplicate(old, bv / scale, count);
        PipeBarrier<PIPE_ALL>();
        FusedMulAdd(input, old, acc, count);
        PipeBarrier<PIPE_ALL>();
        Muls(acc, input, scale, count);
        PipeBarrier<PIPE_ALL>();
    }
}

template <bool COMBINE>
__aicore__ inline void ComputeVector(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR temp, const GemmSb22Tiling& t, TPipe& pipe)
{
    // acc/input/old hold one row tile; aBlk/bBlk stage A/B data with room for
    // the 32-byte row alignment used by padded two-dimensional DMA. idx carries
    // the transposed-A gather offsets.
    TBuf<TPosition::VECCALC> accBuf, inBuf, oldBuf, aBuf, bBuf, idxBuf;
    pipe.InitBuffer(accBuf, VECTOR_TILE * sizeof(float));
    pipe.InitBuffer(inBuf, VECTOR_TILE * sizeof(float));
    pipe.InitBuffer(oldBuf, VECTOR_TILE * sizeof(float));
    pipe.InitBuffer(aBuf, A_STAGE_MAX * sizeof(float));
    pipe.InitBuffer(bBuf, B_STAGE_MAX * sizeof(float));
    pipe.InitBuffer(idxBuf, VECTOR_TILE * sizeof(int32_t));
    auto acc = accBuf.Get<float>();
    auto input = inBuf.Get<float>();
    auto old = oldBuf.Get<float>();
    auto aBlk = aBuf.Get<float>();
    auto bBlk = bBuf.Get<float>();
    auto idx = idxBuf.Get<int32_t>();
    if (!COMBINE && t.transA) {
        const uint32_t pitch = t.lda == t.k ? static_cast<uint32_t>(t.k) : AlignFloatRow(t.k);
        for (uint32_t r = 0; r < VECTOR_TILE; ++r)
            idx.SetValue(r, static_cast<int32_t>(r * pitch * sizeof(float)));
        PipeBarrier<PIPE_ALL>();
    }
    GlobalTensor<float> ag, bg, cg, tg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    tg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(temp));
    const uint64_t rowTiles = (t.m + VECTOR_TILE - 1) / VECTOR_TILE;
    const uint64_t jobs = rowTiles * t.n * t.batches;
    for (uint64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum()) {
        const uint64_t row = job % rowTiles * VECTOR_TILE;
        const uint64_t col = job / rowTiles % t.n;
        const uint64_t batch = job / (rowTiles * t.n);
        const uint32_t count = t.m - row < VECTOR_TILE ? t.m - row : VECTOR_TILE;
        const uint64_t offset = batch * t.strideC + col * t.ldc + row;
        const DataCopyExtParams copy{1, count * 4, 0, 0, 0};
        const DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        if constexpr (!COMBINE) {
            if (NeedsLargeAlphaPath(t)) {
                StageBColumn(bg, bBlk, t, batch, col);
                PipeBarrier<PIPE_ALL>();
                VectorLargeAlphaRow(ag, bg, cg, input, old, bBlk, acc, t, batch, col, row, count, offset, copy, pad);
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(cg[offset], acc, copy);
                PipeBarrier<PIPE_ALL>();
                continue;
            }
        }
        if constexpr (COMBINE) {
            DataCopyPad(acc, tg[(batch * t.n + col) * t.tempLd + row], copy, pad);
            PipeBarrier<PIPE_ALL>();
            Muls(acc, acc, t.alpha, count);
        } else {
            Duplicate(acc, 0.0f, count);
            PipeBarrier<PIPE_ALL>();
            if (t.alpha != 0.0f && t.k > 0) {
                StageBColumn(bg, bBlk, t, batch, col);
                if (t.transA)
                    StageATransposed(ag, aBlk, t, batch, row, count);
                PipeBarrier<PIPE_ALL>();
                VectorAccumulateLoop(ag, bg, input, old, acc, aBlk, bBlk, idx, t, batch, col, row, count, copy, pad);
                Muls(acc, acc, t.alpha, count);
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (t.beta != 0.0f) {
            DataCopyPad(old, cg[offset], copy, pad);
            PipeBarrier<PIPE_ALL>();
            Axpy(acc, old, t.beta, count);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(cg[offset], acc, copy);
        PipeBarrier<PIPE_ALL>();
    }
}
} // namespace

namespace {

template <uint32_t S>
__aicore__ inline void TinyInitIndices(
    const LocalTensor<int32_t>& ia, const LocalTensor<int32_t>& ib, const LocalTensor<uint16_t>& mask, uint32_t count)
{
    CreateVecIndex(ia, int32_t(0), count);
    CreateVecIndex(ib, int32_t(0), count);
    Duplicate(mask, static_cast<uint16_t>(~(S * (S - 1))), count * 2);
    PipeBarrier<PIPE_V>();
    And(ia.ReinterpretCast<uint16_t>(), ia.ReinterpretCast<uint16_t>(), mask, count * 2);
    PipeBarrier<PIPE_V>();
    Duplicate(mask, static_cast<uint16_t>(~(S - 1)), count * 2);
    PipeBarrier<PIPE_V>();
    And(ib.ReinterpretCast<uint16_t>(), ib.ReinterpretCast<uint16_t>(), mask, count * 2);
    PipeBarrier<PIPE_V>();
    ShiftLeft(ia, ia, int32_t(2), count);
    ShiftLeft(ib, ib, int32_t(2), count);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline bool TinyDetectWide(const LocalTensor<float>& la, const LocalTensor<float>& lb,
    LocalTensor<float>& ga, LocalTensor<float>& gb, LocalTensor<float>& acc, uint32_t count)
{
    Abs(ga, la, count);
    Abs(gb, lb, count);
    PipeBarrier<PIPE_V>();
    Max(ga, ga, gb, count);
    PipeBarrier<PIPE_V>();
    ReduceMax(acc, ga, gb, count, false);
    PipeBarrier<PIPE_ALL>();
    return acc.GetValue(0) > EXTREME_VALUE_THRESHOLD;
}

template <uint32_t S>
__aicore__ inline float TinyDotProduct(
    const LocalTensor<float>& la, const LocalTensor<float>& lb, uint32_t matrix, uint32_t col, uint32_t row)
{
    float value = 0.0f;
    for (uint32_t p = 0; p < S; ++p) {
        const float bv = lb.GetValue(matrix * S * S + col * S + p);
        if (bv != 0.0f)
            value = GemmSbFp32::Fma(la.GetValue(matrix * S * S + p * S + row), bv, value);
    }
    return value;
}

// Match the reference's single-rounding FMA when extreme terms cancel.
template <uint32_t S>
__aicore__ inline void TinySoftFma(
    const LocalTensor<float>& la, const LocalTensor<float>& lb, LocalTensor<float>& acc, uint32_t count)
{
    for (uint32_t matrix = 0; matrix < count / (S * S); ++matrix)
        for (uint32_t col = 0; col < S; ++col)
            for (uint32_t row = 0; row < S; ++row)
                acc.SetValue((matrix * S + col) * S + row, TinyDotProduct<S>(la, lb, matrix, col, row));
}

} // namespace

template <uint32_t S, uint32_t GROUP>
__aicore__ inline void ComputeTiny(GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmSb22Tiling& t, TPipe& pipe)
{
    constexpr uint32_t COUNT = GROUP * S * S;
    TBuf<TPosition::VECCALC> aBuf, bBuf, accBuf, gaBuf, gbBuf, iaBuf, ibBuf;
    pipe.InitBuffer(aBuf, COUNT * 4);
    pipe.InitBuffer(bBuf, COUNT * 4);
    pipe.InitBuffer(accBuf, COUNT * 4);
    pipe.InitBuffer(gaBuf, COUNT * 4);
    pipe.InitBuffer(gbBuf, COUNT * 4);
    pipe.InitBuffer(iaBuf, COUNT * 4);
    pipe.InitBuffer(ibBuf, COUNT * 4);
    auto la = aBuf.Get<float>(), lb = bBuf.Get<float>(), acc = accBuf.Get<float>();
    auto ga = gaBuf.Get<float>(), gb = gbBuf.Get<float>();
    auto ia = iaBuf.Get<int32_t>(), ib = ibBuf.Get<int32_t>();
    auto mask = gaBuf.Get<uint16_t>();
    TinyInitIndices<S>(ia, ib, mask, COUNT);
    GlobalTensor<float> ag, bg, cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    for (uint64_t batch = GetBlockIdx() * GROUP; batch < t.batches; batch += GetBlockNum() * GROUP) {
        const uint32_t count = (t.batches - batch < GROUP ? t.batches - batch : GROUP) * S * S;
        const uint64_t offset = batch * S * S;
        DataCopy(la, ag[offset], count);
        DataCopy(lb, bg[offset], count);
        PipeBarrier<PIPE_ALL>();
        const bool wide = TinyDetectWide(la, lb, ga, gb, acc, count);
        Duplicate(acc, 0.0f, count);
        PipeBarrier<PIPE_V>();
        if (wide) {
            TinySoftFma<S>(la, lb, acc, count);
        } else {
            for (uint32_t p = 0; p < S; ++p) {
                Gather(ga, la, ia.ReinterpretCast<uint32_t>(), p * S * 4, count);
                Gather(gb, lb, ib.ReinterpretCast<uint32_t>(), p * 4, count);
                PipeBarrier<PIPE_V>();
                FusedMulAdd(ga, gb, acc, count);
                PipeBarrier<PIPE_V>();
                auto previous = acc;
                acc = ga;
                ga = previous;
            }
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(cg[offset], acc, count);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" __global__ __aicore__ __vector__ void gemm_sb22_tiny(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmSb22Tiling t)
{
    TPipe pipe;
    if (t.batches <= 64)
        ComputeTiny<8, 2>(a, b, c, t, pipe);
    else if (t.tinyBatchGroup == 32)
        ComputeTiny<8, 32>(a, b, c, t, pipe);
    else
        ComputeTiny<8, 8>(a, b, c, t, pipe);
}

void GemmSb22Tiny(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmSb22Tiling t)
{
    gemm_sb22_tiny<<<blocks, nullptr, stream>>>(a, b, c, t);
}

using BatchInput = MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NORMAL>;
using BatchOutput = MatmulType<TPosition::GM, CubeFormat::ND, float, false, LayoutMode::NORMAL>;
__aicore__ constexpr MatmulConfig MakeBatchConfig()
{
    auto cfg = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1);
    cfg.bmmOutMode = BatchOutMode::MULTI_BATCH;
    return cfg;
}
constexpr MatmulConfig BATCH_CONFIG = MakeBatchConfig();
template <uint32_t S>
__aicore__ constexpr MatmulApiStaticTiling MakeBatchStatic()
{
    MatmulApiStaticTiling v;
    v.cfg = BATCH_CONFIG;
    v.usedCoreNum = 1;
    v.M = v.N = v.Ka = v.Kb = S;
    v.singleCoreM = v.singleCoreN = v.singleCoreK = S;
    v.baseM = v.baseN = v.baseK = S;
    v.depthA1 = v.depthB1 = v.stepM = v.stepN = 1;
    v.isBias = v.transLength = v.iterateOrder = v.shareMode = 0;
    v.shareL1Size = 2 * 16 * S * S * sizeof(float);
    v.shareL0CSize = 8 * S * S * sizeof(float);
    v.shareUbSize = 0;

    v.stepKa = v.stepKb = 1;
    v.depthAL1CacheUB = v.depthBL1CacheUB = 0;
    v.dbL0A = v.dbL0B = 2;
    v.dbL0C = 1;
    v.ALayoutInfoB = v.BLayoutInfoB = v.CLayoutInfoB = 16;
    v.ALayoutInfoS = v.ALayoutInfoD = v.BLayoutInfoS = v.BLayoutInfoD = S;
    v.CLayoutInfoS1 = v.CLayoutInfoS2 = S;
    v.ALayoutInfoN = v.ALayoutInfoG = v.BLayoutInfoN = v.BLayoutInfoG = 1;
    v.CLayoutInfoN = v.CLayoutInfoG = 1;
    // Two 8-matrix L1 buffers, with up to 8 results cached in L0C.
    v.BatchNum = 16;
    v.mxTypePara = 0;
    return v;
}
template <uint32_t S>
constexpr MatmulApiStaticTiling BATCH_STATIC = MakeBatchStatic<S>();
template <uint32_t S>
__aicore__ inline void ComputeBatchCube(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmSb22Tiling t, TPipe& pipe)
{
    MatmulImpl<BatchInput, BatchInput, BatchOutput, Output, BATCH_STATIC<S>> mm;
    mm.SetNBatchOutNum(8);
    mm.SetSubBlockIdx(0);
    mm.Init(static_cast<const TCubeTiling*>(nullptr), &pipe);
    mm.DisableBias();
    mm.SetHF32(false);
    GlobalTensor<float> ag, bg, cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    const uint32_t group = t.batches <= 256 ? 12 : 16;
    for (uint64_t batch = GetBlockIdx() * group; batch < t.batches; batch += GetBlockNum() * group) {
        const uint32_t count = t.batches - batch < group ? t.batches - batch : group;
        mm.SetTensorA(bg[batch * t.strideB]);
        mm.SetTensorB(ag[batch * t.strideA]);
        mm.SetBatchNum(count, count);
        mm.IterateBatch(cg[batch * t.strideC], false, 0, false);
        mm.End();
    }
}

namespace {

__aicore__ inline bool ScanNeedsRepair(
    LocalTensor<float>& av, LocalTensor<float>& bv, LocalTensor<float>& cv, LocalTensor<float>& x,
    LocalTensor<float>& y, uint32_t count)
{
    Abs(cv, av, count);
    Abs(x, bv, count);
    PipeBarrier<PIPE_V>();
    Max(cv, cv, x, count);
    PipeBarrier<PIPE_V>();
    ReduceMax(x, cv, y, count, false);
    PipeBarrier<PIPE_ALL>();
    return x.GetValue(0) > EXTREME_VALUE_THRESHOLD;
}

// Rebuilds the small-matrix sums with scaled FMA when the Cube path may have overflowed.
template <uint32_t S>
__aicore__ inline void RepairMatrixSums(const LocalTensor<float>& av, const LocalTensor<float>& bv,
    LocalTensor<float>& cv, LocalTensor<float>& x, LocalTensor<float>& y, uint32_t count)
{
    for (uint32_t matrix = 0; matrix < count / (S * S); ++matrix) {
        for (uint32_t col = 0; col < S; ++col) {
            auto dst = cv[(matrix * S + col) * S];
            for (uint32_t k = 0; k < S; ++k) {
                DataCopy(x, av[matrix * S * S + k * S], S);
                const float bval = bv.GetValue(matrix * S * S + col * S + k);
                const float magnitude = bval < 0.0f ? -bval : bval;
                const float scale = magnitude > 1.0f && magnitude <= FP32_MAX_FINITE ?
                                        (magnitude <= 8.0f ? 8.0f : magnitude) :
                                        1.0f;
                PipeBarrier<PIPE_ALL>();
                Muls(dst, dst, 1.0f / scale, S);
                Duplicate(y, bval / scale, S);
                PipeBarrier<PIPE_V>();
                FusedMulAdd(x, y, dst, S);
                PipeBarrier<PIPE_V>();
                Muls(dst, x, scale, S);
                PipeBarrier<PIPE_V>();
            }
        }
    }
}

} // namespace

template <uint32_t S>
__aicore__ inline void RepairBatchOverflow(GM_ADDR a, GM_ADDR b, GM_ADDR c, const GemmSb22Tiling& t, TPipe& pipe)
{
    constexpr uint32_t GROUP = 8, COUNT = GROUP * S * S;
    TBuf<TPosition::VECCALC> ab, bb, cb, xb, yb;
    pipe.InitBuffer(ab, COUNT * 4);
    pipe.InitBuffer(bb, COUNT * 4);
    pipe.InitBuffer(cb, COUNT * 4);
    pipe.InitBuffer(xb, COUNT * 4);
    pipe.InitBuffer(yb, COUNT * 4);
    auto av = ab.Get<float>(), bv = bb.Get<float>(), cv = cb.Get<float>();
    auto x = xb.Get<float>(), y = yb.Get<float>();
    GlobalTensor<float> ag, bg, cg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    // Scan while Cube runs; synchronize before any corrective C write.
    bool waited = false;
    const uint32_t group = t.batches <= 256 ? 6 : 8;
    for (uint64_t batch = GetBlockIdx() * group; batch < t.batches; batch += GetBlockNum() * 2 * group) {
        const uint32_t count = (t.batches - batch < group ? t.batches - batch : group) * S * S;
        DataCopy(av, ag[batch * S * S], count);
        DataCopy(bv, bg[batch * S * S], count);
        PipeBarrier<PIPE_ALL>();
        if (!ScanNeedsRepair(av, bv, cv, x, y, count))
            continue;
        if (!waited) {
            CrossCoreWaitFlag<2>(0);
            waited = true;
        }
        Duplicate(cv, 0.0f, count);
        PipeBarrier<PIPE_V>();
        RepairMatrixSums<S>(av, bv, cv, x, y, count);
        DataCopy(cg[batch * S * S], cv, count);
        PipeBarrier<PIPE_ALL>();
    }
    if (!waited)
        CrossCoreWaitFlag<2>(0);
}
extern "C" __global__ __aicore__ void gemm_sb22_batch_cube(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmSb22Tiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    if ASCEND_IS_AIC {
        if (t.m == 16) {
            ComputeBatchCube<16>(a, b, c, t, pipe);
        } else {
            ComputeBatchCube<32>(a, b, c, t, pipe);
        }
        CrossCoreSetFlag<2, PIPE_FIX>(0);
    } else {
        if (t.m == 16)
            RepairBatchOverflow<16>(a, b, c, t, pipe);
        else
            RepairBatchOverflow<32>(a, b, c, t, pipe);
    }
}

void GemmSb22BatchCube(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmSb22Tiling t)
{
    gemm_sb22_batch_cube<<<blocks, nullptr, stream>>>(a, b, c, t);
}

extern "C" __global__ __aicore__ __cube__ void gemm_sb22_cube(GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmSb22Tiling t)
{
    TPipe pipe;
    if (!t.transA && !t.transB && t.m == t.n && t.n == t.k && t.lda == t.m && t.ldb == t.m) {
        if (t.m == 16) {
            ComputeSmallCube<16>(a, b, c, t, pipe);
            return;
        }
        if (t.m == 32) {
            ComputeSmallCube<32>(a, b, c, t, pipe);
            return;
        }
    }
    if (t.transA) {
        if (t.transB)
            ComputeCube<true, true>(a, b, c, t, pipe);
        else
            ComputeCube<true, false>(a, b, c, t, pipe);
    } else {
        if (t.transB)
            ComputeCube<false, true>(a, b, c, t, pipe);
        else
            ComputeCube<false, false>(a, b, c, t, pipe);
    }
}

extern "C" __global__ __aicore__ __vector__ void gemm_sb22_vector(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR temp, GemmSb22Tiling t, bool combine)
{
    TPipe pipe;
    if (combine)
        ComputeVector<true>(a, b, c, temp, t, pipe);
    else
        ComputeVector<false>(a, b, c, temp, t, pipe);
}

void GemmSb22Cube(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmSb22Tiling t)
{
    gemm_sb22_cube<<<blocks, nullptr, stream>>>(a, b, c, t);
}

void GemmSb22Vector(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR temp, GemmSb22Tiling t, bool combine)
{
    gemm_sb22_vector<<<blocks, nullptr, stream>>>(a, b, c, temp, t, combine);
}
