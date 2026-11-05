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
 * \file csymm_kernel.cpp
 * \brief CSYMM kernels for Atlas A2 training series (arch22).
 *
 *        side='L': C = alpha*A*B + beta*C   (A is m x m symmetric)
 *        side='R': C = alpha*B*A + beta*C   (A is n x n symmetric)
 *        alpha and beta are both complex; C is a general m x n matrix. All matrices
 *        are column-major complex64 and only A's uplo triangle is referenced.
 *
 *        Phases 0 and 1 are the shared implementations in
 *        common/helper/complex_blas3_arch22.h; only the combine stage below is
 *        operator-specific.
 *
 *        The combine stage is a plain streaming pass: unlike the rank-K operators it
 *        has no triangle to respect and reads its temps untransposed, because the
 *        GEMM produced C^T in row-major and a column of the column-major C is a row
 *        of C^T.
 *
 *        Identical to CHEMM except that Phase 0b mirrors A's imaginary part with a
 *        plus rather than a minus, which the host selects through imagSign.
 */

#include "kernel_operator.h"
#include "common/helper/complex_blas3_arch22.h"
#include "common/helper/complex_blas3_expand_arch22.h"
#include "csymm_kernel.h"
#include "csymm_tiling_data.h"

using namespace AscendC;

namespace {
// Complex elements per Phase 2 inner iteration; a whole multiple of a 256B fp32
// repeat so the GatherMask repeat count is exact.
constexpr uint32_t COMBINE_CHUNK = 1024;
static_assert(COMBINE_CHUNK % cblas3::ELENUM_REPEAT_FP32 == 0, "COMBINE_CHUNK must be a whole number of fp32 repeats");

// TPipe is deliberately NOT a member: rule R3 requires it to stay a local of the
// kernel entry.
struct CsymmCombineCtx {
    GlobalTensor<float> t1Gm;
    GlobalTensor<float> t2Gm;
    GlobalTensor<float> t3Gm;
    GlobalTensor<float> t4Gm;
    GlobalTensor<float> cGm;
    LocalTensor<float> crci; // [re | im] halves, interleaved by Gather on the way out
    LocalTensor<float> cr;
    LocalTensor<float> ci;
    LocalTensor<float> pr;
    LocalTensor<float> pi;
    LocalTensor<float> tmpA;
    LocalTensor<float> tmpB;
    LocalTensor<float> stage;
    LocalTensor<uint32_t> idxU;
    bool applyAlpha;
    bool applyBeta;
};

// The nine UB scratch buffers backing a CsymmCombineCtx. Declared in the caller's
// frame so their lifetime spans every CombineOneChunk call: the LocalTensors handed
// to ctx below stay in use for the whole column loop, so the TBufs they come from
// must outlive this function. Same arrangement as the shared layer's
// MirrorTriBuffers.
struct CsymmCombineBuffers {
    TBuf<TPosition::VECCALC> crci;
    TBuf<TPosition::VECCALC> pr;
    TBuf<TPosition::VECCALC> pi;
    TBuf<TPosition::VECCALC> tmpA;
    TBuf<TPosition::VECCALC> tmpB;
    TBuf<TPosition::VECCALC> stage;
    TBuf<TPosition::VECCALC> idx;
    TBuf<TPosition::VECCALC> aux0;
    TBuf<TPosition::VECCALC> aux1;
};

__aicore__ inline void InitCombineCtx(
    TPipe& pipe, CsymmCombineBuffers& bufs, CsymmCombineCtx& ctx, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4,
    GM_ADDR c, const CsymmCombineTilingData& tiling)
{
    // alpha = (0,0) makes the host skip the whole temp allocation, so t1..t4 come
    // in null. The alpha term is not evaluated in that case either (skipAlphaTerm
    // gates it below), so leave those four tensors unbound instead of handing a
    // null address to SetGlobalBuffer. beta may still be anything, and C is always
    // bound.
    if (tiling.skipAlphaTerm == 0U) {
        ctx.t1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t1));
        ctx.t2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t2));
        ctx.t3Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t3));
        ctx.t4Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t4));
    }
    ctx.cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));

    constexpr uint32_t chunk = COMBINE_CHUNK;
    constexpr uint32_t pairElems = chunk * cblas3::COMPLEX_ELENUM;

    pipe.InitBuffer(bufs.crci, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.pr, chunk * sizeof(float));
    pipe.InitBuffer(bufs.pi, chunk * sizeof(float));
    pipe.InitBuffer(bufs.tmpA, chunk * sizeof(float));
    pipe.InitBuffer(bufs.tmpB, chunk * sizeof(float));
    pipe.InitBuffer(bufs.stage, pairElems * sizeof(float));
    pipe.InitBuffer(bufs.idx, pairElems * sizeof(int32_t));
    pipe.InitBuffer(bufs.aux0, pairElems * sizeof(int32_t));
    pipe.InitBuffer(bufs.aux1, pairElems * sizeof(int32_t));

    ctx.crci = bufs.crci.Get<float>();
    ctx.cr = ctx.crci;
    ctx.ci = ctx.crci[chunk];
    ctx.pr = bufs.pr.Get<float>();
    ctx.pi = bufs.pi.Get<float>();
    ctx.tmpA = bufs.tmpA.Get<float>();
    ctx.tmpB = bufs.tmpB.Get<float>();
    ctx.stage = bufs.stage.Get<float>();

    LocalTensor<int32_t> idx = bufs.idx.Get<int32_t>();
    cblas3::BuildInterleaveIndex(idx, bufs.aux0.Get<int32_t>(), bufs.aux1.Get<int32_t>(), chunk, chunk);
    ctx.idxU = idx.ReinterpretCast<uint32_t>();

    ctx.applyAlpha = (tiling.skipAlphaTerm == 0U);
    ctx.applyBeta = (tiling.isBetaZero == 0U);
}

//   Pr = t1 - t2                       Pi = t3 + t4
//   Cr = ar*Pr - ai*Pi + br*Cr_old - bi*Ci_old
//   Ci = ar*Pi + ai*Pr + br*Ci_old + bi*Cr_old
// over `len` consecutive rows of column `col`.
// Pr = t1 - t2, Pi = t3 + t4, then Cr = ar*Pr - ai*Pi and Ci = ar*Pi + ai*Pr.
// A zero component of alpha is skipped rather than multiplied through. The real
// and imaginary sums are independent and either can reach the fp32 range limit on
// its own, so multiplying an Inf by a component of 0 would yield NaN and carry it
// into the component that is otherwise finite. Skipping matches the BLAS
// convention that a zero scalar leaves its operand unreferenced.
__aicore__ inline void ApplyAlphaScale(
    CsymmCombineCtx& ctx, const CsymmCombineTilingData& tiling, uint64_t tempOffset, uint32_t len)
{
    DataCopyExtParams realParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};

    const bool paired = (tiling.pairedTemps != 0U);
    DataCopyPad(ctx.pr, ctx.t1Gm[tempOffset], realParams, padParams);
    DataCopyPad(ctx.pi, ctx.t3Gm[tempOffset], realParams, padParams);
    if (!paired) {
        DataCopyPad(ctx.tmpA, ctx.t2Gm[tempOffset], realParams, padParams);
        DataCopyPad(ctx.tmpB, ctx.t4Gm[tempOffset], realParams, padParams);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    if (!paired) {
        Sub(ctx.pr, ctx.pr, ctx.tmpA, len);
        Add(ctx.pi, ctx.pi, ctx.tmpB, len);
        PipeBarrier<PIPE_V>();
    }
    // Undo the Phase 0 scaling. Both factors are powers of two, so for any value
    // that stayed normal on the way down this pair of multiplies is exact; where
    // the unscaled sum is genuinely out of range it overflows here, as it should.
    if (tiling.unscaleEnabled != 0U) {
        Muls(ctx.pr, ctx.pr, tiling.unscaleFactor, len);
        Muls(ctx.pi, ctx.pi, tiling.unscaleFactor, len);
        PipeBarrier<PIPE_V>();
    }

    if (tiling.alphaRe != 0.0F) {
        Muls(ctx.cr, ctx.pr, tiling.alphaRe, len);
        Muls(ctx.ci, ctx.pi, tiling.alphaRe, len);
    } else {
        Duplicate(ctx.cr, 0.0F, len);
        Duplicate(ctx.ci, 0.0F, len);
    }
    if (tiling.alphaIm != 0.0F) {
        Muls(ctx.tmpA, ctx.pi, tiling.alphaIm, len);
        Muls(ctx.tmpB, ctx.pr, tiling.alphaIm, len);
        PipeBarrier<PIPE_V>();
        Sub(ctx.cr, ctx.cr, ctx.tmpA, len);
        Add(ctx.ci, ctx.ci, ctx.tmpB, len);
    }
    PipeBarrier<PIPE_V>();
}

// Adds beta * C_old. beta is complex, so the old value contributes to both parts.
__aicore__ inline void ApplyBetaScale(
    CsymmCombineCtx& ctx, const CsymmCombineTilingData& tiling, uint64_t cOffset, uint32_t len)
{
    uint64_t rsvdCnt = 0;
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    DataCopyExtParams cParams{1U, len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(ctx.stage, ctx.cGm[cOffset], cParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    const uint16_t repeats = static_cast<uint16_t>(
        cblas3::CeilDivU(len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), cblas3::BYTENUM_REPEAT));
    GatherMask(ctx.pr, ctx.stage, 1, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    GatherMask(ctx.pi, ctx.stage, 2, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    if (tiling.betaRe != 0.0F) {
        Axpy(ctx.cr, ctx.pr, tiling.betaRe, static_cast<int32_t>(len));
        Axpy(ctx.ci, ctx.pi, tiling.betaRe, static_cast<int32_t>(len));
        PipeBarrier<PIPE_V>();
    }
    if (tiling.betaIm != 0.0F) {
        Muls(ctx.tmpA, ctx.pi, tiling.betaIm, len);
        Muls(ctx.tmpB, ctx.pr, tiling.betaIm, len);
        PipeBarrier<PIPE_V>();
        Sub(ctx.cr, ctx.cr, ctx.tmpA, len);
        Add(ctx.ci, ctx.ci, ctx.tmpB, len);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void CombineOneChunk(
    CsymmCombineCtx& ctx, const CsymmCombineTilingData& tiling, uint32_t col, uint32_t row, uint32_t len)
{
    const uint64_t tempOffset = static_cast<uint64_t>(col) * tiling.tempLdc + row;
    const uint64_t cOffset = (static_cast<uint64_t>(col) * tiling.ldc + row) * cblas3::COMPLEX_ELENUM;

    if (ctx.applyAlpha) {
        ApplyAlphaScale(ctx, tiling, tempOffset, len);
    } else {
        Duplicate(ctx.cr, 0.0F, len);
        Duplicate(ctx.ci, 0.0F, len);
        PipeBarrier<PIPE_V>();
    }

    if (ctx.applyBeta) {
        ApplyBetaScale(ctx, tiling, cOffset, len);
    }

    Gather(ctx.stage, ctx.crci, ctx.idxU, 0U, len * cblas3::COMPLEX_ELENUM);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopyExtParams outParams{1U, len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(ctx.cGm[cOffset], ctx.stage, outParams);

    // Both queues must wait on MTE3 before the next iteration reuses `stage`.
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}
} // namespace

// ===========================================================================
//  Kernel entries
// ===========================================================================
extern "C" __global__ __aicore__ void csymm_split_kernel(
    GM_ADDR src, GM_ADDR re, GM_ADDR im, CBlas3SplitTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    cblas3::SplitBody(src, re, im, tiling);
}

extern "C" __global__ __aicore__ void csymm_expand_kernel(GM_ADDR re, GM_ADDR im, CBlas3ExpandTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    cblas3::MirrorTriBody(re, im, tiling);
}

extern "C" __global__ __aicore__ void csymm_gemm_kernel(
    GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    cblas3::GemmBody<cblas3::MatmulPlainBoth, CBLAS3_GEMM_TRANS_NONE>(left, right, out, tiling);
}

extern "C" __global__ __aicore__ void csymm_combine_kernel(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c, CsymmCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());

    TPipe pipe;
    CsymmCombineBuffers bufs;
    CsymmCombineCtx ctx;
    InitCombineCtx(pipe, bufs, ctx, t1, t2, t3, t4, c, tiling);

    // Every column carries the same m elements, so a plain round robin balances.
    for (uint32_t col = coreIdx; col < tiling.n; col += coreNum) {
        for (uint32_t off = 0; off < tiling.m; off += COMBINE_CHUNK) {
            const uint32_t len = (off + COMBINE_CHUNK > tiling.m) ? (tiling.m - off) : COMBINE_CHUNK;
            CombineOneChunk(ctx, tiling, col, off, len);
        }
    }
}

// ===========================================================================
//  Launchers
// ===========================================================================
void csymm_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR src, GM_ADDR re, GM_ADDR im, CBlas3SplitTilingData tiling)
{
    csymm_split_kernel<<<blockDim, nullptr, stream>>>(src, re, im, tiling);
}

void csymm_expand_kernel_do(uint32_t blockDim, void* stream, GM_ADDR re, GM_ADDR im, CBlas3ExpandTilingData tiling)
{
    csymm_expand_kernel<<<blockDim, nullptr, stream>>>(re, im, tiling);
}

void csymm_gemm_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    csymm_gemm_kernel<<<blockDim, nullptr, stream>>>(left, right, out, tiling);
}

void csymm_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c,
    CsymmCombineTilingData tiling)
{
    csymm_combine_kernel<<<blockDim, nullptr, stream>>>(t1, t2, t3, t4, c, tiling);
}

uint32_t csymm_gemm_single_k() { return cblas3::SINGLE_K; }
