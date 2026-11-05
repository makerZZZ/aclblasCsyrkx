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
 * \file csyrk_kernel.cpp
 * \brief CSYRK kernels for Atlas A2 training series (arch22).
 *
 *        C = alpha * A * A^T + beta * C   (trans = 'N', A is n x k)
 *        C = alpha * A^T * A + beta * C   (trans = 'T', A is k x n)
 *        alpha and beta are complex; C is symmetric and only the uplo triangle is
 *        referenced and updated. All matrices are column-major complex64.
 *
 *        Phases 0 and 1 are the shared implementations in
 *        common/helper/complex_blas3_arch22.h; only the combine stage below is
 *        operator-specific.
 *
 *        Phase 1 uses K-concatenated operands, so it needs only two GEMMs and
 *        produces Pr and Pi directly:
 *          Pr = P*Q^T = Ar*Ar^T - Ai*Ai^T
 *          Pi = P*R^T = Ar*Ai^T + Ai*Ar^T
 *        with P = [Ar|Ai], Q = [Ar|-Ai], R = [Ai|Ar] concatenated along K. Doing
 *        the subtraction inside one Cube accumulation chain, rather than between
 *        two separately rounded GEMM results, is what keeps the real part
 *        accurate: Ar*Ar^T and Ai*Ai^T are each O(k) in magnitude while their
 *        difference is only O(sqrt(k)).
 *
 *        Both Pr and Pi are symmetric, so the column-major (i.e. transposed) read
 *        that Phase 2 performs on the temps yields the same value -- unlike CHERK,
 *        no pointer swap is needed.
 */

#include "kernel_operator.h"
#include "common/helper/complex_blas3_arch22.h"
#include "csyrk_kernel.h"
#include "csyrk_tiling_data.h"

using namespace AscendC;

namespace {
// Complex elements handled by one Phase 2 inner iteration. A whole multiple of a
// 256B fp32 repeat so the GatherMask repeat count is exact. The buffer set below
// totals roughly 70KB, well inside the 192KB AIV UB.
constexpr uint32_t COMBINE_CHUNK = 1024;
static_assert(COMBINE_CHUNK % cblas3::ELENUM_REPEAT_FP32 == 0, "COMBINE_CHUNK must be a whole number of fp32 repeats");

// Everything the combine chunk loop needs. TPipe is deliberately NOT a member:
// rule R3 requires it to stay a local of the kernel entry.
struct CsyrkCombineCtx {
    GlobalTensor<float> prGm;
    GlobalTensor<float> piGm;
    GlobalTensor<float> cGm;
    LocalTensor<float> crci; // [re | im] halves, interleaved by Gather
    LocalTensor<float> cr;
    LocalTensor<float> ci;
    LocalTensor<float> pr; // Pr before scaling
    LocalTensor<float> pi; // Pi before scaling
    LocalTensor<float> tmpA;
    LocalTensor<float> tmpB;
    LocalTensor<float> stage; // interleaved C_old in / C_new out
    LocalTensor<uint32_t> idxU;
    bool applyAlpha;
    bool applyBeta;
};

// The nine UB scratch buffers backing a CsyrkCombineCtx. Declared in the caller's
// frame so their lifetime spans every CombineOneChunk call: the LocalTensors handed
// to ctx below stay in use for the whole column loop, so the TBufs they come from
// must outlive this function. Same arrangement as the shared layer's
// MirrorTriBuffers.
struct CsyrkCombineBuffers {
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
    TPipe& pipe, CsyrkCombineBuffers& bufs, CsyrkCombineCtx& ctx, GM_ADDR pr, GM_ADDR pi, GM_ADDR c,
    const CsyrkCombineTilingData& tiling)
{
    // alpha = (0,0) or k = 0 makes the host skip the whole temp allocation, so pr
    // and pi come in null. The alpha term is not evaluated in that case either
    // (skipAlphaTerm gates it below), so leave those two tensors unbound instead of
    // handing a null address to SetGlobalBuffer. beta may still be anything, and C
    // is always bound.
    if (tiling.skipAlphaTerm == 0U) {
        ctx.prGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pr));
        ctx.piGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pi));
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

// Cr = ar*Pr - ai*Pi, Ci = ar*Pi + ai*Pr. Pr and Pi come straight from Phase 1:
// the sign pattern was folded into the K-concatenated operands, so the only work
// left here is the complex multiply by alpha.
//
// A zero component of alpha is skipped rather than multiplied through. Pr and Pi
// are independent sums and either can reach the fp32 range limit on its own: on
// the RANDOM_EXTREME shapes the input carries +/-FLT_MAX, a single product
// overflows, and the imaginary sum (whose two terms share a sign, so nothing
// cancels) goes to +/-Inf while the real sum stays finite. Multiplying that Inf
// by an alpha component of 0 yields NaN, and subtracting it would carry the NaN
// into a real part that is otherwise exact. Skipping matches the BLAS convention
// that a zero scalar leaves its operand unreferenced.
__aicore__ inline void ApplyAlphaScale(
    CsyrkCombineCtx& ctx, const CsyrkCombineTilingData& tiling, uint64_t tempOffset, uint32_t len)
{
    DataCopyExtParams realParams{1U, len * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};

    DataCopyPad(ctx.pr, ctx.prGm[tempOffset], realParams, padParams);
    DataCopyPad(ctx.pi, ctx.piGm[tempOffset], realParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

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

// Cr += br*Cr_old - bi*Ci_old, Ci += br*Ci_old + bi*Cr_old. pr/pi are free to
// use as scratch here, the alpha term having already been folded into cr/ci.
// A zero component of beta is skipped for the same reason as in ApplyAlphaScale:
// the incoming C may itself carry a non-finite value, and 0 * Inf is NaN.
__aicore__ inline void ApplyBetaScale(
    CsyrkCombineCtx& ctx, const CsyrkCombineTilingData& tiling, uint64_t cOffset, uint32_t len)
{
    uint64_t rsvdCnt = 0;
    DataCopyPadExtParams<float> padParams{false, 0U, 0U, 0.0F};
    DataCopyExtParams cParams{1U, len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(ctx.stage, ctx.cGm[cOffset], cParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    const uint16_t repeats = static_cast<uint16_t>(
        cblas3::CeilDivU(len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), cblas3::BYTENUM_REPEAT));
    // tmpA = Cr_old, tmpB = Ci_old
    GatherMask(ctx.tmpA, ctx.stage, 1, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    GatherMask(ctx.tmpB, ctx.stage, 2, false, 0, {1, repeats, 8, 8}, rsvdCnt);
    PipeBarrier<PIPE_V>();

    if (tiling.betaRe != 0.0F) {
        Muls(ctx.pr, ctx.tmpA, tiling.betaRe, len);
        Muls(ctx.pi, ctx.tmpB, tiling.betaRe, len);
        PipeBarrier<PIPE_V>();
        Add(ctx.cr, ctx.cr, ctx.pr, len);
        Add(ctx.ci, ctx.ci, ctx.pi, len);
        PipeBarrier<PIPE_V>();
    }
    if (tiling.betaIm != 0.0F) {
        Muls(ctx.pr, ctx.tmpB, tiling.betaIm, len);
        Muls(ctx.pi, ctx.tmpA, tiling.betaIm, len);
        PipeBarrier<PIPE_V>();
        Sub(ctx.cr, ctx.cr, ctx.pr, len);
        Add(ctx.ci, ctx.ci, ctx.pi, len);
        PipeBarrier<PIPE_V>();
    }
}

// Complex scaling, expanded from C = alpha * P + beta * C_old:
//   Cr = ar*Pr - ai*Pi + br*Cr_old - bi*Ci_old
//   Ci = ar*Pi + ai*Pr + br*Ci_old + bi*Cr_old
// `len` elements of one contiguous run of rows inside column `col`.
__aicore__ inline void CombineOneChunk(
    CsyrkCombineCtx& ctx, const CsyrkCombineTilingData& tiling, uint32_t col, uint32_t row, uint32_t len)
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

    // C is symmetric, not Hermitian: the diagonal imaginary part is a real value
    // of the result and must be kept. This is the one place where CSYRK must NOT
    // call cblas3::ZeroDiagonalImag.

    Gather(ctx.stage, ctx.crci, ctx.idxU, 0U, len * cblas3::COMPLEX_ELENUM);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopyExtParams outParams{1U, len * cblas3::COMPLEX_ELENUM * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U};
    DataCopyPad(ctx.cGm[cOffset], ctx.stage, outParams);

    // Both queues must wait on MTE3 before the next iteration reuses `stage`:
    // MTE3_V covers the Gather that writes it, MTE3_MTE2 the DataCopyPad that
    // reads C_old into it. Relying on MTE2_V transitivity would break on the path
    // where neither alpha nor beta is applied, which issues no MTE2 at all.
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}
} // namespace

// ===========================================================================
//  Kernel entries
// ===========================================================================
extern "C" __global__ __aicore__ void csyrk_split_kernel(
    GM_ADDR a, GM_ADDR p, GM_ADDR q, GM_ADDR r, CBlas3SplitConcatTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    cblas3::SplitConcatBody(a, p, q, r, tiling);
}

extern "C" __global__ __aicore__ void csyrk_gemm_kernel_tl(
    GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    cblas3::GemmBody<cblas3::MatmulTransLeft, CBLAS3_GEMM_TRANS_LEFT>(left, right, out, tiling);
}

extern "C" __global__ __aicore__ void csyrk_gemm_kernel_tr(
    GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    cblas3::GemmBody<cblas3::MatmulTransRight, CBLAS3_GEMM_TRANS_RIGHT>(left, right, out, tiling);
}

extern "C" __global__ __aicore__ void csyrk_combine_kernel(
    GM_ADDR pr, GM_ADDR pi, GM_ADDR c, CsyrkCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum());
    if (coreNum == 0U) {
        return;
    }
    const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
    const bool isUpper = (tiling.uploMode == CBLAS3_UPLO_UPPER);

    TPipe pipe;
    CsyrkCombineBuffers bufs;
    CsyrkCombineCtx ctx;
    InitCombineCtx(pipe, bufs, ctx, pr, pi, c, tiling);

    // Columns of a triangle carry very unequal work, so they are dealt out in a
    // round robin: every core then gets a uniform sample of the run lengths.
    for (uint32_t col = coreIdx; col < tiling.n; col += coreNum) {
        const uint32_t lo = isUpper ? 0U : col;
        const uint32_t hi = isUpper ? col : (tiling.n - 1U);
        const uint32_t runLen = hi - lo + 1U;

        for (uint32_t off = 0; off < runLen; off += COMBINE_CHUNK) {
            const uint32_t len = (off + COMBINE_CHUNK > runLen) ? (runLen - off) : COMBINE_CHUNK;
            CombineOneChunk(ctx, tiling, col, lo + off, len);
        }
    }
}

// ===========================================================================
//  Launchers
// ===========================================================================
void csyrk_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR a, GM_ADDR p, GM_ADDR q, GM_ADDR r, CBlas3SplitConcatTilingData tiling)
{
    csyrk_split_kernel<<<blockDim, nullptr, stream>>>(a, p, q, r, tiling);
}

void csyrk_gemm_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling)
{
    if (tiling.transMode == CBLAS3_GEMM_TRANS_LEFT) {
        csyrk_gemm_kernel_tl<<<blockDim, nullptr, stream>>>(left, right, out, tiling);
    } else {
        csyrk_gemm_kernel_tr<<<blockDim, nullptr, stream>>>(left, right, out, tiling);
    }
}

void csyrk_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR pr, GM_ADDR pi, GM_ADDR c, CsyrkCombineTilingData tiling)
{
    csyrk_combine_kernel<<<blockDim, nullptr, stream>>>(pr, pi, c, tiling);
}

uint32_t csyrk_gemm_single_k() { return cblas3::SINGLE_K; }
