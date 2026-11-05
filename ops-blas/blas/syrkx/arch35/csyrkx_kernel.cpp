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
 * \file csyrkx_kernel.cpp
 * \brief CSYRKX Phase 0 and Phase 2 AIV kernels (arch35).
 *
 *        Phase 0 (Deinterleave): splits a complex matrix into real/imag float
 *          matrices. Called twice by the host: A -> Ar, Ai and B -> Br, Bi.
 *        Phase 2 (Combine): merges the 4 real GEMM results into complex C.
 *          trans=N : t1=Ar*Br^T, t2=Ai*Bi^T, t3=Ar*Bi^T, t4=Ai*Br^T
 *          trans=T/C: t1=Ar^T*Br, t2=Ai^T*Bi, t3=Ar^T*Bi, t4=Ai^T*Br
 *                    (OP_C maps to OP_T: syrkx is symmetric, not Hermitian,
 *                     so no conjugation is applied)
 *          P_r = t1 - t2, P_i = t3 + t4
 *          Cr = alpha_r*P_r - alpha_i*P_i, Ci = alpha_r*P_i + alpha_i*P_r
 *          C  = (Cr + i*Ci) + beta*C_old, uplo triangle only
 *        Uses Te::Copy for GM<->UB, Te::Transform for vector compute,
 *        DeInterleave/Interleave for complex pack/unpack.
 *        LOWER diagonal blocks: build the full column image in UB (computed
 *        uplo part + C_old restored strictly-upper segment) and store once.
 */

#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#include "csyrkx_tiling_data.h"
#include "csyrkx_kernel.h"
#include "tensor_api/tensor.h"

using namespace AscendC;
namespace te = AscendC::Te;

constexpr uint32_t SCALE_BLOCK = CSYRKX_ARCH35_SCALE_BLOCK;
constexpr uint32_t SCALE_UB_FLOATS = SCALE_BLOCK * SCALE_BLOCK;
constexpr uint32_t SCALE_UB_CPLX_FLOATS = SCALE_BLOCK * SCALE_BLOCK * 2;

template <typename T>
__aicore__ inline auto MakeUbTensor1D(const LocalTensor<T>& lt, uint32_t offset, uint32_t count)
{
    return te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, T>(lt.GetPhyAddr() + static_cast<uint64_t>(offset) * sizeof(T)),
        te::MakeFrameLayout<te::NDLayoutPtn>(static_cast<uint64_t>(1), static_cast<uint64_t>(count)));
}

// ==========================================================================
//  Phase 0: Deinterleave complex matrix -> real, imag (AIV-only)
//  Reads a column-major complex matrix from GM and writes tightly packed real
//  and imag float matrices (row stride = rows). Multi-core split by rows;
//  inner 2D tiles of SCALE_BLOCK x SCALE_BLOCK.
// ==========================================================================

struct DeintCtx {
    GlobalTensor<float> aGM, arGM, aiGM;
    TBuf<TPosition::VECIN> aInBuf, arOutBuf, aiOutBuf;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t lda = 0;
    uint32_t rowStart = 0;
    uint32_t rowEnd = 0;
};

// GM->UB: load complex block via te::Copy (2D strided NDExtLayoutPtn)
__aicore__ inline void LoadComplexABlock(
    DeintCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t blockRows, uint32_t blockCols, uint32_t ubCplxStride)
{
    auto copyGM2UB = te::MakeCopy(te::CopyGM2UB{});
    auto aGm = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.aGM.GetPhyAddr())),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(ctx.cols), static_cast<uint64_t>(ctx.lda * 2)));
    auto aBlock = aGm.Slice(
        te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase * 2)),
        te::MakeShape(static_cast<uint64_t>(blockCols), static_cast<uint64_t>(blockRows * 2)));
    LocalTensor<float> aInUb = ctx.aInBuf.Get<float>();
    auto aUb = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(aInUb.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(blockCols), static_cast<uint64_t>(ubCplxStride)));
    te::Copy(copyGM2UB, aUb, aBlock);
}

// V: DeInterleave complex [re0,im0,re1,im1,...] -> [re0,re1,...] + [im0,im1,...]
// No trailing barrier: the caller issues the V->MTE3 handshake right after
// this returns, which covers the arUb/aiUb read-after-write dependency.
__aicore__ inline void DeinterleaveBlock(DeintCtx& ctx, uint32_t cplxCnt)
{
    LocalTensor<float> aInUb = ctx.aInBuf.Get<float>();
    LocalTensor<float> arUb = ctx.arOutBuf.Get<float>();
    LocalTensor<float> aiUb = ctx.aiOutBuf.Get<float>();
    DeInterleave(arUb, aiUb, aInUb, static_cast<int32_t>(cplxCnt));
}

// UB->GM: store real and imag blocks via te::Copy (2D strided NDExtLayoutPtn)
__aicore__ inline void StoreRealBlocks(
    DeintCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t blockRows, uint32_t blockCols, uint32_t ubTempStride)
{
    auto copyUB2GM = te::MakeCopy(te::CopyUB2GM{});
    LocalTensor<float> arUb = ctx.arOutBuf.Get<float>();
    LocalTensor<float> aiUb = ctx.aiOutBuf.Get<float>();

    auto gmLayout = te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(ctx.cols), static_cast<uint64_t>(ctx.rows));
    auto ubLayout =
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(blockCols), static_cast<uint64_t>(ubTempStride));
    auto coord = te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase));
    auto shape = te::MakeShape(static_cast<uint64_t>(blockCols), static_cast<uint64_t>(blockRows));

    auto arGm = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.arGM.GetPhyAddr())), gmLayout);
    auto arUbT = te::MakeTensor(te::MakeMemPtr<te::Location::UB, float>(arUb.GetPhyAddr()), ubLayout);
    te::Copy(copyUB2GM, arGm.Slice(coord, shape), arUbT);

    auto aiGm = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.aiGM.GetPhyAddr())), gmLayout);
    auto aiUbT = te::MakeTensor(te::MakeMemPtr<te::Location::UB, float>(aiUb.GetPhyAddr()), ubLayout);
    te::Copy(copyUB2GM, aiGm.Slice(coord, shape), aiUbT);
}

// Process one 2D tile: load -> deinterleave -> store
__aicore__ inline void ProcessDeinterleaveTile(
    DeintCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t blockRows, uint32_t blockCols)
{
    uint32_t ubTempStride = RoundUp<uint32_t>(blockRows, CSYRKX_ARCH35_ELEMENTS_PER_BLOCK);
    uint32_t ubCplxStride = 2 * ubTempStride;
    uint32_t cplxCnt = ubCplxStride * blockCols;

    // aInBuf is shared by every tile of this core: the previous tile's V-stage
    // DeinterleaveBlock may still read it when this tile's MTE2 load is issued.
    // MTE2 and V are independent pipes, so handshake before overwriting.
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);

    LoadComplexABlock(ctx, iBase, jBase, blockRows, blockCols, ubCplxStride);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    DeinterleaveBlock(ctx, cplxCnt);

    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    StoreRealBlocks(ctx, iBase, jBase, blockRows, blockCols, ubTempStride);

    // WAR hazard on arOutBuf/aiOutBuf: StoreRealBlocks issues MTE3 reads while
    // the NEXT tile's DeinterleaveBlock (V pipe) writes the same buffers.
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
}

// Loop over 2D tiles within the core's row range
__aicore__ inline void ProcessDeinterleaveLoop(DeintCtx& ctx)
{
    for (uint32_t jBase = 0; jBase < ctx.cols; jBase += SCALE_BLOCK) {
        uint32_t blockCols = Min<uint32_t>(SCALE_BLOCK, ctx.cols - jBase);
        for (uint32_t iBase = ctx.rowStart; iBase < ctx.rowEnd; iBase += SCALE_BLOCK) {
            uint32_t blockRows = Min<uint32_t>(SCALE_BLOCK, ctx.rowEnd - iBase);
            ProcessDeinterleaveTile(ctx, iBase, jBase, blockRows, blockCols);
        }
    }
}

extern "C" __global__ __aicore__ void csyrkx_deinterleave_kernel(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, const CsyrkxDeinterleaveTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    DeintCtx ctx;
    ctx.aGM.SetGlobalBuffer((__gm__ float*)a);
    ctx.arGM.SetGlobalBuffer((__gm__ float*)ar);
    ctx.aiGM.SetGlobalBuffer((__gm__ float*)ai);
    ctx.rows = tiling.rows;
    ctx.cols = tiling.cols;
    ctx.lda = tiling.lda;
    ctx.rowStart = GetBlockIdx() * tiling.rowsPerCore;
    ctx.rowEnd = Min<uint32_t>(ctx.rowStart + tiling.rowsPerCore, tiling.rows);
    pipe.InitBuffer(ctx.aInBuf, SCALE_UB_CPLX_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.arOutBuf, SCALE_UB_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.aiOutBuf, SCALE_UB_FLOATS * sizeof(float));
    ProcessDeinterleaveLoop(ctx);
}

void csyrkx_deinterleave_kernel_do(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, const CsyrkxDeinterleaveTilingData& tiling, uint32_t numBlocks, void* stream)
{
    csyrkx_deinterleave_kernel<<<numBlocks, nullptr, stream>>>(a, ar, ai, tiling);
}

// ==========================================================================
//  Phase 2: Combine/Scale (AIV-only)
//  t1..t4 are the four real GEMM results, all with the same n x tempLdc layout.
//  Only the uplo triangle of C is written; the other triangle is preserved.
// ==========================================================================

struct CsyrkxCtx {
    CsyrkxCombineTilingData tiling;
    GlobalTensor<float> t1GM, t2GM, t3GM, t4GM, cGM;
    TBuf<TPosition::VECIN> t1Buf, t2Buf, t3Buf, t4Buf, cInBuf;
    TBuf<TPosition::VECOUT> cOutBuf;
    // Scratch for the beta-imaginary cross terms. Kept separate from cInBuf:
    // the interleaved C_old copy in cInBuf must stay intact for the LOWER
    // diagonal non-uplo restore in InterleaveAndStore.
    TBuf<TPosition::VECCALC> scratchBuf;
    uint32_t rowStart = 0;
    uint32_t rowEnd = 0;
};

static __aicore__ inline uint32_t UploCount(bool uploUpper, uint32_t iBase, uint32_t absJ, uint32_t rows)
{
    if (uploUpper) {
        return (absJ >= iBase) ? Min(absJ - iBase + 1, rows) : 0;
    }
    return (absJ < iBase) ? rows : (rows - (absJ - iBase));
}

// GM->UB: load t1..t4 (real GEMM results) via Te::Copy (2D strided NDExtLayoutPtn)
__aicore__ inline void LoadGemmResults(CsyrkxCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    uint32_t tempLdc = ctx.tiling.tempLdc;
    uint32_t ubStride = RoundUp<uint32_t>(rows, CSYRKX_ARCH35_ELEMENTS_PER_BLOCK);

    LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
    LocalTensor<float> t2Ub = ctx.t2Buf.Get<float>();
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
    LocalTensor<float> t4Ub = ctx.t4Buf.Get<float>();

    auto copyGM2UB = te::MakeCopy(te::CopyGM2UB{});
    auto gmLayout =
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(ctx.tiling.n), static_cast<uint64_t>(tempLdc));
    auto ubLayout = te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(cols), static_cast<uint64_t>(ubStride));
    auto coord = te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase));
    auto shape = te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows));

    auto t1Gm = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.t1GM.GetPhyAddr())), gmLayout);
    auto t2Gm = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.t2GM.GetPhyAddr())), gmLayout);
    auto t3Gm = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.t3GM.GetPhyAddr())), gmLayout);
    auto t4Gm = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.t4GM.GetPhyAddr())), gmLayout);

    te::Copy(
        copyGM2UB, te::MakeTensor(te::MakeMemPtr<te::Location::UB, float>(t1Ub.GetPhyAddr()), ubLayout),
        t1Gm.Slice(coord, shape));
    te::Copy(
        copyGM2UB, te::MakeTensor(te::MakeMemPtr<te::Location::UB, float>(t2Ub.GetPhyAddr()), ubLayout),
        t2Gm.Slice(coord, shape));
    te::Copy(
        copyGM2UB, te::MakeTensor(te::MakeMemPtr<te::Location::UB, float>(t3Ub.GetPhyAddr()), ubLayout),
        t3Gm.Slice(coord, shape));
    te::Copy(
        copyGM2UB, te::MakeTensor(te::MakeMemPtr<te::Location::UB, float>(t4Ub.GetPhyAddr()), ubLayout),
        t4Gm.Slice(coord, shape));
}

// GM->UB: load C_old (interleaved complex) via Te::Copy (2D strided).
// ubCplxStride = 2 * ubTempStride keeps DeInterleave's output aligned with t1/t3.
__aicore__ inline void LoadCOld(CsyrkxCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    uint32_t ldc = ctx.tiling.ldc;
    uint32_t ubTempStride = RoundUp<uint32_t>(rows, CSYRKX_ARCH35_ELEMENTS_PER_BLOCK);
    uint32_t ubCplxStride = 2 * ubTempStride;
    LocalTensor<float> cInUb = ctx.cInBuf.Get<float>();

    auto copyGM2UB = te::MakeCopy(te::CopyGM2UB{});
    auto gmT = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(const_cast<__gm__ float*>(ctx.cGM.GetPhyAddr())),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(
            static_cast<uint64_t>(ctx.tiling.n), static_cast<uint64_t>(ldc * 2)));
    auto gmBlock = gmT.Slice(
        te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase * 2)),
        te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows * 2)));
    auto ubT = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(cInUb.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(cols), static_cast<uint64_t>(ubCplxStride)));
    te::Copy(copyGM2UB, ubT, gmBlock);
}

// V: compute Cr/Ci from the 4 real GEMM results via Te::Transform.
//   P_r = t1 - t2, P_i = t3 + t4   (4M decomposition of op(A)*op(B)^T)
// Real alpha (alphaImag == 0.0f — uniform tiling-scalar branch):
//   Cr = αr*P_r, Ci = αr*P_i (bit-identical to the real-alpha legacy sequence).
// Complex alpha (alphaImag != 0.0f):
//   Cr = αr*P_r - αi*P_i, Ci = αr*P_i + αi*P_r
// t2/t4 are fully consumed by P_r/P_i and are reused as scratch for the αi terms.
__aicore__ inline void ComputeCrCi(CsyrkxCtx& ctx, uint32_t rows, uint32_t cols, float alphaReal, float alphaImag)
{
    uint32_t ubTempStride = RoundUp<uint32_t>(rows, CSYRKX_ARCH35_ELEMENTS_PER_BLOCK);
    uint32_t cnt = ubTempStride * cols;

    LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
    LocalTensor<float> t2Ub = ctx.t2Buf.Get<float>();
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
    LocalTensor<float> t4Ub = ctx.t4Buf.Get<float>();

    auto t1T = MakeUbTensor1D(t1Ub, 0U, cnt);
    auto t2T = MakeUbTensor1D(t2Ub, 0U, cnt);
    auto t3T = MakeUbTensor1D(t3Ub, 0U, cnt);
    auto t4T = MakeUbTensor1D(t4Ub, 0U, cnt);

    Transform<te::Inst::Sub>(t1T, t1T, t2T);   // t1 = P_r = t1-t2
    Transform<te::Inst::Add>(t3T, t3T, t4T);   // t3 = P_i = t3+t4
    if (alphaImag == 0.0f) {
        Transform<te::Inst::MulScalar>(t1T, t1T, alphaReal);
        Transform<te::Inst::MulScalar>(t3T, t3T, alphaReal);
    } else {
        // Complex alpha cross-scaling; t2/t4 take over the scratch role.
        Transform<te::Inst::MulScalar>(t2T, t3T, alphaImag); // t2 = αi*P_i
        Transform<te::Inst::MulScalar>(t4T, t1T, alphaImag); // t4 = αi*P_r
        Transform<te::Inst::MulScalar>(t1T, t1T, alphaReal); // t1 = αr*P_r
        Transform<te::Inst::MulScalar>(t3T, t3T, alphaReal); // t3 = αr*P_i
        Transform<te::Inst::Sub>(t1T, t1T, t2T);             // Cr = αr*P_r - αi*P_i
        Transform<te::Inst::Add>(t3T, t3T, t4T);             // Ci = αr*P_i + αi*P_r
    }
    PipeBarrier<PIPE_ALL>();
}

// V: DeInterleave C_old -> t2(real), t4(imag); then add beta * C_old into Cr/Ci.
// beta is complex: (Cr,Ci) += (βr*or - βi*oi, βr*oi + βi*or).
// When skipTemp (alpha == 0 or k == 0), Cr/Ci start from 0, so the same
// expression reduces to C = beta*C_old.
__aicore__ inline void ApplyBeta(
    CsyrkxCtx& ctx, uint32_t rows, uint32_t cols, float betaReal, float betaImag, bool skipTemp)
{
    uint32_t ubTempStride = RoundUp<uint32_t>(rows, CSYRKX_ARCH35_ELEMENTS_PER_BLOCK);
    uint32_t ubCplxStride = 2 * ubTempStride;
    uint32_t cnt = ubTempStride * cols;
    uint32_t srcCnt = ubCplxStride * cols;

    LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
    LocalTensor<float> t2Ub = ctx.t2Buf.Get<float>();
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
    LocalTensor<float> t4Ub = ctx.t4Buf.Get<float>();
    LocalTensor<float> cInUb = ctx.cInBuf.Get<float>();

    // DeInterleave the whole block: [re0,im0,...] -> t2 real, t4 imag.
    DeInterleave(t2Ub, t4Ub, cInUb, static_cast<int32_t>(srcCnt));
    PipeBarrier<PIPE_V>();

    auto t1T = MakeUbTensor1D(t1Ub, 0U, cnt);
    auto t2T = MakeUbTensor1D(t2Ub, 0U, cnt);
    auto t3T = MakeUbTensor1D(t3Ub, 0U, cnt);
    auto t4T = MakeUbTensor1D(t4Ub, 0U, cnt);

    if (skipTemp) {
        Transform<te::Inst::MulScalar>(t1T, t2T, betaReal);
        Transform<te::Inst::MulScalar>(t3T, t4T, betaReal);
    } else {
        Transform<te::Inst::Axpy>(t1T, t2T, betaReal);
        Transform<te::Inst::Axpy>(t3T, t4T, betaReal);
    }

    if (betaImag != 0.0f) {
        // Dedicated scratch (cInBuf must keep the interleaved C_old image used
        // by the LOWER diagonal non-uplo restore).
        LocalTensor<float> scratchUb = ctx.scratchBuf.Get<float>();
        auto scratchT = MakeUbTensor1D(scratchUb, 0U, cnt);
        Transform<te::Inst::MulScalar>(scratchT, t4T, betaImag); // s = βi*old_i
        Transform<te::Inst::Sub>(t1T, t1T, scratchT);            // Cr -= βi*old_i
        Transform<te::Inst::MulScalar>(scratchT, t2T, betaImag); // s = βi*old_r
        Transform<te::Inst::Add>(t3T, t3T, scratchT);            // Ci += βi*old_r
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename CopyOp>
__aicore__ inline void StoreUploColumn(
    LocalTensor<float>& cOutDst0, CopyOp& copyUB2GM, __gm__ float* cGmBase, uint64_t gmOff, uint64_t ubOff,
    uint32_t uploCnt)
{
    if (uploCnt == 0) {
        return;
    }
    uint32_t copyCount = uploCnt * 2;
    auto gmCol = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(cGmBase + gmOff),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(1), static_cast<uint64_t>(copyCount)));
    auto ubCol = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(cOutDst0.GetPhyAddr() + ubOff * sizeof(float)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(1), static_cast<uint64_t>(copyCount)));
    te::Copy(copyUB2GM, gmCol, ubCol);
}

// LOWER diagonal column, step 1: copy the non-uplo (strictly-upper) segment of
// C_old from cInUb into cOutDst0 (UB-side) so the final column image is built
// entirely in UB and the store below is single-shot. Storing the whole column
// and then re-storing the non-uplo segment would be two MTE3 bursts targeting
// the same GM range with no guaranteed completion order.
__aicore__ inline void RestoreLowerDiagSegment(
    LocalTensor<float>& cOutUb, LocalTensor<float>& cInUb, uint64_t ubOff, uint32_t nonUploCnt)
{
    if (nonUploCnt == 0) {
        return;
    }
    const uint32_t restoreCount = nonUploCnt * 2;
    Copy(cOutUb[static_cast<uint32_t>(ubOff)], cInUb[static_cast<uint32_t>(ubOff)], restoreCount);
}

// LOWER diagonal column, step 2: emit the single full-column MTE3 store. The
// caller must have run RestoreLowerDiagSegment and crossed a barrier.
template <typename CopyOp>
__aicore__ inline void StoreLowerDiagColumn(
    LocalTensor<float>& cOutUb, CopyOp& copyUB2GM, __gm__ float* cGmBase, uint64_t gmOff, uint64_t ubOff, uint32_t rows)
{
    const uint32_t fullCount = rows * 2;
    auto gmFull = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(cGmBase + gmOff),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(1), static_cast<uint64_t>(fullCount)));
    auto ubFull = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(cOutUb.GetPhyAddr() + ubOff * sizeof(float)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(1), static_cast<uint64_t>(fullCount)));
    te::Copy(copyUB2GM, gmFull, ubFull);
}

// UB->GM: store a non-diagonal block (every column fully inside the uplo
// triangle) with a single 2D te::Copy.
template <typename CopyOp>
__aicore__ inline void StoreNonDiagBlock(
    LocalTensor<float>& cOutDst0, CopyOp& copyUB2GM, __gm__ float* cGmBase, uint32_t n, uint32_t ldc, uint32_t iBase,
    uint32_t jBase, uint32_t rows, uint32_t cols, uint32_t ubCplxStride)
{
    auto gmT = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(cGmBase),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(n), static_cast<uint64_t>(ldc * 2)));
    auto gmBlock = gmT.Slice(
        te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase * 2)),
        te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows * 2)));
    auto ubT = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(cOutDst0.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(cols), static_cast<uint64_t>(ubCplxStride)));
    te::Copy(copyUB2GM, gmBlock, ubT);
}

// Diagonal-block store (uploCnt varies per column). Two passes: all LOWER
// restores first (disjoint UB spans), one barrier, then all stores.
template <typename CopyOp>
__aicore__ inline void StoreDiagonalBlock(
    LocalTensor<float>& cOutDst0, LocalTensor<float>& cInUb, CopyOp& copyUB2GM, __gm__ float* cGmBase, bool uploUpper,
    uint32_t ldc, uint32_t ubCplxStride, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    for (uint32_t c = 0; c < cols; c++) {
        uint32_t absJ = jBase + c;
        uint64_t ubOff = static_cast<uint64_t>(c) * ubCplxStride;
        bool isDiagonal = (absJ >= iBase && absJ < iBase + rows);
        if (isDiagonal && !uploUpper) {
            RestoreLowerDiagSegment(cOutDst0, cInUb, ubOff, absJ - iBase);
        }
    }
    // The pass-1 restores (V-pipe Copy writes into cOutUb) must be observed by
    // the pass-2 MTE3 stores reading the same buffer.
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    for (uint32_t c = 0; c < cols; c++) {
        uint32_t absJ = jBase + c;
        uint64_t gmOff = static_cast<uint64_t>(absJ) * ldc * 2 + iBase * 2;
        uint64_t ubOff = static_cast<uint64_t>(c) * ubCplxStride;
        bool isDiagonal = (absJ >= iBase && absJ < iBase + rows);

        if (!isDiagonal || uploUpper) {
            uint32_t uploCnt = UploCount(uploUpper, iBase, absJ, rows);
            StoreUploColumn(cOutDst0, copyUB2GM, cGmBase, gmOff, ubOff, uploCnt);
        } else {
            StoreLowerDiagColumn(cOutDst0, copyUB2GM, cGmBase, gmOff, ubOff, rows);
        }
    }
}

// V->MTE3: Interleave Cr,Ci -> cOutBuf; store the uplo triangle to GM.
__aicore__ inline void InterleaveAndStore(CsyrkxCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    bool uploUpper = (ctx.tiling.uploMode == ACLBLAS_UPPER);
    uint32_t ldc = ctx.tiling.ldc;
    uint32_t ubTempStride = RoundUp<uint32_t>(rows, CSYRKX_ARCH35_ELEMENTS_PER_BLOCK);
    uint32_t ubCplxStride = 2 * ubTempStride;
    uint32_t cnt = ubTempStride * cols;

    LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
    LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
    LocalTensor<float> cInUb = ctx.cInBuf.Get<float>();
    LocalTensor<float> cOutDst0 = ctx.cOutBuf.Get<float>();
    LocalTensor<float> cOutDst1 = ctx.cOutBuf.GetWithOffset<float>(cnt, cnt * sizeof(float));

    Interleave(cOutDst0, cOutDst1, t1Ub, t3Ub, static_cast<int32_t>(cnt));
    // The V->MTE3 handshake below covers cOutBuf's read-after-write dependency.
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    auto copyUB2GM = te::MakeCopy(te::CopyUB2GM{});
    auto cGmBase = const_cast<__gm__ float*>(ctx.cGM.GetPhyAddr());

    // Non-diagonal block: every column is fully in the uplo triangle.
    bool fullyUplo = uploUpper ? (jBase >= iBase + rows) : (jBase + cols <= iBase);

    if (fullyUplo) {
        StoreNonDiagBlock(cOutDst0, copyUB2GM, cGmBase, ctx.tiling.n, ldc, iBase, jBase, rows, cols, ubCplxStride);
    } else {
        StoreDiagonalBlock(
            cOutDst0, cInUb, copyUB2GM, cGmBase, uploUpper, ldc, ubCplxStride, iBase, jBase, rows, cols);
    }

    // Cross-block sync: this block's MTE3 stores must finish before the next
    // ProcessBlock reuses the same UB buffers.
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(0);
}

__aicore__ inline void ProcessBlock(CsyrkxCtx& ctx, uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    float alphaReal = ctx.tiling.alphaReal;
    float alphaImag = ctx.tiling.alphaImag;
    float betaReal = ctx.tiling.betaReal;
    float betaImag = ctx.tiling.betaImag;
    bool skipTemp = (ctx.tiling.isAlphaZero || ctx.tiling.isKZero);
    bool isBetaZero = ctx.tiling.isBetaZero;

    // alpha==0 or k==0 with beta==0: C = 0 on the uplo triangle, non-uplo
    // restored from C_old. GEMM results are unused; zero t1/t3 so
    // InterleaveAndStore writes (0,0). C_old is still needed for the
    // LOWER diagonal block restore.
    if (skipTemp && isBetaZero) {
        uint32_t ubTempStride = RoundUp<uint32_t>(rows, CSYRKX_ARCH35_ELEMENTS_PER_BLOCK);
        uint32_t cnt = ubTempStride * cols;
        LocalTensor<float> t1Ub = ctx.t1Buf.Get<float>();
        LocalTensor<float> t3Ub = ctx.t3Buf.Get<float>();
        Duplicate(t1Ub, 0.0f, cnt);
        Duplicate(t3Ub, 0.0f, cnt);
        LoadCOld(ctx, iBase, jBase, rows, cols);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        InterleaveAndStore(ctx, iBase, jBase, rows, cols);
        return;
    }

    if (!skipTemp) {
        LoadGemmResults(ctx, iBase, jBase, rows, cols);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        ComputeCrCi(ctx, rows, cols, alphaReal, alphaImag);
    }

    // Always load C_old: for beta scaling when !isBetaZero, and for the
    // non-uplo restore of LOWER diagonal blocks when isBetaZero.
    LoadCOld(ctx, iBase, jBase, rows, cols);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    if (!isBetaZero) {
        ApplyBeta(ctx, rows, cols, betaReal, betaImag, skipTemp);
    }

    InterleaveAndStore(ctx, iBase, jBase, rows, cols);
}

__aicore__ inline void ProcessLoop(CsyrkxCtx& ctx)
{
    bool uploUpper = (ctx.tiling.uploMode == ACLBLAS_UPPER);
    for (uint32_t iBase = ctx.rowStart; iBase < ctx.rowEnd; iBase += SCALE_BLOCK) {
        uint32_t rows = Min<uint32_t>(SCALE_BLOCK, ctx.rowEnd - iBase);
        uint32_t jStart = uploUpper ? iBase : 0;
        uint32_t jLimit = uploUpper ? ctx.tiling.n : Min<uint32_t>(iBase + rows, ctx.tiling.n);
        for (uint32_t jBase = jStart; jBase < jLimit; jBase += SCALE_BLOCK) {
            uint32_t cols = Min<uint32_t>(SCALE_BLOCK, jLimit - jBase);
            ProcessBlock(ctx, iBase, jBase, rows, cols);
        }
    }
}

extern "C" __global__ __aicore__ void csyrkx_combine_kernel(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c, const CsyrkxCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CsyrkxCtx ctx;
    ctx.tiling = tiling;
    ctx.t1GM.SetGlobalBuffer((__gm__ float*)t1);
    ctx.t2GM.SetGlobalBuffer((__gm__ float*)t2);
    ctx.t3GM.SetGlobalBuffer((__gm__ float*)t3);
    ctx.t4GM.SetGlobalBuffer((__gm__ float*)t4);
    ctx.cGM.SetGlobalBuffer((__gm__ float*)c);
    pipe.InitBuffer(ctx.t1Buf, SCALE_UB_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.t2Buf, SCALE_UB_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.t3Buf, SCALE_UB_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.t4Buf, SCALE_UB_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.cInBuf, SCALE_UB_CPLX_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.cOutBuf, SCALE_UB_CPLX_FLOATS * sizeof(float));
    pipe.InitBuffer(ctx.scratchBuf, SCALE_UB_FLOATS * sizeof(float));
    uint32_t blockIdx = GetBlockIdx();
    ctx.rowStart = blockIdx * tiling.rowsPerCore;
    ctx.rowEnd = Min<uint32_t>(ctx.rowStart + tiling.rowsPerCore, tiling.n);
    ProcessLoop(ctx);
}

void csyrkx_combine_kernel_do(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c, const CsyrkxCombineTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    csyrkx_combine_kernel<<<numBlocks, nullptr, stream>>>(t1, t2, t3, t4, c, tiling);
}
