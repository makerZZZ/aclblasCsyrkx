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
 * \brief CSYRK Phase 0 (deinterleave) and Phase 2 (combine) AIV kernels (arch35).
 *
 *   Phase 0: complex A -> Ar, Ai, AiNeg(= -Ai) real matrices (native DataCopyPad
 *            strided deinterleave).
 *   Phase 2: C = alpha*(Cr + i*Ci) + beta*C_old (complex alpha/beta), writes
 *            only the uplo/lower triangle (csyrk is symmetric, not Hermitian).
 *            alpha==0 / k==0 short-circuit to C = beta*C_old.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cann_ops_blas_common.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#include "common/helper/kernel_constant.h"
#include "csyrk_tiling_data.h"
#include "csyrk_kernel.h"

using namespace AscendC;

// ==========================================================================
//  Phase 0: Deinterleave complex A -> Ar, Ai, AiNeg (AIV-only)
//  A is column-major complex (nRows x nCols, lda complex elements). Each
//  column is contiguous: [re0, im0, re1, im1, ...]. Read the whole interleaved
//  column segment contiguously (Compact 8B blocks) into UB, then split real/
//  imag with the vector DeInterleave (VECTOR pipe, not MTE2). The old strided
//  4B-read-per-8B pattern made MTE2 read-bound (89% mte2 ratio).
// ==========================================================================

constexpr uint32_t DEINT_BLOCK = 4096;
constexpr uint32_t DEINT_MAX_ELEMS = 2040;

// Batched path: nRows 8-aligned and a 2*total-element 32B-aligned read, so one
// contiguous 8B-block read + DeInterleave covers maxBatch columns at a time.
__aicore__ inline void CsyrkDeintBatched(
    uint32_t colStart, uint32_t colEnd, uint32_t nRows, uint32_t outLdc, const GlobalTensor<float>& aGM,
    const GlobalTensor<float>& arGM, const GlobalTensor<float>& aiGM, const LocalTensor<float>& abUb,
    const LocalTensor<float>& arUb, const LocalTensor<float>& aiUb, const DataCopyPadExtParams<float>& pp)
{
    if (nRows == 0) {
        return;
    }
    uint32_t maxBatch = Min<uint32_t>(DEINT_BLOCK / nRows, DEINT_MAX_ELEMS / (2u * nRows));
    if (maxBatch == 0) {
        maxBatch = 1;
    }
    for (uint32_t j = colStart; j < colEnd; j += maxBatch) {
        uint32_t nb = Min<uint32_t>(maxBatch, colEnd - j);
        uint32_t total = nb * nRows;
        uint64_t srcBase = static_cast<uint64_t>(j) * nRows;
        DataCopyExtParams cp{static_cast<uint16_t>(total), 8, 0, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(abUb, aGM[2 * srcBase], cp, pp);
        PipeBarrier<PIPE_ALL>();
        DeInterleave(arUb, aiUb, abUb, static_cast<int32_t>(2 * total));
        PipeBarrier<PIPE_ALL>();
        for (uint32_t c = 0; c < nb; c++) {
            uint64_t outBase = static_cast<uint64_t>(j + c) * outLdc;
            DataCopyExtParams cpOut{static_cast<uint16_t>(nRows), 4, 0, 0, 0};
            DataCopyPad<float, PaddingMode::Compact>(arGM[outBase], arUb[c * nRows], cpOut);
            DataCopyPad<float, PaddingMode::Compact>(aiGM[outBase], aiUb[c * nRows], cpOut);
        }
        PipeBarrier<PIPE_ALL>();
    }
}

// Per-column software-pipelined path: double-buffered A read slots let the MTE2
// read of column j+1 overlap the vector deinterleave and MTE3 store of column j.
// ar/ai are single-buffered, ordered with explicit V->MTE3 / MTE3->V events.
// Event ids: 0/1 = MTE2_V per read slot, 2 = V_MTE3, 3 = MTE3_V.
__aicore__ inline void CsyrkDeintPipelined(
    uint32_t colStart, uint32_t colEnd, uint32_t nRows, uint32_t lda, uint32_t outLdc, const GlobalTensor<float>& aGM,
    const GlobalTensor<float>& arGM, const GlobalTensor<float>& aiGM, const LocalTensor<float>& ab0,
    const LocalTensor<float>& ab1, const LocalTensor<float>& arUb, const LocalTensor<float>& aiUb,
    const DataCopyPadExtParams<float>& pp)
{
    const uint32_t colCountP = colEnd - colStart;
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::MTE3_V>(3);
    WaitFlag<HardEvent::V_MTE2>(0);
    {
        uint64_t base = static_cast<uint64_t>(colStart) * lda;
        DataCopyExtParams cp{static_cast<uint16_t>(nRows), 8, 0, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(ab0, aGM[2 * base], cp, pp);
    }
    SetFlag<HardEvent::MTE2_V>(0);
    for (uint32_t idx = 0; idx < colCountP; idx++) {
        uint32_t slot = idx & 1u;
        uint32_t j = colStart + idx;
        LocalTensor<float> src = slot ? ab1 : ab0;
        WaitFlag<HardEvent::MTE2_V>(slot);
        if (idx + 1 < colCountP) {
            uint32_t nslot = slot ^ 1u;
            LocalTensor<float> nsrc = nslot ? ab1 : ab0;
            WaitFlag<HardEvent::V_MTE2>(nslot);
            uint64_t baseN = static_cast<uint64_t>(j + 1) * lda;
            DataCopyExtParams cp{static_cast<uint16_t>(nRows), 8, 0, 0, 0};
            DataCopyPad<float, PaddingMode::Compact>(nsrc, aGM[2 * baseN], cp, pp);
            SetFlag<HardEvent::MTE2_V>(nslot);
        }
        WaitFlag<HardEvent::MTE3_V>(3);   // ar/ai free (prev store done)
        DeInterleave(arUb, aiUb, src, static_cast<int32_t>(2 * nRows));
        SetFlag<HardEvent::V_MTE2>(slot); // read slot consumed
        SetFlag<HardEvent::V_MTE3>(2);    // ar/ai ready for the store
        WaitFlag<HardEvent::V_MTE3>(2);
        uint64_t outBase = static_cast<uint64_t>(j) * outLdc;
        DataCopyExtParams cpOut{static_cast<uint16_t>(nRows), 4, 0, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(arGM[outBase], arUb, cpOut);
        DataCopyPad<float, PaddingMode::Compact>(aiGM[outBase], aiUb, cpOut);
        SetFlag<HardEvent::MTE3_V>(3); // this column's store done
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::MTE3_V>(3);
}

// Generic per-column path (strided lda or nRows > DEINT_BLOCK).
__aicore__ inline void CsyrkDeintGeneric(
    uint32_t colStart, uint32_t colEnd, uint32_t nRows, uint32_t lda, uint32_t outLdc, const GlobalTensor<float>& aGM,
    const GlobalTensor<float>& arGM, const GlobalTensor<float>& aiGM, const LocalTensor<float>& abUb,
    const LocalTensor<float>& arUb, const LocalTensor<float>& aiUb, const DataCopyPadExtParams<float>& pp)
{
    for (uint32_t j = colStart; j < colEnd; j++) {
        for (uint32_t r0 = 0; r0 < nRows; r0 += DEINT_BLOCK) {
            uint32_t cnt = Min<uint32_t>(DEINT_BLOCK, nRows - r0);
            uint64_t base = static_cast<uint64_t>(j) * lda + r0;
            DataCopyExtParams cp{static_cast<uint16_t>(cnt), 8, 0, 0, 0};
            DataCopyPad<float, PaddingMode::Compact>(abUb, aGM[2 * base], cp, pp);
            PipeBarrier<PIPE_ALL>();
            DeInterleave(arUb, aiUb, abUb, static_cast<int32_t>(2 * cnt));
            PipeBarrier<PIPE_ALL>();
            uint64_t outBase = static_cast<uint64_t>(j) * outLdc + r0;
            DataCopyExtParams cpOut{static_cast<uint16_t>(cnt), 4, 0, 0, 0};
            DataCopyPad<float, PaddingMode::Compact>(arGM[outBase], arUb, cpOut);
            DataCopyPad<float, PaddingMode::Compact>(aiGM[outBase], aiUb, cpOut);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

extern "C" __global__ __aicore__ void csyrk_deinterleave_kernel(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, const CsyrkDeinterleaveTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECIN> abInBuf, arInBuf, aiInBuf;
    pipe.InitBuffer(abInBuf, 2 * DEINT_BLOCK * sizeof(float));
    pipe.InitBuffer(arInBuf, DEINT_BLOCK * sizeof(float));
    pipe.InitBuffer(aiInBuf, DEINT_BLOCK * sizeof(float));

    GlobalTensor<float> aGM, arGM, aiGM;
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    arGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ar));
    aiGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ai));

    const uint32_t nRows = tiling.nRows;
    const uint32_t lda = tiling.lda;
    const uint32_t outLdc = tiling.outLdc;
    const uint32_t colStart = GetBlockIdx() * tiling.colsPerCore;
    const uint32_t colEnd = Min<uint32_t>(colStart + tiling.colsPerCore, tiling.nCols);
    const LocalTensor<float> abUb = abInBuf.Get<float>();
    const LocalTensor<float> arUb = arInBuf.Get<float>();
    const LocalTensor<float> aiUb = aiInBuf.Get<float>();
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};

    if (lda == nRows && nRows > 0 && nRows <= DEINT_BLOCK && (nRows & 0x7u) == 0) {
        CsyrkDeintBatched(colStart, colEnd, nRows, outLdc, aGM, arGM, aiGM, abUb, arUb, aiUb, pp);
        return;
    }
    if (nRows > 0 && nRows <= DEINT_BLOCK && colEnd > colStart) {
        LocalTensor<float> ab0 = abInBuf.Get<float>();
        LocalTensor<float> ab1 = abInBuf.Get<float>()[DEINT_BLOCK];
        CsyrkDeintPipelined(colStart, colEnd, nRows, lda, outLdc, aGM, arGM, aiGM, ab0, ab1, arUb, aiUb, pp);
        return;
    }
    CsyrkDeintGeneric(colStart, colEnd, nRows, lda, outLdc, aGM, arGM, aiGM, abUb, arUb, aiUb, pp);
}

void csyrk_deinterleave_kernel_do(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, const CsyrkDeinterleaveTilingData& tiling, uint32_t numBlocks, void* stream)
{
    csyrk_deinterleave_kernel<<<numBlocks, nullptr, stream>>>(a, ar, ai, tiling);
}

// ==========================================================================
//  HF32 residual kernel (SIMT): reads deinterleaved fp32 Ar/Ai and writes the
//  HF32 low residual xl = x - HF32(x) for every element. HF32 keeps ten stored
//  mantissa bits (nearest-even), so HF32(x) = round10(x); xl is exact in fp32.
//  The cube's HF32x3 path forms xh*yh + xl*yh + xh*yl (three HF32 mmads), where
//  the hardware rounds ar/ai to xh and arLow/aiLow hold xl.
// ==========================================================================
__simt_callee__ __aicore__ inline float CsyrkHf32Residual(float value)
{
    union {
        float value;
        uint32_t bits;
    } part{value};
    if ((part.bits & 0x7f800000U) == 0x7f800000U) {
        return 0.0f; // inf/nan: residual undefined, keep 0
    }
    part.bits = (part.bits + 0xfffU + ((part.bits >> 13) & 1U)) & 0xffffe000U;
    return value - part.value;
}

// Threads stride the rows of this block's column span; contiguous threads hit
// consecutive rows of one column (adjacent memory), so GM access coalesces.
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void csyrk_hf32resid_vf(
    __gm__ float* srcRe, __gm__ float* srcIm, __gm__ float* dstRe, __gm__ float* dstIm, uint32_t rowLo, uint32_t rows,
    uint32_t colLo, uint32_t cols, uint32_t ldc)
{
    const uint64_t total = static_cast<uint64_t>(cols) * rows;
    for (uint64_t t = threadIdx.x; t < total; t += blockDim.x) {
        const uint32_t j = colLo + static_cast<uint32_t>(t / rows);
        const uint32_t i = rowLo + static_cast<uint32_t>(t % rows);
        const uint64_t off = static_cast<uint64_t>(j) * ldc + i;
        dstRe[off] = CsyrkHf32Residual(srcRe[off]);
        dstIm[off] = CsyrkHf32Residual(srcIm[off]);
    }
}

extern "C" __global__ __aicore__ void csyrk_hf32resid_kernel(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR arLow, GM_ADDR aiLow, const CsyrkDeinterleaveTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t nRows = tiling.nRows;
    const uint32_t nCols = tiling.nCols;
    if (nRows == 0 || nCols == 0) {
        return;
    }
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    // A (and Ar/Ai) are column-major: contiguous threads must walk the rows of
    // one column (adjacent memory), and blocks split the columns so each block
    // reads a contiguous column span. Deinterleave uses the same split, so the
    // residual pass sees the same column band layout and coalesces.
    const uint32_t colsPer = CeilDiv<uint32_t>(nCols, blockNum);
    const uint32_t colLo = Min<uint32_t>(blockIdx * colsPer, nCols);
    const uint32_t colHi = Min<uint32_t>(colLo + colsPer, nCols);
    if (colLo >= colHi) {
        return;
    }
    asc_vf_call<csyrk_hf32resid_vf>(
        dim3{SIMT_MAX_THREAD_NUM, 1, 1}, reinterpret_cast<__gm__ float*>(ar), reinterpret_cast<__gm__ float*>(ai),
        reinterpret_cast<__gm__ float*>(arLow), reinterpret_cast<__gm__ float*>(aiLow), 0, nRows, colLo, colHi - colLo,
        tiling.outLdc);
}

void csyrk_hf32resid_kernel_do(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR arLow, GM_ADDR aiLow, const CsyrkDeinterleaveTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    csyrk_hf32resid_kernel<<<numBlocks, nullptr, stream>>>(ar, ai, arLow, aiLow, tiling);
}

// ==========================================================================
//  Phase 2: Combine/Scale (AIV-only)
// ==========================================================================

constexpr uint32_t CB_BLOCK = 64;
constexpr uint32_t CB_HALF = 32;
constexpr uint32_t DIAG_SLOT = 64; // 256 B per diagonal-row slot (32B-aligned)
// Diagonal reduction chunk (multiple of 64 = ReduceSum Level-1 unit).
constexpr uint32_t DIAG_CHUNK = 1024;

class CsyrkKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR ar, GM_ADDR ai, GM_ADDR temp, GM_ADDR c, const CsyrkCombineTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessFast();
    __aicore__ inline void ProcessFull();
    __aicore__ inline void FastTileLoop(uint32_t blockIdx, uint32_t blockNum);
    __aicore__ inline void FastDiagPhase(uint32_t blockIdx, uint32_t blockNum);
    __aicore__ inline void FastRowBand();
    __aicore__ inline void FastRowBandDiag(
        uint32_t iBase, uint32_t jBase, uint32_t rLo, uint32_t rHi, uint32_t cols, bool upper);
    __aicore__ inline void FullTileBlock(
        uint32_t mb, uint32_t jb, uint32_t rLo, uint32_t rHi, bool upper, bool betaOnly);
    __aicore__ inline void ProcessHalf(uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols, uint32_t colOff);
    __aicore__ inline void ProcessHalfFast(uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols);
    __aicore__ inline void FastQuadBulk(uint32_t iBase, uint32_t jBase, uint32_t rowsA, uint32_t cols);
    __aicore__ inline void FullQuadRead2D(uint32_t iBase, uint32_t jAbs, uint32_t rowsA, uint32_t cols);
    __aicore__ inline void FullQuadCompute(uint32_t iBase, uint32_t jAbs, uint32_t rowsA, uint32_t cols);
    __aicore__ inline void ProcessDiagRow(uint32_t iAbs, uint32_t jBase, uint32_t len);
    __aicore__ inline void ProcessDiagRowFast(uint32_t iAbs, uint32_t jBase, uint32_t len);
    __aicore__ inline void ProcessRowStrip(uint32_t iAbs, uint32_t jBase, uint32_t len, bool overwriteDiag);
    __aicore__ inline void FullRowLoadCompute(
        uint32_t iAbs, uint32_t jBase, uint32_t len, bool overwriteDiag, const LocalTensor<float>& crUb,
        const LocalTensor<float>& ciUb, const LocalTensor<float>& q2Ub, const LocalTensor<float>& q3Ub);
    __aicore__ inline void ProcessRowStripFast(uint32_t iAbs, uint32_t jBase, uint32_t len, bool overwriteDiag);
    __aicore__ inline void FastStripLoad(
        uint32_t iAbs, uint32_t jBase, uint32_t len, const LocalTensor<float>& crUb, const LocalTensor<float>& ciUb,
        const LocalTensor<float>& q2Ub, const LocalTensor<float>& q3Ub);
    __aicore__ inline void ProcessDiagBand8(
        uint32_t iBase, uint32_t jBase, uint32_t rLo, uint32_t rows, uint32_t cols, bool upper);
    __aicore__ inline void DiagBand8Read(
        uint32_t iBase, uint32_t jBase, uint32_t rLo, uint32_t rows, const uint32_t* jStart, const uint32_t* len);
    __aicore__ inline void DiagBand8Overwrite(
        uint32_t iBase, uint32_t rLo, uint32_t rows, const uint32_t* len, bool upper);
    __aicore__ inline void DiagBand8Write(
        uint32_t iBase, uint32_t jBase, uint32_t rLo, uint32_t rows, const uint32_t* jStart, const uint32_t* len);
    __aicore__ inline void PrefetchHalfFast(uint32_t bufSel, uint32_t iBase, uint32_t jBase);
    __aicore__ inline void CombineHalfFast(uint32_t bufSel, uint32_t iBase, uint32_t jBase);
    __aicore__ inline void ProcessBetaHalf(
        uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols, uint32_t colOff);
    __aicore__ inline void ProcessBetaDiagRow(uint32_t iAbs, uint32_t jBase, uint32_t len);
    __aicore__ inline void ComputeDirectDiag(uint32_t iAbs, float& sumRe, float& sumIm);
    __aicore__ inline void DiagLoadChunk(
        bool transN, uint32_t iAbs, uint32_t kOff, uint32_t cnt, const LocalTensor<float>& arUb,
        const LocalTensor<float>& aiUb, const DataCopyPadExtParams<float>& pp);
    __aicore__ inline void ComputeDiagBlock8(uint32_t i0);
    __aicore__ inline void ComputeDiagBlock8T(uint32_t i0);
    __aicore__ inline void ComputeDiagBatch8(uint32_t i0, bool transN);
    __aicore__ inline void DiagBlock8Aligned(uint32_t i0, uint32_t j0, uint32_t cnt, uint32_t valid);
    __aicore__ inline void DiagBlock8Tail(
        uint32_t i0, uint32_t j0, uint32_t cnt, uint32_t valid, float* accTailRe, float* accTailIm);
    __aicore__ inline void DiagBlock8TAligned(uint32_t i0, uint32_t j0, uint32_t cnt, uint32_t valid);
    __aicore__ inline void DiagBlock8TTail(
        uint32_t i0, uint32_t j0, uint32_t cnt, uint32_t valid, float* accTailRe, float* accTailIm);
    __aicore__ inline void InitBuffersFast();
    __aicore__ inline void InitBuffersFull();

private:
    TPipe* pipe_;
    CsyrkCombineTilingData tiling_;
    GlobalTensor<float> tempGM_;
    GlobalTensor<float> cGM_;
    GlobalTensor<float> arGM_;
    GlobalTensor<float> aiGM_;
    TBuf<TPosition::VECIN> crBuf_;
    TBuf<TPosition::VECIN> ciBuf_;
    TBuf<TPosition::VECIN> q2Buf_;
    TBuf<TPosition::VECIN> q3Buf_;
    TBuf<TPosition::VECIN> cInBuf_;
    TBuf<TPosition::VECIN> arBuf_;
    TBuf<TPosition::VECIN> aiBuf_;
    TBuf<TPosition::VECCALC> dSumReBuf_;
    TBuf<TPosition::VECCALC> dSumImBuf_;
    TBuf<TPosition::VECCALC> dScalarBuf_;
    TBuf<TPosition::VECCALC> dWorkBuf_;
    TBuf<TPosition::VECCALC> reBuf_;
    TBuf<TPosition::VECCALC> imBuf_;
    TBuf<TPosition::VECCALC> t1Buf_;
    TBuf<TPosition::VECCALC> t2Buf_;
    // Batch direct-diagonal results (fast large-n path): diagRow0_/diagRow1_
    // delimit the <= 8 diagonal rows this core pre-computed with 8-way shared
    // contiguous reads; ProcessRowStripFast reads the scalar from these instead
    // of a per-row strided reduction.
    TBuf<TPosition::VECCALC> diagReBuf_;
    TBuf<TPosition::VECCALC> diagImBuf_;
    uint32_t diagRow0_;
    uint32_t diagRow1_;
    TBuf<TPosition::VECOUT> outReBuf_;
    TBuf<TPosition::VECOUT> outImBuf_;
    TBuf<TPosition::VECOUT> outBuf_;
    uint32_t rowStart_;
    uint32_t rowEnd_;
};

__aicore__ inline void CsyrkKernel::InitBuffersFast()
{
    // Fast config (alpha == (1,0) && beta == 0): the four quad buffers hold a
    // full 64x64 tile and complex scaling is dropped, so a non-diagonal block is
    // 4 reads + Sub/Add + Interleave + 1 write with no C_old load.
    pipe_->InitBuffer(crBuf_, 2 * CB_BLOCK * CB_BLOCK * sizeof(float));
    pipe_->InitBuffer(ciBuf_, 2 * CB_BLOCK * CB_BLOCK * sizeof(float));
    pipe_->InitBuffer(q2Buf_, 2 * CB_BLOCK * CB_BLOCK * sizeof(float));
    pipe_->InitBuffer(q3Buf_, 2 * CB_BLOCK * CB_BLOCK * sizeof(float));
    pipe_->InitBuffer(outBuf_, 2 * CB_BLOCK * CB_BLOCK * sizeof(float));
    pipe_->InitBuffer(t1Buf_, DIAG_CHUNK * sizeof(float)); // ComputeDirectDiag
    pipe_->InitBuffer(t2Buf_, DIAG_CHUNK * sizeof(float));
    // Row strips use 4B-strided interleaved writes, so the two halves must be
    // separate 32B-aligned buffers (an offset view would misalign).
    pipe_->InitBuffer(outReBuf_, CB_BLOCK * CB_BLOCK * sizeof(float));
    pipe_->InitBuffer(outImBuf_, CB_BLOCK * CB_BLOCK * sizeof(float));
}

__aicore__ inline void CsyrkKernel::InitBuffersFull()
{
    pipe_->InitBuffer(crBuf_, CB_BLOCK * CB_HALF * sizeof(float));
    pipe_->InitBuffer(ciBuf_, CB_BLOCK * CB_HALF * sizeof(float));
    pipe_->InitBuffer(q2Buf_, CB_BLOCK * CB_HALF * sizeof(float));
    pipe_->InitBuffer(q3Buf_, CB_BLOCK * CB_HALF * sizeof(float));
    pipe_->InitBuffer(cInBuf_, 2 * CB_BLOCK * CB_HALF * sizeof(float));
    pipe_->InitBuffer(reBuf_, CB_BLOCK * CB_HALF * sizeof(float));
    pipe_->InitBuffer(imBuf_, CB_BLOCK * CB_HALF * sizeof(float));
    pipe_->InitBuffer(t1Buf_, CB_BLOCK * CB_HALF * sizeof(float));
    pipe_->InitBuffer(t2Buf_, CB_BLOCK * CB_HALF * sizeof(float));
    pipe_->InitBuffer(outReBuf_, CB_BLOCK * CB_HALF * sizeof(float));
    pipe_->InitBuffer(outImBuf_, CB_BLOCK * CB_HALF * sizeof(float));
    pipe_->InitBuffer(outBuf_, 2 * CB_BLOCK * CB_HALF * sizeof(float));
}

__aicore__ inline void CsyrkKernel::Init(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR temp, GM_ADDR c, const CsyrkCombineTilingData& tiling, TPipe* pipe)
{
    pipe_ = pipe;
    tiling_ = tiling;
    tempGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(temp));
    cGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));
    arGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ar));
    aiGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ai));
    // Diagonal reduction buffers (shared by both paths).
    pipe_->InitBuffer(arBuf_, DIAG_CHUNK * sizeof(float));
    pipe_->InitBuffer(aiBuf_, DIAG_CHUNK * sizeof(float));
    pipe_->InitBuffer(dSumReBuf_, DIAG_CHUNK * sizeof(float));
    pipe_->InitBuffer(dSumImBuf_, DIAG_CHUNK * sizeof(float));
    pipe_->InitBuffer(dScalarBuf_, 64 * sizeof(float));
    pipe_->InitBuffer(dWorkBuf_, 4096 * sizeof(float));
    pipe_->InitBuffer(diagReBuf_, 32 * sizeof(float));
    pipe_->InitBuffer(diagImBuf_, 32 * sizeof(float));
    if (tiling.isFastCfg) {
        InitBuffersFast();
    } else {
        InitBuffersFull();
    }
    uint32_t blockIdx = GetBlockIdx();
    rowStart_ = blockIdx * tiling_.rowsPerCore;
    rowEnd_ = Min<uint32_t>(rowStart_ + tiling_.rowsPerCore, tiling_.n);
}

// out = alpha * (Cr + i*Ci) + beta * C_old, element-wise over cnt elements.
// reUb/imUb hold the split C_old; when beta==0 the caller passes betaRe=betaIm=0.
__aicore__ inline void CsyrkScaleAlphaBeta(
    const LocalTensor<float>& crUb, const LocalTensor<float>& ciUb, const LocalTensor<float>& reUb,
    const LocalTensor<float>& imUb, const LocalTensor<float>& t1, const LocalTensor<float>& t2,
    const LocalTensor<float>& outRe, const LocalTensor<float>& outIm, float alphaRe, float alphaIm, float betaRe,
    float betaIm, uint32_t cnt)
{
    Muls(t1, crUb, alphaRe, cnt);
    Muls(t2, ciUb, alphaIm, cnt);
    Sub(outRe, t1, t2, cnt);
    Muls(t1, ciUb, alphaRe, cnt);
    Muls(t2, crUb, alphaIm, cnt);
    Add(outIm, t1, t2, cnt);
    Muls(t1, reUb, betaRe, cnt);
    Add(outRe, outRe, t1, cnt);
    Muls(t2, imUb, betaIm, cnt);
    Sub(outRe, outRe, t2, cnt);
    Muls(t1, imUb, betaRe, cnt);
    Add(outIm, outIm, t1, cnt);
    Muls(t2, reUb, betaIm, cnt);
    Add(outIm, outIm, t2, cnt);
}

// Load C_old half tile (complex) into UB and split real/imag.
__aicore__ inline void LoadCOldHalf(
    LocalTensor<float>& cInUb, LocalTensor<float>& reUb, LocalTensor<float>& imUb, GlobalTensor<float>& cGM,
    uint64_t cOff, uint32_t rows, uint32_t cols, uint32_t ldc)
{
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    int64_t cSrcStride = static_cast<int64_t>(ldc - rows) * 2 * sizeof(float);
    DataCopyExtParams cpC{
        static_cast<uint16_t>(cols), static_cast<uint32_t>(rows * 2 * sizeof(float)), cSrcStride, 0, 0};
    DataCopyPad(cInUb, cGM[cOff], cpC, pp);
    PipeBarrier<PIPE_ALL>();
    DeInterleave(reUb, imUb, cInUb, static_cast<int32_t>(2 * rows * cols));
    PipeBarrier<PIPE_ALL>();
}

// Direct diagonal: the 4M decomposition computes C[i][i] = Q0[i][i] - Q1[i][i]
// where Q0/Q1 are sums of squares accumulated in fp32. For i==j the two large
// positive sums cancel, amplifying rounding error by ~sqrt(k) (fails the 1e-2
// ULP limit for k >= ~800). Recompute the diagonal element directly from the
// deinterleaved Ar/Ai with a plain fp32 accumulation, matching the reference's
// conditioning. trans=N reads row iAbs of Ar/Ai (strided by arLdc); trans=T/C
// reads column iAbs (contiguous).
// Split an 8-row-interleaved stream (cnt columns, layout col*8 + row) into 8
// contiguous row streams of cnt floats each, using 3 DeInterleave levels
// (8->4->2->1). scr must hold 16*cnt floats of scratch; dst 8*cnt floats.
__aicore__ inline void Deint8Rows(LocalTensor<float> src, LocalTensor<float> dst, LocalTensor<float> scr, uint32_t cnt)
{
    uint32_t c2 = 2 * cnt;
    uint32_t c4 = 4 * cnt;
    // L1: even src positions -> rows {0,2,4,6} (scr[0..4cnt)), odd -> {1,3,5,7}.
    DeInterleave(scr, scr[c4], src, static_cast<int32_t>(8 * cnt));
    // L2 on the even set -> {0,4} then {2,6}; on the odd set -> {1,5} then {3,7}.
    DeInterleave(scr[8 * cnt], scr[10 * cnt], scr, static_cast<int32_t>(4 * cnt));
    DeInterleave(scr[12 * cnt], scr[14 * cnt], scr[c4], static_cast<int32_t>(4 * cnt));
    // L3 leaves: each 2-row pair -> two single rows of cnt floats.
    DeInterleave(dst, dst[4 * cnt], scr[8 * cnt], static_cast<int32_t>(2 * cnt));
    DeInterleave(dst[2 * cnt], dst[6 * cnt], scr[10 * cnt], static_cast<int32_t>(2 * cnt));
    DeInterleave(dst[cnt], dst[5 * cnt], scr[12 * cnt], static_cast<int32_t>(2 * cnt));
    DeInterleave(dst[3 * cnt], dst[7 * cnt], scr[14 * cnt], static_cast<int32_t>(2 * cnt));
}

// Accumulate 8 diagonal rows' (Ar^2 - Ai^2, 2*Ar*Ai) over `cnt` columns into
// accRe/accIm (8 x colch), reading the per-row streams rowArB/rowAiB.
__aicore__ inline void DiagAccumRows(
    const LocalTensor<float>& accRe, const LocalTensor<float>& accIm, const LocalTensor<float>& rowArB,
    const LocalTensor<float>& rowAiB, const LocalTensor<float>& t1, const LocalTensor<float>& t2, uint32_t cnt,
    uint32_t colch, uint32_t valid)
{
    for (uint32_t r = 0; r < valid; r++) {
        LocalTensor<float> arRow = rowArB[r * cnt];
        LocalTensor<float> aiRow = rowAiB[r * cnt];
        LocalTensor<float> accReRow = accRe[r * colch];
        LocalTensor<float> accImRow = accIm[r * colch];
        Mul(t1, arRow, arRow, cnt);
        Mul(t2, aiRow, aiRow, cnt);
        Sub(t1, t1, t2, cnt);
        Add(accReRow, accReRow, t1, cnt);
        Mul(t1, arRow, aiRow, cnt);
        Add(t1, t1, t1, cnt);
        Add(accImRow, accImRow, t1, cnt);
    }
}

// Accumulate one tail column (< 8 remaining) for up to 8 rows: read the 8
// per-row values already staged in arUb/aiUb and add the scalar terms.
__aicore__ inline void DiagTailAccum(
    float* accTailRe, float* accTailIm, const LocalTensor<float>& arUb, const LocalTensor<float>& aiUb, uint32_t valid)
{
    for (uint32_t r = 0; r < valid; r++) {
        float arv = arUb.GetValue(r);
        float aiv = aiUb.GetValue(r);
        accTailRe[r] += arv * arv - aiv * aiv;
        accTailIm[r] += 2.0f * arv * aiv;
    }
}

// Reduce the per-row accumulators to scalars, adding the tail-column terms.
__aicore__ inline void DiagReduceRows(
    const LocalTensor<float>& accRe, const LocalTensor<float>& accIm, const LocalTensor<float>& scalar,
    const LocalTensor<float>& work, const LocalTensor<float>& dstRe, const LocalTensor<float>& dstIm,
    const float* accTailRe, const float* accTailIm, uint32_t colch, uint32_t valid)
{
    for (uint32_t r = 0; r < valid; r++) {
        ReduceSum(scalar, accRe[r * colch], work, colch);
        PipeBarrier<PIPE_ALL>();
        dstRe.SetValue(r, scalar.GetValue(0) + accTailRe[r]);
        ReduceSum(scalar, accIm[r * colch], work, colch);
        PipeBarrier<PIPE_ALL>();
        dstIm.SetValue(r, scalar.GetValue(0) + accTailIm[r]);
    }
}

// Batch direct diagonal for 8 aligned diagonal rows starting at i0 (trans=N
// only). Ar/Ai are column-major, so 8 consecutive rows of one column are 8
// contiguous floats (one 32 B read); a 512-column block is read as 512 such
// reads instead of 8*512 strided 4 B reads, then Deint8Rows splits the block
// into per-row streams that accumulate into eight scalar diagonals. The tail
// columns (< 8 per block) and the per-row accumulation order (ascending
// columns, fp32) match ComputeDirectDiag, so precision is unchanged. Runs on
// the fast large-n diagonal rows, reusing the tile buffers that are idle
// during the diagonal phase.
__aicore__ inline void CsyrkKernel::DiagBlock8Aligned(uint32_t i0, uint32_t j0, uint32_t cnt, uint32_t valid)
{
    constexpr uint32_t COLCH = 512;
    LocalTensor<float> xarB = q2Buf_.Get<float>();
    LocalTensor<float> rowArB = q2Buf_.Get<float>()[8 * COLCH];
    LocalTensor<float> xaiB = q3Buf_.Get<float>();
    LocalTensor<float> rowAiB = q3Buf_.Get<float>()[8 * COLCH];
    LocalTensor<float> scrB = outBuf_.Get<float>();
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    uint64_t srcOff = static_cast<uint64_t>(i0) + static_cast<uint64_t>(j0) * tiling_.arLdc;
    DataCopyExtParams cpBlock{static_cast<uint16_t>(cnt), 32, static_cast<int64_t>(tiling_.arLdc - 8) * 4, 0, 0};
    DataCopyPad<float, PaddingMode::Compact>(xarB, arGM_[srcOff], cpBlock, pp);
    DataCopyPad<float, PaddingMode::Compact>(xaiB, aiGM_[srcOff], cpBlock, pp);
    PipeBarrier<PIPE_ALL>();
    Deint8Rows(xarB, rowArB, scrB, cnt);
    Deint8Rows(xaiB, rowAiB, scrB, cnt);
    PipeBarrier<PIPE_ALL>();
    DiagAccumRows(
        crBuf_.Get<float>(), ciBuf_.Get<float>(), rowArB, rowAiB, t1Buf_.Get<float>(), t2Buf_.Get<float>(), cnt, COLCH,
        valid);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void CsyrkKernel::DiagBlock8Tail(
    uint32_t i0, uint32_t j0, uint32_t cnt, uint32_t valid, float* accTailRe, float* accTailIm)
{
    LocalTensor<float> t1 = t1Buf_.Get<float>();
    LocalTensor<float> t2 = t2Buf_.Get<float>();
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    // Tail columns (< 8): 32B read + scalar per row.
    for (uint32_t jj = j0; jj < j0 + cnt; jj++) {
        uint64_t off = static_cast<uint64_t>(i0) + static_cast<uint64_t>(jj) * tiling_.arLdc;
        DataCopyExtParams cp8{1, 32, 0, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(t1, arGM_[off], cp8, pp);
        DataCopyPad<float, PaddingMode::Compact>(t2, aiGM_[off], cp8, pp);
        PipeBarrier<PIPE_ALL>();
        DiagTailAccum(accTailRe, accTailIm, t1, t2, valid);
    }
}

__aicore__ inline void CsyrkKernel::DiagBlock8TAligned(uint32_t i0, uint32_t j0, uint32_t cnt, uint32_t valid)
{
    constexpr uint32_t COLCH = 512;
    LocalTensor<float> rowArB = q2Buf_.Get<float>();
    LocalTensor<float> rowAiB = q3Buf_.Get<float>();
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    // 8 columns (= diagonal rows) x cnt contiguous floats, strided by arLdc.
    uint64_t off = static_cast<uint64_t>(i0) * tiling_.arLdc + j0;
    DataCopyExtParams cp{
        static_cast<uint16_t>(valid), static_cast<uint32_t>(cnt * sizeof(float)),
        static_cast<int64_t>(tiling_.arLdc - cnt) * 4, 0, 0};
    DataCopyPad<float, PaddingMode::Compact>(rowArB, arGM_[off], cp, pp);
    DataCopyPad<float, PaddingMode::Compact>(rowAiB, aiGM_[off], cp, pp);
    PipeBarrier<PIPE_ALL>();
    DiagAccumRows(
        crBuf_.Get<float>(), ciBuf_.Get<float>(), rowArB, rowAiB, t1Buf_.Get<float>(), t2Buf_.Get<float>(), cnt, COLCH,
        valid);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void CsyrkKernel::DiagBlock8TTail(
    uint32_t i0, uint32_t j0, uint32_t cnt, uint32_t valid, float* accTailRe, float* accTailIm)
{
    LocalTensor<float> staging = outBuf_.Get<float>();
    LocalTensor<float> t2 = t2Buf_.Get<float>();
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    // Tail columns (< 8): read 8 rows x 8 floats (32B each), scalar per row.
    for (uint32_t jj = j0; jj < j0 + cnt; jj++) {
        uint64_t off = static_cast<uint64_t>(i0) * tiling_.arLdc + jj;
        DataCopyExtParams cp8{static_cast<uint16_t>(valid), 4, static_cast<int64_t>(tiling_.arLdc - 1) * 4, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(staging, arGM_[off], cp8, pp);
        DataCopyPad<float, PaddingMode::Compact>(t2, aiGM_[off], cp8, pp);
        PipeBarrier<PIPE_ALL>();
        DiagTailAccum(accTailRe, accTailIm, staging, t2, valid);
    }
}

// Batched direct diagonal for 8 diagonal rows at i0. The two data layouts
// (trans=N: Ar/Ai rows via 8-way DeInterleave; trans=T/C: contiguous columns)
// share the same K-chunk loop, per-row accumulation and final reduction; only
// the chunk read/store differ, handled by the *Aligned / *Tail helpers.
__aicore__ inline void CsyrkKernel::ComputeDiagBatch8(uint32_t i0, bool transN)
{
    const bool modeOk = transN ? (tiling_.isTransN != 0) : (tiling_.isTransN == 0);
    if (i0 >= Min<uint32_t>(i0 + 8, tiling_.n) || !modeOk || tiling_.useDirectDiag == 0) {
        diagRow0_ = 0;
        diagRow1_ = 0;
        return;
    }
    constexpr uint32_t COLCH = 512; // 8 rows x 512 cols; fits the idle tile bufs
    const uint32_t k = tiling_.k;
    diagRow0_ = i0;
    diagRow1_ = Min<uint32_t>(i0 + 8, tiling_.n);
    const uint32_t valid = diagRow1_ - i0;
    LocalTensor<float> accRe = crBuf_.Get<float>();
    LocalTensor<float> accIm = ciBuf_.Get<float>();
    float accTailRe[8] = {0.0f};
    float accTailIm[8] = {0.0f};
    // Elementwise-accumulate across K chunks into 8 x COLCH vectors and reduce
    // once per row at the end (a per-chunk ReduceSum + scalar read costs a full
    // pipeline drain each and made this phase sync-bound).
    Duplicate(accRe, 0.0f, 8 * COLCH);
    Duplicate(accIm, 0.0f, 8 * COLCH);
    PipeBarrier<PIPE_V>();
    for (uint32_t j0 = 0; j0 < k;) {
        uint32_t cntFull = Min<uint32_t>(COLCH, k - j0);
        uint32_t cnt = cntFull & ~0x7u;
        if (cnt > 0) {
            if (transN) {
                DiagBlock8Aligned(i0, j0, cnt, valid);
            } else {
                DiagBlock8TAligned(i0, j0, cnt, valid);
            }
            j0 += cnt;
        }
        if (transN) {
            DiagBlock8Tail(i0, j0, cntFull - cnt, valid, accTailRe, accTailIm);
        } else {
            DiagBlock8TTail(i0, j0, cntFull - cnt, valid, accTailRe, accTailIm);
        }
        j0 += (cntFull - cnt);
    }
    DiagReduceRows(
        accRe, accIm, dScalarBuf_.Get<float>(), dWorkBuf_.Get<float>(), diagReBuf_.Get<float>(),
        diagImBuf_.Get<float>(), accTailRe, accTailIm, COLCH, valid);
}

__aicore__ inline void CsyrkKernel::ComputeDiagBlock8(uint32_t i0) { ComputeDiagBatch8(i0, true); }

__aicore__ inline void CsyrkKernel::ComputeDiagBlock8T(uint32_t i0) { ComputeDiagBatch8(i0, false); }

// Load one K chunk of diagonal row iAbs into arUb/aiUb: trans=N reads row iAbs
// strided by arLdc; trans=T/C reads column iAbs contiguously.
__aicore__ inline void CsyrkKernel::DiagLoadChunk(
    bool transN, uint32_t iAbs, uint32_t kOff, uint32_t cnt, const LocalTensor<float>& arUb,
    const LocalTensor<float>& aiUb, const DataCopyPadExtParams<float>& pp)
{
    uint32_t arLdc = tiling_.arLdc;
    if (transN) {
        uint64_t off = static_cast<uint64_t>(iAbs) + static_cast<uint64_t>(kOff) * arLdc;
        DataCopyExtParams cp{static_cast<uint16_t>(cnt), 4, static_cast<int64_t>(arLdc - 1) * 4, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(arUb, arGM_[off], cp, pp);
        DataCopyPad<float, PaddingMode::Compact>(aiUb, aiGM_[off], cp, pp);
    } else {
        uint64_t off = static_cast<uint64_t>(iAbs) * arLdc + kOff;
        DataCopyExtParams cp{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(arUb, arGM_[off], cp, pp);
        DataCopyPad<float, PaddingMode::Compact>(aiUb, aiGM_[off], cp, pp);
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void CsyrkKernel::ComputeDirectDiag(uint32_t iAbs, float& sumRe, float& sumIm)
{
    uint32_t k = tiling_.k;
    uint32_t arLdc = tiling_.arLdc;
    bool transN = (tiling_.isTransN != 0);

    LocalTensor<float> arUb = arBuf_.Get<float>();
    LocalTensor<float> aiUb = aiBuf_.Get<float>();
    LocalTensor<float> sumReUb = dSumReBuf_.Get<float>();
    LocalTensor<float> sumImUb = dSumImBuf_.Get<float>();
    LocalTensor<float> t1 = t1Buf_.Get<float>();
    LocalTensor<float> t2 = t2Buf_.Get<float>();
    LocalTensor<float> scalar = dScalarBuf_.Get<float>();
    LocalTensor<float> work = dWorkBuf_.Get<float>();

    Duplicate(sumReUb, 0.0f, DIAG_CHUNK);
    Duplicate(sumImUb, 0.0f, DIAG_CHUNK);
    PipeBarrier<PIPE_ALL>();
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};

    for (uint32_t kOff = 0; kOff < k; kOff += DIAG_CHUNK) {
        uint32_t cnt = Min<uint32_t>(DIAG_CHUNK, k - kOff);
        DiagLoadChunk(transN, iAbs, kOff, cnt, arUb, aiUb, pp);
        Mul(t1, arUb, arUb, cnt);
        Mul(t2, aiUb, aiUb, cnt);
        Sub(t1, t1, t2, cnt); // ar^2 - ai^2
        Mul(t2, arUb, aiUb, cnt);
        Add(t2, t2, t2, cnt); // 2*ar*ai
        PipeBarrier<PIPE_ALL>();
        Add(sumReUb, sumReUb, t1, cnt);
        Add(sumImUb, sumImUb, t2, cnt);
        PipeBarrier<PIPE_ALL>();
    }

    ReduceSum(scalar, sumReUb, work, DIAG_CHUNK);
    PipeBarrier<PIPE_ALL>();
    sumRe = scalar.GetValue(0);
    ReduceSum(scalar, sumImUb, work, DIAG_CHUNK);
    PipeBarrier<PIPE_ALL>();
    sumIm = scalar.GetValue(0);
}

// Full-config bulk tile: read the four quads, fold in C_old (beta != 0), apply
// the complex alpha/beta scaling, interleave and write the complex C tile.
__aicore__ inline void CsyrkKernel::FullQuadCompute(uint32_t iBase, uint32_t jAbs, uint32_t rowsA, uint32_t cols)
{
    const uint32_t ldc = tiling_.ldc;
    const uint32_t cnt = rowsA * cols;
    FullQuadRead2D(iBase, jAbs, rowsA, cols);
    uint64_t cOff = jAbs * ldc * 2 + iBase * 2;
    LocalTensor<float> cInUb = cInBuf_.Get<float>();
    LocalTensor<float> reUb = reBuf_.Get<float>();
    LocalTensor<float> imUb = imBuf_.Get<float>();
    if (tiling_.isBetaZero == 0) {
        LoadCOldHalf(cInUb, reUb, imUb, cGM_, cOff, rowsA, cols, ldc);
    }
    PipeBarrier<PIPE_ALL>();
    Sub(crBuf_.Get<float>(), crBuf_.Get<float>(), ciBuf_.Get<float>(), cnt); // Cr = Q0 - Q1
    Add(ciBuf_.Get<float>(), q2Buf_.Get<float>(), q3Buf_.Get<float>(), cnt); // Ci = Q2 + Q3
    PipeBarrier<PIPE_ALL>();
    CsyrkScaleAlphaBeta(
        crBuf_.Get<float>(), ciBuf_.Get<float>(), reUb, imUb, t1Buf_.Get<float>(), t2Buf_.Get<float>(),
        outReBuf_.Get<float>(), outImBuf_.Get<float>(), tiling_.alphaReal, tiling_.alphaImag, tiling_.betaReal,
        tiling_.betaImag, cnt);
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> outDst0 = outBuf_.Get<float>();
    LocalTensor<float> outDst1 = outBuf_.GetWithOffset<float>(cnt, cnt * sizeof(float));
    Interleave(outDst0, outDst1, outReBuf_.Get<float>(), outImBuf_.Get<float>(), static_cast<int32_t>(cnt));
    PipeBarrier<PIPE_ALL>();
    int64_t cDstStride = static_cast<int64_t>(ldc - rowsA) * 2 * sizeof(float);
    DataCopyExtParams cpOut{
        static_cast<uint16_t>(cols), static_cast<uint32_t>(rowsA * 2 * sizeof(float)), 0, cDstStride, 0};
    DataCopyPad(cGM_[cOff], outDst0, cpOut);
    PipeBarrier<PIPE_ALL>();
}

// Full (complex alpha/beta) config: read a rowsA x cols tile of the four quads
// (temp is DNExt column-major; one 2D copy with cols blocks of rowsA floats).
__aicore__ inline void CsyrkKernel::FullQuadRead2D(uint32_t iBase, uint32_t jAbs, uint32_t rowsA, uint32_t cols)
{
    const uint32_t tempLdc = tiling_.tempLdc;
    LocalTensor<float> crUb = crBuf_.Get<float>();
    LocalTensor<float> ciUb = ciBuf_.Get<float>();
    LocalTensor<float> q2Ub = q2Buf_.Get<float>();
    LocalTensor<float> q3Ub = q3Buf_.Get<float>();
    uint64_t qStride = static_cast<uint64_t>(tiling_.n) * tempLdc;
    uint64_t baseOff = jAbs * tempLdc + iBase;
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    DataCopyExtParams cpT{
        static_cast<uint16_t>(cols), static_cast<uint32_t>(rowsA * sizeof(float)),
        static_cast<int64_t>((tempLdc - rowsA) * sizeof(float)), 0, 0};
    DataCopyPad(crUb, tempGM_[baseOff], cpT, pp);
    DataCopyPad(ciUb, tempGM_[qStride + baseOff], cpT, pp);
    DataCopyPad(q2Ub, tempGM_[2 * qStride + baseOff], cpT, pp);
    DataCopyPad(q3Ub, tempGM_[3 * qStride + baseOff], cpT, pp);
}

// Fast config bulk: read a full rowsA x cols tile of the four quads, form
// Cr = Q0-Q1 / Ci = Q2+Q3, interleave and write the complex C tile.
__aicore__ inline void CsyrkKernel::FastQuadBulk(uint32_t iBase, uint32_t jBase, uint32_t rowsA, uint32_t cols)
{
    const uint32_t ldc = tiling_.ldc;
    const uint32_t tempLdc = tiling_.tempLdc;
    const uint32_t cnt = rowsA * cols;
    LocalTensor<float> crUb = crBuf_.Get<float>();
    LocalTensor<float> ciUb = ciBuf_.Get<float>();
    LocalTensor<float> q2Ub = q2Buf_.Get<float>();
    LocalTensor<float> q3Ub = q3Buf_.Get<float>();
    LocalTensor<float> outUb = outBuf_.Get<float>();
    uint64_t qStride = static_cast<uint64_t>(tiling_.n) * tempLdc;
    uint64_t baseOff = static_cast<uint64_t>(jBase) * tempLdc + iBase;
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    DataCopyExtParams cpT{
        static_cast<uint16_t>(cols), static_cast<uint32_t>(rowsA * sizeof(float)),
        static_cast<int64_t>((tempLdc - rowsA) * sizeof(float)), 0, 0};
    DataCopyPad(crUb, tempGM_[baseOff], cpT, pp);
    DataCopyPad(ciUb, tempGM_[qStride + baseOff], cpT, pp);
    DataCopyPad(q2Ub, tempGM_[2 * qStride + baseOff], cpT, pp);
    DataCopyPad(q3Ub, tempGM_[3 * qStride + baseOff], cpT, pp);
    PipeBarrier<PIPE_ALL>();
    Sub(crUb, crUb, ciUb, cnt);
    Add(ciUb, q2Ub, q3Ub, cnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> outDst0 = outUb;
    LocalTensor<float> outDst1 = outUb[cnt];
    Interleave(outDst0, outDst1, crUb, ciUb, static_cast<int32_t>(cnt));
    PipeBarrier<PIPE_ALL>();
    uint64_t cOff = static_cast<uint64_t>(jBase) * ldc * 2 + iBase * 2;
    int64_t cDstStride = static_cast<int64_t>(ldc - rowsA) * 2 * sizeof(float);
    DataCopyExtParams cpOut{
        static_cast<uint16_t>(cols), static_cast<uint32_t>(rowsA * 2 * sizeof(float)), 0, cDstStride, 0};
    DataCopyPad(cGM_[cOff], outDst0, cpOut);
    PipeBarrier<PIPE_ALL>();
}

// Fast (alpha==(1,0), beta==0) single 64x64-tile combine: C = (Q0-Q1) + i(Q2+Q3).
__aicore__ inline void CsyrkKernel::ProcessHalfFast(uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols)
{
    uint32_t n = tiling_.n;
    uint32_t ldc = tiling_.ldc;
    uint32_t tempLdc = tiling_.tempLdc;

    LocalTensor<float> crUb = crBuf_.Get<float>();
    LocalTensor<float> ciUb = ciBuf_.Get<float>();
    LocalTensor<float> q2Ub = q2Buf_.Get<float>();
    LocalTensor<float> q3Ub = q3Buf_.Get<float>();
    LocalTensor<float> outUb = outBuf_.Get<float>();

    // 8-aligned bulk: every DataCopyPad blockLen is a multiple of 32 B on both
    // the temp read and the interleaved C write, so no 32 B padding skew can occur.
    uint32_t rowsA = rows & ~0x7u;
    if (rowsA > 0) {
        FastQuadBulk(iBase, jBase, rowsA, cols);
    }

    // Leftover 0..7 rows: per-row strips.
    for (uint32_t m = 0; m < rows - rowsA; m++) {
        ProcessRowStripFast(iBase + rowsA + m, jBase, cols, false);
    }
}

// Prefetch a full 64x64 half tile's four quad reads into ping/pong slot
// bufSel. Waits for the previous Combine on this slot (V_MTE2) before MTE2
// overwrites, then issues the reads and signals MTE2_V. rows/cols = CB_BLOCK.
__aicore__ inline void CsyrkKernel::PrefetchHalfFast(uint32_t bufSel, uint32_t iBase, uint32_t jBase)
{
    const uint32_t n = tiling_.n;
    const uint32_t tempLdc = tiling_.tempLdc;
    const uint32_t slot = bufSel ? (CB_BLOCK * CB_BLOCK) : 0;
    LocalTensor<float> crUb = crBuf_.Get<float>()[slot];
    LocalTensor<float> ciUb = ciBuf_.Get<float>()[slot];
    LocalTensor<float> q2Ub = q2Buf_.Get<float>()[slot];
    LocalTensor<float> q3Ub = q3Buf_.Get<float>()[slot];
    WaitFlag<HardEvent::V_MTE2>(bufSel + 4);
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    uint64_t qStride = static_cast<uint64_t>(n) * tempLdc;
    uint64_t baseOff = static_cast<uint64_t>(jBase) * tempLdc + iBase;
    DataCopyExtParams cpT{
        static_cast<uint16_t>(CB_BLOCK), static_cast<uint32_t>(CB_BLOCK * sizeof(float)),
        static_cast<int64_t>((tempLdc - CB_BLOCK) * sizeof(float)), 0, 0};
    DataCopyPad(crUb, tempGM_[baseOff], cpT, pp);
    DataCopyPad(ciUb, tempGM_[qStride + baseOff], cpT, pp);
    DataCopyPad(q2Ub, tempGM_[2 * qStride + baseOff], cpT, pp);
    DataCopyPad(q3Ub, tempGM_[3 * qStride + baseOff], cpT, pp);
    SetFlag<HardEvent::MTE2_V>(bufSel);
}

// Combine + write a tile prefetched by PrefetchHalfFast(bufSel). Waits for
// its quad reads, forms Cr/Ci, interleaves into outBuf (single-buffered, so
// the previous tile's MTE3 store is drained first via a same-round
// Set+Wait MTE3_V at the end of the previous Combine), writes C. Releases
// the quad slot for the next prefetch.
__aicore__ inline void CsyrkKernel::CombineHalfFast(uint32_t bufSel, uint32_t iBase, uint32_t jBase)
{
    const uint32_t n = tiling_.n;
    const uint32_t ldc = tiling_.ldc;
    const uint32_t slot = bufSel ? (CB_BLOCK * CB_BLOCK) : 0;
    LocalTensor<float> crUb = crBuf_.Get<float>()[slot];
    LocalTensor<float> ciUb = ciBuf_.Get<float>()[slot];
    LocalTensor<float> q2Ub = q2Buf_.Get<float>()[slot];
    LocalTensor<float> q3Ub = q3Buf_.Get<float>()[slot];
    LocalTensor<float> outUb = outBuf_.Get<float>();
    const uint32_t cnt = CB_BLOCK * CB_BLOCK;
    WaitFlag<HardEvent::MTE2_V>(bufSel);
    Sub(crUb, crUb, ciUb, cnt);
    Add(ciUb, q2Ub, q3Ub, cnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> outDst0 = outUb;
    LocalTensor<float> outDst1 = outUb[cnt];
    Interleave(outDst0, outDst1, crUb, ciUb, static_cast<int32_t>(cnt));
    // Quad slot free for the next prefetch on this bufSel.
    SetFlag<HardEvent::V_MTE2>(bufSel + 4);
    PipeBarrier<PIPE_ALL>();
    const uint64_t cOff = static_cast<uint64_t>(jBase) * ldc * 2 + iBase * 2;
    const int64_t cDstStride = static_cast<int64_t>(ldc - CB_BLOCK) * 2 * sizeof(float);
    DataCopyExtParams cpOut{
        static_cast<uint16_t>(CB_BLOCK), static_cast<uint32_t>(CB_BLOCK * 2 * sizeof(float)), 0, cDstStride, 0};
    DataCopyPad(cGM_[cOff], outDst0, cpOut);
    // Drain: this round's MTE3 store completes before the next Interleave
    // overwrites outBuf (same-round Set+Wait, herk pattern).
    SetFlag<HardEvent::MTE3_V>(6);
    WaitFlag<HardEvent::MTE3_V>(6);
}

// Fast (alpha==(1,0), beta==0) batch of up to 8 diagonal rows. Same addressing
// as ProcessRowStripFast (each row reads len strided 4B elements from the four
// quads into an aligned 256B UB slot, forms Cr/Ci, applies the precomputed
// direct-diagonal value, and writes the contiguous C segment with a strided
// dst), but the reads of all rows are issued together and the three
// PipeBarrier<PIPE_ALL> are hoisted out of the row loop: 3 barriers per band
// instead of 3 per row (the diagonal phase is barrier-bound).
__aicore__ inline void CsyrkKernel::DiagBand8Read(
    uint32_t iBase, uint32_t jBase, uint32_t rLo, uint32_t rows, const uint32_t* jStart, const uint32_t* len)
{
    const uint32_t tempLdc = tiling_.tempLdc;
    const uint64_t qStride = static_cast<uint64_t>(tiling_.n) * tempLdc;
    LocalTensor<float> crUb = crBuf_.Get<float>();
    LocalTensor<float> ciUb = ciBuf_.Get<float>();
    LocalTensor<float> q2Ub = q2Buf_.Get<float>();
    LocalTensor<float> q3Ub = q3Buf_.Get<float>();
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    for (uint32_t m = 0; m < rows; m++) {
        uint32_t iAbs = iBase + rLo + m;
        uint64_t tempOff = static_cast<uint64_t>(iAbs) + static_cast<uint64_t>(jBase + jStart[m]) * tempLdc;
        DataCopyExtParams cpR{static_cast<uint16_t>(len[m]), 4, static_cast<int64_t>(tempLdc - 1) * 4, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(crUb[m * DIAG_SLOT], tempGM_[tempOff], cpR, pp);
        DataCopyPad<float, PaddingMode::Compact>(ciUb[m * DIAG_SLOT], tempGM_[qStride + tempOff], cpR, pp);
        DataCopyPad<float, PaddingMode::Compact>(q2Ub[m * DIAG_SLOT], tempGM_[2 * qStride + tempOff], cpR, pp);
        DataCopyPad<float, PaddingMode::Compact>(q3Ub[m * DIAG_SLOT], tempGM_[3 * qStride + tempOff], cpR, pp);
    }
}

__aicore__ inline void CsyrkKernel::DiagBand8Overwrite(
    uint32_t iBase, uint32_t rLo, uint32_t rows, const uint32_t* len, bool upper)
{
    LocalTensor<float> crUb = crBuf_.Get<float>();
    LocalTensor<float> ciUb = ciBuf_.Get<float>();
    LocalTensor<float> dRe = diagReBuf_.Get<float>();
    LocalTensor<float> dIm = diagImBuf_.Get<float>();
    for (uint32_t m = 0; m < rows; m++) {
        uint32_t rAbs = iBase + rLo + m;
        if (rAbs >= diagRow0_ && rAbs < diagRow1_) {
            // UPPER: diagonal element is the first column of this row's segment;
            // LOWER: the last one.
            uint32_t idx = upper ? 0 : (len[m] - 1);
            crUb.SetValue(m * DIAG_SLOT + idx, dRe.GetValue(rAbs - diagRow0_));
            ciUb.SetValue(m * DIAG_SLOT + idx, dIm.GetValue(rAbs - diagRow0_));
        }
    }
}

__aicore__ inline void CsyrkKernel::DiagBand8Write(
    uint32_t iBase, uint32_t jBase, uint32_t rLo, uint32_t rows, const uint32_t* jStart, const uint32_t* len)
{
    const uint32_t ldc = tiling_.ldc;
    LocalTensor<float> crUb = crBuf_.Get<float>();
    LocalTensor<float> ciUb = ciBuf_.Get<float>();
    for (uint32_t m = 0; m < rows; m++) {
        uint32_t iAbs = iBase + rLo + m;
        uint64_t cOff = static_cast<uint64_t>(jBase + jStart[m]) * ldc * 2 + static_cast<uint64_t>(iAbs) * 2;
        DataCopyExtParams cpC{
            static_cast<uint16_t>(len[m]), 4, 0, static_cast<uint32_t>((ldc * 2 - 1) * sizeof(float)), 0};
        DataCopyPad<float, PaddingMode::Compact>(cGM_[cOff], crUb[m * DIAG_SLOT], cpC);
        DataCopyPad<float, PaddingMode::Compact>(cGM_[cOff + 1], ciUb[m * DIAG_SLOT], cpC);
    }
}

__aicore__ inline void CsyrkKernel::ProcessDiagBand8(
    uint32_t iBase, uint32_t jBase, uint32_t rLo, uint32_t rows, uint32_t cols, bool upper)
{
    if (rows == 0 || cols == 0) {
        return;
    }
    uint32_t jStart[8];
    uint32_t len[8];
    for (uint32_t m = 0; m < rows; m++) {
        uint32_t rRel = rLo + m;
        jStart[m] = upper ? rRel : 0;
        len[m] = upper ? (cols - rRel) : (rRel + 1);
    }
    DiagBand8Read(iBase, jBase, rLo, rows, jStart, len);
    PipeBarrier<PIPE_ALL>();
    LocalTensor<float> crUb = crBuf_.Get<float>();
    LocalTensor<float> ciUb = ciBuf_.Get<float>();
    LocalTensor<float> q2Ub = q2Buf_.Get<float>();
    LocalTensor<float> q3Ub = q3Buf_.Get<float>();
    for (uint32_t m = 0; m < rows; m++) {
        Sub(crUb[m * DIAG_SLOT], crUb[m * DIAG_SLOT], ciUb[m * DIAG_SLOT], len[m]);
        Add(ciUb[m * DIAG_SLOT], q2Ub[m * DIAG_SLOT], q3Ub[m * DIAG_SLOT], len[m]);
    }
    if (tiling_.useDirectDiag != 0 && iBase == jBase) {
        DiagBand8Overwrite(iBase, rLo, rows, len, upper);
    }
    PipeBarrier<PIPE_ALL>();
    DiagBand8Write(iBase, jBase, rLo, rows, jStart, len);
    PipeBarrier<PIPE_ALL>();
}

// Read one diagonal/edge row's four quadrants (len strided 4B elements each)
// into cr/ci/q2/q3. temp is DNExt column-major (addr = row + col*tempLdc).
__aicore__ inline void CsyrkKernel::FastStripLoad(
    uint32_t iAbs, uint32_t jBase, uint32_t len, const LocalTensor<float>& crUb, const LocalTensor<float>& ciUb,
    const LocalTensor<float>& q2Ub, const LocalTensor<float>& q3Ub)
{
    uint64_t qStride = static_cast<uint64_t>(tiling_.n) * tiling_.tempLdc;
    uint64_t tempOff = static_cast<uint64_t>(iAbs) + static_cast<uint64_t>(jBase) * tiling_.tempLdc;
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    DataCopyExtParams cpR{static_cast<uint16_t>(len), 4, static_cast<int64_t>(tiling_.tempLdc - 1) * 4, 0, 0};
    DataCopyPad<float, PaddingMode::Compact>(crUb, tempGM_[tempOff], cpR, pp);
    DataCopyPad<float, PaddingMode::Compact>(ciUb, tempGM_[qStride + tempOff], cpR, pp);
    DataCopyPad<float, PaddingMode::Compact>(q2Ub, tempGM_[2 * qStride + tempOff], cpR, pp);
    DataCopyPad<float, PaddingMode::Compact>(q3Ub, tempGM_[3 * qStride + tempOff], cpR, pp);
    PipeBarrier<PIPE_ALL>();
}

// Fast (alpha==(1,0), beta==0) row strip using only the 64x64 quad/out buffers.
__aicore__ inline void CsyrkKernel::ProcessRowStripFast(uint32_t iAbs, uint32_t jBase, uint32_t len, bool overwriteDiag)
{
    if (len == 0) {
        return;
    }
    uint32_t n = tiling_.n;
    uint32_t ldc = tiling_.ldc;
    uint32_t tempLdc = tiling_.tempLdc;

    LocalTensor<float> crUb = crBuf_.Get<float>();
    LocalTensor<float> ciUb = ciBuf_.Get<float>();
    LocalTensor<float> q2Ub = q2Buf_.Get<float>();
    LocalTensor<float> q3Ub = q3Buf_.Get<float>();
    LocalTensor<float> outRe = outReBuf_.Get<float>();
    LocalTensor<float> outIm = outImBuf_.Get<float>();

    uint64_t cOff = static_cast<uint64_t>(jBase) * ldc * 2 + iAbs * 2;
    DataCopyPadExtParams<float> ppRead{true, 0, 0, 0.0f};
    FastStripLoad(iAbs, jBase, len, crUb, ciUb, q2Ub, q3Ub);

    Sub(crUb, crUb, ciUb, len);
    Add(ciUb, q2Ub, q3Ub, len);
    PipeBarrier<PIPE_ALL>();

    if (overwriteDiag && tiling_.useDirectDiag) {
        uint32_t diagIdx = (tiling_.uploMode == ACLBLAS_UPPER) ? 0 : (len - 1);
        float diagRe = 0.0f;
        float diagIm = 0.0f;
        if (iAbs >= diagRow0_ && iAbs < diagRow1_) {
            diagRe = diagReBuf_.Get<float>().GetValue(iAbs - diagRow0_);
            diagIm = diagImBuf_.Get<float>().GetValue(iAbs - diagRow0_);
        } else {
            ComputeDirectDiag(iAbs, diagRe, diagIm);
        }
        crUb.SetValue(diagIdx, diagRe);
        ciUb.SetValue(diagIdx, diagIm);
        PipeBarrier<PIPE_ALL>();
    }

    // Cr/Ci already hold (Q0-Q1) / (Q2+Q3); the strided MTE3 write interleaves
    // them on the fly (matches ProcessRowStrip).
    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    DataCopyExtParams cpDiag{static_cast<uint16_t>(len), 4, 0, static_cast<uint32_t>(ldc * 2 - 1) * 4, 0};
    DataCopyPad<float, PaddingMode::Compact>(cGM_[cOff], crUb, cpDiag);
    DataCopyPad<float, PaddingMode::Compact>(cGM_[cOff + 1], ciUb, cpDiag);
    PipeBarrier<PIPE_ALL>();
}

// Fast (alpha==(1,0), beta==0) diagonal row.
__aicore__ inline void CsyrkKernel::ProcessDiagRowFast(uint32_t iAbs, uint32_t jBase, uint32_t len)
{
    ProcessRowStripFast(iAbs, jBase, len, true);
}

// Combine one full (non-diagonal) half tile: rows x cols at (iBase, jBase+colOff).
__aicore__ inline void CsyrkKernel::ProcessHalf(
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols, uint32_t colOff)
{
    uint64_t jAbs = static_cast<uint64_t>(jBase) + colOff;

    uint32_t rowsA = rows & ~0x7u;
    if (rowsA > 0) {
        FullQuadCompute(iBase, jAbs, rowsA, cols);
    }

    // Leftover 0..7 rows: per-row strips (proven diag-style addressing).
    for (uint32_t m = 0; m < rows - rowsA; m++) {
        ProcessRowStrip(iBase + rowsA + m, jAbs, cols, false);
    }
}

// Read and fold the stray C_old row (beta*C), then overwrite the diagonal when
// the quad value is not accurate enough. Returns Cr/Ci in crUb/ciUb.
__aicore__ inline void CsyrkKernel::FullRowLoadCompute(
    uint32_t iAbs, uint32_t jBase, uint32_t len, bool overwriteDiag, const LocalTensor<float>& crUb,
    const LocalTensor<float>& ciUb, const LocalTensor<float>& q2Ub, const LocalTensor<float>& q3Ub)
{
    const uint32_t ldc = tiling_.ldc;
    uint64_t cOff = static_cast<uint64_t>(jBase) * ldc * 2 + iAbs * 2;
    FastStripLoad(iAbs, jBase, len, crUb, ciUb, q2Ub, q3Ub);
    if (tiling_.isBetaZero == 0) {
        LocalTensor<float> cInUb = cInBuf_.Get<float>();
        DataCopyPadExtParams<float> ppRead{true, 0, 0, 0.0f};
        DataCopyExtParams cpC{static_cast<uint16_t>(len), 8, static_cast<int64_t>(ldc * 2 - 2) * 4, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(cInUb, cGM_[cOff], cpC, ppRead);
        PipeBarrier<PIPE_ALL>();
        DeInterleave(reBuf_.Get<float>(), imBuf_.Get<float>(), cInUb, static_cast<int32_t>(2 * len));
        PipeBarrier<PIPE_ALL>();
    }
    Sub(crUb, crUb, ciUb, len); // Cr = Q0 - Q1
    Add(ciUb, q2Ub, q3Ub, len); // Ci = Q2 + Q3
    PipeBarrier<PIPE_ALL>();
    if (overwriteDiag) {
        // The quad value loses precision to cancellation for large k.
        uint32_t diagIdx = (tiling_.uploMode == ACLBLAS_UPPER) ? 0 : (len - 1);
        float diagRe = 0.0f;
        float diagIm = 0.0f;
        ComputeDirectDiag(iAbs, diagRe, diagIm);
        crUb.SetValue(diagIdx, diagRe);
        ciUb.SetValue(diagIdx, diagIm);
        PipeBarrier<PIPE_ALL>();
    }
}

// One diagonal row: len elements at absolute row iAbs, temp/C columns from jBase.
__aicore__ inline void CsyrkKernel::ProcessRowStrip(uint32_t iAbs, uint32_t jBase, uint32_t len, bool overwriteDiag)
{
    if (len == 0) {
        return;
    }
    const uint32_t ldc = tiling_.ldc;
    LocalTensor<float> crUb = crBuf_.Get<float>();
    LocalTensor<float> ciUb = ciBuf_.Get<float>();
    FullRowLoadCompute(iAbs, jBase, len, overwriteDiag, crUb, ciUb, q2Buf_.Get<float>(), q3Buf_.Get<float>());
    // out = alpha * (Cr + i*Ci)  (+ beta * C_old when beta != 0)
    const float bRe = (tiling_.isBetaZero == 0) ? tiling_.betaReal : 0.0f;
    const float bIm = (tiling_.isBetaZero == 0) ? tiling_.betaImag : 0.0f;
    CsyrkScaleAlphaBeta(
        crUb, ciUb, reBuf_.Get<float>(), imBuf_.Get<float>(), t1Buf_.Get<float>(), t2Buf_.Get<float>(),
        outReBuf_.Get<float>(), outImBuf_.Get<float>(), tiling_.alphaReal, tiling_.alphaImag, bRe, bIm, len);
    PipeBarrier<PIPE_ALL>();
    uint64_t cOff = static_cast<uint64_t>(jBase) * ldc * 2 + iAbs * 2;
    DataCopyExtParams cpDiag{static_cast<uint16_t>(len), 4, 0, static_cast<uint32_t>(ldc * 2 - 1) * 4, 0};
    DataCopyPad<float, PaddingMode::Compact>(cGM_[cOff], outReBuf_.Get<float>(), cpDiag);
    DataCopyPad<float, PaddingMode::Compact>(cGM_[cOff + 1], outImBuf_.Get<float>(), cpDiag);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void CsyrkKernel::ProcessDiagRow(uint32_t iAbs, uint32_t jBase, uint32_t len)
{
    ProcessRowStrip(iAbs, jBase, len, true);
}

// beta-only path (alpha==0 or k==0): C = beta*C_old on the triangle.
__aicore__ inline void CsyrkKernel::ProcessBetaHalf(
    uint32_t iBase, uint32_t jBase, uint32_t rows, uint32_t cols, uint32_t colOff)
{
    uint32_t ldc = tiling_.ldc;
    float betaRe = tiling_.betaReal;
    float betaIm = tiling_.betaImag;
    uint64_t jAbs = static_cast<uint64_t>(jBase) + colOff;

    LocalTensor<float> cInUb = cInBuf_.Get<float>();
    LocalTensor<float> reUb = reBuf_.Get<float>();
    LocalTensor<float> imUb = imBuf_.Get<float>();
    LocalTensor<float> t1 = t1Buf_.Get<float>();
    LocalTensor<float> t2 = t2Buf_.Get<float>();
    LocalTensor<float> outRe = outReBuf_.Get<float>();
    LocalTensor<float> outIm = outImBuf_.Get<float>();
    LocalTensor<float> outUb = outBuf_.Get<float>();

    // 8-aligned bulk (see ProcessHalf): blockLen multiples of 32 B on read and write.
    uint32_t rowsA = rows & ~0x7u;
    if (rowsA > 0) {
        uint32_t cnt = rowsA * cols;
        uint64_t cOff = jAbs * ldc * 2 + iBase * 2;
        LoadCOldHalf(cInUb, reUb, imUb, cGM_, cOff, rowsA, cols, ldc);
        PipeBarrier<PIPE_ALL>();

        Muls(t1, reUb, betaRe, cnt);
        Muls(t2, imUb, betaIm, cnt);
        Sub(outRe, t1, t2, cnt);
        Muls(t1, imUb, betaRe, cnt);
        Muls(t2, reUb, betaIm, cnt);
        Add(outIm, t1, t2, cnt);
        PipeBarrier<PIPE_ALL>();

        LocalTensor<float> outDst0 = outBuf_.Get<float>();
        LocalTensor<float> outDst1 = outBuf_.GetWithOffset<float>(cnt, cnt * sizeof(float));
        Interleave(outDst0, outDst1, outRe, outIm, static_cast<int32_t>(cnt));
        PipeBarrier<PIPE_ALL>();

        DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
        int64_t cDstStride = static_cast<int64_t>(ldc - rowsA) * 2 * sizeof(float);
        DataCopyExtParams cpOut{
            static_cast<uint16_t>(cols), static_cast<uint32_t>(rowsA * 2 * sizeof(float)), 0, cDstStride, 0};
        DataCopyPad(cGM_[cOff], outUb, cpOut);
        PipeBarrier<PIPE_ALL>();
    }

    // Leftover 0..7 rows: per-row beta strips.
    for (uint32_t m = 0; m < rows - rowsA; m++) {
        ProcessBetaDiagRow(iBase + rowsA + m, jAbs, cols);
    }
}

__aicore__ inline void CsyrkKernel::ProcessBetaDiagRow(uint32_t iAbs, uint32_t jBase, uint32_t len)
{
    if (len == 0) {
        return;
    }
    uint32_t ldc = tiling_.ldc;
    float betaRe = tiling_.betaReal;
    float betaIm = tiling_.betaImag;

    LocalTensor<float> cInUb = cInBuf_.Get<float>();
    LocalTensor<float> reUb = reBuf_.Get<float>();
    LocalTensor<float> imUb = imBuf_.Get<float>();
    LocalTensor<float> t1 = t1Buf_.Get<float>();
    LocalTensor<float> t2 = t2Buf_.Get<float>();
    LocalTensor<float> outRe = outReBuf_.Get<float>();
    LocalTensor<float> outIm = outImBuf_.Get<float>();
    LocalTensor<float> outUb = outBuf_.Get<float>();

    uint64_t cOff = static_cast<uint64_t>(jBase) * ldc * 2 + iAbs * 2;
    DataCopyPadExtParams<float> ppRead{true, 0, 0, 0.0f};
    DataCopyExtParams cpC{static_cast<uint16_t>(len), 8, static_cast<int64_t>(ldc * 2 - 2) * 4, 0, 0};
    DataCopyPad<float, PaddingMode::Compact>(cInUb, cGM_[cOff], cpC, ppRead);
    PipeBarrier<PIPE_ALL>();
    DeInterleave(reUb, imUb, cInUb, static_cast<int32_t>(2 * len));
    PipeBarrier<PIPE_ALL>();

    Muls(t1, reUb, betaRe, len);
    Muls(t2, imUb, betaIm, len);
    Sub(outRe, t1, t2, len);
    Muls(t1, imUb, betaRe, len);
    Muls(t2, reUb, betaIm, len);
    Add(outIm, t1, t2, len);
    PipeBarrier<PIPE_ALL>();

    DataCopyPadExtParams<float> pp{true, 0, 0, 0.0f};
    DataCopyExtParams cpDiag{static_cast<uint16_t>(len), 4, 0, static_cast<uint32_t>(ldc * 2 - 1) * 4, 0};
    DataCopyPad<float, PaddingMode::Compact>(cGM_[cOff], outRe, cpDiag);
    DataCopyPad<float, PaddingMode::Compact>(cGM_[cOff + 1], outIm, cpDiag);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void CsyrkKernel::FastTileLoop(uint32_t blockIdx, uint32_t blockNum)
{
    // Non-diagonal half tiles keep the tile-stride distribution; the diagonal
    // tile is excluded (handled row-by-row below). Software-pipelined: full
    // 64x64 half tiles are prefetched one iteration ahead; edge tiles go serial.
    SetFlag<HardEvent::V_MTE2>(4);
    SetFlag<HardEvent::V_MTE2>(5);
    const uint32_t n = tiling_.n;
    const bool upper = (tiling_.uploMode == 121);
    const uint32_t nBlocks = CeilDiv<uint32_t>(n, CB_BLOCK);
    uint32_t curBuf = 0;
    uint32_t pIBase = 0;
    uint32_t pJBase = 0;
    bool havePrev = false;
    for (uint64_t tileIdx = blockIdx; tileIdx < static_cast<uint64_t>(nBlocks) * nBlocks; tileIdx += blockNum) {
        const uint32_t mb = static_cast<uint32_t>(tileIdx / nBlocks);
        const uint32_t jb = static_cast<uint32_t>(tileIdx % nBlocks);
        if (mb == jb) { // diagonal tile: handled row-by-row
            continue;
        }
        const uint32_t iBase = mb * CB_BLOCK;
        const uint32_t blockRows = Min<uint32_t>(CB_BLOCK, n - iBase);
        const uint32_t jBase = jb * CB_BLOCK;
        const uint32_t cols = Min<uint32_t>(CB_BLOCK, n - jBase);
        if (!((upper && mb < jb) || (!upper && mb > jb))) {
            continue;
        }
        if (blockRows == CB_BLOCK && cols == CB_BLOCK) {
            PrefetchHalfFast(curBuf, iBase, jBase);
            if (havePrev) {
                CombineHalfFast(curBuf ^ 1, pIBase, pJBase);
            } else {
                havePrev = true;
            }
            pIBase = iBase;
            pJBase = jBase;
            curBuf ^= 1;
        } else {
            if (havePrev) {
                CombineHalfFast(curBuf ^ 1, pIBase, pJBase);
                havePrev = false;
            }
            ProcessHalfFast(iBase, jBase, blockRows, cols);
        }
    }
    if (havePrev) {
        CombineHalfFast(curBuf ^ 1, pIBase, pJBase);
    }
}

// Diagonal phase for the fast config: 8-row groups distributed over cores. Each
// group is pre-computed with 8-way shared contiguous reads (ComputeDiagBlock8/8T)
// and rows are independent, so any balanced partition is exact.
__aicore__ inline void CsyrkKernel::FastDiagPhase(uint32_t blockIdx, uint32_t blockNum)
{
    const uint32_t n = tiling_.n;
    const bool upper = (tiling_.uploMode == 121);
    // The diagonal phase reuses the quad/out tile buffers; drain pending ops.
    PipeBarrier<PIPE_ALL>();
    uint32_t nGroups = CeilDiv<uint32_t>(n, 8);
    for (uint32_t g = blockIdx; g < nGroups; g += blockNum) {
        uint32_t gRow = g * 8;
        if (tiling_.isTransN != 0) {
            ComputeDiagBlock8(gRow);
        } else {
            ComputeDiagBlock8T(gRow);
        }
        uint32_t gEnd = Min<uint32_t>(gRow + 8, n);
        // ProcessDiagBand8 needs 8 rows of the SAME tile, so use it only when
        // the whole group lies in one tile; otherwise handle row by row.
        uint32_t mb0 = gRow / CB_BLOCK;
        uint32_t mb1 = (gEnd - 1) / CB_BLOCK;
        if (mb0 == mb1 && (gEnd - gRow) == 8) {
            ProcessDiagBand8(
                mb0 * CB_BLOCK, mb0 * CB_BLOCK, gRow - mb0 * CB_BLOCK, 8, Min<uint32_t>(CB_BLOCK, n - mb0 * CB_BLOCK),
                upper);
            continue;
        }
        for (uint32_t row = gRow; row < gEnd; row++) {
            const uint32_t mb = row / CB_BLOCK;
            const uint32_t r = row - mb * CB_BLOCK;
            const uint32_t jBase = mb * CB_BLOCK;
            const uint32_t cols = Min<uint32_t>(CB_BLOCK, n - jBase);
            if (upper) {
                ProcessDiagRowFast(row, jBase + r, cols - r);
            } else {
                ProcessDiagRowFast(row, jBase, r + 1);
            }
        }
    }
}

// Fast-config diagonal rows for a small-n row band: precompute the 8-row
// shared diagonal values (ComputeDiagBlock8/8T) then emit the row strips.
__aicore__ inline void CsyrkKernel::FastRowBandDiag(
    uint32_t iBase, uint32_t jBase, uint32_t rLo, uint32_t rHi, uint32_t cols, bool upper)
{
    const uint32_t n = tiling_.n;
    if (tiling_.useDirectDiag != 0) {
        for (uint32_t r0 = rLo; r0 < rHi; r0 += 8) {
            uint32_t gRow = iBase + r0;
            if (tiling_.isTransN != 0) {
                ComputeDiagBlock8(gRow);
            } else {
                ComputeDiagBlock8T(gRow);
            }
            uint32_t gEndR = Min<uint32_t>(Min<uint32_t>(gRow + 8, n), iBase + rHi);
            for (uint32_t row = gRow; row < gEndR; row++) {
                uint32_t rr = row - iBase;
                if (upper) {
                    ProcessDiagRowFast(row, jBase + rr, cols - rr);
                } else {
                    ProcessDiagRowFast(row, jBase, rr + 1);
                }
            }
        }
        return;
    }
    // useDirectDiag==0 (k<=512): no per-element recompute, so full 8-row bands
    // go through ProcessDiagBand8 (3 barriers/band).
    uint32_t r = rLo;
    for (; r + 8 <= rHi; r += 8) {
        ProcessDiagBand8(iBase, jBase, r, 8, cols, upper);
    }
    for (; r < rHi; r++) {
        uint32_t iAbs = iBase + r;
        if (upper) {
            ProcessDiagRowFast(iAbs, jBase + r, cols - r);
        } else {
            ProcessDiagRowFast(iAbs, jBase, r + 1);
        }
    }
}

// Small-n fast config: row-band distribution so every core has work.
__aicore__ inline void CsyrkKernel::FastRowBand()
{
    const uint32_t n = tiling_.n;
    const uint32_t nBlocks = CeilDiv<uint32_t>(n, CB_BLOCK);
    const bool upper = (tiling_.uploMode == 121);
    for (uint32_t mb = rowStart_ / CB_BLOCK; mb < CeilDiv<uint32_t>(rowEnd_, CB_BLOCK); mb++) {
        uint32_t iBase = mb * CB_BLOCK;
        uint32_t blockRows = Min<uint32_t>(CB_BLOCK, n - iBase);
        uint32_t rLo = (rowStart_ > iBase) ? (rowStart_ - iBase) : 0;
        uint32_t rHi = Min<uint32_t>(blockRows, rowEnd_ - iBase);
        if (rLo >= rHi) {
            continue;
        }
        for (uint32_t jb = 0; jb < nBlocks; jb++) {
            uint32_t jBase = jb * CB_BLOCK;
            uint32_t cols = Min<uint32_t>(CB_BLOCK, n - jBase);
            if (mb == jb) {
                FastRowBandDiag(iBase, jBase, rLo, rHi, cols, upper);
            } else if ((upper && mb <= jb) || (!upper && mb >= jb)) {
                ProcessHalfFast(iBase + rLo, jBase, rHi - rLo, cols);
            }
        }
    }
}

__aicore__ inline void CsyrkKernel::ProcessFast()
{
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    const uint32_t n = tiling_.n;
    const uint32_t nBlocks = CeilDiv<uint32_t>(n, CB_BLOCK);
    const uint64_t triTilesF = static_cast<uint64_t>(nBlocks) * (nBlocks + 1) / 2;
    if (triTilesF >= blockNum) {
        FastTileLoop(blockIdx, blockNum);
        FastDiagPhase(blockIdx, blockNum);
        return;
    }
    FastRowBand();
}

// Full (complex alpha/beta) config, one triangular block at (mb, jb) rows
// [rLo, rHi) of the tile. Dispatches to the diag/half/beta variants.
__aicore__ inline void CsyrkKernel::FullTileBlock(
    uint32_t mb, uint32_t jb, uint32_t rLo, uint32_t rHi, bool upper, bool betaOnly)
{
    const uint32_t n = tiling_.n;
    uint32_t iBase = mb * CB_BLOCK;
    uint32_t rows = rHi - rLo;
    uint32_t jBase = jb * CB_BLOCK;
    uint32_t cols = Min<uint32_t>(CB_BLOCK, n - jBase);
    if (mb == jb) {
        for (uint32_t r = rLo; r < rHi; r++) {
            uint32_t iAbs = iBase + r;
            uint32_t len = upper ? (cols - r) : (r + 1);
            uint32_t jCol = upper ? (jBase + r) : jBase;
            if (betaOnly) {
                ProcessBetaDiagRow(iAbs, jCol, len);
            } else {
                ProcessDiagRow(iAbs, jCol, len);
            }
        }
        return;
    }
    if (!((upper && mb < jb) || (!upper && mb > jb))) {
        return;
    }
    uint32_t half1 = Min<uint32_t>(CB_HALF, cols);
    if (betaOnly) {
        ProcessBetaHalf(iBase + rLo, jBase, rows, half1, 0);
    } else {
        ProcessHalf(iBase + rLo, jBase, rows, half1, 0);
    }
    if (cols > CB_HALF) {
        uint32_t half2 = cols - CB_HALF;
        if (betaOnly) {
            ProcessBetaHalf(iBase + rLo, jBase, rows, half2, CB_HALF);
        } else {
            ProcessHalf(iBase + rLo, jBase, rows, half2, CB_HALF);
        }
    }
}

__aicore__ inline void CsyrkKernel::ProcessFull()
{
    // Tile-stride distribution over the upper/lower triangle: balanced block
    // counts per core. Small n (fewer triTiles than cores) falls back to a
    // row-band split so no core idles.
    const uint32_t n = tiling_.n;
    const uint32_t nBlocks = CeilDiv<uint32_t>(n, CB_BLOCK);
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    const bool upper = (tiling_.uploMode == 121);
    const bool betaOnly = (tiling_.isAlphaZero != 0 || tiling_.isKZero != 0);
    const uint64_t triTiles = static_cast<uint64_t>(nBlocks) * (nBlocks + 1) / 2;
    if (triTiles >= blockNum) {
        for (uint64_t tileIdx = blockIdx; tileIdx < static_cast<uint64_t>(nBlocks) * nBlocks; tileIdx += blockNum) {
            uint32_t mb = static_cast<uint32_t>(tileIdx / nBlocks);
            uint32_t jb = static_cast<uint32_t>(tileIdx % nBlocks);
            uint32_t blockRows = Min<uint32_t>(CB_BLOCK, n - mb * CB_BLOCK);
            FullTileBlock(mb, jb, 0, blockRows, upper, betaOnly);
        }
        return;
    }
    for (uint32_t mb = rowStart_ / CB_BLOCK; mb < CeilDiv<uint32_t>(rowEnd_, CB_BLOCK); mb++) {
        uint32_t iBase = mb * CB_BLOCK;
        uint32_t blockRows = Min<uint32_t>(CB_BLOCK, n - iBase);
        uint32_t rLo = (rowStart_ > iBase) ? (rowStart_ - iBase) : 0;
        uint32_t rHi = Min<uint32_t>(blockRows, rowEnd_ - iBase);
        if (rLo >= rHi) {
            continue;
        }
        for (uint32_t jb = 0; jb < nBlocks; jb++) {
            FullTileBlock(mb, jb, rLo, rHi, upper, betaOnly);
        }
    }
}

__aicore__ inline void CsyrkKernel::Process()
{
    if (tiling_.isFastCfg != 0) {
        ProcessFast();
        return;
    }
    ProcessFull();
}

// ==========================================================================
//  Phase 2 (SIMT): small-n combine. One thread per (i,j) element of the uplo
//  triangle; no TPipe/TBuf, no pipe barriers. The SIMD path's fixed overhead
//  (pipe init + per-tile PIPE_ALL) dominates small n (n=128: 12us measured vs
//  a 0.13us GM floor), which thread-level parallelism avoids. Fast config only
//  (alpha==(1,0), beta==0): C = (Q0-Q1) + i(Q2+Q3).
// ==========================================================================
struct CsyrkSimtArgs {
    uint32_t n;
    uint32_t ldc;
    uint32_t tempLdc;
    uint32_t qStride;      // quadrant stride (n * tempLdc)
    uint8_t upper;
    uint8_t overwriteDiag; // useDirectDiag: overwrite C[i][i] from ar/ai
    uint32_t arLdc;
    uint32_t k;
    uint8_t isTransN;
};

// Direct diagonal element (r,r): sum over k of (ar^2 - ai^2, 2*ar*ai) with
// the SIMD ComputeDirectDiag accumulation order (ascending k, fp32).
__simt_callee__ inline void CsyrkSimtDirectDiag(
    const CsyrkSimtArgs& a, uint32_t r, __gm__ float* arGm, __gm__ float* aiGm, float& re, float& im)
{
    float sr = 0.0f, si = 0.0f;
    for (uint32_t kk = 0; kk < a.k; kk++) {
        const uint64_t o =
            a.isTransN ? (static_cast<uint64_t>(kk) * a.arLdc + r) : (static_cast<uint64_t>(r) * a.arLdc + kk);
        const float arv = arGm[o];
        const float aiv = aiGm[o];
        sr += arv * arv - aiv * aiv;
        si += 2.0f * arv * aiv;
    }
    re = sr;
    im = si;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void csyrk_combine_simt(
    uint32_t rowLo, uint32_t rowHi, CsyrkSimtArgs a, __gm__ float* tempGm, __gm__ float* cGm, __gm__ float* arGm,
    __gm__ float* aiGm)
{
    const uint32_t n = a.n;
    const uint32_t ldc = a.ldc;
    const uint32_t tempLdc = a.tempLdc;
    const uint64_t qStride = a.qStride;
    // Row-band ownership: this block covers rows [rowLo, rowHi) of every column.
    // Rows are the fastest thread dimension (temp and C are column-major, so
    // consecutive rows of one column are adjacent in memory).
    const uint32_t band = rowHi - rowLo;
    const uint32_t total = band * n;
    for (uint32_t t = threadIdx.x; t < total; t += blockDim.x) {
        const uint32_t j = t / band;             // absolute column j
        const uint32_t r = rowLo + t - j * band; // absolute row i (fastest)
        if (a.upper ? (j < r) : (j > r)) {
            continue;                            // outside the uplo triangle
        }
        // temp column-major DNExt: quad q element (i,j) at q*qStride + j*tempLdc + i.
        const uint64_t off = static_cast<uint64_t>(j) * tempLdc + r;
        float re = tempGm[off] - tempGm[qStride + off];
        float im = tempGm[2 * qStride + off] + tempGm[3 * qStride + off];
        if (a.overwriteDiag && r == j) {
            CsyrkSimtDirectDiag(a, r, arGm, aiGm, re, im);
        }
        const uint64_t cOff = (static_cast<uint64_t>(j) * ldc + r) * 2;
        cGm[cOff] = re;
        cGm[cOff + 1] = im;
    }
}

extern "C" __global__ __aicore__ void csyrk_combine_simt_kernel(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR temp, GM_ADDR c, const CsyrkCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t n = tiling.n;
    if (n == 0) {
        return;
    }
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();
    const uint32_t rowsPer = CeilDiv<uint32_t>(n, blockNum);
    const uint32_t rowLo = Min<uint32_t>(blockIdx * rowsPer, n);
    const uint32_t rowHi = Min<uint32_t>(rowLo + rowsPer, n);
    if (rowLo >= rowHi) {
        return;
    }
    CsyrkSimtArgs a;
    a.n = n;
    a.ldc = tiling.ldc;
    a.tempLdc = tiling.tempLdc;
    a.qStride = static_cast<uint32_t>(static_cast<uint64_t>(n) * tiling.tempLdc);
    a.upper = (tiling.uploMode == 121) ? 1 : 0;
    a.overwriteDiag = tiling.useDirectDiag;
    a.arLdc = tiling.arLdc;
    a.k = tiling.k;
    a.isTransN = tiling.isTransN;
    asc_vf_call<csyrk_combine_simt>(
        dim3{tiling.simtThreads, 1, 1}, rowLo, rowHi, a, reinterpret_cast<__gm__ float*>(temp),
        reinterpret_cast<__gm__ float*>(c), reinterpret_cast<__gm__ float*>(ar), reinterpret_cast<__gm__ float*>(ai));
}

void csyrk_combine_simt_kernel_do(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR temp, GM_ADDR c, const CsyrkCombineTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    csyrk_combine_simt_kernel<<<numBlocks, nullptr, stream>>>(ar, ai, temp, c, tiling);
}

extern "C" __global__ __aicore__ void csyrk_combine_kernel(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR temp, GM_ADDR c, const CsyrkCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CsyrkKernel op;
    op.Init(ar, ai, temp, c, tiling, &pipe);
    op.Process();
}

void csyrk_combine_kernel_do(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR temp, GM_ADDR c, const CsyrkCombineTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    csyrk_combine_kernel<<<numBlocks, nullptr, stream>>>(ar, ai, temp, c, tiling);
}
