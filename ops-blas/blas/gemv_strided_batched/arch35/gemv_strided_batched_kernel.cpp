/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "gemv_strided_batched_tiling_data.h"
#include "common/helper/kernel_constant.h"

using namespace AscendC;

constexpr uint32_t MTE_BLOCK_BYTES = 32u;
constexpr uint32_t SIMD_REG_BYTES = 256u;
constexpr uint32_t VEC_FLOAT_PER_REPEAT = MTE_BLOCK_BYTES / sizeof(float);
constexpr uint32_t ELENUM_BLOCK_FP16    = 16;
constexpr uint32_t ELENUM_REPEAT_FP16   = 128;

// ============================================================
// AIV SIMD regbase path (Transpose + incx=1 && incy=1)
// Supports FP32, FP16 (HSH/HSS), BF16 (TST/TSS)
// ============================================================
template <typename T, const bool TRANS_T = true, bool HSS_MODE = false>
class GemvStridedBatchedAIV {
public:
    __aicore__ inline GemvStridedBatchedAIV() : pipe(nullptr) {}
    __aicore__ inline void Init(TPipe& p, GM_ADDR A, GM_ADDR x, GM_ADDR y, const GemvStridedBatchedTilingData& tiling);
    __aicore__ inline void Process();

    __aicore__ inline void InitUbuf();
    __aicore__ inline void ParseTilingData(const GemvStridedBatchedTilingData& tiling);
    __aicore__ inline void CopyInMatAndVec(uint32_t curBatchId, uint32_t nOffset, uint32_t nCurr,
                                           uint32_t mOffset, uint32_t mCurr);
    __aicore__ inline void CopyInY(uint32_t curBatchId, uint32_t mOffset, uint32_t mCurr);
    __aicore__ inline void ApplyAlphaBeta(uint32_t curCalMatNum);
    __aicore__ inline void CastleHalfToFloat(uint32_t dotCnt);
    __aicore__ inline void CastleYHalfToFloat();
    __aicore__ inline void CastFloatToHalf();
    __aicore__ inline void CopyOutRsltFloat(uint32_t curBatchId, uint32_t mOffset, uint32_t mCurr);
    __aicore__ inline void CopyOutRsltHalf(uint32_t curBatchId, uint32_t mOffset, uint32_t mCurr);
    __aicore__ inline void ComputeDotProduct(uint32_t outCnt, uint32_t dotCnt, uint32_t dotTile, bool isFirstDot);
    __aicore__ inline void OutputMChunkAndSync(uint32_t b, uint32_t mOffset, uint32_t mCurr, uint32_t mCurrAlign);

private:
    static constexpr bool IS_FLOAT     = IsSameType<T, float>::value;
    static constexpr bool IS_FLOAT_OUT = IS_FLOAT || HSS_MODE;

    TPipe* pipe;

    GlobalTensor<T> inAGM;
    GlobalTensor<T> inxGM;
    GlobalTensor<T> inyGM;
    GlobalTensor<T> outGM;
    GlobalTensor<float> inyFloatGM;
    GlobalTensor<float> outFloatGM;

    TBuf<QuePosition::VECCALC> inABuf;
    TBuf<QuePosition::VECCALC> inxBuf;
    TBuf<QuePosition::VECCALC> inyBuf;
    TBuf<QuePosition::VECCALC> outBuf;
    TBuf<QuePosition::VECCALC> matTmpBuf;
    TBuf<QuePosition::VECCALC> vecTmpBuf;
    TBuf<QuePosition::VECCALC> matPreBuf;
    TBuf<QuePosition::VECCALC> vecPreBuf;

    LocalTensor<float> matTmpLocal;
    LocalTensor<float> vecTmpLocal;
    LocalTensor<T>   matPreLocal;
    LocalTensor<T>   vecPreLocal;
    LocalTensor<float> matFloatLocal;
    LocalTensor<float> vecFloatLocal;
    LocalTensor<float> mulResultLocal;
    LocalTensor<float> sumResultLocal;
    LocalTensor<float> yInFloatLocal;

    float alpha = 1.0f;
    float beta  = 0.0f;

    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t outDim = 0;
    uint32_t dotDim = 0;
    uint32_t outTile = 0;
    uint32_t dotTile = 0;
    uint32_t mCurr = 0;
    uint32_t calMatNum  = 0;
    uint32_t startMatId = 0;
    uint32_t maxMatEleNum = 0;
    uint32_t maxVecEleNum = 0;
    uint32_t szInA=0, szInx=0, szInY=0, szMatTmp=0, szVecTmp=0;
    int32_t lda = 0;
    int32_t vecIdx = 0;
    int64_t strideA = 0;
    int64_t stridex = 0;
    int64_t stridey = 0;
};

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::Init(
    TPipe& p, GM_ADDR A, GM_ADDR x, GM_ADDR y, const GemvStridedBatchedTilingData& tiling)
{
    this->pipe = &p;
    this->vecIdx = GetBlockIdx();
    ParseTilingData(tiling);
    this->inAGM.SetGlobalBuffer((__gm__ T *)A);
    this->inxGM.SetGlobalBuffer((__gm__ T *)x);
    if constexpr (HSS_MODE) {
        this->inyFloatGM.SetGlobalBuffer((__gm__ float *)y);
        this->outFloatGM.SetGlobalBuffer((__gm__ float *)y);
    } else {
        this->inyGM.SetGlobalBuffer((__gm__ T *)y);
        this->outGM.SetGlobalBuffer((__gm__ T *)y);
    }
    InitUbuf();
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::ParseTilingData(
    const GemvStridedBatchedTilingData& tiling)
{
    this->alpha          = tiling.alpha;
    this->beta           = tiling.beta;
    this->m              = tiling.m;
    this->n              = tiling.n;
    this->outDim         = tiling.outSize;
    this->dotDim         = tiling.dotSize;
    this->dotTile        = tiling.dotTile;
    this->outTile        = tiling.outTile;
    this->szInA          = tiling.bufInA;
    this->szInx          = tiling.bufInx;
    this->szInY          = tiling.bufInY;
    this->szMatTmp       = tiling.bufMatTmp;
    this->szVecTmp       = tiling.bufVecTmp;
    this->lda            = tiling.lda;

    this->strideA  = tiling.strideA;
    this->stridex  = tiling.stridex;
    this->stridey  = tiling.stridey;

    this->startMatId = this->vecIdx * tiling.batchPerCore;
    this->calMatNum  = (this->vecIdx == tiling.usedCoreNum - 1) ? tiling.batchTail : tiling.batchPerCore;
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::InitUbuf()
{
    this->maxMatEleNum = this->outTile * this->dotTile;
    this->maxVecEleNum = (this->dotTile < 64u) ? 64u : this->dotTile;
    if (this->outTile > this->maxVecEleNum) {
        this->maxVecEleNum = this->outTile;
    }

    this->pipe->InitBuffer(this->inABuf,    this->szInA);
    this->pipe->InitBuffer(this->inxBuf,    this->szInx);
    this->pipe->InitBuffer(this->inyBuf,    this->szInY);
    this->pipe->InitBuffer(this->matTmpBuf, this->szMatTmp);
    this->pipe->InitBuffer(this->vecTmpBuf, this->szVecTmp);

    if constexpr (!IS_FLOAT && !HSS_MODE) {
        this->pipe->InitBuffer(this->outBuf, this->szInY);
    }

    if constexpr (!IS_FLOAT) {
        this->pipe->InitBuffer(this->matPreBuf, this->maxMatEleNum * sizeof(T));
        this->pipe->InitBuffer(this->vecPreBuf, this->maxVecEleNum * sizeof(T));
        this->matPreLocal = matPreBuf.Get<T>();
        this->vecPreLocal = vecPreBuf.Get<T>();
    }

    this->matTmpLocal = matTmpBuf.Get<float>();
    this->vecTmpLocal = vecTmpBuf.Get<float>();

    this->matFloatLocal = this->matTmpLocal[0];
    uint32_t yOff = ((this->outTile + VEC_FLOAT_PER_REPEAT - 1) & ~(VEC_FLOAT_PER_REPEAT - 1));

    if constexpr (IS_FLOAT) {
        this->mulResultLocal = this->matTmpLocal[0];
        this->sumResultLocal = this->vecTmpLocal[0];
        this->yInFloatLocal  = this->vecTmpLocal[yOff];
    } else {
        this->mulResultLocal = this->matTmpLocal[this->maxMatEleNum];
        this->vecFloatLocal  = this->vecTmpLocal[0];
        this->sumResultLocal = this->vecTmpLocal[this->dotTile];
        this->yInFloatLocal  = this->vecTmpLocal[this->dotTile + yOff];
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::CopyInMatAndVec(
    uint32_t curBatchId, uint32_t nOffset, uint32_t nCurr,
    uint32_t mOffset, uint32_t mCurr)
{
    LocalTensor<T> inALocal = inABuf.Get<T>();
    LocalTensor<T> inxLocal = inxBuf.Get<T>();

    uint64_t inABase = static_cast<uint64_t>(this->startMatId + curBatchId) * static_cast<uint64_t>(this->strideA)
                     + static_cast<uint64_t>(nOffset) * static_cast<uint64_t>(this->lda) + mOffset;

    uint32_t elemAlign = MTE_BLOCK_BYTES / sizeof(T);
    uint32_t mCurrAligned = ((mCurr + elemAlign - 1) & ~(elemAlign - 1));
    uint32_t matSrcStride = uint32_t((static_cast<uint32_t>(this->lda) - mCurr) * sizeof(T));
    uint32_t matDstStride = (this->dotTile >= mCurrAligned)
        ? uint32_t((this->dotTile - mCurrAligned) * sizeof(T) / MTE_BLOCK_BYTES) : 0u;

    DataCopyExtParams cp{static_cast<uint16_t>(nCurr),
        static_cast<uint32_t>(mCurr * sizeof(T)), matSrcStride, matDstStride, 0};
    DataCopyPadExtParams<T> pp{false, 0, 0, 0};
    DataCopyPad(inALocal, this->inAGM[inABase], cp, pp);

    uint64_t inxOffset = static_cast<uint64_t>(this->startMatId + curBatchId) * static_cast<uint64_t>(this->stridex)
                       + mOffset;
    DataCopyPad(inxLocal, this->inxGM[inxOffset],
        DataCopyExtParams{1, static_cast<uint32_t>(mCurr * sizeof(T)), 0, 0, 0},
        DataCopyPadExtParams<T>{false, 0, 0, 0});
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::CopyInY(uint32_t curBatchId,
    uint32_t mOffset, uint32_t mCurr)
{
    uint64_t offset = static_cast<uint64_t>(this->startMatId + curBatchId) * static_cast<uint64_t>(this->stridey)
                    + mOffset;
    if constexpr (HSS_MODE) {
        auto inyLocal = inyBuf.Get<float>();
        DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(inyLocal, this->inyFloatGM[offset], copyParams, padParams);
    } else {
        LocalTensor<T> inyLocal = inyBuf.Get<T>();
        DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inyLocal, this->inyGM[offset], copyParams, padParams);
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::ApplyAlphaBeta(uint32_t curCalMatNum)
{
    uint32_t count = curCalMatNum;
    if (alpha != 1.0f) { Muls(this->sumResultLocal, this->sumResultLocal, alpha, count); }
    if (beta != 0.0f) {
        Muls(this->mulResultLocal, this->yInFloatLocal, beta, count);
        PipeBarrier<PIPE_V>();
        Add(this->sumResultLocal, this->sumResultLocal, this->mulResultLocal, count);
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::CastleYHalfToFloat()
{
    LocalTensor<T> inyLocal = inyBuf.Get<T>();
    uint32_t cnt = ((this->mCurr + (MTE_BLOCK_BYTES / sizeof(T) - 1)) & ~(MTE_BLOCK_BYTES / sizeof(T) - 1));
    if (cnt > this->maxVecEleNum) {
        cnt = this->maxVecEleNum;
    }
    DataCopy(this->vecPreLocal, inyLocal, cnt);
    PipeBarrier<PIPE_V>();
    uint16_t castRepeatNum = (cnt - 1) / ELENUM_REPEAT_FP16 + 1;
    uint32_t castCnt = castRepeatNum * ELENUM_REPEAT_FP16;
    if (castCnt > this->maxVecEleNum) {
        castCnt = this->maxVecEleNum;
    }
    Cast(this->yInFloatLocal, this->vecPreLocal, RoundMode::CAST_NONE, castCnt);
    PipeBarrier<PIPE_V>();
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::CastleHalfToFloat(uint32_t dotCnt)
{
    LocalTensor<T> inALocal = inABuf.Get<T>();
    LocalTensor<T> inxLocal = inxBuf.Get<T>();

    uint32_t matRows = this->mCurr;
    uint16_t matCastRepeatNum = (matRows * this->dotTile - 1) / ELENUM_REPEAT_FP16 + 1;
    DataCopy(this->matPreLocal, inALocal, matRows * this->dotTile);

    DataCopy(this->vecPreLocal, inxLocal, this->dotTile);
    PipeBarrier<PIPE_V>();
    Cast(this->vecFloatLocal, this->vecPreLocal, RoundMode::CAST_NONE, this->dotTile);
    PipeBarrier<PIPE_V>();
    uint32_t matCastCnt = matCastRepeatNum * ELENUM_REPEAT_FP16;
    if (matCastCnt > this->maxMatEleNum) {
        matCastCnt = this->maxMatEleNum;
    }
    Cast(this->matFloatLocal, this->matPreLocal, RoundMode::CAST_NONE, matCastCnt);
    PipeBarrier<PIPE_V>();
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::CastFloatToHalf()
{
    LocalTensor<T> outLocal = outBuf.Get<T>();
    constexpr RoundMode rm = IsSameType<T, bfloat16_t>::value ? RoundMode::CAST_RINT : RoundMode::CAST_ROUND;
    uint16_t blockNum = (this->mCurr - 1) / ELENUM_BLOCK_FP16 + 1;
    Cast(outLocal, this->sumResultLocal, rm, blockNum * ELENUM_BLOCK_FP16);
    PipeBarrier<PIPE_V>();
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::CopyOutRsltFloat(uint32_t curBatchId,
    uint32_t mOffset, uint32_t mCurr)
{
    uint64_t outOffset = static_cast<uint64_t>(this->startMatId + curBatchId) * static_cast<uint64_t>(this->stridey)
                       + mOffset;
    DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(float)), 0, 0, 0};
    if constexpr (HSS_MODE)
        DataCopyPad(this->outFloatGM[outOffset], this->sumResultLocal, copyParams);
    else
        DataCopyPad(this->outGM[outOffset], this->sumResultLocal, copyParams);
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::CopyOutRsltHalf(uint32_t curBatchId,
    uint32_t mOffset, uint32_t mCurr)
{
    LocalTensor<T> outLocal = outBuf.Get<T>();
    uint64_t outOffset = static_cast<uint64_t>(this->startMatId + curBatchId) * static_cast<uint64_t>(this->stridey)
                       + mOffset;
    DataCopyExtParams copyParams{uint16_t(1), uint32_t(mCurr * sizeof(T)), 0, 0, 0};
    DataCopyPad(this->outGM[outOffset], outLocal, copyParams);
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::ComputeDotProduct(
    uint32_t outCnt, uint32_t dotCnt, uint32_t dotTile, bool isFirstDot)
{
    constexpr uint32_t VL = SIMD_REG_BYTES / sizeof(float);
    uint32_t lM = outCnt;
    uint32_t lNTile = dotTile;
    auto& dstLT = isFirstDot ? this->sumResultLocal : this->yInFloatLocal;
    __ubuf__ float* aAddr = IS_FLOAT ? (__ubuf__ float*)inABuf.Get<float>().GetPhyAddr()
                                     : (__ubuf__ float*)this->matFloatLocal.GetPhyAddr();
    __ubuf__ float* xAddr = IS_FLOAT ? (__ubuf__ float*)inxBuf.Get<float>().GetPhyAddr()
                                     : (__ubuf__ float*)this->vecFloatLocal.GetPhyAddr();
    __ubuf__ float* dAddr = (__ubuf__ float*)dstLT.GetPhyAddr();
    uint16_t vLoopNum = (dotCnt + VL - 1) / VL;
    uint32_t tailLen  = dotCnt % VL;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregA;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregX;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregMul;
        AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregSum;
        auto maskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t row = 0; row < static_cast<uint16_t>(lM); row++) {
            AscendC::MicroAPI::Duplicate<float>(vregSum, 0.0f, maskAll);
            uint32_t matOff = row * static_cast<uint32_t>(lNTile);
            for (uint16_t i = 0; i < vLoopNum; i++) {
                uint32_t col = i * static_cast<uint32_t>(VL);
                uint32_t chunk = (i == vLoopNum - 1 && tailLen) ? tailLen : VL;
                auto mask = (chunk < VL)
                    ? AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(chunk) : maskAll;
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                    vregA, (__ubuf__ float*)(aAddr + matOff + col));
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                    vregX, (__ubuf__ float*)(xAddr + col));
                AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vregMul, vregA, vregX, mask);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(vregSum, vregSum, vregMul, maskAll);
            }
            AscendC::MicroAPI::ReduceSum(vregSum, vregSum, maskAll);
            __ubuf__ float* ptr = dAddr + row;
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(ptr, vregSum, 1, maskAll);
        }
    }
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::OutputMChunkAndSync(
    uint32_t b, uint32_t mOffset, uint32_t mCurr, uint32_t mCurrAlign)
{
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if constexpr (IS_FLOAT_OUT) {
        DataCopy(this->yInFloatLocal, inyBuf.Get<float>(), mCurrAlign);
        PipeBarrier<PIPE_V>();
    } else {
        CastleYHalfToFloat();
    }
    ApplyAlphaBeta(mCurrAlign);

    if constexpr (!IS_FLOAT && !HSS_MODE)
        CastFloatToHalf();

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if constexpr (IS_FLOAT_OUT)
        CopyOutRsltFloat(b, mOffset, mCurr);
    else
        CopyOutRsltHalf(b, mOffset, mCurr);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
}

template <typename T, const bool TRANS_T, bool HSS_MODE>
__aicore__ inline void GemvStridedBatchedAIV<T, TRANS_T, HSS_MODE>::Process()
{
    if (0 == this->calMatNum)
        return;

    uint32_t outTile = this->outTile;
    uint32_t dotTile = this->dotTile;
    uint32_t outDim = this->outDim;
    uint32_t dotDim = this->dotDim;

    for (uint32_t b = 0; b < this->calMatNum; b++) {
        for (uint32_t outOff = 0; outOff < outDim; outOff += outTile) {
            uint32_t outCnt = (outOff + outTile > outDim) ? (outDim - outOff) : outTile;
            uint32_t outCntAlign = ((outCnt + VEC_FLOAT_PER_REPEAT - 1) & ~(VEC_FLOAT_PER_REPEAT - 1));
            this->mCurr = outCnt;

            CopyInY(b, outOff, outCnt);

            bool isFirstDot = true;
            for (uint32_t dotOff = 0; dotOff < dotDim; dotOff += dotTile) {
                uint32_t dotCnt = (dotOff + dotTile > dotDim) ? (dotDim - dotOff) : dotTile;
                CopyInMatAndVec(b, outOff, outCnt, dotOff, dotCnt);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                if constexpr (!IS_FLOAT)
                    CastleHalfToFloat(dotCnt);

                ComputeDotProduct(outCnt, dotCnt, dotTile, isFirstDot);

                if (!isFirstDot)
                    Add(this->sumResultLocal, this->sumResultLocal, this->yInFloatLocal, outCntAlign);
                isFirstDot = false;
            }
            OutputMChunkAndSync(b, outOff, outCnt, outCntAlign);
        }
    }
}

// ============================================================
// SIMT — one thread per output element
// Explicit strideA/stridex/stridey (int64_t) for batch addressing
// Column-major: A(row, col) = A[row + col * lda]
// ============================================================
template <typename T_IN, typename T_OUT, bool IS_TRANS>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM)
inline void GemvSimtStrided(
    uint32_t m, uint32_t n, uint32_t numB, uint32_t startB,
    float alpha, float beta,
    int32_t lda, int32_t incx, int32_t incy,
    int64_t strideA, int64_t stridex, int64_t stridey,
    __gm__ const T_IN *aGm, __gm__ const T_IN *xGm, __gm__ T_OUT *yGm)
{
    constexpr bool IN_FP16  = IsSameType<T_IN, half>::value;
    constexpr bool IN_BF16  = IsSameType<T_IN, bfloat16_t>::value;
    constexpr bool OUT_FP16 = IsSameType<T_OUT, half>::value;
    constexpr bool OUT_BF16 = IsSameType<T_OUT, bfloat16_t>::value;
    uint32_t outDim = IS_TRANS ? n : m;
    uint32_t dotDim = IS_TRANS ? m : n;
    int64_t lda64 = static_cast<int64_t>(lda);
    int64_t incx64 = -static_cast<int64_t>(incx);
    int64_t incy64 = -static_cast<int64_t>(incy);

    for (uint32_t b = 0; b < numB; b++) {
        int64_t aBase = static_cast<int64_t>(startB + b) * strideA;
        int64_t xBase = static_cast<int64_t>(startB + b) * stridex;
        int64_t yBase = static_cast<int64_t>(startB + b) * stridey;
        if (incx < 0) { xBase += static_cast<int64_t>(dotDim - 1) * incx64; }
        if (incy < 0) { yBase += static_cast<int64_t>(outDim - 1) * incy64; }

        for (uint32_t outIdx = threadIdx.x; outIdx < outDim; outIdx += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t dotIdx = 0; dotIdx < dotDim; dotIdx++) {
                int64_t aIdx = IS_TRANS ? (static_cast<int64_t>(outIdx) * lda64 + static_cast<int64_t>(dotIdx))
                                        : (static_cast<int64_t>(dotIdx) * lda64 + static_cast<int64_t>(outIdx));
                int64_t xIdx = static_cast<int64_t>(dotIdx) * static_cast<int64_t>(incx);
                float av, xv;
                if constexpr (IN_BF16) {
                    av = __bfloat162float(aGm[aBase + aIdx]);
                    xv = __bfloat162float(xGm[xBase + xIdx]);
                } else if constexpr (IN_FP16) {
                    av = __half2float(aGm[aBase + aIdx]);
                    xv = __half2float(xGm[xBase + xIdx]);
                } else {
                    av = static_cast<float>(aGm[aBase + aIdx]);
                    xv = static_cast<float>(xGm[xBase + xIdx]);
                }
                acc += av * xv;
            }
            int64_t yIdx = static_cast<int64_t>(outIdx) * static_cast<int64_t>(incy);
            float oldY;
            if constexpr (OUT_BF16)
                oldY = __bfloat162float(yGm[yBase + yIdx]);
            else if constexpr (OUT_FP16)
                oldY = __half2float(yGm[yBase + yIdx]);
            else
                oldY = static_cast<float>(yGm[yBase + yIdx]);
            float result = alpha * acc + beta * oldY;
            if constexpr (OUT_BF16)      yGm[yBase + yIdx] = __float2bfloat16_rn_sat(result);
            else if constexpr (OUT_FP16) yGm[yBase + yIdx] = __float2half_rn_sat(result);
            else                         yGm[yBase + yIdx] = static_cast<T_OUT>(result);
        }
    }
}

// ============================================================
// Kernel entry helpers
// ============================================================
template <typename T_IN, typename T_OUT>
__aicore__ inline void DispatchNormalStrided(
    const GemvStridedBatchedTilingData& tiling, uint32_t nB, uint32_t sB,
    int32_t lda, int32_t incx, int32_t incy,
    GM_ADDR A, GM_ADDR x, GM_ADDR y)
{
    asc_vf_call<GemvSimtStrided<T_IN, T_OUT, false>>(
        dim3{tiling.numThreads, 1, 1},
        tiling.m, tiling.n, nB, sB, tiling.alpha, tiling.beta, lda, incx, incy,
        tiling.strideA, tiling.stridex, tiling.stridey,
        reinterpret_cast<__gm__ const T_IN *>(A),
        reinterpret_cast<__gm__ const T_IN *>(x),
        reinterpret_cast<__gm__ T_OUT *>(y));
}

template <typename T, bool HSS>
__aicore__ inline void DispatchTransposeStrided(
    const GemvStridedBatchedTilingData& tiling, uint32_t nB, uint32_t sB,
    int32_t lda, int32_t incx, int32_t incy,
    GM_ADDR A, GM_ADDR x, GM_ADDR y)
{
    using TO = typename std::conditional<HSS, float, T>::type;
    constexpr bool IS_FLOAT = IsSameType<T, float>::value;
    constexpr uint32_t AIV_DOTDIM_THRESHOLD = IS_FLOAT ? 256u : 512u;
    if (tiling.incx == 1 && tiling.incy == 1 && tiling.dotSize <= AIV_DOTDIM_THRESHOLD) {
        TPipe pipe;
        GemvStridedBatchedAIV<T, true, HSS> op;
        op.Init(pipe, A, x, y, tiling);
        op.Process();
        return;
    }
    asc_vf_call<GemvSimtStrided<T, TO, true>>(
        dim3{tiling.numThreads, 1, 1},
        tiling.m, tiling.n, nB, sB, tiling.alpha, tiling.beta, lda, incx, incy,
        tiling.strideA, tiling.stridex, tiling.stridey,
        reinterpret_cast<__gm__ const T *>(A),
        reinterpret_cast<__gm__ const T *>(x),
        reinterpret_cast<__gm__ TO *>(y));
}

extern "C" __global__ __aicore__ void gemv_strided_batched(
    GM_ADDR A, GM_ADDR x, GM_ADDR y, GemvStridedBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t sB = GetBlockIdx() * tiling.batchPerCore;
    uint32_t nB = (GetBlockIdx() == tiling.usedCoreNum - 1) ? tiling.batchTail : tiling.batchPerCore;
    if (nB == 0) return;
    int32_t lda = tiling.lda, incx = tiling.incx, incy = tiling.incy;

#define D_NORM(T_IN, T_OUT) DispatchNormalStrided<T_IN, T_OUT>(tiling, nB, sB, lda, incx, incy, A, x, y)
#define D_TRANS(T, HSS)     DispatchTransposeStrided<T, HSS>(tiling, nB, sB, lda, incx, incy, A, x, y)

    if (tiling.trans == 0) {
        if      (tiling.dtype == 1) D_NORM(float,      float);
        else if (tiling.dtype == 0) D_NORM(half,       half);
        else if (tiling.dtype == 2) D_NORM(half,       float);
        else if (tiling.dtype == 3) D_NORM(bfloat16_t, bfloat16_t);
        else                        D_NORM(bfloat16_t, float);
    } else {
        if      (tiling.dtype == 1) D_TRANS(float,      false);
        else if (tiling.dtype == 0) D_TRANS(half,       false);
        else if (tiling.dtype == 2) D_TRANS(half,       true);
        else if (tiling.dtype == 3) D_TRANS(bfloat16_t, false);
        else                        D_TRANS(bfloat16_t, true);
    }

#undef D_NORM
#undef D_TRANS
}

void gemv_strided_batched_kernel_do(
    GM_ADDR A, GM_ADDR x, GM_ADDR y,
    const GemvStridedBatchedTilingData& tiling,
    uint32_t numBlocks, void *stream)
{
    gemv_strided_batched<<<numBlocks, nullptr, stream>>>(A, x, y, tiling);
}
