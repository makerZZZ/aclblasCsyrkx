/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace CgemvBatched {

constexpr uint8_t BUFFER_NUM = 2;
constexpr uint32_t TMPBUF_NUM = 3;
constexpr uint32_t BYTENUM_BLOCK = 32;
constexpr uint32_t BYTENUM_REPEAT = 256;

constexpr uint32_t COMPLEX_ELENUM = 2;
constexpr uint32_t MAX_CORE_CNT = 40;
constexpr uint32_t ELENUM_LINE_ALIGNED = 32;  // max elements in line

constexpr uint32_t ELENUM_BLOCK_FP32 = 8;
constexpr uint32_t ELENUM_REPEAT_FP32 = 64;
constexpr uint32_t ELENUM_BLOCK_FP16 = 16;
constexpr uint32_t ELENUM_REPEAT_FP16 = 128;

constexpr uint32_t OP_CONJ_TRANS = 2;  // N:0, T:1, C:2

template <typename T, const bool TRANS_T>
class CgemvBatchedAIV {
public:
    __aicore__ inline CgemvBatchedAIV(){};
    __aicore__ inline void Init(
        GM_ADDR A, GM_ADDR x, GM_ADDR y, GM_ADDR mask, uint32_t transVal, GM_ADDR workSpace, GM_ADDR tilingGm);
    __aicore__ inline void Process();

    __aicore__ inline void InitUbuf();
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline void CopyInGatherMask();
    __aicore__ inline void CopyInMatAndVec(uint32_t loopId, uint32_t curCalColNum, uint32_t pingFlag);
    __aicore__ inline void MatAMulVecx(uint32_t curCalMatNum, uint32_t pingFlag);
    __aicore__ inline void PrepareRealImag(uint32_t curCalMatNum, uint32_t pingFlag);
    __aicore__ inline void PrepareRealImagForFloat(uint16_t matRepeatNum, uint16_t vecRepeatNum, uint32_t pingFlag);
    __aicore__ inline void PrepareRealImagForHalf(
        uint32_t curCalMatNum, uint16_t matRepeatNum, uint16_t vecRepeatNum, uint32_t pingFlag);
    __aicore__ inline void MatAMulVecxByLineNormal(uint32_t curCalMatNum);
    __aicore__ inline void ReduceSumMatAByLineNormal(uint32_t curCalMatNum);
    __aicore__ inline void MatAMulVecxByLineTrans(uint32_t curCalMatNum);
    __aicore__ inline void ReduceSumMatAByLineTrans(uint32_t curCalMatNum);
    __aicore__ inline void GatherRealImagToComplex(uint32_t curCalMatNum, uint32_t pingFlag);
    __aicore__ inline void CopyOutRslt(uint32_t loopId, uint32_t curCalMatNum, uint32_t pingFlag);

private:
    static constexpr bool IS_FLOAT = IsSameType<T, float>::value;  // half or float
    static constexpr bool IS_TRANS = TRANS_T;

    TPipe pipe;

    GlobalTensor<T> inAGM;
    GlobalTensor<T> inxGM;
    GlobalTensor<uint32_t> maskGM;
    GlobalTensor<T> outGM;
    GlobalTensor<float> workGM;

    TBuf<QuePosition::VECCALC> inABuf;
    TBuf<QuePosition::VECCALC> inxBuf;
    TBuf<QuePosition::VECCALC> outBuf;

    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> matTmpBuf;
    TBuf<QuePosition::VECCALC> vecTmpBuf;
    TBuf<QuePosition::VECCALC> matPreBuf;
    TBuf<QuePosition::VECCALC> vecPreBuf;

    LocalTensor<uint32_t> maskLocal;
    LocalTensor<float> matTmpLocal;
    LocalTensor<float> vecTmpLocal;
    LocalTensor<half> matPreLocal;
    LocalTensor<half> vecPreLocal;

    LocalTensor<float> matRealLocal;
    LocalTensor<float> matImagLocal;
    LocalTensor<float> matMulVecRealLocal;
    LocalTensor<float> matMulVecImagLocal;
    LocalTensor<float> vecRealLocal;
    LocalTensor<float> vecImagLocal;

    uint32_t m = 0;  // row num
    uint32_t n = 0;  // column num

    uint32_t calMatNum = 0;
    uint32_t startMatId = 0;

    uint32_t maxMatNum = 0;

    uint32_t maxMatEleNum = 0;  // complex number
    uint32_t maxVecEleNum = 0;  // complex number

    int32_t vecIdx;
    uint32_t transOpt = 0;

    uint32_t eleNumPerBlock = 8;
    uint32_t eleNumPerRepeat = 64;
};

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::Init(
    GM_ADDR A, GM_ADDR x, GM_ADDR y, GM_ADDR mask, uint32_t transVal, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    this->vecIdx = GetBlockIdx();
    this->transOpt = transVal;

    ParseTilingData(tilingGm);

    this->inAGM.SetGlobalBuffer((__gm__ T *)A);
    this->inxGM.SetGlobalBuffer((__gm__ T *)x);
    this->maskGM.SetGlobalBuffer((__gm__ uint32_t *)mask);
    this->outGM.SetGlobalBuffer((__gm__ T *)y);

    this->workGM.SetGlobalBuffer((__gm__ float *)workSpace);
    InitUbuf();
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::InitUbuf()
{
    this->maxMatEleNum = this->maxMatNum * this->m * ELENUM_LINE_ALIGNED;
    this->maxVecEleNum = this->maxMatNum * ELENUM_LINE_ALIGNED;

    // reserve 256bytes at the end of memory
    // use for input and output
    this->pipe.InitBuffer(
        this->inABuf, BUFFER_NUM * (this->maxMatEleNum * sizeof(T) + BYTENUM_REPEAT) * COMPLEX_ELENUM);
    this->pipe.InitBuffer(
        this->inxBuf, BUFFER_NUM * (this->maxVecEleNum * sizeof(T) + BYTENUM_REPEAT) * COMPLEX_ELENUM);
    this->pipe.InitBuffer(
        this->outBuf, BUFFER_NUM * (this->maxVecEleNum * sizeof(T) + BYTENUM_REPEAT) * COMPLEX_ELENUM);
    this->pipe.InitBuffer(this->maskBuf, this->maxVecEleNum * COMPLEX_ELENUM * sizeof(uint32_t));

    // use for saving temp result
    uint32_t vecTmpBufSize = 0;
    if constexpr (IS_TRANS) {
        vecTmpBufSize = this->maxVecEleNum * COMPLEX_ELENUM * BYTENUM_BLOCK;
    } else {
        vecTmpBufSize = (this->maxVecEleNum * sizeof(float) + BYTENUM_REPEAT) * COMPLEX_ELENUM;
    }
    this->pipe.InitBuffer(
        this->matTmpBuf, TMPBUF_NUM * (this->maxMatEleNum * sizeof(float) + BYTENUM_REPEAT) * COMPLEX_ELENUM);
    this->pipe.InitBuffer(this->vecTmpBuf, vecTmpBufSize);

    if constexpr (!IS_FLOAT) {
        uint32_t vecPreBufSize = 0;
        if constexpr (IS_TRANS) {
            vecPreBufSize = (this->maxVecEleNum * sizeof(float) + BYTENUM_REPEAT) * COMPLEX_ELENUM;
        } else {
            vecPreBufSize = (this->maxVecEleNum * sizeof(half) + BYTENUM_REPEAT) * COMPLEX_ELENUM;
        }

        this->pipe.InitBuffer(this->matPreBuf, (this->maxMatEleNum * sizeof(half) + BYTENUM_REPEAT) * COMPLEX_ELENUM);
        this->pipe.InitBuffer(this->vecPreBuf, vecPreBufSize);

        this->matPreLocal = matPreBuf.Get<half>();
        this->vecPreLocal = vecPreBuf.Get<half>();
    }
    this->matTmpLocal = matTmpBuf.Get<float>();
    this->vecTmpLocal = vecTmpBuf.Get<float>();
    this->maskLocal = maskBuf.Get<uint32_t>();

    this->matRealLocal = this->matTmpLocal[0];
    this->matImagLocal = this->matTmpLocal[this->maxMatEleNum + ELENUM_REPEAT_FP32];
    this->matMulVecRealLocal = this->matTmpLocal[2 * (this->maxMatEleNum + ELENUM_REPEAT_FP32)];
    this->matMulVecImagLocal = this->matTmpLocal[3 * (this->maxMatEleNum + ELENUM_REPEAT_FP32)];
    this->vecRealLocal = this->vecTmpLocal[0];
    this->vecImagLocal = this->vecTmpLocal[this->maxVecEleNum + ELENUM_REPEAT_FP32];
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::ParseTilingData(GM_ADDR tilingGm)
{
    // skip dtype, isTrans
    uint32_t locId = 2;
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);
    this->m = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t)));
    locId++;
    this->n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t)));
    locId++;
    this->maxMatNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t)));
    locId++;
    this->calMatNum = (*(
        __gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t) + sizeof(uint32_t) * this->vecIdx));
    locId += MAX_CORE_CNT;
    this->startMatId = (*(
        __gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t) + sizeof(uint32_t) * this->vecIdx));
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::CopyInGatherMask()
{
    uint32_t maskNum = this->maxMatNum * ELENUM_LINE_ALIGNED * COMPLEX_ELENUM;
    DataCopy(this->maskLocal, this->maskGM, maskNum);
    PipeBarrier<PIPE_ALL>();
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::CopyInMatAndVec(
    uint32_t loopId, uint32_t curCalMatNum, uint32_t pingFlag)
{
    LocalTensor<T> inALocal =
        inABuf.Get<T>()[pingFlag * COMPLEX_ELENUM * (this->maxMatEleNum + BYTENUM_REPEAT / sizeof(T))];
    LocalTensor<T> inxLocal =
        inxBuf.Get<T>()[pingFlag * COMPLEX_ELENUM * (this->maxVecEleNum + BYTENUM_REPEAT / sizeof(T))];

    uint64_t curStartMatId = this->startMatId + loopId * this->maxMatNum;
    uint64_t inAOffset = curStartMatId * this->m * this->n * COMPLEX_ELENUM;

    uint16_t matBlockCnt = curCalMatNum * this->m;
    uint32_t matBlockLen = this->n * COMPLEX_ELENUM * sizeof(T);
    uint32_t matDstStride = (ELENUM_LINE_ALIGNED - this->n) * COMPLEX_ELENUM * sizeof(T) / BYTENUM_BLOCK;

    uint16_t vecBlockCnt = 0;
    uint32_t vecBlockLen = 0;
    uint32_t vecDstStride = 0;
    uint64_t inxOffset = 0;

    if constexpr (IS_TRANS) {
        vecBlockCnt = 1;
        vecBlockLen = curCalMatNum * this->m * COMPLEX_ELENUM * sizeof(T);
        vecDstStride = 0;
        inxOffset = curStartMatId * this->m * COMPLEX_ELENUM;
    } else {
        vecBlockCnt = curCalMatNum;
        vecBlockLen = matBlockLen;
        vecDstStride = matDstStride;
        inxOffset = curStartMatId * this->n * COMPLEX_ELENUM;
    }

    DataCopyExtParams copyMatParams{matBlockCnt, matBlockLen, 0, matDstStride, 0};
    DataCopyExtParams copyVecParams{vecBlockCnt, vecBlockLen, 0, vecDstStride, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    // copy in matrix A and vec x
    DataCopyPad(inALocal, this->inAGM[inAOffset], copyMatParams, padParams);
    DataCopyPad(inxLocal, this->inxGM[inxOffset], copyVecParams, padParams);
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::PrepareRealImag(uint32_t curCalMatNum, uint32_t pingFlag)
{
    uint16_t matRepeatNum =
        (curCalMatNum * this->m * ELENUM_LINE_ALIGNED * COMPLEX_ELENUM * sizeof(T) - 1) / BYTENUM_REPEAT + 1;
    uint16_t vecRepeatNum = 0;
    if constexpr (IS_TRANS) {
        vecRepeatNum = (curCalMatNum * this->m * COMPLEX_ELENUM - 1) / ELENUM_BLOCK_FP32 + 1;
    } else {
        vecRepeatNum = (curCalMatNum * ELENUM_LINE_ALIGNED * COMPLEX_ELENUM * sizeof(T) - 1) / BYTENUM_REPEAT + 1;
    }

    if constexpr (IS_FLOAT) {
        PrepareRealImagForFloat(matRepeatNum, vecRepeatNum, pingFlag);
    } else {
        PrepareRealImagForHalf(curCalMatNum, matRepeatNum, vecRepeatNum, pingFlag);
    }
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::PrepareRealImagForFloat(
    uint16_t matRepeatNum, uint16_t vecRepeatNum, uint32_t pingFlag)
{
    uint64_t rsvdCnt = 0;

    LocalTensor<float> inALocal =
        inABuf.Get<float>()[pingFlag * COMPLEX_ELENUM * (this->maxMatEleNum + ELENUM_REPEAT_FP32)];
    LocalTensor<float> inxLocal =
        inxBuf.Get<float>()[pingFlag * COMPLEX_ELENUM * (this->maxVecEleNum + ELENUM_REPEAT_FP32)];

    GatherMask(this->matRealLocal, inALocal, 1, false, 0, {1, matRepeatNum, 8, 8}, rsvdCnt);
    GatherMask(this->matImagLocal, inALocal, 2, false, 0, {1, matRepeatNum, 8, 8}, rsvdCnt);

    if constexpr (IS_TRANS) {
        Brcb(this->vecTmpLocal, inxLocal, vecRepeatNum, {1, 8});
    } else {
        GatherMask(this->vecRealLocal, inxLocal, 1, false, 0, {1, vecRepeatNum, 8, 8}, rsvdCnt);
        GatherMask(this->vecImagLocal, inxLocal, 2, false, 0, {1, vecRepeatNum, 8, 8}, rsvdCnt);
    }

    PipeBarrier<PIPE_V>();
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::PrepareRealImagForHalf(
    uint32_t curCalMatNum, uint16_t matRepeatNum, uint16_t vecRepeatNum, uint32_t pingFlag)
{
    LocalTensor<half> inALocal =
        inABuf.Get<half>()[pingFlag * COMPLEX_ELENUM * (this->maxMatEleNum + ELENUM_REPEAT_FP16)];
    LocalTensor<half> inxLocal =
        inxBuf.Get<half>()[pingFlag * COMPLEX_ELENUM * (this->maxVecEleNum + ELENUM_REPEAT_FP16)];
    LocalTensor<half> matPreRealLocal = this->matPreLocal[0];
    LocalTensor<half> matPreImagLocal = this->matPreLocal[this->maxMatEleNum + ELENUM_REPEAT_FP16];

    uint8_t matCastRepeatNum = (curCalMatNum * this->m * ELENUM_LINE_ALIGNED - 1) / ELENUM_REPEAT_FP16 + 1;
    uint8_t vecCastRepeatNum = 0;
    uint64_t rsvdCnt = 0;

    GatherMask(matPreRealLocal, inALocal, 1, false, 0, {1, matRepeatNum, 8, 8}, rsvdCnt);
    GatherMask(matPreImagLocal, inALocal, 2, false, 0, {1, matRepeatNum, 8, 8}, rsvdCnt);

    if constexpr (IS_TRANS) {
        LocalTensor<float> vecCastLocal = vecPreBuf.Get<float>();
        vecCastRepeatNum = (curCalMatNum * this->m * COMPLEX_ELENUM - 1) / ELENUM_REPEAT_FP16 + 1;
        Cast(vecCastLocal, inxLocal, RoundMode::CAST_NONE, vecCastRepeatNum * ELENUM_REPEAT_FP16);
        PipeBarrier<PIPE_V>();

        Brcb(this->vecTmpLocal, vecCastLocal, vecRepeatNum, {1, 8});
    } else {
        LocalTensor<half> vecPreRealLocal = this->vecPreLocal[0];
        LocalTensor<half> vecPreImagLocal = this->vecPreLocal[this->maxVecEleNum + ELENUM_REPEAT_FP16];
        vecCastRepeatNum = (curCalMatNum * ELENUM_LINE_ALIGNED - 1) / ELENUM_REPEAT_FP16 + 1;

        GatherMask(vecPreRealLocal, inxLocal, 1, false, 0, {1, vecRepeatNum, 8, 8}, rsvdCnt);
        GatherMask(vecPreImagLocal, inxLocal, 2, false, 0, {1, vecRepeatNum, 8, 8}, rsvdCnt);
        PipeBarrier<PIPE_V>();

        Cast(this->vecRealLocal, vecPreRealLocal, RoundMode::CAST_NONE, vecCastRepeatNum * ELENUM_REPEAT_FP16);
        Cast(this->vecImagLocal, vecPreImagLocal, RoundMode::CAST_NONE, vecCastRepeatNum * ELENUM_REPEAT_FP16);
    }
    Cast(this->matRealLocal, matPreRealLocal, RoundMode::CAST_NONE, matCastRepeatNum * ELENUM_REPEAT_FP16);
    Cast(this->matImagLocal, matPreImagLocal, RoundMode::CAST_NONE, matCastRepeatNum * ELENUM_REPEAT_FP16);

    PipeBarrier<PIPE_V>();
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::MatAMulVecxByLineNormal(uint32_t curCalMatNum)
{
    LocalTensor<float> tmpRRLocal = this->matMulVecRealLocal;
    LocalTensor<float> tmpIILocal = this->matTmpLocal[4 * (this->maxMatEleNum + ELENUM_REPEAT_FP32)];
    LocalTensor<float> tmpRILocal = this->matMulVecImagLocal;
    LocalTensor<float> tmpIRLocal = this->matTmpLocal[5 * (this->maxMatEleNum + ELENUM_REPEAT_FP32)];

    if (curCalMatNum > this->m) {
        uint8_t dstRepStride = this->m * ELENUM_LINE_ALIGNED / ELENUM_BLOCK_FP32;
        uint8_t src0RepStride = dstRepStride;
        uint8_t src1RepStride = ELENUM_LINE_ALIGNED / ELENUM_BLOCK_FP32;
        for (int32_t i = 0; i < this->m; i++) {
            uint32_t offset = i * ELENUM_LINE_ALIGNED;
            Mul(tmpRRLocal[offset],
                this->matRealLocal[offset],
                this->vecRealLocal,
                this->n,
                curCalMatNum,
                {1, 1, 1, dstRepStride, src0RepStride, src1RepStride});
            Mul(tmpIILocal[offset],
                this->matImagLocal[offset],
                this->vecImagLocal,
                this->n,
                curCalMatNum,
                {1, 1, 1, dstRepStride, src0RepStride, src1RepStride});
            Mul(tmpRILocal[offset],
                this->matRealLocal[offset],
                this->vecImagLocal,
                this->n,
                curCalMatNum,
                {1, 1, 1, dstRepStride, src0RepStride, src1RepStride});
            Mul(tmpIRLocal[offset],
                this->matImagLocal[offset],
                this->vecRealLocal,
                this->n,
                curCalMatNum,
                {1, 1, 1, dstRepStride, src0RepStride, src1RepStride});
        }
    } else {
        uint8_t dstRepStride = ELENUM_LINE_ALIGNED / ELENUM_BLOCK_FP32;
        uint8_t src0RepStride = dstRepStride;
        uint8_t src1RepStride = 0;
        for (int32_t i = 0; i < curCalMatNum; i++) {
            uint32_t matOffset = i * this->m * ELENUM_LINE_ALIGNED;
            uint32_t vecOffset = i * ELENUM_LINE_ALIGNED;
            Mul(tmpRRLocal[matOffset],
                this->matRealLocal[matOffset],
                this->vecRealLocal[vecOffset],
                this->n,
                this->m,
                {1, 1, 1, dstRepStride, src0RepStride, src1RepStride});
            Mul(tmpIILocal[matOffset],
                this->matImagLocal[matOffset],
                this->vecImagLocal[vecOffset],
                this->n,
                this->m,
                {1, 1, 1, dstRepStride, src0RepStride, src1RepStride});
            Mul(tmpRILocal[matOffset],
                this->matRealLocal[matOffset],
                this->vecImagLocal[vecOffset],
                this->n,
                this->m,
                {1, 1, 1, dstRepStride, src0RepStride, src1RepStride});
            Mul(tmpIRLocal[matOffset],
                this->matImagLocal[matOffset],
                this->vecRealLocal[vecOffset],
                this->n,
                this->m,
                {1, 1, 1, dstRepStride, src0RepStride, src1RepStride});
        }
    }
    PipeBarrier<PIPE_V>();

    uint8_t repeatNum = (curCalMatNum * this->m * ELENUM_LINE_ALIGNED - 1) / ELENUM_REPEAT_FP32 + 1;
    uint32_t calCount = repeatNum * ELENUM_REPEAT_FP32;
    Sub(this->matMulVecRealLocal, tmpRRLocal, tmpIILocal, calCount);
    Add(this->matMulVecImagLocal, tmpRILocal, tmpIRLocal, calCount);
    PipeBarrier<PIPE_V>();
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::ReduceSumMatAByLineNormal(uint32_t curCalMatNum)
{
    int32_t repeatNum = curCalMatNum * this->m;
    int32_t srcRepStride = ELENUM_LINE_ALIGNED / ELENUM_BLOCK_FP32;
    WholeReduceSum(this->vecRealLocal, this->matMulVecRealLocal, this->n, repeatNum, 1, 1, srcRepStride);
    WholeReduceSum(this->vecImagLocal, this->matMulVecImagLocal, this->n, repeatNum, 1, 1, srcRepStride);
    PipeBarrier<PIPE_V>();
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::MatAMulVecxByLineTrans(uint32_t curCalMatNum)
{
    LocalTensor<float> tmpRRLocal = this->matMulVecRealLocal;
    LocalTensor<float> tmpIILocal = this->matTmpLocal[4 * (this->maxMatEleNum + ELENUM_REPEAT_FP32)];
    LocalTensor<float> tmpRILocal = this->matMulVecImagLocal;
    LocalTensor<float> tmpIRLocal = this->matTmpLocal[5 * (this->maxMatEleNum + ELENUM_REPEAT_FP32)];

    int32_t dupCnt = COMPLEX_ELENUM * (this->maxVecEleNum + ELENUM_REPEAT_FP32);
    uint8_t repeatNum = (curCalMatNum * this->m * ELENUM_LINE_ALIGNED - 1) / ELENUM_REPEAT_FP32 + 1;
    uint32_t calCount = repeatNum * ELENUM_REPEAT_FP32;

    uint8_t mulRepeatNum = curCalMatNum * this->m;  // max 255
    uint8_t dstRepStride = ELENUM_LINE_ALIGNED / ELENUM_BLOCK_FP32;
    uint8_t src0RepStride = dstRepStride;
    uint8_t src1RepStride = COMPLEX_ELENUM;
    uint32_t offset = ELENUM_BLOCK_FP32;

    Mul(tmpRRLocal,
        this->matRealLocal,
        this->vecTmpLocal,
        this->n,
        mulRepeatNum,
        {1, 1, 0, dstRepStride, src0RepStride, src1RepStride});
    Mul(tmpIILocal,
        this->matImagLocal,
        this->vecTmpLocal[offset],
        this->n,
        mulRepeatNum,
        {1, 1, 0, dstRepStride, src0RepStride, src1RepStride});
    Mul(tmpRILocal,
        this->matRealLocal,
        this->vecTmpLocal[offset],
        this->n,
        mulRepeatNum,
        {1, 1, 0, dstRepStride, src0RepStride, src1RepStride});
    Mul(tmpIRLocal,
        this->matImagLocal,
        this->vecTmpLocal,
        this->n,
        mulRepeatNum,
        {1, 1, 0, dstRepStride, src0RepStride, src1RepStride});
    PipeBarrier<PIPE_V>();

    Duplicate(this->vecTmpLocal, 0.0f, dupCnt);

    if (OP_CONJ_TRANS == this->transOpt) {
        // conj trans
        Add(this->matMulVecRealLocal, tmpRRLocal, tmpIILocal, calCount);
        Sub(this->matMulVecImagLocal, tmpRILocal, tmpIRLocal, calCount);
    } else {
        // trans
        Sub(this->matMulVecRealLocal, tmpRRLocal, tmpIILocal, calCount);
        Add(this->matMulVecImagLocal, tmpRILocal, tmpIRLocal, calCount);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::ReduceSumMatAByLineTrans(uint32_t curCalMatNum)
{
    uint8_t dstRepStride = ELENUM_LINE_ALIGNED / ELENUM_BLOCK_FP32;
    uint8_t src0RepStride = dstRepStride;
    uint8_t src1RepStride = this->m * ELENUM_LINE_ALIGNED / ELENUM_BLOCK_FP32;
    for (int32_t i = 0; i < this->m; i++) {
        uint32_t offset = i * ELENUM_LINE_ALIGNED;
        Add(this->vecRealLocal,
            this->vecRealLocal,
            this->matMulVecRealLocal[offset],
            this->n,
            curCalMatNum,
            {1, 1, 1, dstRepStride, src0RepStride, src1RepStride});
        Add(this->vecImagLocal,
            this->vecImagLocal,
            this->matMulVecImagLocal[offset],
            this->n,
            curCalMatNum,
            {1, 1, 1, dstRepStride, src0RepStride, src1RepStride});
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::GatherRealImagToComplex(uint32_t curCalMatNum, uint32_t pingFlag)
{
    int32_t calCount = 0;
    if constexpr (IS_TRANS) {
        calCount = curCalMatNum * ELENUM_LINE_ALIGNED * COMPLEX_ELENUM;
    } else {
        uint8_t blockNum = (curCalMatNum * this->m * COMPLEX_ELENUM * sizeof(T) - 1) / BYTENUM_BLOCK + 1;
        calCount = blockNum * BYTENUM_BLOCK / sizeof(T);
    }

    if constexpr (IS_FLOAT) {
        LocalTensor<float> rsltLocal =
            outBuf.Get<float>()[pingFlag * COMPLEX_ELENUM * (this->maxVecEleNum + ELENUM_REPEAT_FP32)];
        Gather(rsltLocal, this->vecTmpLocal, this->maskLocal, (uint32_t)0, calCount);
    } else {
        LocalTensor<half> rsltLocal =
            outBuf.Get<half>()[pingFlag * COMPLEX_ELENUM * (this->maxVecEleNum + ELENUM_REPEAT_FP16)];

        LocalTensor<half> vecPreRealLocal = this->vecPreLocal[0];
        LocalTensor<half> vecPreImagLocal = this->vecPreLocal[this->maxVecEleNum + ELENUM_REPEAT_FP16];
        uint8_t castRepeatNum = 0;
        if constexpr (IS_TRANS) {
            castRepeatNum = (curCalMatNum * ELENUM_LINE_ALIGNED - 1) / ELENUM_REPEAT_FP16 + 1;
        } else {
            castRepeatNum = (curCalMatNum * this->m - 1) / ELENUM_REPEAT_FP16 + 1;
        }

        Cast(vecPreRealLocal, this->vecRealLocal, RoundMode::CAST_NONE, castRepeatNum * ELENUM_REPEAT_FP16);
        Cast(vecPreImagLocal, this->vecImagLocal, RoundMode::CAST_NONE, castRepeatNum * ELENUM_REPEAT_FP16);
        PipeBarrier<PIPE_V>();

        Gather(rsltLocal, this->vecPreLocal, this->maskLocal, (uint32_t)0, calCount);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::MatAMulVecx(uint32_t curCalMatNum, uint32_t pingFlag)
{
    PrepareRealImag(curCalMatNum, pingFlag);
    if constexpr (IS_TRANS) {
        MatAMulVecxByLineTrans(curCalMatNum);
        ReduceSumMatAByLineTrans(curCalMatNum);
    } else {
        MatAMulVecxByLineNormal(curCalMatNum);
        ReduceSumMatAByLineNormal(curCalMatNum);
    }
    GatherRealImagToComplex(curCalMatNum, pingFlag);
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::CopyOutRslt(
    uint32_t loopId, uint32_t curCalMatNum, uint32_t pingFlag)
{
    uint16_t blockCnt = 0;
    uint32_t blockLen = 0;
    uint32_t srcStride = 0;
    uint64_t outOffset = 0;
    uint64_t curStartMatId = this->startMatId + loopId * this->maxMatNum;

    LocalTensor<T> outLocal =
        outBuf.Get<T>()[pingFlag * COMPLEX_ELENUM * (this->maxVecEleNum + BYTENUM_REPEAT / sizeof(T))];
    if constexpr (IS_TRANS) {
        blockCnt = curCalMatNum;
        blockLen = this->n * COMPLEX_ELENUM * sizeof(T);
        srcStride = (ELENUM_LINE_ALIGNED - this->n) * COMPLEX_ELENUM * sizeof(T) / BYTENUM_BLOCK;
        outOffset = curStartMatId * this->n * COMPLEX_ELENUM;
    } else {
        blockCnt = 1;
        blockLen = curCalMatNum * this->m * COMPLEX_ELENUM * sizeof(T);
        srcStride = 0;
        outOffset = curStartMatId * this->m * COMPLEX_ELENUM;
    }
    DataCopyExtParams copyParams{blockCnt, blockLen, srcStride, 0, 0};
    DataCopyPad(this->outGM[outOffset], outLocal, copyParams);
}

template <typename T, const bool TRANS_T>
__aicore__ inline void CgemvBatchedAIV<T, TRANS_T>::Process()
{
    if (0 == this->calMatNum)
        return;

    // copy in gather mask
    CopyInGatherMask();

    uint32_t loopNum = this->calMatNum / this->maxMatNum;
    uint32_t tailNum = this->calMatNum % this->maxMatNum;

    if (tailNum > 0) {
        loopNum++;
    }
    uint32_t pingFlag = 1;

    for (uint32_t loopId = 0; loopId < loopNum; loopId++) {
        uint32_t curCalMatNum = this->maxMatNum;
        if ((loopId == loopNum - 1) && (tailNum > 0))
            curCalMatNum = tailNum;

        if (loopId == 0) {
            CopyInMatAndVec(0, curCalMatNum, pingFlag);
        }
        auto eventId = pingFlag ? EVENT_ID0 : EVENT_ID1;

        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        uint32_t nextLoopId = loopId + 1;
        if (nextLoopId < loopNum) {
            uint32_t nextCalMatNum = this->maxMatNum;
            if ((nextLoopId == loopNum - 1) && (tailNum > 0))
                nextCalMatNum = tailNum;

            CopyInMatAndVec(nextLoopId, nextCalMatNum, 1 - pingFlag);
        }
        MatAMulVecx(curCalMatNum, pingFlag);

        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        CopyOutRslt(loopId, curCalMatNum, pingFlag);

        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
        pingFlag = 1 - pingFlag;
    }
}
}  // namespace CgemvBatched

