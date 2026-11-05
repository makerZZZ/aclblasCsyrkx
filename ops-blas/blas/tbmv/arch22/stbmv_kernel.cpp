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
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "stbmv_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t UB_BYTENUM_PER_REPEAT = 256;

template <typename T>
class TbmvAIV {
public:
    __aicore__ inline TbmvAIV() = default;
    __aicore__ inline void Init(GM_ADDR aBanded, GM_ADDR x, GM_ADDR y, const TbmvTilingData& tiling);
    __aicore__ inline void Process();

private:
    TPipe pipe;

    __aicore__ inline void CopyIn(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void CopyInPad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutPad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void ProcessFast();
    __aicore__ inline void ProcessGeneralBand(uint32_t bandIdx);
    __aicore__ inline void ProcessGeneralBandCol(
        uint32_t bandIdx, uint32_t col, uint32_t colBatchSize, uint32_t aRowBase);
    __aicore__ inline void LoadXCol(LocalTensor<T>& xLocal, uint32_t col, uint32_t count);
    __aicore__ inline void CopyOutYCol(LocalTensor<T>& yLocal, uint32_t yStart, uint32_t count, uint32_t bandIdx);

    __aicore__ inline uint32_t XPhysicalPos(uint32_t logical)
    {
        return (incx >= 0) ? (logical * absIncx) : ((n - 1U - logical) * absIncx);
    }
    __aicore__ inline uint32_t YPhysicalPos(uint32_t logical) { return XPhysicalPos(logical); }

    GlobalTensor<T> aGM;
    GlobalTensor<T> xGM;
    GlobalTensor<T> yGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> aQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;

    uint32_t vecIdx = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t lda = 0;
    uint32_t useCoreNum = 0;
    uint32_t maxDataCount = 0;

    int64_t incx = 1;
    uint32_t absIncx = 1;
    uint32_t uplo = ACLBLAS_LOWER;
    uint32_t trans = ACLBLAS_OP_N;
    uint32_t diag = ACLBLAS_NON_UNIT;

    int elementsPerRepeat = UB_BYTENUM_PER_REPEAT / BYTENUM_PER_FLOAT32;
    int elementsPerBlock = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;
};

template <typename T>
__aicore__ inline void TbmvAIV<T>::Init(GM_ADDR aBanded, GM_ADDR x, GM_ADDR y, const TbmvTilingData& tiling)
{
    vecIdx = GetBlockIdx();

    n = tiling.n;
    k = tiling.k;
    lda = tiling.lda;
    useCoreNum = tiling.useCoreNum;
    incx = tiling.incx;
    uplo = tiling.uplo;
    trans = tiling.trans;
    diag = tiling.diag;

    if (useCoreNum == 0 || useCoreNum > TBMV_MAX_CORE_NUM) {
        useCoreNum = 1;
    }

    absIncx = (incx >= 0) ? static_cast<uint32_t>(incx) : static_cast<uint32_t>(-incx);

    xGM.SetGlobalBuffer((__gm__ T*)x, (n - 1U) * absIncx + 1U);
    yGM.SetGlobalBuffer((__gm__ T*)y, (n - 1U) * absIncx + 1U);
    aGM.SetGlobalBuffer((__gm__ T*)aBanded, static_cast<uint64_t>(this->k + 1U) * this->lda);

    maxDataCount = 30 * 1024 / BYTENUM_PER_FLOAT32;

    pipe.InitBuffer(aQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(xQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(yQueue, BUFFER_NUM, maxDataCount * sizeof(T));
}

template <typename T>
__aicore__ inline void TbmvAIV<T>::CopyIn(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    uint64_t r = static_cast<uint64_t>(rowOffset) * lda + colOffset;
    DataCopy(LocalA, aGM[r], dataCount);
    aQueue.EnQue<T>(LocalA);
}

template <typename T>
__aicore__ inline void TbmvAIV<T>::Compute(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.DeQue<T>();
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalY = yQueue.AllocTensor<T>();

    int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    if (diag == ACLBLAS_UNIT && rowOffset == 0) {
        for (uint32_t i = 0; i < dataCount; i++) {
            LocalA.SetValue(i, static_cast<T>(1.0f));
        }
    }
    Mul(LocalY, LocalA, LocalX, dataCount);

    yQueue.EnQue<T>(LocalY);
    aQueue.FreeTensor(LocalA);
}

template <typename T>
__aicore__ inline void TbmvAIV<T>::CopyOut(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> yLocal = yQueue.DeQue<T>();
    if (incx == 1) {
        DataCopy(yGM[colOffset + rowOffset], yLocal, dataCount);
    } else {
        for (uint32_t i = 0; i < dataCount; i++) {
            uint32_t pos = XPhysicalPos(colOffset + rowOffset + i);
            yGM.SetValue(pos, yGM.GetValue(pos) + yLocal.GetValue(i));
        }
    }
    yQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void TbmvAIV<T>::CopyInPad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    uint64_t r = static_cast<uint64_t>(rowOffset) * lda + colOffset;
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    DataCopyPad(LocalA, aGM[r], copyParams, padParams);
    aQueue.EnQue<T>(LocalA);
}

template <typename T>
__aicore__ inline void TbmvAIV<T>::CopyOutPad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> yLocal = yQueue.DeQue<T>();
    if (incx == 1) {
        uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
        DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
        DataCopyPad(yGM[colOffset + rowOffset], yLocal, copyParams);
    } else {
        uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
        DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
        for (uint32_t i = 0; i < dataCount; i++) {
            uint32_t pos = XPhysicalPos(colOffset + rowOffset + i);
            yGM.SetValue(pos, yGM.GetValue(pos) + yLocal.GetValue(i));
        }
    }
    yQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void TbmvAIV<T>::ProcessFast()
{
    for (uint32_t col = 0; col < n; col += maxDataCount) {
        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        uint32_t colBatchSize = (col + maxDataCount <= n) ? maxDataCount : (n - col);
        LoadXCol(xLocal, col, colBatchSize);

        for (uint32_t bandIdx = vecIdx; bandIdx <= k; bandIdx += useCoreNum) {
            uint32_t bandLen = n - bandIdx;
            if (col >= bandLen) {
                continue;
            }
            uint32_t dataCount = maxDataCount;
            if (col + dataCount > bandLen) {
                dataCount = bandLen - col;
                xQueue.EnQue<T>(xLocal);
                CopyInPad(bandIdx, col, dataCount);
                Compute(bandIdx, col, dataCount);
                CopyOutPad(bandIdx, col, dataCount);
                continue;
            }
            xQueue.EnQue<T>(xLocal);
            CopyIn(bandIdx, col, dataCount);
            Compute(bandIdx, col, dataCount);
            CopyOut(bandIdx, col, dataCount);
        }
        xQueue.FreeTensor(xLocal);
    }
}

template <typename T>
__aicore__ inline void TbmvAIV<T>::LoadXCol(LocalTensor<T>& xLocal, uint32_t col, uint32_t count)
{
    if (incx == 1 && count % static_cast<uint32_t>(elementsPerBlock) == 0) {
        DataCopy(xLocal, xGM[col], count);
    } else if (incx == 1) {
        uint8_t paddingNum = elementsPerBlock - count % elementsPerBlock;
        DataCopyExtParams copyParams{1, count * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
        DataCopyPad(xLocal, xGM[col], copyParams, padParams);
    } else {
        for (uint32_t i = 0; i < count; i++) {
            xLocal.SetValue(i, xGM.GetValue(XPhysicalPos(col + i)));
        }
    }
}

template <typename T>
__aicore__ inline void TbmvAIV<T>::CopyOutYCol(
    LocalTensor<T>& yLocal, uint32_t yStart, uint32_t count, uint32_t bandIdx)
{
    (void)bandIdx;
    if (incx == 1 && count % static_cast<uint32_t>(elementsPerBlock) == 0) {
        DataCopy(yGM[yStart], yLocal, count);
    } else if (incx == 1) {
        uint8_t paddingNum = elementsPerBlock - count % elementsPerBlock;
        DataCopyExtParams copyParams{1, count * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
        DataCopyPad(yGM[yStart], yLocal, copyParams);
    } else {
        for (uint32_t i = 0; i < count; i++) {
            uint32_t pos = XPhysicalPos(yStart + i);
            yGM.SetValue(pos, yGM.GetValue(pos) + yLocal.GetValue(i));
        }
    }
}

template <typename T>
__aicore__ inline void TbmvAIV<T>::ProcessGeneralBandCol(
    uint32_t bandIdx, uint32_t col, uint32_t colBatchSize, uint32_t aRowBase)
{
    bool isLwrN = (uplo == ACLBLAS_LOWER) && (trans == ACLBLAS_OP_N);
    bool isUprT = (uplo == ACLBLAS_UPPER) && ((trans == ACLBLAS_OP_T) || (trans == ACLBLAS_OP_C));
    bool isLwrT = (uplo == ACLBLAS_LOWER) && ((trans == ACLBLAS_OP_T) || (trans == ACLBLAS_OP_C));

    uint32_t aColBase = col;
    if (isLwrT) {
        aColBase = col - bandIdx;
    } else if (isUprT) {
        aColBase = col + bandIdx;
    }
    uint32_t aOffset = aRowBase * lda + aColBase;

    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    LoadXCol(xLocal, col, colBatchSize);
    xQueue.EnQue<T>(xLocal);

    if (colBatchSize % static_cast<uint32_t>(elementsPerBlock) == 0) {
        LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
        DataCopy(LocalA, aGM[aOffset], colBatchSize);
        aQueue.EnQue<T>(LocalA);
    } else {
        uint8_t paddingNum = elementsPerBlock - colBatchSize % elementsPerBlock;
        DataCopyExtParams copyParams{1, colBatchSize * BYTENUM_PER_FLOAT32, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
        LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
        DataCopyPad(LocalA, aGM[aOffset], copyParams, padParams);
        aQueue.EnQue<T>(LocalA);
    }

    {
        LocalTensor<T> LocalA = aQueue.DeQue<T>();
        LocalTensor<T> LocalX = xQueue.DeQue<T>();
        LocalTensor<T> LocalY = yQueue.AllocTensor<T>();

        int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        if (diag == ACLBLAS_UNIT && bandIdx == 0) {
            for (uint32_t i = 0; i < colBatchSize; i++) {
                LocalA.SetValue(i, static_cast<T>(1.0f));
            }
        }
        Mul(LocalY, LocalA, LocalX, colBatchSize);
        yQueue.EnQue<T>(LocalY);
        aQueue.FreeTensor(LocalA);
    }

    {
        uint32_t yRowBaseVal = 0;
        if (isLwrN || isUprT) {
            yRowBaseVal = bandIdx;
        }
        uint32_t yStart = col + yRowBaseVal;
        if (!isLwrN && !isUprT) {
            yStart = col - bandIdx;
        }

        LocalTensor<T> yLocal = yQueue.DeQue<T>();
        CopyOutYCol(yLocal, yStart, colBatchSize, bandIdx);
        yQueue.FreeTensor(yLocal);
    }

    xQueue.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void TbmvAIV<T>::ProcessGeneralBand(uint32_t bandIdx)
{
    uint32_t firstCol = 0;
    uint32_t bandLen = n;
    uint32_t aRowBase = 0;

    if ((uplo == ACLBLAS_LOWER) && (trans == ACLBLAS_OP_N)) {
        bandLen = n - bandIdx;
        aRowBase = bandIdx;
    } else if ((uplo == ACLBLAS_LOWER) && ((trans == ACLBLAS_OP_T) || (trans == ACLBLAS_OP_C))) {
        firstCol = bandIdx;
        aRowBase = bandIdx;
    } else if ((uplo == ACLBLAS_UPPER) && (trans == ACLBLAS_OP_N)) {
        firstCol = bandIdx;
        aRowBase = k - bandIdx;
    } else {
        bandLen = n - bandIdx;
        aRowBase = k - bandIdx;
    }

    for (uint32_t col = firstCol; col < bandLen; col += maxDataCount) {
        uint32_t colBatchSize = (col + maxDataCount <= bandLen) ? maxDataCount : (bandLen - col);
        if (colBatchSize == 0) {
            continue;
        }
        ProcessGeneralBandCol(bandIdx, col, colBatchSize, aRowBase);
    }
}

template <typename T>
__aicore__ inline void TbmvAIV<T>::Process()
{
    SetAtomicAdd<T>();
    if (vecIdx >= useCoreNum) {
        SetAtomicNone();
        return;
    }

    if ((uplo == ACLBLAS_LOWER) && (trans == ACLBLAS_OP_N)) {
        ProcessFast();
    } else {
        for (uint32_t bandIdx = vecIdx; bandIdx <= k; bandIdx += useCoreNum) {
            ProcessGeneralBand(bandIdx);
        }
    }
    SetAtomicNone();
}

__global__ __aicore__ void stbmv_kernel(GM_ADDR aBanded, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, TbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TbmvAIV<float> op;
    op.Init(aBanded, x, y, tiling);
    op.Process();
}

void stbmv_kernel_do(
    GM_ADDR aBanded, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const TbmvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    stbmv_kernel<<<numBlocks, nullptr, stream>>>(aBanded, x, y, workSpace, tiling);
}
