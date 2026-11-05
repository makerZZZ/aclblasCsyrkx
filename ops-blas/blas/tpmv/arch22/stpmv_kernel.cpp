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
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "stpmv_tiling_data.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t UB_BYTENUM_PER_REPEAT = 256;

template <typename T>
class TpmvAIV {
public:
    __aicore__ inline TpmvAIV() = default;
    __aicore__ inline void Init(GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, const TpmvTilingData& tiling);
    __aicore__ inline void Process();

private:
    TPipe pipe;

    __aicore__ inline bool IsFormulaA()
    {
        return (
            (uplo == ACLBLAS_LOWER && trans == ACLBLAS_OP_N) ||
            (uplo == ACLBLAS_UPPER && (trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C)));
    }

    __aicore__ inline uint64_t PackedIndex(uint32_t row, uint32_t col)
    {
        if (IsFormulaA()) {
            return static_cast<uint64_t>(row) * (row + 1ULL) / 2ULL + col;
        } else {
            return static_cast<uint64_t>(col) * (col + 1ULL) / 2ULL + row;
        }
    }

    __aicore__ inline uint32_t XPhysicalPos(uint32_t logical)
    {
        return (incx >= 0) ? (logical * absIncx) : ((n - 1U - logical) * absIncx);
    }

    __aicore__ inline uint32_t YPhysicalPos(uint32_t logical) { return XPhysicalPos(logical); }

    __aicore__ inline void LoadX(LocalTensor<T>& xLocal, uint32_t colOffset, uint32_t dataCount)
    {
        if (dataCount == 0) {
            return;
        }
        if (incx == 1) {
            if (dataCount % static_cast<uint32_t>(elementsPerBlock) == 0) {
                DataCopy(xLocal, xGM[colOffset], dataCount);
            } else {
                uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
                DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
                DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
                DataCopyPad(xLocal, xGM[colOffset], copyParams, padParams);
            }
        } else {
            for (uint32_t i = 0; i < dataCount; i++) {
                uint32_t pos = XPhysicalPos(colOffset + i);
                xLocal.SetValue(i, xGM.GetValue(pos));
            }
        }
    }

    __aicore__ inline void CopyIn(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void CopyInPad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void ComputePad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);
    __aicore__ inline void CopyOutPad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void CopyInStrided(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount);

    __aicore__ inline void ProcessFast();
    __aicore__ inline void ProcessStrided();

    GlobalTensor<T> aGM;
    GlobalTensor<T> xGM;
    GlobalTensor<T> yGM;

    TQue<QuePosition::VECIN, BUFFER_NUM> aQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;
    TBuf<TPosition::VECCALC> tmpBuf;

    uint32_t vecIdx = 0;
    uint32_t n = 0;
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
__aicore__ inline void TpmvAIV<T>::Init(GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, const TpmvTilingData& tiling)
{
    vecIdx = GetBlockIdx();

    n = tiling.n;
    useCoreNum = tiling.useCoreNum;
    incx = tiling.incx;
    uplo = tiling.uplo;
    trans = tiling.trans;
    diag = tiling.diag;

    absIncx = (incx >= 0) ? static_cast<uint32_t>(incx) : static_cast<uint32_t>(-incx);

    if (useCoreNum == 0 || useCoreNum > TPMV_MAX_CORE_NUM) {
        useCoreNum = 1;
    }

    uint32_t physicalSize = (n - 1U) * absIncx + 1U;
    xGM.SetGlobalBuffer((__gm__ T*)x, physicalSize);
    yGM.SetGlobalBuffer((__gm__ T*)y, physicalSize);
    aGM.SetGlobalBuffer((__gm__ T*)aPacked, static_cast<uint64_t>(this->n) * this->n);

    maxDataCount = 30 * 1024 / BYTENUM_PER_FLOAT32;

    pipe.InitBuffer(aQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(xQueue, BUFFER_NUM, maxDataCount * sizeof(T));
    pipe.InitBuffer(yQueue, BUFFER_NUM, maxDataCount * sizeof(T));

    int tmpCount = (maxDataCount / elementsPerRepeat + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
    pipe.InitBuffer(tmpBuf, tmpCount * sizeof(T));
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::CopyIn(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    uint64_t r = PackedIndex(rowOffset, colOffset);
    DataCopy(LocalA, aGM[r], dataCount);
    aQueue.EnQue<T>(LocalA);
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::Compute(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.DeQue<T>();
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalY = yQueue.AllocTensor<T>();

    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);

    if (diag == ACLBLAS_UNIT && colOffset <= rowOffset && rowOffset < colOffset + dataCount) {
        uint32_t diagIdx = rowOffset - colOffset;
        LocalA.SetValue(diagIdx, (T)1.0f);
    }

    LocalTensor<T> tmpLocal = tmpBuf.Get<T>();
    Mul(LocalY, LocalA, LocalX, dataCount);
    ReduceSum(LocalY, LocalY, tmpLocal, dataCount);

    yQueue.EnQue<T>(LocalY);
    aQueue.FreeTensor(LocalA);
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::CopyOut(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    DataCopyExtParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};

    LocalTensor<T> yLocal = yQueue.DeQue<T>();
    uint32_t yPos = YPhysicalPos(rowOffset);
    DataCopyPad(yGM[yPos], yLocal, copyParams);
    yQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::CopyInPad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    uint64_t r = PackedIndex(rowOffset, colOffset);
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    DataCopyPad(LocalA, aGM[r], copyParams, padParams);
    aQueue.EnQue<T>(LocalA);
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::ComputePad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.DeQue<T>();
    LocalTensor<T> LocalX = xQueue.DeQue<T>();
    LocalTensor<T> LocalY = yQueue.AllocTensor<T>();

    int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);

    if (diag == ACLBLAS_UNIT && colOffset <= rowOffset && rowOffset < colOffset + dataCount) {
        uint32_t diagIdx = rowOffset - colOffset;
        LocalA.SetValue(diagIdx, (T)1.0f);
    }

    LocalTensor<T> tmpLocal = tmpBuf.Get<T>();
    Mul(LocalY, LocalA, LocalX, dataCount);
    ReduceSum(LocalY, LocalY, tmpLocal, dataCount);

    yQueue.EnQue<T>(LocalY);
    aQueue.FreeTensor(LocalA);
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::CopyOutPad(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, static_cast<uint16_t>(sizeof(T)), 0, 0, 0};

    LocalTensor<T> yLocal = yQueue.DeQue<T>();
    uint32_t yPos = YPhysicalPos(rowOffset);
    DataCopyPad(yGM[yPos], yLocal, copyParams);
    yQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::CopyInStrided(uint32_t rowOffset, uint32_t colOffset, uint32_t dataCount)
{
    LocalTensor<T> LocalA = aQueue.AllocTensor<T>();
    for (uint32_t i = 0; i < dataCount; i++) {
        uint32_t col = colOffset + i;
        uint64_t index = PackedIndex(rowOffset, col);
        LocalA.SetValue(i, aGM.GetValue(index));
    }
    aQueue.EnQue<T>(LocalA);
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::ProcessFast()
{
    for (uint32_t col = 0; col < n; col += maxDataCount) {
        uint32_t colBatchSize = (col + maxDataCount <= n) ? maxDataCount : (n - col);
        if (colBatchSize == 0) {
            continue;
        }

        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        LoadX(xLocal, col, colBatchSize);

        for (uint32_t row = vecIdx; row < n; row += useCoreNum) {
            if (col > row) {
                continue;
            }
            uint32_t dataCount = (col + colBatchSize <= row + 1U) ? colBatchSize : (row + 1U - col);
            if (dataCount == 0) {
                continue;
            }

            xQueue.EnQue<T>(xLocal);
            bool needsPad = (dataCount < maxDataCount);
            if (needsPad) {
                CopyInPad(row, col, dataCount);
                ComputePad(row, col, dataCount);
                CopyOutPad(row, col, dataCount);
            } else {
                CopyIn(row, col, dataCount);
                Compute(row, col, dataCount);
                CopyOut(row, col, dataCount);
            }
        }
        xQueue.FreeTensor(xLocal);
    }
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::ProcessStrided()
{
    bool isFormulaA = IsFormulaA();

    for (uint32_t col = 0; col < n; col += maxDataCount) {
        uint32_t colBatchSize = (col + maxDataCount <= n) ? maxDataCount : (n - col);
        if (colBatchSize == 0) {
            continue;
        }

        LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
        for (uint32_t row = vecIdx; row < n; row += useCoreNum) {
            uint32_t overlapStart = 0;
            uint32_t dataCount = 0;

            if (isFormulaA) {
                if (col > row) {
                    continue;
                }
                overlapStart = col;
                dataCount = (col + colBatchSize <= row + 1U) ? colBatchSize : (row + 1U - col);
            } else {
                if (col + colBatchSize <= row) {
                    continue;
                }
                overlapStart = (col >= row) ? col : row;
                dataCount = col + colBatchSize - overlapStart;
            }

            if (dataCount == 0) {
                continue;
            }

            LoadX(xLocal, overlapStart, dataCount);
            xQueue.EnQue<T>(xLocal);

            CopyInStrided(row, overlapStart, dataCount);
            Compute(row, overlapStart, dataCount);
            CopyOut(row, overlapStart, dataCount);
        }
        xQueue.FreeTensor(xLocal);
    }
}

template <typename T>
__aicore__ inline void TpmvAIV<T>::Process()
{
    bool isFormulaA = IsFormulaA();
    SetAtomicAdd<T>();
    if (vecIdx >= useCoreNum) {
        SetAtomicNone();
        return;
    }

    if (isFormulaA && (incx == 1)) {
        ProcessFast();
    } else {
        ProcessStrided();
    }
    SetAtomicNone();
}

__global__ __aicore__ void stpmv_kernel(GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, TpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TpmvAIV<float> op;
    op.Init(aPacked, x, y, tiling);
    op.Process();
}

void stpmv_kernel_do(
    GM_ADDR aPacked, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, const TpmvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    stpmv_kernel<<<numBlocks, nullptr, stream>>>(aPacked, x, y, workSpace, tiling);
}
