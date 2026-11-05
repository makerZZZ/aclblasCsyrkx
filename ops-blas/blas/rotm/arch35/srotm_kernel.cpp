/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file srotm_kernel.cpp
 * \brief Modified Givens rotation host implementation
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "srotm_tiling_data.h"

using namespace AscendC;

constexpr uint32_t ALIGN_BYTES = 32;

// ==========================================================================
// SIMT VF path — grid-stride for non-unit / mixed-sign strides
// ==========================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SrotmSimt(
    uint32_t n, int32_t incx, int32_t incy, int64_t kx, int64_t ky,
    float alpha1, float beta1, float alpha2, float beta2,
    __gm__ float* xGm, __gm__ float* yGm)
{
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += gridDim.x * blockDim.x) {
        int64_t xi = kx + static_cast<int64_t>(idx) * incx;
        int64_t yi = ky + static_cast<int64_t>(idx) * incy;
        float xVal = xGm[xi];
        float yVal = yGm[yi];
        xGm[xi] = alpha1 * xVal + beta1 * yVal;
        yGm[yi] = alpha2 * xVal + beta2 * yVal;
    }
}

// ==========================================================================
// SIMD path — for unit stride
// ==========================================================================
template <typename T>
class SrotmKernel {
public:
    __aicore__ inline SrotmKernel() {}
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR x, GM_ADDR y, const SrotmTilingData& t);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count);
    __aicore__ inline void Compute(uint32_t count);
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count);

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> outXQueue;
    TQue<QuePosition::VECOUT, 1> outYQueue;

    GlobalTensor<T> xGM;
    GlobalTensor<T> yGM;
    T alpha1, beta1, alpha2, beta2;

    uint32_t startOffset;
    uint32_t calCount;
    uint32_t tileSize;
    uint32_t tileNum;
    uint32_t remainderCount;
};

template <typename T>
__aicore__ inline void SrotmKernel<T>::Init(TPipe* pipe, GM_ADDR x, GM_ADDR y, const SrotmTilingData& t)
{
    pipe_ = pipe;
    int32_t elementCount = t.elementCount;
    alpha1 = t.alpha1; beta1 = t.beta1;
    alpha2 = t.alpha2; beta2 = t.beta2;

    uint32_t blockNum = GetBlockNum();
    if (blockNum == 0) blockNum = 1;
    uint32_t blockIdx = GetBlockIdx();

    int32_t perCore = elementCount / static_cast<int32_t>(blockNum);
    int32_t remain = elementCount % static_cast<int32_t>(blockNum);
    startOffset = blockIdx * static_cast<uint32_t>(perCore);
    if (blockIdx < static_cast<uint32_t>(remain)) {
        startOffset += blockIdx;
        perCore += 1;
    } else {
        startOffset += static_cast<uint32_t>(remain);
    }
    calCount = (perCore > 0) ? static_cast<uint32_t>(perCore) : 0;
    if (calCount == 0) return;

    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    yGM.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));

    uint32_t ubPerQueue = UB_SIZE / (4 * BUFFER_NUM);
    tileSize = ubPerQueue / sizeof(T);
    tileNum = calCount / tileSize;
    remainderCount = calCount % tileSize;

    pipe_->InitBuffer(xQueue, BUFFER_NUM, ubPerQueue);
    pipe_->InitBuffer(yQueue, BUFFER_NUM, ubPerQueue);
    pipe_->InitBuffer(outXQueue, BUFFER_NUM, ubPerQueue);
    pipe_->InitBuffer(outYQueue, BUFFER_NUM, ubPerQueue);
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::CopyIn(uint32_t offset, uint32_t count)
{
    uint32_t nbytes = count * sizeof(T);
    DataCopyExtParams ext{1, nbytes, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    LocalTensor<T> xLocal = xQueue.AllocTensor<T>();
    DataCopyPad(xLocal, xGM[startOffset + offset], ext, padParams);
    xQueue.EnQue(xLocal);

    LocalTensor<T> yLocal = yQueue.AllocTensor<T>();
    DataCopyPad(yLocal, yGM[startOffset + offset], ext, padParams);
    yQueue.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::Compute(uint32_t count)
{
    LocalTensor<T> xLocal = xQueue.DeQue<T>();
    LocalTensor<T> yLocal = yQueue.DeQue<T>();

    LocalTensor<T> outXLocal = outXQueue.AllocTensor<T>();
    LocalTensor<T> outYLocal = outYQueue.AllocTensor<T>();

    Muls(outXLocal, xLocal, alpha1, count);
    Axpy(outXLocal, yLocal, beta1, count);
    Muls(outYLocal, xLocal, alpha2, count);
    Axpy(outYLocal, yLocal, beta2, count);

    outXQueue.EnQue(outXLocal);
    outYQueue.EnQue(outYLocal);
    xQueue.FreeTensor(xLocal);
    yQueue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::CopyOut(uint32_t offset, uint32_t count)
{
    uint32_t nbytes = count * sizeof(T);
    DataCopyExtParams ext{1, nbytes, 0, 0, 0};

    LocalTensor<T> outXLocal = outXQueue.DeQue<T>();
    DataCopyPad(xGM[startOffset + offset], outXLocal, ext);
    outXQueue.FreeTensor(outXLocal);

    LocalTensor<T> outYLocal = outYQueue.DeQue<T>();
    DataCopyPad(yGM[startOffset + offset], outYLocal, ext);
    outYQueue.FreeTensor(outYLocal);
}

template <typename T>
__aicore__ inline void SrotmKernel<T>::Process()
{
    if (calCount == 0) return;

    for (uint32_t i = 0; i < tileNum; i++) {
        uint32_t offset = i * tileSize;
        CopyIn(offset, tileSize);
        Compute(tileSize);
        CopyOut(offset, tileSize);
    }
    if (remainderCount > 0) {
        uint32_t offset = tileNum * tileSize;
        CopyIn(offset, remainderCount);
        Compute(remainderCount);
        CopyOut(offset, remainderCount);
    }
}

__global__ __aicore__ void srotm_kernel(GM_ADDR x, GM_ADDR y, SrotmTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if ((t.incx == 1 && t.incy == 1) || (t.incx == -1 && t.incy == -1)) {
        TPipe pipe;
        SrotmKernel<float> op;
        op.Init(&pipe, x, y, t);
        op.Process();
    } else {
        asc_vf_call<SrotmSimt>(
            dim3{t.numThreads, 1, 1},
            static_cast<uint32_t>(t.elementCount), t.incx, t.incy, t.kx, t.ky,
            t.alpha1, t.beta1, t.alpha2, t.beta2,
            reinterpret_cast<__gm__ float*>(x), reinterpret_cast<__gm__ float*>(y));
    }
}

void srotm_kernel_do_arch35(GM_ADDR x, GM_ADDR y, const SrotmTilingData& tilingData,
                     uint32_t numBlocks, void *stream)
{
    srotm_kernel<<<numBlocks, nullptr, stream>>>(x, y, tilingData);
}
