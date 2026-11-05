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
#include <cstdlib>
#include <type_traits>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "scalex_tiling_data.h"
#include "scalex_kernel.h"

using namespace AscendC;

template<typename XType>
class ScalexAIV {
public:
    __aicore__ inline ScalexAIV() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR alpha, const ScalexTilingData& tdata, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SingleIteration(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline void ProcessFp32(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline void ProcessMixed(uint32_t curOffset, uint32_t dataCount);

    static constexpr uint32_t ELEMENTS_PER_BLOCK = 32 / sizeof(XType);

    TPipe* pipe_;
    GlobalTensor<XType> xGM_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> midBuf_;
    ScalexTilingData tiling_;
    float alphaVal_;
    uint32_t blockIdx_;
    uint32_t myOffset_;
    uint32_t myCount_;
};

template<typename XType>
__aicore__ inline void ScalexAIV<XType>::Init(GM_ADDR x, GM_ADDR alpha,
    const ScalexTilingData& tdata, TPipe* pipe)
{
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    tiling_ = tdata;
    alphaVal_ = tiling_.alphaIsDevice ? *reinterpret_cast<__gm__ float*>(alpha) : tiling_.alpha;

    myOffset_ = blockIdx_ * tiling_.perCoreN;
    myCount_ = tiling_.perCoreN;
    if (blockIdx_ == GetBlockNum() - 1) {
        myCount_ += tiling_.remainder;
    }

    xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(x), tiling_.totalN);

    uint32_t bufSize = tiling_.tileSize * sizeof(XType);
    pipe_->InitBuffer(inQueue_, 1, bufSize);
    pipe_->InitBuffer(outQueue_, 1, bufSize);

    if constexpr (!std::is_same<XType, float>::value) {
        pipe_->InitBuffer(midBuf_, tiling_.tileSize * sizeof(float));
    }
}

template<typename XType>
__aicore__ inline void ScalexAIV<XType>::Process()
{
    if (myCount_ == 0) {
        return;
    }

    uint32_t tileLoop = myCount_ / tiling_.tileSize;
    uint32_t tileTail = myCount_ % tiling_.tileSize;
    uint32_t curOffset = myOffset_;

    for (uint32_t i = 0; i < tileLoop; i++) {
        SingleIteration(curOffset, tiling_.tileSize);
        curOffset += tiling_.tileSize;
    }

    if (tileTail > 0) {
        SingleIteration(curOffset, tileTail);
    }
}

template<typename XType>
__aicore__ inline void ScalexAIV<XType>::SingleIteration(uint32_t curOffset, uint32_t dataCount)
{
    if constexpr (std::is_same<XType, float>::value) {
        ProcessFp32(curOffset, dataCount);
    } else {
        ProcessMixed(curOffset, dataCount);
    }
}

template<typename XType>
__aicore__ inline void ScalexAIV<XType>::ProcessFp32(uint32_t curOffset, uint32_t dataCount)
{
    uint32_t alignedCount = (dataCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    uint32_t tailCount = dataCount - alignedCount;

    LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();

    if (alignedCount > 0) {
        DataCopy(inLocal, xGM_[curOffset], alignedCount);
    }

    if (tailCount > 0) {
        uint8_t paddingNum = ELEMENTS_PER_BLOCK - tailCount;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, paddingNum, 0.0f};
        DataCopyPad(inLocal[alignedCount], xGM_[curOffset + alignedCount], copyParams, padParams);
    }

    inQueue_.EnQue<float>(inLocal);

    LocalTensor<float> vData = inQueue_.DeQue<float>();
    LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();

    Muls(outLocal, vData, alphaVal_, static_cast<int32_t>(dataCount));

    inQueue_.FreeTensor(vData);
    outQueue_.EnQue<float>(outLocal);

    LocalTensor<float> writeData = outQueue_.DeQue<float>();

    if (alignedCount > 0) {
        DataCopy(xGM_[curOffset], writeData, alignedCount);
    }

    if (tailCount > 0) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(float)), 0, 0, 0};
        DataCopyPad(xGM_[curOffset + alignedCount], writeData[alignedCount], copyParams);
    }

    outQueue_.FreeTensor(writeData);
}

template<typename XType>
__aicore__ inline void ScalexAIV<XType>::ProcessMixed(uint32_t curOffset, uint32_t dataCount)
{
    uint32_t alignedCount = (dataCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    uint32_t tailCount = dataCount - alignedCount;

    LocalTensor<XType> inLocal = inQueue_.AllocTensor<XType>();

    if (alignedCount > 0) {
        DataCopy(inLocal, xGM_[curOffset], alignedCount);
    }

    if (tailCount > 0) {
        uint8_t paddingNum = ELEMENTS_PER_BLOCK - tailCount;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(XType)), 0, 0, 0};
        DataCopyPadExtParams<XType> padParams{true, 0, paddingNum, 0};
        DataCopyPad(inLocal[alignedCount], xGM_[curOffset + alignedCount], copyParams, padParams);
    }

    inQueue_.EnQue<XType>(inLocal);

    LocalTensor<XType> vData = inQueue_.DeQue<XType>();

    LocalTensor<float> midLocal = midBuf_.Get<float>();
    Cast(midLocal, vData, RoundMode::CAST_NONE, static_cast<uint32_t>(dataCount));
    inQueue_.FreeTensor(vData);

    Muls(midLocal, midLocal, alphaVal_, static_cast<int32_t>(dataCount));

    LocalTensor<XType> outLocal = outQueue_.AllocTensor<XType>();
    Cast(outLocal, midLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(dataCount));
    outQueue_.EnQue<XType>(outLocal);

    LocalTensor<XType> writeData = outQueue_.DeQue<XType>();

    if (alignedCount > 0) {
        DataCopy(xGM_[curOffset], writeData, alignedCount);
    }

    if (tailCount > 0) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(XType)), 0, 0, 0};
        DataCopyPad(xGM_[curOffset + alignedCount], writeData[alignedCount], copyParams);
    }

    outQueue_.FreeTensor(writeData);
}

// SIMT VF functions for incx != 1 paths (interleaved distribution)

template<typename XType>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void ScalexSimtCompute(
    uint32_t totalN, int64_t incx, uint32_t numBlocks, __gm__ float* alphaGm,
    __gm__ XType* xGm, float alphaHost, uint32_t alphaIsDevice)
{
    uint32_t stride = static_cast<uint32_t>(incx);
    float alpha = alphaIsDevice ? *alphaGm : alphaHost;

    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < totalN; i += blockDim.x * numBlocks) {
        uint32_t idx = i * stride;

        if constexpr (std::is_same<XType, float>::value) {
            xGm[idx] = alpha * xGm[idx];
        } else {
            float val = static_cast<float>(xGm[idx]);
            val = alpha * val;
            xGm[idx] = static_cast<XType>(val);
        }
    }
}

// AIV Kernel entry
__global__ __aicore__ void scalex_aiv_kernel(GM_ADDR x, GM_ADDR alpha, ScalexTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    if (tdata.xType == static_cast<uint32_t>(ACL_FLOAT)) {
        ScalexAIV<float> op;
        op.Init(x, alpha, tdata, &pipe);
        op.Process();
    } else if (tdata.xType == static_cast<uint32_t>(ACL_FLOAT16)) {
        ScalexAIV<half> op;
        op.Init(x, alpha, tdata, &pipe);
        op.Process();
    } else {
        ScalexAIV<bfloat16_t> op;
        op.Init(x, alpha, tdata, &pipe);
        op.Process();
    }
}

// SIMT Kernel entry (interleaved: elements spread across blocks)
__global__ __aicore__ void scalex_simt_kernel(GM_ADDR x, GM_ADDR alpha, ScalexTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tdata.xType == static_cast<uint32_t>(ACL_FLOAT)) {
        asc_vf_call<ScalexSimtCompute<float>>(dim3{tdata.nthreads, 1, 1},
            tdata.totalN, tdata.incx, tdata.numBlocks,
            reinterpret_cast<__gm__ float*>(alpha),
            reinterpret_cast<__gm__ float*>(x),
            tdata.alpha, tdata.alphaIsDevice);
    } else if (tdata.xType == static_cast<uint32_t>(ACL_FLOAT16)) {
        asc_vf_call<ScalexSimtCompute<half>>(dim3{tdata.nthreads, 1, 1},
            tdata.totalN, tdata.incx, tdata.numBlocks,
            reinterpret_cast<__gm__ float*>(alpha),
            reinterpret_cast<__gm__ half*>(x),
            tdata.alpha, tdata.alphaIsDevice);
    } else {
        asc_vf_call<ScalexSimtCompute<bfloat16_t>>(dim3{tdata.nthreads, 1, 1},
            tdata.totalN, tdata.incx, tdata.numBlocks,
            reinterpret_cast<__gm__ float*>(alpha),
            reinterpret_cast<__gm__ bfloat16_t*>(x),
            tdata.alpha, tdata.alphaIsDevice);
    }
}

void scalex_kernel_do(uint8_t* x, void* alpha, const ScalexTilingData& tiling,
                      uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    GM_ADDR xGm = x;
    GM_ADDR alphaGm = static_cast<uint8_t*>(alpha);

    if (tiling.incx == 1) {
        scalex_aiv_kernel<<<numBlocks, nullptr, aclStream>>>(xGm, alphaGm, tiling);
    } else {
        scalex_simt_kernel<<<numBlocks, nullptr, aclStream>>>(xGm, alphaGm, tiling);
    }
}
