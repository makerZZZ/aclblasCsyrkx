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
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "axpy_ex_tiling_data.h"
#include "axpy_ex_kernel.h"

using namespace AscendC;

#ifndef __NPU_HOST__
static constexpr Reg::CastTrait CAST_B16_TO_B32 = {
    Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

static constexpr Reg::CastTrait CAST_B32_TO_B16 = {
    Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
#endif

template <typename XType>
__simd_vf__ inline void AxpyMixedVf(__ubuf__ XType* xAddr, __ubuf__ XType* yAddr, float a, uint32_t n, uint16_t loopNum)
{
    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    Reg::RegTensor<XType> vx16, vy16, vz16;
    Reg::RegTensor<float> vx32, vy32, vt;
    Reg::MaskReg mask;
    uint32_t count = n;

    for (uint16_t i = 0; i < loopNum; ++i) {
        mask = Reg::UpdateMask<float>(count);
        Reg::LoadAlign<XType, Reg::LoadDist::DIST_UNPACK_B16>(vx16, xAddr + i * VL);
        Reg::LoadAlign<XType, Reg::LoadDist::DIST_UNPACK_B16>(vy16, yAddr + i * VL);
        Reg::Cast<float, XType, CAST_B16_TO_B32>(vx32, vx16, mask);
        Reg::Cast<float, XType, CAST_B16_TO_B32>(vy32, vy16, mask);
        Reg::Muls(vt, vx32, a, mask);
        Reg::Add(vt, vt, vy32, mask);
        Reg::Cast<XType, float, CAST_B32_TO_B16>(vz16, vt, mask);
        Reg::StoreAlign<XType, Reg::StoreDist::DIST_PACK_B32>(yAddr + i * VL, vz16, mask);
    }
}

template <typename XType>
class AxpyExAIV {
public:
    __aicore__ inline AxpyExAIV() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR alpha, const AxpyExTilingData& tdata, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SingleIteration(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline void ProcessFp32(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline void ProcessMixed(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline void CopyInAligned(uint32_t curOffset, uint32_t alignedCount);
    __aicore__ inline void CopyInTail(uint32_t curOffset, uint32_t alignedCount, uint32_t tailCount);
    __aicore__ inline void CopyOutAligned(uint32_t curOffset, uint32_t alignedCount);
    __aicore__ inline void CopyOutTail(uint32_t curOffset, uint32_t alignedCount, uint32_t tailCount);

    static constexpr uint32_t ELEMENTS_PER_BLOCK = 32 / sizeof(XType);

    TPipe* pipe_;
    GlobalTensor<XType> xGM_;
    GlobalTensor<XType> yGM_;
    TBuf<TPosition::VECIN> xBuf_;
    TBuf<TPosition::VECIN> yBuf_;
    AxpyExTilingData tiling_;
    float alphaVal_;
    uint32_t blockIdx_;
    uint32_t myOffset_;
    uint32_t myCount_;
};

template <typename XType>
__aicore__ inline void AxpyExAIV<XType>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR alpha, const AxpyExTilingData& tdata, TPipe* pipe)
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
    yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(y), tiling_.totalN);

    pipe_->InitBuffer(xBuf_, tiling_.tileSize * sizeof(XType));
    pipe_->InitBuffer(yBuf_, tiling_.tileSize * sizeof(XType));
}

template <typename XType>
__aicore__ inline void AxpyExAIV<XType>::Process()
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

template <typename XType>
__aicore__ inline void AxpyExAIV<XType>::SingleIteration(uint32_t curOffset, uint32_t dataCount)
{
    if constexpr (std::is_same<XType, float>::value) {
        ProcessFp32(curOffset, dataCount);
    } else {
        ProcessMixed(curOffset, dataCount);
    }
}

template <typename XType>
__aicore__ inline void AxpyExAIV<XType>::CopyInAligned(uint32_t curOffset, uint32_t alignedCount)
{
    if (alignedCount == 0) {
        return;
    }
    LocalTensor<XType> xLocal = xBuf_.Get<XType>();
    LocalTensor<XType> yLocal = yBuf_.Get<XType>();
    DataCopy(xLocal, xGM_[curOffset], alignedCount);
    DataCopy(yLocal, yGM_[curOffset], alignedCount);
}

template <typename XType>
__aicore__ inline void AxpyExAIV<XType>::CopyInTail(uint32_t curOffset, uint32_t alignedCount, uint32_t tailCount)
{
    if (tailCount == 0) {
        return;
    }
    LocalTensor<XType> xLocal = xBuf_.Get<XType>();
    LocalTensor<XType> yLocal = yBuf_.Get<XType>();
    uint8_t paddingNum = static_cast<uint8_t>(ELEMENTS_PER_BLOCK - tailCount);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(XType)), 0, 0, 0};
    DataCopyPadExtParams<XType> padParams{true, 0, paddingNum, 0};
    DataCopyPad(xLocal[alignedCount], xGM_[curOffset + alignedCount], copyParams, padParams);
    DataCopyPad(yLocal[alignedCount], yGM_[curOffset + alignedCount], copyParams, padParams);
}

template <typename XType>
__aicore__ inline void AxpyExAIV<XType>::CopyOutAligned(uint32_t curOffset, uint32_t alignedCount)
{
    if (alignedCount == 0) {
        return;
    }
    LocalTensor<XType> yLocal = yBuf_.Get<XType>();
    DataCopy(yGM_[curOffset], yLocal, alignedCount);
}

template <typename XType>
__aicore__ inline void AxpyExAIV<XType>::CopyOutTail(uint32_t curOffset, uint32_t alignedCount, uint32_t tailCount)
{
    if (tailCount == 0) {
        return;
    }
    LocalTensor<XType> yLocal = yBuf_.Get<XType>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(tailCount * sizeof(XType)), 0, 0, 0};
    DataCopyPad(yGM_[curOffset + alignedCount], yLocal[alignedCount], copyParams);
}

template <typename XType>
__aicore__ inline void AxpyExAIV<XType>::ProcessFp32(uint32_t curOffset, uint32_t dataCount)
{
    uint32_t alignedCount = (dataCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    uint32_t tailCount = dataCount - alignedCount;

    CopyInAligned(curOffset, alignedCount);
    CopyInTail(curOffset, alignedCount, tailCount);

    PipeBarrier<PIPE_MTE2>();

    LocalTensor<float> xLocal = xBuf_.Get<float>();
    LocalTensor<float> yLocal = yBuf_.Get<float>();
    uint32_t axpyCount = alignedCount + (tailCount > 0 ? ELEMENTS_PER_BLOCK : 0);
    Axpy(yLocal, xLocal, alphaVal_, static_cast<int32_t>(axpyCount));

    PipeBarrier<PIPE_V>();

    CopyOutAligned(curOffset, alignedCount);
    CopyOutTail(curOffset, alignedCount, tailCount);

    PipeBarrier<PIPE_ALL>();
}

template <typename XType>
__aicore__ inline void AxpyExAIV<XType>::ProcessMixed(uint32_t curOffset, uint32_t dataCount)
{
    uint32_t alignedCount = (dataCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    uint32_t tailCount = dataCount - alignedCount;

    CopyInAligned(curOffset, alignedCount);
    CopyInTail(curOffset, alignedCount, tailCount);

    PipeBarrier<PIPE_MTE2>();

    LocalTensor<XType> xLocal = xBuf_.Get<XType>();
    LocalTensor<XType> yLocal = yBuf_.Get<XType>();
    __ubuf__ XType* xAddr = (__ubuf__ XType*)xLocal.GetPhyAddr();
    __ubuf__ XType* yAddr = (__ubuf__ XType*)yLocal.GetPhyAddr();

    constexpr uint32_t VL = VECTOR_REG_WIDTH / sizeof(float);
    uint16_t loopNum = static_cast<uint16_t>((dataCount + VL - 1) / VL);
    asc_vf_call<AxpyMixedVf<XType>>(xAddr, yAddr, alphaVal_, dataCount, loopNum);

    PipeBarrier<PIPE_V>();

    CopyOutAligned(curOffset, alignedCount);
    CopyOutTail(curOffset, alignedCount, tailCount);

    PipeBarrier<PIPE_ALL>();
}

template <typename XType>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void AxpyExSimtCompute(
    uint32_t totalN, int64_t startOffsetX, int64_t startOffsetY, int64_t incx, int64_t incy, uint32_t numBlocks,
    __gm__ float* alphaGm, float alphaHost, uint32_t alphaIsDevice, __gm__ XType* xGm, __gm__ XType* yGm)
{
    float alpha = alphaIsDevice ? *alphaGm : alphaHost;

    uint32_t perBlock = totalN / numBlocks;
    uint32_t blockRemainder = totalN % numBlocks;
    uint32_t myBlockCount = perBlock + (blockIdx.x < blockRemainder ? 1u : 0u);
    uint32_t myBlockStart = blockIdx.x * perBlock + (blockIdx.x < blockRemainder ? blockIdx.x : blockRemainder);

    uint32_t myEnd = myBlockStart + myBlockCount;
    for (uint32_t k = myBlockStart + threadIdx.x; k < myEnd; k += blockDim.x) {
        int64_t xIdx = startOffsetX + static_cast<int64_t>(k) * incx;
        int64_t yIdx = startOffsetY + static_cast<int64_t>(k) * incy;

        if constexpr (std::is_same<XType, float>::value) {
            yGm[yIdx] = alpha * xGm[xIdx] + yGm[yIdx];
        } else {
            float xVal = static_cast<float>(xGm[xIdx]);
            float yVal = static_cast<float>(yGm[yIdx]);
            yVal = alpha * xVal + yVal;
            yGm[yIdx] = static_cast<XType>(yVal);
        }
    }
}

extern "C" __global__ __aicore__ void axpy_ex_aiv_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR alpha, AxpyExTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    if (tdata.xType == static_cast<uint32_t>(ACL_FLOAT)) {
        AxpyExAIV<float> op;
        op.Init(x, y, alpha, tdata, &pipe);
        op.Process();
    } else if (tdata.xType == static_cast<uint32_t>(ACL_FLOAT16)) {
        AxpyExAIV<half> op;
        op.Init(x, y, alpha, tdata, &pipe);
        op.Process();
    } else {
        AxpyExAIV<bfloat16_t> op;
        op.Init(x, y, alpha, tdata, &pipe);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void axpy_ex_simt_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR alpha, AxpyExTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tdata.xType == static_cast<uint32_t>(ACL_FLOAT)) {
        asc_vf_call<AxpyExSimtCompute<float>>(
            dim3{tdata.nthreads, 1, 1}, tdata.totalN, tdata.startOffsetX, tdata.startOffsetY, tdata.incx, tdata.incy,
            tdata.numBlocks, reinterpret_cast<__gm__ float*>(alpha), tdata.alpha, tdata.alphaIsDevice,
            reinterpret_cast<__gm__ float*>(x), reinterpret_cast<__gm__ float*>(y));
    } else if (tdata.xType == static_cast<uint32_t>(ACL_FLOAT16)) {
        asc_vf_call<AxpyExSimtCompute<half>>(
            dim3{tdata.nthreads, 1, 1}, tdata.totalN, tdata.startOffsetX, tdata.startOffsetY, tdata.incx, tdata.incy,
            tdata.numBlocks, reinterpret_cast<__gm__ float*>(alpha), tdata.alpha, tdata.alphaIsDevice,
            reinterpret_cast<__gm__ half*>(x), reinterpret_cast<__gm__ half*>(y));
    } else {
        asc_vf_call<AxpyExSimtCompute<bfloat16_t>>(
            dim3{tdata.nthreads, 1, 1}, tdata.totalN, tdata.startOffsetX, tdata.startOffsetY, tdata.incx, tdata.incy,
            tdata.numBlocks, reinterpret_cast<__gm__ float*>(alpha), tdata.alpha, tdata.alphaIsDevice,
            reinterpret_cast<__gm__ bfloat16_t*>(x), reinterpret_cast<__gm__ bfloat16_t*>(y));
    }
}

void axpy_ex_kernel_do(
    uint8_t* x, uint8_t* y, void* alpha, const AxpyExTilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    GM_ADDR xGm = x;
    GM_ADDR yGm = y;
    GM_ADDR alphaGm = static_cast<uint8_t*>(alpha);

    if (tiling.incx == 1 && tiling.incy == 1) {
        axpy_ex_aiv_kernel<<<numBlocks, nullptr, aclStream>>>(xGm, yGm, alphaGm, tiling);
    } else {
        axpy_ex_simt_kernel<<<numBlocks, nullptr, aclStream>>>(xGm, yGm, alphaGm, tiling);
    }
}
