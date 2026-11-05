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
#include <cfloat>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "adv_api/math/is_nan.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "scnrm2_tiling_data.h"

using namespace AscendC;

namespace {

constexpr uint32_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint32_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint32_t ELEMENTS_PER_BLOCK = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;
constexpr uint32_t REDUCE_REPEAT_BYTES = 256;
constexpr uint32_t ELEMENTS_PER_REPEAT = REDUCE_REPEAT_BYTES / BYTENUM_PER_FLOAT32;
constexpr uint32_t BUFFER_NUM = 1;
// DataCopyPad expresses blockLen in bytes through uint16_t. Keep each whole
// chunk 32-byte aligned as well, so both the GM address and the byte length can
// be unaligned while the copy size still fits in the API parameter.
constexpr uint32_t MAX_COPY_PAD_NUM =
    UINT16_MAX / BYTENUM_PER_FLOAT32 / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
static_assert(MAX_COPY_PAD_NUM * BYTENUM_PER_FLOAT32 <= UINT16_MAX,
    "DataCopyPad block length must fit in uint16_t");

__aicore__ inline void MergeScaleSsq(float s, float q, float& scale, float& ssq)
{
    if (s != s || scale != scale) {
        scale = s + scale;
        ssq = 1.0f;
        return;
    }
    if (s > FLT_MAX || scale > FLT_MAX) {
        scale = s + scale;
        ssq = 1.0f;
        return;
    }
    if (s == 0.0f) {
        return;
    }
    if (scale == 0.0f) {
        scale = s;
        ssq = q;
    } else if (scale >= s) {
        float r = s / scale;
        ssq += q * r * r;
    } else {
        float r = scale / s;
        ssq = q + ssq * r * r;
        scale = s;
    }
}

class Scnrm2AIV {
public:
    __aicore__ inline void Init(
        TPipe* pipe, GM_ADDR xGM, GM_ADDR wsGM, uint32_t blockIdx, uint32_t useCoreNum,
        const Scnrm2TilingData& tdata);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t dataCount);
    __aicore__ inline float ComputeMax(uint32_t dataCount);
    __aicore__ inline float ComputeScaledSsq(uint32_t dataCount, float scale);
    __aicore__ inline void WriteWorkspace(float scaleLocal, float ssqLocal);
    __aicore__ inline float Pass1ComputeMax();
    __aicore__ inline float Pass2ComputeSsq(float scaleLocal);

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;
    TBuf<TPosition::VECCALC> absBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> outBuf_;

    GlobalTensor<float> xGM_;
    GlobalTensor<float> wsGM_;

    uint32_t blockIdx_ = 0;
    uint32_t useCoreNum_ = 0;
    uint32_t calNum_ = 0;
    uint32_t startOffset_ = 0;
    uint32_t maxDataCount_ = 0;
};

__aicore__ inline void Scnrm2AIV::Init(
    TPipe* pipe, GM_ADDR xGM, GM_ADDR wsGM, uint32_t blockIdx, uint32_t useCoreNum,
    const Scnrm2TilingData& tdata)
{
    pipe_ = pipe;
    blockIdx_ = blockIdx;
    useCoreNum_ = useCoreNum;
    calNum_ = tdata.batchPerCore + (blockIdx_ < tdata.remain ? 1 : 0);
    startOffset_ = blockIdx_ * tdata.batchPerCore + (blockIdx_ < tdata.remain ? blockIdx_ : tdata.remain);
    maxDataCount_ = tdata.maxDataCount;

    xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xGM), static_cast<uint64_t>(tdata.n) * 2);
    xGM_.SetL2CacheHint(CacheMode::CACHE_MODE_PERSISTENT);
    wsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsGM), static_cast<uint64_t>(useCoreNum_) * 2);

    pipe_->InitBuffer(inQueue_, BUFFER_NUM, maxDataCount_ * sizeof(float));
    pipe_->InitBuffer(absBuf_, maxDataCount_ * sizeof(float));

    uint32_t level1Rep = (maxDataCount_ + ELEMENTS_PER_REPEAT - 1) / ELEMENTS_PER_REPEAT;
    uint32_t level1Align = ((level1Rep + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    pipe_->InitBuffer(workBuf_, level1Align * sizeof(float) + UB_BYTENUM_PER_BLOCK);
    pipe_->InitBuffer(outBuf_, UB_BYTENUM_PER_BLOCK);
}

__aicore__ inline void Scnrm2AIV::CopyIn(uint32_t offset, uint32_t dataCount)
{
    LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
    DataCopyParams copyParams{1, static_cast<uint16_t>(dataCount * sizeof(float)), 0, 0};
    uint32_t mod = dataCount % ELEMENTS_PER_BLOCK;
    uint8_t paddingNum = static_cast<uint8_t>(mod == 0 ? 0 : ELEMENTS_PER_BLOCK - mod);
    DataCopyPadParams padParams{true, 0, paddingNum, 0};

    DataCopyPad(inLocal, xGM_[offset], copyParams, padParams);
    inQueue_.EnQue<float>(inLocal);
}

__aicore__ inline float Scnrm2AIV::ComputeMax(uint32_t dataCount)
{
    LocalTensor<float> inLocal = inQueue_.DeQue<float>();
    LocalTensor<float> absLocal = absBuf_.Get<float>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();
    LocalTensor<float> outLocal = outBuf_.Get<float>();

    IsNan(absLocal, inLocal, dataCount);
    PipeBarrier<PIPE_V>();
    ReduceSum(outLocal, absLocal, workLocal, static_cast<int32_t>(dataCount));
    event_t nanEvt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(nanEvt);
    WaitFlag<HardEvent::V_S>(nanEvt);
    if (outLocal.GetValue(0) > 0.0f) {
        inQueue_.FreeTensor(inLocal);
        return 0.0f / 0.0f;
    }

    Abs(absLocal, inLocal, dataCount);
    PipeBarrier<PIPE_V>();
    ReduceMax(outLocal, absLocal, workLocal, static_cast<int32_t>(dataCount), false);

    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evt);
    WaitFlag<HardEvent::V_S>(evt);
    float tileMax = outLocal.GetValue(0);

    inQueue_.FreeTensor(inLocal);
    return tileMax;
}

__aicore__ inline float Scnrm2AIV::ComputeScaledSsq(uint32_t dataCount, float scale)
{
    LocalTensor<float> inLocal = inQueue_.DeQue<float>();
    LocalTensor<float> workLocal = workBuf_.Get<float>();
    LocalTensor<float> outLocal = outBuf_.Get<float>();
    LocalTensor<float> computeLocal = inLocal.template ReinterpretCast<float>();

    Divs(computeLocal, computeLocal, scale, dataCount);
    PipeBarrier<PIPE_V>();
    Mul(computeLocal, computeLocal, computeLocal, dataCount);
    PipeBarrier<PIPE_V>();
    ReduceSum(outLocal, computeLocal, workLocal, static_cast<int32_t>(dataCount));

    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evt);
    WaitFlag<HardEvent::V_S>(evt);
    float tileSsq = outLocal.GetValue(0);

    inQueue_.FreeTensor(inLocal);
    return tileSsq;
}

__aicore__ inline void Scnrm2AIV::WriteWorkspace(float scaleLocal, float ssqLocal)
{
    LocalTensor<float> outLocal = outBuf_.Get<float>();
    outLocal.SetValue(0, scaleLocal);
    outLocal.SetValue(1, ssqLocal);

    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evt);
    WaitFlag<HardEvent::S_MTE3>(evt);

    DataCopyParams copyParams{1, static_cast<uint16_t>(2 * sizeof(float)), 0, 0};
    DataCopyPad(wsGM_[blockIdx_ * 2], outLocal, copyParams);
}

__aicore__ inline float Scnrm2AIV::Pass1ComputeMax()
{
    float scaleLocal = 0.0f;
    const uint32_t copyChunk = maxDataCount_ < MAX_COPY_PAD_NUM ? maxDataCount_ : MAX_COPY_PAD_NUM;
    uint32_t repeatTimes = calNum_ / copyChunk;
    uint32_t remainNum = calNum_ % copyChunk;
    uint32_t currOffset = startOffset_;

    for (uint32_t i = 0; i < repeatTimes; i++) {
        CopyIn(currOffset, copyChunk);
        float tileMax = ComputeMax(copyChunk);
        if (tileMax != tileMax) {
            return tileMax;
        }
        if (tileMax > scaleLocal) {
            scaleLocal = tileMax;
        }
        currOffset += copyChunk;
    }
    if (remainNum > 0) {
        CopyIn(currOffset, remainNum);
        uint32_t alignedCount = (remainNum + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
        float tileMax = ComputeMax(alignedCount);
        if (tileMax != tileMax) {
            return tileMax;
        }
        if (tileMax > scaleLocal) {
            scaleLocal = tileMax;
        }
    }
    return scaleLocal;
}

__aicore__ inline float Scnrm2AIV::Pass2ComputeSsq(float scaleLocal)
{
    float ssqLocal = 0.0f;
    if (scaleLocal <= 0.0f) {
        return ssqLocal;
    }

    const uint32_t copyChunk = maxDataCount_ < MAX_COPY_PAD_NUM ? maxDataCount_ : MAX_COPY_PAD_NUM;
    uint32_t repeatTimes = calNum_ / copyChunk;
    uint32_t remainNum = calNum_ % copyChunk;
    uint32_t currOffset = startOffset_;

    for (uint32_t i = 0; i < repeatTimes; i++) {
        CopyIn(currOffset, copyChunk);
        ssqLocal += ComputeScaledSsq(copyChunk, scaleLocal);
        // Pass 1 already scanned all values for NaN. Two maximum-magnitude
        // components make the final float norm overflow regardless of the tail.
        if (scaleLocal == FLT_MAX && ssqLocal >= 2.0f) {
            return ssqLocal;
        }
        currOffset += copyChunk;
    }
    if (remainNum > 0) {
        CopyIn(currOffset, remainNum);
        uint32_t alignedCount = (remainNum + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
        ssqLocal += ComputeScaledSsq(alignedCount, scaleLocal);
    }
    return ssqLocal;
}

__aicore__ inline void Scnrm2AIV::Process()
{
    float scale = Pass1ComputeMax();
    float ssq = (scale != scale || scale > FLT_MAX) ? 1.0f : Pass2ComputeSsq(scale);
    WriteWorkspace(scale, ssq);
}

__simt_callee__ inline void Scnrm2SimtUpdate(float value, float& scale, float& ssq)
{
    if (value != value || scale != scale) {
        scale = value + scale;
        ssq = 1.0f;
        return;
    }
    float ax = (value >= 0.0f) ? value : -value;
    if (ax > FLT_MAX || scale > FLT_MAX) {
        scale = ax + scale;
        ssq = 1.0f;
        return;
    }
    if (ax == 0.0f) {
        return;
    }
    if (scale < ax) {
        float ratio = scale / ax;
        ssq = 1.0f + ssq * ratio * ratio;
        scale = ax;
        } else if (scale > 0.0f) {
            float ratio = ax / scale;
            ssq += ratio * ratio;
    }
}

__simt_callee__ inline void Scnrm2SimtMerge(
    const __ubuf__ float* sBuf, const __ubuf__ float* qBuf, uint32_t blockDimX, __gm__ float* wsOut)
{
    float scale = 0.0f;
    float ssq = 0.0f;
    for (uint32_t i = 0; i < blockDimX; i++) {
        float s = sBuf[i];
        float q = qBuf[i];
        if (s != s || scale != scale || s > FLT_MAX || scale > FLT_MAX) {
            scale = s + scale;
            ssq = 1.0f;
            continue;
        }
        if (s <= 0.0f) {
            continue;
        }
        float ratio = (scale > s) ? (s / scale) : (scale / s);
        if (scale <= 0.0f) {
            scale = s;
            ssq = q;
        } else if (scale >= s) {
            ssq += q * ratio * ratio;
        } else {
            ssq = q + ssq * ratio * ratio;
            scale = s;
        }
    }
    wsOut[0] = scale;
    wsOut[1] = ssq;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void Scnrm2Simt(
    uint32_t calNum, uint32_t elemStart, int32_t incx, uint32_t n, __gm__ const float* xGm, __gm__ float* wsOut)
{
    uint32_t absInc = static_cast<uint32_t>((incx > 0) ? incx : -incx);
    float scale = 0.0f;
    float ssq = 0.0f;

    for (uint32_t t = threadIdx.x; t < calNum; t += blockDim.x) {
        uint32_t elemIdx = elemStart + t;
        uint64_t complexIdx = (incx > 0) ? static_cast<uint64_t>(elemIdx) * absInc :
            static_cast<uint64_t>(n - 1 - elemIdx) * absInc;
        uint64_t floatIdx = complexIdx * 2;
        Scnrm2SimtUpdate(xGm[floatIdx], scale, ssq);
        Scnrm2SimtUpdate(xGm[floatIdx + 1], scale, ssq);
    }

    __ubuf__ float sBuf[SIMT_MAX_THREAD_NUM];
    __ubuf__ float qBuf[SIMT_MAX_THREAD_NUM];
    sBuf[threadIdx.x] = scale;
    qBuf[threadIdx.x] = ssq;
    asc_syncthreads();

    if (threadIdx.x == 0) {
        Scnrm2SimtMerge(sBuf, qBuf, blockDim.x, wsOut);
    }
}

class Scnrm2Final {
public:
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR wsGM, GM_ADDR resultGM, uint32_t useCoreNum, bool reduceSquareSum);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void ComputeResult();
    __aicore__ inline void CopyOut();

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TBuf<TPosition::VECCALC> outBuf_;

    GlobalTensor<float> wsGM_;
    GlobalTensor<float> outGM_;

    uint32_t useCoreNum_ = 0;
    uint32_t count_ = 0;
    uint32_t paddedCount_ = 0;
    float globalScale_ = 0.0f;
    float globalSsq_ = 0.0f;
    bool reduceSquareSum_ = false;
};

__aicore__ inline void Scnrm2Final::Init(TPipe* pipe, GM_ADDR wsGM, GM_ADDR resultGM, uint32_t useCoreNum, bool reduceSquareSum)
{
    pipe_ = pipe;
    useCoreNum_ = useCoreNum;
    reduceSquareSum_ = reduceSquareSum;
    count_ = useCoreNum_ * 2;
    paddedCount_ = (count_ + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK * ELEMENTS_PER_BLOCK;
    if (paddedCount_ < ELEMENTS_PER_BLOCK) {
        paddedCount_ = ELEMENTS_PER_BLOCK;
    }

    wsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsGM), paddedCount_);
    outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(resultGM), 1);

    pipe_->InitBuffer(inQueue_, 1, paddedCount_ * sizeof(float));
    pipe_->InitBuffer(outBuf_, UB_BYTENUM_PER_BLOCK);
}

__aicore__ inline void Scnrm2Final::CopyIn()
{
    LocalTensor<float> inLocal = inQueue_.AllocTensor<float>();
    if (count_ % ELEMENTS_PER_BLOCK != 0) {
        DataCopyParams copyParams{1, static_cast<uint16_t>(count_ * sizeof(float)), 0, 0};
        uint8_t paddingNum = static_cast<uint8_t>(paddedCount_ - count_);
        DataCopyPadParams padParams{true, 0, paddingNum, 0};
        DataCopyPad(inLocal, wsGM_, copyParams, padParams);
    } else {
        DataCopy(inLocal, wsGM_, count_);
    }
    inQueue_.EnQue<float>(inLocal);
}

__aicore__ inline void Scnrm2Final::ComputeResult()
{
    LocalTensor<float> inLocal = inQueue_.DeQue<float>();

    event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(evt);
    WaitFlag<HardEvent::MTE2_S>(evt);

    float globalScale = 0.0f;
    float globalSsq = 0.0f;
    for (uint32_t i = 0; i < useCoreNum_; i++) {
        if (reduceSquareSum_) {
            globalSsq += inLocal.GetValue(i * 2);
        } else {
            MergeScaleSsq(inLocal.GetValue(i * 2), inLocal.GetValue(i * 2 + 1), globalScale, globalSsq);
        }
    }

    inQueue_.FreeTensor(inLocal);
    globalScale_ = reduceSquareSum_ ? 1.0f : globalScale;
    globalSsq_ = globalSsq;
}

__aicore__ inline void Scnrm2Final::CopyOut()
{
    LocalTensor<float> outLocal = outBuf_.Get<float>();
    outLocal.SetValue(0, globalSsq_);

    event_t evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(evtSV);
    WaitFlag<HardEvent::S_V>(evtSV);

    Sqrt(outLocal, outLocal, static_cast<int32_t>(ELEMENTS_PER_BLOCK));

    event_t evtVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evtVS);
    WaitFlag<HardEvent::V_S>(evtVS);
    float rootSsq = outLocal.GetValue(0);
    outLocal.SetValue(0, globalScale_ * rootSsq);

    event_t evtSM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evtSM);
    WaitFlag<HardEvent::S_MTE3>(evtSM);

    DataCopyParams copyParams{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
    DataCopyPad(outGM_, outLocal, copyParams);
}

__aicore__ inline void Scnrm2Final::Process()
{
    CopyIn();
    ComputeResult();
    CopyOut();
}

} // namespace

__aicore__ inline bool KeepNormResult(GM_ADDR result, const Scnrm2TilingData& tiling)
{
    if (tiling.checkNormResult == 0) {
        return false;
    }
    GlobalTensor<float> resultGM;
    resultGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(result), 1);
    float value = resultGM.GetValue(0);
    // Recompute zero, tiny and overflowed results without a Host synchronization.
    return value != value || (value >= 1.0e-15f && value < FLT_MAX);
}

__global__ __aicore__ void scnrm2_compute_kernel(GM_ADDR x, GM_ADDR result, GM_ADDR workspace, Scnrm2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (KeepNormResult(result, tiling)) {
        return;
    }

    uint32_t blkIdx = GetBlockIdx();
    uint32_t calNum = tiling.batchPerCore + (blkIdx < tiling.remain ? 1 : 0);
    uint32_t elemStart = blkIdx * tiling.batchPerCore + (blkIdx < tiling.remain ? blkIdx : tiling.remain);

    if (tiling.incx == 1 && tiling.n >= 64) {
        TPipe pipe;
        Scnrm2AIV op;
        op.Init(&pipe, x, workspace, blkIdx, tiling.useCoreNum, tiling);
        op.Process();
    } else {
        asc_vf_call<Scnrm2Simt>(
            dim3{tiling.nthreads, 1, 1}, calNum, elemStart, tiling.incx, static_cast<uint32_t>(tiling.n),
            reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ float*>(workspace) + blkIdx * 2);
    }
}

__global__ __aicore__ void scnrm2_final_kernel(GM_ADDR result, GM_ADDR workspace, Scnrm2TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (KeepNormResult(result, tiling)) {
        return;
    }

    TPipe pipe;
    Scnrm2Final op;
    op.Init(&pipe, workspace, result, tiling.useCoreNum, false);
    op.Process();
}

void scnrm2_kernel_do(uint8_t* x, uint8_t* result, uint8_t* workspace,
    const Scnrm2TilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    scnrm2_compute_kernel<<<numBlocks, nullptr, aclStream>>>(x, result, workspace, tiling);
    scnrm2_final_kernel<<<1, nullptr, aclStream>>>(result, workspace, tiling);
}
