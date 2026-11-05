/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * aclblasIcamin kernel (arch35 / Ascend 950PR).
 *
 *   result = argmin_i (|Re(x[k])| + |Im(x[k])|),  k = 1+(i-1)*incx,  1-based, ties -> smallest index
 *
 * Structure mirrors the merged arch35 isamin operator:
 *   - icamin_aiv_kernel<<<useCoreNum>>>   unit-stride scan, one (minVal,minIdx) slot per core
 *   - icamin_small_kernel<<<1>>>          single-core fast path, writes the result directly
 *   - icamin_simt_kernel<<<useCoreNum>>>  strided (incx != 1) scan, one slot per core
 *   - icamin_reduce_kernel<<<1>>>         merges the slots, writes the 1-based INT32 result
 * The unit-stride scan folds each tile with pure vector ops so MTE2 and V overlap
 * through a depth-2 TQue: copy interleaved complex64, Abs + DeInterleave + Add
 * into a magnitude array, then ReduceMin(calIndex) drops one (value,index) pair
 * per tile into a UB slot. A NaN magnitude is rewritten to +Inf in the fold so a
 * lone NaN cannot poison its tile's reduce; a slice whose merged minimum is not
 * strictly below FLT_MAX (all NaN/Inf, or an exact FLT_MAX) is re-scanned element
 * by element instead. After the tile loop the per-tile slots merge into one
 * (value,index) per core, written back to GM over MTE3. The reduce launch orders
 * the two phases, so no cross-core barrier is needed.
 *
 * NaN semantics (cblas / golden): seed with abs1(x[0]), then update only on strict <.
 * NaN never satisfies <, so a NaN first element keeps index 1 and later finite values
 * cannot displace it; all-NaN also yields index 1. Vector tiles rewrite NaN magnitudes
 * to +Inf for ReduceMin stability; when global element 0 is NaN the owning core forces
 * (NaN, idx 0) before publish so the cross-core merge matches the sequential scan.
 */

#include <cstdint>
#include <cfloat>
#include <cmath>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "common/helper/kernel_constant.h"
#include "icamin_tiling_data.h"

using namespace AscendC;

constexpr uint32_t ICAMIN_BYTES_PER_FLOAT = 4;
constexpr uint32_t ICAMIN_UB_BLOCK_BYTES = 32;
constexpr uint32_t ICAMIN_ELEMENTS_PER_BLOCK = ICAMIN_UB_BLOCK_BYTES / ICAMIN_BYTES_PER_FLOAT;
// Complex elements folded per iteration, also the accumulator lane count and the
// workspace slot alignment granularity.
constexpr uint32_t ICAMIN_VF_FLOATS = 64;
// TQue depth for the unit-stride fold pipeline: two live slots so tile t+1's MTE2
// copy overlaps tile t's V-pipe magnitude+ReduceMin.
constexpr uint32_t ICAMIN_PIPE_BUFFERS = 2;

// Empty SIMT lane marker (no element visited). Distinct from a real NaN seed at idx 0.
constexpr uint32_t ICAMIN_EMPTY_IDX = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Sequential cblas-style update: seed the first seen (val,idx), then update only
// on strict <. NaN never updates (IEEE: any < involving NaN is false).
// ---------------------------------------------------------------------------
__simt_callee__ __aicore__ inline void IcaminAccumulate(
    float& bestVal, uint32_t& bestIdx, bool& hasValue, float val, uint32_t idx)
{
    if (!hasValue) {
        bestVal = val;
        bestIdx = idx;
        hasValue = true;
        return;
    }
    if (val < bestVal) {
        bestVal = val;
        bestIdx = idx;
    }
}

// ---------------------------------------------------------------------------
// Block-wide pow2 tree reduce. Empty lanes carry ICAMIN_EMPTY_IDX. Two live
// candidates merge in index order so the fold matches a left-to-right scan:
// start with the earlier index, accept the later only on strict <.
// ---------------------------------------------------------------------------
__simt_callee__ __aicore__ inline void IcaminSimtTreeReduce(
    __ubuf__ float* ubPartialVals, __ubuf__ uint32_t* ubPartialIdxs)
{
    unsigned int blockPow2 = 1;
    while (blockPow2 < blockDim.x) {
        blockPow2 <<= 1;
    }

    for (unsigned int s = blockPow2 >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && (threadIdx.x + s) < blockDim.x) {
            float otherVal = ubPartialVals[threadIdx.x + s];
            uint32_t otherIdx = ubPartialIdxs[threadIdx.x + s];
            float curVal = ubPartialVals[threadIdx.x];
            uint32_t curIdx = ubPartialIdxs[threadIdx.x];
            if (otherIdx == ICAMIN_EMPTY_IDX) {
                // partner empty: keep ours
            } else if (curIdx == ICAMIN_EMPTY_IDX) {
                ubPartialVals[threadIdx.x] = otherVal;
                ubPartialIdxs[threadIdx.x] = otherIdx;
            } else if (otherIdx < curIdx) {
                // other is earlier in the vector: seed=other, try cur as update
                if (!(curVal < otherVal)) {
                    ubPartialVals[threadIdx.x] = otherVal;
                    ubPartialIdxs[threadIdx.x] = otherIdx;
                }
            } else if (otherVal < curVal) {
                ubPartialVals[threadIdx.x] = otherVal;
                ubPartialIdxs[threadIdx.x] = otherIdx;
            }
        }
        asc_syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Scalar scan of a GM slice, one strided element per thread iteration.
// writeMode 0 -> (minVal,minIdx) slot, 1 -> 1-based INT32 (single-core path).
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void IcaminSimtCompute(
    uint32_t calNum, uint32_t blockStartOffset, uint32_t stride, __gm__ const float* xGm, __gm__ float* wsSlotPtr,
    uint32_t writeMode, __gm__ int32_t* resultPtr)
{
    if (calNum == 0) {
        return;
    }

    __ubuf__ float ubPartialVals[SIMT_MAX_THREAD_NUM];
    __ubuf__ uint32_t ubPartialIdxs[SIMT_MAX_THREAD_NUM];

    float bestVal = NAN;
    uint32_t bestIdx = 0;
    bool hasValue = false;

    for (uint32_t i = threadIdx.x; i < calNum; i += blockDim.x) {
        // int64_t avoids uint32 wrap when (blockStartOffset+i)*stride approaches 2^32.
        const int64_t complexIdx =
            (static_cast<int64_t>(blockStartOffset) + static_cast<int64_t>(i)) * static_cast<int64_t>(stride);
        float re = xGm[complexIdx * 2];
        float im = xGm[complexIdx * 2 + 1];
        float reAbs = (re >= 0.0f) ? re : -re;
        float imAbs = (im >= 0.0f) ? im : -im;
        IcaminAccumulate(bestVal, bestIdx, hasValue, reAbs + imAbs, blockStartOffset + i);
    }

    ubPartialVals[threadIdx.x] = bestVal;
    ubPartialIdxs[threadIdx.x] = hasValue ? bestIdx : ICAMIN_EMPTY_IDX;
    asc_syncthreads();
    IcaminSimtTreeReduce(ubPartialVals, ubPartialIdxs);

    // The store stays in the __simt_vf__ body: a GM write issued from a
    // __simt_callee__ was not visible to the following kernel in this build.
    if (threadIdx.x == 0) {
        uint32_t outIdx = ubPartialIdxs[0];
        if (outIdx == ICAMIN_EMPTY_IDX) {
            outIdx = blockStartOffset;
        }
        if (writeMode == 0) {
            wsSlotPtr[0] = ubPartialVals[0];
            *reinterpret_cast<__gm__ uint32_t*>(wsSlotPtr + 1) = outIdx;
        } else {
            resultPtr[0] = static_cast<int32_t>(outIdx + 1);
        }
    }
}

constexpr uint32_t ICAMIN_INF_BITS = 0x7F800000u;
// (value, index) slots written by the per-tile ReduceMin: 8 floats (32B) keeps every
// destination 32B-aligned; the work-buffer repeat granularity for ReduceMin.
constexpr uint32_t ICAMIN_SLOT_STRIDE = ICAMIN_ELEMENTS_PER_BLOCK;
constexpr uint32_t ICAMIN_REDUCE_REPEAT_BYTES = 256;
// Magnitudes per ReduceMin repeat (256 B of fp32) -- the fold granularity.
constexpr uint32_t ICAMIN_FOLD_LANES = ICAMIN_REDUCE_REPEAT_BYTES / ICAMIN_BYTES_PER_FLOAT; // 64
// Shortest slice the V-pipe fold is used for. Below it the indexed ReduceMin does
// not reliably return its index output (see ProcessScalar), and a scalar scan of a
// slice this short is both exact and cheaper than one more tile round-trip.
constexpr uint32_t ICAMIN_SCALAR_MAX = 8 * ICAMIN_FOLD_LANES; // 512

// ---------------------------------------------------------------------------
// Unit-stride per-core scan: copy a tile, fold it into a register (value,
// index) accumulator, and publish the slice once at the end.
// ---------------------------------------------------------------------------
class IcaminAIVBase {
protected:
    TPipe* pipe_;
    TQue<TPosition::VECIN, ICAMIN_PIPE_BUFFERS> inQue_;
    TBuf<TPosition::VECCALC> reBuf_;
    TBuf<TPosition::VECCALC> imBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> slotBuf_;
    TBuf<TPosition::VECCALC> outBufT_;
    TBuf<TPosition::VECCALC> workBufT_;
    GlobalTensor<float> gmX_;
    GM_ADDR x_;
    int32_t blockIdx_;
    uint32_t calNum_;
    uint32_t startOffset_;
    uint32_t tileComplex_;
    uint32_t nthreads_;
    uint32_t maxTiles_;
    float bestVal_;
    uint32_t bestIdx_;
    bool hasValue_;
    uint32_t writeMode_;
    __gm__ float* wsSlotPtr_;
    __gm__ int32_t* resultPtr_;

    __aicore__ inline void InitBase(TPipe* pipe, GM_ADDR inDevice, const IcaminTilingData& tdata)
    {
        pipe_ = pipe;
        x_ = inDevice;
        blockIdx_ = GetBlockIdx();
        uint32_t blockId = static_cast<uint32_t>(blockIdx_);
        startOffset_ = blockId * tdata.perCoreN;
        calNum_ = (blockId == tdata.useCoreNum - 1) ? tdata.lastCoreN : tdata.perCoreN;
        tileComplex_ = tdata.tileSize;
        nthreads_ = tdata.nthreads;
        writeMode_ = 0u;
        wsSlotPtr_ = nullptr;
        resultPtr_ = nullptr;

        gmX_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inDevice), tdata.totalN * 2);

        maxTiles_ = (calNum_ + tileComplex_ - 1) / tileComplex_;
        // ReduceMin work buffer sized like the sibling isamin (per-repeat partials).
        uint32_t elementsPerRepeat = ICAMIN_REDUCE_REPEAT_BYTES / sizeof(float);
        uint32_t level1RepeatCnt = (tileComplex_ + elementsPerRepeat - 1) / elementsPerRepeat;
        uint32_t level1AlignEnd =
            (level1RepeatCnt + ICAMIN_ELEMENTS_PER_BLOCK - 1) / ICAMIN_ELEMENTS_PER_BLOCK * ICAMIN_ELEMENTS_PER_BLOCK;

        // Double-buffered interleaved-complex DMA slots + magnitude scratch + per-tile
        // (value,index) slots. This is the unit-stride fold pipeline.
        pipe_->InitBuffer(inQue_, ICAMIN_PIPE_BUFFERS, tileComplex_ * 2 * sizeof(float));
        pipe_->InitBuffer(reBuf_, tileComplex_ * sizeof(float));
        pipe_->InitBuffer(imBuf_, tileComplex_ * sizeof(float));
        pipe_->InitBuffer(maskBuf_, tileComplex_ * sizeof(uint8_t));
        pipe_->InitBuffer(workBufT_, level1AlignEnd * sizeof(float) + ICAMIN_UB_BLOCK_BYTES);
        pipe_->InitBuffer(slotBuf_, maxTiles_ * ICAMIN_SLOT_STRIDE * sizeof(float) + ICAMIN_UB_BLOCK_BYTES);
        pipe_->InitBuffer(outBufT_, ICAMIN_UB_BLOCK_BYTES);

        bestVal_ = NAN;
        bestIdx_ = 0;
        hasValue_ = false;
    }

    // Magnitude count a tile is folded and reduced over, rounded up to the
    // ReduceMin repeat granularity (256 B = 64 fp32 lanes). The level-2 reduce
    // steps whole repeats, so a count that is not a whole number of repeats reads
    // lanes past the ones it was given -- stale scratch that can win the minimum.
    // The lanes above the real data carry +Inf magnitude, so the extra width
    // cannot change the winner.
    __aicore__ inline uint32_t AlignedTile(uint32_t tile)
    {
        return (tile + ICAMIN_FOLD_LANES - 1) / ICAMIN_FOLD_LANES * ICAMIN_FOLD_LANES;
    }

    // One unit-stride tile: copy interleaved complex64 into a TQue slot so the
    // MTE2 fetch of tile t+1 overlaps the vector fold of tile t. The host
    // 4-complex-aligns every slice, so a full tile is always 32B-aligned in both
    // address and length. A final tile that is not a whole 8-magnitude block must
    // still be reduced over a whole number of blocks -- the V pipe works in 32B
    // blocks, so ReduceMin over a raw sub-block count yields a wrong index -- and
    // the extra lanes carry +Inf magnitude so they can never win. MTE2's pad count
    // is a 3-bit field (it can only complete the last block), so the remainder is
    // filled by a Duplicate at a 32B-aligned offset.
    __aicore__ inline void CopyInTile(uint32_t tileStart, uint32_t tile)
    {
        uint32_t tileFloats = tile * 2;
        uint32_t alignedTile = AlignedTile(tile);
        LocalTensor<float> inLocal = inQue_.AllocTensor<float>();
        uint32_t copiedFloats =
            (tileFloats + ICAMIN_ELEMENTS_PER_BLOCK - 1) / ICAMIN_ELEMENTS_PER_BLOCK * ICAMIN_ELEMENTS_PER_BLOCK;
        if (tileFloats % ICAMIN_ELEMENTS_PER_BLOCK == 0) {
            DataCopy(inLocal, gmX_[tileStart * 2], tileFloats);
        } else {
            DataCopyParams copyParams{1, static_cast<uint16_t>(tileFloats * ICAMIN_BYTES_PER_FLOAT), 0, 0};
            DataCopyPadParams padParams{true, 0, static_cast<uint8_t>(copiedFloats - tileFloats), ICAMIN_INF_BITS};
            DataCopyPad(inLocal, gmX_[tileStart * 2], copyParams, padParams);
        }
        if (copiedFloats < alignedTile * 2) {
            float posInf = *reinterpret_cast<const float*>(&ICAMIN_INF_BITS);
            Duplicate(inLocal[copiedFloats], posInf, alignedTile * 2 - copiedFloats);
        }
        inQue_.EnQue(inLocal);
    }

    __aicore__ inline uint32_t TileSize(uint32_t t)
    {
        uint32_t rem = calNum_ - t * tileComplex_;
        return (rem >= tileComplex_) ? tileComplex_ : rem;
    }

    // Pure-SIMD magnitude fold over one tile of interleaved complex64:
    //   Abs(|re|,|im|) -> DeInterleave(re,im) -> Add(mag = |re|+|im|).
    // Every NaN magnitude is then rewritten to +Inf: CMPMODE::EQ of a value with
    // itself is 0 only for NaN (IEEE ordered compare), so the mask marks exactly
    // the non-NaN lanes; the Select keeps the magnitude there and drops a +Inf on
    // the NaN lanes. This removes all reliance on the ReduceMin NaN behaviour,
    // which is not stable across build modes. A tile of nothing but NaN/Inf
    // reduces to +Inf at its first lane; the owning core of global element 0
    // still forces index 0 when that element is NaN (see CombineSlice).
    __aicore__ inline void ComputeMagnitude(
        LocalTensor<float>& inLocal, LocalTensor<float>& reLocal, LocalTensor<float>& imLocal, uint32_t tile)
    {
        uint32_t tileFloats = tile * 2;
        LocalTensor<uint8_t> maskLocal = maskBuf_.Get<uint8_t>();
        Abs(inLocal, inLocal, static_cast<int32_t>(tileFloats));
        DeInterleave(reLocal, imLocal, inLocal, static_cast<int32_t>(tileFloats));
        Add(reLocal, reLocal, imLocal, tile);
        float posInf = *reinterpret_cast<const float*>(&ICAMIN_INF_BITS);
        Duplicate(imLocal, posInf, tile);
        Compare(maskLocal, reLocal, reLocal, CMPMODE::EQ, tile);
        Select(reLocal, maskLocal, reLocal, imLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, tile);
    }

    // Software-pipelined tile loop: prime tile 0, then for each tile DeQue it
    // (waiting on its MTE2 copy), issue tile t+1's copy, and fold tile t into its
    // own (value,index) slot with ReduceMin(calIndex). The depth-2 queue keeps at
    // most two copies in flight, so MTE2 does not idle while V folds.
    __aicore__ inline void ProcessTiles()
    {
        if (maxTiles_ == 0) {
            return;
        }
        LocalTensor<float> reLocal = reBuf_.Get<float>();
        LocalTensor<float> imLocal = imBuf_.Get<float>();
        LocalTensor<float> workLocal = workBufT_.Get<float>();
        LocalTensor<float> slotAll = slotBuf_.Get<float>();

        CopyInTile(startOffset_, TileSize(0));
        for (uint32_t t = 0; t < maxTiles_; ++t) {
            uint32_t tile = TileSize(t);
            LocalTensor<float> inLocal = inQue_.DeQue<float>();
            if (t + 1 < maxTiles_) {
                CopyInTile(startOffset_ + (t + 1) * tileComplex_, TileSize(t + 1));
            }
            uint32_t alignedTile = AlignedTile(tile);
            ComputeMagnitude(inLocal, reLocal, imLocal, alignedTile);
            ReduceMin(slotAll[t * ICAMIN_SLOT_STRIDE], reLocal, workLocal, static_cast<int32_t>(alignedTile), true);
            inQue_.FreeTensor(inLocal);
        }
    }

    // Merge the per-tile (value,index) slots into the slice candidate. PipeBarrier
    // drains the V pipe so the scalar GetValue reads see every finished ReduceMin.
    // Ties across tiles resolve to the smallest global index (ReduceMin already
    // returns the first -- smallest local -- index within a tile).
    //
    // Vector tiles rewrite NaN -> +Inf, so a NaN at global element 0 would otherwise
    // lose to a later finite. Force (NaN, idx 0) on the owning core to match cblas.
    __aicore__ inline bool CombineSlice()
    {
        LocalTensor<float> slotAll = slotBuf_.Get<float>();
        PipeBarrier<PIPE_ALL>();
        if (startOffset_ == 0) {
            float re0 = gmX_.GetValue(0);
            float im0 = gmX_.GetValue(1);
            if (re0 != re0 || im0 != im0) {
                bestVal_ = NAN;
                bestIdx_ = 0;
                hasValue_ = true;
                return false;
            }
        }
        float bestVal = FLT_MAX;
        uint32_t bestIdx = 0;
        bool hasValue = false;
        bool sawNaN = false;
        for (uint32_t t = 0; t < maxTiles_; ++t) {
            float v = slotAll.GetValue(t * ICAMIN_SLOT_STRIDE);
            float idxBitsF = slotAll.GetValue(t * ICAMIN_SLOT_STRIDE + 1);
            uint32_t localIdx = *reinterpret_cast<uint32_t*>(&idxBitsF);
            uint32_t gidx = startOffset_ + t * tileComplex_ + localIdx;
            if (v != v) {
                sawNaN = true;
            } else if (!hasValue || v < bestVal || (v == bestVal && gidx < bestIdx)) {
                bestVal = v;
                bestIdx = gidx;
                hasValue = true;
            }
        }
        bestVal_ = bestVal;
        bestIdx_ = bestIdx;
        hasValue_ = hasValue;
        return sawNaN || !hasValue;
    }

    // The slot/result write stays on MTE3. Handing it to a __simt_vf__ body ended
    // the block with errcode 341 (VEC access to UB out of bounds) in this build,
    // even though the same shape works when the SIMT call is the block's only
    // remaining work (the fallback path below).
    __aicore__ inline void PublishSlice()
    {
        if (writeMode_ == 0u) {
            LocalTensor<float> outLocal = outBufT_.Get<float>();
            outLocal.SetValue(0, bestVal_);
            outLocal.SetValue(1, *reinterpret_cast<float*>(&bestIdx_));

            GlobalTensor<float> wsGM;
            wsGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wsSlotPtr_), 2);
            DataCopyParams writeParams{1, static_cast<uint16_t>(2 * ICAMIN_BYTES_PER_FLOAT), 0, 0};
            DataCopyPad(wsGM, outLocal, writeParams);
            return;
        }

        int32_t result = static_cast<int32_t>(bestIdx_) + 1;
        LocalTensor<int32_t> outLocal = outBufT_.Get<int32_t>();
        Duplicate<int32_t>(outLocal, result, ICAMIN_ELEMENTS_PER_BLOCK);
        // Duplicate runs on the V pipe, the publish below reads UB on MTE3. Without
        // this barrier the copy wins the race and ships whatever stale bytes the
        // VECCALC slot held (observed: every single-block case returned float bits
        // -- +Inf / magnitudes from a previous kernel -- instead of the index).
        PipeBarrier<PIPE_ALL>();

        GlobalTensor<int32_t> resultGM;
        resultGM.SetGlobalBuffer(resultPtr_, 1);
        DataCopyParams writeParams{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        DataCopyPad(resultGM, outLocal, writeParams);
    }

    // Short-slice scan: one DataCopy into UB, then cblas-style strict-< walk
    // (seed with first magnitude, NaN never updates). The V-pipe fold is not used
    // below ICAMIN_SCALAR_MAX because indexed ReduceMin is unreliable on short
    // tiles. No §3.3 shape reaches this branch (min per-core slice ~18728).
    __aicore__ inline void ProcessScalar()
    {
        CopyInTile(startOffset_, calNum_);
        LocalTensor<float> inLocal = inQue_.DeQue<float>();
        PipeBarrier<PIPE_ALL>();

        float re0 = inLocal.GetValue(0);
        float im0 = inLocal.GetValue(1);
        float bestVal = ((re0 < 0.0f) ? -re0 : re0) + ((im0 < 0.0f) ? -im0 : im0);
        uint32_t bestIdx = startOffset_;
        for (uint32_t i = 1; i < calNum_; ++i) {
            const uint32_t local = i * 2;
            float re = inLocal.GetValue(local);
            float im = inLocal.GetValue(local + 1);
            float mag = ((re < 0.0f) ? -re : re) + ((im < 0.0f) ? -im : im);
            if (mag < bestVal) {
                bestVal = mag;
                bestIdx = startOffset_ + i;
            }
        }
        inQue_.FreeTensor(inLocal);
        bestVal_ = bestVal;
        bestIdx_ = bestIdx;
        hasValue_ = true;
        PublishSlice();
    }

    __aicore__ inline void ProcessBase()
    {
        if (calNum_ == 0) {
            return;
        }
        if (calNum_ < ICAMIN_SCALAR_MAX) {
            ProcessScalar();
            return;
        }
        ProcessTiles();
        if (CombineSlice()) {
            // Only reachable if a slot came back NaN despite the magnitude
            // sanitize; redo the slice element by element rather than publish a
            // value the fold cannot vouch for.
            asc_vf_call<IcaminSimtCompute>(
                dim3{nthreads_, 1, 1}, calNum_, startOffset_, 1u, reinterpret_cast<__gm__ const float*>(x_), wsSlotPtr_,
                writeMode_, resultPtr_);
            return;
        }
        PublishSlice();
    }
};

class IcaminAIV : public IcaminAIVBase {
public:
    __aicore__ inline IcaminAIV() {}
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR inDevice, GM_ADDR wsDevice, const IcaminTilingData& tdata);
    __aicore__ inline void Process() { ProcessBase(); }
};

__aicore__ inline void IcaminAIV::Init(TPipe* pipe, GM_ADDR inDevice, GM_ADDR wsDevice, const IcaminTilingData& tdata)
{
    InitBase(pipe, inDevice, tdata);
    wsSlotPtr_ = reinterpret_cast<__gm__ float*>(wsDevice) + static_cast<uint32_t>(blockIdx_) * 2;
}

class IcaminAIVSmall : public IcaminAIVBase {
public:
    __aicore__ inline IcaminAIVSmall() {}
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR inDevice, GM_ADDR resultDevice, const IcaminTilingData& tdata);
    __aicore__ inline void Process() { ProcessBase(); }
};

__aicore__ inline void IcaminAIVSmall::Init(
    TPipe* pipe, GM_ADDR inDevice, GM_ADDR resultDevice, const IcaminTilingData& tdata)
{
    InitBase(pipe, inDevice, tdata);
    writeMode_ = 1u;
    resultPtr_ = reinterpret_cast<__gm__ int32_t*>(resultDevice);
}

extern "C" __global__ __aicore__ void icamin_aiv_kernel(GM_ADDR x, GM_ADDR workSpace, IcaminTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    IcaminAIV op;
    op.Init(&pipe, x, workSpace, tdata);
    op.Process();
}

extern "C" __global__ __aicore__ void icamin_small_kernel(GM_ADDR x, GM_ADDR result, IcaminTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    IcaminAIVSmall op;
    op.Init(&pipe, x, result, tdata);
    op.Process();
}

extern "C" __global__ __aicore__ void icamin_simt_kernel(GM_ADDR x, GM_ADDR workSpace, IcaminTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    int32_t blockIdx = GetBlockIdx();
    uint32_t blockId = static_cast<uint32_t>(blockIdx);
    uint32_t calNum = (blockId == tdata.useCoreNum - 1) ? tdata.lastCoreN : tdata.perCoreN;
    uint32_t startOffset = blockId * tdata.perCoreN;

    if (calNum > 0) {
        __gm__ float* wsSlotPtr = reinterpret_cast<__gm__ float*>(workSpace) + blockId * 2;
        asc_vf_call<IcaminSimtCompute>(
            dim3{tdata.nthreads, 1, 1}, calNum, startOffset, tdata.incx, reinterpret_cast<__gm__ const float*>(x),
            wsSlotPtr, 0u, nullptr);
    }
}

// ---------------------------------------------------------------------------
// Merge the per-core slots: one bulk read into UB, then a 64-thread tree reduce
// (a scalar GetValue loop over the slots cost ~12us, most of this kernel).
// ---------------------------------------------------------------------------
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void IcaminReduceSimtCompute(
    __ubuf__ const float* wsData, uint32_t slotCount, __gm__ int32_t* resultGm)
{
    __ubuf__ float ubPartialVals[SIMT_MAX_THREAD_NUM];
    __ubuf__ uint32_t ubPartialIdxs[SIMT_MAX_THREAD_NUM];

    float bestVal = NAN;
    uint32_t bestIdx = 0;
    bool hasValue = false;

    for (uint32_t i = threadIdx.x; i < slotCount; i += blockDim.x) {
        float val = wsData[i * 2];
        uint32_t idx = *reinterpret_cast<const __ubuf__ uint32_t*>(&wsData[i * 2 + 1]);
        IcaminAccumulate(bestVal, bestIdx, hasValue, val, idx);
    }

    ubPartialVals[threadIdx.x] = bestVal;
    ubPartialIdxs[threadIdx.x] = hasValue ? bestIdx : ICAMIN_EMPTY_IDX;
    asc_syncthreads();
    IcaminSimtTreeReduce(ubPartialVals, ubPartialIdxs);

    if (threadIdx.x == 0) {
        uint32_t outIdx = ubPartialIdxs[0];
        if (outIdx == ICAMIN_EMPTY_IDX) {
            outIdx = 0;
        }
        resultGm[0] = static_cast<int32_t>(outIdx + 1);
    }
}

extern "C" __global__ __aicore__ void icamin_reduce_kernel(GM_ADDR workSpace, GM_ADDR resultGM, IcaminTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    uint32_t useCoreNum = tdata.useCoreNum;
    uint32_t totalFloats = useCoreNum * 2;
    uint32_t alignedFloats = ((totalFloats + ICAMIN_VF_FLOATS - 1) / ICAMIN_VF_FLOATS) * ICAMIN_VF_FLOATS;

    TBuf<TPosition::VECCALC> buf;
    pipe.InitBuffer(buf, alignedFloats * sizeof(float));
    LocalTensor<float> wsData = buf.Get<float>();

    GlobalTensor<float> wsGM;
    wsGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace), alignedFloats);
    DataCopy(wsData, wsGM, alignedFloats);

    event_t copyInEvent = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(copyInEvent);
    WaitFlag<HardEvent::MTE2_V>(copyInEvent);

    __ubuf__ float* wsUb = reinterpret_cast<__ubuf__ float*>(wsData.GetPhyAddr());
    __gm__ int32_t* resGm = reinterpret_cast<__gm__ int32_t*>(resultGM);
    constexpr uint32_t REDUCE_NTHREADS = 64;
    asc_vf_call<IcaminReduceSimtCompute>(dim3{REDUCE_NTHREADS, 1, 1}, wsUb, useCoreNum, resGm);
}

void icamin_kernel_do(
    GM_ADDR x, GM_ADDR result, GM_ADDR workSpace, const IcaminTilingData& tiling, uint32_t numBlocks, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    // The unit-stride host may pack slices 4-complex-aligned and end up using
    // fewer cores than numBlocks; launching the idle ones only costs time.
    uint32_t blocks = (tiling.useCoreNum > 0 && tiling.useCoreNum <= numBlocks) ? tiling.useCoreNum : numBlocks;

    if (tiling.incx == 1) {
        if (tiling.useCoreNum == 1 && tiling.lastCoreN <= tiling.tileSize) {
            icamin_small_kernel<<<1, nullptr, aclStream>>>(x, result, tiling);
        } else {
            icamin_aiv_kernel<<<blocks, nullptr, aclStream>>>(x, workSpace, tiling);
            icamin_reduce_kernel<<<1, nullptr, aclStream>>>(workSpace, result, tiling);
        }
    } else {
        icamin_simt_kernel<<<blocks, nullptr, aclStream>>>(x, workSpace, tiling);
        icamin_reduce_kernel<<<1, nullptr, aclStream>>>(workSpace, result, tiling);
    }
}
