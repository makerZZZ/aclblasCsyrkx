/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/devkit_version_compat.h"

#if ASC_DEVKIT_GE_9_1

#include <cstdlib>
#include <cstdio>
#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "csymm_kernel.h"
#include "csymm_tiling_data.h"
#define KERNEL_UTILS_LITE
#include "common/arch/hardware.h"
#include "common/helper/kernel_utils.h"
#include "tensor_api/tensor.h"

namespace te = AscendC::Te;

#include "common/helper/gemm_cube_arch35.h"
#include "common/helper/gemm_scale_arch35.h"

// ============================================================================
// GEMM Cube Kernel — software-pipelined (GM2L1 prefetch overlaps compute)
// ============================================================================
// Prefetch: issue GM2L1 copies for one K-chunk into L1 (async, no waiting).
template <uint32_t BM, uint32_t BK, uint32_t BN, typename TensorA, typename TensorB>
__aicore__ inline void PrefetchKChunk(
    const GemmCubeState& st, TensorA gmATensor, TensorB gmBTensor,
    const GemmTileState& tile, uint64_t l1OffsetA, uint64_t l1OffsetB,
    uint32_t kOffset, uint32_t curKChunk)
{
    TensorCopyGM2L1A<BM, BK>(
        st, gmATensor, l1OffsetA, tile.mi, kOffset, tile.curM, curKChunk);
    TensorCopyGM2L1B<BK, BN>(
        st, gmBTensor, l1OffsetB, tile.ni, kOffset, curKChunk, tile.curN);
}

// Compute: wait for the chunk's GM2L1, then run the BK-level L1->L0/Mmad
// pipeline, and finally release the L1 buffer.
template <uint32_t BM, uint32_t BK, uint32_t BN>
__aicore__ inline void ComputeKChunk(
    const GemmCubeState& st, const GemmTileState& tile,
    uint64_t l1OffsetA, uint64_t l1OffsetB, uint64_t l1BufId,
    uint32_t kOffset, uint32_t curKChunk,
    uint32_t segStart, uint32_t segEnd, uint64_t& l0PingPong)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    for (uint32_t kInnerOffset = 0; kInnerOffset < curKChunk; kInnerOffset += BK) {
        uint32_t remainingK = curKChunk - kInnerOffset;
        uint32_t curK = remainingK < BK ? remainingK : BK;
        // BK=64 uses the full L0A/L0B (single buffer, no ping-pong).
        uint64_t l0BufId = (BK == 64) ? 0 : (l0PingPong & GEMM_BUFFER_MASK);
        uint32_t globalKOffset = kOffset + kInnerOffset;
        bool isFirstK = (globalKOffset == segStart);
        bool isLastK = (globalKOffset + curK == segEnd);
        RunL1ToL0Mmad<BM, BK, BN>(
            st, tile, l1OffsetA, l1OffsetB, l0BufId, kInnerOffset, curKChunk, curK,
            isFirstK, isLastK);
        l0PingPong++;
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
}

template <uint32_t BM, uint32_t BK, uint32_t BN, typename TensorA, typename TensorB, typename TensorC>
__aicore__ inline void ProcessMNChunk(
    const GemmCubeState& st, TensorA gmATensor, TensorB gmBTensor,
    TensorC gmCTensor0, TensorC gmCTensor1,
    const GemmTileState& tile, uint64_t l1BufferBytes,
    uint64_t& l1PingPong, uint64_t& l0PingPong,
    uint32_t seg, uint32_t segStart, uint32_t segEnd, uint32_t chunk)
{
    uint32_t nChunks = CeilDiv(segEnd - segStart, chunk);
    for (uint32_t c = 0; c < nChunks; c++) {
        uint32_t kOffset = segStart + c * chunk;
        uint32_t curKChunk = (segEnd - kOffset) < chunk ? (segEnd - kOffset) : chunk;
        uint64_t l1BufId = l1PingPong & GEMM_BUFFER_MASK;
        uint64_t l1OffsetA = l1BufId * l1BufferBytes;
        uint64_t l1OffsetB = l1OffsetA +
            static_cast<uint64_t>(BM) * static_cast<uint64_t>(chunk) * sizeof(float);
        uint32_t nextKOffset = kOffset + curKChunk;
        if (nextKOffset < segEnd) {
            uint64_t nextBufId = (l1PingPong + 1) & GEMM_BUFFER_MASK;
            uint64_t nextOffsetA = nextBufId * l1BufferBytes;
            uint64_t nextOffsetB = nextOffsetA +
                static_cast<uint64_t>(BM) * static_cast<uint64_t>(chunk) * sizeof(float);
            uint32_t nextKChunk = (segEnd - nextKOffset) < chunk ? (segEnd - nextKOffset) : chunk;
            // Prefetch the next K-chunk into the other L1 bank while the cube
            // is computing the current one (input double buffering).
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(nextBufId);
            PrefetchKChunk<BM, BK, BN>(
                st, gmATensor, gmBTensor, tile, nextOffsetA, nextOffsetB,
                nextKOffset, nextKChunk);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(nextBufId);
        }
        ComputeKChunk<BM, BK, BN>(
            st, tile, l1OffsetA, l1OffsetB, l1BufId,
            kOffset, curKChunk, segStart, segEnd, l0PingPong);
        l1PingPong++;
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(GEMM_ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(GEMM_ZERO_FLAG);
        TensorWriteFixpipeBlock<BM, BN>(
            st, (seg == 0) ? gmCTensor0 : gmCTensor1,
            tile.mi, tile.ni, tile.mStart, tile.nStart);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(GEMM_ZERO_FLAG);
    }
}

template <uint32_t BM, uint32_t BK, uint32_t BN, typename TensorA, typename TensorB, typename TensorC>
__aicore__ inline void ProcessMNTile(
    const GemmCubeState& st, TensorA gmATensor, TensorB gmBTensor,
    TensorC gmCTensor0, TensorC gmCTensor1,
    const GemmTileState& tile, uint64_t l1BufferBytes,
    uint64_t& l1PingPong, uint64_t& l0PingPong)
{
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(GEMM_ZERO_FLAG);
    uint32_t kStart = static_cast<uint32_t>(st.tiling.kStart);
    uint32_t kEnd = static_cast<uint32_t>(st.tiling.kEnd);
    if (kEnd <= kStart) {
        kStart = 0;
        kEnd = static_cast<uint32_t>(st.tiling.k);
    }
    uint32_t chunk = static_cast<uint32_t>(st.tiling.tileKChunk);
    if (chunk == 0) {
        chunk = kEnd - kStart;
    }
    uint32_t segCount = (st.tiling.kSegmentCount == 2) ? 2u : 1u;
    uint32_t kMid = kStart + (kEnd - kStart) / 2;
    for (uint32_t seg = 0; seg < segCount; seg++) {
        // Segment 1 must wait until segment 0's Fixpipe (L0C drain) finished,
        // otherwise its Mmad would race the still-active writeback.
        if (seg > 0) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(GEMM_ZERO_FLAG);
        }
        uint32_t segStart = (seg == 0) ? kStart : kMid;
        uint32_t segEnd = (seg == 0) ? ((segCount == 2) ? kMid : kEnd) : kEnd;
        if (segEnd <= segStart) {
            break;
        }
        // Prefetch chunk 0 of this segment into L1 slot 0 (async).
        {
            uint64_t l1BufId = l1PingPong & GEMM_BUFFER_MASK;
            uint64_t l1OffsetA = l1BufId * l1BufferBytes;
            uint64_t l1OffsetB = l1OffsetA +
                static_cast<uint64_t>(BM) * static_cast<uint64_t>(chunk) * sizeof(float);
            uint32_t curKChunk = (segEnd - segStart) < chunk ? (segEnd - segStart) : chunk;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            PrefetchKChunk<BM, BK, BN>(
                st, gmATensor, gmBTensor, tile, l1OffsetA, l1OffsetB, segStart, curKChunk);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
        }
        ProcessMNChunk<BM, BK, BN>(st, gmATensor, gmBTensor, gmCTensor0, gmCTensor1,
            tile, l1BufferBytes, l1PingPong, l0PingPong, seg, segStart, segEnd, chunk);
    }
}

template <uint32_t BM, uint32_t BK, uint32_t BN, uint32_t C0_VAL, bool IsTransA, bool IsTransB,
          typename TensorA, typename TensorB, typename TensorC>
__aicore__ inline void ProcessMNBlock(
    const GemmCubeState& st, TensorA gmATensor, TensorB gmBTensor,
    TensorC gmCTensor0, TensorC gmCTensor1,
    uint64_t l1BufferBytes, uint64_t& l1PingPong, uint64_t& l0PingPong,
    uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd)
{
    for (uint32_t mi = mStart; mi < mEnd; mi++) {
        for (uint32_t ni = nStart; ni < nEnd; ni++) {
            GemmTileState tile = MakeGemmTileState<BM, BN>(st, mi, ni, mStart, nStart);
            ProcessMNTile<BM, BK, BN>(
                st, gmATensor, gmBTensor, gmCTensor0, gmCTensor1, tile,
                l1BufferBytes, l1PingPong, l0PingPong);
        }
    }
}

template <uint32_t BM, uint32_t BK, uint32_t BN, uint32_t C0_VAL, bool IsTransA, bool IsTransB,
          typename TensorA, typename TensorB, typename TensorC>
__aicore__ inline void ProcessAllTiles(
    const GemmCubeState& st, TensorA gmATensor, TensorB gmBTensor,
    TensorC gmCTensor0, TensorC gmCTensor1)
{
    // DAV-3510 real L1 is 512KB (HardwareInfo's 32KB is wrong for V350);
    // use half per ping-pong buffer.
    constexpr uint64_t L1_BUFFER_BYTES =
        CSYMM_ARCH35_L1_SIZE_BYTES / GEMM_BUFFER_COUNT;
    InitGemmPipelineFlags();
    uint64_t l1PingPong = 0;
    uint64_t l0PingPong = 0;

    for (uint32_t mb = 0; mb < st.mBlockCount; mb++) {
        for (uint32_t nb = 0; nb < st.nBlockCount; nb++) {
            uint32_t mStart, mEnd, nStart, nEnd;
            MakeGemmBlockRange(st, mb, nb, mStart, mEnd, nStart, nEnd);
            ProcessMNBlock<BM, BK, BN, C0_VAL, IsTransA, IsTransB>(
                st, gmATensor, gmBTensor, gmCTensor0, gmCTensor1,
                L1_BUFFER_BYTES, l1PingPong, l0PingPong,
                mStart, mEnd, nStart, nEnd);
        }
    }

    // L1 slots are drained by each tile's initial-prefetch wait; only drain
    // the M (Mmad) and FIX (writeback) stages here. Consume the final
    // MTE1_MTE2 release so a fused (multi-GEMM) kernel starts with a clean
    // L1-flag state.
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(GEMM_ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(GEMM_FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(GEMM_ZERO_FLAG);
    uint64_t lastL1BufId = (l1PingPong == 0) ? 0 : ((l1PingPong - 1) & GEMM_BUFFER_MASK);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(lastL1BufId);
}

template <uint32_t BM, uint32_t BK, uint32_t BN, uint32_t C0_VAL, bool IsTransA, bool IsTransB>
__aicore__ inline void GemmCubeKernelImpl(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c,
    __gm__ uint8_t* c1, GemmTilingData& tiling, uint32_t blockIdx)
{
    GemmCubeState st{};
    if (!InitGemmState<BM, BK, BN, C0_VAL>(st, tiling, blockIdx)) {
        return;
    }

    auto operands = MakeGemmOperandTensors<IsTransA, IsTransB>(st, a, b, c);
    auto gmCTensor1 = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(c1)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(st.tiling.m),
                                                static_cast<uint64_t>(st.tiling.ldc)));

    ProcessAllTiles<BM, BK, BN, C0_VAL, IsTransA, IsTransB>(
        st, operands.a, operands.b, operands.c, gmCTensor1);
}

// The template tuple (BM, BK, BN) MUST equal the host-side (baseM, baseK,
// baseN) that CalcMultiCorePartition used to size the core grid (mBlocks x
// nBlocks) and that InitGemmState uses to partition the M/N axes. Dispatching
// on baseM alone was wrong for skewed shapes: the host halves the SHORT side's
// tile to 128 (baseN=128 for wide m<n, baseM=128 for tall m>n), but a
// baseM-only check sent both to the (256,64,256) / (128,64,128) templates whose
// other axis disagrees with the grid -> nBlocks blocks that InitGemmState
// returns early on (actualN/actualM == 0), idling most of the 28 AIC cores.
template <bool IsTransA, bool IsTransB>
__aicore__ inline void GemmCubeKernelDispatch(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c,
    __gm__ uint8_t* c1, GemmTilingData& tiling, uint32_t blockIdx)
{
    if (tiling.baseM == 256 && tiling.baseN == 256 && tiling.baseK == 64) {
        GemmCubeKernelImpl<256, 64, 256, GEMM_C0_SIZE, IsTransA, IsTransB>(a, b, c, c1, tiling, blockIdx);
    } else if (tiling.baseM == 256 && tiling.baseN == 128 && tiling.baseK == 64) {
        // wide skew (m < n): short side m is tiled at 128 for core parallelism
        GemmCubeKernelImpl<256, 64, 128, GEMM_C0_SIZE, IsTransA, IsTransB>(a, b, c, c1, tiling, blockIdx);
    } else if (tiling.baseM == 128 && tiling.baseN == 256 && tiling.baseK == 64) {
        // tall skew (n < m): short side n is tiled at 128 for core parallelism
        GemmCubeKernelImpl<128, 64, 256, GEMM_C0_SIZE, IsTransA, IsTransB>(a, b, c, c1, tiling, blockIdx);
    } else if (tiling.baseM == 128 && tiling.baseN == 128 && tiling.baseK == 64) {
        GemmCubeKernelImpl<128, 64, 128, GEMM_C0_SIZE, IsTransA, IsTransB>(a, b, c, c1, tiling, blockIdx);
    } else if (tiling.baseM == 256 && tiling.baseK == 32) {
        GemmCubeKernelImpl<256, 32, 256, GEMM_C0_SIZE, IsTransA, IsTransB>(a, b, c, c1, tiling, blockIdx);
    } else {
        GemmCubeKernelImpl<GEMM_BASE_M, GEMM_BASE_K, GEMM_BASE_N, GEMM_C0_SIZE, IsTransA, IsTransB>(
            a, b, c, c1, tiling, blockIdx);
    }
}

extern "C" __global__ __cube__ void csymm_cube_gemm_kernel(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c,
    __gm__ uint8_t* c1, GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    bool isTransA = (tiling.isTransA != 0);
    bool isTransB = (tiling.isTransB != 0);
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (isTransA && isTransB) {
        GemmCubeKernelDispatch<true, true>(a, b, c, c1, tiling, blockIdx);
    } else if (isTransA && !isTransB) {
        GemmCubeKernelDispatch<true, false>(a, b, c, c1, tiling, blockIdx);
    } else if (!isTransA && isTransB) {
        GemmCubeKernelDispatch<false, true>(a, b, c, c1, tiling, blockIdx);
    } else {
        GemmCubeKernelDispatch<false, false>(a, b, c, c1, tiling, blockIdx);
    }
}

// Fused 3-GEMM kernel: runs the three 3m real GEMMs (w1/w2/w3) of one csymm
// call from a SINGLE launch with 3 * usedCoreNum blocks. The three GEMMs share
// the same tiling/m/n/k and differ only in their A/B/C plane pointers, and the
// cube kernel is pure per-block (blockIdx -> tile, no cross-block sync), so
// the fused launch is bit-identical per block to three serial launches while
// letting the 28 AIC cores schedule blocks from all three GEMMs concurrently.
// Serial dispatch under-utilized the cores whenever one GEMM used < ~19 cores
// (e.g. 1024^2 uses 16); fusing cuts the cube phase from ~3x one GEMM toward
// ceil(3*usedCoreNum/28) batches; the 3m GEMMs are always fused into a single launch.
extern "C" __global__ __cube__ void csymm_cube_gemm3_kernel(
    __gm__ uint8_t* a0, __gm__ uint8_t* b0, __gm__ uint8_t* c0, __gm__ uint8_t* d0,
    __gm__ uint8_t* a1, __gm__ uint8_t* b1, __gm__ uint8_t* c1_, __gm__ uint8_t* d1,
    __gm__ uint8_t* a2, __gm__ uint8_t* b2, __gm__ uint8_t* c2, __gm__ uint8_t* d2,
    GemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    const uint32_t perGemm = static_cast<uint32_t>(tiling.usedCoreNum);
    if (perGemm == 0) {
        return;
    }
    const uint32_t bid = AscendC::GetBlockIdx();
    const uint32_t gid = bid / perGemm;
    if (gid > 2) {
        return;
    }
    const uint32_t local = bid - gid * perGemm;
    __gm__ uint8_t* a = (gid == 0) ? a0 : (gid == 1) ? a1 : a2;
    __gm__ uint8_t* b = (gid == 0) ? b0 : (gid == 1) ? b1 : b2;
    __gm__ uint8_t* c = (gid == 0) ? c0 : (gid == 1) ? c1_ : c2;
    __gm__ uint8_t* d = (gid == 0) ? d0 : (gid == 1) ? d1 : d2;
    bool isTransA = (tiling.isTransA != 0);
    bool isTransB = (tiling.isTransB != 0);
    if (isTransA && isTransB) {
        GemmCubeKernelDispatch<true, true>(a, b, c, d, tiling, local);
    } else if (isTransA && !isTransB) {
        GemmCubeKernelDispatch<true, false>(a, b, c, d, tiling, local);
    } else if (!isTransA && isTransB) {
        GemmCubeKernelDispatch<false, true>(a, b, c, d, tiling, local);
    } else {
        GemmCubeKernelDispatch<false, false>(a, b, c, d, tiling, local);
    }
}


// ============================================================================
// Scale kernel — C = beta * C (in-place), for alpha==0 / k==0 fast path
// (implementation shared with gemm via common/helper/gemm_scale_arch35.h)
// ============================================================================
extern "C" __global__ __aicore__ void csymm_scale_kernel(
    __gm__ uint8_t* cInOut, int32_t m, int32_t n, int32_t ldc,
    float betaReal, float betaImag, int32_t isComplex)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    gemm_scale::ScaleFp32Kernel op;
    op.Init(cInOut, m, n, ldc, betaReal, betaImag, isComplex, &pipe);
    op.Process();
}


// ============================================================================
// Csymm combine kernel — merge 3 FP32 GEMM results (3m) into complex C
// w1=(Ar+Ai)(Br+Bi), w2=Ar*Br, w3=Ai*Bi  =>  Re=w2-w3, Im=w1-w2-w3
// C = alpha*Re + i*alpha*Im + beta*C
// ============================================================================
namespace csymm_combine {
using namespace AscendC;

constexpr int32_t COMBINE_TILE = 2048;
constexpr int32_t COMBINE_BUF_NUM = 2;
// Scratch layout inside calcBuf_: [abR|abI|cR|cI|outR|outI|tmp|outBuf(2x)]
constexpr int32_t COMBINE_SCRATCH_NUM = 9;

class CsymmCombineKernel {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* w1a, __gm__ uint8_t* w2a, __gm__ uint8_t* w3a,
        __gm__ uint8_t* w1b, __gm__ uint8_t* w2b, __gm__ uint8_t* w3b,
        int32_t kSplit, int32_t tempLdc, __gm__ uint8_t* cInOut,
        int32_t m, int32_t n, int32_t ldc,
        float ar, float ai, float br, float bi, AscendC::TPipe* pipe)
    {
        pipe_ = pipe;
        m_ = m;
        n_ = n;
        ldc_ = ldc;
        tempLdc_ = tempLdc;
        kSplit_ = (kSplit != 0);
        ar_ = ar; ai_ = ai; br_ = br; bi_ = bi;
        w1aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w1a),
            static_cast<uint64_t>(tempLdc) * n);
        w2aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w2a),
            static_cast<uint64_t>(tempLdc) * n);
        w3aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w3a),
            static_cast<uint64_t>(tempLdc) * n);
        w1bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w1b),
            static_cast<uint64_t>(tempLdc) * n);
        w2bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w2b),
            static_cast<uint64_t>(tempLdc) * n);
        w3bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w3b),
            static_cast<uint64_t>(tempLdc) * n);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cInOut),
            static_cast<uint64_t>(ldc) * n * 2);
        PartitionColsByBlock(n_, colsPerCore_, startCol_, endCol_);
    }

    __aicore__ inline void Process()
    {
        if (startCol_ >= endCol_) return;
        pipe_->InitBuffer(t1Que_, COMBINE_BUF_NUM, COMBINE_TILE * sizeof(float));
        pipe_->InitBuffer(t2Que_, COMBINE_BUF_NUM, COMBINE_TILE * sizeof(float));
        pipe_->InitBuffer(cQue_, COMBINE_BUF_NUM, 2 * COMBINE_TILE * sizeof(float));
        pipe_->InitBuffer(outQue_, COMBINE_BUF_NUM, 2 * COMBINE_TILE * sizeof(float));
        pipe_->InitBuffer(calcBuf_, COMBINE_SCRATCH_NUM * COMBINE_TILE * sizeof(float));

        const bool betaZero = (br_ == 0.0f && bi_ == 0.0f);
        if (!kSplit_ && betaZero && m_ <= COMBINE_TILE && ldc_ == m_) {
            // Wide shapes ran one tiny tile per output column (30-150 tiles per
            // core on 1129/1186/1173/1075-1078...), each paying the full
            // queue-handshake latency (~0.4us). Pack whole columns into one
            // flat block so the handshakes amortize over colsPerBlock columns.
            // The w planes are column-major with stride tempLdc_; a 2D read
            // with a per-column source gap lands the nCols columns
            // concatenated in UB, the element-wise subs/adds keep that layout,
            // and Interleave turns the two concatenated re/im planes into one
            // column-concatenated complex run that maps back onto C
            // (a single contiguous write when ldc == m).
            int32_t colsPerBlock = (m_ > 0) ? COMBINE_TILE / m_ : 1;
            if (colsPerBlock < 1) colsPerBlock = 1;
            int32_t col = startCol_;
            while (col < endCol_) {
                int32_t nCols = endCol_ - col;
                if (nCols > colsPerBlock) nCols = colsPerBlock;
                ProcessColBlock(col, nCols);
                col += nCols;
            }
            return;
        }
        ForEachColumnTile(*this, startCol_, endCol_, m_, COMBINE_TILE);
    }

    // beta==0 / kSplit==0 flat path: process nCols whole columns (count =
    // nCols*m_ <= COMBINE_TILE) with a single set of queue handshakes.
    // beta == 0 fast path (every TC_PF perf case) and the general beta != 0 path
    // both need the alpha*(t_a + i*u_a) product; shared with ProcessTile.
    __aicore__ inline void EmitAlphaCombined(
        LocalTensor<float> abR, LocalTensor<float> abI,
        LocalTensor<float> outR, LocalTensor<float> outI, LocalTensor<float> tmp,
        LocalTensor<float> outBuf, int32_t vecCount, int32_t tail,
        const float tA[8], const float uA[8], const float tB[8], const float uB[8])
    {
        if (vecCount > 0) {
            Muls(outR, abR, ar_, vecCount);
            Muls(tmp, abI, ai_, vecCount);
            Sub(outR, outR, tmp, vecCount);
            Muls(outI, abI, ar_, vecCount);
            Muls(tmp, abR, ai_, vecCount);
            Add(outI, outI, tmp, vecCount);
        }
        for (int32_t i = 0; i < tail; i++) {
            int32_t idx = vecCount + i;
            float ab_r = kSplit_ ? (tA[i] + tB[i]) : tA[i];
            float ab_i = kSplit_ ? (uA[i] + uB[i]) : uA[i];
            outBuf.SetValue(2 * idx, ar_ * ab_r - ai_ * ab_i);
            outBuf.SetValue(2 * idx + 1, ar_ * ab_i + ai_ * ab_r);
        }
    }

    // Read the three real planes (w1/w2/w3 = Aplus/Br/Bi) of nCols columns into
    // the de-interleave queues (Compact 2D copy: column gap = tempLdc_ - m_).
    __aicore__ inline void ReadCombinePlanes(
        LocalTensor<float>& w1aLocal, LocalTensor<float>& w2aLocal, LocalTensor<float>& w3aLocal,
        uint64_t tempOffset, int32_t nCols)
    {
        DataCopyExtParams ext{static_cast<uint16_t>(nCols),
            static_cast<uint32_t>(static_cast<uint32_t>(m_) * sizeof(float)),
            static_cast<int64_t>(tempLdc_ - m_) * static_cast<int64_t>(sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        LocalTensor<float> w1aTile = t1Que_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Compact>(w1aTile, w1aGlobal_[tempOffset], ext, padParams);
        LocalTensor<float> w2aTile = t1Que_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Compact>(w2aTile, w2aGlobal_[tempOffset], ext, padParams);
        LocalTensor<float> w3aTile = t2Que_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Compact>(w3aTile, w3aGlobal_[tempOffset], ext, padParams);
        t1Que_.EnQue(w1aTile);
        t1Que_.EnQue(w2aTile);
        t2Que_.EnQue(w3aTile);
        w1aLocal = t1Que_.DeQue<float>();
        w2aLocal = t1Que_.DeQue<float>();
        w3aLocal = t2Que_.DeQue<float>();
    }

    // beta==0 / kSplit==0 flat path: process nCols whole columns (count =
    // nCols*m_ <= COMBINE_TILE) with a single set of queue handshakes.
    __aicore__ inline void ProcessColBlock(int32_t colStart, int32_t nCols)
    {
        int32_t count = m_ * nCols;
        const uint64_t tempOffset = static_cast<uint64_t>(colStart) * tempLdc_;
        LocalTensor<float> scratch = calcBuf_.Get<float>();
        LocalTensor<float> abR = scratch[0];
        LocalTensor<float> abI = scratch[COMBINE_TILE];
        LocalTensor<float> outR = scratch[2 * COMBINE_TILE];
        LocalTensor<float> outI = scratch[3 * COMBINE_TILE];
        LocalTensor<float> tmp = scratch[4 * COMBINE_TILE];
        LocalTensor<float> outBuf = outQue_.AllocTensor<float>();
        int32_t vecCount = count & ~7; // vector ops need a multiple of 8 floats
        int32_t tail = count - vecCount;
        float tA[8] = {0.0f};
        float uA[8] = {0.0f};

        LocalTensor<float> w1aLocal;
        LocalTensor<float> w2aLocal;
        LocalTensor<float> w3aLocal;
        ReadCombinePlanes(w1aLocal, w2aLocal, w3aLocal, tempOffset, nCols);
        if (vecCount > 0) {
            Sub(abR, w2aLocal, w3aLocal, vecCount);      // t_a (beta=0: straight to abR)
            Sub(tmp, w1aLocal, w2aLocal, vecCount);
            Sub(abI, tmp, w3aLocal, vecCount);           // u_a
        }
        for (int32_t i = 0; i < tail; i++) {
            int32_t idx = vecCount + i;
            tA[i] = w2aLocal.GetValue(idx) - w3aLocal.GetValue(idx);
            uA[i] = w1aLocal.GetValue(idx) - w2aLocal.GetValue(idx) - w3aLocal.GetValue(idx);
        }
        t1Que_.FreeTensor(w1aLocal);
        t1Que_.FreeTensor(w2aLocal);
        t2Que_.FreeTensor(w3aLocal);

        EmitAlphaCombined(abR, abI, outR, outI, tmp, outBuf, vecCount, tail, tA, uA, tA, uA);
        if (vecCount > 0) {
            LocalTensor<float> dst0 = outBuf[0];
            LocalTensor<float> dst1 = outBuf[vecCount];
            Interleave(dst0, dst1, outR, outI, vecCount);
        }
        outQue_.EnQue(outBuf);
        LocalTensor<float> oLocal = outQue_.DeQue<float>();
        // Only reached when ldc_ == m_ (enforced by the caller): the
        // column-concatenated complex run is one contiguous C write. C with
        // padding (ldc > m) keeps using the legacy per-column ProcessTile
        // path, whose per-tile GM writes handle arbitrary ldc.
        const uint64_t cOff = static_cast<uint64_t>(colStart) * ldc_ * 2;
        DataCopyExtParams cext{1, static_cast<uint32_t>(static_cast<uint32_t>(count) * 2 * sizeof(float)), 0, 0, 0};
        DataCopyPad<float, PaddingMode::Normal>(cGlobal_[cOff], oLocal, cext);
        outQue_.FreeTensor(oLocal);
    }

    // Stage segment-a (t_a/u_a) into abR/abI (non-kSplit) or cR/cI (kSplit).
    __aicore__ inline void CsymmStageSegA(
        LocalTensor<float>& w1aLocal, LocalTensor<float>& w2aLocal, LocalTensor<float>& w3aLocal,
        LocalTensor<float>& abR, LocalTensor<float>& abI, LocalTensor<float>& cR, LocalTensor<float>& cI,
        LocalTensor<float>& tmp, float tA[8], float uA[8], int32_t vecCount, int32_t tail)
    {
        LocalTensor<float> ta = kSplit_ ? cR : abR;
        LocalTensor<float> ua = kSplit_ ? cI : abI;
        if (vecCount > 0) {
            Sub(ta, w2aLocal, w3aLocal, vecCount);      // t_a
            Sub(tmp, w1aLocal, w2aLocal, vecCount);
            Sub(ua, tmp, w3aLocal, vecCount);           // u_a
        }
        for (int32_t i = 0; i < tail; i++) {
            const int32_t idx = vecCount + i;
            tA[i] = w2aLocal.GetValue(idx) - w3aLocal.GetValue(idx);
            uA[i] = w1aLocal.GetValue(idx) - w2aLocal.GetValue(idx) - w3aLocal.GetValue(idx);
        }
    }

    // Stage segment-b (t_b/u_b) and accumulate onto abR/abI (kSplit only).
    __aicore__ inline void CsymmStageSegB(
        LocalTensor<float>& w1bLocal, LocalTensor<float>& w2bLocal, LocalTensor<float>& w3bLocal,
        LocalTensor<float>& abR, LocalTensor<float>& abI, LocalTensor<float>& cR, LocalTensor<float>& cI,
        LocalTensor<float>& tmp, float tB[8], float uB[8], int32_t vecCount, int32_t tail)
    {
        if (vecCount > 0) {
            Sub(tmp, w2bLocal, w3bLocal, vecCount); // t_b
            Add(abR, cR, tmp, vecCount);            // abR = t_a + t_b
            Sub(tmp, w1bLocal, w2bLocal, vecCount);
            Sub(tmp, tmp, w3bLocal, vecCount);      // u_b
            Add(abI, cI, tmp, vecCount);            // abI = u_a + u_b
        }
        for (int32_t i = 0; i < tail; i++) {
            const int32_t idx = vecCount + i;
            tB[i] = w2bLocal.GetValue(idx) - w3bLocal.GetValue(idx);
            uB[i] = w1bLocal.GetValue(idx) - w2bLocal.GetValue(idx) - w3bLocal.GetValue(idx);
        }
    }

    // Stage t_a/u_a (segment a) and, when K is split, t_b/u_b (segment b) into the
    // scratch planes / scalar arrays. abR/abI hold the combined t_a+t_b for the
    // non-kSplit case; cR/cI hold t_a for the kSplit case (later merged by b).
    __aicore__ inline void CombineAccumulate(
        LocalTensor<float>& w1aLocal, LocalTensor<float>& w2aLocal, LocalTensor<float>& w3aLocal,
        LocalTensor<float>& abR, LocalTensor<float>& abI, LocalTensor<float>& cR, LocalTensor<float>& cI,
        LocalTensor<float>& tmp,
        float tA[8], float uA[8], float tB[8], float uB[8],
        int32_t col, int32_t rowOffset, int32_t count, int32_t vecCount, int32_t tail)
    {
        const uint64_t tempOffset = static_cast<uint64_t>(col) * tempLdc_ + rowOffset;
        DataCopyExtParams ext{1, static_cast<uint32_t>(count) * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        LocalTensor<float> w1aTile = t1Que_.AllocTensor<float>();
        DataCopyPad(w1aTile, w1aGlobal_[tempOffset], ext, padParams);
        LocalTensor<float> w2aTile = t1Que_.AllocTensor<float>();
        DataCopyPad(w2aTile, w2aGlobal_[tempOffset], ext, padParams);
        LocalTensor<float> w3aTile = t2Que_.AllocTensor<float>();
        DataCopyPad(w3aTile, w3aGlobal_[tempOffset], ext, padParams);
        t1Que_.EnQue(w1aTile);
        t1Que_.EnQue(w2aTile);
        t2Que_.EnQue(w3aTile);
        w1aLocal = t1Que_.DeQue<float>();
        w2aLocal = t1Que_.DeQue<float>();
        w3aLocal = t2Que_.DeQue<float>();
        CsymmStageSegA(w1aLocal, w2aLocal, w3aLocal, abR, abI, cR, cI, tmp, tA, uA, vecCount, tail);
        t1Que_.FreeTensor(w1aLocal);
        t1Que_.FreeTensor(w2aLocal);
        t2Que_.FreeTensor(w3aLocal);
        if (kSplit_) {
            // Parallel issue of the three b-plane reads, like segment a.
            LocalTensor<float> w1bTile = t1Que_.AllocTensor<float>();
            DataCopyPad(w1bTile, w1bGlobal_[tempOffset], ext, padParams);
            LocalTensor<float> w2bTile = t1Que_.AllocTensor<float>();
            DataCopyPad(w2bTile, w2bGlobal_[tempOffset], ext, padParams);
            LocalTensor<float> w3bTile = t2Que_.AllocTensor<float>();
            DataCopyPad(w3bTile, w3bGlobal_[tempOffset], ext, padParams);
            t1Que_.EnQue(w1bTile);
            t1Que_.EnQue(w2bTile);
            t2Que_.EnQue(w3bTile);
            LocalTensor<float> w1bLocal = t1Que_.DeQue<float>();
            LocalTensor<float> w2bLocal = t1Que_.DeQue<float>();
            LocalTensor<float> w3bLocal = t2Que_.DeQue<float>();
            CsymmStageSegB(w1bLocal, w2bLocal, w3bLocal, abR, abI, cR, cI, tmp, tB, uB, vecCount, tail);
            t1Que_.FreeTensor(w1bLocal);
            t1Que_.FreeTensor(w2bLocal);
            t2Que_.FreeTensor(w3bLocal);
        }
    }

    // Load complex C (beta != 0), de-interleave, accumulate the combined abR/abI
    // planes, and write the per-element result into outBuf (vector + scalar tail).
    __aicore__ inline void CsymmEmitLoadAccumC(
        LocalTensor<float>& cLocal, LocalTensor<float>& cR, LocalTensor<float>& cI,
        LocalTensor<float>& abR, LocalTensor<float>& abI,
        LocalTensor<float>& outR, LocalTensor<float>& outI, LocalTensor<float>& tmp,
        LocalTensor<float>& outBuf,
        const float tA[8], const float uA[8], const float tB[8], const float uB[8],
        int32_t count, int32_t vecCount, int32_t tail)
    {
        if (vecCount > 0) {
            DeInterleave(cR, cI, cLocal, 2 * count);
            // outR = ar*abR - ai*abI + br*cR - bi*cI
            Muls(outR, abR, ar_, vecCount);
            Muls(tmp, abI, ai_, vecCount);
            Sub(outR, outR, tmp, vecCount);
            Muls(tmp, cR, br_, vecCount);
            Add(outR, outR, tmp, vecCount);
            Muls(tmp, cI, bi_, vecCount);
            Sub(outR, outR, tmp, vecCount);
            // outI = ar*abI + ai*abR + br*cI + bi*cR
            Muls(outI, abI, ar_, vecCount);
            Muls(tmp, abR, ai_, vecCount);
            Add(outI, outI, tmp, vecCount);
            Muls(tmp, cI, br_, vecCount);
            Add(outI, outI, tmp, vecCount);
            Muls(tmp, cR, bi_, vecCount);
            Add(outI, outI, tmp, vecCount);
        }
        for (int32_t i = 0; i < tail; i++) {
            const int32_t idx = vecCount + i;
            const float ab_r = kSplit_ ? (tA[i] + tB[i]) : tA[i];
            const float ab_i = kSplit_ ? (uA[i] + uB[i]) : uA[i];
            const float c_r = cLocal.GetValue(2 * idx);
            const float c_i = cLocal.GetValue(2 * idx + 1);
            const float out_r = ar_ * ab_r - ai_ * ab_i + br_ * c_r - bi_ * c_i;
            const float out_i = ar_ * ab_i + ai_ * ab_r + br_ * c_i + bi_ * c_r;
            outBuf.SetValue(2 * idx, out_r);
            outBuf.SetValue(2 * idx + 1, out_i);
        }
    }

    // Interleave the real/imag planes back and write the combined tile to GM.
    __aicore__ inline void CsymmEmitInterleaveWrite(
        LocalTensor<float> outR, LocalTensor<float> outI, LocalTensor<float> outBuf,
        const uint64_t cOffset, const DataCopyExtParams& cplxExt, int32_t vecCount)
    {
        if (vecCount > 0) {
            LocalTensor<float> dst0 = outBuf[0];
            LocalTensor<float> dst1 = outBuf[vecCount];
            Interleave(dst0, dst1, outR, outI, vecCount);
        }
        outQue_.EnQue(outBuf);
        LocalTensor<float> oLocal = outQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(cGlobal_[cOffset], oLocal, cplxExt);
        outQue_.FreeTensor(oLocal);
    }

    // De-interleave C (beta != 0), apply alpha/beta, interleave and write back.
    __aicore__ inline void CombineEmit(
        LocalTensor<float> abR, LocalTensor<float> abI,
        LocalTensor<float> cR, LocalTensor<float> cI,
        LocalTensor<float> outR, LocalTensor<float> outI, LocalTensor<float> tmp,
        LocalTensor<float> outBuf,
        const float tA[8], const float uA[8], const float tB[8], const float uB[8],
        int32_t col, int32_t rowOffset, int32_t count, int32_t vecCount, int32_t tail)
    {
        const uint64_t cOffset = static_cast<uint64_t>(col) * ldc_ * 2 + rowOffset * 2;
        const uint32_t cplxBytes = static_cast<uint32_t>(count) * 2 * sizeof(float);
        DataCopyExtParams cplxExt{1, cplxBytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        const bool betaZero = (br_ == 0.0f && bi_ == 0.0f);
        if (!betaZero) {
            // ---- load complex C ----
            LocalTensor<float> cTile = cQue_.AllocTensor<float>();
            DataCopyPad(cTile, cGlobal_[cOffset], cplxExt, padParams);
            cQue_.EnQue(cTile);
            LocalTensor<float> cLocal = cQue_.DeQue<float>();
            CsymmEmitLoadAccumC(cLocal, cR, cI, abR, abI, outR, outI, tmp, outBuf,
                                tA, uA, tB, uB, count, vecCount, tail);
            cQue_.FreeTensor(cLocal);
        } else {
            EmitAlphaCombined(abR, abI, outR, outI, tmp, outBuf, vecCount, tail, tA, uA, tB, uB);
        }
        CsymmEmitInterleaveWrite(outR, outI, outBuf, cOffset, cplxExt, vecCount);
    }

    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        LocalTensor<float> scratch = calcBuf_.Get<float>();
        LocalTensor<float> abR = scratch[0];
        LocalTensor<float> abI = scratch[COMBINE_TILE];
        LocalTensor<float> cR = scratch[2 * COMBINE_TILE];
        LocalTensor<float> cI = scratch[3 * COMBINE_TILE];
        LocalTensor<float> outR = scratch[4 * COMBINE_TILE];
        LocalTensor<float> outI = scratch[5 * COMBINE_TILE];
        LocalTensor<float> tmp = scratch[6 * COMBINE_TILE];
        LocalTensor<float> outBuf = outQue_.AllocTensor<float>();
        const int32_t vecCount = count & ~7; // vector ops need a multiple of 8 floats
        const int32_t tail = count - vecCount;
        float tA[8] = {0.0f};
        float uA[8] = {0.0f};
        float tB[8] = {0.0f};
        float uB[8] = {0.0f};
        LocalTensor<float> w1aLocal;
        LocalTensor<float> w2aLocal;
        LocalTensor<float> w3aLocal;
        CombineAccumulate(w1aLocal, w2aLocal, w3aLocal, abR, abI, cR, cI, tmp,
            tA, uA, tB, uB, col, rowOffset, count, vecCount, tail);
        CombineEmit(abR, abI, cR, cI, outR, outI, tmp, outBuf,
            tA, uA, tB, uB, col, rowOffset, count, vecCount, tail);
    }

    AscendC::TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, COMBINE_BUF_NUM> t1Que_;
    TQue<QuePosition::VECIN, COMBINE_BUF_NUM> t2Que_;
    TQue<QuePosition::VECIN, COMBINE_BUF_NUM> cQue_;
    TQue<QuePosition::VECOUT, COMBINE_BUF_NUM> outQue_;
    TBuf<QuePosition::VECCALC> calcBuf_;
    GlobalTensor<float> w1aGlobal_;
    GlobalTensor<float> w2aGlobal_;
    GlobalTensor<float> w3aGlobal_;
    GlobalTensor<float> w1bGlobal_;
    GlobalTensor<float> w2bGlobal_;
    GlobalTensor<float> w3bGlobal_;
    GlobalTensor<float> cGlobal_;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldc_ = 0;
    int32_t tempLdc_ = 0;
    bool kSplit_ = false;
    float ar_ = 1.0f;
    float ai_ = 0.0f;
    float br_ = 0.0f;
    float bi_ = 0.0f;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
    int32_t colsPerCore_ = 0;
};

} // namespace csymm_combine

extern "C" __global__ __aicore__ void csymm_combine_kernel(
    __gm__ uint8_t* w1a, __gm__ uint8_t* w2a, __gm__ uint8_t* w3a,
    __gm__ uint8_t* w1b, __gm__ uint8_t* w2b, __gm__ uint8_t* w3b,
    int32_t kSplit, int32_t tempLdc,
    __gm__ uint8_t* cInOut, int32_t m, int32_t n, int32_t ldc,
    float ar, float ai, float br, float bi)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    csymm_combine::CsymmCombineKernel op;
    op.Init(w1a, w2a, w3a, w1b, w2b, w3b, kSplit, tempLdc,
            cInOut, m, n, ldc, ar, ai, br, bi, &pipe);
    op.Process();
}

// ============================================================================
// Kernel launchers
// ============================================================================

void csymm_cube_gemm_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR c1,
    const GemmTilingData& tilingData)
{
    csymm_cube_gemm_kernel<<<numBlocks, nullptr, stream>>>(a, b, c, c1, tilingData);
}

// Fused 3-GEMM launch wrapper (see csymm_cube_gemm3_kernel). perGemmBlocks is
// the per-GEMM block count (tiling.usedCoreNum); the launch uses 3x blocks.
void csymm_cube_gemm3_do(
    uint32_t perGemmBlocks, void* stream,
    GM_ADDR a0, GM_ADDR b0, GM_ADDR c0, GM_ADDR d0,
    GM_ADDR a1, GM_ADDR b1, GM_ADDR c1_, GM_ADDR d1,
    GM_ADDR a2, GM_ADDR b2, GM_ADDR c2, GM_ADDR d2,
    const GemmTilingData& tilingData)
{
    csymm_cube_gemm3_kernel<<<3u * perGemmBlocks, nullptr, stream>>>(
        a0, b0, c0, d0, a1, b1, c1_, d1, a2, b2, c2, d2, tilingData);
}

void csymm_scale_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR cInOut, int32_t m, int32_t n, int32_t ldc,
    float betaReal, float betaImag, int32_t isComplex)
{
    csymm_scale_kernel<<<numBlocks, nullptr, stream>>>(
        cInOut, m, n, ldc, betaReal, betaImag, isComplex);
}

void csymm_combine_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR w1a, GM_ADDR w2a, GM_ADDR w3a,
    GM_ADDR w1b, GM_ADDR w2b, GM_ADDR w3b, int32_t kSplit, int32_t tempLdc,
    GM_ADDR cInOut, int32_t m, int32_t n, int32_t ldc,
    float ar, float ai, float br, float bi)
{
    csymm_combine_kernel<<<numBlocks, nullptr, stream>>>(
        w1a, w2a, w3a, w1b, w2b, w3b, kSplit, tempLdc,
        cInOut, m, n, ldc, ar, ai, br, bi);
}

#endif
