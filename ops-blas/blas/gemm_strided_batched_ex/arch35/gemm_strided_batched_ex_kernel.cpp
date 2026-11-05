/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_strided_batched_ex_kernel.cpp
 * \brief Independent strided-batched GEMM kernel for arch35 (DAV-3510).
 *
 * SIMD membase implementation using BlockMmad low-level API.
 * C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
 *
 * Batch pointers are derived from base + batch * stride in every device kernel.
 */

#include "kernel_operator.h"
#include <type_traits>
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#define ASCENDC_CUBE_ONLY
#include "gemm_strided_batched_ex_tiling_data.h"
#include "common/arch/hardware.h"
#include "common/helper/kernel_utils.h"

// ============================================================================
// GemmStridedCubeState — POD runtime state for strided cube kernel
// ============================================================================
struct GemmStridedCubeState {
    GemmStridedBatchedExTilingData tiling;
    uint32_t actualM;
    uint32_t actualN;
    uint32_t baseMCount;
    uint32_t mnTasks;
    uint32_t mBlockIdx;
    uint32_t nBlockIdx;
    uint32_t tailM;
    uint32_t tailMAlign;
    uint32_t baseNCount;
    uint32_t tailN;
    uint32_t tailNAlign;
    uint32_t mLoopCount;
    uint32_t nLoopCount;
    uint32_t kLoopCount;
    uint32_t batchIdx;
    uint32_t splitIdx;
    uint32_t kStart;
    uint32_t kLength;
    uint64_t aBaseOffset;
    uint64_t bBaseOffset;
    uint64_t cBaseOffset;
    uint32_t mBlockSize;
    uint32_t nBlockSize;
    uint32_t mBlockCount;
    uint32_t nBlockCount;
};

// ============================================================================
// Standalone __aicore__ helper functions
// ============================================================================

__aicore__ inline bool InitStridedTilingBase(GemmStridedCubeState& st, GemmStridedBatchedExTilingData tiling)
{
    st.tiling = tiling;
    if (st.tiling.mBlocks <= 0 || st.tiling.nBlocks <= 0) {
        return false;
    }
    const uint64_t mnTasks = static_cast<uint64_t>(st.tiling.mBlocks) * static_cast<uint64_t>(st.tiling.nBlocks);
    if (mnTasks > UINT32_MAX) {
        return false;
    }
    st.mnTasks = static_cast<uint32_t>(mnTasks);
    return true;
}

__aicore__ inline void ComputeStridedBaseOffsets(GemmStridedCubeState& st)
{
    if (!st.tiling.isTransB) {
        st.bBaseOffset = static_cast<uint64_t>(st.nBlockIdx) * st.tiling.singleCoreN;
    } else {
        st.bBaseOffset = static_cast<uint64_t>(st.nBlockIdx) * st.tiling.singleCoreN * st.tiling.ldb;
    }
    if (!st.tiling.isTransA) {
        st.aBaseOffset = static_cast<uint64_t>(st.mBlockIdx) * st.tiling.singleCoreM * st.tiling.lda;
    } else {
        st.aBaseOffset = static_cast<uint64_t>(st.mBlockIdx) * st.tiling.singleCoreM;
    }
    st.cBaseOffset = static_cast<uint64_t>(st.mBlockIdx) * st.tiling.ldc * st.tiling.singleCoreM +
                     static_cast<uint64_t>(st.nBlockIdx) * st.tiling.singleCoreN;
}

template <uint32_t BASE_M, uint32_t BASE_N>
__aicore__ inline void ComputeStrided2DBlocking(GemmStridedCubeState& st)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    constexpr uint32_t MAX_C0_TILES = HardwareInfo<ArchType::ASCEND_V350>::l0CSize / C0_TILE_BYTES;
    static_assert(MAX_C0_TILES > 0, "L0C is too small for one output tile");
    st.mBlockSize = st.mLoopCount;
    st.nBlockSize = st.nLoopCount;
    while (static_cast<uint64_t>(st.mBlockSize) * st.nBlockSize > MAX_C0_TILES) {
        if (st.mBlockSize < st.nBlockSize) {
            st.nBlockSize = (st.nBlockSize + 1) / 2;
        } else {
            st.mBlockSize = (st.mBlockSize + 1) / 2;
        }
    }
    // R8: mBlockSize/nBlockSize >= 1 guaranteed (mLoopCount/nLoopCount >= 1 when actualM/N > 0)
    if (st.nBlockSize == 0) {
        st.nBlockSize = 1;
    }
    if (st.mBlockSize == 0) {
        st.mBlockSize = 1;
    }
    st.mBlockCount = CeilDiv(st.mLoopCount, st.mBlockSize);
    st.nBlockCount = CeilDiv(st.nLoopCount, st.nBlockSize);
}

__aicore__ inline void DecodeStridedTask(GemmStridedCubeState& st, uint64_t taskId, uint32_t& mnTask)
{
    const uint32_t batchCount = static_cast<uint32_t>(st.tiling.batchCount);
    const uint32_t splitK = static_cast<uint32_t>(st.tiling.splitK);
    const uint64_t batchSplit = static_cast<uint64_t>(batchCount) * splitK;
    uint64_t batchSplitTask = 0;
    switch (st.tiling.selectedAlgo) {
        case GEMM_STRIDED_KERNEL_BATCH_FIRST:
        case GEMM_STRIDED_KERNEL_PERSISTENT:
            batchSplitTask = taskId % batchSplit;
            mnTask = static_cast<uint32_t>(taskId / batchSplit);
            break;
        default:
            mnTask = static_cast<uint32_t>(taskId % st.mnTasks);
            batchSplitTask = taskId / st.mnTasks;
            break;
    }
    st.batchIdx = static_cast<uint32_t>(batchSplitTask / splitK);
    st.splitIdx = static_cast<uint32_t>(batchSplitTask % splitK);

    if (st.tiling.selectedAlgo == GEMM_STRIDED_KERNEL_M_MAJOR) {
        st.mBlockIdx = mnTask / static_cast<uint32_t>(st.tiling.nBlocks);
        st.nBlockIdx = mnTask % static_cast<uint32_t>(st.tiling.nBlocks);
    } else {
        st.mBlockIdx = mnTask % static_cast<uint32_t>(st.tiling.mBlocks);
        st.nBlockIdx = mnTask / static_cast<uint32_t>(st.tiling.mBlocks);
    }
}

template <uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline void InitStridedTaskState(GemmStridedCubeState& st, uint64_t taskId)
{
    constexpr uint32_t CUBE_BLOCK = HardwareInfo<ArchType::ASCEND_V350>::l1l0BlockSize;
    uint32_t mnTask = 0;
    DecodeStridedTask(st, taskId, mnTask);
    st.actualM = static_cast<uint32_t>(st.tiling.m) - st.mBlockIdx * static_cast<uint32_t>(st.tiling.singleCoreM);
    if (st.actualM > static_cast<uint32_t>(st.tiling.singleCoreM)) {
        st.actualM = static_cast<uint32_t>(st.tiling.singleCoreM);
    }
    st.actualN = static_cast<uint32_t>(st.tiling.n) - st.nBlockIdx * static_cast<uint32_t>(st.tiling.singleCoreN);
    if (st.actualN > static_cast<uint32_t>(st.tiling.singleCoreN)) {
        st.actualN = static_cast<uint32_t>(st.tiling.singleCoreN);
    }
    st.mLoopCount = CeilDiv(st.actualM, BASE_M);
    st.nLoopCount = CeilDiv(st.actualN, BASE_N);
    const uint32_t fullK = static_cast<uint32_t>(st.tiling.k);
    st.kStart =
        static_cast<uint32_t>((static_cast<uint64_t>(fullK) * st.splitIdx) / static_cast<uint32_t>(st.tiling.splitK));
    const uint32_t kEnd = static_cast<uint32_t>(
        (static_cast<uint64_t>(fullK) * (st.splitIdx + 1U)) / static_cast<uint32_t>(st.tiling.splitK));
    st.kLength = kEnd - st.kStart;
    st.kLoopCount = CeilDiv(st.kLength, BASE_K);
    st.baseMCount = st.actualM / BASE_M;
    st.tailM = st.actualM % BASE_M;
    st.tailMAlign = RoundUp(st.tailM, CUBE_BLOCK);
    st.baseNCount = st.actualN / BASE_N;
    st.tailN = st.actualN % BASE_N;
    st.tailNAlign = RoundUp(st.tailN, CUBE_BLOCK);
    ComputeStridedBaseOffsets(st);
    ComputeStrided2DBlocking<BASE_M, BASE_N>(st);
}

template <typename A_TYPE, uint32_t BASE_M, uint32_t BASE_K, uint32_t C0_VAL>
__aicore__ inline void LoadAStridedTile(
    const GemmStridedCubeState& st, AscendC::GlobalTensor<A_TYPE>& aGM, AscendC::LocalTensor<A_TYPE>& a1,
    AscendC::LocalTensor<A_TYPE>& a2, uint32_t mi, uint32_t kIdx, uint32_t curM, uint32_t curK, uint32_t curMAlign)
{
    constexpr uint32_t CUBE_BLOCK = HardwareInfo<ArchType::ASCEND_V350>::l1l0BlockSize;
    uint64_t aOffset;
    AscendC::Nd2NzParams ndA;
    ndA.ndNum = 1;
    ndA.dstNzNStride = 1;
    ndA.dstNzMatrixStride = 0;
    if (!st.tiling.isTransA) {
        aOffset = st.aBaseOffset + static_cast<uint64_t>(mi) * BASE_M * st.tiling.lda +
                  static_cast<uint64_t>(st.kStart + kIdx * BASE_K);
        ndA.nValue = curM;
        ndA.dValue = curK;
        ndA.srcNdMatrixStride = 0;
        ndA.srcDValue = static_cast<uint32_t>(st.tiling.lda);
        ndA.dstNzC0Stride = RoundUp(curM, CUBE_BLOCK);
    } else {
        aOffset = st.aBaseOffset + static_cast<uint64_t>(st.kStart + kIdx * BASE_K) * st.tiling.lda +
                  static_cast<uint64_t>(mi) * BASE_M;
        ndA.nValue = curK;
        ndA.dValue = curM;
        ndA.srcNdMatrixStride = 0;
        ndA.srcDValue = static_cast<uint32_t>(st.tiling.lda);
        ndA.dstNzC0Stride = RoundUp(BASE_K, CUBE_BLOCK);
    }
    AscendC::DataCopy(a1, aGM[aOffset], ndA);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::LoadData2DParamsV2 ldA;
    ldA.mStartPosition = 0;
    ldA.kStartPosition = 0;
    ldA.sid = 0;
    if (st.tiling.isTransA) {
        ldA.mStep = CeilDiv(BASE_K, CUBE_BLOCK);
        ldA.kStep = RoundUp(CeilDiv(curMAlign, C0_VAL), 2u);
        ldA.srcStride = CeilDiv(BASE_K, CUBE_BLOCK);
        ldA.dstStride = CeilDiv(curMAlign, CUBE_BLOCK);
        ldA.ifTranspose = true;
    } else {
        ldA.mStep = CeilDiv(curMAlign, CUBE_BLOCK);
        ldA.kStep = CeilDiv(BASE_K, C0_VAL);
        ldA.srcStride = CeilDiv(curMAlign, CUBE_BLOCK);
        ldA.dstStride = CeilDiv(curMAlign, CUBE_BLOCK);
        ldA.ifTranspose = false;
    }
    AscendC::LoadData(a2, a1, ldA);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename B_TYPE, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline void LoadBStridedTile(
    const GemmStridedCubeState& st, AscendC::GlobalTensor<B_TYPE>& bGM, AscendC::LocalTensor<B_TYPE>& b1,
    AscendC::LocalTensor<B_TYPE>& b2, uint32_t ni, uint32_t kIdx, uint32_t curK, uint32_t curN, uint32_t curNAlign)
{
    constexpr uint32_t CUBE_BLOCK = HardwareInfo<ArchType::ASCEND_V350>::l1l0BlockSize;
    uint64_t bOffset;
    AscendC::Nd2NzParams ndB;
    ndB.ndNum = 1;
    ndB.dstNzNStride = 1;
    ndB.dstNzMatrixStride = 0;
    if (!st.tiling.isTransB) {
        bOffset = st.bBaseOffset + static_cast<uint64_t>(st.kStart + kIdx * BASE_K) * st.tiling.ldb +
                  static_cast<uint64_t>(ni) * BASE_N;
        ndB.nValue = curK;
        ndB.dValue = curN;
        ndB.srcNdMatrixStride = 0;
        ndB.srcDValue = static_cast<uint32_t>(st.tiling.ldb);
        ndB.dstNzC0Stride = RoundUp(BASE_K, CUBE_BLOCK);
    } else {
        bOffset = st.bBaseOffset + static_cast<uint64_t>(ni) * BASE_N * st.tiling.ldb +
                  static_cast<uint64_t>(st.kStart + kIdx * BASE_K);
        ndB.nValue = curN;
        ndB.dValue = curK;
        ndB.srcNdMatrixStride = 0;
        ndB.dstNzC0Stride = RoundUp(curN, CUBE_BLOCK);
        ndB.srcDValue = static_cast<uint32_t>(st.tiling.ldb);
    }
    AscendC::DataCopy(b1, bGM[bOffset], ndB);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::LoadData2DParamsV2 ldB;
    ldB.mStartPosition = 0;
    ldB.kStartPosition = 0;
    ldB.sid = 0;
    if (st.tiling.isTransB) {
        ldB.mStep = CeilDiv(curNAlign, CUBE_BLOCK);
        ldB.kStep = CeilDiv(BASE_K, C0_VAL);
        ldB.srcStride = CeilDiv(curNAlign, CUBE_BLOCK);
        ldB.dstStride = CeilDiv(curNAlign, CUBE_BLOCK);
        ldB.ifTranspose = false;
    } else {
        ldB.mStep = CeilDiv(BASE_K, CUBE_BLOCK);
        ldB.kStep = CeilDiv(static_cast<uint32_t>(curNAlign * sizeof(B_TYPE)), 32u);
        ldB.srcStride = CeilDiv(BASE_K, CUBE_BLOCK);
        ldB.dstStride = CeilDiv(curNAlign, CUBE_BLOCK);
        ldB.ifTranspose = true;
    }
    AscendC::LoadData(b2, b1, ldB);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <
    typename A_TYPE, typename B_TYPE, typename ACC_TYPE, uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N,
    uint32_t C0_VAL>
__aicore__ inline void ProcessStridedNTile(
    const GemmStridedCubeState& st, AscendC::GlobalTensor<B_TYPE>& bGM, AscendC::LocalTensor<A_TYPE>& a2,
    AscendC::LocalTensor<B_TYPE>& b1, AscendC::LocalTensor<B_TYPE>& b2, uint32_t mi, uint32_t ni, uint32_t kIdx,
    uint32_t mStart, uint32_t nStart, uint32_t curM, uint32_t curK)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    if (ni != nStart) {
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    }
    uint32_t curN = (ni != st.baseNCount) ? BASE_N : st.tailN;
    uint32_t curNAlign = (ni != st.baseNCount) ? BASE_N : st.tailNAlign;
    LoadBStridedTile<B_TYPE, BASE_K, BASE_N, C0_VAL>(st, bGM, b1, b2, ni, kIdx, curK, curN, curNAlign);
    uint32_t cTileIdx = (mi - mStart) * st.nBlockSize + (ni - nStart);
    AscendC::LocalTensor<ACC_TYPE> c0(AscendC::TPosition::CO1, cTileIdx * C0_TILE_BYTES, BASE_M * BASE_N);
    AscendC::MmadParams mp{};
    mp.m = curM;
    mp.n = curN;
    mp.k = curK;
    mp.cmatrixInitVal = (kIdx == 0);
    AscendC::Mmad(c0, a2, b2, mp);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
}

template <typename C_GM_TYPE, typename ACC_TYPE, uint32_t BASE_M, uint32_t BASE_N, QuantMode_t QUANT_MODE>
__aicore__ inline void WriteStridedFixpipe(
    const GemmStridedCubeState& st, AscendC::GlobalTensor<C_GM_TYPE>& cGM, uint32_t mStart, uint32_t mEnd,
    uint32_t nStart, uint32_t nEnd)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    for (uint32_t mi = mStart; mi < mEnd; mi++) {
        uint32_t curM = (mi != st.baseMCount) ? BASE_M : st.tailM;
        uint32_t curMAlign = (mi != st.baseMCount) ? BASE_M : st.tailMAlign;
        for (uint32_t ni = nStart; ni < nEnd; ni++) {
            uint32_t curN = (ni != st.baseNCount) ? BASE_N : st.tailN;
            uint32_t cTileIdx = (mi - mStart) * st.nBlockSize + (ni - nStart);
            AscendC::LocalTensor<ACC_TYPE> c0(AscendC::TPosition::CO1, cTileIdx * C0_TILE_BYTES, BASE_M * BASE_N);
            uint64_t cOffset = st.cBaseOffset + static_cast<uint64_t>(mi) * BASE_M * st.tiling.ldc +
                               static_cast<uint64_t>(ni) * BASE_N;
            AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp;
            fp.mSize = curM;
            fp.nSize = curN;
            fp.srcStride = curMAlign;
            fp.dstStride = static_cast<uint32_t>(st.tiling.ldc);
            fp.quantPre = QUANT_MODE;
            AscendC::Fixpipe(cGM[cOffset], c0, fp);
            AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);
        }
    }
}

// ============================================================================
// RunKSlice — Process one K slice across M×N tiles (depth 2: mi→ni)
// ============================================================================
template <
    typename A_TYPE, typename B_TYPE, typename ACC_TYPE, uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N,
    uint32_t C0_VAL>
__aicore__ inline void RunKSlice(
    GemmStridedCubeState& st, AscendC::GlobalTensor<A_TYPE>& aGM, AscendC::GlobalTensor<B_TYPE>& bGM,
    AscendC::LocalTensor<A_TYPE>& a1, AscendC::LocalTensor<B_TYPE>& b1, AscendC::LocalTensor<A_TYPE>& a2,
    AscendC::LocalTensor<B_TYPE>& b2, uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd, uint32_t kIdx,
    uint32_t curK, bool& firstMmad)
{
    for (uint32_t mi = mStart; mi < mEnd; mi++) {
        if (!firstMmad) {
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        }
        firstMmad = false;
        uint32_t curM = (mi != st.baseMCount) ? BASE_M : st.tailM;
        uint32_t curMAlign = (mi != st.baseMCount) ? BASE_M : st.tailMAlign;
        LoadAStridedTile<A_TYPE, BASE_M, BASE_K, C0_VAL>(st, aGM, a1, a2, mi, kIdx, curM, curK, curMAlign);
        for (uint32_t ni = nStart; ni < nEnd; ni++) {
            ProcessStridedNTile<A_TYPE, B_TYPE, ACC_TYPE, BASE_M, BASE_K, BASE_N, C0_VAL>(
                st, bGM, a2, b1, b2, mi, ni, kIdx, mStart, nStart, curM, curK);
        }
    }
}

// ============================================================================
// RunStridedTask — Inner computation for one task (depth 3: mb→nb→kIdx)
// ============================================================================
template <
    typename A_TYPE, typename B_TYPE, typename C_GM_TYPE, typename ACC_TYPE, uint32_t BASE_M, uint32_t BASE_K,
    uint32_t BASE_N, uint32_t C0_VAL, QuantMode_t QMODE>
__aicore__ inline void RunStridedTask(
    GemmStridedCubeState& st, AscendC::GlobalTensor<A_TYPE>& aGM, AscendC::GlobalTensor<B_TYPE>& bGM,
    AscendC::GlobalTensor<C_GM_TYPE>& cGM, AscendC::LocalTensor<A_TYPE>& a1, AscendC::LocalTensor<B_TYPE>& b1,
    AscendC::LocalTensor<A_TYPE>& a1Alt, AscendC::LocalTensor<B_TYPE>& b1Alt, AscendC::LocalTensor<A_TYPE>& a2,
    AscendC::LocalTensor<B_TYPE>& b2)
{
    bool firstMmad = true;
    for (uint32_t mb = 0; mb < st.mBlockCount; mb++) {
        for (uint32_t nb = 0; nb < st.nBlockCount; nb++) {
            uint32_t mStart = mb * st.mBlockSize;
            uint32_t mEnd = (mStart + st.mBlockSize < st.mLoopCount) ? mStart + st.mBlockSize : st.mLoopCount;
            uint32_t nStart = nb * st.nBlockSize;
            uint32_t nEnd = (nStart + st.nBlockSize < st.nLoopCount) ? nStart + st.nBlockSize : st.nLoopCount;
            for (uint32_t kIdx = 0; kIdx < st.kLoopCount; kIdx++) {
                uint32_t curK = (kIdx == st.kLoopCount - 1) ? (st.kLength - kIdx * BASE_K) : BASE_K;
                if (st.tiling.selectedAlgo == GEMM_STRIDED_KERNEL_DEEP_K && (kIdx & 1U) != 0) {
                    RunKSlice<A_TYPE, B_TYPE, ACC_TYPE, BASE_M, BASE_K, BASE_N, C0_VAL>(
                        st, aGM, bGM, a1Alt, b1Alt, a2, b2, mStart, mEnd, nStart, nEnd, kIdx, curK, firstMmad);
                } else {
                    RunKSlice<A_TYPE, B_TYPE, ACC_TYPE, BASE_M, BASE_K, BASE_N, C0_VAL>(
                        st, aGM, bGM, a1, b1, a2, b2, mStart, mEnd, nStart, nEnd, kIdx, curK, firstMmad);
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            WriteStridedFixpipe<C_GM_TYPE, ACC_TYPE, BASE_M, BASE_N, QMODE>(st, cGM, mStart, mEnd, nStart, nEnd);
            firstMmad = true;
        }
    }
}

// ============================================================================
// GEMM_STRIDED_CUBE_KERNEL — Macro generating all strided cube kernel variants
//
// Supports separate A_TYPE and B_TYPE for FP8 mixed-type kernels.
// Task loop iterates over batch × mBlocks × nTasks with grid-stride.
//
// Parameters:
//   FUNC_NAME  - kernel function name
//   A_TYPE     - input A element type
//   B_TYPE     - input B element type
//   C_GM_TYPE  - output GM element type
//   ACC_TYPE   - MMAD accumulator type (float or int32_t)
//   BM, BK, BN - base tile dimensions
//   C0_VAL     - C0 parameter for LoadData
//   QUANT_MODE - Fixpipe quantization mode
// ============================================================================
#define GEMM_STRIDED_CUBE_KERNEL(FUNC_NAME, A_TYPE, B_TYPE, C_GM_TYPE, ACC_TYPE, BM, BK, BN, C0_VAL, QUANT_MODE) \
    __cube__ __global__ void FUNC_NAME(                                                                          \
        __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, GemmStridedBatchedExTilingData tiling)          \
    {                                                                                                            \
        AscendC::InitSocState();                                                                                 \
        GemmStridedCubeState st{};                                                                               \
        if (!InitStridedTilingBase(st, tiling)) {                                                                \
            return;                                                                                              \
        }                                                                                                        \
        constexpr uint32_t L1_SIZE = HardwareInfo<ArchType::ASCEND_V350>::l1Size;                                \
        constexpr uint32_t L1_HALF = L1_SIZE / 2;                                                                \
        /* DAV-3510 uses a 512-byte L1 offset to avoid A/B ping-pong bank conflicts. */                          \
        constexpr uint32_t L1_BANK_SKEW = 512;                                                                   \
        static_assert(L1_HALF > L1_BANK_SKEW, "L1 buffer must leave room for the bank skew");                    \
        constexpr uint32_t L1_SKEWED_BUFFER = L1_HALF - L1_BANK_SKEW;                                            \
        AscendC::LocalTensor<A_TYPE> a1(AscendC::TPosition::A1, 0, L1_SKEWED_BUFFER);                            \
        AscendC::LocalTensor<A_TYPE> a1Alt(AscendC::TPosition::A1, L1_HALF, L1_SKEWED_BUFFER);                   \
        AscendC::LocalTensor<B_TYPE> b1(AscendC::TPosition::B1, L1_SIZE + L1_BANK_SKEW, L1_SKEWED_BUFFER);       \
        AscendC::LocalTensor<B_TYPE> b1Alt(                                                                      \
            AscendC::TPosition::B1, L1_SIZE + L1_HALF + L1_BANK_SKEW, L1_SKEWED_BUFFER);                         \
        AscendC::LocalTensor<A_TYPE> a2(AscendC::TPosition::A2, 0, (BM) * (BK));                                 \
        AscendC::LocalTensor<B_TYPE> b2(AscendC::TPosition::B2, 0, (BK) * (BN));                                 \
        const uint64_t blockIdxVal = AscendC::GetBlockIdx();                                                     \
        const uint64_t gridDim = AscendC::GetBlockNum();                                                         \
        for (uint64_t taskId = blockIdxVal; taskId < st.tiling.totalTasks; taskId += gridDim) {                  \
            InitStridedTaskState<(BM), (BK), (BN), (C0_VAL)>(st, taskId);                                        \
            AscendC::GlobalTensor<A_TYPE> aGM;                                                                   \
            AscendC::GlobalTensor<B_TYPE> bGM;                                                                   \
            AscendC::GlobalTensor<C_GM_TYPE> cGM;                                                                \
            const int64_t aBatchOffset = static_cast<int64_t>(st.batchIdx) * st.tiling.strideA;                  \
            const int64_t bBatchOffset = static_cast<int64_t>(st.batchIdx) * st.tiling.strideB;                  \
            const int64_t cBatchOffset = static_cast<int64_t>(st.batchIdx) * st.tiling.workspaceBatchStride +    \
                                         static_cast<int64_t>(st.splitIdx) * st.tiling.workspaceSplitStride;     \
            aGM.SetGlobalBuffer(reinterpret_cast<__gm__ A_TYPE*>(a) + aBatchOffset);                             \
            bGM.SetGlobalBuffer(reinterpret_cast<__gm__ B_TYPE*>(b) + bBatchOffset);                             \
            cGM.SetGlobalBuffer(reinterpret_cast<__gm__ C_GM_TYPE*>(c) + cBatchOffset);                          \
            RunStridedTask<A_TYPE, B_TYPE, C_GM_TYPE, ACC_TYPE, (BM), (BK), (BN), (C0_VAL), QUANT_MODE>(         \
                st, aGM, bGM, cGM, a1, b1, a1Alt, b1Alt, a2, b2);                                                \
        }                                                                                                        \
        AscendC::PipeBarrier<PIPE_ALL>();                                                                        \
    }

// ALGO0-7 are separate entry points. ALGO4 uses a smaller spatial tile and
// ALGO5 doubles BK; the remaining algorithms differ in task decoding.
#define GEMM_FP16_ALGOS(SUFFIX, C_TYPE, QMODE)                                                           \
    GEMM_STRIDED_CUBE_KERNEL(                                                                            \
        gemm_strided_batched_ex_algo0_fp16_##SUFFIX, half, half, C_TYPE, float, 128, 16, 128, 16, QMODE) \
    GEMM_STRIDED_CUBE_KERNEL(                                                                            \
        gemm_strided_batched_ex_algo1_fp16_##SUFFIX, half, half, C_TYPE, float, 128, 16, 128, 16, QMODE) \
    GEMM_STRIDED_CUBE_KERNEL(                                                                            \
        gemm_strided_batched_ex_algo2_fp16_##SUFFIX, half, half, C_TYPE, float, 128, 16, 128, 16, QMODE) \
    GEMM_STRIDED_CUBE_KERNEL(                                                                            \
        gemm_strided_batched_ex_algo3_fp16_##SUFFIX, half, half, C_TYPE, float, 128, 16, 128, 16, QMODE) \
    GEMM_STRIDED_CUBE_KERNEL(                                                                            \
        gemm_strided_batched_ex_algo4_fp16_##SUFFIX, half, half, C_TYPE, float, 64, 16, 64, 16, QMODE)   \
    GEMM_STRIDED_CUBE_KERNEL(                                                                            \
        gemm_strided_batched_ex_algo5_fp16_##SUFFIX, half, half, C_TYPE, float, 64, 32, 64, 16, QMODE)   \
    GEMM_STRIDED_CUBE_KERNEL(                                                                            \
        gemm_strided_batched_ex_algo6_fp16_##SUFFIX, half, half, C_TYPE, float, 128, 16, 128, 16, QMODE) \
    GEMM_STRIDED_CUBE_KERNEL(                                                                            \
        gemm_strided_batched_ex_algo7_fp16_##SUFFIX, half, half, C_TYPE, float, 128, 16, 128, 16, QMODE)

GEMM_FP16_ALGOS(f16, half, QuantMode_t::F322F16)
GEMM_FP16_ALGOS(f32, float, QuantMode_t::NoQuant)

GEMM_STRIDED_CUBE_KERNEL(
    gemm_strided_batched_ex_algo0_bf16_bf16, bfloat16_t, bfloat16_t, bfloat16_t, float, 128, 16, 128, 16,
    QuantMode_t::F322BF16)
GEMM_STRIDED_CUBE_KERNEL(
    gemm_strided_batched_ex_algo0_bf16_f32, bfloat16_t, bfloat16_t, float, float, 128, 16, 128, 16,
    QuantMode_t::NoQuant)
GEMM_STRIDED_CUBE_KERNEL(
    gemm_strided_batched_ex_algo0_fp32_f32, float, float, float, float, 32, 8, 16, 8, QuantMode_t::NoQuant)
GEMM_STRIDED_CUBE_KERNEL(
    gemm_strided_batched_ex_algo0_e4m3_f16, fp8_e4m3fn_t, fp8_e4m3fn_t, half, float, 32, 32, 16, 32,
    QuantMode_t::F322F16)
GEMM_STRIDED_CUBE_KERNEL(
    gemm_strided_batched_ex_algo0_e5m2_f16, fp8_e5m2_t, fp8_e5m2_t, half, float, 32, 32, 16, 32, QuantMode_t::F322F16)
GEMM_STRIDED_CUBE_KERNEL(
    gemm_strided_batched_ex_algo0_e4m3_e5m2_f16, fp8_e5m2_t, fp8_e4m3fn_t, half, float, 32, 32, 16, 32,
    QuantMode_t::F322F16)
GEMM_STRIDED_CUBE_KERNEL(
    gemm_strided_batched_ex_algo0_e5m2_e4m3_f16, fp8_e4m3fn_t, fp8_e5m2_t, half, float, 32, 32, 16, 32,
    QuantMode_t::F322F16)
GEMM_STRIDED_CUBE_KERNEL(
    gemm_strided_batched_ex_algo0_e4m3_f32, fp8_e4m3fn_t, fp8_e4m3fn_t, float, float, 32, 32, 16, 32,
    QuantMode_t::NoQuant)
GEMM_STRIDED_CUBE_KERNEL(
    gemm_strided_batched_ex_algo0_e5m2_f32, fp8_e5m2_t, fp8_e5m2_t, float, float, 32, 32, 16, 32, QuantMode_t::NoQuant)
GEMM_STRIDED_CUBE_KERNEL(
    gemm_strided_batched_ex_algo0_e4m3_e5m2_f32, fp8_e5m2_t, fp8_e4m3fn_t, float, float, 32, 32, 16, 32,
    QuantMode_t::NoQuant)
GEMM_STRIDED_CUBE_KERNEL(
    gemm_strided_batched_ex_algo0_e5m2_e4m3_f32, fp8_e4m3fn_t, fp8_e5m2_t, float, float, 32, 32, 16, 32,
    QuantMode_t::NoQuant)
GEMM_STRIDED_CUBE_KERNEL(
    gemm_strided_batched_ex_algo0_int8_i32, int8_t, int8_t, int32_t, int32_t, 128, 32, 128, 32, QuantMode_t::NoQuant)

#define LAUNCH_FP16_ALGO(SUFFIX)                                                                              \
    switch (tilingData.selectedAlgo) {                                                                        \
        case GEMM_STRIDED_KERNEL_BALANCED:                                                                    \
            gemm_strided_batched_ex_algo0_fp16_##SUFFIX<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData); \
            break;                                                                                            \
        case GEMM_STRIDED_KERNEL_BATCH_FIRST:                                                                 \
            gemm_strided_batched_ex_algo1_fp16_##SUFFIX<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData); \
            break;                                                                                            \
        case GEMM_STRIDED_KERNEL_M_MAJOR:                                                                     \
            gemm_strided_batched_ex_algo2_fp16_##SUFFIX<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData); \
            break;                                                                                            \
        case GEMM_STRIDED_KERNEL_N_MAJOR:                                                                     \
            gemm_strided_batched_ex_algo3_fp16_##SUFFIX<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData); \
            break;                                                                                            \
        case GEMM_STRIDED_KERNEL_SMALL:                                                                       \
            gemm_strided_batched_ex_algo4_fp16_##SUFFIX<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData); \
            break;                                                                                            \
        case GEMM_STRIDED_KERNEL_DEEP_K:                                                                      \
            gemm_strided_batched_ex_algo5_fp16_##SUFFIX<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData); \
            break;                                                                                            \
        case GEMM_STRIDED_KERNEL_SPLIT_K:                                                                     \
            gemm_strided_batched_ex_algo6_fp16_##SUFFIX<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData); \
            break;                                                                                            \
        case GEMM_STRIDED_KERNEL_PERSISTENT:                                                                  \
            gemm_strided_batched_ex_algo7_fp16_##SUFFIX<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData); \
            break;                                                                                            \
        default:                                                                                              \
            break;                                                                                            \
    }

void gemm_strided_batched_ex_kernel_do(
    uint32_t numBlocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c,
    const GemmStridedBatchedExTilingData& tilingData, GemmStridedBatchedExDtypeCase dtypeCase)
{
    switch (dtypeCase) {
        case GEMM_STRIDED_DTYPE_FP16_F16:
            LAUNCH_FP16_ALGO(f16);
            break;
        case GEMM_STRIDED_DTYPE_FP16_F32:
            LAUNCH_FP16_ALGO(f32);
            break;
        case GEMM_STRIDED_DTYPE_BF16_BF16:
            gemm_strided_batched_ex_algo0_bf16_bf16<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_STRIDED_DTYPE_BF16_F32:
            gemm_strided_batched_ex_algo0_bf16_f32<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_STRIDED_DTYPE_FP32_F32:
            gemm_strided_batched_ex_algo0_fp32_f32<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_STRIDED_DTYPE_FP8_E4M3_F16:
            gemm_strided_batched_ex_algo0_e4m3_f16<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_STRIDED_DTYPE_FP8_E5M2_F16:
            gemm_strided_batched_ex_algo0_e5m2_f16<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_STRIDED_DTYPE_FP8_E4M3_E5M2_F16:
            gemm_strided_batched_ex_algo0_e4m3_e5m2_f16<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_STRIDED_DTYPE_FP8_E5M2_E4M3_F16:
            gemm_strided_batched_ex_algo0_e5m2_e4m3_f16<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_STRIDED_DTYPE_FP8_E4M3_F32:
            gemm_strided_batched_ex_algo0_e4m3_f32<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_STRIDED_DTYPE_FP8_E5M2_F32:
            gemm_strided_batched_ex_algo0_e5m2_f32<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_STRIDED_DTYPE_FP8_E4M3_E5M2_F32:
            gemm_strided_batched_ex_algo0_e4m3_e5m2_f32<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_STRIDED_DTYPE_FP8_E5M2_E4M3_F32:
            gemm_strided_batched_ex_algo0_e5m2_e4m3_f32<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_STRIDED_DTYPE_INT8_I32:
        case GEMM_STRIDED_DTYPE_INT8_F32:
            gemm_strided_batched_ex_algo0_int8_i32<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        default:
            break;
    }
}

namespace {

constexpr uint32_t VECTOR_THREADS = 256;

struct StridedLogicalParams {
    int32_t logicalM;
    int32_t logicalN;
    int32_t ldc;
    int32_t batchCount;
    int64_t strideC;
};

struct StridedComplexParams {
    StridedLogicalParams logical;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t isTransA;
    int32_t isTransB;
    int64_t strideA;
    int64_t strideB;
    float alpha;
    float beta;
    float alphaImag;
    float betaImag;
};

struct StridedEpilogueParams {
    StridedLogicalParams logical;
    int32_t splitK;
    int32_t hasBeta;
    int64_t workspaceBatchStride;
    int64_t workspaceSplitStride;
};

__simt_callee__ __aicore__ inline void StridedComplexAccumulate(
    __gm__ float* a, __gm__ float* b, const StridedComplexParams& tiling, int32_t batch, int32_t row, int32_t col,
    float& sumReal, float& sumImag)
{
    sumReal = 0.0f;
    sumImag = 0.0f;
    if (tiling.k <= 0) {
        return;
    }
    __gm__ float* aBatch = a + 2 * static_cast<int64_t>(batch) * tiling.strideA;
    __gm__ float* bBatch = b + 2 * static_cast<int64_t>(batch) * tiling.strideB;
    for (int32_t inner = 0; inner < tiling.k; ++inner) {
        const int64_t aOffset = tiling.isTransA == 0 ? static_cast<int64_t>(inner) * tiling.lda + row :
                                                       static_cast<int64_t>(row) * tiling.lda + inner;
        const int64_t bOffset = tiling.isTransB == 0 ? static_cast<int64_t>(col) * tiling.ldb + inner :
                                                       static_cast<int64_t>(inner) * tiling.ldb + col;
        const float avReal = aBatch[2 * aOffset];
        float avImag = aBatch[2 * aOffset + 1];
        const float bvReal = bBatch[2 * bOffset];
        float bvImag = bBatch[2 * bOffset + 1];
        if (tiling.isTransA == 2) {
            avImag = -avImag;
        }
        if (tiling.isTransB == 2) {
            bvImag = -bvImag;
        }
        sumReal += avReal * bvReal - avImag * bvImag;
        sumImag += avReal * bvImag + avImag * bvReal;
    }
}

__simt_callee__ __aicore__ inline void StridedComplexStore(
    __gm__ float* c, const StridedComplexParams& tiling, int32_t batch, int32_t row, int32_t col, float sumReal,
    float sumImag)
{
    __gm__ float* cBatch = c + 2 * static_cast<int64_t>(batch) * tiling.logical.strideC;
    const int64_t cOffset = 2 * (static_cast<int64_t>(col) * tiling.logical.ldc + row);
    float resultReal = tiling.alpha * sumReal - tiling.alphaImag * sumImag;
    float resultImag = tiling.alpha * sumImag + tiling.alphaImag * sumReal;
    if (tiling.beta != 0.0f || tiling.betaImag != 0.0f) {
        const float oldReal = cBatch[cOffset];
        const float oldImag = cBatch[cOffset + 1];
        resultReal += tiling.beta * oldReal - tiling.betaImag * oldImag;
        resultImag += tiling.beta * oldImag + tiling.betaImag * oldReal;
    }
    cBatch[cOffset] = resultReal;
    cBatch[cOffset + 1] = resultImag;
}

struct StridedLogicalOffset {
    int32_t batch;
    int32_t row;
    int32_t col;
    int64_t matrixIndex;
    int64_t cOffset;
};

struct StridedLogicalStep {
    int64_t batches;
    int32_t rows;
    int32_t cols;
};

__simt_callee__ __aicore__ inline bool InitStridedLogicalOffset(
    int64_t index, const StridedLogicalParams& tiling, StridedLogicalOffset& offset)
{
    const int64_t logicalMatrix = static_cast<int64_t>(tiling.logicalM) * tiling.logicalN;
    if (logicalMatrix <= 0 || tiling.logicalM <= 0 || tiling.logicalN <= 0) {
        return false;
    }
    offset.batch = static_cast<int32_t>(index / logicalMatrix);
    offset.matrixIndex = index - static_cast<int64_t>(offset.batch) * logicalMatrix;
    offset.col = static_cast<int32_t>(offset.matrixIndex / tiling.logicalM);
    offset.row = static_cast<int32_t>(offset.matrixIndex - static_cast<int64_t>(offset.col) * tiling.logicalM);
    offset.cOffset = static_cast<int64_t>(offset.batch) * tiling.strideC +
                     static_cast<int64_t>(offset.col) * tiling.ldc + offset.row;
    return true;
}

__simt_callee__ __aicore__ inline bool InitStridedLogicalStep(
    int64_t step, const StridedLogicalParams& tiling, StridedLogicalStep& logicalStep)
{
    const int64_t logicalMatrix = static_cast<int64_t>(tiling.logicalM) * tiling.logicalN;
    if (logicalMatrix <= 0 || tiling.logicalM <= 0 || tiling.logicalN <= 0) {
        return false;
    }
    logicalStep.batches = step / logicalMatrix;
    const int64_t matrixStep = step - logicalStep.batches * logicalMatrix;
    logicalStep.cols = static_cast<int32_t>(matrixStep / tiling.logicalM);
    logicalStep.rows = static_cast<int32_t>(matrixStep - static_cast<int64_t>(logicalStep.cols) * tiling.logicalM);
    return true;
}

__simt_callee__ __aicore__ inline void AdvanceStridedLogicalOffset(
    const StridedLogicalParams& tiling, const StridedLogicalStep& step, StridedLogicalOffset& offset)
{
    int32_t nextRow = offset.row + step.rows;
    int32_t colCarry = 0;
    if (nextRow >= tiling.logicalM) {
        nextRow -= tiling.logicalM;
        colCarry = 1;
    }
    int32_t nextCol = offset.col + step.cols + colCarry;
    int32_t batchCarry = 0;
    if (nextCol >= tiling.logicalN) {
        nextCol -= tiling.logicalN;
        batchCarry = 1;
    }
    offset.batch += static_cast<int32_t>(step.batches) + batchCarry;
    offset.row = nextRow;
    offset.col = nextCol;
    offset.matrixIndex = static_cast<int64_t>(nextCol) * tiling.logicalM + nextRow;
    offset.cOffset =
        static_cast<int64_t>(offset.batch) * tiling.strideC + static_cast<int64_t>(nextCol) * tiling.ldc + nextRow;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(VECTOR_THREADS) inline void StridedComplexGemmVf(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, int32_t k, int32_t lda, int32_t ldb, int32_t ldc,
    int32_t logicalM, int32_t logicalN, int32_t isTransA, int32_t isTransB, int32_t batchCount, int64_t strideA,
    int64_t strideB, int64_t strideC, float alpha, float beta, float alphaImag, float betaImag, uint32_t block,
    uint32_t blocks)
{
    const StridedComplexParams tiling{
        {logicalM, logicalN, ldc, batchCount, strideC},
        k,
        lda,
        ldb,
        isTransA,
        isTransB,
        strideA,
        strideB,
        alpha,
        beta,
        alphaImag,
        betaImag};
    const int64_t matrixElements = static_cast<int64_t>(logicalM) * logicalN;
    const int64_t total = matrixElements * batchCount;
    const int64_t thread = static_cast<int64_t>(block) * blockDim.x + threadIdx.x;
    const int64_t threadCount = static_cast<int64_t>(blocks) * blockDim.x;
    StridedLogicalOffset offset{};
    StridedLogicalStep step{};
    if (!InitStridedLogicalOffset(thread, tiling.logical, offset) ||
        !InitStridedLogicalStep(threadCount, tiling.logical, step)) {
        return;
    }
    for (int64_t index = thread; index < total; index += threadCount) {
        float sumReal = 0.0f;
        float sumImag = 0.0f;
        StridedComplexAccumulate(a, b, tiling, offset.batch, offset.row, offset.col, sumReal, sumImag);
        StridedComplexStore(c, tiling, offset.batch, offset.row, offset.col, sumReal, sumImag);
        AdvanceStridedLogicalOffset(tiling.logical, step, offset);
    }
}

template <typename T>
__simt_callee__ __aicore__ inline float StridedToFloat(T value)
{
    if constexpr (std::is_same<T, float>::value) {
        return value;
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        return __bfloat162float(value);
    } else {
        return __half2float(value);
    }
}

template <typename T>
__simt_callee__ __aicore__ inline T StridedFromFloat(float value)
{
    if constexpr (std::is_same<T, float>::value) {
        return value;
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        return __float2bfloat16_rn_sat(value);
    } else {
        return __float2half_rn_sat(value);
    }
}

struct StridedEpilogueLoop {
    int64_t logicalMatrix;
    int64_t total;
    int64_t index;
    int64_t threadCount;
};

struct StridedEpilogueState {
    StridedEpilogueLoop loop;
    StridedLogicalOffset offset;
    StridedLogicalStep step;
};

__simt_callee__ __aicore__ inline void InitStridedEpilogueLoop(
    const StridedLogicalParams& tiling, uint32_t block, uint32_t blocks, StridedEpilogueLoop& loop)
{
    loop.logicalMatrix = static_cast<int64_t>(tiling.logicalM) * tiling.logicalN;
    loop.total = loop.logicalMatrix * tiling.batchCount;
    loop.index = static_cast<int64_t>(block) * blockDim.x + threadIdx.x;
    loop.threadCount = static_cast<int64_t>(blocks) * blockDim.x;
}

__simt_callee__ __aicore__ inline bool InitStridedEpilogueState(
    const StridedLogicalParams& tiling, uint32_t block, uint32_t blocks, StridedEpilogueState& state)
{
    InitStridedEpilogueLoop(tiling, block, blocks, state.loop);
    return InitStridedLogicalOffset(state.loop.index, tiling, state.offset) &&
           InitStridedLogicalStep(state.loop.threadCount, tiling, state.step);
}

template <typename PARTIAL_TYPE, typename C_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(VECTOR_THREADS) inline void StridedFloatEpilogueVf(
    __gm__ PARTIAL_TYPE* partial, __gm__ C_TYPE* c, int32_t ldc, int32_t logicalM, int32_t logicalN, int32_t batchCount,
    int32_t splitK, int32_t hasBeta, int64_t strideC, int64_t workspaceBatchStride, int64_t workspaceSplitStride,
    float alpha, float beta, uint32_t block, uint32_t blocks, bool earlyExit)
{
    const StridedEpilogueParams tiling{
        {logicalM, logicalN, ldc, batchCount, strideC}, splitK, hasBeta, workspaceBatchStride, workspaceSplitStride};
    StridedEpilogueState state{};
    if (!InitStridedEpilogueState(tiling.logical, block, blocks, state)) {
        return;
    }
    for (; state.loop.index < state.loop.total; state.loop.index += state.loop.threadCount) {
        float result = 0.0f;
        if (!earlyExit) {
            const int64_t partialOffset =
                static_cast<int64_t>(state.offset.batch) * tiling.workspaceBatchStride + state.offset.matrixIndex;
            // Fixed split order makes ALGO6 reduction deterministic.
            for (int32_t split = 0; split < tiling.splitK; ++split) {
                result += static_cast<float>(
                    partial[partialOffset + static_cast<int64_t>(split) * tiling.workspaceSplitStride]);
            }
            result *= alpha;
        }
        if (tiling.hasBeta != 0) {
            result += beta * StridedToFloat(c[state.offset.cOffset]);
        }
        c[state.offset.cOffset] = StridedFromFloat<C_TYPE>(result);
        AdvanceStridedLogicalOffset(tiling.logical, state.step, state.offset);
    }
}

template <typename C_TYPE>
__aicore__ inline void LaunchStridedEpilogueVf(
    __gm__ uint8_t* partial, __gm__ uint8_t* c, GemmStridedBatchedExTilingData tiling, uint32_t blocks, bool earlyExit)
{
    asc_vf_call<StridedFloatEpilogueVf<float, C_TYPE>>(
        dim3{VECTOR_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(partial), reinterpret_cast<__gm__ C_TYPE*>(c),
        tiling.ldc, tiling.logicalM, tiling.logicalN, tiling.batchCount, tiling.splitK, tiling.hasBeta, tiling.strideC,
        tiling.workspaceBatchStride, tiling.workspaceSplitStride, tiling.alpha, tiling.beta, AscendC::GetBlockIdx(),
        blocks, earlyExit);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(VECTOR_THREADS) inline void StridedInt32EpilogueVf(
    __gm__ int32_t* partial, __gm__ int32_t* c, int32_t ldc, int32_t logicalM, int32_t logicalN, int32_t batchCount,
    int32_t splitK, int32_t hasBeta, int64_t strideC, int64_t workspaceBatchStride, int64_t workspaceSplitStride,
    int32_t alphaInt, int32_t betaInt, uint32_t block, uint32_t blocks, bool earlyExit)
{
    const StridedEpilogueParams tiling{
        {logicalM, logicalN, ldc, batchCount, strideC}, splitK, hasBeta, workspaceBatchStride, workspaceSplitStride};
    StridedEpilogueState state{};
    if (!InitStridedEpilogueState(tiling.logical, block, blocks, state)) {
        return;
    }
    for (; state.loop.index < state.loop.total; state.loop.index += state.loop.threadCount) {
        uint32_t result = 0;
        if (!earlyExit) {
            const int64_t partialOffset =
                static_cast<int64_t>(state.offset.batch) * tiling.workspaceBatchStride + state.offset.matrixIndex;
            for (int32_t split = 0; split < tiling.splitK; ++split) {
                result += static_cast<uint32_t>(
                    partial[partialOffset + static_cast<int64_t>(split) * tiling.workspaceSplitStride]);
            }
            result *= static_cast<uint32_t>(alphaInt);
        }
        if (tiling.hasBeta != 0) {
            result += static_cast<uint32_t>(betaInt) * static_cast<uint32_t>(c[state.offset.cOffset]);
        }
        c[state.offset.cOffset] = static_cast<int32_t>(result);
        AdvanceStridedLogicalOffset(tiling.logical, state.step, state.offset);
    }
}

} // namespace

extern "C" __global__ __aicore__ void gemm_strided_batched_ex_complex(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, GemmStridedBatchedExTilingData tiling, uint32_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<StridedComplexGemmVf>(
        dim3{VECTOR_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(b),
        reinterpret_cast<__gm__ float*>(c), tiling.k, tiling.lda, tiling.ldb, tiling.ldc, tiling.logicalM,
        tiling.logicalN, tiling.isTransA, tiling.isTransB, tiling.batchCount, tiling.strideA, tiling.strideB,
        tiling.strideC, tiling.alpha, tiling.beta, tiling.alphaImag, tiling.betaImag, AscendC::GetBlockIdx(), blocks);
}

void gemm_strided_batched_ex_complex_do(
    uint32_t numBlocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c,
    const GemmStridedBatchedExTilingData& tilingData)
{
    gemm_strided_batched_ex_complex<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData, numBlocks);
}

#define GEMM_STRIDED_EPILOGUE_KERNEL(NAME, C_TYPE, EARLY)                                                   \
    extern "C" __global__ __aicore__ void NAME(                                                             \
        __gm__ uint8_t* partial, __gm__ uint8_t* c, GemmStridedBatchedExTilingData tiling, uint32_t blocks) \
    {                                                                                                       \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                                     \
        LaunchStridedEpilogueVf<C_TYPE>(partial, c, tiling, blocks, EARLY);                                 \
    }

GEMM_STRIDED_EPILOGUE_KERNEL(gemm_strided_batched_ex_epilogue_f16, half, false)
GEMM_STRIDED_EPILOGUE_KERNEL(gemm_strided_batched_ex_epilogue_bf16, bfloat16_t, false)
GEMM_STRIDED_EPILOGUE_KERNEL(gemm_strided_batched_ex_epilogue_f32, float, false)
GEMM_STRIDED_EPILOGUE_KERNEL(gemm_strided_batched_ex_early_f16, half, true)
GEMM_STRIDED_EPILOGUE_KERNEL(gemm_strided_batched_ex_early_bf16, bfloat16_t, true)
GEMM_STRIDED_EPILOGUE_KERNEL(gemm_strided_batched_ex_early_f32, float, true)

extern "C" __global__ __aicore__ void gemm_strided_batched_ex_epilogue_i32(
    __gm__ uint8_t* partial, __gm__ uint8_t* c, GemmStridedBatchedExTilingData tiling, uint32_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<StridedInt32EpilogueVf>(
        dim3{VECTOR_THREADS, 1, 1}, reinterpret_cast<__gm__ int32_t*>(partial), reinterpret_cast<__gm__ int32_t*>(c),
        tiling.ldc, tiling.logicalM, tiling.logicalN, tiling.batchCount, tiling.splitK, tiling.hasBeta, tiling.strideC,
        tiling.workspaceBatchStride, tiling.workspaceSplitStride, tiling.alphaInt, tiling.betaInt,
        AscendC::GetBlockIdx(), blocks, false);
}

extern "C" __global__ __aicore__ void gemm_strided_batched_ex_early_i32(
    __gm__ uint8_t* partial, __gm__ uint8_t* c, GemmStridedBatchedExTilingData tiling, uint32_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<StridedInt32EpilogueVf>(
        dim3{VECTOR_THREADS, 1, 1}, reinterpret_cast<__gm__ int32_t*>(partial), reinterpret_cast<__gm__ int32_t*>(c),
        tiling.ldc, tiling.logicalM, tiling.logicalN, tiling.batchCount, tiling.splitK, tiling.hasBeta, tiling.strideC,
        tiling.workspaceBatchStride, tiling.workspaceSplitStride, tiling.alphaInt, tiling.betaInt,
        AscendC::GetBlockIdx(), blocks, true);
}

extern "C" __global__ __aicore__ void gemm_strided_batched_ex_epilogue_int8_f32(
    __gm__ uint8_t* partial, __gm__ uint8_t* c, GemmStridedBatchedExTilingData tiling, uint32_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<StridedFloatEpilogueVf<int32_t, float>>(
        dim3{VECTOR_THREADS, 1, 1}, reinterpret_cast<__gm__ int32_t*>(partial), reinterpret_cast<__gm__ float*>(c),
        tiling.ldc, tiling.logicalM, tiling.logicalN, tiling.batchCount, tiling.splitK, tiling.hasBeta, tiling.strideC,
        tiling.workspaceBatchStride, tiling.workspaceSplitStride, tiling.alpha, tiling.beta, AscendC::GetBlockIdx(),
        blocks, false);
}

extern "C" __global__ __aicore__ void gemm_strided_batched_ex_early_int8_f32(
    __gm__ uint8_t* partial, __gm__ uint8_t* c, GemmStridedBatchedExTilingData tiling, uint32_t blocks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<StridedFloatEpilogueVf<int32_t, float>>(
        dim3{VECTOR_THREADS, 1, 1}, reinterpret_cast<__gm__ int32_t*>(partial), reinterpret_cast<__gm__ float*>(c),
        tiling.ldc, tiling.logicalM, tiling.logicalN, tiling.batchCount, tiling.splitK, tiling.hasBeta, tiling.strideC,
        tiling.workspaceBatchStride, tiling.workspaceSplitStride, tiling.alpha, tiling.beta, AscendC::GetBlockIdx(),
        blocks, true);
}

static bool GemmStridedOutputIsFp32(GemmStridedBatchedExDtypeCase dtypeCase)
{
    return dtypeCase == GEMM_STRIDED_DTYPE_FP16_F32 || dtypeCase == GEMM_STRIDED_DTYPE_BF16_F32 ||
           dtypeCase == GEMM_STRIDED_DTYPE_FP32_F32 || dtypeCase == GEMM_STRIDED_DTYPE_FP8_E4M3_F32 ||
           dtypeCase == GEMM_STRIDED_DTYPE_FP8_E5M2_F32 || dtypeCase == GEMM_STRIDED_DTYPE_FP8_E4M3_E5M2_F32 ||
           dtypeCase == GEMM_STRIDED_DTYPE_FP8_E5M2_E4M3_F32 || dtypeCase == GEMM_STRIDED_DTYPE_INT8_F32;
}

void gemm_strided_batched_ex_epilogue_do(
    uint32_t numBlocks, void* stream, uint8_t* partial, uint8_t* c, const GemmStridedBatchedExTilingData& tilingData,
    GemmStridedBatchedExDtypeCase dtypeCase, bool earlyExit)
{
    if (dtypeCase == GEMM_STRIDED_DTYPE_INT8_I32) {
        if (earlyExit) {
            gemm_strided_batched_ex_early_i32<<<numBlocks, nullptr, stream>>>(partial, c, tilingData, numBlocks);
        } else {
            gemm_strided_batched_ex_epilogue_i32<<<numBlocks, nullptr, stream>>>(partial, c, tilingData, numBlocks);
        }
        return;
    }
    if (dtypeCase == GEMM_STRIDED_DTYPE_INT8_F32) {
        if (earlyExit) {
            gemm_strided_batched_ex_early_int8_f32<<<numBlocks, nullptr, stream>>>(partial, c, tilingData, numBlocks);
        } else {
            gemm_strided_batched_ex_epilogue_int8_f32<<<numBlocks, nullptr, stream>>>(
                partial, c, tilingData, numBlocks);
        }
        return;
    }
    if (GemmStridedOutputIsFp32(dtypeCase)) {
        if (earlyExit) {
            gemm_strided_batched_ex_early_f32<<<numBlocks, nullptr, stream>>>(partial, c, tilingData, numBlocks);
        } else {
            gemm_strided_batched_ex_epilogue_f32<<<numBlocks, nullptr, stream>>>(partial, c, tilingData, numBlocks);
        }
        return;
    }
    if (dtypeCase == GEMM_STRIDED_DTYPE_BF16_BF16) {
        if (earlyExit) {
            gemm_strided_batched_ex_early_bf16<<<numBlocks, nullptr, stream>>>(partial, c, tilingData, numBlocks);
        } else {
            gemm_strided_batched_ex_epilogue_bf16<<<numBlocks, nullptr, stream>>>(partial, c, tilingData, numBlocks);
        }
        return;
    }
    if (earlyExit) {
        gemm_strided_batched_ex_early_f16<<<numBlocks, nullptr, stream>>>(partial, c, tilingData, numBlocks);
    } else {
        gemm_strided_batched_ex_epilogue_f16<<<numBlocks, nullptr, stream>>>(partial, c, tilingData, numBlocks);
    }
}
