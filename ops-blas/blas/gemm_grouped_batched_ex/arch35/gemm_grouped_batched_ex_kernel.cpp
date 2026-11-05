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
 * \file gemm_grouped_batched_ex_kernel.cpp
 * \brief Grouped GEMM kernel for arch35 (DAV_3510).
 *
 * SIMD membase implementation using BlockMmad low-level API.
 * C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]
 *
 * Refactored: common helpers, POD state, standalone functions, unified macro.
 */

#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "gemm_grouped_batched_ex_tiling_data.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#undef KERNEL_UTILS_LITE

// DAV_3510 has 512 KiB L1 shared by the A1 and B1 tensors.
constexpr uint32_t GROUPED_L1_SIZE = 256 * 1024;
constexpr uint32_t GROUPED_L0C_SIZE = 128 * 1024;
constexpr uint32_t GROUPED_CUBE_BLOCK = 32;

__aicore__ inline void CopyGroupData(GroupedGemmGroupData* dst, const __gm__ GroupedGemmGroupData* src)
{
    static_assert(sizeof(GroupedGemmGroupData) % sizeof(uint64_t) == 0,
        "GroupedGemmGroupData must be copied in 64-bit chunks");
    uint64_t* ptr = reinterpret_cast<uint64_t*>(dst);
    auto src64 = reinterpret_cast<const __gm__ uint64_t*>(src);
    for (uint32_t i = 0; i < sizeof(GroupedGemmGroupData) / sizeof(uint64_t); i++) {
        ptr[i] = src64[i];
    }
}

// ============================================================================
// GroupedGemmCubeState — POD runtime state for batched cube kernel
// ============================================================================
struct GroupedGemmCubeState {
    GroupedGemmGroupData tiling;
    uint32_t mnTasks;
    uint32_t mBlockIdx;
    uint32_t nBlockIdx;
    uint32_t actualM;
    uint32_t actualN;
    uint32_t baseMCount;
    uint32_t tailM;
    uint32_t tailMAlign;
    uint32_t baseNCount;
    uint32_t tailN;
    uint32_t tailNAlign;
    uint32_t mLoopCount;
    uint32_t nLoopCount;
    uint32_t kLoopCount;
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

__aicore__ inline bool LoadGroupForCubeTask(
    GroupedGemmCubeState& st, __gm__ uint8_t* tilingGm, uint32_t taskId, uint32_t& localTask)
{
    const auto* header = reinterpret_cast<const __gm__ GroupedGemmTilingHeader*>(tilingGm);
    const auto* groups = reinterpret_cast<const __gm__ GroupedGemmGroupData*>(
        tilingGm + sizeof(GroupedGemmTilingHeader));
    uint32_t low = 0;
    uint32_t high = header->groupCount;
    while (low < high) {
        uint32_t mid = low + (high - low) / 2;
        if (groups[mid].cubeTaskStart <= taskId) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (low == 0) { return false; }
    uint32_t index = low - 1;
    uint32_t start = groups[index].cubeTaskStart;
    uint32_t count = groups[index].cubeTaskCount;
    if (taskId - start < count) {
        CopyGroupData(&st.tiling, &groups[index]);
        localTask = taskId - start;
        st.mnTasks = static_cast<uint32_t>(st.tiling.mBlocks) *
                     static_cast<uint32_t>(st.tiling.nBlocks);
        return st.mnTasks != 0;
    }
    return false;
}

__aicore__ inline void ComputeGroupedBaseOffsets(GroupedGemmCubeState& st)
{
    if (!st.tiling.isTransA) {
        st.aBaseOffset = static_cast<uint64_t>(st.mBlockIdx) * st.tiling.singleCoreM * st.tiling.lda;
    } else {
        st.aBaseOffset = static_cast<uint64_t>(st.mBlockIdx) * st.tiling.singleCoreM;
    }
    if (!st.tiling.isTransB) {
        st.bBaseOffset = static_cast<uint64_t>(st.nBlockIdx) * st.tiling.singleCoreN;
    } else {
        st.bBaseOffset = static_cast<uint64_t>(st.nBlockIdx) * st.tiling.singleCoreN * st.tiling.ldb;
    }
    st.cBaseOffset = static_cast<uint64_t>(st.mBlockIdx) * st.tiling.ldc * st.tiling.singleCoreM +
                     static_cast<uint64_t>(st.nBlockIdx) * st.tiling.singleCoreN;
}

template <uint32_t BASE_M, uint32_t BASE_N>
__aicore__ inline void ComputeGrouped2DBlocking(GroupedGemmCubeState& st)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    constexpr uint32_t MAX_C0_TILES = GROUPED_L0C_SIZE / C0_TILE_BYTES;
    st.mBlockSize = st.mLoopCount;
    st.nBlockSize = st.nLoopCount;
    while (st.mBlockSize * st.nBlockSize > MAX_C0_TILES) {
        if (st.mBlockSize >= st.nBlockSize) {
            st.mBlockSize = (st.mBlockSize + 1) / 2;
        } else {
            st.nBlockSize = (st.nBlockSize + 1) / 2;
        }
    }
    // R8: mBlockSize/nBlockSize >= 1 guaranteed (mLoopCount/nLoopCount >= 1 when actualM/N > 0)
    if (st.mBlockSize == 0) { st.mBlockSize = 1; }
    if (st.nBlockSize == 0) { st.nBlockSize = 1; }
    st.mBlockCount = CeilDiv(st.mLoopCount, st.mBlockSize);
    st.nBlockCount = CeilDiv(st.nLoopCount, st.nBlockSize);
}

template <uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline void InitGroupedTaskState(GroupedGemmCubeState& st, uint32_t taskId)
{
    constexpr uint32_t CUBE_BLOCK = GROUPED_CUBE_BLOCK;
    // R8: mnTasks > 0 guaranteed by InitGroupedTilingBase
    uint32_t mnTask = taskId % st.mnTasks;
    st.mBlockIdx = mnTask % static_cast<uint32_t>(st.tiling.mBlocks);
    st.nBlockIdx = mnTask / static_cast<uint32_t>(st.tiling.mBlocks);
    st.actualM = static_cast<uint32_t>(st.tiling.m) - st.mBlockIdx * static_cast<uint32_t>(st.tiling.singleCoreM);
    if (st.actualM > static_cast<uint32_t>(st.tiling.singleCoreM)) {
        st.actualM = static_cast<uint32_t>(st.tiling.singleCoreM);
    }
    st.actualN = static_cast<uint32_t>(st.tiling.n) - st.nBlockIdx * static_cast<uint32_t>(st.tiling.singleCoreN);
    if (st.actualN > static_cast<uint32_t>(st.tiling.singleCoreN)) {
        st.actualN = static_cast<uint32_t>(st.tiling.singleCoreN);
    }
    st.baseMCount = st.actualM / BASE_M;
    st.tailM = st.actualM % BASE_M;
    st.tailMAlign = RoundUp(st.tailM, CUBE_BLOCK);
    st.baseNCount = st.actualN / BASE_N;
    st.tailN = st.actualN % BASE_N;
    st.tailNAlign = RoundUp(st.tailN, CUBE_BLOCK);
    st.mLoopCount = CeilDiv(st.actualM, BASE_M);
    st.nLoopCount = CeilDiv(st.actualN, BASE_N);
    st.kLoopCount = CeilDiv(static_cast<uint32_t>(st.tiling.k), BASE_K);
    ComputeGroupedBaseOffsets(st);
    ComputeGrouped2DBlocking<BASE_M, BASE_N>(st);
}

template <typename A_TYPE, uint32_t BASE_M, uint32_t BASE_K, uint32_t C0_VAL>
__aicore__ inline void LoadAGroupedTile(
    const GroupedGemmCubeState& st,
    AscendC::GlobalTensor<A_TYPE>& aGM,
    AscendC::LocalTensor<A_TYPE>& a1,
    AscendC::LocalTensor<A_TYPE>& a2,
    uint32_t mi, uint32_t kIdx, uint32_t curM, uint32_t curK, uint32_t curMAlign)
{
    constexpr uint32_t CUBE_BLOCK = GROUPED_CUBE_BLOCK;
    uint64_t aOffset;
    AscendC::Nd2NzParams ndA;
    ndA.ndNum = 1;
    ndA.dstNzNStride = 1;
    ndA.dstNzMatrixStride = 0;
    if (!st.tiling.isTransA) {
        aOffset = st.aBaseOffset + static_cast<uint64_t>(mi) * BASE_M * st.tiling.lda +
                  static_cast<uint64_t>(kIdx) * BASE_K;
        ndA.nValue = curM;
        ndA.dValue = curK;
        ndA.srcNdMatrixStride = 0;
        ndA.srcDValue = static_cast<uint32_t>(st.tiling.lda);
        ndA.dstNzC0Stride = RoundUp(curM, CUBE_BLOCK);
    } else {
        aOffset = st.aBaseOffset + static_cast<uint64_t>(kIdx) * BASE_K * st.tiling.lda +
                  static_cast<uint64_t>(mi) * BASE_M;
        ndA.nValue = curK;
        ndA.dValue = curM;
        ndA.srcNdMatrixStride = 0;
        ndA.srcDValue = static_cast<uint32_t>(st.tiling.lda);
        ndA.dstNzC0Stride = RoundUp(curK, CUBE_BLOCK);
    }
    AscendC::DataCopy(a1, aGM[aOffset], ndA);
    PIPE_BARRIER(ALL);
    AscendC::LoadData2DParamsV2 ldA;
    ldA.mStartPosition = 0;
    ldA.kStartPosition = 0;
    ldA.sid = 0;
    if (!st.tiling.isTransA) {
        ldA.mStep = CeilDiv(curMAlign, CUBE_BLOCK);
        ldA.kStep = CeilDiv(BASE_K, C0_VAL);
        ldA.srcStride = CeilDiv(curMAlign, CUBE_BLOCK);
        ldA.dstStride = CeilDiv(curMAlign, CUBE_BLOCK);
        ldA.ifTranspose = false;
    } else {
        ldA.mStep = CeilDiv(BASE_K, CUBE_BLOCK);
        ldA.kStep = RoundUp(CeilDiv(curMAlign, C0_VAL), 2u);
        ldA.srcStride = CeilDiv(BASE_K, CUBE_BLOCK);
        ldA.dstStride = CeilDiv(curMAlign, CUBE_BLOCK);
        ldA.ifTranspose = true;
    }
    AscendC::LoadData(a2, a1, ldA);
    PIPE_BARRIER(ALL);
}

template <typename B_TYPE, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline void LoadBGroupedTile(
    const GroupedGemmCubeState& st,
    AscendC::GlobalTensor<B_TYPE>& bGM,
    AscendC::LocalTensor<B_TYPE>& b1,
    AscendC::LocalTensor<B_TYPE>& b2,
    uint32_t ni, uint32_t kIdx, uint32_t curK, uint32_t curN, uint32_t curNAlign)
{
    constexpr uint32_t CUBE_BLOCK = GROUPED_CUBE_BLOCK;
    uint64_t bOffset;
    AscendC::Nd2NzParams ndB;
    ndB.ndNum = 1;
    ndB.dstNzNStride = 1;
    ndB.dstNzMatrixStride = 0;
    if (!st.tiling.isTransB) {
        bOffset = st.bBaseOffset + static_cast<uint64_t>(kIdx) * BASE_K * st.tiling.ldb +
                  static_cast<uint64_t>(ni) * BASE_N;
        ndB.nValue = curK;
        ndB.dValue = curN;
        ndB.srcNdMatrixStride = 0;
        ndB.srcDValue = static_cast<uint32_t>(st.tiling.ldb);
        ndB.dstNzC0Stride = RoundUp(curK, CUBE_BLOCK);
    } else {
        bOffset = st.bBaseOffset + static_cast<uint64_t>(ni) * BASE_N * st.tiling.ldb +
                  static_cast<uint64_t>(kIdx) * BASE_K;
        ndB.nValue = curN;
        ndB.dValue = curK;
        ndB.srcNdMatrixStride = 0;
        ndB.srcDValue = static_cast<uint32_t>(st.tiling.ldb);
        ndB.dstNzC0Stride = RoundUp(curN, CUBE_BLOCK);
    }
    AscendC::DataCopy(b1, bGM[bOffset], ndB);
    PIPE_BARRIER(ALL);
    AscendC::LoadData2DParamsV2 ldB;
    ldB.mStartPosition = 0;
    ldB.kStartPosition = 0;
    ldB.sid = 0;
    if (!st.tiling.isTransB) {
        ldB.mStep = CeilDiv(BASE_K, CUBE_BLOCK);
        ldB.kStep = CeilDiv(static_cast<uint32_t>(curNAlign * sizeof(B_TYPE)), 32u);
        ldB.srcStride = CeilDiv(BASE_K, CUBE_BLOCK);
        ldB.dstStride = CeilDiv(curNAlign, CUBE_BLOCK);
        ldB.ifTranspose = true;
    } else {
        ldB.mStep = CeilDiv(curNAlign, CUBE_BLOCK);
        ldB.kStep = CeilDiv(BASE_K, C0_VAL);
        ldB.srcStride = CeilDiv(curNAlign, CUBE_BLOCK);
        ldB.dstStride = CeilDiv(curNAlign, CUBE_BLOCK);
        ldB.ifTranspose = false;
    }
    AscendC::LoadData(b2, b1, ldB);
    PIPE_BARRIER(ALL);
}

template <typename A_TYPE, typename B_TYPE, uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline void ProcessGroupedNTile(
    const GroupedGemmCubeState& st,
    AscendC::GlobalTensor<B_TYPE>& bGM,
    AscendC::LocalTensor<A_TYPE>& a2,
    AscendC::LocalTensor<B_TYPE>& b1,
    AscendC::LocalTensor<B_TYPE>& b2,
    uint32_t mi, uint32_t ni, uint32_t kIdx,
    uint32_t mStart, uint32_t nStart,
    uint32_t curM, uint32_t curK)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    if (ni != nStart) {
        WAIT_FLAG(M, FIX, EVENT_ID0);
    }
    uint32_t curN = (ni != st.baseNCount) ? BASE_N : st.tailN;
    uint32_t curNAlign = (ni != st.baseNCount) ? BASE_N : st.tailNAlign;
    LoadBGroupedTile<B_TYPE, BASE_K, BASE_N, C0_VAL>(st, bGM, b1, b2, ni, kIdx, curK, curN, curNAlign);
    uint32_t cTileIdx = (mi - mStart) * st.nBlockSize + (ni - nStart);
    AscendC::LocalTensor<float> c0(AscendC::TPosition::CO1, cTileIdx * C0_TILE_BYTES, BASE_M * BASE_N);
    AscendC::MmadParams mp{};
    mp.m = curM;
    mp.n = curN;
    mp.k = curK;
    mp.cmatrixInitVal = (kIdx == 0);
    AscendC::Mmad(c0, a2, b2, mp);
    SET_FLAG(M, FIX, EVENT_ID0);
}

template <typename C_GM_TYPE, uint32_t BASE_M, uint32_t BASE_N, QuantMode_t QUANT_MODE>
__aicore__ inline void WriteGroupedFixpipe(
    const GroupedGemmCubeState& st,
    AscendC::GlobalTensor<C_GM_TYPE>& cGM,
    uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    for (uint32_t mi = mStart; mi < mEnd; mi++) {
        uint32_t curM = (mi != st.baseMCount) ? BASE_M : st.tailM;
        uint32_t curMAlign = (mi != st.baseMCount) ? BASE_M : st.tailMAlign;
        for (uint32_t ni = nStart; ni < nEnd; ni++) {
            uint32_t curN = (ni != st.baseNCount) ? BASE_N : st.tailN;
            uint32_t cTileIdx = (mi - mStart) * st.nBlockSize + (ni - nStart);
            AscendC::LocalTensor<float> c0(AscendC::TPosition::CO1,
                cTileIdx * C0_TILE_BYTES, BASE_M * BASE_N);
            uint64_t cOffset = st.cBaseOffset +
                static_cast<uint64_t>(mi) * BASE_M * st.tiling.ldc +
                static_cast<uint64_t>(ni) * BASE_N;
            AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp;
            fp.mSize = curM;
            fp.nSize = curN;
            fp.srcStride = curMAlign;
            fp.dstStride = static_cast<uint32_t>(st.tiling.ldc);
            fp.quantPre = QUANT_MODE;
            AscendC::Fixpipe(cGM[cOffset], c0, fp);
            SET_FLAG(FIX, MTE2, EVENT_ID0);
            WAIT_FLAG(FIX, MTE2, EVENT_ID0);
        }
    }
}

// ============================================================================
// RunKSlice — Process one K slice across M×N tiles (depth 2: mi→ni)
// ============================================================================
template <typename A_TYPE, typename B_TYPE,
          uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline void RunKSlice(
    GroupedGemmCubeState& st,
    AscendC::GlobalTensor<A_TYPE>& aGM,
    AscendC::GlobalTensor<B_TYPE>& bGM,
    AscendC::LocalTensor<A_TYPE>& a1,
    AscendC::LocalTensor<B_TYPE>& b1,
    AscendC::LocalTensor<A_TYPE>& a2,
    AscendC::LocalTensor<B_TYPE>& b2,
    uint32_t mStart, uint32_t mEnd,
    uint32_t nStart, uint32_t nEnd,
    uint32_t kIdx, uint32_t curK,
    bool& firstMmad)
{
    for (uint32_t mi = mStart; mi < mEnd; mi++) {
        if (!firstMmad) { WAIT_FLAG(M, FIX, EVENT_ID0); }
        firstMmad = false;
        uint32_t curM = (mi != st.baseMCount) ? BASE_M : st.tailM;
        uint32_t curMAlign = (mi != st.baseMCount) ? BASE_M : st.tailMAlign;
        LoadAGroupedTile<A_TYPE, BASE_M, BASE_K, C0_VAL>(st, aGM, a1, a2, mi, kIdx, curM, curK, curMAlign);
        for (uint32_t ni = nStart; ni < nEnd; ni++) {
            ProcessGroupedNTile<A_TYPE, B_TYPE, BASE_M, BASE_K, BASE_N, C0_VAL>(
                st, bGM, a2, b1, b2, mi, ni, kIdx, mStart, nStart, curM, curK);
        }
    }
}

// ============================================================================
// RunGroupedTask — Inner computation for one task (depth 3: mb→nb→kIdx)
// ============================================================================
template <typename A_TYPE, typename B_TYPE, typename C_GM_TYPE,
          uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL, QuantMode_t QMODE>
__aicore__ inline void RunGroupedTask(
    GroupedGemmCubeState& st,
    AscendC::GlobalTensor<A_TYPE>& aGM,
    AscendC::GlobalTensor<B_TYPE>& bGM,
    AscendC::GlobalTensor<C_GM_TYPE>& cGM,
    AscendC::LocalTensor<A_TYPE>& a1,
    AscendC::LocalTensor<B_TYPE>& b1,
    AscendC::LocalTensor<A_TYPE>& a2,
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
                uint32_t curK = (kIdx == st.kLoopCount - 1)
                    ? (static_cast<uint32_t>(st.tiling.k) - kIdx * BASE_K) : BASE_K;
                RunKSlice<A_TYPE, B_TYPE, BASE_M, BASE_K, BASE_N, C0_VAL>(
                    st, aGM, bGM, a1, b1, a2, b2,
                    mStart, mEnd, nStart, nEnd, kIdx, curK, firstMmad);
            }
            WAIT_FLAG(M, FIX, EVENT_ID0);
            WriteGroupedFixpipe<C_GM_TYPE, BASE_M, BASE_N, QMODE>(st, cGM, mStart, mEnd, nStart, nEnd);
            firstMmad = true;
        }
    }
}

// ============================================================================
// GEMM_GROUPED_CUBE_KERNEL — Macro generating all batched cube kernel variants
//
// Supports separate A_TYPE and B_TYPE for FP8 mixed-type kernels.
// Task loop iterates over batch × mBlocks × nTasks with grid-stride.
//
// Parameters:
//   FUNC_NAME  - kernel function name
//   A_TYPE     - input A element type
//   B_TYPE     - input B element type
//   C_GM_TYPE  - output GM element type
//   BM, BK, BN - base tile dimensions
//   C0_VAL     - C0 parameter for LoadData
//   QUANT_MODE - Fixpipe quantization mode
// ============================================================================
#define GEMM_GROUPED_CUBE_KERNEL(FUNC_NAME, A_TYPE, B_TYPE, BM, BK, BN, C0_VAL) \
__cube__ __global__ void FUNC_NAME(__gm__ uint8_t* aarray, __gm__ uint8_t* barray, \
    __gm__ uint8_t* workspace, __gm__ uint8_t* tilingGm) \
{ \
    AscendC::InitSocState(); \
    GroupedGemmCubeState st{}; \
    const auto* header = reinterpret_cast<const __gm__ GroupedGemmTilingHeader*>(tilingGm); \
    constexpr uint32_t L1_SIZE = GROUPED_L1_SIZE; \
    AscendC::LocalTensor<A_TYPE> a1(AscendC::TPosition::A1, 0, L1_SIZE); \
    AscendC::LocalTensor<B_TYPE> b1(AscendC::TPosition::B1, L1_SIZE, L1_SIZE); \
    AscendC::LocalTensor<A_TYPE> a2(AscendC::TPosition::A2, 0, (BM) * (BK)); \
    AscendC::LocalTensor<B_TYPE> b2(AscendC::TPosition::B2, 0, (BK) * (BN)); \
    uint32_t blockIdxVal = AscendC::GetBlockIdx(); \
    uint32_t gridDim = AscendC::GetBlockNum(); \
    for (uint32_t taskId = blockIdxVal; taskId < header->totalCubeTasks; taskId += gridDim) { \
        uint32_t localTask = 0; \
        if (!LoadGroupForCubeTask(st, tilingGm, taskId, localTask)) { continue; } \
        InitGroupedTaskState<(BM), (BK), (BN), (C0_VAL)>(st, localTask); \
        uint32_t localBatchIdx = localTask / st.mnTasks; \
        uint32_t problemIdx = st.tiling.batchStart + localBatchIdx; \
        __gm__ uint64_t* aPtrArray = reinterpret_cast<__gm__ uint64_t*>(aarray); \
        __gm__ uint64_t* bPtrArray = reinterpret_cast<__gm__ uint64_t*>(barray); \
        AscendC::GlobalTensor<A_TYPE> aGM; \
        AscendC::GlobalTensor<B_TYPE> bGM; \
        AscendC::GlobalTensor<float> cGM; \
        aGM.SetGlobalBuffer(reinterpret_cast<__gm__ A_TYPE*>(aPtrArray[problemIdx])); \
        bGM.SetGlobalBuffer(reinterpret_cast<__gm__ B_TYPE*>(bPtrArray[problemIdx])); \
        uint64_t matrixOffset = st.tiling.workspaceOffset + \
            static_cast<uint64_t>(localBatchIdx) * st.tiling.originalM * st.tiling.originalN; \
        cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + matrixOffset); \
        RunGroupedTask<A_TYPE, B_TYPE, float, (BM), (BK), (BN), (C0_VAL), QuantMode_t::NoQuant>( \
            st, aGM, bGM, cGM, a1, b1, a2, b2); \
    } \
    PIPE_BARRIER(ALL); \
}

union GroupedFp8FloatBits {
    uint32_t bits;
    float value;
};

template <bool IS_E5M2>
__aicore__ inline float DecodeGroupedFp8(uint8_t raw)
{
    uint32_t sign = static_cast<uint32_t>(raw & 0x80u) << 24;
    uint32_t exponent;
    uint32_t mantissa;
    if constexpr (IS_E5M2) {
        exponent = (raw >> 2) & 0x1fu;
        mantissa = raw & 0x3u;
        if (exponent == 0) {
            float value = static_cast<float>(mantissa) * 0.0000152587890625f; // 2^-16
            return (raw & 0x80u) != 0 ? -value : value;
        }
        if (exponent == 0x1fu) {
            GroupedFp8FloatBits special{};
            special.bits = sign | 0x7f800000u | (mantissa << 21);
            return special.value;
        }
        GroupedFp8FloatBits normal{};
        normal.bits = sign | ((exponent + 112u) << 23) | (mantissa << 21);
        return normal.value;
    } else {
        exponent = (raw >> 3) & 0x0fu;
        mantissa = raw & 0x7u;
        if (exponent == 0) {
            float value = static_cast<float>(mantissa) * 0.001953125f; // 2^-9
            return (raw & 0x80u) != 0 ? -value : value;
        }
        if (exponent == 0x0fu && mantissa == 0x7u) {
            GroupedFp8FloatBits nanValue{};
            nanValue.bits = sign | 0x7fc00000u;
            return nanValue.value;
        }
        GroupedFp8FloatBits normal{};
        normal.bits = sign | ((exponent + 120u) << 23) | (mantissa << 20);
        return normal.value;
    }
}

template <bool A_IS_E5M2, bool B_IS_E5M2>
__aicore__ inline void RunGroupedFp8Kernel(__gm__ uint8_t* aarray, __gm__ uint8_t* barray,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tilingGm)
{
    GroupedGemmCubeState st{};
    const auto* header = reinterpret_cast<const __gm__ GroupedGemmTilingHeader*>(tilingGm);
    uint32_t blockIdxVal = AscendC::GetBlockIdx();
    uint32_t gridDim = AscendC::GetBlockNum();
    for (uint32_t taskId = blockIdxVal; taskId < header->totalCubeTasks; taskId += gridDim) {
        uint32_t localTask = 0;
        if (!LoadGroupForCubeTask(st, tilingGm, taskId, localTask)) { continue; }
        uint32_t mnTask = localTask % st.mnTasks;
        uint32_t localBatchIdx = localTask / st.mnTasks;
        uint32_t mBlockIdx = mnTask % static_cast<uint32_t>(st.tiling.mBlocks);
        uint32_t nBlockIdx = mnTask / static_cast<uint32_t>(st.tiling.mBlocks);
        uint32_t rowStart = mBlockIdx * static_cast<uint32_t>(st.tiling.singleCoreM);
        uint32_t columnStart = nBlockIdx * static_cast<uint32_t>(st.tiling.singleCoreN);
        uint32_t rowEnd = rowStart + static_cast<uint32_t>(st.tiling.singleCoreM);
        uint32_t columnEnd = columnStart + static_cast<uint32_t>(st.tiling.singleCoreN);
        if (rowEnd > static_cast<uint32_t>(st.tiling.m)) { rowEnd = static_cast<uint32_t>(st.tiling.m); }
        if (columnEnd > static_cast<uint32_t>(st.tiling.n)) { columnEnd = static_cast<uint32_t>(st.tiling.n); }

        uint32_t problemIdx = st.tiling.batchStart + localBatchIdx;
        __gm__ uint64_t* aPtrArray = reinterpret_cast<__gm__ uint64_t*>(aarray);
        __gm__ uint64_t* bPtrArray = reinterpret_cast<__gm__ uint64_t*>(barray);
        __gm__ uint8_t* aGM = reinterpret_cast<__gm__ uint8_t*>(aPtrArray[problemIdx]);
        __gm__ uint8_t* bGM = reinterpret_cast<__gm__ uint8_t*>(bPtrArray[problemIdx]);
        __gm__ float* output = reinterpret_cast<__gm__ float*>(workspace) + st.tiling.workspaceOffset +
            static_cast<uint64_t>(localBatchIdx) * st.tiling.originalM * st.tiling.originalN;

        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            for (uint32_t column = columnStart; column < columnEnd; ++column) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < static_cast<uint32_t>(st.tiling.k); ++k) {
                    uint64_t aOffset = st.tiling.isTransA == 0 ?
                        static_cast<uint64_t>(row) * st.tiling.lda + k :
                        static_cast<uint64_t>(k) * st.tiling.lda + row;
                    uint64_t bOffset = st.tiling.isTransB == 0 ?
                        static_cast<uint64_t>(k) * st.tiling.ldb + column :
                        static_cast<uint64_t>(column) * st.tiling.ldb + k;
                    sum += DecodeGroupedFp8<A_IS_E5M2>(aGM[aOffset]) *
                           DecodeGroupedFp8<B_IS_E5M2>(bGM[bOffset]);
                }
                output[static_cast<uint64_t>(row) * st.tiling.ldc + column] = sum;
            }
        }
    }
}

#define GEMM_GROUPED_FP8_KERNEL(FUNC_NAME, A_IS_E5M2, B_IS_E5M2) \
__cube__ __global__ void FUNC_NAME(__gm__ uint8_t* aarray, __gm__ uint8_t* barray, \
    __gm__ uint8_t* workspace, __gm__ uint8_t* tilingGm) \
{ \
    AscendC::InitSocState(); \
    RunGroupedFp8Kernel<(A_IS_E5M2), (B_IS_E5M2)>(aarray, barray, workspace, tilingGm); \
    PIPE_BARRIER(ALL); \
}

// ── Kernel instantiations ──
// FP16: BASE_M=128, BASE_K=16, BASE_N=128, C0=16, F322F16
GEMM_GROUPED_CUBE_KERNEL(gemm_grouped_batched_ex_kernel_fp16, half, half,
    128, 16, 128, 16)

// BF16: same tiles as FP16, F322BF16
GEMM_GROUPED_CUBE_KERNEL(gemm_grouped_batched_ex_kernel_bf16, bfloat16_t, bfloat16_t,
    128, 16, 128, 16)

// FP8 is decoded and accumulated on the NPU scalar pipeline. This avoids the
// unsupported raw-FP8 LoadData/Mmad layout while keeping all GEMM work on device.
GEMM_GROUPED_FP8_KERNEL(gemm_grouped_batched_ex_kernel_fp8_e4m3, false, false)
GEMM_GROUPED_FP8_KERNEL(gemm_grouped_batched_ex_kernel_fp8_e5m2, true, true)
GEMM_GROUPED_FP8_KERNEL(gemm_grouped_batched_ex_kernel_fp8_e5m2_e4m3, true, false)
GEMM_GROUPED_FP8_KERNEL(gemm_grouped_batched_ex_kernel_fp8_e4m3_e5m2, false, true)

// ── Kernel launcher ──
void gemm_grouped_batched_ex_cube_kernel_do(uint32_t numBlocks, void* stream,
    uint8_t* aarray, uint8_t* barray, uint8_t* workspace, uint8_t* tilingGm, int dtypeCase)
{
    switch (dtypeCase) {
        case GROUPED_GEMM_FP16:
            gemm_grouped_batched_ex_kernel_fp16<<<numBlocks, nullptr, stream>>>(
                aarray, barray, workspace, tilingGm);
            break;
        case GROUPED_GEMM_BF16:
            gemm_grouped_batched_ex_kernel_bf16<<<numBlocks, nullptr, stream>>>(
                aarray, barray, workspace, tilingGm);
            break;
        case GROUPED_GEMM_FP8_E4M3_E4M3:
            gemm_grouped_batched_ex_kernel_fp8_e4m3<<<numBlocks, nullptr, stream>>>(
                aarray, barray, workspace, tilingGm);
            break;
        case GROUPED_GEMM_FP8_E5M2_E5M2:
            gemm_grouped_batched_ex_kernel_fp8_e5m2<<<numBlocks, nullptr, stream>>>(
                aarray, barray, workspace, tilingGm);
            break;
        case GROUPED_GEMM_FP8_E4M3_E5M2:
            gemm_grouped_batched_ex_kernel_fp8_e5m2_e4m3<<<numBlocks, nullptr, stream>>>(
                aarray, barray, workspace, tilingGm);
            break;
        case GROUPED_GEMM_FP8_E5M2_E4M3:
            gemm_grouped_batched_ex_kernel_fp8_e4m3_e5m2<<<numBlocks, nullptr, stream>>>(
                aarray, barray, workspace, tilingGm);
            break;
        default:
            break;
    }
}
