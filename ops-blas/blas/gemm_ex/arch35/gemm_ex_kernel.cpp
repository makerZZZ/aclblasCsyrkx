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
 * \file gemm_ex_kernel.cpp
 * \brief General matrix multiply (GEMM) kernel for arch35 (DAV_3510).
 *
 * SIMD membase implementation using BlockMmad low-level API.
 * All dtype variants use K-outer + 2D blocking for optimal Cube utilization.
 * A single macro (GEMM_CUBE_KERNEL) generates all cube kernel variants to eliminate
 * code duplication while preserving standalone-function pattern (required on arch35).
 *
 * C = alpha * op(A) * op(B) + beta * C
 * (alpha/beta handled by separate Vector kernel post-processing)
 */

#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "gemm_ex_tiling_data.h"
#include "common/arch/hardware.h"
#include "common/helper/kernel_utils.h"

// ============================================================================
// GemmCubeState — Runtime state for cube kernel (POD struct, not a class)
// ============================================================================
struct GemmCubeState {
    GemmExTilingData tiling;
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
// Standalone __aicore__ helper functions for GEMM_CUBE_KERNEL
// (arch35: class template methods cause Mmad hang; standalone functions required)
// ============================================================================

__aicore__ inline void ComputeBaseOffsets(GemmCubeState& st)
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
__aicore__ inline void Compute2DBlocking(GemmCubeState& st)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    constexpr uint32_t MAX_C0_TILES = HardwareInfo<ArchType::ASCEND_V350>::l0CSize / C0_TILE_BYTES;
    st.mBlockSize = st.mLoopCount;
    st.nBlockSize = st.nLoopCount;
    while (st.mBlockSize * st.nBlockSize > MAX_C0_TILES) {
        if (st.mBlockSize >= st.nBlockSize) {
            st.mBlockSize = (st.mBlockSize + 1) / 2;
        } else {
            st.nBlockSize = (st.nBlockSize + 1) / 2;
        }
    }
    st.mBlockCount = CeilDiv(st.mLoopCount, st.mBlockSize);
    st.nBlockCount = CeilDiv(st.nLoopCount, st.nBlockSize);
}

template <uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline bool InitGemmState(GemmCubeState& st, GemmExTilingData tiling)
{
    constexpr uint32_t CUBE_BLOCK = HardwareInfo<ArchType::ASCEND_V350>::l1l0BlockSize;
    st.tiling = tiling;
    if (AscendC::GetBlockIdx() >= static_cast<uint32_t>(st.tiling.usedCoreNum)) {
        return false;
    }
    if (st.tiling.mBlocks == 0 || st.tiling.nBlocks == 0) {
        return false;
    }
    st.mBlockIdx = AscendC::GetBlockIdx() % static_cast<uint32_t>(st.tiling.mBlocks);
    st.nBlockIdx = AscendC::GetBlockIdx() / static_cast<uint32_t>(st.tiling.mBlocks);
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
    ComputeBaseOffsets(st);
    Compute2DBlocking<BASE_M, BASE_N>(st);
    return true;
}

template <typename A_TYPE, uint32_t BASE_M, uint32_t BASE_K, uint32_t C0_VAL>
__aicore__ inline void LoadATile(
    const GemmCubeState& st, AscendC::GlobalTensor<A_TYPE>& aGM, AscendC::LocalTensor<A_TYPE>& a1,
    AscendC::LocalTensor<A_TYPE>& a2, uint32_t mi, uint32_t kIdx, uint32_t curM, uint32_t curK, uint32_t curMAlign)
{
    constexpr uint32_t CUBE_BLOCK = HardwareInfo<ArchType::ASCEND_V350>::l1l0BlockSize;
    uint64_t aOffset;
    AscendC::Nd2NzParams ndA;
    ndA.ndNum = 1;
    ndA.dstNzNStride = 1;
    ndA.dstNzMatrixStride = 0;
    if (!st.tiling.isTransA) {
        aOffset =
            st.aBaseOffset + static_cast<uint64_t>(mi) * BASE_M * st.tiling.lda + static_cast<uint64_t>(kIdx) * BASE_K;
        ndA.nValue = curM;
        ndA.dValue = curK;
        ndA.srcNdMatrixStride = 0;
        ndA.srcDValue = static_cast<uint32_t>(st.tiling.lda);
        ndA.dstNzC0Stride = RoundUp(curM, CUBE_BLOCK);
    } else {
        aOffset =
            st.aBaseOffset + static_cast<uint64_t>(kIdx) * BASE_K * st.tiling.lda + static_cast<uint64_t>(mi) * BASE_M;
        ndA.nValue = curK;
        ndA.dValue = curM;
        ndA.srcNdMatrixStride = 0;
        ndA.srcDValue = static_cast<uint32_t>(st.tiling.lda);
        ndA.dstNzC0Stride = RoundUp(curK, CUBE_BLOCK);
    }
    AscendC::DataCopy(a1, aGM[aOffset], ndA);
    AscendC::PipeBarrier<PIPE_ALL>();
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
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename B_TYPE, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline void LoadBTile(
    const GemmCubeState& st, AscendC::GlobalTensor<B_TYPE>& bGM, AscendC::LocalTensor<B_TYPE>& b1,
    AscendC::LocalTensor<B_TYPE>& b2, uint32_t ni, uint32_t kIdx, uint32_t curK, uint32_t curN, uint32_t curNAlign)
{
    constexpr uint32_t CUBE_BLOCK = HardwareInfo<ArchType::ASCEND_V350>::l1l0BlockSize;
    uint64_t bOffset;
    AscendC::Nd2NzParams ndB;
    ndB.ndNum = 1;
    ndB.dstNzNStride = 1;
    ndB.dstNzMatrixStride = 0;
    if (!st.tiling.isTransB) {
        bOffset =
            st.bBaseOffset + static_cast<uint64_t>(kIdx) * BASE_K * st.tiling.ldb + static_cast<uint64_t>(ni) * BASE_N;
        ndB.nValue = curK;
        ndB.dValue = curN;
        ndB.srcNdMatrixStride = 0;
        ndB.srcDValue = static_cast<uint32_t>(st.tiling.ldb);
        ndB.dstNzC0Stride = RoundUp(curK, CUBE_BLOCK);
    } else {
        bOffset =
            st.bBaseOffset + static_cast<uint64_t>(ni) * BASE_N * st.tiling.ldb + static_cast<uint64_t>(kIdx) * BASE_K;
        ndB.nValue = curN;
        ndB.dValue = curK;
        ndB.srcNdMatrixStride = 0;
        ndB.srcDValue = static_cast<uint32_t>(st.tiling.ldb);
        ndB.dstNzC0Stride = RoundUp(curN, CUBE_BLOCK);
    }
    AscendC::DataCopy(b1, bGM[bOffset], ndB);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::LoadData2DParamsV2 ldB;
    ldB.mStartPosition = 0;
    ldB.kStartPosition = 0;
    ldB.sid = 0;
    if (!st.tiling.isTransB) {
        ldB.mStep = CeilDiv(BASE_K, CUBE_BLOCK);
        constexpr uint32_t VECTOR_ALIGN_BYTES = 32;
        ldB.kStep = CeilDiv(static_cast<uint32_t>(curNAlign * sizeof(B_TYPE)), VECTOR_ALIGN_BYTES);
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
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <typename A_TYPE, typename B_TYPE, uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline void ProcessNTile(
    const GemmCubeState& st, AscendC::GlobalTensor<B_TYPE>& bGM, AscendC::LocalTensor<A_TYPE>& a2,
    AscendC::LocalTensor<B_TYPE>& b1, AscendC::LocalTensor<B_TYPE>& b2, uint32_t mi, uint32_t ni, uint32_t kIdx,
    uint32_t mStart, uint32_t nStart, uint32_t curM, uint32_t curK)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    uint32_t curN = (ni != st.baseNCount) ? BASE_N : st.tailN;
    uint32_t curNAlign = (ni != st.baseNCount) ? BASE_N : st.tailNAlign;
    LoadBTile<B_TYPE, BASE_K, BASE_N, C0_VAL>(st, bGM, b1, b2, ni, kIdx, curK, curN, curNAlign);
    uint32_t cTileIdx = (mi - mStart) * st.nBlockSize + (ni - nStart);
    AscendC::LocalTensor<float> c0(AscendC::TPosition::CO1, cTileIdx * C0_TILE_BYTES, BASE_M * BASE_N);
    AscendC::MmadParams mp{};
    mp.m = curM;
    mp.n = curN;
    mp.k = curK;
    mp.cmatrixInitVal = (kIdx == 0);
    AscendC::Mmad(c0, a2, b2, mp);
}

template <typename C_GM_TYPE, uint32_t BASE_M, uint32_t BASE_N, QuantMode_t QUANT_MODE>
__aicore__ inline void WriteFixpipeBlock(
    const GemmCubeState& st, AscendC::GlobalTensor<C_GM_TYPE>& cGM, uint32_t mStart, uint32_t mEnd, uint32_t nStart,
    uint32_t nEnd)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    for (uint32_t mi = mStart; mi < mEnd; mi++) {
        uint32_t curM = (mi != st.baseMCount) ? BASE_M : st.tailM;
        uint32_t curMAlign = (mi != st.baseMCount) ? BASE_M : st.tailMAlign;
        for (uint32_t ni = nStart; ni < nEnd; ni++) {
            uint32_t curN = (ni != st.baseNCount) ? BASE_N : st.tailN;
            uint32_t cTileIdx = (mi - mStart) * st.nBlockSize + (ni - nStart);
            AscendC::LocalTensor<float> c0(AscendC::TPosition::CO1, cTileIdx * C0_TILE_BYTES, BASE_M * BASE_N);
            uint64_t cOffset = st.cBaseOffset + static_cast<uint64_t>(mi) * BASE_M * st.tiling.ldc +
                static_cast<uint64_t>(ni) * BASE_N;
            AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp;
            fp.mSize = curM;
            fp.nSize = curN;
            fp.srcStride = curMAlign;
            fp.dstStride = static_cast<uint32_t>(st.tiling.ldc);
            fp.quantPre = QUANT_MODE;
            AscendC::Fixpipe(cGM[cOffset], c0, fp);
        }
    }
}

// ============================================================================
// GEMM_CUBE_KERNEL — Macro generating all BlockMmad cube kernel variants
// ============================================================================
#define GEMM_CUBE_KERNEL(FUNC_NAME, A_TYPE, B_TYPE, C_GM_TYPE, BM, BK, BN, C0_VAL, QUANT_MODE)                     \
    extern "C" __global__ __cube__ void FUNC_NAME(                                                                            \
        __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, GemmExTilingData tiling)                          \
    {                                                                                                              \
        AscendC::InitSocState();                                                                                   \
        GemmCubeState st{};                                                                                        \
        if (!InitGemmState<(BM), (BK), (BN), (C0_VAL)>(st, tiling)) {                                              \
            return;                                                                                                \
        }                                                                                                          \
        AscendC::GlobalTensor<A_TYPE> aGM;                                                                         \
        AscendC::GlobalTensor<B_TYPE> bGM;                                                                         \
        AscendC::GlobalTensor<C_GM_TYPE> cGM;                                                                      \
        aGM.SetGlobalBuffer(reinterpret_cast<__gm__ A_TYPE*>(a));                                                  \
        bGM.SetGlobalBuffer(reinterpret_cast<__gm__ B_TYPE*>(b));                                                  \
        cGM.SetGlobalBuffer(reinterpret_cast<__gm__ C_GM_TYPE*>(c));                                               \
        constexpr uint32_t L1_HALF_SIZE = HardwareInfo<ArchType::ASCEND_V350>::l1Size / 2;                         \
        AscendC::LocalTensor<A_TYPE> a1(AscendC::TPosition::A1, 0, L1_HALF_SIZE);                                  \
        AscendC::LocalTensor<B_TYPE> b1(AscendC::TPosition::B1, L1_HALF_SIZE, L1_HALF_SIZE);                       \
        AscendC::LocalTensor<A_TYPE> a2(AscendC::TPosition::A2, 0, (BM) * (BK));                                   \
        AscendC::LocalTensor<B_TYPE> b2(AscendC::TPosition::B2, 0, (BK) * (BN));                                   \
        for (uint32_t mb = 0; mb < st.mBlockCount; mb++) {                                                         \
            for (uint32_t nb = 0; nb < st.nBlockCount; nb++) {                                                     \
                uint32_t mStart = mb * st.mBlockSize;                                                              \
            uint32_t mEnd = (mStart + st.mBlockSize < st.mLoopCount) ? mStart + st.mBlockSize : st.mLoopCount; \
                uint32_t nStart = nb * st.nBlockSize;                                                              \
            uint32_t nEnd = (nStart + st.nBlockSize < st.nLoopCount) ? nStart + st.nBlockSize : st.nLoopCount; \
                for (uint32_t kIdx = 0; kIdx < st.kLoopCount; kIdx++) {                                            \
                    uint32_t curK =                                                                                \
                        (kIdx == st.kLoopCount - 1) ? (static_cast<uint32_t>(st.tiling.k) - kIdx * (BK)) : (BK);   \
                    for (uint32_t mi = mStart; mi < mEnd; mi++) {                                                  \
                        uint32_t curM = (mi != st.baseMCount) ? (BM) : st.tailM;                                   \
                        uint32_t curMAlign = (mi != st.baseMCount) ? (BM) : st.tailMAlign;                         \
                    LoadATile<A_TYPE, (BM), (BK), (C0_VAL)>(st, aGM, a1, a2, mi, kIdx, curM, curK, curMAlign); \
                        for (uint32_t ni = nStart; ni < nEnd; ni++) {                                              \
                            ProcessNTile<A_TYPE, B_TYPE, (BM), (BK), (BN), (C0_VAL)>(                              \
                                st, bGM, a2, b1, b2, mi, ni, kIdx, mStart, nStart, curM, curK);                    \
                        }                                                                                          \
                    }                                                                                              \
                }                                                                                                  \
                AscendC::PipeBarrier<PIPE_ALL>();                                                                  \
                WriteFixpipeBlock<C_GM_TYPE, (BM), (BN), QUANT_MODE>(st, cGM, mStart, mEnd, nStart, nEnd);         \
            }                                                                                                      \
        }                                                                                                          \
        AscendC::PipeBarrier<PIPE_ALL>();                                                                          \
}

// ── Kernel instantiations ──
GEMM_CUBE_KERNEL(gemm_ex_kernel_fp16, half, half, half, 128, 16, 128, 16, QuantMode_t::F322F16)
GEMM_CUBE_KERNEL(gemm_ex_kernel_bf16, bfloat16_t, bfloat16_t, bfloat16_t, 128, 16, 128, 16, QuantMode_t::F322BF16)
GEMM_CUBE_KERNEL(gemm_ex_kernel_fp32, float, float, float, 32, 8, 16, 8, QuantMode_t::NoQuant)
GEMM_CUBE_KERNEL(gemm_ex_kernel_fp8_e4m3, fp8_e4m3fn_t, fp8_e4m3fn_t, half, 32, 32, 16, 32, QuantMode_t::F322F16)
GEMM_CUBE_KERNEL(gemm_ex_kernel_fp8_e5m2, fp8_e5m2_t, fp8_e5m2_t, half, 32, 32, 16, 32, QuantMode_t::F322F16)
GEMM_CUBE_KERNEL(gemm_ex_kernel_fp8_e5m2_e4m3, fp8_e5m2_t, fp8_e4m3fn_t, half, 32, 32, 16, 32, QuantMode_t::F322F16)
GEMM_CUBE_KERNEL(gemm_ex_kernel_fp8_e4m3_e5m2, fp8_e4m3fn_t, fp8_e5m2_t, half, 32, 32, 16, 32, QuantMode_t::F322F16)
GEMM_CUBE_KERNEL(gemm_ex_kernel_fp16_out_f32, half, half, float, 128, 16, 128, 16, QuantMode_t::NoQuant)
GEMM_CUBE_KERNEL(gemm_ex_kernel_bf16_out_f32, bfloat16_t, bfloat16_t, float, 128, 16, 128, 16, QuantMode_t::NoQuant)

// ── Kernel launcher ──
void gemm_ex_kernel_do(
    uint32_t numBlocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, const GemmExTilingData& tilingData,
    bool isTransA, bool isTransB, GemmDTypeCase dtypeCase)
{
    (void)isTransA;
    (void)isTransB;
    switch (dtypeCase) {
        case GEMM_DTYPE_FP16:
            gemm_ex_kernel_fp16<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_DTYPE_BF16:
            gemm_ex_kernel_bf16<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_DTYPE_FP32:
            gemm_ex_kernel_fp32<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_DTYPE_FP8_E4M3:
            gemm_ex_kernel_fp8_e4m3<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_DTYPE_FP8_E5M2:
            gemm_ex_kernel_fp8_e5m2<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_DTYPE_FP8_E5M2_E4M3:
            gemm_ex_kernel_fp8_e5m2_e4m3<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_DTYPE_FP8_E4M3_E5M2:
            gemm_ex_kernel_fp8_e4m3_e5m2<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_DTYPE_FP16_OUT_F32:
            gemm_ex_kernel_fp16_out_f32<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        case GEMM_DTYPE_BF16_OUT_F32:
            gemm_ex_kernel_bf16_out_f32<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
            break;
        default:
            break;
    }
}

// ============================================================================
// Alpha/Beta post-processing Vector kernels
// ============================================================================
namespace ab_kernel {
using namespace AscendC;

constexpr int32_t AB_TILE_SIZE = 256;
constexpr int32_t AB_BUF_NUM = 1;

template <typename T, typename U>
struct IsSameType {
    static constexpr bool value = false;
};

template <typename T>
struct IsSameType<T, T> {
    static constexpr bool value = true;
};

template <typename TEMP_TYPE, typename C_TYPE, RoundMode OUTPUT_ROUND>
class AlphaBetaKernel {
    static constexpr bool NEED_TEMP_CAST = !IsSameType<TEMP_TYPE, float>::value;
    static constexpr bool IS_FP32_OUTPUT = IsSameType<C_TYPE, float>::value;

public:
    __aicore__ inline void Init(
        __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut, GemmExTilingData tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        m_ = tiling.m;
        n_ = tiling.n;
        ldc_ = tiling.ldc;
        alpha_ = tiling.alpha;
        beta_ = tiling.beta;
        hasBeta_ = tiling.hasBeta;

        tempGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ TEMP_TYPE*>(tempAB), static_cast<uint64_t>(m_) * n_);
        cOrigGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_TYPE*>(cOrig), static_cast<uint64_t>(ldc_) * n_);
        cOutGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_TYPE*>(cOut), static_cast<uint64_t>(ldc_) * n_);

        pipe_->InitBuffer(tempQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(TEMP_TYPE));
        pipe_->InitBuffer(cOrigQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(C_TYPE));
        pipe_->InitBuffer(outQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(C_TYPE));
        if constexpr (NEED_TEMP_CAST) {
            pipe_->InitBuffer(tempFP32Que_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        }
        if constexpr (!IS_FP32_OUTPUT) {
            pipe_->InitBuffer(outFP32Que_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
            pipe_->InitBuffer(cFP32Que_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
            pipe_->InitBuffer(scaledCQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        }
        if constexpr (IS_FP32_OUTPUT) {
            pipe_->InitBuffer(calcQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        }

        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockNum > 1) {
            int32_t colsPerCore = (n_ + blockNum - 1) / blockNum;
            startCol_ = blockIdx * colsPerCore;
            endCol_ = startCol_ + colsPerCore;
            if (endCol_ > n_) {
                endCol_ = n_;
            }
            if (startCol_ >= n_) {
                startCol_ = 0;
                endCol_ = 0;
            }
        } else {
            startCol_ = 0;
            endCol_ = n_;
        }
    }

    __aicore__ inline void Process()
    {
        for (int32_t col = startCol_; col < endCol_; col++) {
            int32_t rowOffset = 0;
            while (rowOffset < m_) {
                int32_t count = m_ - rowOffset;
                if (count > AB_TILE_SIZE) {
                    count = AB_TILE_SIZE;
                }
                ProcessTile(col, rowOffset, count);
                rowOffset += count;
            }
        }
    }

private:
    __aicore__ inline LocalTensor<float> ScaleTemp(LocalTensor<TEMP_TYPE> tempLocal, int32_t count)
    {
        LocalTensor<float> result;
        if constexpr (NEED_TEMP_CAST) {
            LocalTensor<float> tempFP32 = tempFP32Que_.AllocTensor<float>();
            Cast(tempFP32, tempLocal, RoundMode::CAST_NONE, count);
            if constexpr (IS_FP32_OUTPUT) {
                result = outQue_.AllocTensor<float>();
            } else {
                result = outFP32Que_.AllocTensor<float>();
            }
            Muls(result, tempFP32, alpha_, count);
            tempFP32Que_.FreeTensor(tempFP32);
        } else {
            if constexpr (IS_FP32_OUTPUT) {
                result = outQue_.AllocTensor<float>();
            } else {
                result = outFP32Que_.AllocTensor<float>();
            }
            Muls(result, tempLocal, alpha_, count);
        }
        return result;
    }

    __aicore__ inline void AddBetaTerm(LocalTensor<float> alphaResult, LocalTensor<C_TYPE> cOrigLocal, int32_t count)
    {
        if constexpr (IS_FP32_OUTPUT) {
            LocalTensor<float> scaledC = calcQue_.AllocTensor<float>();
            Muls(scaledC, cOrigLocal, beta_, count);
            Add(alphaResult, alphaResult, scaledC, count);
            calcQue_.FreeTensor(scaledC);
        } else {
            LocalTensor<float> cFP32 = cFP32Que_.AllocTensor<float>();
            Cast(cFP32, cOrigLocal, RoundMode::CAST_NONE, count);
            LocalTensor<float> scaledC = scaledCQue_.AllocTensor<float>();
            Muls(scaledC, cFP32, beta_, count);
            cFP32Que_.FreeTensor(cFP32);
            Add(alphaResult, alphaResult, scaledC, count);
            scaledCQue_.FreeTensor(scaledC);
        }
    }

    __aicore__ inline void WriteResult(LocalTensor<float> alphaResult, int32_t col, int32_t rowOffset, int32_t count)
    {
        uint64_t cOutOffset = static_cast<uint64_t>(col) * ldc_ + rowOffset;
        if constexpr (IS_FP32_OUTPUT) {
            outQue_.EnQue(alphaResult);
            LocalTensor<float> resultTile = outQue_.DeQue<float>();
            DataCopy(cOutGlobal_[cOutOffset], resultTile, count);
            outQue_.FreeTensor(resultTile);
        } else {
            LocalTensor<C_TYPE> outTile = outQue_.AllocTensor<C_TYPE>();
            Cast(outTile, alphaResult, OUTPUT_ROUND, count);
            AscendC::PipeBarrier<PIPE_ALL>();
            outFP32Que_.FreeTensor(alphaResult);
            outQue_.EnQue(outTile);
            LocalTensor<C_TYPE> resultTile = outQue_.DeQue<C_TYPE>();
            DataCopy(cOutGlobal_[cOutOffset], resultTile, count);
            outQue_.FreeTensor(resultTile);
        }
    }

    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        uint64_t tempOffset = static_cast<uint64_t>(col) * m_ + rowOffset;
        LocalTensor<TEMP_TYPE> tempTile = tempQue_.AllocTensor<TEMP_TYPE>();
        DataCopy(tempTile, tempGlobal_[tempOffset], count);
        tempQue_.EnQue(tempTile);

        uint64_t cOrigOffset = static_cast<uint64_t>(col) * ldc_ + rowOffset;
        LocalTensor<C_TYPE> cOrigTile = cOrigQue_.AllocTensor<C_TYPE>();
        if (hasBeta_) {
            DataCopy(cOrigTile, cOrigGlobal_[cOrigOffset], count);
        }
        cOrigQue_.EnQue(cOrigTile);

        LocalTensor<TEMP_TYPE> tempLocal = tempQue_.DeQue<TEMP_TYPE>();
        LocalTensor<C_TYPE> cOrigLocal = cOrigQue_.DeQue<C_TYPE>();

        LocalTensor<float> alphaResult = ScaleTemp(tempLocal, count);
        if (hasBeta_) {
            AddBetaTerm(alphaResult, cOrigLocal, count);
        }
        WriteResult(alphaResult, col, rowOffset, count);

        cOrigQue_.FreeTensor(cOrigLocal);
        tempQue_.FreeTensor(tempLocal);
    }

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, AB_BUF_NUM> tempQue_;
    TQue<QuePosition::VECIN, AB_BUF_NUM> cOrigQue_;
    TQue<QuePosition::VECOUT, AB_BUF_NUM> outQue_;
    TQue<QuePosition::VECCALC, AB_BUF_NUM> tempFP32Que_;
    TQue<QuePosition::VECCALC, AB_BUF_NUM> outFP32Que_;
    TQue<QuePosition::VECCALC, AB_BUF_NUM> cFP32Que_;
    TQue<QuePosition::VECCALC, AB_BUF_NUM> scaledCQue_;
    TQue<QuePosition::VECCALC, AB_BUF_NUM> calcQue_;
    GlobalTensor<TEMP_TYPE> tempGlobal_;
    GlobalTensor<C_TYPE> cOrigGlobal_;
    GlobalTensor<C_TYPE> cOutGlobal_;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldc_ = 0;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    int32_t hasBeta_ = 0;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
};

// ── Kernel entry points (one per dtype variant) ──

extern "C" __global__ __aicore__ void gemm_ex_alpha_beta_kernel_fp32(
    __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut, GemmExTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    AlphaBetaKernel<float, float, RoundMode::CAST_NONE> op;
    op.Init(tempAB, cOrig, cOut, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void gemm_ex_alpha_beta_kernel_fp16(
    __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut, GemmExTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    AlphaBetaKernel<half, half, RoundMode::CAST_NONE> op;
    op.Init(tempAB, cOrig, cOut, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void gemm_ex_alpha_beta_kernel_bf16(
    __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut, GemmExTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    AlphaBetaKernel<bfloat16_t, bfloat16_t, RoundMode::CAST_RINT> op;
    op.Init(tempAB, cOrig, cOut, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void gemm_ex_alpha_beta_kernel_f32_to_f16(
    __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut, GemmExTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    AlphaBetaKernel<float, half, RoundMode::CAST_NONE> op;
    op.Init(tempAB, cOrig, cOut, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void gemm_ex_alpha_beta_kernel_f32_to_bf16(
    __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut, GemmExTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    AlphaBetaKernel<float, bfloat16_t, RoundMode::CAST_RINT> op;
    op.Init(tempAB, cOrig, cOut, tiling, &pipe);
    op.Process();
}

} // namespace ab_kernel

void gemm_ex_alpha_beta_do(
    uint32_t numBlocks, void* stream, uint8_t* tempAB, uint8_t* cOrig, uint8_t* cOut,
    const GemmExTilingData& tilingData, GemmDTypeCase dtypeCase, bool useFP32Temp)
{
    if (useFP32Temp) {
        switch (dtypeCase) {
            case GEMM_DTYPE_FP16:
                ab_kernel::gemm_ex_alpha_beta_kernel_f32_to_f16<<<numBlocks, nullptr, stream>>>(
                    tempAB, cOrig, cOut, tilingData);
                break;
            case GEMM_DTYPE_BF16:
                ab_kernel::gemm_ex_alpha_beta_kernel_f32_to_bf16<<<numBlocks, nullptr, stream>>>(
                    tempAB, cOrig, cOut, tilingData);
                break;
            default:
                break;
        }
        return;
    }
    switch (dtypeCase) {
        case GEMM_DTYPE_FP16:
        case GEMM_DTYPE_FP8_E4M3:
        case GEMM_DTYPE_FP8_E5M2:
        case GEMM_DTYPE_FP8_E5M2_E4M3:
        case GEMM_DTYPE_FP8_E4M3_E5M2:
            ab_kernel::gemm_ex_alpha_beta_kernel_fp16<<<numBlocks, nullptr, stream>>>(tempAB, cOrig, cOut, tilingData);
            break;
        case GEMM_DTYPE_BF16:
            ab_kernel::gemm_ex_alpha_beta_kernel_bf16<<<numBlocks, nullptr, stream>>>(tempAB, cOrig, cOut, tilingData);
            break;
        case GEMM_DTYPE_FP32:
            ab_kernel::gemm_ex_alpha_beta_kernel_fp32<<<numBlocks, nullptr, stream>>>(tempAB, cOrig, cOut, tilingData);
            break;
        default:
            break;
    }
}
