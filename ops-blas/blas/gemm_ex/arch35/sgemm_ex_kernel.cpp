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
 * \file sgemm_ex_kernel.cpp
 * \brief FP32 GEMM kernel for arch35 (DAV_3510).
 *
 * SIMD membase implementation using BlockMmad low-level API.
 * Dual-kernel architecture:
 *   1. Cube kernel:  op(B') * op(A') matrix multiply (column-major trick)
 *   2. Vector kernel: C = alpha * tempAB + beta * C post-processing
 *
 * FP32 tile parameters: baseM=32, baseK=8, baseN=16, C0=8
 */

#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "sgemm_ex_tiling_data.h"
#include "sgemm_ex_kernel.h"
#include "common/arch/hardware.h"
#include "common/helper/kernel_utils.h"

// ============================================================================
// Cube Kernel — standalone __aicore__ functions (arch35 constraint)
// ============================================================================

struct SgemmCubeState {
    SgemmExTilingData tiling;
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

__aicore__ inline void ComputeBaseOffsets(SgemmCubeState& st)
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
__aicore__ inline void Compute2DBlocking(SgemmCubeState& st)
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
__aicore__ inline bool InitGemmState(SgemmCubeState& st, SgemmExTilingData tiling)
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
    // Guard against excess cores: when nBlocks*singleCoreN > n (or mBlocks*singleCoreM > m),
    // the subtraction would underflow (uint32_t). Return early for cores with no work.
    uint32_t mStart = st.mBlockIdx * static_cast<uint32_t>(st.tiling.singleCoreM);
    uint32_t nStart = st.nBlockIdx * static_cast<uint32_t>(st.tiling.singleCoreN);
    if (mStart >= static_cast<uint32_t>(st.tiling.m) || nStart >= static_cast<uint32_t>(st.tiling.n)) {
        return false;
    }
    st.actualM = static_cast<uint32_t>(st.tiling.m) - mStart;
    if (st.actualM > static_cast<uint32_t>(st.tiling.singleCoreM)) {
        st.actualM = static_cast<uint32_t>(st.tiling.singleCoreM);
    }
    st.actualN = static_cast<uint32_t>(st.tiling.n) - nStart;
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

template <uint32_t BASE_M, uint32_t BASE_K, uint32_t C0_VAL>
__aicore__ inline void LoadATile(
    const SgemmCubeState& st, AscendC::GlobalTensor<float>& aGM, AscendC::LocalTensor<float>& a1,
    AscendC::LocalTensor<float>& a2, uint32_t mi, uint32_t kIdx, uint32_t curM, uint32_t curK, uint32_t curMAlign)
{
    constexpr uint32_t CUBE_BLOCK = HardwareInfo<ArchType::ASCEND_V350>::l1l0BlockSize;
    uint64_t aOffset;
    AscendC::Nd2NzParams ndA;
    ndA.ndNum = 1;
    ndA.dstNzNStride = 1;
    ndA.dstNzMatrixStride = 0;
    ndA.srcNdMatrixStride = 0;
    if (!st.tiling.isTransA) {
        aOffset = st.aBaseOffset + static_cast<uint64_t>(mi) * BASE_M * st.tiling.lda + static_cast<uint64_t>(kIdx) * BASE_K;
        ndA.nValue = curM;
        ndA.dValue = curK;
        ndA.srcDValue = static_cast<uint32_t>(st.tiling.lda);
        ndA.dstNzC0Stride = RoundUp(curM, CUBE_BLOCK);
    } else {
        aOffset = st.aBaseOffset + static_cast<uint64_t>(kIdx) * BASE_K * st.tiling.lda + static_cast<uint64_t>(mi) * BASE_M;
        ndA.nValue = curK;
        ndA.dValue = curM;
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

template <uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline void LoadBTile(
    const SgemmCubeState& st, AscendC::GlobalTensor<float>& bGM, AscendC::LocalTensor<float>& b1,
    AscendC::LocalTensor<float>& b2, uint32_t ni, uint32_t kIdx, uint32_t curK, uint32_t curN, uint32_t curNAlign)
{
    constexpr uint32_t CUBE_BLOCK = HardwareInfo<ArchType::ASCEND_V350>::l1l0BlockSize;
    uint64_t bOffset;
    AscendC::Nd2NzParams ndB;
    ndB.ndNum = 1;
    ndB.dstNzNStride = 1;
    ndB.dstNzMatrixStride = 0;
    ndB.srcNdMatrixStride = 0;
    if (!st.tiling.isTransB) {
        bOffset = st.bBaseOffset + static_cast<uint64_t>(kIdx) * BASE_K * st.tiling.ldb +
                  static_cast<uint64_t>(ni) * BASE_N;
        ndB.nValue = curK;
        ndB.dValue = curN;
        ndB.srcDValue = static_cast<uint32_t>(st.tiling.ldb);
        ndB.dstNzC0Stride = RoundUp(curK, CUBE_BLOCK);
    } else {
        bOffset = st.bBaseOffset + static_cast<uint64_t>(ni) * BASE_N * st.tiling.ldb +
                  static_cast<uint64_t>(kIdx) * BASE_K;
        ndB.nValue = curN;
        ndB.dValue = curK;
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
        ldB.kStep = CeilDiv(static_cast<uint32_t>(curNAlign * sizeof(float)), VECTOR_ALIGN_BYTES);
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

template <uint32_t BASE_M, uint32_t BASE_K, uint32_t BASE_N, uint32_t C0_VAL>
__aicore__ inline void ProcessNTile(
    const SgemmCubeState& st,
    AscendC::GlobalTensor<float>& bGM,
    AscendC::LocalTensor<float>& a2,
    AscendC::LocalTensor<float>& b1,
    AscendC::LocalTensor<float>& b2,
    uint32_t mi, uint32_t ni, uint32_t kIdx,
    uint32_t mStart, uint32_t nStart, uint32_t curM, uint32_t curK)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    uint32_t curN = (ni != st.baseNCount) ? BASE_N : st.tailN;
    uint32_t curNAlign = (ni != st.baseNCount) ? BASE_N : st.tailNAlign;
    LoadBTile<BASE_K, BASE_N, C0_VAL>(st, bGM, b1, b2, ni, kIdx, curK, curN, curNAlign);
    uint32_t cTileIdx = (mi - mStart) * st.nBlockSize + (ni - nStart);
    AscendC::LocalTensor<float> c0(AscendC::TPosition::CO1, cTileIdx * C0_TILE_BYTES, BASE_M * BASE_N);
    AscendC::MmadParams mp{};
    mp.m = curM;
    mp.n = curN;
    mp.k = curK;
    mp.cmatrixInitVal = (kIdx == 0);
    AscendC::Mmad(c0, a2, b2, mp);
}

// Fixpipe nSize alignment: tail block nSize aligned to 16 when safe.
// Fixpipe requires nSize to be a multiple of 16 (without channelSplit).
// Safe condition: ni*BASE_N + alignedN <= ldc (padding region absorbs extra elements).
// Otherwise fallback to curN (hardware tolerates non-aligned nSize, ref: gemm_ex).
// tempAB scenario: ldc=CeilAlign(m,16) >= ni*BASE_N + 16 (always holds), always aligned.
// Direct-write-C scenario: ldc=original ldc, aligned when ldc >= ni*BASE_N + CeilAlign(tailN,16).
template <uint32_t BASE_M, uint32_t BASE_N>
__aicore__ inline void WriteFixpipeBlock(
    const SgemmCubeState& st,
    AscendC::GlobalTensor<float>& cGM,
    uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd)
{
    constexpr uint32_t C0_TILE_BYTES = BASE_M * BASE_N * sizeof(float);
    for (uint32_t mi = mStart; mi < mEnd; mi++) {
        uint32_t curM = (mi != st.baseMCount) ? BASE_M : st.tailM;
        uint32_t curMAlign = (mi != st.baseMCount) ? BASE_M : st.tailMAlign;
        for (uint32_t ni = nStart; ni < nEnd; ni++) {
            uint32_t curN = (ni != st.baseNCount) ? BASE_N : st.tailN;
            uint32_t cTileIdx = (mi - mStart) * st.nBlockSize + (ni - nStart);
            AscendC::LocalTensor<float> c0(
                AscendC::TPosition::CO1, cTileIdx * C0_TILE_BYTES, BASE_M * BASE_N);
            uint64_t cOffset = st.cBaseOffset +
                               static_cast<uint64_t>(mi) * BASE_M * st.tiling.ldc +
                               static_cast<uint64_t>(ni) * BASE_N;
            // Tail block nSize alignment: align to 16 when safe, fallback to curN otherwise
            uint32_t nSizeFixpipe = curN;
            if (ni == st.baseNCount && curN != BASE_N) {
                uint32_t alignedN = RoundUp(curN, 16u);
                uint32_t blockEnd = ni * BASE_N + alignedN;
                nSizeFixpipe = (blockEnd <= static_cast<uint32_t>(st.tiling.ldc)) ? alignedN : curN;
            }
            AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp;
            fp.mSize = curM;
            fp.nSize = nSizeFixpipe;
            fp.srcStride = curMAlign;
            fp.dstStride = static_cast<uint32_t>(st.tiling.ldc);
            fp.quantPre = QuantMode_t::NoQuant;
            AscendC::Fixpipe(cGM[cOffset], c0, fp);
        }
    }
}

// FP32 Cube kernel entry: baseM=32, baseK=8, baseN=16, c0=8
extern "C" __global__ __cube__ void sgemm_ex_cube_kernel(
    __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, SgemmExTilingData tiling)
{
    AscendC::InitSocState();
    SgemmCubeState st{};
    if (!InitGemmState<32, 8, 16, 8>(st, tiling)) {
        return;
    }

    AscendC::GlobalTensor<float> aGM;
    AscendC::GlobalTensor<float> bGM;
    AscendC::GlobalTensor<float> cGM;
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    bGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(c));

    constexpr uint32_t L1_HALF_SIZE = HardwareInfo<ArchType::ASCEND_V350>::l1Size / 2;
    AscendC::LocalTensor<float> a1(AscendC::TPosition::A1, 0, L1_HALF_SIZE);
    AscendC::LocalTensor<float> b1(AscendC::TPosition::B1, L1_HALF_SIZE, L1_HALF_SIZE);
    AscendC::LocalTensor<float> a2(AscendC::TPosition::A2, 0, 32 * 8);
    AscendC::LocalTensor<float> b2(AscendC::TPosition::B2, 0, 8 * 16);

    for (uint32_t mb = 0; mb < st.mBlockCount; mb++) {
        for (uint32_t nb = 0; nb < st.nBlockCount; nb++) {
            uint32_t mStart = mb * st.mBlockSize;
            uint32_t mEnd = (mStart + st.mBlockSize < st.mLoopCount) ? mStart + st.mBlockSize : st.mLoopCount;
            uint32_t nStart = nb * st.nBlockSize;
            uint32_t nEnd = (nStart + st.nBlockSize < st.nLoopCount) ? nStart + st.nBlockSize : st.nLoopCount;
            for (uint32_t kIdx = 0; kIdx < st.kLoopCount; kIdx++) {
                uint32_t curK = (kIdx == st.kLoopCount - 1)
                                    ? (static_cast<uint32_t>(st.tiling.k) - kIdx * 8)
                                    : 8;
                for (uint32_t mi = mStart; mi < mEnd; mi++) {
                    uint32_t curM = (mi != st.baseMCount) ? 32 : st.tailM;
                    uint32_t curMAlign = (mi != st.baseMCount) ? 32 : st.tailMAlign;
                    LoadATile<32, 8, 8>(st, aGM, a1, a2, mi, kIdx, curM, curK, curMAlign);
                    for (uint32_t ni = nStart; ni < nEnd; ni++) {
                        ProcessNTile<32, 8, 16, 8>(
                            st, bGM, a2, b1, b2, mi, ni, kIdx, mStart, nStart, curM, curK);
                    }
                }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            WriteFixpipeBlock<32, 16>(st, cGM, mStart, mEnd, nStart, nEnd);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// Cube kernel launcher
void sgemm_ex_kernel_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR a, GM_ADDR b, GM_ADDR c,
    const SgemmExTilingData& tilingData)
{
    sgemm_ex_cube_kernel<<<numBlocks, nullptr, stream>>>(a, b, c, tilingData);
}

// ============================================================================
// Vector Kernel — alpha/beta post-processing (FP32, no Cast needed)
// ============================================================================

using namespace AscendC;

constexpr int32_t AB_TILE_SIZE = 256;
constexpr int32_t AB_BUF_NUM = 1;

class SgemmExAlphaBetaKernel {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut,
        SgemmExTilingData tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        m_ = tiling.m;
        n_ = tiling.n;
        ldc_ = tiling.ldc;     // tempAB row stride (= CeilAlign(m, 16))
        cLdc_ = tiling.cLdc;   // C matrix original ldc
        alpha_ = tiling.alpha;
        beta_ = tiling.beta;
        hasBeta_ = tiling.hasBeta;

        tempGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tempAB), static_cast<uint64_t>(ldc_) * n_);
        cOrigGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cOrig), static_cast<uint64_t>(cLdc_) * n_);
        cOutGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cOut), static_cast<uint64_t>(cLdc_) * n_);

        pipe_->InitBuffer(tempQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        pipe_->InitBuffer(cOrigQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        pipe_->InitBuffer(outQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        pipe_->InitBuffer(calcQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));

        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        if (blockNum > 1) {
            int32_t colsPerCore = (n_ + static_cast<int32_t>(blockNum) - 1) / static_cast<int32_t>(blockNum);
            startCol_ = static_cast<int32_t>(blockIdx) * colsPerCore;
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
    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        // 1. Load tempAB tile from workspace (row stride = ldc_ = CeilAlign(m, 16))
        uint64_t tempOffset = static_cast<uint64_t>(col) * ldc_ + rowOffset;
        LocalTensor<float> tempTile = tempQue_.AllocTensor<float>();
        DataCopy(tempTile, tempGlobal_[tempOffset], count);
        tempQue_.EnQue(tempTile);

        // 2. Load C_orig tile if hasBeta (row stride = cLdc_ = original ldc)
        uint64_t cOrigOffset = static_cast<uint64_t>(col) * cLdc_ + rowOffset;
        LocalTensor<float> cOrigTile = cOrigQue_.AllocTensor<float>();
        if (hasBeta_) {
            DataCopy(cOrigTile, cOrigGlobal_[cOrigOffset], count);
        }
        cOrigQue_.EnQue(cOrigTile);

        LocalTensor<float> tempLocal = tempQue_.DeQue<float>();
        LocalTensor<float> cOrigLocal = cOrigQue_.DeQue<float>();

        // 3. result = alpha * tempAB
        LocalTensor<float> alphaResult = outQue_.AllocTensor<float>();
        Muls(alphaResult, tempLocal, alpha_, count);

        // 4. result += beta * C_orig (if hasBeta)
        if (hasBeta_) {
            LocalTensor<float> scaledC = calcQue_.AllocTensor<float>();
            Muls(scaledC, cOrigLocal, beta_, count);
            Add(alphaResult, alphaResult, scaledC, count);
            calcQue_.FreeTensor(scaledC);
        }

        // 5. Write result back to GM (row stride = cLdc_ = original ldc)
        uint64_t cOutOffset = static_cast<uint64_t>(col) * cLdc_ + rowOffset;
        outQue_.EnQue(alphaResult);
        LocalTensor<float> resultTile = outQue_.DeQue<float>();
        DataCopy(cOutGlobal_[cOutOffset], resultTile, count);
        outQue_.FreeTensor(resultTile);

        cOrigQue_.FreeTensor(cOrigLocal);
        tempQue_.FreeTensor(tempLocal);
    }

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, AB_BUF_NUM> tempQue_;
    TQue<QuePosition::VECIN, AB_BUF_NUM> cOrigQue_;
    TQue<QuePosition::VECOUT, AB_BUF_NUM> outQue_;
    TQue<QuePosition::VECCALC, AB_BUF_NUM> calcQue_;
    GlobalTensor<float> tempGlobal_;
    GlobalTensor<float> cOrigGlobal_;
    GlobalTensor<float> cOutGlobal_;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldc_ = 0;
    int32_t cLdc_ = 0;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    int32_t hasBeta_ = 0;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
};

extern "C" __global__ __aicore__ void sgemm_ex_alpha_beta_kernel(
    __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut,
    SgemmExTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SgemmExAlphaBetaKernel op;
    op.Init(tempAB, cOrig, cOut, tiling, &pipe);
    op.Process();
}

// Vector kernel launcher
void sgemm_ex_alpha_beta_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR tempAB, GM_ADDR cOrig, GM_ADDR cOut,
    const SgemmExTilingData& tilingData)
{
    sgemm_ex_alpha_beta_kernel<<<numBlocks, nullptr, stream>>>(tempAB, cOrig, cOut, tilingData);
}
