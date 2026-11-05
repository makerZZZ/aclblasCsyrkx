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
 * \file strmm_kernel.cpp
 * \brief STRMM Kernel implementation for ascend950 (DAV_3510)
 *        Phase 1: Mirror kernel   (AIV-only, SIMT) - mirrors triangular matrix, fills zeros in missing part
 *        Phase 2: GEMM kernel     (AIC-only, tensor_api) - MMAD result to temp GM
 *        Phase 3: Scale kernel     (AIV-only, SIMT) - alpha*temp -> C
 */

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#include "strmm_tiling_data.h"

namespace te = AscendC::Te;

constexpr uint16_t PIPE_FLAG = 0;

constexpr int64_t L0A_SIZE = 64 * 1024;
constexpr int64_t L1_SIZE = 512 * 1024;

constexpr uint32_t FINAL_ACCUMULATION = 3;
constexpr uint32_t NON_FINAL_ACCUMULATION = 2;

// ================================================================================
// Phase 1: Mirror Kernel (AIV-only, SIMT)
// For TRMM: missing triangle is filled with zeros (NOT symmetric values).
// trans=T/C: mirror stage reads A^T (transpose on read) into workspaceA.
// diag=UNIT: workspaceA diagonal = 1.0 (not A's diagonal).
//
// Column-major storage: A[col*lda + row] stores A[row][col]. The mirror kernel
// reads A and fills workspaceGm in column-major (workspaceGm[col*lda + row] =
// workspaceA[row][col]). The row-major GEMM kernel then reads workspaceGm as
// row-major, which is workspaceA^T — this is the column-major adaptation (see
// strmm_host.cpp for the m↔n swap + side flip trick).
// ================================================================================

template <bool UPLO_IS_UPPER, bool TRANS_IS_T, bool DIAG_IS_UNIT>
__simt_callee__ __aicore__ inline float ComputeMirrorVal(
    const __gm__ float* aGm, int64_t lda64, uint32_t row, uint32_t col)
{
    int64_t row64 = static_cast<int64_t>(row);
    int64_t col64 = static_cast<int64_t>(col);
    float val = 0.0f;

    // Column-major: aGm[col*lda + row] = A[row][col], aGm[row*lda + col] = A[col][row].
    // trans=N reads A[row][col] → aGm[col*lda + row].
    // trans=T reads A^T[row][col] = A[col][row] → aGm[row*lda + col].
    if constexpr (UPLO_IS_UPPER) {
        if constexpr (TRANS_IS_T) {
            if (col <= row) {
                val = aGm[row64 * lda64 + col64];
            }
        } else {
            if (row <= col) {
                val = aGm[col64 * lda64 + row64];
            }
        }
    } else {
        if constexpr (TRANS_IS_T) {
            if (col >= row) {
                val = aGm[row64 * lda64 + col64];
            }
        } else {
            if (row >= col) {
                val = aGm[col64 * lda64 + row64];
            }
        }
    }

    if constexpr (DIAG_IS_UNIT) {
        if (row == col) {
            val = 1.0f;
        }
    }

    return val;
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_T, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StrmmMirrorCompute(
    uint32_t dimA, uint32_t lda, const __gm__ float* aGm, __gm__ float* workspaceGm,
    uint32_t rowStart, uint32_t rowEnd)
{
    int64_t lda64 = static_cast<int64_t>(lda);

    for (uint32_t row = rowStart + threadIdx.x; row < rowEnd; row += blockDim.x) {
        int64_t row64 = static_cast<int64_t>(row);
        for (uint32_t col = 0; col < dimA; ++col) {
            int64_t col64 = static_cast<int64_t>(col);
            workspaceGm[col64 * lda64 + row64] =
                ComputeMirrorVal<UPLO_IS_UPPER, TRANS_IS_T, DIAG_IS_UNIT>(
                    aGm, lda64, row, col);
        }
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_T>
__aicore__ inline void DispatchMirrorDiag(
    bool diagUnit, uint32_t dimA, uint32_t lda,
    const __gm__ float* aGm, __gm__ float* workspaceGm,
    uint32_t rowStart, uint32_t rowEnd, uint32_t nthreads)
{
    if (diagUnit) {
        asc_vf_call<StrmmMirrorCompute<UPLO_IS_UPPER, TRANS_IS_T, true>>(
            dim3{nthreads, 1, 1}, dimA, lda, aGm, workspaceGm, rowStart, rowEnd);
    } else {
        asc_vf_call<StrmmMirrorCompute<UPLO_IS_UPPER, TRANS_IS_T, false>>(
            dim3{nthreads, 1, 1}, dimA, lda, aGm, workspaceGm, rowStart, rowEnd);
    }
}

template <bool UPLO_IS_UPPER>
__aicore__ inline void DispatchMirrorTrans(
    bool transT, bool diagUnit, uint32_t dimA, uint32_t lda,
    const __gm__ float* aGm, __gm__ float* workspaceGm,
    uint32_t rowStart, uint32_t rowEnd, uint32_t nthreads)
{
    if (transT) {
        DispatchMirrorDiag<UPLO_IS_UPPER, true>(diagUnit, dimA, lda, aGm, workspaceGm, rowStart, rowEnd, nthreads);
    } else {
        DispatchMirrorDiag<UPLO_IS_UPPER, false>(diagUnit, dimA, lda, aGm, workspaceGm, rowStart, rowEnd, nthreads);
    }
}

__global__ __aicore__ void strmm_mirror_kernel(
    const __gm__ uint8_t* gmA, __gm__ uint8_t* gmWorkspaceA, const StrmmMirrorTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto ctx = InitMirrorKernelCtx(gmA, gmWorkspaceA, tiling.dimA, tiling.lda, tiling.mirrorRowsPerCore);
    if (ctx.rowStart >= ctx.rowEnd) {
        return;
    }

    bool uploUpper = (tiling.uploMode == ACLBLAS_UPPER);
    bool transT = (tiling.transMode == ACLBLAS_OP_T || tiling.transMode == ACLBLAS_OP_C);
    bool diagUnit = (tiling.diagMode == ACLBLAS_UNIT);

    if (uploUpper) {
        DispatchMirrorTrans<true>(transT, diagUnit, ctx.dimA, ctx.lda, ctx.aGm, ctx.workspaceGm, ctx.rowStart, ctx.rowEnd, tiling.nthreads);
    } else {
        DispatchMirrorTrans<false>(transT, diagUnit, ctx.dimA, ctx.lda, ctx.aGm, ctx.workspaceGm, ctx.rowStart, ctx.rowEnd, tiling.nthreads);
    }
}

void strmm_mirror_kernel_do(
    const uint8_t* gmA, uint8_t* gmWorkspaceA, const StrmmMirrorTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    strmm_mirror_kernel<<<numBlocks, nullptr, stream>>>(gmA, gmWorkspaceA, tiling);
}

// ================================================================================
// Phase 2: GEMM Kernel (AIC-only, tensor_api)
// Side=Left:  left=workspaceA(M×K), right=B(K×N), K=m
// Side=Right: left=B(M×K),    right=workspaceA(K×N), K=n
//
// Column-major adaptation (host-side m↔n swap, see ExecuteStrmmKernels in strmm_host.cpp):
//   The host swaps m↔n and flips side before calling this kernel, so the kernel computes
//   C^T (row-major n×m) instead of C (row-major m×n). This works because:
//   - workspaceA is stored column-major by the mirror kernel: the row-major GEMM kernel
//     reads it as workspaceA^T, which is A^T (trans=N) or A (trans=T).
//   - Column-major B(m×n) in memory == row-major B^T(n×m): the kernel reads B as B^T.
//   The kernel code itself is unchanged; only the tiling parameters (m, n, side, lda, ldb)
//   are swapped by the host. The temp buffer stores C^T in row-major (n rows × tempRowStride).
//
// Uses AscendC::Te (tensor_api) for all data movement and computation:
//   GM→L1: Te::Copy(Te::CopyGM2L1{})       — replaces Nd2Nz + DataCopy
//   L1→L0A: Te::Copy(Te::CopyL12L0A{})     — replaces LoadData2D
//   L1→L0B: Te::Copy(Te::CopyL12L0B{})     — ZN→ZN normal load
//   MMAD:   Te::Mmad(Te::MmadAtom{...})    — replaces Mmad
//   L0C→GM: Te::Copy(Te::CopyL0C2GM{})     — replaces DataCopy + SetFixpipeNz2ndFlag
// ================================================================================

constexpr uint64_t STRMM_FP32_C0 = 8;
constexpr uint64_t STRMM_FRACTAL = 16;
constexpr uint64_t STRMM_L0C_C0 = 16;
constexpr uint64_t STRMM_L1_BUF_NUM = 2;
constexpr uint64_t STRMM_L1_BUF_MASK = STRMM_L1_BUF_NUM - 1;
constexpr uint64_t STRMM_L0_BUF_MASK = 0x1;
constexpr uint64_t STRMM_HALF_L0_SIZE = L0A_SIZE / 2;

template <typename TensorGM, typename TensorL1>
__aicore__ inline void StrmmCopyGM2L1(
    TensorGM gmTensor, TensorL1 tensorL1,
    uint64_t off0, uint64_t off1, uint64_t dim0, uint64_t dim1)
{
    auto gmBlock = gmTensor.Slice(te::MakeCoord(off0, off1), te::MakeShape(dim0, dim1));
    auto copyGM2L1 = te::MakeCopy(te::CopyGM2L1{});
    te::Copy(copyGM2L1, tensorL1, gmBlock);
}

template <typename TensorAL1, typename TensorBL1>
__aicore__ inline void StrmmL0MmadLoop(
    TensorAL1 tensorAL1, TensorBL1 tensorBL1,
    uint64_t curML1, uint64_t curKL1, uint64_t nL0,
    uint64_t kL1Iter, uint64_t iter0, uint64_t baseK,
    uint64_t& l0PingPong)
{
    using T = float;
    uint64_t kL0Iter = CeilDiv<uint64_t>(curKL1, baseK);
    for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
        uint64_t kL0Offset = iter1 * baseK;
        uint64_t curKL0 = (kL0Offset + baseK > curKL1) ? (curKL1 - kL0Offset) : baseK;
        uint64_t l0BufId = l0PingPong & STRMM_L0_BUF_MASK;
        uint64_t l0Offset = STRMM_HALF_L0_SIZE * l0BufId;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        auto layoutAL0 = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<STRMM_FP32_C0>>(curML1, curKL0);
        auto tensorAL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0A, T>(l0Offset), layoutAL0);
        auto tensorBlockAL1 = tensorAL1.Slice(te::MakeCoord(0, kL0Offset), te::MakeShape(curML1, curKL0));
        te::Copy(te::MakeCopy(te::CopyL12L0A{}), tensorAL0, tensorBlockAL1);
        auto layoutBL0 = te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<STRMM_FP32_C0>>(curKL0, nL0);
        auto tensorBL0 = te::MakeTensor(te::MakeMemPtr<te::Location::L0B, T>(l0Offset), layoutBL0);
        auto tensorBlockBL1 = tensorBL1.Slice(te::MakeCoord(kL0Offset, 0), te::MakeShape(curKL0, nL0));
        te::Copy(te::MakeCopy(te::CopyL12L0B{}), tensorBL0, tensorBlockBL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        bool isLastK = (iter0 + 1 == kL1Iter) && (iter1 + 1 == kL0Iter);
        bool isFirstK = (iter0 == 0 && iter1 == 0);
        uint8_t unitFlag = isLastK ? FINAL_ACCUMULATION : NON_FINAL_ACCUMULATION;
        te::MmadParams mmadParams{
            static_cast<uint16_t>(curML1), static_cast<uint16_t>(nL0),
            static_cast<uint16_t>(curKL0), unitFlag, isFirstK};
        auto layoutL0C = te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<STRMM_L0C_C0>>(curML1, nL0);
        auto tensorL0C = te::MakeTensor(te::MakeMemPtr<te::Location::L0C, float>(0), layoutL0C);
        te::Mmad(te::MmadAtom<te::MmadTraits<te::MmadOperation>>{}.with(mmadParams),
            tensorL0C, tensorAL0, tensorBL0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        l0PingPong++;
    }
}

template <typename TensorA, typename TensorB, typename TensorAL1, typename TensorBL1>
__aicore__ inline void StrmmGemmProcessKChunk(
    TensorA gmLeftTensor, TensorB gmRightTensor,
    TensorAL1 tensorAL1, TensorBL1 tensorBL1,
    uint64_t l1BufId,
    uint64_t mOff, uint64_t nOff, uint64_t kOff, uint64_t curK,
    uint64_t mL0, uint64_t nL0,
    uint64_t kL1Iter, uint64_t iter0,
    uint64_t baseK,
    uint64_t& l0PingPong)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    StrmmCopyGM2L1(gmLeftTensor, tensorAL1, mOff, kOff, mL0, curK);
    StrmmCopyGM2L1(gmRightTensor, tensorBL1, kOff, nOff, curK, nL0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    StrmmL0MmadLoop(tensorAL1, tensorBL1, mL0, curK, nL0,
        kL1Iter, iter0, baseK, l0PingPong);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
}

template <typename TensorA, typename TensorB>
__aicore__ inline void StrmmProcessKChunks(
    TensorA gmLeftTensor, TensorB gmRightTensor,
    uint32_t K, uint32_t tileKChunk, uint64_t kL1Iter, uint64_t baseK,
    uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0,
    uint64_t& abL1LoopCnt, uint64_t& l0PingPong)
{
    using T = float;
    for (uint32_t kOff = 0; kOff < K; kOff += tileKChunk) {
        uint32_t curK = Min<uint32_t>(tileKChunk, K - kOff);
        uint64_t l1BufId = abL1LoopCnt & STRMM_L1_BUF_MASK;
        uint64_t l1OffsetA = l1BufId * (L1_SIZE / STRMM_L1_BUF_NUM);
        uint64_t aSideL1Size = RoundUp<uint64_t>(mL0, STRMM_FRACTAL) * RoundUp<uint64_t>(curK, STRMM_FP32_C0);
        uint64_t l1OffsetB = l1OffsetA + aSideL1Size * sizeof(T);
        auto tensorAL1 = te::MakeTensor(te::MakeMemPtr<te::Location::L1, T>(l1OffsetA),
            te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<STRMM_FP32_C0>>(mL0, curK));
        auto tensorBL1 = te::MakeTensor(te::MakeMemPtr<te::Location::L1, T>(l1OffsetB),
            te::MakeFrameLayout<te::ZNLayoutPtn, AscendC::Std::Int<STRMM_FP32_C0>>(curK, nL0));
        StrmmGemmProcessKChunk(gmLeftTensor, gmRightTensor,
            tensorAL1, tensorBL1, l1BufId,
            mOff, nOff, kOff, curK, mL0, nL0,
            kL1Iter, kOff / tileKChunk, baseK, l0PingPong);
        abL1LoopCnt++;
    }
}

template <typename TensorTemp>
__aicore__ inline void StrmmCopyOutL0C2GM(
    TensorTemp gmTempTensor, uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0)
{
    auto gmBlockC = gmTempTensor.Slice(te::MakeCoord(mOff, nOff), te::MakeShape(mL0, nL0));
    auto tensorL0C = te::MakeTensor(te::MakeMemPtr<te::Location::L0C, float>(0),
        te::MakeFrameLayout<te::NZLayoutPtn, AscendC::Std::Int<STRMM_L0C_C0>>(mL0, nL0));
    te::MakeCopy(te::CopyL0C2GM{}).Call(gmBlockC, tensorL0C, te::FixpipeParams{FINAL_ACCUMULATION});
}

template <typename TensorA, typename TensorB, typename TensorTemp>
__aicore__ inline void StrmmProcessTile(
    TensorA gmLeftTensor, TensorB gmRightTensor, TensorTemp gmTempTensor,
    uint32_t K, uint32_t tileKChunk, uint64_t kL1Iter, uint64_t baseK,
    uint32_t mStart, uint32_t mEnd, uint32_t nStart, uint32_t nEnd,
    uint32_t tileM, uint32_t tileN,
    uint64_t& abL1LoopCnt, uint64_t& l0PingPong)
{
    for (uint32_t mOff = mStart; mOff < mEnd; mOff += tileM) {
        uint32_t curTileM = Min<uint32_t>(tileM, mEnd - mOff);
        for (uint32_t nOff = nStart; nOff < nEnd; nOff += tileN) {
            uint32_t curTileN = Min<uint32_t>(tileN, nEnd - nOff);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);
            StrmmProcessKChunks(gmLeftTensor, gmRightTensor,
                K, tileKChunk, kL1Iter, baseK,
                mOff, nOff, curTileM, curTileN, abL1LoopCnt, l0PingPong);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(PIPE_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(PIPE_FLAG);
            StrmmCopyOutL0C2GM(gmTempTensor, mOff, nOff, curTileM, curTileN);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);
        }
    }
}

__global__ __aicore__ void strmm_gemm_kernel(
    __gm__ uint8_t* gmA, const __gm__ uint8_t* gmB, __gm__ uint8_t* gmTemp,
    const StrmmGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    uint32_t K = (tiling.sideMode == ACLBLAS_SIDE_LEFT) ? tiling.m : tiling.n;
    // gmB is const __gm__ uint8_t* (read-only input B). gmLeft/gmRight are kept const to
    // preserve const safety. The const_cast at te::MakeTensor is required because
    // the Blaze tensor API (te::MakeMemPtr<te::Location::GM>) accepts only non-const
    // __gm__ float*. The cast is safe: tensors are only read (GM→L1 copy), never written.
    const __gm__ uint8_t* gmLeft = (tiling.sideMode == ACLBLAS_SIDE_LEFT) ? gmA : gmB;
    const __gm__ uint8_t* gmRight = (tiling.sideMode == ACLBLAS_SIDE_LEFT) ? gmB : gmA;
    uint64_t leftLd = (tiling.sideMode == ACLBLAS_SIDE_LEFT) ? tiling.lda : tiling.ldb;
    uint64_t leftRows = tiling.m;
    uint64_t rightLd = (tiling.sideMode == ACLBLAS_SIDE_LEFT) ? tiling.ldb : tiling.lda;
    uint64_t rightRows = K;
    auto gmLeftTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(const_cast<__gm__ uint8_t*>(gmLeft))),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(leftRows, leftLd));
    auto gmRightTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(const_cast<__gm__ uint8_t*>(gmRight))),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(rightRows, rightLd));
    auto gmTempTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmTemp)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.m, tiling.tempRowStride));
    uint32_t divM = CeilDiv<uint32_t>(tiling.m, tiling.singleCoreM);
    uint32_t divN = CeilDiv<uint32_t>(tiling.n, tiling.singleCoreN);
    uint64_t totalTiles = static_cast<uint64_t>(divM) * static_cast<uint64_t>(divN);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);
    uint64_t l0PingPong = 0;
    uint64_t abL1LoopCnt = 0;
    uint64_t kL1Iter = CeilDiv<uint64_t>(K, tiling.tileKChunk);
    for (uint64_t tileIdx = AscendC::GetBlockIdx(); tileIdx < totalTiles; tileIdx += AscendC::GetBlockNum()) {
        uint64_t coreIdxM = tileIdx / divN;
        uint64_t coreIdxN = tileIdx % divN;
        if (coreIdxM % 2 == 1) { coreIdxN = divN - 1 - coreIdxN; }
        uint32_t mStart = coreIdxM * tiling.singleCoreM;
        uint32_t nStart = coreIdxN * tiling.singleCoreN;
        StrmmProcessTile(gmLeftTensor, gmRightTensor, gmTempTensor,
            K, tiling.tileKChunk, kL1Iter, STRMM_ARCH35_BASE_K,
            mStart, Min<uint32_t>(mStart + tiling.singleCoreM, tiling.m),
            nStart, Min<uint32_t>(nStart + tiling.singleCoreN, tiling.n),
            tiling.tileM, tiling.tileN, abL1LoopCnt, l0PingPong);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);
}

void strmm_gemm_kernel_do(
    uint8_t* gmA, const uint8_t* gmB, uint8_t* gmTemp,
    const StrmmGemmTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    strmm_gemm_kernel<<<numBlocks, nullptr, stream>>>(gmA, gmB, gmTemp, tiling);
}

// ================================================================================
// Phase 3: Scale Kernel (AIV-only, SIMT)
// C = alpha * temp
// alpha value is always carried in tiling (host resolves device scalar via
// ResolveStrmmScalars before launching the pipeline).
// temp stores C^T in row-major (n rows × tempRowStride, tempRowStride = CeilAlign(m)).
//   Reading temp[j*tempRowStride + i] gives C^T[j][i] = C[i][j].
// C is column-major (ldc is column stride, ldc >= m).
//   C[i][j] is at cGm[j*ldc + i].
// ================================================================================

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StrmmScaleCompute(
    uint32_t n, uint32_t ldc, uint32_t tempRowStride,
    float alphaVal,
    const __gm__ float* __restrict tempGm, __gm__ float* __restrict cGm,
    uint32_t rowStart, uint32_t rowEnd)
{
    int64_t ldc64 = static_cast<int64_t>(ldc);
    int64_t tempStride64 = static_cast<int64_t>(tempRowStride);

    for (uint32_t i = rowStart + threadIdx.x; i < rowEnd; i += blockDim.x) {
        int64_t i64 = static_cast<int64_t>(i);
        for (uint32_t j = 0; j < n; ++j) {
            int64_t j64 = static_cast<int64_t>(j);
            float tempVal = tempGm[j64 * tempStride64 + i64];
            cGm[j64 * ldc64 + i64] = alphaVal * tempVal;
        }
    }
}

__global__ __aicore__ void strmm_scale_kernel(
    const __gm__ uint8_t* gmTemp, __gm__ uint8_t* gmC,
    const StrmmScaleTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* tempGm = reinterpret_cast<const __gm__ float* __restrict>(gmTemp);
    auto* cGm = reinterpret_cast<__gm__ float* __restrict>(gmC);

    uint32_t m = tiling.m;
    uint32_t n = tiling.n;
    uint32_t ldc = tiling.ldc;
    uint32_t tempRowStride = tiling.tempRowStride;
    float alphaVal = tiling.alphaVal;

    int32_t blkIdx = AscendC::GetBlockIdx();
    uint32_t rowStart = static_cast<uint32_t>(blkIdx) * tiling.scaleRowsPerCore;
    uint32_t rowEnd = Min<uint32_t>(rowStart + tiling.scaleRowsPerCore, m);
    if (rowStart >= rowEnd) {
        return;
    }

    asc_vf_call<StrmmScaleCompute>(
        dim3{tiling.nthreads, 1, 1},
        n, ldc, tempRowStride, alphaVal,
        tempGm, cGm, rowStart, rowEnd);
}

void strmm_scale_kernel_do(
    const uint8_t* gmTemp, uint8_t* gmC,
    const StrmmScaleTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    strmm_scale_kernel<<<numBlocks, nullptr, stream>>>(gmTemp, gmC, tiling);
}
