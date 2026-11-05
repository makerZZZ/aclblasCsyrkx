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
 * \file ssymm_kernel.cpp
 * \brief SSYMM Kernel implementation for ascend950 (DAV_3510)
 *        Phase 1: Mirror kernel   (AIV-only, SIMT) - mirrors symmetric triangle
 *        Phase 2: GEMM kernel     (AIC-only, SIMD membase, double-buffered) - MMAD result to temp GM
 *        Phase 3: Scale kernel     (AIV-only, SIMT) - alpha*temp + beta*C -> C
 */

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#include "ssymm_tiling_data.h"

constexpr int64_t B32_C0_SIZE = 8;
constexpr uint16_t TWO_ALIGN = 2;

constexpr uint16_t ZERO_FLAG = 0;
constexpr uint16_t FIRST_FLAG = 1;
constexpr uint32_t DB_COUNT = 2;

// ================================================================================
// L0/L1 buffer sizes (bytes) — matching matmul_fp32 arch35
// ================================================================================

constexpr int64_t L0A_SIZE = 64 * 1024;
constexpr int64_t L0B_SIZE = 64 * 1024;
constexpr int64_t L0C_SIZE = 256 * 1024;
constexpr int64_t L1_SIZE = 512 * 1024;

constexpr uint64_t HALF_L0_SIZE = L0A_SIZE / DB_COUNT / sizeof(float);

// ================================================================================
// Phase 1: Mirror Kernel (AIV-only, SIMT)
// Copies A to workspace and mirrors the symmetric triangle in-place.
// ================================================================================

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SsymmMirrorCompute(
    uint32_t dimA, uint32_t lda, const __gm__ float* aGm, __gm__ float* workspaceGm,
    uint32_t rowStart, uint32_t rowEnd)
{
    int64_t lda64 = static_cast<int64_t>(lda);
    // Column-major: A[col*lda + row] stores A[row][col]. The mirror kernel reads A and
    // fills workspaceGm in column-major. Since the resulting workspace is a full symmetric
    // matrix, its row-major and column-major memory layouts are identical — the row-major
    // GEMM kernel reads it correctly either way.
    // vec4 optimization is dropped because column-major access by row is non-contiguous
    // (stride = lda between elements); correctness takes priority.
    for (uint32_t row = rowStart + threadIdx.x; row < rowEnd; row += blockDim.x) {
        int64_t row64 = static_cast<int64_t>(row);
        for (uint32_t col = 0; col < dimA; ++col) {
            int64_t col64 = static_cast<int64_t>(col);
            if constexpr (UPLO_IS_UPPER) {
                // UPPER: upper triangle (col >= row) stored directly,
                //         lower triangle (col < row) mirrored from symmetric position.
                if (col >= row) {
                    workspaceGm[col64 * lda64 + row64] = aGm[col64 * lda64 + row64];
                } else {
                    workspaceGm[col64 * lda64 + row64] = aGm[row64 * lda64 + col64];
                }
            } else {
                // LOWER: lower triangle (col <= row) stored directly,
                //         upper triangle (col > row) mirrored from symmetric position.
                if (col <= row) {
                    workspaceGm[col64 * lda64 + row64] = aGm[col64 * lda64 + row64];
                } else {
                    workspaceGm[col64 * lda64 + row64] = aGm[row64 * lda64 + col64];
                }
            }
        }
    }
}

__global__ __aicore__ void ssymm_mirror_kernel(
    const __gm__ uint8_t* gmA, __gm__ uint8_t* gmWorkspaceA, const SsymmMirrorTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto ctx = InitMirrorKernelCtx(gmA, gmWorkspaceA, tiling.dimA, tiling.lda, tiling.mirrorRowsPerCore);
    if (ctx.rowStart >= ctx.rowEnd) {
        return;
    }

    if (tiling.uploMode == ACLBLAS_UPPER) {
        asc_vf_call<SsymmMirrorCompute<true>>(
            dim3{tiling.nthreads, 1, 1}, ctx.dimA, ctx.lda, ctx.aGm, ctx.workspaceGm, ctx.rowStart, ctx.rowEnd);
    } else {
        asc_vf_call<SsymmMirrorCompute<false>>(
            dim3{tiling.nthreads, 1, 1}, ctx.dimA, ctx.lda, ctx.aGm, ctx.workspaceGm, ctx.rowStart, ctx.rowEnd);
    }
}

void ssymm_mirror_kernel_do(
    const uint8_t* gmA, uint8_t* gmWorkspaceA, const SsymmMirrorTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    ssymm_mirror_kernel<<<numBlocks, nullptr, stream>>>(gmA, gmWorkspaceA, tiling);
}

// ================================================================================
// Phase 2: GEMM Kernel (AIC-only, SIMD membase, double-buffered)
// MMAD: left_matrix * right_matrix -> temp GM (row-major, NZ2ND via DataCopyCO12DstParams)
//
// Column-major adaptation (host-side m↔n swap, see ExecuteSsymmKernels in ssymm_host.cpp):
//   The host swaps m↔n and flips side before calling this kernel, so the kernel computes
//   C^T (row-major n×m) instead of C (row-major m×n). This works because:
//   - workspaceA is a full symmetric matrix: row-major == column-major in memory.
//   - Column-major B(m×n) in memory == row-major B^T(n×m): the kernel reads B as B^T.
//   The kernel code itself is unchanged; only the tiling parameters (m, n, side, lda, ldb)
//   are swapped by the host. The temp buffer stores C^T in row-major (n rows × tempRowStride).
//
//   Kernel view (post-swap):  Side=Left:  left=A_sym(M×K), right=B(K×N), K=orig_m
//                             Side=Right: left=B(M×K),    right=A_sym(K×N), K=orig_n
// ================================================================================

template <typename T>
__aicore__ inline void SsymmCopyInA1(
    const AscendC::GlobalTensor<T>& aGlobal, const AscendC::LocalTensor<T>& al1Local,
    uint64_t curML1, uint64_t curKL1, uint64_t lda)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = curML1;
    nd2nzParams.dValue = curKL1;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = lda;
    nd2nzParams.dstNzC0Stride = RoundUp<int64_t>(curML1, AscendC::BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
}

template <typename T>
__aicore__ inline void SsymmCopyInB1(
    const AscendC::GlobalTensor<T>& bGlobal, const AscendC::LocalTensor<T>& bl1Local,
    uint64_t curNL1, uint64_t curKL1, uint64_t ldb)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = curKL1;
    nd2nzParams.dValue = curNL1;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = ldb;
    nd2nzParams.dstNzC0Stride = RoundUp<int64_t>(curKL1, AscendC::BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
}

template <typename T>
__aicore__ inline void SsymmCopyInA2(
    const AscendC::LocalTensor<T>& al0Local, const AscendC::LocalTensor<T>& al1Local,
    uint64_t curML1, uint64_t curKL1, uint64_t mL0, uint64_t kL0)
{
    AscendC::LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;
    loadDataParams.mStep = CeilDiv<int64_t>(mL0, AscendC::BLOCK_CUBE);
    loadDataParams.kStep = CeilDiv<int64_t>(kL0, B32_C0_SIZE);
    loadDataParams.srcStride = CeilDiv<int64_t>(curML1, AscendC::BLOCK_CUBE);
    loadDataParams.dstStride = loadDataParams.mStep;
    loadDataParams.ifTranspose = false;
    AscendC::LoadData<T>(al0Local, al1Local, loadDataParams);
}

template <typename T>
__aicore__ inline void SsymmCopyInB2(
    const AscendC::LocalTensor<T>& bl0Local, const AscendC::LocalTensor<T>& bl1Local,
    uint64_t curKL1, uint64_t curKL0, uint64_t nL0)
{
    AscendC::LoadData2DParamsV2 b2LoadDataParams;
    b2LoadDataParams.mStartPosition = 0;
    b2LoadDataParams.kStartPosition = 0;
    b2LoadDataParams.mStep = CeilDiv<int64_t>(curKL0, AscendC::BLOCK_CUBE);
    b2LoadDataParams.kStep = CeilDiv<int64_t>(nL0, AscendC::BLOCK_CUBE) * TWO_ALIGN;
    b2LoadDataParams.dstStride = b2LoadDataParams.kStep >> 1;
    b2LoadDataParams.srcStride = CeilDiv<int64_t>(curKL1, AscendC::BLOCK_CUBE);
    b2LoadDataParams.ifTranspose = true;
    AscendC::LoadData<T>(bl0Local, bl1Local, b2LoadDataParams);
}

template <typename T>
__aicore__ inline void SsymmCopyOut(
    const AscendC::GlobalTensor<T>& tempGlobal, const AscendC::LocalTensor<float>& c1Local,
    uint64_t mL0, uint64_t nL0, uint64_t tempRowStride)
{
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.nSize = nL0;
    intriParams.mSize = mL0;
    intriParams.dstStride = tempRowStride;
    intriParams.srcStride = RoundUp<int64_t>(mL0, AscendC::BLOCK_CUBE);
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.reluPre = 0;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    AscendC::DataCopy(tempGlobal, c1Local, intriParams);
}

template <typename T>
__aicore__ inline void ProcessKChunkDB(
    const AscendC::GlobalTensor<T>& aGlobal, const AscendC::GlobalTensor<T>& bGlobal,
    const AscendC::LocalTensor<T>& l1Local,
    const AscendC::LocalTensor<T>& al0Local, const AscendC::LocalTensor<T>& bl0Local,
    const AscendC::LocalTensor<T>& l0cLocal,
    const SsymmGemmTilingData& tiling,
    uint32_t mOff, uint32_t nOff, uint32_t kOff, uint32_t curK,
    uint64_t mL0, uint64_t nL0,
    uint64_t l1BufStrideA, uint64_t l1BufStrideB,
    uint64_t l1BufId, uint64_t& l0PingPong)
{
    uint64_t curML1 = mL0;
    uint64_t curKL1 = curK;

    uint64_t offsetAL1 = l1BufStrideA * l1BufId;
    if (tiling.sideMode == ACLBLAS_SIDE_LEFT) {
        SsymmCopyInA1<T>(aGlobal[mOff * tiling.lda + kOff], l1Local[offsetAL1],
            curML1, curKL1, tiling.lda);
    } else {
        SsymmCopyInA1<T>(bGlobal[mOff * tiling.ldb + kOff], l1Local[offsetAL1],
            curML1, curKL1, tiling.ldb);
    }

    uint64_t offsetBL1 = l1BufStrideA * DB_COUNT + l1BufStrideB * l1BufId;
    if (tiling.sideMode == ACLBLAS_SIDE_LEFT) {
        SsymmCopyInB1<T>(bGlobal[kOff * tiling.ldb + nOff], l1Local[offsetBL1],
            nL0, curKL1, tiling.ldb);
    } else {
        SsymmCopyInB1<T>(aGlobal[kOff * tiling.lda + nOff], l1Local[offsetBL1],
            nL0, curKL1, tiling.lda);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

    uint64_t kL0TileNum = CeilDiv<int64_t>(curKL1, B32_C0_SIZE);
    uint64_t tailKL0 = curKL1 - (kL0TileNum - 1) * B32_C0_SIZE;
    uint64_t offsetAL0 = offsetAL1;
    uint64_t offsetBL0 = offsetBL1;
    for (uint64_t iter1 = 0; iter1 < kL0TileNum; ++iter1) {
        uint64_t curKL0 = (iter1 + 1 == kL0TileNum) ? tailKL0 : B32_C0_SIZE;
        bool isFirstLoop = (kOff == 0 && iter1 == 0);
        uint64_t l0BufId = l0PingPong & 0x1;
        uint64_t l0Offset = HALF_L0_SIZE * l0BufId;

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);

        SsymmCopyInA2<T>(al0Local[l0Offset], l1Local[offsetAL0], curML1, curKL1, mL0, curKL0);
        SsymmCopyInB2<T>(bl0Local[l0Offset], l1Local[offsetBL0], curKL1, curKL0, nL0);

        offsetAL0 += RoundUp<int64_t>(curML1, AscendC::BLOCK_CUBE) * B32_C0_SIZE;
        offsetBL0 += B32_C0_SIZE * B32_C0_SIZE;

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);

        AscendC::MmadParams mmadParams;
        mmadParams.m = mL0;
        mmadParams.n = nL0;
        mmadParams.k = curKL0;
        mmadParams.cmatrixSource = false;
        mmadParams.cmatrixInitVal = isFirstLoop;
        mmadParams.unitFlag = 0;
        AscendC::Mmad(l0cLocal, al0Local[l0Offset], bl0Local[l0Offset], mmadParams);

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
        l0PingPong++;
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
}

__global__ __aicore__ void ssymm_gemm_kernel(
    __gm__ uint8_t* gmA, const __gm__ uint8_t* gmB, __gm__ uint8_t* gmTemp,
    const SsymmGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    uint32_t K = (tiling.sideMode == ACLBLAS_SIDE_LEFT) ? tiling.m : tiling.n;

    AscendC::GlobalTensor<float> aGlobal, bGlobal, tempGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmA));
    // gmB is const __gm__ uint8_t* (read-only input B). const_cast is required because
    // SetGlobalBuffer accepts only non-const __gm__ float*. The cast is safe:
    // bGlobal is only read (GM→L1 copy), never written.
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(const_cast<__gm__ uint8_t*>(gmB)));
    tempGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gmTemp));

    AscendC::LocalTensor<float> al0Local{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<float> bl0Local{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> l0cLocal{AscendC::TPosition::CO1, 0, L0C_SIZE};
    AscendC::LocalTensor<float> l1Local{AscendC::TPosition::A1, 0, L1_SIZE};

    uint32_t curBlockIdx = AscendC::GetBlockIdx();
    uint32_t blockNum = AscendC::GetBlockNum();
    uint32_t divM = CeilDiv<uint32_t>(tiling.m, tiling.singleCoreM);
    uint32_t divN = CeilDiv<uint32_t>(tiling.n, tiling.singleCoreN);
    uint32_t totalTiles = divM * divN;

    uint64_t l1BufStrideA = static_cast<uint64_t>(tiling.tileM) * static_cast<uint64_t>(tiling.tileKChunk);
    uint64_t l1BufStrideB = RoundUp<int64_t>(static_cast<uint64_t>(tiling.tileKChunk), AscendC::BLOCK_CUBE)
        * static_cast<uint64_t>(tiling.tileN);
    uint64_t l1PingPong = 0;
    uint64_t l0PingPong = 0;

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);

    for (uint32_t tileIdx = curBlockIdx; tileIdx < totalTiles; tileIdx += blockNum) {
        uint32_t coreIdxM = tileIdx / divN;
        uint32_t coreIdxN = tileIdx % divN;
        if (coreIdxM % 2 == 1) { coreIdxN = divN - 1 - coreIdxN; }

        uint32_t mStart = coreIdxM * tiling.singleCoreM;
        uint32_t nStart = coreIdxN * tiling.singleCoreN;
        uint32_t mEnd = Min<uint32_t>(mStart + tiling.singleCoreM, tiling.m);
        uint32_t nEnd = Min<uint32_t>(nStart + tiling.singleCoreN, tiling.n);

        for (uint32_t mOff = mStart; mOff < mEnd; mOff += tiling.tileM) {
            uint32_t curTileM = Min<uint32_t>(tiling.tileM, mEnd - mOff);
            for (uint32_t nOff = nStart; nOff < nEnd; nOff += tiling.tileN) {
                uint32_t curTileN = Min<uint32_t>(tiling.tileN, nEnd - nOff);
                uint64_t mL0 = curTileM;
                uint64_t nL0 = curTileN;

                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);

                for (uint32_t kOff = 0; kOff < K; kOff += tiling.tileKChunk) {
                    uint32_t curK = Min<uint32_t>(tiling.tileKChunk, K - kOff);
                    uint64_t l1BufId = l1PingPong & 0x1;

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
                    ProcessKChunkDB<float>(aGlobal, bGlobal, l1Local, al0Local, bl0Local, l0cLocal,
                        tiling, mOff, nOff, kOff, curK, mL0, nL0,
                        l1BufStrideA, l1BufStrideB, l1BufId, l0PingPong);
                    l1PingPong++;
                }

                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);

                uint64_t offsetD = mOff * tiling.tempRowStride + nOff;
                SsymmCopyOut<float>(tempGlobal[offsetD], l0cLocal, mL0, nL0, tiling.tempRowStride);

                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
            }
        }
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
}
void ssymm_gemm_kernel_do(
    uint8_t* gmA, const uint8_t* gmB, uint8_t* gmTemp,
    const SsymmGemmTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    ssymm_gemm_kernel<<<numBlocks, nullptr, stream>>>(gmA, gmB, gmTemp, tiling);
}

// ================================================================================
// Phase 3: Scale Kernel (AIV-only, SIMT)
// C = alpha * temp + beta * C  (skipTemp == 0)
// C = beta * C                 (skipTemp == 1, alpha==0 fast path — temp not read)
// alpha/beta values are always carried in tiling (host resolves device scalars via
// ReadAlphaBetaFromDevice before launching the pipeline).
// temp stores C^T in row-major (n rows × tempRowStride, tempRowStride = CeilAlign(m)).
//   Reading temp[col*tempRowStride + row] gives C^T[col][row] = C[row][col].
// C is column-major (ldc is column stride, ldc >= m).
//   C[row][col] is at cGm[col*ldc + row].
// ================================================================================

__simt_callee__ __aicore__ inline void SsymmScaleOneElement(
    float alphaVal, float betaVal, uint32_t skipTemp,
    int64_t i64, int64_t j64, int64_t ldc64, int64_t tempStride64,
    __gm__ float* __restrict tempGm, __gm__ float* __restrict cGm)
{
    if (skipTemp != 0) {
        if (betaVal == 0.0f) {
            cGm[j64 * ldc64 + i64] = 0.0f;
        } else if (betaVal != 1.0f) {
            cGm[j64 * ldc64 + i64] = betaVal * cGm[j64 * ldc64 + i64];
        }
    } else {
        float tempVal = tempGm[j64 * tempStride64 + i64];
        if (betaVal == 0.0f) {
            cGm[j64 * ldc64 + i64] = alphaVal * tempVal;
        } else if (betaVal == 1.0f) {
            cGm[j64 * ldc64 + i64] = alphaVal * tempVal + cGm[j64 * ldc64 + i64];
        } else {
            cGm[j64 * ldc64 + i64] = alphaVal * tempVal + betaVal * cGm[j64 * ldc64 + i64];
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SsymmScaleCompute(
    uint32_t m, uint32_t n, uint32_t ldc, uint32_t tempRowStride,
    float alphaVal, float betaVal, uint32_t skipTemp,
    __gm__ float* __restrict tempGm, __gm__ float* __restrict cGm,
    uint32_t rowStart, uint32_t rowEnd)
{
    int64_t ldc64 = static_cast<int64_t>(ldc);
    int64_t tempStride64 = static_cast<int64_t>(tempRowStride);

    for (uint32_t i = rowStart + threadIdx.x; i < rowEnd; i += blockDim.x) {
        int64_t i64 = static_cast<int64_t>(i);
        for (uint32_t j = 0; j < n; ++j) {
            SsymmScaleOneElement(alphaVal, betaVal, skipTemp, i64,
                static_cast<int64_t>(j), ldc64, tempStride64, tempGm, cGm);
        }
    }
}

__global__ __aicore__ void ssymm_scale_kernel(
    __gm__ uint8_t* gmTemp, __gm__ uint8_t* gmC,
    const SsymmScaleTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* tempGm = reinterpret_cast<__gm__ float* __restrict>(gmTemp);
    auto* cGm = reinterpret_cast<__gm__ float* __restrict>(gmC);

    uint32_t m = tiling.m;
    uint32_t n = tiling.n;
    uint32_t ldc = tiling.ldc;
    uint32_t tempRowStride = tiling.tempRowStride;
    float alphaVal = tiling.alphaVal;
    float betaVal = tiling.betaVal;
    uint32_t skipTemp = tiling.skipTemp;

    int32_t blkIdx = AscendC::GetBlockIdx();
    uint32_t rowStart = static_cast<uint32_t>(blkIdx) * tiling.scaleRowsPerCore;
    uint32_t rowEnd = Min<uint32_t>(rowStart + tiling.scaleRowsPerCore, m);
    if (rowStart >= rowEnd) {
        return;
    }

    asc_vf_call<SsymmScaleCompute>(
        dim3{tiling.nthreads, 1, 1},
        m, n, ldc, tempRowStride, alphaVal, betaVal, skipTemp,
        tempGm, cGm, rowStart, rowEnd);
}

void ssymm_scale_kernel_do(
    uint8_t* gmTemp, uint8_t* gmC,
    const SsymmScaleTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    ssymm_scale_kernel<<<numBlocks, nullptr, stream>>>(gmTemp, gmC, tiling);
}
