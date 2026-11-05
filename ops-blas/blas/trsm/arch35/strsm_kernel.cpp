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
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "strsm_tiling_data.h"
#include "common/helper/kernel_constant.h"

using namespace AscendC;

static constexpr uint32_t STRSM_AUX_GRID_DIM = 128;

namespace trsm_common {

constexpr uint32_t MAX_BLOCK_NB = 128;
static_assert(MAX_BLOCK_NB >= 64, "MAX_BLOCK_NB must be >= 64");
constexpr uint32_t UB_A_BLOCK_FLOATS = MAX_BLOCK_NB * MAX_BLOCK_NB;
constexpr uint32_t SIMT_MAX_THREADS = 64;

template <typename GmT>
__simt_callee__ inline float LoadFromGm(__gm__ GmT* ptr, uint64_t idx)
{
    return static_cast<float>(ptr[idx]);
}

template <typename GmT>
__simt_callee__ inline void StoreToGm(__gm__ GmT* ptr, uint64_t idx, float val)
{
    ptr[idx] = static_cast<GmT>(val);
}

template <bool RIGHT>
__simt_callee__ inline uint64_t BOff(uint64_t freeIdx, uint64_t aIdx, uint64_t ldb)
{
    if constexpr (RIGHT) { return freeIdx + aIdx * ldb; }
    else { return freeIdx * ldb + aIdx; }
}

template <typename GmT, bool TRANS_IS_TRANS>
__simt_callee__ inline float LoadAFromUbOrGm(
    __gm__ GmT* aGm, __ubuf__ float* aBlockUb, uint32_t ldaU32, uint32_t row, uint32_t kk, uint32_t blockStart,
    uint32_t blockSize, bool rowInBlock)
{
    uint64_t ldaU64 = static_cast<uint64_t>(ldaU32);
    bool inUb = rowInBlock && kk >= blockStart && kk < blockStart + blockSize;
    if constexpr (TRANS_IS_TRANS) {
        return inUb ? aBlockUb[(row - blockStart) * blockSize + (kk - blockStart)] :
                      LoadFromGm<GmT>(aGm, static_cast<uint64_t>(row) * ldaU64 + kk);
    } else {
        return inUb ? aBlockUb[(kk - blockStart) * blockSize + (row - blockStart)] :
                      LoadFromGm<GmT>(aGm, static_cast<uint64_t>(kk) * ldaU64 + row);
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_TRANS>
__simt_callee__ inline void GetDotRange(uint32_t mEff, uint32_t row, uint32_t& dotStart, uint32_t& dotEnd)
{
    if constexpr (TRANS_IS_TRANS) {
        dotStart = UPLO_IS_UPPER ? 0 : row + 1;
        dotEnd = UPLO_IS_UPPER ? row : mEff;
    } else {
        dotStart = UPLO_IS_UPPER ? row + 1 : 0;
        dotEnd = UPLO_IS_UPPER ? mEff : row;
    }
}

template <typename GmT, bool TRANS_IS_TRANS, typename BType = GmT, bool RIGHT = false>
__simt_callee__ inline void ComputeDotProduct(
    __gm__ GmT* aGm, __ubuf__ float* aBlockUb, uint32_t ldaU32, uint32_t row, uint32_t colIdx, int32_t ldb,
    uint32_t dotStart, uint32_t dotEnd, uint32_t blockStart, uint32_t blockSize, bool rowInBlock, __gm__ BType* bGm,
    __ubuf__ float* partialSums, float& dotTotal)
{
    uint64_t ldbU64 = static_cast<uint64_t>(static_cast<uint32_t>(ldb));
    uint32_t dotLen = dotEnd - dotStart;
    if (dotLen <= blockDim.x) {
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (uint32_t kk = dotStart; kk < dotEnd; ++kk) {
                float aVal =
                    LoadAFromUbOrGm<GmT, TRANS_IS_TRANS>(aGm, aBlockUb, ldaU32, row, kk, blockStart, blockSize, rowInBlock);
                float bVal = LoadFromGm<BType>(bGm, BOff<RIGHT>(colIdx, kk, ldbU64));
                sum += aVal * bVal;
            }
            dotTotal = sum;
        }
    } else {
        float sum = 0.0f;
        for (uint32_t kk = dotStart + threadIdx.x; kk < dotEnd; kk += blockDim.x) {
            float aVal =
                LoadAFromUbOrGm<GmT, TRANS_IS_TRANS>(aGm, aBlockUb, ldaU32, row, kk, blockStart, blockSize, rowInBlock);
            float bVal = LoadFromGm<BType>(bGm, BOff<RIGHT>(colIdx, kk, ldbU64));
            sum += aVal * bVal;
        }
        partialSums[threadIdx.x] = sum;
        if (blockDim.x > 1) {
            asc_syncthreads();
            for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
                }
                asc_syncthreads();
            }
        }
        dotTotal = partialSums[0];
    }
    if (blockDim.x > 1) {
        asc_syncthreads();
    }
}

template <typename GmT, typename BType = GmT, bool RIGHT = false>
__simt_callee__ inline void ScaleB(
    uint32_t m, uint32_t colStart, uint32_t colEnd, int32_t ldb, float alpha, __gm__ BType* bGm)
{
    uint64_t ldbU64 = static_cast<uint64_t>(static_cast<uint32_t>(ldb));
    for (uint32_t col = colStart + threadIdx.x; col < colEnd; col += blockDim.x) {
        for (uint32_t row = 0; row < m; ++row) {
            uint64_t offset = BOff<RIGHT>(col, row, ldbU64);
            float cur = LoadFromGm<BType>(bGm, offset);
            StoreToGm<BType>(bGm, offset, alpha * cur);
        }
    }
    if (blockDim.x > 1) {
        asc_syncthreads();
    }
}

template <typename GmT, bool UPLO_IS_UPPER, bool DIAG_IS_UNIT>
__simt_callee__ inline void LoadABlock(
    uint32_t blockStart, uint32_t blockSize, uint32_t mEff, uint32_t ldaU32, __gm__ GmT* aGm, __ubuf__ float* aUb)
{
    uint64_t ldaU64 = static_cast<uint64_t>(ldaU32);
    for (uint32_t j = threadIdx.x; j < blockSize; j += blockDim.x) {
        uint32_t col = blockStart + j;
        if (col >= mEff) continue;
        uint32_t iStart = UPLO_IS_UPPER ? 0 : j;
        uint32_t iEnd = UPLO_IS_UPPER ? (j + 1) : blockSize;
        for (uint32_t i = iStart; i < iEnd; ++i) {
            uint32_t row = blockStart + i;
            if (row >= mEff) break;
            if constexpr (DIAG_IS_UNIT) {
                if (row == col) {
                    aUb[j * blockSize + i] = 1.0f;
                    continue;
                }
            }
            aUb[j * blockSize + i] = LoadFromGm<GmT>(aGm, static_cast<uint64_t>(col) * ldaU64 + row);
        }
    }
    if (blockDim.x > 1) {
        asc_syncthreads();
    }
}

template <typename GmT, bool DIAG_IS_UNIT, typename BType = GmT, bool RIGHT = false>
__simt_callee__ inline void ComputeAndStoreResult(
    __gm__ GmT* aGm, __ubuf__ float* aBlockUb, uint32_t ldaU32, uint32_t row, uint32_t colIdx, int32_t ldb,
    uint32_t blockStart, uint32_t blockSize, float dotTotal, __gm__ BType* bGm)
{
    uint64_t ldbU64 = static_cast<uint64_t>(static_cast<uint32_t>(ldb));
    uint64_t ldaU64 = static_cast<uint64_t>(ldaU32);
    float bRow = LoadFromGm<BType>(bGm, BOff<RIGHT>(colIdx, row, ldbU64));
    float bCurr = bRow - dotTotal;
    if constexpr (!DIAG_IS_UNIT) {
        float diagVal = (row >= blockStart && row < blockStart + blockSize) ?
                            aBlockUb[(row - blockStart) * blockSize + (row - blockStart)] :
                            LoadFromGm<GmT>(aGm, static_cast<uint64_t>(row) * ldaU64 + row);
        bCurr = bCurr / diagVal;
    }
    StoreToGm<BType>(bGm, BOff<RIGHT>(colIdx, row, ldbU64), bCurr);
}


template <typename GmT, bool UPLO_IS_UPPER, bool TRANS_IS_TRANS, bool DIAG_IS_UNIT, typename BType = GmT, bool RIGHT = false>
__simt_callee__ inline void SolveRow(
    uint32_t mEff, uint32_t row, uint32_t colIdx, int32_t lda, int32_t ldb, uint32_t blockStart, uint32_t blockSize,
    __gm__ GmT* aGm, __gm__ BType* bGm, __ubuf__ float* partialSums, __ubuf__ float* aBlockUb)
{
    uint32_t ldaU32 = static_cast<uint32_t>(lda);
    uint32_t dotStart, dotEnd;
    GetDotRange<UPLO_IS_UPPER, TRANS_IS_TRANS>(mEff, row, dotStart, dotEnd);
    bool rowInBlock = (row >= blockStart && row < blockStart + blockSize);
    float dotTotal = 0.0f;
    ComputeDotProduct<GmT, TRANS_IS_TRANS, BType, RIGHT>(
        aGm, aBlockUb, ldaU32, row, colIdx, ldb, dotStart, dotEnd, blockStart, blockSize, rowInBlock, bGm,
        partialSums, dotTotal);
    if (threadIdx.x == 0) {
        ComputeAndStoreResult<GmT, DIAG_IS_UNIT, BType, RIGHT>(
            aGm, aBlockUb, ldaU32, row, colIdx, ldb, blockStart, blockSize, dotTotal, bGm);
    }
    if (blockDim.x > 1) {
        asc_syncthreads();
    }
}

template <typename GmT, bool UPLO_IS_UPPER, bool TRANS_IS_TRANS, bool DIAG_IS_UNIT, typename BType = GmT, bool RIGHT = false>
__simt_callee__ inline void ProcessBlockStep(
    uint32_t mEff, uint32_t colIdx, uint32_t k, int32_t lda, int32_t ldb, uint32_t ldaU32, __gm__ GmT* aGm,
    __gm__ BType* bGm, __ubuf__ float* partialSums, __ubuf__ float* aBlockUb)
{
    uint32_t blockStart = k;
    uint32_t blockSize = (k + MAX_BLOCK_NB <= mEff) ? MAX_BLOCK_NB : (mEff - k);
    if (blockStart < mEff && blockSize > 0) {
        LoadABlock<GmT, UPLO_IS_UPPER, DIAG_IS_UNIT>(blockStart, blockSize, mEff, ldaU32, aGm, aBlockUb);
    }
    for (uint32_t ii = 0; ii < blockSize; ++ii) {
        uint32_t row;
        if constexpr (TRANS_IS_TRANS) {
            row = UPLO_IS_UPPER ? (blockStart + ii) : (blockStart + blockSize - 1 - ii);
        } else {
            row = UPLO_IS_UPPER ? (blockStart + blockSize - 1 - ii) : (blockStart + ii);
        }
        SolveRow<GmT, UPLO_IS_UPPER, TRANS_IS_TRANS, DIAG_IS_UNIT, BType, RIGHT>(
            mEff, row, colIdx, lda, ldb, blockStart, blockSize, aGm, bGm, partialSums, aBlockUb);
    }
}

template <typename GmT, bool UPLO_IS_UPPER, bool TRANS_IS_TRANS, bool DIAG_IS_UNIT, typename BType = GmT, bool RIGHT = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREADS) inline void SimtCompute(
    uint32_t mEff, uint32_t colStart, uint32_t colEnd, int32_t lda, int32_t ldb, float alpha, __gm__ GmT* aGm,
    __gm__ BType* bGm)
{
    __ubuf__ float partialSums[SIMT_MAX_THREADS];
    __ubuf__ float aBlockUb[UB_A_BLOCK_FLOATS];
    ScaleB<GmT, BType, RIGHT>(mEff, colStart, colEnd, ldb, alpha, bGm);
    constexpr bool kForward = (!UPLO_IS_UPPER && !TRANS_IS_TRANS) || (UPLO_IS_UPPER && TRANS_IS_TRANS);
    uint32_t ldaU32 = static_cast<uint32_t>(lda);
    uint32_t numBlks = (mEff + MAX_BLOCK_NB - 1) / MAX_BLOCK_NB;
    for (uint32_t colIdx = colStart; colIdx < colEnd; ++colIdx) {
        for (uint32_t bi = 0; bi < numBlks; ++bi) {
            uint32_t blk = kForward ? bi : (numBlks - 1 - bi);
            ProcessBlockStep<GmT, UPLO_IS_UPPER, TRANS_IS_TRANS, DIAG_IS_UNIT, BType, RIGHT>(
                mEff, colIdx, blk * MAX_BLOCK_NB, lda, ldb, ldaU32, aGm, bGm, partialSums, aBlockUb);
        }
    }
}

template <typename GmT, bool UPLO_IS_UPPER, bool TRANS_IS_TRANS, bool DIAG_IS_UNIT, typename BType = GmT, bool RIGHT = false>
__aicore__ inline void DispatchDiag(
    uint32_t mEff, uint32_t colStart, uint32_t colEnd, int32_t lda, int32_t ldb, float alpha, __gm__ GmT* aGm,
    __gm__ BType* bGm, uint32_t numThreads)
{
    asc_vf_call<SimtCompute<GmT, UPLO_IS_UPPER, TRANS_IS_TRANS, DIAG_IS_UNIT, BType, RIGHT>>(
        dim3{numThreads, 1, 1}, mEff, colStart, colEnd, lda, ldb, alpha, aGm, bGm);
}

template <typename GmT, bool UPLO_IS_UPPER, bool TRANS_IS_TRANS, typename BType = GmT, bool RIGHT = false>
__aicore__ inline void DispatchUplo(
    uint32_t diag, uint32_t mEff, uint32_t colStart, uint32_t colEnd, int32_t lda, int32_t ldb, float alpha,
    __gm__ GmT* aGm, __gm__ BType* bGm, uint32_t numThreads)
{
    if (diag == ACLBLAS_UNIT) {
        DispatchDiag<GmT, UPLO_IS_UPPER, TRANS_IS_TRANS, true, BType, RIGHT>(
            mEff, colStart, colEnd, lda, ldb, alpha, aGm, bGm, numThreads);
    } else {
        DispatchDiag<GmT, UPLO_IS_UPPER, TRANS_IS_TRANS, false, BType, RIGHT>(
            mEff, colStart, colEnd, lda, ldb, alpha, aGm, bGm, numThreads);
    }
}

template <typename GmT, typename BType = GmT, bool RIGHT = false>
__aicore__ inline void DispatchKernel(const StrsmTilingData& tiling, __gm__ GmT* aGm, __gm__ BType* bGm)
{
    uint32_t mEff = tiling.m;
    uint32_t nEff = tiling.n;
    uint32_t coreId = GetBlockIdx();
    uint32_t colStart = coreId * tiling.perCoreN + (coreId < tiling.coreRemainder ? coreId : tiling.coreRemainder);
    uint32_t colEnd = colStart + tiling.perCoreN + (coreId < tiling.coreRemainder ? 1 : 0);
    if (colStart >= nEff) {
        return;
    }
    if (colEnd > nEff) {
        colEnd = nEff;
    }
    bool isUpper = (tiling.uplo == ACLBLAS_UPPER);
    bool isTrans = (tiling.trans != ACLBLAS_OP_N);
    if (isUpper && isTrans) {
        DispatchUplo<GmT, true, true, BType, RIGHT>(
            tiling.diag, mEff, colStart, colEnd, tiling.lda, tiling.ldb, tiling.alpha, aGm, bGm, tiling.numThreads);
    } else if (isUpper && !isTrans) {
        DispatchUplo<GmT, true, false, BType, RIGHT>(
            tiling.diag, mEff, colStart, colEnd, tiling.lda, tiling.ldb, tiling.alpha, aGm, bGm, tiling.numThreads);
    } else if (!isUpper && isTrans) {
        DispatchUplo<GmT, false, true, BType, RIGHT>(
            tiling.diag, mEff, colStart, colEnd, tiling.lda, tiling.ldb, tiling.alpha, aGm, bGm, tiling.numThreads);
    } else {
        DispatchUplo<GmT, false, false, BType, RIGHT>(
            tiling.diag, mEff, colStart, colEnd, tiling.lda, tiling.ldb, tiling.alpha, aGm, bGm, tiling.numThreads);
    }
}

template <typename GmT>
__simt_vf__ __aicore__
LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void ZeroFill(
    uint32_t m, uint32_t colStart, uint32_t colEnd, uint32_t ldbU32, __gm__ GmT* bGm)
{
    uint64_t ldbU64 = static_cast<uint64_t>(ldbU32);
    for (uint32_t col = colStart + threadIdx.x; col < colEnd; col += blockDim.x) {
        for (uint32_t row = 0; row < m; ++row) {
            StoreToGm<GmT>(bGm, static_cast<uint64_t>(col) * ldbU64 + row, 0.0f);
        }
    }
}

template <typename GmT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void Transpose(
    uint32_t rowStart, uint32_t rowEnd, uint32_t cols, int32_t ldIn, int32_t ldOut,
    __gm__ const GmT* inGm, __gm__ GmT* outGm)
{
    uint32_t ldInU32 = static_cast<uint32_t>(ldIn);
    uint32_t ldOutU32 = static_cast<uint32_t>(ldOut);
    uint64_t ldInU64 = static_cast<uint64_t>(ldInU32);
    uint64_t ldOutU64 = static_cast<uint64_t>(ldOutU32);
    for (uint32_t i = rowStart + threadIdx.x; i < rowEnd; i += blockDim.x) {
        for (uint32_t j = 0; j < cols; ++j) {
            outGm[static_cast<uint64_t>(i) * ldOutU64 + j] = inGm[static_cast<uint64_t>(j) * ldInU64 + i];
        }
    }
}

template <typename GmT, bool UPLO_IS_UPPER, bool TRANS_IS_TRANS, bool DIAG_IS_UNIT, typename BType = GmT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void PanelSolve(
    uint32_t mEff, uint32_t colStart, uint32_t colEnd, uint32_t panelStart, uint32_t panelSize,
    int32_t lda, int32_t ldb, __gm__ GmT* aGm, __gm__ BType* bGm)
{
    __ubuf__ float aBlockUb[UB_A_BLOCK_FLOATS];
    uint32_t panelEnd = panelStart + panelSize;
    uint32_t ldaU32 = static_cast<uint32_t>(lda);
    uint64_t ldbU64 = static_cast<uint64_t>(static_cast<uint32_t>(ldb));

    LoadABlock<GmT, UPLO_IS_UPPER, DIAG_IS_UNIT>(panelStart, panelSize, mEff, ldaU32, aGm, aBlockUb);

    for (uint32_t colIdx = colStart + threadIdx.x; colIdx < colEnd; colIdx += blockDim.x) {
        for (uint32_t ii = 0; ii < panelSize; ++ii) {
            uint32_t row;
            if constexpr (TRANS_IS_TRANS) {
                row = UPLO_IS_UPPER ? (panelStart + ii) : (panelStart + panelSize - 1 - ii);
            } else {
                row = UPLO_IS_UPPER ? (panelStart + panelSize - 1 - ii) : (panelStart + ii);
            }

            uint32_t dotStart, dotEnd;
            if constexpr (!TRANS_IS_TRANS) {
                if constexpr (UPLO_IS_UPPER) {
                    dotStart = row + 1;
                    dotEnd = panelEnd;
                } else {
                    dotStart = panelStart;
                    dotEnd = row;
                }
            } else {
                if constexpr (UPLO_IS_UPPER) {
                    dotStart = panelStart;
                    dotEnd = row;
                } else {
                    dotStart = row + 1;
                    dotEnd = panelEnd;
                }
            }

            float dotTotal = 0.0f;
            for (uint32_t kk = dotStart; kk < dotEnd; ++kk) {
                float aVal = LoadAFromUbOrGm<GmT, TRANS_IS_TRANS>(
                    aGm, aBlockUb, ldaU32, row, kk, panelStart, panelSize, true);
                float bVal = LoadFromGm<BType>(bGm, static_cast<uint64_t>(colIdx) * ldbU64 + kk);
                dotTotal += aVal * bVal;
            }

            float bCurr = LoadFromGm<BType>(bGm, static_cast<uint64_t>(colIdx) * ldbU64 + row) - dotTotal;
            if constexpr (!DIAG_IS_UNIT) {
                float diagVal = aBlockUb[(row - panelStart) * panelSize + (row - panelStart)];
                bCurr = bCurr / diagVal;
            }
            StoreToGm<BType>(bGm, static_cast<uint64_t>(colIdx) * ldbU64 + row, bCurr);
        }
    }
}

} // namespace trsm_common

__global__ __aicore__ void strsm_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR workSpace, const StrsmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    auto* aGm = reinterpret_cast<__gm__ float*>(a);
    auto* bGm = reinterpret_cast<__gm__ float*>(b);
    trsm_common::DispatchKernel<float>(tiling, aGm, bGm);
}

__global__ __aicore__ void strsm_right_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR workSpace, const StrsmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    auto* aGm = reinterpret_cast<__gm__ float*>(a);
    auto* bGm = reinterpret_cast<__gm__ float*>(b);
    trsm_common::DispatchKernel<float, float, true>(tiling, aGm, bGm);
}

__global__ __aicore__ void strsm_zero_kernel(GM_ADDR b, uint32_t m, uint32_t n, int32_t ldb)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t colsPerBlock = (n + GetBlockNum() - 1) / GetBlockNum();
    uint32_t colStart = GetBlockIdx() * colsPerBlock;
    if (colStart >= n) return;
    uint32_t colEnd = colStart + colsPerBlock;
    if (colEnd > n) colEnd = n;
    dim3 grid = {STRSM_AUX_GRID_DIM, 1, 1};
    asc_vf_call<trsm_common::ZeroFill<float>>(
        grid, m, colStart, colEnd, static_cast<uint32_t>(ldb), reinterpret_cast<__gm__ float*>(b));
}

__global__ __aicore__ void strsm_transpose_kernel(
    GM_ADDR in, GM_ADDR out, uint32_t rows, uint32_t cols, int32_t ldIn, int32_t ldOut)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t rowsPerBlock = (rows + GetBlockNum() - 1) / GetBlockNum();
    uint32_t rowStart = GetBlockIdx() * rowsPerBlock;
    if (rowStart >= rows) return;
    uint32_t rowEnd = rowStart + rowsPerBlock;
    if (rowEnd > rows) rowEnd = rows;
    dim3 grid = {STRSM_AUX_GRID_DIM, 1, 1};
    asc_vf_call<trsm_common::Transpose<float>>(
        grid, rowStart, rowEnd, cols, ldIn, ldOut,
        reinterpret_cast<__gm__ const float*>(in), reinterpret_cast<__gm__ float*>(out));
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_TRANS>
__aicore__ inline void PanelSolveDispatch(
    uint32_t diag, uint32_t m, uint32_t colStart, uint32_t colEnd, uint32_t panelStart, uint32_t panelSize,
    int32_t lda, int32_t ldb, __gm__ float* aGm, __gm__ float* bGm)
{
    dim3 grid = {64, 1, 1};
    if (diag == ACLBLAS_UNIT) {
        asc_vf_call<trsm_common::PanelSolve<float, UPLO_IS_UPPER, TRANS_IS_TRANS, true>>(
            grid, m, colStart, colEnd, panelStart, panelSize, lda, ldb, aGm, bGm);
    } else {
        asc_vf_call<trsm_common::PanelSolve<float, UPLO_IS_UPPER, TRANS_IS_TRANS, false>>(
            grid, m, colStart, colEnd, panelStart, panelSize, lda, ldb, aGm, bGm);
    }
}

__global__ __aicore__ void strsm_panel_kernel(GM_ADDR a, GM_ADDR b, const StrsmPanelTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* aGm = reinterpret_cast<__gm__ float*>(a);
    auto* bGm = reinterpret_cast<__gm__ float*>(b);

    uint32_t n = tiling.n;
    uint32_t colsPerCore = (n + GetBlockNum() - 1) / GetBlockNum();
    uint32_t colStart = GetBlockIdx() * colsPerCore;
    if (colStart >= n) return;
    uint32_t colEnd = colStart + colsPerCore;
    if (colEnd > n) colEnd = n;

    bool isUpper = (tiling.uplo == ACLBLAS_UPPER);
    bool isTrans = (tiling.trans != ACLBLAS_OP_N);

    if (isUpper && isTrans) {
        PanelSolveDispatch<true, true>(tiling.diag, tiling.m, colStart, colEnd, tiling.panelStart, tiling.panelSize,
            tiling.lda, tiling.ldb, aGm, bGm);
    } else if (isUpper && !isTrans) {
        PanelSolveDispatch<true, false>(tiling.diag, tiling.m, colStart, colEnd, tiling.panelStart, tiling.panelSize,
            tiling.lda, tiling.ldb, aGm, bGm);
    } else if (!isUpper && isTrans) {
        PanelSolveDispatch<false, true>(tiling.diag, tiling.m, colStart, colEnd, tiling.panelStart, tiling.panelSize,
            tiling.lda, tiling.ldb, aGm, bGm);
    } else {
        PanelSolveDispatch<false, false>(tiling.diag, tiling.m, colStart, colEnd, tiling.panelStart, tiling.panelSize,
            tiling.lda, tiling.ldb, aGm, bGm);
    }
}

void strsm_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* workSpace, const StrsmTilingData& tiling, uint32_t numBlocks, void* stream)
{
    strsm_kernel<<<numBlocks, nullptr, stream>>>(a, b, workSpace, tiling);
}

void strsm_right_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* workSpace, const StrsmTilingData& tiling, uint32_t numBlocks, void* stream)
{
    strsm_right_kernel<<<numBlocks, nullptr, stream>>>(a, b, workSpace, tiling);
}

void strsm_zero_kernel_do(uint8_t* b, uint32_t m, uint32_t n, int32_t ldb, uint32_t numBlocks, void* stream)
{
    strsm_zero_kernel<<<numBlocks, nullptr, stream>>>(b, m, n, ldb);
}

void strsm_transpose_kernel_do(
    uint8_t* in, uint8_t* out, uint32_t rows, uint32_t cols, int32_t ldIn, int32_t ldOut,
    uint32_t numBlocks, void* stream)
{
    strsm_transpose_kernel<<<numBlocks, nullptr, stream>>>(in, out, rows, cols, ldIn, ldOut);
}

void strsm_panel_kernel_do(uint8_t* a, uint8_t* b, const StrsmPanelTilingData& tiling, uint32_t numBlocks, void* stream)
{
    strsm_panel_kernel<<<numBlocks, nullptr, stream>>>(a, b, tiling);
}
