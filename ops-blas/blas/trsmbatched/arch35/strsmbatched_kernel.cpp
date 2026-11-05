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
#include "strsmbatched_tiling_data.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"

using namespace AscendC::Te;

// ===================== AIV kernels: scale / zero / panel / axpy / extract / transpose =====================

// VEC_THREAD_NUM=128 for element-wise auxiliary kernels (scale/zero/extract/axpy/transpose).
// Panel solve uses SIMT_MAX_THREADS=64 (in trsmbatched_panel namespace) because it shares
// UB-resident A block data across threads, requiring fewer threads for UB capacity.
constexpr uint32_t VEC_THREAD_NUM = 128;

// ---------- Scale kernel ----------

__simt_vf__ __aicore__ LAUNCH_BOUND(VEC_THREAD_NUM) inline void TrsmbatchedScaleSimt(
    uint32_t m, uint32_t colStart, uint32_t colEnd, uint32_t ldb, float alpha, __gm__ float* bGm)
{
    uint64_t ldbU64 = static_cast<uint64_t>(ldb);
    for (uint32_t col = colStart + threadIdx.x; col < colEnd; col += blockDim.x) {
        for (uint32_t row = 0; row < m; ++row) {
            uint64_t offset = static_cast<uint64_t>(col) * ldbU64 + row;
            bGm[offset] = alpha * bGm[offset];
        }
    }
}

extern "C" __global__ __aicore__ void trsmbatched_scale_kernel(GM_ADDR b, float alpha, uint32_t m, uint32_t n, int32_t ldb)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t totalCols = n;
    uint32_t colsPerBlock = (totalCols + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
    uint32_t colStart = AscendC::GetBlockIdx() * colsPerBlock;
    if (colStart >= totalCols) return;
    uint32_t colEnd = colStart + colsPerBlock;
    if (colEnd > totalCols) colEnd = totalCols;
    asc_vf_call<TrsmbatchedScaleSimt>(
        dim3{VEC_THREAD_NUM, 1, 1}, m, colStart, colEnd, static_cast<uint32_t>(ldb), alpha, reinterpret_cast<__gm__ float*>(b));
}

void trsmbatched_scale_kernel_do(
    GM_ADDR b, float alpha, uint32_t m, uint32_t n, int32_t ldb, uint32_t numBlocks, void* stream)
{
    trsmbatched_scale_kernel<<<numBlocks, nullptr, stream>>>(b, alpha, m, n, ldb);
}

// ---------- Zero kernel ----------

__simt_vf__ __aicore__ LAUNCH_BOUND(VEC_THREAD_NUM) inline void TrsmbatchedZeroSimt(
    uint32_t m, uint32_t colStart, uint32_t colEnd, uint32_t ldb, __gm__ float* bGm)
{
    uint64_t ldbU64 = static_cast<uint64_t>(ldb);
    for (uint32_t col = colStart + threadIdx.x; col < colEnd; col += blockDim.x) {
        for (uint32_t row = 0; row < m; ++row) {
            bGm[static_cast<uint64_t>(col) * ldbU64 + row] = 0.0f;
        }
    }
}

extern "C" __global__ __aicore__ void trsmbatched_zero_kernel(GM_ADDR b, uint32_t m, uint32_t n, int32_t ldb)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t nU32 = static_cast<uint32_t>(n);
    uint32_t colsPerBlock = (nU32 + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
    uint32_t colStart = AscendC::GetBlockIdx() * colsPerBlock;
    if (colStart >= nU32) return;
    uint32_t colEnd = colStart + colsPerBlock;
    if (colEnd > nU32) colEnd = nU32;
    asc_vf_call<TrsmbatchedZeroSimt>(
        dim3{VEC_THREAD_NUM, 1, 1}, m, colStart, colEnd, static_cast<uint32_t>(ldb), reinterpret_cast<__gm__ float*>(b));
}

void trsmbatched_zero_kernel_do(
    GM_ADDR b, uint32_t m, uint32_t n, int32_t ldb, uint32_t numBlocks, void* stream)
{
    trsmbatched_zero_kernel<<<numBlocks, nullptr, stream>>>(b, m, n, ldb);
}

// ---------- Panel kernel (SIMT triangular solve) ----------

namespace trsmbatched_panel {

constexpr uint32_t MAX_BLOCK_NB = 128;
static_assert(MAX_BLOCK_NB >= 64, "MAX_BLOCK_NB must be >= 64");
constexpr uint32_t UB_A_BLOCK_FLOATS = MAX_BLOCK_NB * MAX_BLOCK_NB;
// Panel solve uses 64 threads (vs 128 for element-wise kernels) because all threads
// share the UB-resident A block, and fewer threads allow larger per-thread UB footprint.
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
    asc_syncthreads();
}

template <typename GmT, bool TRANS_IS_TRANS>
__simt_callee__ inline float LoadAFromUbOrGm(
    __gm__ GmT* aGm, __ubuf__ float* aBlockUb, uint32_t ldaU32, uint32_t row, uint32_t dotIdx,
    uint32_t blockStart, uint32_t blockSize, bool rowInBlock)
{
    uint64_t ldaU64 = static_cast<uint64_t>(ldaU32);
    bool inUb = rowInBlock && dotIdx >= blockStart && dotIdx < blockStart + blockSize;
    if constexpr (TRANS_IS_TRANS) {
        return inUb ? aBlockUb[(row - blockStart) * blockSize + (dotIdx - blockStart)]
                     : LoadFromGm<GmT>(aGm, static_cast<uint64_t>(row) * ldaU64 + dotIdx);
    } else {
        return inUb ? aBlockUb[(dotIdx - blockStart) * blockSize + (row - blockStart)]
                     : LoadFromGm<GmT>(aGm, static_cast<uint64_t>(dotIdx) * ldaU64 + row);
    }
}

template <typename GmT, bool UPLO_IS_UPPER, bool TRANS_IS_TRANS, bool DIAG_IS_UNIT, typename BType = GmT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREADS) inline void PanelSolve(
    uint32_t mEff, uint32_t colStart, uint32_t colEnd, uint32_t panelStart, uint32_t panelSize,
    int32_t lda, int32_t ldb, __gm__ GmT* aGm, __gm__ BType* bGm)
{
    __ubuf__ float aBlockUb[UB_A_BLOCK_FLOATS];
    uint32_t panelEnd = panelStart + panelSize;
    uint32_t ldaU32 = static_cast<uint32_t>(lda);
    uint64_t ldbU64 = static_cast<uint64_t>(static_cast<uint32_t>(ldb));

    LoadABlock<GmT, UPLO_IS_UPPER, DIAG_IS_UNIT>(panelStart, panelSize, mEff, ldaU32, aGm, aBlockUb);

    for (uint32_t colIdx = colStart + threadIdx.x; colIdx < colEnd; colIdx += blockDim.x) {
        for (uint32_t rowIdx = 0; rowIdx < panelSize; ++rowIdx) {
            uint32_t row;
            if constexpr (TRANS_IS_TRANS) {
                row = UPLO_IS_UPPER ? (panelStart + rowIdx) : (panelStart + panelSize - 1 - rowIdx);
            } else {
                row = UPLO_IS_UPPER ? (panelStart + panelSize - 1 - rowIdx) : (panelStart + rowIdx);
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
            for (uint32_t dotIdx = dotStart; dotIdx < dotEnd; ++dotIdx) {
                float aVal = LoadAFromUbOrGm<GmT, TRANS_IS_TRANS>(
                    aGm, aBlockUb, ldaU32, row, dotIdx, panelStart, panelSize, true);
                float bVal = LoadFromGm<BType>(bGm, static_cast<uint64_t>(colIdx) * ldbU64 + dotIdx);
                dotTotal += aVal * bVal;
            }

            float bCurr = LoadFromGm<BType>(bGm, static_cast<uint64_t>(colIdx) * ldbU64 + row) - dotTotal;
            if constexpr (!DIAG_IS_UNIT) {
                float diagVal = aBlockUb[(row - panelStart) * panelSize + (row - panelStart)];
                // BLAS assumes A is non-singular (diagonal elements are non-zero).
                bCurr = bCurr / diagVal;
            }
            StoreToGm<BType>(bGm, static_cast<uint64_t>(colIdx) * ldbU64 + row, bCurr);
        }
    }
}

} // namespace trsmbatched_panel

template <bool UPLO_IS_UPPER, bool TRANS_IS_TRANS>
__aicore__ inline void TrsmbatchedPanelSolveDispatch(
    uint32_t diag, uint32_t m, uint32_t colStart, uint32_t colEnd, uint32_t panelStart, uint32_t panelSize,
    int32_t lda, int32_t ldb, __gm__ float* aGm, __gm__ float* bGm)
{
    dim3 grid = {trsmbatched_panel::SIMT_MAX_THREADS, 1, 1};
    if (diag == ACLBLAS_UNIT) {
        asc_vf_call<trsmbatched_panel::PanelSolve<float, UPLO_IS_UPPER, TRANS_IS_TRANS, true>>(
            grid, m, colStart, colEnd, panelStart, panelSize, lda, ldb, aGm, bGm);
    } else {
        asc_vf_call<trsmbatched_panel::PanelSolve<float, UPLO_IS_UPPER, TRANS_IS_TRANS, false>>(
            grid, m, colStart, colEnd, panelStart, panelSize, lda, ldb, aGm, bGm);
    }
}

extern "C" __global__ __aicore__ void trsmbatched_panel_kernel(GM_ADDR a, GM_ADDR b, const TrsmbatchedPanelTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* aGm = reinterpret_cast<__gm__ float*>(a);
    auto* bGm = reinterpret_cast<__gm__ float*>(b);

    uint32_t n = tiling.n;
    uint32_t colsPerCore = (n + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
    uint32_t colStart = AscendC::GetBlockIdx() * colsPerCore;
    if (colStart >= n) return;
    uint32_t colEnd = colStart + colsPerCore;
    if (colEnd > n) colEnd = n;

    bool isUpper = (tiling.uplo == ACLBLAS_UPPER);
    bool isTrans = (tiling.trans != ACLBLAS_OP_N);

    if (isUpper && isTrans) {
        TrsmbatchedPanelSolveDispatch<true, true>(tiling.diag, tiling.m, colStart, colEnd,
            tiling.panelStart, tiling.panelSize, tiling.lda, tiling.ldb, aGm, bGm);
    } else if (isUpper && !isTrans) {
        TrsmbatchedPanelSolveDispatch<true, false>(tiling.diag, tiling.m, colStart, colEnd,
            tiling.panelStart, tiling.panelSize, tiling.lda, tiling.ldb, aGm, bGm);
    } else if (!isUpper && isTrans) {
        TrsmbatchedPanelSolveDispatch<false, true>(tiling.diag, tiling.m, colStart, colEnd,
            tiling.panelStart, tiling.panelSize, tiling.lda, tiling.ldb, aGm, bGm);
    } else {
        TrsmbatchedPanelSolveDispatch<false, false>(tiling.diag, tiling.m, colStart, colEnd,
            tiling.panelStart, tiling.panelSize, tiling.lda, tiling.ldb, aGm, bGm);
    }
}

void trsmbatched_panel_kernel_do(
    GM_ADDR a, GM_ADDR b, const TrsmbatchedPanelTilingData& tiling, uint32_t numBlocks, void* stream)
{
    trsmbatched_panel_kernel<<<numBlocks, nullptr, stream>>>(a, b, tiling);
}

// ---------- Extract A kernel ----------

__simt_vf__ __aicore__ LAUNCH_BOUND(VEC_THREAD_NUM) inline void TrsmbatchedExtractASimt(
    uint32_t mC, uint32_t bs, uint32_t aWsStride, uint32_t lda, uint64_t aOffset,
    uint64_t startIdx, uint64_t calNum, __gm__ const float* aGm, __gm__ float* wsGm)
{
    if (calNum == 0) return;
    for (uint64_t idx = threadIdx.x; idx < calNum; idx += blockDim.x) {
        uint64_t linear = startIdx + idx;
        uint32_t i = static_cast<uint32_t>(linear / bs);
        uint32_t l = static_cast<uint32_t>(linear % bs);
        uint64_t srcIdx = aOffset + l + static_cast<uint64_t>(i) * lda;
        wsGm[static_cast<uint64_t>(i) * aWsStride + l] = aGm[srcIdx];
    }
}

extern "C" __global__ __aicore__ void trsmbatched_extract_a_kernel(
    GM_ADDR a, GM_ADDR ws, uint32_t mC, uint32_t bs, uint32_t aWsStride, uint32_t lda,
    uint64_t aOffset)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint64_t totalElements = static_cast<uint64_t>(mC) * bs;
    uint64_t elementsPerBlock = (totalElements + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
    uint64_t startIdx = static_cast<uint64_t>(AscendC::GetBlockIdx()) * elementsPerBlock;
    if (startIdx >= totalElements) return;
    uint64_t endIdx = startIdx + elementsPerBlock;
    if (endIdx > totalElements) endIdx = totalElements;
    uint64_t calNum = endIdx - startIdx;
    asc_vf_call<TrsmbatchedExtractASimt>(
        dim3{VEC_THREAD_NUM, 1, 1}, mC, bs, aWsStride, lda, aOffset, startIdx,
        calNum,
        reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ float*>(ws));
}

void trsmbatched_extract_a_kernel_do(
    GM_ADDR a, GM_ADDR ws, uint32_t mC, uint32_t bs, uint32_t aWsStride, uint32_t lda,
    uint64_t aOffset, uint32_t numBlocks, void* stream)
{
    trsmbatched_extract_a_kernel<<<numBlocks, nullptr, stream>>>(a, ws, mC, bs, aWsStride, lda, aOffset);
}

// ---------- Extract B kernel ----------

__simt_vf__ __aicore__ LAUNCH_BOUND(VEC_THREAD_NUM) inline void TrsmbatchedExtractBSimt(
    uint32_t bs, uint32_t n, uint32_t bWsStride, uint32_t ldb, uint64_t bOffset,
    uint64_t startIdx, uint64_t calNum, __gm__ const float* bGm, __gm__ float* wsGm)
{
    if (calNum == 0) return;
    for (uint64_t idx = threadIdx.x; idx < calNum; idx += blockDim.x) {
        uint64_t linear = startIdx + idx;
        uint32_t l = static_cast<uint32_t>(linear / n);
        uint32_t j = static_cast<uint32_t>(linear % n);
        wsGm[static_cast<uint64_t>(l) * bWsStride + j] = bGm[bOffset + l + static_cast<uint64_t>(j) * ldb];
    }
}

extern "C" __global__ __aicore__ void trsmbatched_extract_b_kernel(
    GM_ADDR b, GM_ADDR ws, uint32_t bs, uint32_t n, uint32_t bWsStride, uint32_t ldb, uint64_t bOffset)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint64_t totalElements = static_cast<uint64_t>(bs) * n;
    uint64_t elementsPerBlock = (totalElements + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
    uint64_t startIdx = static_cast<uint64_t>(AscendC::GetBlockIdx()) * elementsPerBlock;
    if (startIdx >= totalElements) return;
    uint64_t endIdx = startIdx + elementsPerBlock;
    if (endIdx > totalElements) endIdx = totalElements;
    uint64_t calNum = endIdx - startIdx;
    asc_vf_call<TrsmbatchedExtractBSimt>(
        dim3{VEC_THREAD_NUM, 1, 1}, bs, n, bWsStride, ldb, bOffset, startIdx,
        calNum,
        reinterpret_cast<__gm__ const float*>(b), reinterpret_cast<__gm__ float*>(ws));
}

void trsmbatched_extract_b_kernel_do(
    GM_ADDR b, GM_ADDR ws, uint32_t bs, uint32_t n, uint32_t bWsStride, uint32_t ldb,
    uint64_t bOffset, uint32_t numBlocks, void* stream)
{
    trsmbatched_extract_b_kernel<<<numBlocks, nullptr, stream>>>(b, ws, bs, n, bWsStride, ldb, bOffset);
}

// ---------- AXPY kernel (no trans) ----------

template <bool TRANS_TEMP>
__simt_vf__ __aicore__ LAUNCH_BOUND(VEC_THREAD_NUM) inline void TrsmbatchedAxpySimt(
    uint32_t m, uint32_t n, uint32_t ldb, uint32_t tempRowStride, uint64_t bOffset,
    uint64_t startIdx, uint64_t calNum, __gm__ float* bGm, __gm__ const float* tempGm)
{
    if (calNum == 0) return;
    for (uint64_t idx = threadIdx.x; idx < calNum; idx += blockDim.x) {
        uint64_t linear = startIdx + idx;
        uint32_t row = static_cast<uint32_t>(linear / n);
        uint32_t col = static_cast<uint32_t>(linear % n);
        uint64_t tempOff = TRANS_TEMP
            ? (static_cast<uint64_t>(col) * tempRowStride + row)
            : (static_cast<uint64_t>(row) * tempRowStride + col);
        bGm[bOffset + row + static_cast<uint64_t>(col) * ldb] -= tempGm[tempOff];
    }
}

extern "C" __global__ __aicore__ void trsmbatched_axpy_kernel(GM_ADDR b, GM_ADDR temp, const TrsmbatchedAxpyTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint64_t totalElements = static_cast<uint64_t>(tiling.m) * tiling.n;
    uint64_t elementsPerBlock = (totalElements + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
    uint64_t startIdx = static_cast<uint64_t>(AscendC::GetBlockIdx()) * elementsPerBlock;
    if (startIdx >= totalElements) return;
    uint64_t endIdx = startIdx + elementsPerBlock;
    if (endIdx > totalElements) endIdx = totalElements;
    uint64_t calNum = endIdx - startIdx;
    asc_vf_call<TrsmbatchedAxpySimt<false>>(
        dim3{VEC_THREAD_NUM, 1, 1}, tiling.m, tiling.n, tiling.ldb, tiling.tempRowStride, tiling.bOffset,
        startIdx, calNum, reinterpret_cast<__gm__ float*>(b), reinterpret_cast<__gm__ const float*>(temp));
}

void trsmbatched_axpy_kernel_do(
    GM_ADDR b, GM_ADDR temp, const TrsmbatchedAxpyTilingData& tiling, uint32_t numBlocks, void* stream)
{
    trsmbatched_axpy_kernel<<<numBlocks, nullptr, stream>>>(b, temp, tiling);
}

// ---------- AXPY trans kernel ----------

extern "C" __global__ __aicore__ void trsmbatched_axpy_trans_kernel(
    GM_ADDR b, GM_ADDR temp, const TrsmbatchedAxpyTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint64_t totalElements = static_cast<uint64_t>(tiling.m) * tiling.n;
    uint64_t elementsPerBlock = (totalElements + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
    uint64_t startIdx = static_cast<uint64_t>(AscendC::GetBlockIdx()) * elementsPerBlock;
    if (startIdx >= totalElements) return;
    uint64_t endIdx = startIdx + elementsPerBlock;
    if (endIdx > totalElements) endIdx = totalElements;
    uint64_t calNum = endIdx - startIdx;
    asc_vf_call<TrsmbatchedAxpySimt<true>>(
        dim3{VEC_THREAD_NUM, 1, 1}, tiling.m, tiling.n, tiling.ldb, tiling.tempRowStride, tiling.bOffset,
        startIdx, calNum, reinterpret_cast<__gm__ float*>(b), reinterpret_cast<__gm__ const float*>(temp));
}

void trsmbatched_axpy_trans_kernel_do(
    GM_ADDR b, GM_ADDR temp, const TrsmbatchedAxpyTilingData& tiling, uint32_t numBlocks, void* stream)
{
    trsmbatched_axpy_trans_kernel<<<numBlocks, nullptr, stream>>>(b, temp, tiling);
}

// ---------- Transpose kernel ----------

__simt_vf__ __aicore__ LAUNCH_BOUND(VEC_THREAD_NUM) inline void TrsmbatchedTransposeSimt(
    uint32_t rowStart, uint32_t rowEnd, uint32_t cols, int32_t ldIn, int32_t ldOut,
    __gm__ const float* inGm, __gm__ float* outGm)
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

extern "C" __global__ __aicore__ void trsmbatched_transpose_kernel(
    GM_ADDR in, GM_ADDR out, uint32_t rows, uint32_t cols, int32_t ldIn, int32_t ldOut)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t rowsPerBlock = (rows + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
    uint32_t rowStart = AscendC::GetBlockIdx() * rowsPerBlock;
    if (rowStart >= rows) return;
    uint32_t rowEnd = rowStart + rowsPerBlock;
    if (rowEnd > rows) rowEnd = rows;
    asc_vf_call<TrsmbatchedTransposeSimt>(
        dim3{VEC_THREAD_NUM, 1, 1}, rowStart, rowEnd, cols, ldIn, ldOut,
        reinterpret_cast<__gm__ const float*>(in), reinterpret_cast<__gm__ float*>(out));
}

void trsmbatched_transpose_kernel_do(
    GM_ADDR in, GM_ADDR out, uint32_t rows, uint32_t cols, int32_t ldIn, int32_t ldOut,
    uint32_t numBlocks, void* stream)
{
    trsmbatched_transpose_kernel<<<numBlocks, nullptr, stream>>>(in, out, rows, cols, ldIn, ldOut);
}

// ===================== AIC kernel: Cube GEMM =====================

constexpr uint16_t PIPE_FLAG = 0;
constexpr int64_t L0A_SIZE = 64 * 1024;
constexpr int64_t L1_SIZE = 512 * 1024;
constexpr uint32_t FINAL_ACC = 3;
constexpr uint32_t NON_FINAL_ACC = 2;
constexpr uint64_t FP32_C0 = 8;
constexpr uint64_t FRACTAL = 16;
constexpr uint64_t L0C_C0 = 16;
constexpr uint64_t L1_BUF_NUM = 2;
constexpr uint64_t L1_BUF_MASK = L1_BUF_NUM - 1;
constexpr uint64_t HALF_L0_SIZE = L0A_SIZE / 2;
constexpr uint64_t BASE_K = 8;

template <typename TensorGM, typename TensorL1>
__aicore__ inline void TrsmbatchedGemmCopyGM2L1(TensorGM gmTensor, TensorL1 tensorL1,
    uint64_t off0, uint64_t off1, uint64_t dim0, uint64_t dim1)
{
    auto gmBlock = gmTensor.Slice(MakeCoord(off0, off1), MakeShape(dim0, dim1));
    Copy(MakeCopy(CopyGM2L1{}), tensorL1, gmBlock);
}

template <typename TensorAL1, typename TensorBL1>
__aicore__ inline void TrsmbatchedGemmLoadL0Chunk(TensorAL1 tensorAL1, TensorBL1 tensorBL1,
    uint64_t curML1, uint64_t curKL1, uint64_t nL0, uint64_t iter1, uint64_t& l0PingPong)
{
    using T = float;
    uint64_t kL0Offset = iter1 * BASE_K;
    uint64_t curKL0 = (kL0Offset + BASE_K > curKL1) ? (curKL1 - kL0Offset) : BASE_K;
    uint64_t l0BufId = l0PingPong & 0x1;
    uint64_t l0Offset = HALF_L0_SIZE * l0BufId;
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
    auto layoutAL0 = MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(curML1, curKL0);
    auto tensorAL0 = MakeTensor(MakeMemPtr<Location::L0A, T>(l0Offset), layoutAL0);
    auto tensorBlockAL1 = tensorAL1.Slice(MakeCoord(0, kL0Offset), MakeShape(curML1, curKL0));
    Copy(MakeCopy(CopyL12L0A{}), tensorAL0, tensorBlockAL1);
    auto layoutBL0 = MakeFrameLayout<ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(curKL0, nL0);
    auto tensorBL0 = MakeTensor(MakeMemPtr<Location::L0B, T>(l0Offset), layoutBL0);
    auto tensorBlockBL1 = tensorBL1.Slice(MakeCoord(kL0Offset, 0), MakeShape(curKL0, nL0));
    Copy(MakeCopy(CopyL12L0B{}), tensorBL0, tensorBlockBL1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
    l0PingPong++;
}

template <typename TensorAL1, typename TensorBL1>
__aicore__ inline void TrsmbatchedGemmL0MmadLoop(TensorAL1 tensorAL1, TensorBL1 tensorBL1,
    uint64_t curML1, uint64_t curKL1, uint64_t nL0,
    uint64_t kL1Iter, uint64_t iter0, uint64_t& l0PingPong)
{
    using T = float;
    uint64_t kL0Iter = CeilDiv<uint64_t>(curKL1, BASE_K);
    if (kL0Iter == 0) return;

    TrsmbatchedGemmLoadL0Chunk(tensorAL1, tensorBL1, curML1, curKL1, nL0, 0, l0PingPong);

    auto layoutL0C = MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(curML1, nL0);
    auto tensorL0C = MakeTensor(MakeMemPtr<Location::L0C, float>(0), layoutL0C);

    for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
        uint64_t l0BufIdCur = (l0PingPong - 1) & 0x1;
        uint64_t l0OffsetCur = HALF_L0_SIZE * l0BufIdCur;
        uint64_t kL0Offset = iter1 * BASE_K;
        uint64_t curKL0 = (kL0Offset + BASE_K > curKL1) ? (curKL1 - kL0Offset) : BASE_K;

        if (iter1 + 1 < kL0Iter) {
            TrsmbatchedGemmLoadL0Chunk(tensorAL1, tensorBL1, curML1, curKL1, nL0, iter1 + 1, l0PingPong);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufIdCur);
        bool isLastK = (iter0 + 1 == kL1Iter) && (iter1 + 1 == kL0Iter);
        uint8_t unitFlag = isLastK ? FINAL_ACC : NON_FINAL_ACC;
        MmadParams mmadParams{
            static_cast<uint16_t>(curML1), static_cast<uint16_t>(nL0),
            static_cast<uint16_t>(curKL0), unitFlag, (iter0 == 0 && iter1 == 0)};
        auto layoutAL0Cur = MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(curML1, curKL0);
        auto tensorAL0Cur = MakeTensor(MakeMemPtr<Location::L0A, T>(l0OffsetCur), layoutAL0Cur);
        auto layoutBL0Cur = MakeFrameLayout<ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(curKL0, nL0);
        auto tensorBL0Cur = MakeTensor(MakeMemPtr<Location::L0B, T>(l0OffsetCur), layoutBL0Cur);
        Mmad(MmadAtom<MmadTraits<MmadOperation>>{}.with(mmadParams),
            tensorL0C, tensorAL0Cur, tensorBL0Cur);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufIdCur);
    }
}

template <typename TensorGM>
__aicore__ inline void TrsmbatchedGemmLoadL1Chunk(TensorGM gmLeftTensor, TensorGM gmRightTensor,
    uint64_t mOff, uint64_t nOff, uint64_t k, uint64_t tileKChunk,
    uint64_t mL0, uint64_t nL0, uint64_t kOff, uint64_t& abL1LoopCnt)
{
    uint64_t curK = Min<uint64_t>(tileKChunk, k - kOff);
    uint64_t l1BufId = abL1LoopCnt & L1_BUF_MASK;
    uint64_t l1OffsetA = l1BufId * (L1_SIZE / L1_BUF_NUM);
    uint64_t aSideL1Size = RoundUp<uint64_t>(mL0, FRACTAL) * RoundUp<uint64_t>(curK, FP32_C0);
    uint64_t l1OffsetB = l1OffsetA + aSideL1Size * sizeof(float);
    auto tensorAL1 = MakeTensor(MakeMemPtr<Location::L1, float>(l1OffsetA),
        MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(mL0, curK));
    auto tensorBL1 = MakeTensor(MakeMemPtr<Location::L1, float>(l1OffsetB),
        MakeFrameLayout<ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(curK, nL0));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
    TrsmbatchedGemmCopyGM2L1(gmLeftTensor, tensorAL1, mOff, kOff, mL0, curK);
    TrsmbatchedGemmCopyGM2L1(gmRightTensor, tensorBL1, kOff, nOff, curK, nL0);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
    abL1LoopCnt++;
}

template <typename TensorGM>
__aicore__ inline void TrsmbatchedGemmProcessTile(TensorGM gmLeftTensor, TensorGM gmRightTensor, TensorGM gmTempTensor,
    uint64_t mOff, uint64_t nOff, uint64_t mL0, uint64_t nL0, uint64_t k, uint64_t tileKChunk,
    uint64_t kL1Iter, uint64_t& l0PingPong, uint64_t& abL1LoopCnt)
{
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);

    TrsmbatchedGemmLoadL1Chunk(gmLeftTensor, gmRightTensor, mOff, nOff, k, tileKChunk, mL0, nL0, 0,
        abL1LoopCnt);
    uint64_t curBufId = (abL1LoopCnt - 1) & L1_BUF_MASK;
    uint64_t curKForL1 = Min<uint64_t>(tileKChunk, k);

    for (uint64_t kOff = 0; kOff < k; kOff += tileKChunk) {
        if (kOff + tileKChunk < k) {
            TrsmbatchedGemmLoadL1Chunk(gmLeftTensor, gmRightTensor, mOff, nOff, k, tileKChunk, mL0,
                nL0, kOff + tileKChunk, abL1LoopCnt);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(curBufId);

        uint64_t l1OffsetA = curBufId * (L1_SIZE / L1_BUF_NUM);
        uint64_t aSideL1Size = RoundUp<uint64_t>(mL0, FRACTAL) * RoundUp<uint64_t>(curKForL1, FP32_C0);
        uint64_t l1OffsetB = l1OffsetA + aSideL1Size * sizeof(float);
        auto tensorAL1Cur = MakeTensor(MakeMemPtr<Location::L1, float>(l1OffsetA),
            MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(mL0, curKForL1));
        auto tensorBL1Cur = MakeTensor(MakeMemPtr<Location::L1, float>(l1OffsetB),
            MakeFrameLayout<ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(curKForL1, nL0));

        TrsmbatchedGemmL0MmadLoop(tensorAL1Cur, tensorBL1Cur, mL0, curKForL1, nL0,
            kL1Iter, kOff / tileKChunk, l0PingPong);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(curBufId);

        if (kOff + tileKChunk < k) {
            curBufId = (abL1LoopCnt - 1) & L1_BUF_MASK;
            curKForL1 = Min<uint64_t>(tileKChunk, k - (kOff + tileKChunk));
        }
    }

    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(PIPE_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(PIPE_FLAG);

    auto gmBlockC = gmTempTensor.Slice(MakeCoord(mOff, nOff), MakeShape(mL0, nL0));
    auto tensorL0C = MakeTensor(MakeMemPtr<Location::L0C, float>(0),
        MakeFrameLayout<NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(mL0, nL0));
    MakeCopy(CopyL0C2GM{}).Call(gmBlockC, tensorL0C, FixpipeParams{FINAL_ACC});

    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);
}

__aicore__ inline void TrsmbatchedGemmSetFlags()
{
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);
}

__aicore__ inline void TrsmbatchedGemmWaitFlags()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);
}

extern "C" __global__ __aicore__ void trsmbatched_gemm_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR temp, const TrsmbatchedGemmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();

    const uint32_t m = tiling.m;
    const uint32_t n = tiling.n;
    const uint32_t k = tiling.k;
    const uint32_t lda = tiling.lda;
    const uint32_t ldb = tiling.ldb;

    auto gmLeftTensor = MakeTensor(
        MakeMemPtr<Location::GM>(reinterpret_cast<__gm__ float*>(a) + tiling.aOffset),
        MakeFrameLayout<NDExtLayoutPtn>(m, lda));
    auto gmRightTensor = MakeTensor(
        MakeMemPtr<Location::GM>(reinterpret_cast<__gm__ float*>(x) + tiling.bOffset),
        MakeFrameLayout<NDExtLayoutPtn>(k, ldb));
    auto gmTempTensor = MakeTensor(
        MakeMemPtr<Location::GM>(reinterpret_cast<__gm__ float*>(temp)),
        MakeFrameLayout<NDExtLayoutPtn>(m, tiling.tempRowStride));

    const uint64_t tileM = tiling.tileM;
    const uint64_t tileN = tiling.tileN;
    const uint64_t divM = CeilDiv<uint64_t>(m, tileM);
    const uint64_t divN = CeilDiv<uint64_t>(n, tileN);
    const uint64_t totalTiles = divM * divN;
    const uint64_t kL1Iter = CeilDiv<uint64_t>(k, tiling.tileKChunk);

    TrsmbatchedGemmSetFlags();
    uint64_t l0PingPong = 0;
    uint64_t abL1LoopCnt = 0;

    for (uint64_t tileIdx = AscendC::GetBlockIdx(); tileIdx < totalTiles; tileIdx += AscendC::GetBlockNum()) {
        uint64_t coreIdxM = tileIdx / divN;
        uint64_t coreIdxN = tileIdx % divN;
        if (coreIdxM % 2 == 1) { coreIdxN = divN - 1 - coreIdxN; }
        uint64_t mEnd = Min<uint64_t>((coreIdxM + 1) * tileM, m);
        uint64_t nEnd = Min<uint64_t>((coreIdxN + 1) * tileN, n);
        for (uint64_t mOff = coreIdxM * tileM; mOff < mEnd; mOff += tileM) {
            uint64_t mL0 = Min<uint64_t>(tileM, mEnd - mOff);
            for (uint64_t nOff = coreIdxN * tileN; nOff < nEnd; nOff += tileN) {
                uint64_t nL0 = Min<uint64_t>(tileN, nEnd - nOff);
                TrsmbatchedGemmProcessTile(gmLeftTensor, gmRightTensor, gmTempTensor,
                    mOff, nOff, mL0, nL0, k, tiling.tileKChunk, kL1Iter, l0PingPong, abL1LoopCnt);
            }
        }
    }

    TrsmbatchedGemmWaitFlags();
}

void trsmbatched_gemm_kernel_do(
    GM_ADDR a, GM_ADDR x, GM_ADDR temp, const TrsmbatchedGemmTilingData& tiling, uint32_t numBlocks, void* stream)
{
    trsmbatched_gemm_kernel<<<numBlocks, nullptr, stream>>>(a, x, temp, tiling);
}
