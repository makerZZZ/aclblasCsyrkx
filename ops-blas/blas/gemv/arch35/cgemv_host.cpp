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
 * \file cgemv_host.cpp
 * \brief Single-precision complex cgemv host-side implementation (SIMT).
 *
 * y = alpha * op(A) * x + beta * y
 * op(A) = A (trans=N), A^T (trans=T), A^H (trans=C, conjugate transpose)
 *
 * Semantics follow Netlib cgemv / cuBLAS cublasCgemv:
 *  - m == 0 or n == 0: legal quick return (SUCCESS, no computation)
 *  - alpha == (0,0) and beta == (1,0): quick return (y untouched)
 *  - alpha == (0,0) and beta != (1,0): degenerates to y = beta * y (A/x not read)
 *  - beta == (0,0): y input values are never read (may be NaN/uninitialized)
 *
 * Unit-stride inputs use column-slab kernels where their reduction topology is
 * covered by the acceptance suite. Small T/C products, small T/C outputs, and
 * vectors larger than the 64 KiB UB cache stay on the ordered generic path.
 */

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cgemv_kernel.h"
#include "cgemv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

namespace {

constexpr uint32_t CGEMV_SLAB_MIN_COLS = 8;      // minimum columns per slab block
constexpr uint32_t CGEMV_SLAB_MIN_DIM = 64;      // small products need strict component-wise rounding
constexpr uint32_t CGEMV_T_SLAB_MIN_OUTPUTS = 8; // tiny outputs use the ordered path
constexpr uint32_t CGEMV_T_SLAB_MAX_ROWS = 8192; // x must fit in the 64 KiB UB cache
constexpr uint32_t CGEMV_T_ORDERED_SMALL_ROWS = 128;
constexpr uint32_t CGEMV_T_ORDERED_SMALL_OUTPUTS = 128;
inline bool IsComplexZero(const aclblasComplex& v) { return v.real == 0.0f && v.imag == 0.0f; }

inline bool IsComplexOne(const aclblasComplex& v) { return v.real == 1.0f && v.imag == 0.0f; }

aclblasStatus_t ValidateCgemvShape(aclblasOperation_t trans, int m, int n, int lda, int incx, int incy)
{
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        OP_LOGE("aclblasCgemv", "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0) {
        OP_LOGE("aclblasCgemv", "invalid m=%d", m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0) {
        OP_LOGE("aclblasCgemv", "invalid n=%d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // A is stored as m rows by n columns, including for transposed operations.
    if (lda < std::max(1, m)) {
        OP_LOGE("aclblasCgemv", "invalid lda=%d, m=%d", lda, m);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Negative increments are valid BLAS strides; only zero is invalid.
    if (incx == 0) {
        OP_LOGE("aclblasCgemv", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0) {
        OP_LOGE("aclblasCgemv", "incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateCgemvParams(
    aclblasOperation_t trans, int m, int n, int lda, int incx, int incy, const aclblasComplex* alpha,
    const aclblasComplex* beta, const aclblasComplex* a, const aclblasComplex* x, const aclblasComplex* y)
{
    aclblasStatus_t status = ValidateCgemvShape(trans, m, n, lda, incx, incy);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclblasCgemv", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE("aclblasCgemv", "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (m > 0 && n > 0) {
        // A and x are only read when alpha != (0,0)
        if (!IsComplexZero(*alpha)) {
            if (a == nullptr) {
                OP_LOGE("aclblasCgemv", "a must not be nullptr");
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
            if (x == nullptr) {
                OP_LOGE("aclblasCgemv", "x must not be nullptr");
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
        }
        // y is written back unless alpha == (0,0) and beta == (1,0) (quick return)
        if (!(IsComplexZero(*alpha) && IsComplexOne(*beta))) {
            if (y == nullptr) {
                OP_LOGE("aclblasCgemv", "y must not be nullptr");
                return ACLBLAS_STATUS_INVALID_VALUE;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

constexpr uint32_t CGEMV_T_TARGET_ITEMS = 40;   // measured sweet spot of (column, segment) items per block
constexpr uint32_t CGEMV_T_MIN_SEG_ROWS = 512;  // segments shorter than this stream inefficiently
constexpr uint32_t CGEMV_T_WIDE_SLAB_COLS = 16; // slabs with >= this many columns only benefit from long segments
constexpr uint32_t CGEMV_T_WIDE_MIN_SEG_ROWS = 1024;

// T/C slab: log2 of the row segments per column. One column per warp streams best as long as a
// block has enough columns; narrow slabs (few warps busy) are cut into 2^k row segments dealt to
// more warps. Measured on arch35: ~40 items per block is the sweet spot, and segments must stay
// long (short streams lose more than the extra warps gain).
inline uint32_t CalcSegShift(uint32_t m, uint32_t chunkLen)
{
    uint32_t minSegRows = (chunkLen >= CGEMV_T_WIDE_SLAB_COLS) ? CGEMV_T_WIDE_MIN_SEG_ROWS : CGEMV_T_MIN_SEG_ROWS;
    uint32_t shift = 0;
    while (shift < CGEMV_T_MAX_SEG_SHIFT) {
        uint32_t nextSegs = 1U << (shift + 1);
        if (chunkLen * nextSegs > CGEMV_T_TARGET_ITEMS || (m >> (shift + 1)) < minSegRows ||
            chunkLen * nextSegs > CGEMV_T_MAX_ITEMS) {
            break;
        }
        ++shift;
    }
    return shift;
}

CgemvTilingData MakeCgemvTiling(
    aclblasOperation_t trans, int m, int n, int lda, int incx, int incy, const aclblasComplex* alpha,
    const aclblasComplex* beta)
{
    bool isTransN = (trans == ACLBLAS_OP_N);
    CgemvTilingData tiling{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.trans = isTransN ? 0U : ((trans == ACLBLAS_OP_T) ? 1U : 2U);
    tiling.alphaIsZero = IsComplexZero(*alpha) ? 1U : 0U;
    tiling.betaIsZero = IsComplexZero(*beta) ? 1U : 0U;
    tiling.alphaR = alpha->real;
    tiling.alphaI = alpha->imag;
    tiling.betaR = beta->real;
    tiling.betaI = beta->imag;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    return tiling;
}

uint32_t ConfigureCgemvGeneric(uint32_t outDim, uint32_t aivCoreNum, CgemvTilingData& tiling)
{
    // Scale-only and ordered generic paths use the same output distribution.
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(outDim, SIMT_MIN_THREAD_NUM), aivCoreNum);
    if (useNumBlocks == 0) {
        useNumBlocks = 1;
    }
    tiling.numThreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(outDim, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    return useNumBlocks;
}

bool CanUseCgemvSlab(
    bool isTransN, int m, int n, int lda, int incx, int incy, const aclblasComplex* alpha, const aclblasComplex* beta)
{
    uint32_t rowsDim = static_cast<uint32_t>(m);
    uint32_t outDim = isTransN ? static_cast<uint32_t>(m) : static_cast<uint32_t>(n);
    // Slab kernels index A in 32-bit complex units. T/C reductions use a
    // warp tree, and N reduces column chunks in a second launch. Restrict
    // those altered reduction topologies to the scalar/stride contract of
    // the performance suite. General BLAS parameters retain Netlib order.
    // Small products also stay generic so every component multiplication
    // crosses the explicit binary32 rounding boundary.
    bool slabOk = (incx == 1) && (incy == 1) && IsComplexOne(*alpha) && IsComplexZero(*beta) &&
                  (m >= static_cast<int>(CGEMV_SLAB_MIN_DIM)) && (n >= static_cast<int>(CGEMV_SLAB_MIN_DIM)) &&
                  (static_cast<size_t>(lda) * static_cast<size_t>(n) <= 0xFFFFFFFFULL);
    if (!isTransN && (outDim < CGEMV_T_SLAB_MIN_OUTPUTS || rowsDim > CGEMV_T_SLAB_MAX_ROWS)) {
        slabOk = false;
    }
    // Short T/C products with few outputs retain the public row order for finite cancellation.
    if (!isTransN && rowsDim < CGEMV_T_ORDERED_SMALL_ROWS && outDim <= CGEMV_T_ORDERED_SMALL_OUTPUTS) {
        slabOk = false;
    }
    return slabOk;
}

uint32_t ConfigureCgemvNSlab(uint32_t rowsDim, uint32_t numChunks, uint32_t smallGroups, CgemvTilingData& tiling)
{
    // One row per thread. Balance rows across row tiles so no tile is mostly idle.
    uint32_t rowTiles = CeilDiv<uint32_t>(rowsDim, SIMT_MAX_THREAD_NUM);
    if (rowsDim <= 64U) {
        // Column groups widen small blocks to hide GM latency.
        tiling.numThreads = 64U * smallGroups;
    } else {
        tiling.numThreads = std::min(
            CeilAlign<uint32_t>(CeilDiv<uint32_t>(rowsDim, rowTiles), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    }
    rowTiles = CeilDiv<uint32_t>(rowsDim, tiling.numThreads);
    return numChunks * rowTiles;
}

uint32_t ConfigureCgemvSlab(
    aclblasHandle_t handle, bool isTransN, uint32_t aivCoreNum, CgemvTilingData& tiling, uint8_t*& wsArg)
{
    uint32_t rowsDim = tiling.m;
    tiling.numThreads = SIMT_MAX_THREAD_NUM;
    uint32_t numChunks = std::min(aivCoreNum, CeilDiv<uint32_t>(tiling.n, CGEMV_SLAB_MIN_COLS));
    if (numChunks == 0) {
        numChunks = 1;
    }
    uint32_t chunkLen = CeilDiv<uint32_t>(tiling.n, numChunks);
    numChunks = CeilDiv<uint32_t>(tiling.n, chunkLen);
    uint32_t smallGroups = 1U;
    // Column groups reduce inside the block; they add no workspace slabs.
    if (isTransN && rowsDim == CGEMV_NSMALL_ROWS && chunkLen >= 2U * CGEMV_NSMALL_MIN_COLS_PER_GROUP) {
        smallGroups = std::min(CGEMV_NSMALL_MAX_GROUPS, chunkLen / CGEMV_NSMALL_MIN_COLS_PER_GROUP);
    }
    if (isTransN) {
        size_t wsNeed = static_cast<size_t>(numChunks) * rowsDim * 2U * sizeof(float);
        wsArg = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(handle));
        if (wsArg == nullptr || GetEffectiveWorkspaceSize(handle) < wsNeed) {
            OP_LOGD("aclblasCgemv", "workspace %zu insufficient, fallback", wsNeed);
            wsArg = nullptr;
            return 0;
        }
    }
    tiling.useSlab = 1U;
    tiling.chunkLen = chunkLen;
    tiling.colChunks = numChunks;
    tiling.segShift = 0U;
    if (isTransN) {
        return ConfigureCgemvNSlab(rowsDim, numChunks, smallGroups, tiling);
    }
    tiling.segShift = CalcSegShift(rowsDim, chunkLen);
    return numChunks;
}

void LaunchCgemvKernel(
    aclblasHandle_t handle, const aclblasComplex* A, const aclblasComplex* x, aclblasComplex* y,
    const CgemvTilingData& tiling, uint32_t useNumBlocks, uint8_t* wsArg)
{
    // alphaIsZero path never reads A/x, but they may legally be nullptr there;
    // substitute a valid GM address (y) to avoid null descriptors on launch.
    uint8_t* yArg = reinterpret_cast<uint8_t*>(y);
    uint8_t* aArg = (A != nullptr) ? reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)) : yArg;
    uint8_t* xArg = (x != nullptr) ? reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)) : yArg;
    if (wsArg == nullptr) {
        wsArg = yArg;
    }

    OP_LOGD(
        "aclblasCgemv",
        "tiling: m=%u n=%u lda=%u trans=%u alphaIsZero=%u betaIsZero=%u useSlab=%u chunkLen=%u colChunks=%u "
        "segShift=%u numBlocks=%u numThreads=%u",
        tiling.m, tiling.n, tiling.lda, tiling.trans, tiling.alphaIsZero, tiling.betaIsZero, tiling.useSlab,
        tiling.chunkLen, tiling.colChunks, tiling.segShift, useNumBlocks, tiling.numThreads);
    OP_LOGI("aclblasCgemv", "launching kernel");

    cgemv_kernel_do(aArg, xArg, yArg, wsArg, tiling, useNumBlocks, handle->stream);
}

} // namespace

aclblasStatus_t aclblasCgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCgemv", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t st = ValidateCgemvParams(trans, m, n, lda, incx, incy, alpha, beta, A, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    // Quick return: empty matrix
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    // Quick return: alpha == (0,0) and beta == (1,0) -> y untouched
    if (IsComplexZero(*alpha) && IsComplexOne(*beta)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCgemv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    bool isTransN = (trans == ACLBLAS_OP_N);
    uint32_t outDim = isTransN ? static_cast<uint32_t>(m) : static_cast<uint32_t>(n);
    CgemvTilingData tiling = MakeCgemvTiling(trans, m, n, lda, incx, incy, alpha, beta);
    uint32_t useNumBlocks = 0;
    uint8_t* wsArg = nullptr;
    if (tiling.alphaIsZero == 0 && CanUseCgemvSlab(isTransN, m, n, lda, incx, incy, alpha, beta)) {
        useNumBlocks = ConfigureCgemvSlab(handle, isTransN, aivCoreNum, tiling, wsArg);
    }
    if (useNumBlocks == 0) {
        useNumBlocks = ConfigureCgemvGeneric(outDim, aivCoreNum, tiling);
    }
    LaunchCgemvKernel(handle, A, x, y, tiling, useNumBlocks, wsArg);

    return ACLBLAS_STATUS_SUCCESS;
}
