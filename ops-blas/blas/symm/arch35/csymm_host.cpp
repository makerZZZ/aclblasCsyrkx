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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>
#include <complex>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "csymm_kernel.h"
#include "gemm/arch35/gemm_tiling_helpers.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

// ==========================================================================
//  Core-count queries cached per-process (single-device assumption, matching
//  the rest of this file). Querying the platform manager on every call costs
//  real microseconds on the small-shape fast paths.
// ==========================================================================
static uint32_t CsymmAivCores()
{
    static uint32_t n = []() { return GetAivCoreCount(); }();
    return n;
}

static uint32_t CsymmAicCores()
{
    static uint32_t n = []() { return GetAicCoreCount(); }();
    return n;
}

// Try uint64_t multiply/add with overflow check (mirrors ssymm arch22). On
// overflow the caller must refuse to launch rather than compute a wrong,
// wrapped-around workspace size (G.RES.02).
static bool TryMulU64(uint64_t lhs, uint64_t rhs, uint64_t* out)
{
    if (out == nullptr) {
        return false;
    }
    if (lhs == 0 || rhs == 0) {
        *out = 0;
        return true;
    }
    if (lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

static bool TryAddU64(uint64_t lhs, uint64_t rhs, uint64_t* out)
{
    if (out == nullptr) {
        return false;
    }
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

// ==========================================================================
//  Tiling helpers (cube FP32 GEMM, identical scheme to cgemm)
// ==========================================================================
static void CalcMultiCorePartition(GemmTilingData& tiling, uint32_t maxCores, int m, int n)
{
    // Pick the (mBlocks, nBlocks) core grid that minimizes the maximum number of
    // baseM x baseN tiles any single core executes. Maximizing mb*nb (cores)
    // alone is wrong for non-divisible shapes: 1524x3086 with the 256 tile has
    // 6x13 tiles; a 4x7=28-core grid gives the busiest core 2x2=4 tiles, while a
    // 2x13=26-core grid caps it at ceil(6/2)*1=3 tiles (-25% tail time) for only
    // two fewer cores. Square, divisible shapes (1024^2: 4x4=16 cores, 1 tile per
    // core) are unaffected. maxCores lets the caller trade cores for true
    // parallelism when the three 3m GEMMs run on separate streams.
    const int32_t maxC = static_cast<int32_t>(maxCores);
    int32_t mTiles = (m + tiling.baseM - 1) / tiling.baseM;
    int32_t nTiles = (n + tiling.baseN - 1) / tiling.baseN;
    int32_t bestMBlocks = 1;
    int32_t bestNBlocks = 1;
    int32_t bestMaxTiles = INT32_MAX;
    int32_t bestCores = 0;
    for (int32_t mb = 1; mb <= mTiles && mb <= maxC; mb++) {
        int32_t nbMax = std::min(nTiles, maxC / mb);
        for (int32_t nb = 1; nb <= nbMax; nb++) {
            const int32_t perCoreMax = ((mTiles + mb - 1) / mb) * ((nTiles + nb - 1) / nb);
            const int32_t cores = mb * nb;
            if (perCoreMax < bestMaxTiles || (perCoreMax == bestMaxTiles && cores > bestCores)) {
                bestMaxTiles = perCoreMax;
                bestCores = cores;
                bestMBlocks = mb;
                bestNBlocks = nb;
            }
        }
    }
    tiling.mBlocks = bestMBlocks;
    tiling.nBlocks = bestNBlocks;
    tiling.usedCoreNum = bestCores;
    tiling.maxTilesPerCore = bestMaxTiles;
}

static GemmTilingData CalGemmTilingData(
    int m, int n, int k, int lda, int ldb, int ldc,
    aclblasOperation_t transa, aclblasOperation_t transb,
    float alphaReal, float alphaImag, float betaReal, float betaImag)
{
    GemmTilingData tiling{};
    InitGemmTilingBase(tiling, m, n, k, lda, ldb, ldc);
    // Adaptive tile size: large matrices use 64x64x128. The 32KB L0 half-slot
    // holds 64x128x4=32KB A and 64x128x4=32KB B, so L1->L0 double-buffering
    // (ping-pong) overlaps the copy with MMAD and halves the per-BK copy count
    // vs BK=64 (32K/BK Copy invocations per tile; the per-Call fixed cost of
    // tensor_api Copy dominates at small tiles).
    // Tile 256x256x64 is the cube optimum on this part: it minimizes the number
    // of L1->L0 Copy invocations per tile (32) — the dominant fixed cost — and
    // keeps one 256x256 tile per core (16 cores). Smaller tiles or a double-
    // buffered BK=32 both roughly double the Copy count and measure slower
    // (128x128x64: 18.6ms, 64x64x128: 43.9ms, 256x256x32: 9.7ms on 1024^2).
    // Skewed shapes (e.g. 323x1900, m<n) used to fall to the 64x64x128 tile and
    // explode into ~180 tiny tiles per GEMM; any side >= 768 gets the 256 tile so
    // the tile count stays near the 28 AIC cores. L1 still fits: the ping-pong
    // slot holds A(256x128x4=128KB)+B(128x256x4=128KB) regardless of shape.
    const bool useBigTile = (m >= 768 || n >= 768);
    tiling.baseM = useBigTile ? 256 : GEMM_BASE_M;
    tiling.baseN = useBigTile ? 256 : GEMM_BASE_N;
    tiling.baseK = useBigTile ? 64 : GEMM_BASE_K;
    // BK=64 for both tiles. A BK=32 experiment measured ~3.5% faster on big tiles
    // measured ~3.5% faster on big tiles (1575^2 1822->1758us, 1074^2
    // 643->626us) by giving the L1->L0 copies a ping-pong L0 buffer, but it
    // BROKE accuracy: with baseK=32 the kernel's per-chunk (BK-step) MMAD
    // accumulation order changes and the <256,32,256> path also feeds the
    // wrong B layout into the 3m formulas (1247-case accuracy run went
    // 1213/1213 PASS -> 753/753 FAIL). Kept off by default.
    tiling.baseK = useBigTile ? 64 : GEMM_BASE_K;
    // Skewed shapes: a 256-wide tile on the short side leaves the AIC cores
    // idle (256x1024: 4x1 tile grid, 4 cores/GEMM, 12/28 used across 3 GEMMs,
    // cube ~45us). Halving the tile on the short side doubles the cores per
    // GEMM (8) so the three GEMMs spread over 24 cores. L1 still fits: the
    // ping-pong slot holds A(256x128x4=128KB)+B(128x128x4=64KB)=192KB for a
    // 256x128 tile, or 64KB+128KB for 128x256 (both <= 256KB). Only short
    // sides < 512 trigger it, so square/large shapes keep the 256 tile.
    if (useBigTile && std::min(m, n) < 512) {
        if (n > m) tiling.baseN = 128;  // wide: short side is m -> tile along m
        else       tiling.baseM = 128;  // tall: short side is n -> tile along n
    }
    // L1 per ping-pong slot = 256KB: A(256x128x4=128KB) + B(128x256x4=128KB).
    tiling.tileKChunk = useBigTile ? 128 : GEMM_TILE_K_CHUNK;
    FillGemmTilingScalars(tiling, alphaReal, alphaImag, betaReal, betaImag,
                          (transa != ACLBLAS_OP_N) ? 1 : 0,
                          (transb != ACLBLAS_OP_N) ? 1 : 0);
    return tiling;
}

static void PrepareCubeTiling(GemmTilingData& tiling, uint32_t cubeCoreNum)
{
    ApplyColMajorSwap(tiling);
    // Second-order skew: the first-order skew (CalGemmTilingData) only halves
    // the SHORT side's tile, which is useless when that side is already a
    // single <256 tile (e.g. tall 2896x128: cube grid 1x12 = 12 blocks over 28
    // cores). If the resulting tile grid still cannot fill the cores, halve the
    // larger remaining tile axis down to 128 so more blocks spread over the 28
    // AIC cores. Tile splitting is bit-identical per output element (M/N tiles
    // never change the per-element accumulation order), so accuracy is safe.
    uint32_t mTiles = (tiling.baseM > 0) ? CeilDiv(tiling.m, tiling.baseM) : 0;
    uint32_t nTiles = (tiling.baseN > 0) ? CeilDiv(tiling.n, tiling.baseN) : 0;
    // Only split an axis that is NOT a full (full-width, max-efficiency) tile,
    // and only when K is long enough that per-tile fixed cost is amortized.
    if (mTiles > 0 && nTiles > 0 && (mTiles * nTiles) < cubeCoreNum &&
        tiling.k >= 128) {
        const uint32_t m = static_cast<uint32_t>(tiling.m);
        const uint32_t n = static_cast<uint32_t>(tiling.n);
        const bool mFull = (mTiles * tiling.baseM == m);
        const bool nFull = (nTiles * tiling.baseN == n);
        if (!nFull && tiling.baseN > 128) {
            tiling.baseN = 128;
        } else if (!mFull && tiling.baseM > 128) {
            tiling.baseM = 128;
        }
    }
    CalcMultiCorePartition(tiling, cubeCoreNum, tiling.m, tiling.n);
}

// ==========================================================================
//  Parameter validation
// ==========================================================================
static aclblasStatus_t ValidateCsymmParams(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    int m, int n, const void* alpha, int lda, int ldb, const void* beta, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCsymm", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (side != ACLBLAS_SIDE_LEFT && side != ACLBLAS_SIDE_RIGHT) {
        OP_LOGE("aclblasCsymm", "invalid side: %d", static_cast<int>(side));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasCsymm", "invalid uplo: %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0) {
        OP_LOGE("aclblasCsymm", "m, n must be non-negative: m=%d, n=%d", m, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // BLAS standard: m == 0 || n == 0 is a legal no-op — even with unusable
    // leading dimensions or null pointers (checked before ld/alpha/beta).
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    int k = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    int ldaMin = std::max(1, k);
    if (lda < ldaMin) {
        OP_LOGE("aclblasCsymm", "lda=%d must be >= %d", lda, ldaMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    int ldbMin = std::max(1, m);
    if (ldb < ldbMin) {
        OP_LOGE("aclblasCsymm", "ldb=%d must be >= %d", ldb, ldbMin);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, m)) {
        OP_LOGE("aclblasCsymm", "ldc=%d must be >= %d", ldc, std::max(1, m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE("aclblasCsymm", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (beta == nullptr) {
        OP_LOGE("aclblasCsymm", "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateCsymmPointers(
    int k, float alphaAbs, const void* A, const void* B, float betaAbs, const void* C)
{
    if (k > 0 && alphaAbs != 0.0f) {
        if (A == nullptr) {
            OP_LOGE("aclblasCsymm", "A must not be nullptr when k > 0 and alpha != 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (B == nullptr) {
            OP_LOGE("aclblasCsymm", "B must not be nullptr when k > 0 and alpha != 0");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (betaAbs != 0.0f && C == nullptr) {
        OP_LOGE("aclblasCsymm", "C must not be nullptr when beta != 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Workspace size planning (G.RES.02): all byte counts accumulated in uint64_t
//  with overflow checks; on overflow the caller refuses to launch.
// ==========================================================================
struct CsymmWsSizes {
    uint64_t reA = 0;   // Ar/Ai/Aplus plane (S x S), 512B-aligned
    uint64_t reB = 0;   // Br/Bi/Bplus plane (M x N), 512B-aligned
    uint64_t temp = 0;  // one w-plane (tempLdc x N), 512B-aligned
    uint64_t total = 0; // 3*reA + 3*reB + 6*temp
};

static bool CsymmComputeWorkspace(int m, int n, int S, CsymmWsSizes* out)
{
    uint64_t reABytes = 0;
    uint64_t reBBytes = 0;
    uint64_t tempBytes = 0;
    if (!TryMulU64(static_cast<uint64_t>(S), static_cast<uint64_t>(S), &reABytes) ||
        !TryMulU64(reABytes, static_cast<uint64_t>(sizeof(float)), &reABytes) ||
        !TryMulU64(static_cast<uint64_t>(m), static_cast<uint64_t>(n), &reBBytes) ||
        !TryMulU64(reBBytes, static_cast<uint64_t>(sizeof(float)), &reBBytes) ||
        !TryMulU64(static_cast<uint64_t>(CeilAlign(m, GEMM_FRACTAL)), static_cast<uint64_t>(n), &tempBytes) ||
        !TryMulU64(tempBytes, static_cast<uint64_t>(sizeof(float)), &tempBytes)) {
        return false;
    }
    auto align512 = [](uint64_t x) { return (x + 511) & ~static_cast<uint64_t>(511); };
    out->reA = align512(reABytes);
    out->reB = align512(reBBytes);
    out->temp = align512(tempBytes);
    uint64_t termRe = 0;
    uint64_t termRb = 0;
    uint64_t termTemp = 0;
    if (!TryMulU64(out->reA, 3, &termRe) ||
        !TryMulU64(out->reB, 3, &termRb) ||
        !TryMulU64(out->temp, 6, &termTemp)) {
        return false;
    }
    uint64_t total = 0;
    if (!TryAddU64(termRe, termRb, &total) || !TryAddU64(total, termTemp, &total)) {
        return false;
    }
    out->total = total;
    return true;
}

// alpha == 0  ->  C = beta * C (in-place scale kernel).
static aclblasStatus_t CsymmRunAlphaZeroPath(
    aclrtStream stream, uint8_t* cPtr, int m, int n, int ldc,
    float betaRe, float betaIm, uint32_t aivCoreNum)
{
    if (betaRe == 1.0f && betaIm == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCsymm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t vecBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    csymm_scale_do(vecBlocks, stream, cPtr, m, n, ldc, betaRe, betaIm, 1);
    return ACLBLAS_STATUS_SUCCESS;
}

// Small-shape single-launch direct path (m, n <= 16).
static aclblasStatus_t CsymmRunSmallPath(
    aclrtStream stream, const aclblasComplex* A, int lda, aclblasFillMode_t uplo,
    const aclblasComplex* B, int ldb, aclblasComplex* C, int ldc,
    int m, int n, int S, int sideLeft,
    float alphaRe, float alphaIm, float betaRe, float betaIm, uint32_t aivCoreNum)
{
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCsymm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t smallBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    csymm_small_do(smallBlocks, stream,
                   reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)),
                   lda, uplo,
                   reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(B)),
                   ldb,
                   reinterpret_cast<uint8_t*>(C), ldc,
                   m, n, S, sideLeft,
                   alphaRe, alphaIm, betaRe, betaIm);
    return ACLBLAS_STATUS_SUCCESS;
}

// Launch the prep (mirror + de-interleave) and fused 3m cube GEMM for the main path.
static void CsymmLaunchPrepAndCube(
    aclrtStream stream,
    const aclblasComplex* A, int lda, aclblasFillMode_t uplo,
    const aclblasComplex* B, int ldb, aclblasSideMode_t side,
    int m, int n, int S, float alphaRe, float alphaIm, float betaRe, float betaIm,
    uint32_t aivCoreNum, uint32_t cubeCoreNum,
    uint8_t* d_reA, uint8_t* d_imA, uint8_t* d_reB, uint8_t* d_imB,
    uint8_t* d_apA, uint8_t* d_apB, uint8_t* d_w1a, uint8_t* d_w1b,
    uint8_t* d_w2a, uint8_t* d_w2b, uint8_t* d_w3a, uint8_t* d_w3b)
{
    uint32_t prepBlocks = std::min(aivCoreNum,
        static_cast<uint32_t>(m) + static_cast<uint32_t>(n));
    csymm_prep_do(prepBlocks, stream,
                  reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)),
                  lda, static_cast<int32_t>(uplo),
                  reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(B)),
                  ldb, d_reA, d_imA, d_reB, d_imB, d_apA, d_apB, m, n, S, 0, S);

    GemmTilingData cubeTiling = CalGemmTilingData(
        m, n, S, m, S, m, ACLBLAS_OP_N, ACLBLAS_OP_N,
        alphaRe, alphaIm, betaRe, betaIm);
    PrepareCubeTiling(cubeTiling, cubeCoreNum);
    cubeTiling.ldc = static_cast<int32_t>(CeilAlign(m, GEMM_FRACTAL));
    uint32_t numBlocks = static_cast<uint32_t>(cubeTiling.usedCoreNum);

    const int32_t kTotal = cubeTiling.k;
    bool kSplit = (kTotal >= 1500);
    cubeTiling.kStart = 0;
    cubeTiling.kEnd = kTotal;
    cubeTiling.kSegmentCount = kSplit ? 2 : 1;

    if (side == ACLBLAS_SIDE_LEFT) {
        csymm_cube_gemm3_do(numBlocks, stream,
            d_apB, d_apA, d_w1a, d_w1b,  // w1 = Aplus*Bplus
            d_reB, d_reA, d_w2a, d_w2b,  // w2 = Ar*Br
            d_imB, d_imA, d_w3a, d_w3b,  // w3 = Ai*Bi
            cubeTiling);
    } else {
        csymm_cube_gemm3_do(numBlocks, stream,
            d_apA, d_apB, d_w1a, d_w1b,  // w1 = Bplus*Aplus
            d_reA, d_reB, d_w2a, d_w2b,  // w2 = Br*Ar
            d_imA, d_imB, d_w3a, d_w3b,  // w3 = Bi*Ai
            cubeTiling);
    }
}

// Launch the combine (reconstruct C from the 3m planes) kernel for the main path.
static void CsymmLaunchCombine(
    aclrtStream stream, aclblasComplex* C, int m, int n, int ldc,
    float alphaRe, float alphaIm, float betaRe, float betaIm,
    uint32_t aivCoreNum, bool kSplit,
    uint8_t* d_w1a, uint8_t* d_w2a, uint8_t* d_w3a, uint8_t* d_w1b, uint8_t* d_w2b, uint8_t* d_w3b)
{
    uint32_t combineBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    csymm_combine_do(combineBlocks, stream,
        d_w1a, d_w2a, d_w3a, d_w1b, d_w2b, d_w3b,
        (kSplit ? 1 : 0), static_cast<int32_t>(CeilAlign(m, GEMM_FRACTAL)),
        reinterpret_cast<uint8_t*>(C), m, n, ldc,
        alphaRe, alphaIm, betaRe, betaIm);
}

// The twelve 3m workspace planes carved out of the handle workspace by
// CsymmPrepareWorkspace (see the layout comment there).
struct CsymmWsPlanes {
    uint8_t* reA = nullptr;
    uint8_t* imA = nullptr;
    uint8_t* apA = nullptr;
    uint8_t* reB = nullptr;
    uint8_t* imB = nullptr;
    uint8_t* apB = nullptr;
    uint8_t* w1a = nullptr;
    uint8_t* w2a = nullptr;
    uint8_t* w3a = nullptr;
    uint8_t* w1b = nullptr;
    uint8_t* w2b = nullptr;
    uint8_t* w3b = nullptr;
};

// Size the 3m workspace (3 x A plane + 3 x B plane + 6 x temp plane), request it
// from the handle and carve the twelve planes out of the 512B-aligned base:
//   Ar/Ai/Aplus = S x S,  Br/Bi/Bplus = m x n,  w1..w3 (a/b) = tempLdc x n.
static aclblasStatus_t CsymmPrepareWorkspace(
    _aclblas_handle* h, int m, int n, int S, CsymmWsPlanes* planes)
{
    CsymmWsSizes ws{};
    if (!CsymmComputeWorkspace(m, n, S, &ws)) {
        OP_LOGE("aclblasCsymm", "workspace size overflow (S=%d, m=%d, n=%d)", S, m, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // The base below is rounded up to a 512B boundary, which consumes up to 511
    // bytes from the head of the block. Request that slack up front, otherwise a
    // workspace whose address is not already 512B-aligned leaves fewer than
    // ws.total usable bytes and the tail plane (w3b = w2b + ws.temp) is written
    // past the allocated range (G.RES.02). A user-owned workspace sized to
    // exactly ws.total now fails here instead of silently overflowing.
    constexpr uintptr_t kAlignMask = 511;
    uint64_t paddedTotal = 0;
    if (!TryAddU64(ws.total, static_cast<uint64_t>(kAlignMask), &paddedTotal)) {
        OP_LOGE("aclblasCsymm",
                "workspace size overflow while reserving 512B alignment slack (S=%d, m=%d, n=%d)",
                S, m, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, static_cast<size_t>(paddedTotal));
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCsymm",
                "workspace ensure failed, need %zu bytes (incl. 512B align slack), ret=%d",
                static_cast<size_t>(paddedTotal), wsRet);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    uint8_t* workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    uintptr_t wsAddr = reinterpret_cast<uintptr_t>(workspace);
    uint8_t* base = reinterpret_cast<uint8_t*>((wsAddr + kAlignMask) & ~kAlignMask);
    // Belt-and-braces assertion on top of the reserved slack above: the aligned
    // base must still have ws.total contiguous bytes available.
    if (reinterpret_cast<uintptr_t>(base) + static_cast<uintptr_t>(ws.total) >
        wsAddr + h->workspace_size) {
        OP_LOGE("aclblasCsymm",
                "workspace 512B alignment overflow: need %llu bytes after align, "
                "but only %zu bytes available at %p",
                static_cast<unsigned long long>(ws.total), h->workspace_size, workspace);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    planes->reA = base;
    planes->imA = planes->reA + ws.reA;
    planes->apA = planes->imA + ws.reA;   // Ar + Ai (3m)
    planes->reB = planes->apA + ws.reA;
    planes->imB = planes->reB + ws.reB;
    planes->apB = planes->imB + ws.reB;   // Br + Bi (3m)
    planes->w1a = planes->apB + ws.reB;
    planes->w2a = planes->w1a + ws.temp;
    planes->w3a = planes->w2a + ws.temp;
    planes->w1b = planes->w3a + ws.temp;
    planes->w2b = planes->w1b + ws.temp;
    planes->w3b = planes->w2b + ws.temp;
    return ACLBLAS_STATUS_SUCCESS;
}

// Main path: mirror+de-interleave prep, fused 3m cube GEMM, combine.
static aclblasStatus_t CsymmRunMainCompute(
    _aclblas_handle* h, aclrtStream stream,
    aclblasSideMode_t side, aclblasFillMode_t uplo,
    const aclblasComplex* A, const aclblasComplex* B, aclblasComplex* C,
    int lda, int ldb, int ldc, int m, int n, int S,
    float alphaRe, float alphaIm, float betaRe, float betaIm,
    uint32_t aivCoreNum, uint32_t cubeCoreNum)
{
    CsymmWsPlanes p{};
    const aclblasStatus_t wsRet = CsymmPrepareWorkspace(h, m, n, S, &p);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        return wsRet;
    }
    const bool kSplit = (S >= 1500);
    CsymmLaunchPrepAndCube(stream, A, lda, uplo, B, ldb, side, m, n, S,
                           alphaRe, alphaIm, betaRe, betaIm, aivCoreNum, cubeCoreNum,
                           p.reA, p.imA, p.reB, p.imB, p.apA, p.apB,
                           p.w1a, p.w1b, p.w2a, p.w2b, p.w3a, p.w3b);
    CsymmLaunchCombine(stream, C, m, n, ldc, alphaRe, alphaIm, betaRe, betaIm,
                       aivCoreNum, kSplit, p.w1a, p.w2a, p.w3a, p.w1b, p.w2b, p.w3b);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  aclblasCsymm — complex (COMPLEX64) symmetric matrix multiply
//  C = alpha * A * B + beta * C   (LEFT)   /   alpha * B * A + beta * C (RIGHT)
//  A is symmetric (no conjugation).
//  Implemented as a self-contained operator (no dependency on other operators
//  in this repo): a small AIV prep kernel mirrors the symmetric A and
//  de-interleaves B into real/imag float planes, an AIC cube GEMM (own copy of
//  the proven FP32 MMAD kernel) computes the 4 real GEMMs, and an AIV combine
//  kernel merges them into complex C.
// ==========================================================================
extern "C" aclblasStatus_t aclblasCsymm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    int m, int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C, int ldc)
{
    OP_LOGI("aclblasCsymm", "entry: side=%d, uplo=%d, m=%d, n=%d, lda=%d, ldb=%d, ldc=%d",
            static_cast<int>(side), static_cast<int>(uplo), m, n, lda, ldb, ldc);

    aclblasStatus_t st = ValidateCsymmParams(handle, side, uplo, m, n, alpha, lda, ldb, beta, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;
    if (m == 0 || n == 0) return ACLBLAS_STATUS_SUCCESS;

    const float* af = reinterpret_cast<const float*>(alpha);
    const float* bf = reinterpret_cast<const float*>(beta);
    float alphaRe = af[0];
    float alphaIm = af[1];
    float betaRe = bf[0];
    float betaIm = bf[1];

    float alphaAbs = std::abs(alphaRe) + std::abs(alphaIm);
    float betaAbs = std::abs(betaRe) + std::abs(betaIm);
    int k = (side == ACLBLAS_SIDE_LEFT) ? m : n;

    st = ValidateCsymmPointers(k, alphaAbs, A, B, betaAbs, C);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    // BLAS: beta == 0 allows C to be NULL — nothing to compute or write.
    if (C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    aclrtStream stream = h->stream;

    // alpha == 0  ->  C = beta * C
    if (alphaAbs == 0.0f) {
        return CsymmRunAlphaZeroPath(stream, reinterpret_cast<uint8_t*>(C),
            m, n, ldc, betaRe, betaIm, CsymmAivCores());
    }

    uint32_t aivCoreNum = CsymmAivCores();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCsymm", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t cubeCoreNum = CsymmAicCores();
    if (cubeCoreNum == 0) {
        OP_LOGE("aclblasCsymm", "GetAicCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    int S = (side == ACLBLAS_SIDE_LEFT) ? m : n;  // symmetric dim

    // ---- small-shape single-launch direct path (m,n <= 16) ----
    // One AIV launch computes C = alpha*A*B + beta*C with on-the-fly symmetric
    // mirroring of A. It replaces the prep/gemm3/combine 3-launch pipeline,
    // whose fixed cost (~2.9us per launch) exceeds the whole task-book budget on
    // these shapes (thresholds 7.5/10/15us for 1..16/32/64, measured single-
    // launch floor ~7.7us). No workspace and no AIC cores are needed here.
    // Ultra-tiny straight-line AIV kernel (single launch). Sized >16 is slower
    // than the cube main path for these shapes (per-core full A-plane rebuild
    // dominates), so the AIV cut stays small.
    if (m <= 16 && n <= 16) {
        return CsymmRunSmallPath(stream, A, lda, uplo, B, ldb, C, ldc, m, n, S,
            (side == ACLBLAS_SIDE_LEFT) ? 0 : 1,
            alphaRe, alphaIm, betaRe, betaIm, aivCoreNum);
    }

    return CsymmRunMainCompute(h, stream, side, uplo, A, B, C, lda, ldb, ldc,
        m, n, S, alphaRe, alphaIm, betaRe, betaIm, aivCoreNum, cubeCoreNum);
}

#endif
