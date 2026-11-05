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
 * \file gemm_strided_batched_host.cpp
 * \brief Strided Batched GEMM (FP32) host-side implementation, arch35 (DAV_3510), Blaze tensor_api path.
 *
 * C_i = alpha * op(A_i) * op(B_i) + beta * C_i, i = 0..batchCount-1, column-major.
 * All batches share one tiling (identical m/n/k/trans/ld); only GM pointers advance by i*stride.
 * Batches are submitted serially and asynchronously on the handle stream (the caller synchronizes).
 * The row-major Blaze engine computes C^T = op(B)^T op(A)^T, so the GEMM operates on swapped operands
 * (left = B, right = A) with engine dimensions mEff = n, nEff = m, kEff = k (design section 4.3). Fast
 * path (alpha==1, beta==0, ldc%8==0): Fixpipe writes C_i directly. Otherwise the GEMM writes a compact
 * temp (row stride CeilAlign(m, 8)) and an AIV combine fuses alpha*temp + beta*C_i (design HIGH-1).
 */

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "gemm_strided_batched_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

namespace {
// FP32 Blaze GEMM tile constants (arch35), matching the arch35-validated strmm GEMM base.
constexpr uint32_t GEMM_SB_BASE_M = 16;
constexpr uint32_t GEMM_SB_BASE_N = 16;
constexpr uint32_t GEMM_SB_BASE_K = 8;
constexpr uint32_t GEMM_SB_FIXPIPE_N_ALIGN = 8;
constexpr uint32_t GEMM_SB_DEFAULT_TILE_M = 128;
constexpr uint32_t GEMM_SB_DEFAULT_TILE_N = 128;
constexpr uint32_t GEMM_SB_DEFAULT_TILE_K_CHUNK = 256;
constexpr uint32_t GEMM_SB_L1_SIZE_BYTES = 512 * 1024;
constexpr uint32_t GEMM_SB_COMBINE_TILE_LEN = 2048;
constexpr size_t GEMM_SB_FP32_BYTES = 4;
} // namespace

// ============================================================================
// Parameter validation
// ============================================================================

static aclblasStatus_t ValidateDimsAndTrans(
    aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int batchCount)
{
    CHECK_RET(
        transa == ACLBLAS_OP_N || transa == ACLBLAS_OP_T || transa == ACLBLAS_OP_C,
        OP_LOGE("aclblasSgemmStridedBatched", "invalid transa=%d", static_cast<int>(transa));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        transb == ACLBLAS_OP_N || transb == ACLBLAS_OP_T || transb == ACLBLAS_OP_C,
        OP_LOGE("aclblasSgemmStridedBatched", "invalid transb=%d", static_cast<int>(transb));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0, OP_LOGE("aclblasSgemmStridedBatched", "invalid m=%d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSgemmStridedBatched", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE("aclblasSgemmStridedBatched", "invalid k=%d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        batchCount >= 0, OP_LOGE("aclblasSgemmStridedBatched", "invalid batchCount=%d", batchCount);
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateLeadingDims(bool isTransA, bool isTransB, int m, int n, int k, int lda, int ldb, int ldc)
{
    int expectedLda = isTransA ? std::max(1, k) : std::max(1, m);
    CHECK_RET(
        lda >= expectedLda, OP_LOGE("aclblasSgemmStridedBatched", "invalid lda=%d, expected>=%d", lda, expectedLda);
        return ACLBLAS_STATUS_INVALID_VALUE);
    int expectedLdb = isTransB ? std::max(1, n) : std::max(1, k);
    CHECK_RET(
        ldb >= expectedLdb, OP_LOGE("aclblasSgemmStridedBatched", "invalid ldb=%d, expected>=%d", ldb, expectedLdb);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldc >= std::max(1, m), OP_LOGE("aclblasSgemmStridedBatched", "invalid ldc=%d, expected>=max(1,m=%d)", ldc, m);
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidatePointers(
    const void* alpha, const void* beta, int m, int n, int k, int batchCount, const void* A, const void* B,
    const void* C)
{
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasSgemmStridedBatched", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        beta != nullptr, OP_LOGE("aclblasSgemmStridedBatched", "beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE);
    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (k > 0) {
        CHECK_RET(
            A != nullptr, OP_LOGE("aclblasSgemmStridedBatched", "A must not be nullptr when k>0");
            return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(
            B != nullptr, OP_LOGE("aclblasSgemmStridedBatched", "B must not be nullptr when k>0");
            return ACLBLAS_STATUS_INVALID_VALUE);
    }
    const float betaVal = *static_cast<const float*>(beta);
    if (k > 0) {
        CHECK_RET(
            C != nullptr, OP_LOGE("aclblasSgemmStridedBatched", "C must not be nullptr when k>0");
            return ACLBLAS_STATUS_INVALID_VALUE);
    }
    CHECK_RET(
        C != nullptr || betaVal == 0.0f, OP_LOGE("aclblasSgemmStridedBatched", "C must not be nullptr when beta!=0");
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateStrides(int64_t strideA, int64_t strideB, int64_t strideC)
{
    CHECK_RET(
        strideA >= 0, OP_LOGE("aclblasSgemmStridedBatched", "invalid strideA=%ld, must be >= 0", strideA);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        strideB >= 0, OP_LOGE("aclblasSgemmStridedBatched", "invalid strideB=%ld, must be >= 0", strideB);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        strideC >= 0, OP_LOGE("aclblasSgemmStridedBatched", "invalid strideC=%ld, must be >= 0", strideC);
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateSgemmStridedBatchedParams(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda,
    int ldb, int ldc, const void* alpha, const void* beta, const void* A, const void* B, const void* C,
    int64_t strideA, int64_t strideB, int64_t strideC, int batchCount)
{
    CHECK_RET(
        handle != nullptr, OP_LOGE("aclblasSgemmStridedBatched", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    aclblasStatus_t st = ValidateDimsAndTrans(transa, transb, m, n, k, batchCount);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    bool isTransA = (transa != ACLBLAS_OP_N);
    bool isTransB = (transb != ACLBLAS_OP_N);
    st = ValidateLeadingDims(isTransA, isTransB, m, n, k, lda, ldb, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = ValidateStrides(strideA, strideB, strideC);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return ValidatePointers(alpha, beta, m, n, k, batchCount, A, B, C);
}

// ============================================================================
// Tiling computation (GEMM: swapped serpentine M/N tile split; combine: column partition)
// ============================================================================

// AIV block count for a column-partitioned Vector kernel over n columns of length m.
static uint32_t CalcVectorCores(uint32_t aivCoreNum, int n)
{
    uint32_t cores = std::min<uint32_t>(aivCoreNum, static_cast<uint32_t>(std::max(1, n)));
    return std::max<uint32_t>(cores, 1);
}

static void CalcGemmTileSizes(GemmSbGemmTilingData& tiling, uint32_t mEff, uint32_t nEff)
{
    uint32_t tileM = std::min<uint32_t>(GEMM_SB_DEFAULT_TILE_M, tiling.singleCoreM);
    uint32_t tileN = std::min<uint32_t>(GEMM_SB_DEFAULT_TILE_N, tiling.singleCoreN);
    uint32_t tileKChunk = GEMM_SB_DEFAULT_TILE_K_CHUNK;
    tileM = std::max<uint32_t>(tileM, GEMM_SB_BASE_M);
    tileN = std::max<uint32_t>(tileN, GEMM_SB_BASE_N);
    tileKChunk = std::max<uint32_t>(tileKChunk, GEMM_SB_BASE_K);
    if (mEff < GEMM_SB_DEFAULT_TILE_M) {
        tileM = std::min<uint32_t>(tiling.singleCoreM,
            CeilAlign<uint32_t>(std::max<uint32_t>(mEff, GEMM_SB_BASE_M), GEMM_SB_BASE_M));
    }
    if (nEff < GEMM_SB_DEFAULT_TILE_N) {
        tileN = std::min<uint32_t>(tiling.singleCoreN,
            CeilAlign<uint32_t>(std::max<uint32_t>(nEff, GEMM_SB_BASE_N), GEMM_SB_BASE_N));
    }
    uint32_t aSideL1 = CeilAlign<uint32_t>(tileM, GEMM_SB_BASE_M)
                     * CeilAlign<uint32_t>(tileKChunk, GEMM_SB_BASE_K) * GEMM_SB_FP32_BYTES;
    uint32_t bSideL1 = CeilAlign<uint32_t>(tileKChunk, GEMM_SB_BASE_K)
                     * CeilAlign<uint32_t>(tileN, GEMM_SB_BASE_N) * GEMM_SB_FP32_BYTES;
    while (2 * (aSideL1 + bSideL1) > GEMM_SB_L1_SIZE_BYTES && tileKChunk > GEMM_SB_BASE_K) {
        tileKChunk = CeilAlign<uint32_t>(std::max<uint32_t>(tileKChunk / 2, GEMM_SB_BASE_K), GEMM_SB_BASE_K);
        aSideL1 = CeilAlign<uint32_t>(tileM, GEMM_SB_BASE_M)
                * CeilAlign<uint32_t>(tileKChunk, GEMM_SB_BASE_K) * GEMM_SB_FP32_BYTES;
        bSideL1 = CeilAlign<uint32_t>(tileKChunk, GEMM_SB_BASE_K)
                * CeilAlign<uint32_t>(tileN, GEMM_SB_BASE_N) * GEMM_SB_FP32_BYTES;
    }
    tiling.tileM = tileM;
    tiling.tileN = tileN;
    tiling.tileKChunk = tileKChunk;
}

// Build the GEMM tiling in the swapped (C^T) coordinate system: mEff = n, nEff = m, kEff = k,
// ldLeft = original ldb, ldRight = original lda. ldC is set by the caller per path.
static GemmSbGemmTilingData CalcGemmTiling(
    uint32_t aicCoreNum, int m, int n, int k, int lda, int ldb, bool isTransA, bool isTransB)
{
    GemmSbGemmTilingData tiling{};
    uint32_t mEff = static_cast<uint32_t>(n);
    uint32_t nEff = static_cast<uint32_t>(m);
    tiling.mEff = mEff;
    tiling.nEff = nEff;
    tiling.kEff = static_cast<uint32_t>(k);
    tiling.ldLeft = static_cast<uint32_t>(ldb);
    tiling.ldRight = static_cast<uint32_t>(lda);
    tiling.transLeft = isTransB ? 1 : 0;
    tiling.transRight = isTransA ? 1 : 0;
    tiling.baseK = GEMM_SB_BASE_K;

    uint64_t tileCount = static_cast<uint64_t>(CeilDiv<uint32_t>(mEff, GEMM_SB_BASE_M))
                       * static_cast<uint64_t>(CeilDiv<uint32_t>(nEff, GEMM_SB_BASE_N));
    tiling.usedAicCoreNum = std::max<uint32_t>(std::min<uint64_t>(tileCount, aicCoreNum), 1);

    // Factor the core count into M and N directions so that each core handles one
    // large tile (divM × divN = usedAicCoreNum), maximizing L1 reuse.
    uint32_t cores = tiling.usedAicCoreNum;
    uint32_t coreMNum = 1;
    uint32_t coreNNum = cores;
    for (uint32_t d = 1; d <= cores; d++) {
        if (cores % d != 0)
            continue;
        uint32_t dn = cores / d;
        if (d > mEff || dn > nEff)
            continue;
        coreMNum = d;
        coreNNum = dn;
    }
    uint32_t coreM = CeilDiv<uint32_t>(mEff, coreMNum);
    uint32_t coreN = CeilDiv<uint32_t>(nEff, coreNNum);
    tiling.singleCoreM = std::max<uint32_t>(coreM, GEMM_SB_BASE_M);
    tiling.singleCoreN = std::max<uint32_t>(coreN, GEMM_SB_BASE_N);
    CalcGemmTileSizes(tiling, mEff, nEff);
    return tiling;
}

static void FillCombineTiling(
    GemmSbCombineTilingData& tiling, int m, int n, int ldc, uint32_t tempRowStride, float alpha, float beta,
    uint32_t usedAivCoreNum)
{
    // After column-major swap: temp is C^T(n×m), and we need to compute C = alpha*temp + beta*C.
    // Combine kernel processes C columnwise in the original (unswapped) user space:
    //   - C(m×n): m rows, n columns, ldc leading dimension
    //   - temp = C^T(n×m): n rows, m columns, stored with row stride = tempRowStride
    // So combine accesses temp[col * tempRowStride + row] where col=0..n-1, row=0..m-1.
    tiling.m = static_cast<uint32_t>(m);         // Original m (temp's column count)
    tiling.n = static_cast<uint32_t>(n);         // Original n (temp's row count after swap)
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.tempRowStride = tempRowStride;        // Aligned row stride of temp = max(CeilAlign(m,8), CeilAlign(n,8))
    tiling.alpha = alpha;
    tiling.beta = beta;
    tiling.hasBeta = (beta != 0.0f) ? 1 : 0;
    tiling.usedAivCoreNum = usedAivCoreNum;
    tiling.tileLen = GEMM_SB_COMBINE_TILE_LEN;
}

// ============================================================================
// Only-scale path (k=0 or alpha=0): C_i = beta * C_i, per batch, device-side async
// ============================================================================

static aclblasStatus_t ZeroAllBatch(
    _aclblas_handle* h, void* C, int m, int n, int ldc, int64_t strideC, int batchCount)
{
    if (C == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    // Clear only the logical m*n region (BLAS defines just op(C) = m x n); rows [m, ldc) hold user data.
    auto* cBase = reinterpret_cast<uint8_t*>(C);
    const size_t rowBytes = static_cast<size_t>(m) * GEMM_SB_FP32_BYTES;
    const int64_t colStrideBytes = static_cast<int64_t>(ldc) * static_cast<int64_t>(GEMM_SB_FP32_BYTES);
    for (int i = 0; i < batchCount; ++i) {
        int64_t batchByteOffset = static_cast<int64_t>(i) * strideC * static_cast<int64_t>(GEMM_SB_FP32_BYTES);
        if (ldc == m) {
            size_t blockBytes = static_cast<size_t>(m) * n * GEMM_SB_FP32_BYTES;
            aclError aclRet = aclrtMemsetAsync(cBase + batchByteOffset, blockBytes, 0, blockBytes, h->stream);
            CHECK_RET(aclRet == ACL_SUCCESS,
                OP_LOGE("aclblasSgemmStridedBatched", "aclrtMemsetAsync failed (beta=0), batch=%d, ret=%d", i, aclRet);
                return ACLBLAS_STATUS_INTERNAL_ERROR);
            continue;
        }
        for (int col = 0; col < n; ++col) {
            void* ci = cBase + batchByteOffset + static_cast<int64_t>(col) * colStrideBytes;
            aclError aclRet = aclrtMemsetAsync(ci, rowBytes, 0, rowBytes, h->stream);
            CHECK_RET(aclRet == ACL_SUCCESS,
                OP_LOGE("aclblasSgemmStridedBatched", "aclrtMemsetAsync failed (beta=0), batch=%d, col=%d, ret=%d",
                    i, col, aclRet);
                return ACLBLAS_STATUS_INTERNAL_ERROR);
        }
    }
    OP_LOGI("aclblasSgemmStridedBatched", "only-scale path beta=0, zeroed %d batches", batchCount);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t BetaScaleAllBatch(
    _aclblas_handle* h, void* C, int m, int n, int ldc, int64_t strideC, int batchCount, float betaVal)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgemmStridedBatched", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint32_t vecCores = CalcVectorCores(aivCoreNum, n);
    GemmSbCombineTilingData tiling{};
    FillCombineTiling(tiling, m, n, ldc, 0, 0.0f, betaVal, vecCores);
    auto* cBase = reinterpret_cast<uint8_t*>(C);
    OP_LOGI("aclblasSgemmStridedBatched", "only-scale path beta=%.4f, cores=%u, batches=%d", betaVal, vecCores,
            batchCount);
    for (int i = 0; i < batchCount; ++i) {
        int64_t elemOffset = static_cast<int64_t>(i) * strideC;
        auto* ci = reinterpret_cast<GM_ADDR>(cBase + elemOffset * static_cast<int64_t>(GEMM_SB_FP32_BYTES));
        gemm_sb_beta_scale_kernel_do(vecCores, h->stream, ci, tiling);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t HandleOnlyScaleAllBatch(
    _aclblas_handle* h, void* C, int m, int n, int ldc, int64_t strideC, int batchCount, float betaVal)
{
    if (betaVal == 1.0f) {
        OP_LOGI("aclblasSgemmStridedBatched", "only-scale path beta=1, C unchanged");
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (betaVal == 0.0f) {
        return ZeroAllBatch(h, C, m, n, ldc, strideC, batchCount);
    }
    return BetaScaleAllBatch(h, C, m, n, ldc, strideC, batchCount, betaVal);
}

// ============================================================================
// Fast/general path launch (k>0, alpha!=0): per-batch Blaze GEMM [+ AIV combine]
// ============================================================================

struct SgemmStridedBatchedLaunchPlan {
    GemmSbGemmTilingData gemmTiling; // fast path: launched as-is; temp path: template (mEff recomputed per strip)
    uint8_t* workspace;
    bool needTemp;
    uint32_t tempRowStride; // temp path GM row stride = CeilAlign(original m, 8)
    uint32_t nStrip;        // temp path: C columns processed per strip (temp reused across strips/batches)
    // Context to recompute per-strip GEMM/combine tiling in SubmitBatches (temp path only).
    float alphaVal;
    float betaVal;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    bool isTransA;
    bool isTransB;
    uint32_t aicCoreNum;
    uint32_t aivCoreNum;
};

static aclblasStatus_t BuildLaunchPlan(
    _aclblas_handle* h, bool isTransA, bool isTransB, float alphaVal, float betaVal, int m, int n, int k, int lda,
    int ldb, int ldc, uint32_t aicCoreNum, SgemmStridedBatchedLaunchPlan& plan)
{
    plan.workspace = nullptr;
    // Fast path only when alpha==1, beta==0, AND the user ldc is 8-aligned (Fixpipe N=8 requirement,
    // design MED-2). Otherwise a compact temp is written and the AIV combine (or copy) runs.
    bool fastDirect = (alphaVal == 1.0f) && (betaVal == 0.0f) && (ldc % GEMM_SB_FIXPIPE_N_ALIGN == 0);
    plan.needTemp = !fastDirect;

    // tempRowStride must satisfy: (1) >= CeilAlign(m,8) for Fixpipe alignment, (2) >= n for NDExtLayoutPtn(n, ldC)
    // to be legal when used as ldC in GEMM kernel, AND (3) itself be 8-aligned for Fixpipe N=8 requirement.
    // Use CeilAlign(n, 8) to guarantee (2)+(3), then max with CeilAlign(m,8) for (1).
    uint32_t tempRowStride = std::max(CeilAlign<uint32_t>(static_cast<uint32_t>(m), GEMM_SB_FIXPIPE_N_ALIGN),
                                       CeilAlign<uint32_t>(static_cast<uint32_t>(n), GEMM_SB_FIXPIPE_N_ALIGN));
    plan.tempRowStride = tempRowStride;

    plan.gemmTiling = CalcGemmTiling(aicCoreNum, m, n, k, lda, ldb, isTransA, isTransB);
    if (plan.gemmTiling.usedAicCoreNum == 0) {
        OP_LOGE("aclblasSgemmStridedBatched", "invalid tiling, usedAicCoreNum=0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    // ldC: fast path writes C_i directly (user ldc); temp paths write the compact aligned temp.
    plan.gemmTiling.ldC = plan.needTemp ? tempRowStride : static_cast<uint32_t>(ldc);

    if (plan.needTemp) {
        // Strip-mine over C columns so each strip's temp (curN * tempRowStride * 4 bytes) fits the
        // handle workspace. This bounds workspace usage to the default 4 MiB regardless of m/n, instead
        // of requiring the caller to inject a larger workspace for big shapes. temp is reused serially
        // across strips and batches (stream ordering serializes GEMM-write / combine-read).
        const size_t colBytes = static_cast<size_t>(tempRowStride) * GEMM_SB_FP32_BYTES;
        const size_t availBytes = GetEffectiveWorkspaceSize(h);
        if (availBytes < colBytes) {
            // A single C column already exceeds the workspace (pathologically large m). Fall back to the
            // conservative error and let the caller inject a larger workspace via aclblasSetWorkspace.
            OP_LOGE("aclblasSgemmStridedBatched",
                "workspace too small for one C column, required=%zu bytes per column", colBytes);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        plan.nStrip = static_cast<uint32_t>(
            std::min<size_t>(static_cast<size_t>(n), availBytes / colBytes));
        plan.workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
        uint32_t aivCoreNum = GetAivCoreCount();
        if (aivCoreNum == 0) {
            OP_LOGE("aclblasSgemmStridedBatched", "GetAivCoreCount failed");
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        plan.alphaVal = alphaVal;
        plan.betaVal = betaVal;
        plan.m = m;
        plan.n = n;
        plan.k = k;
        plan.lda = lda;
        plan.ldb = ldb;
        plan.ldc = ldc;
        plan.isTransA = isTransA;
        plan.isTransB = isTransB;
        plan.aicCoreNum = aicCoreNum;
        plan.aivCoreNum = aivCoreNum;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Temp path: per (batch, C-column strip) submit GEMM (writes curN rows of C^T into the shared temp) then
// combine (alpha*temp + beta*C over the same C columns). The engine mEff maps to the C-column dimension, so
// a strip over C columns is a strip over mEff: shrink mEff to curN, keep the temp base at workspace start,
// and advance the left (B) / output (C) GM bases by the strip's column offset.
static void SubmitBatchesTemp(
    _aclblas_handle* h, const SgemmStridedBatchedLaunchPlan& plan, const float* aBase, const float* bBase,
    float* cBase, int64_t strideA, int64_t strideB, int64_t strideC, int batchCount)
{
    auto gmTemp = reinterpret_cast<GM_ADDR>(plan.workspace);
    for (int i = 0; i < batchCount; ++i) {
        const float* ai = aBase + static_cast<int64_t>(i) * strideA;
        const float* bi = bBase + static_cast<int64_t>(i) * strideB;
        float* ci = cBase + static_cast<int64_t>(i) * strideC;
        for (uint32_t nOff = 0; nOff < static_cast<uint32_t>(plan.n); nOff += plan.nStrip) {
            uint32_t curN = std::min<uint32_t>(plan.nStrip, static_cast<uint32_t>(plan.n) - nOff);
            // GEMM tiling for this strip: engine mEff = curN (pass curN as the swap "n").
            GemmSbGemmTilingData gt = CalcGemmTiling(
                plan.aicCoreNum, plan.m, static_cast<int>(curN), plan.k, plan.lda, plan.ldb, plan.isTransA,
                plan.isTransB);
            gt.ldC = plan.tempRowStride;
            // Advance left (=B) by nOff columns of op(B): NoTrans B (k x n, ldb) -> nOff*ldb; Trans B
            // (n x k, ldb) -> nOff (op(B) column nOff is B row nOff, contiguous in column-major).
            int64_t bStripElemOff = plan.isTransB ? static_cast<int64_t>(nOff)
                                                  : static_cast<int64_t>(nOff) * static_cast<int64_t>(plan.ldb);
            const float* biStrip = bi + bStripElemOff;
            float* ciStrip = ci + static_cast<int64_t>(nOff) * static_cast<int64_t>(plan.ldc);
            auto gmLeft = reinterpret_cast<GM_ADDR>(const_cast<float*>(biStrip));
            auto gmRight = reinterpret_cast<GM_ADDR>(const_cast<float*>(ai));
            gemm_sb_gemm_kernel_do(gt.usedAicCoreNum, h->stream, gmLeft, gmRight, gmTemp, gt);

            // Combine strip: temp holds curN local columns (index 0..curN-1); C strip base already at
            // column nOff, so combine uses the same local column index for both temp and C.
            uint32_t vecCores = CalcVectorCores(plan.aivCoreNum, static_cast<int>(curN));
            GemmSbCombineTilingData ct{};
            FillCombineTiling(ct, plan.m, static_cast<int>(curN), plan.ldc, plan.tempRowStride, plan.alphaVal,
                plan.betaVal, vecCores);
            gemm_sb_combine_kernel_do(vecCores, h->stream, gmTemp, reinterpret_cast<GM_ADDR>(ciStrip),
                reinterpret_cast<GM_ADDR>(ciStrip), ct);
        }
    }
}

static void SubmitBatches(
    _aclblas_handle* h, const SgemmStridedBatchedLaunchPlan& plan, const void* A, const void* B, void* C,
    int64_t strideA, int64_t strideB, int64_t strideC, int batchCount)
{
    const auto* aBase = static_cast<const float*>(A);
    const auto* bBase = static_cast<const float*>(B);
    auto* cBase = static_cast<float*>(C);
    if (plan.needTemp) {
        SubmitBatchesTemp(h, plan, aBase, bBase, cBase, strideA, strideB, strideC, batchCount);
        return;
    }
    // Fast path: Blaze GEMM Fixpipe writes C_i directly (no workspace, no combine).
    for (int i = 0; i < batchCount; ++i) {
        const float* ai = aBase + static_cast<int64_t>(i) * strideA;
        const float* bi = bBase + static_cast<int64_t>(i) * strideB;
        float* ci = cBase + static_cast<int64_t>(i) * strideC;
        // Column-major swap: left = B_i, right = A_i.
        auto gmLeft = reinterpret_cast<GM_ADDR>(const_cast<float*>(bi));
        auto gmRight = reinterpret_cast<GM_ADDR>(const_cast<float*>(ai));
        gemm_sb_gemm_kernel_do(
            plan.gemmTiling.usedAicCoreNum, h->stream, gmLeft, gmRight, reinterpret_cast<GM_ADDR>(ci),
            plan.gemmTiling);
    }
}

static aclblasStatus_t LaunchSgemmStridedBatchedKernel(
    aclblasHandle_t handle, const void* A, const void* B, void* C, bool isTransA, bool isTransB, float alphaVal,
    float betaVal, int m, int n, int k, int lda, int ldb, int ldc, int64_t strideA, int64_t strideB, int64_t strideC,
    int batchCount, uint32_t aicCoreNum)
{
    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    SgemmStridedBatchedLaunchPlan plan{};
    aclblasStatus_t st = BuildLaunchPlan(
        h, isTransA, isTransB, alphaVal, betaVal, m, n, k, lda, ldb, ldc, aicCoreNum, plan);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    OP_LOGD(
        "aclblasSgemmStridedBatched",
        "gemm tiling: mEff=%u nEff=%u kEff=%u ldLeft=%u ldRight=%u ldC=%u transLeft=%u transRight=%u "
        "aicCores=%u singleCoreM=%u singleCoreN=%u tileM=%u tileN=%u tileKChunk=%u",
        plan.gemmTiling.mEff, plan.gemmTiling.nEff, plan.gemmTiling.kEff, plan.gemmTiling.ldLeft,
        plan.gemmTiling.ldRight, plan.gemmTiling.ldC, plan.gemmTiling.transLeft, plan.gemmTiling.transRight,
        plan.gemmTiling.usedAicCoreNum, plan.gemmTiling.singleCoreM, plan.gemmTiling.singleCoreN,
        plan.gemmTiling.tileM, plan.gemmTiling.tileN, plan.gemmTiling.tileKChunk);
    OP_LOGI(
        "aclblasSgemmStridedBatched",
        "launching: aicBlocks=%u, needTemp=%d, nStrip=%u, m=%d, n=%d, k=%d, ldc=%d, batchCount=%d",
        plan.gemmTiling.usedAicCoreNum, plan.needTemp ? 1 : 0, plan.needTemp ? plan.nStrip : 0, m, n, k, ldc,
        batchCount);
    SubmitBatches(h, plan, A, B, C, strideA, strideB, strideC, batchCount);
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
// Public API entry — dispatch only
// ============================================================================

aclblasStatus_t aclblasSgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda, int64_t strideA, const float* B, int ldb, int64_t strideB,
    const float* beta, float* C, int ldc, int64_t strideC, int batchCount)
{
    OP_LOGI(
        "aclblasSgemmStridedBatched",
        "entry: transA=%d, transB=%d, m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d, batchCount=%d",
        static_cast<int>(transA), static_cast<int>(transB), m, n, k, lda, ldb, ldc, batchCount);

    aclblasStatus_t st = ValidateSgemmStridedBatchedParams(
        handle, transA, transB, m, n, k, lda, ldb, ldc, alpha, beta, A, B, C, strideA, strideB, strideC, batchCount);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasSgemmStridedBatched", "parameter validation failed, status=%d", static_cast<int>(st));
        return st;
    }

    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = reinterpret_cast<_aclblas_handle*>(handle);
    const float alphaVal = *alpha;
    const float betaVal = *beta;

    // Only-scale path: k=0 or alpha=0 -> C_i = beta * C_i.
    if (k == 0 || alphaVal == 0.0f) {
        return HandleOnlyScaleAllBatch(h, C, m, n, ldc, strideC, batchCount, betaVal);
    }

    uint32_t aicCoreNum = GetAicCoreCount();
    if (aicCoreNum == 0) {
        OP_LOGE("aclblasSgemmStridedBatched", "GetAicCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    bool isTransA = (transA != ACLBLAS_OP_N);
    bool isTransB = (transB != ACLBLAS_OP_N);
    return LaunchSgemmStridedBatchedKernel(
        handle, A, B, C, isTransA, isTransB, alphaVal, betaVal, m, n, k, lda, ldb, ldc, strideA, strideB, strideC,
        batchCount, aicCoreNum);
}
