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
 * \file csyrkx_host.cpp
 * \brief aclblasCsyrkx host implementation for ascend950 (DAV_3510).
 *
 *        C = alpha * op(A) * op(B)^T + beta * C, uplo triangle only.
 *
 *        Phase 0 (AIV x 2): deinterleave complex A -> Ar, Ai and B -> Br, Bi
 *        Phase 1 (AIC x 4): four real GEMMs via gemm_kernel_do (4M decomposition)
 *          trans=N : t1=Ar*Br^T, t2=Ai*Bi^T, t3=Ar*Bi^T, t4=Ai*Br^T
 *          trans=T/C: t1=Ar^T*Br, t2=Ai^T*Bi, t3=Ar^T*Bi, t4=Ai^T*Br
 *          P_r = t1 - t2, P_i = t3 + t4
 *        Phase 2 (AIV): combine scale with complex alpha/beta:
 *          Cr = αr*P_r - αi*P_i, Ci = αr*P_i + αi*P_r
 *          C  = (Cr + i*Ci) + beta*C_old
 *
 *        OP_C maps to OP_T: syrkx yields a symmetric (not Hermitian) matrix,
 *        so no conjugation is applied (matches cublasCsyrkx).
 */

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "csyrkx_kernel.h"
#include "gemm/arch35/gemm_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/syrk_host_utils.h"

static const char* OP_TAG = "aclblasCsyrkx";

// ==========================================================================
//  Parameter validation
//  Order: handle -> enums -> dims -> leading dimensions -> scalar pointers
//  -> n == 0 quick return -> resolved scalars -> conditional matrix pointers.
//  Parameter checks precede the no-op decision (task doc §2.5).
// ==========================================================================
static aclblasStatus_t CheckCsyrkxEarlyParams(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k)
{
    CHECK_RET(handle != nullptr, OP_LOGE(OP_TAG, "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE(OP_TAG, "uplo must be UPPER or LOWER, got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE(OP_TAG, "trans must be OP_N, OP_T or OP_C, got %d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE(OP_TAG, "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateCsyrkxLeadingDims(aclblasOperation_t trans, int n, int k, int lda, int ldb, int ldc)
{
    const int minLdab = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    CHECK_RET(
        lda >= minLdab,
        OP_LOGE(OP_TAG, "lda must be >= max(1,%s), got lda=%d n=%d k=%d", (trans == ACLBLAS_OP_N) ? "n" : "k", lda, n, k);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldb >= minLdab,
        OP_LOGE(OP_TAG, "ldb must be >= max(1,%s), got ldb=%d n=%d k=%d", (trans == ACLBLAS_OP_N) ? "n" : "k", ldb, n, k);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldc >= std::max(1, n), OP_LOGE(OP_TAG, "ldc must be >= max(1,n), got ldc=%d n=%d", ldc, n);
        return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Scalar resolution: alpha and beta are complex and may live on host or
//  device (both must use the same location). Device scalars are read back over
//  the handle stream before any value-dependent branching.
// ==========================================================================
static aclblasStatus_t ReadCsyrkxScalarsFromDevice(
    const aclblasComplex* alpha, const aclblasComplex* beta, aclblasComplex& alphaVal, aclblasComplex& betaVal,
    aclrtStream stream)
{
    aclError aclRet =
        aclrtMemcpyAsync(&alphaVal, sizeof(aclblasComplex), alpha, sizeof(aclblasComplex), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "aclrtMemcpyAsync alpha D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtMemcpyAsync(&betaVal, sizeof(aclblasComplex), beta, sizeof(aclblasComplex), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "aclrtMemcpyAsync beta D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(OP_TAG, "aclrtSynchronizeStream for D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ResolveCsyrkxScalars(
    const aclblasComplex* alpha, const aclblasComplex* beta, aclblasComplex& alphaVal, aclblasComplex& betaVal,
    aclrtStream stream)
{
    bool alphaIsDevice = false;
    bool betaIsDevice = false;
    aclblasStatus_t st = CheckPtrLocation(alpha, &alphaIsDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "alpha pointer location query failed, st=%d", static_cast<int>(st));
        return st;
    }
    st = CheckPtrLocation(beta, &betaIsDevice);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "beta pointer location query failed, st=%d", static_cast<int>(st));
        return st;
    }
    if (alphaIsDevice != betaIsDevice) {
        OP_LOGE(OP_TAG, "alpha and beta must be both host or both device pointers");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alphaIsDevice) {
        return ReadCsyrkxScalarsFromDevice(alpha, beta, alphaVal, betaVal, stream);
    }
    alphaVal = *alpha;
    betaVal = *beta;
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  GEMM tiling: C is always n x n, so the column-major swap only flips the
//  transpose flags (the fixpipe output convention transposes the product).
// ==========================================================================
static GemmTilingData CalCsyrkxGemmTilingData(
    int n, int k, int lda, int ldb, int tempLdc, aclblasOperation_t transa, aclblasOperation_t transb)
{
    GemmTilingData tiling{};
    tiling.m = n;
    tiling.n = n;
    tiling.k = k;
    tiling.lda = lda;
    tiling.ldb = ldb;
    tiling.ldc = tempLdc;
    tiling.cLdc = tempLdc;
    tiling.baseM = GEMM_BASE_M;
    tiling.baseN = GEMM_BASE_N;
    tiling.baseK = GEMM_BASE_K;
    tiling.tileKChunk = GEMM_TILE_K_CHUNK;
    tiling.c0Size = GEMM_C0_SIZE;
    tiling.isTransA = (transa != ACLBLAS_OP_N) ? 1 : 0;
    tiling.isTransB = (transb != ACLBLAS_OP_N) ? 1 : 0;
    // All scaling is folded into the combine phase.
    tiling.alphaReal = 1.0f;
    tiling.alphaImag = 0.0f;
    tiling.betaReal = 0.0f;
    tiling.betaImag = 0.0f;
    tiling.hasBeta = 0;
    return tiling;
}

static void ApplyColMajorSwap(GemmTilingData& tiling)
{
    std::swap(tiling.m, tiling.n);
    std::swap(tiling.lda, tiling.ldb);
    std::swap(tiling.isTransA, tiling.isTransB);
}

static void CalcMultiCorePartition(GemmTilingData& tiling, uint32_t cubeCoreNum)
{
    int32_t maxCores = static_cast<int32_t>(cubeCoreNum);
    int32_t mTiles = (tiling.m + tiling.baseM - 1) / tiling.baseM;
    int32_t nTiles = (tiling.n + tiling.baseN - 1) / tiling.baseN;
    int32_t bestMBlocks = 1;
    int32_t bestNBlocks = 1;
    int32_t bestUtilization = 0;
    for (int32_t mb = 1; mb <= mTiles && mb <= maxCores; mb++) {
        int32_t nb = std::min(nTiles, maxCores / mb);
        if (nb < 1) {
            nb = 1;
        }
        int32_t utilization = mb * nb;
        if (utilization > bestUtilization && utilization <= maxCores) {
            bestUtilization = utilization;
            bestMBlocks = mb;
            bestNBlocks = nb;
        }
    }
    tiling.mBlocks = bestMBlocks;
    tiling.nBlocks = bestNBlocks;
    tiling.usedCoreNum = bestMBlocks * bestNBlocks;
}

static void PrepareCubeTiling(GemmTilingData& tiling, uint32_t cubeCoreNum)
{
    ApplyColMajorSwap(tiling);
    CalcMultiCorePartition(tiling, cubeCoreNum);
}

// ==========================================================================
//  Workspace: 4 temp slabs (n x tempLdc) + 4 real slabs (Ar, Ai, Br, Bi).
//  Skipped entirely when no GEMM runs (alpha == 0 or k == 0).
// ==========================================================================
struct CsyrkxWorkspace {
    uint32_t aicCoreNum = 0;
    uint32_t aivCoreNum = 0;
    uint32_t usedAivCoreNum = 0;
    uint32_t tempLdc = 0;
    uint32_t physRows = 0;
    uint32_t physCols = 0;
    size_t realBytes = 0;
    size_t tempBytes = 0;
};

static aclblasStatus_t PrepareCsyrkxWorkspace(
    _aclblas_handle* h, uint32_t n, uint32_t k, aclblasOperation_t trans, bool skipTemp, CsyrkxWorkspace& ws)
{
    ws.aicCoreNum = GetAicCoreCount();
    if (ws.aicCoreNum == 0) {
        OP_LOGE(OP_TAG, "GetAicCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ws.aivCoreNum = GetAivCoreCount();
    if (ws.aivCoreNum == 0) {
        OP_LOGE(OP_TAG, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    ws.usedAivCoreNum = GetUsedAivCoreNum(n, ws.aivCoreNum);
    ws.tempLdc = CeilAlign<uint32_t>(n, GEMM_FRACTAL);

    ws.physRows = (trans == ACLBLAS_OP_N) ? n : k;
    ws.physCols = (trans == ACLBLAS_OP_N) ? k : n;

    size_t workspaceNeed = 0;
    if (!skipTemp) {
        uint64_t realBytes = 0;
        uint64_t tempBytes = 0;
        uint64_t elemCount = 0;
        if (!SyrkMulU64(ws.physRows, ws.physCols, elemCount) ||
            !SyrkMulU64(elemCount, sizeof(float), realBytes) ||
            !SyrkMulU64(ws.tempLdc, n, elemCount) || !SyrkMulU64(elemCount, sizeof(float), tempBytes)) {
            OP_LOGE(OP_TAG, "workspace size overflow: n=%u k=%u", n, k);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        constexpr uint64_t kGroupCount = 4; // 4 real + 4 temp slabs
        uint64_t slabTotal = 0;
        uint64_t need = 0;
        if (!SyrkAddU64(realBytes, tempBytes, slabTotal) ||
            !SyrkMulU64(slabTotal, kGroupCount, need) ||
            need > static_cast<uint64_t>(ACLBLAS_MAX_WORKSPACE_SIZE)) {
            OP_LOGE(OP_TAG, "workspace need exceeds limit, realBytes=%llu tempBytes=%llu",
                static_cast<unsigned long long>(realBytes), static_cast<unsigned long long>(tempBytes));
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        workspaceNeed = static_cast<size_t>(need);
        ws.realBytes = static_cast<size_t>(realBytes);
        ws.tempBytes = static_cast<size_t>(tempBytes);
    }
    constexpr size_t kGmAlign = 512;
    workspaceNeed = (workspaceNeed + kGmAlign - 1) / kGmAlign * kGmAlign;

    aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, workspaceNeed);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_TAG, "workspace ensure failed, required=%zu, ret=%d", workspaceNeed, wsRet);
        return wsRet;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Phase 0 + 1: two deinterleave launches + four real GEMM launches.
//  Operand order follows the 950PR gemm convention: gemm_kernel_do(X, Y, T)
//  stores (X * Y)^T, hence the pairs below produce the logical products
//  documented in csyrkx_kernel.cpp.
// ==========================================================================
static void RunCsyrkxGemmPhase(
    _aclblas_handle* h, uint32_t n, const aclblasComplex* A, uint32_t lda, const aclblasComplex* B, uint32_t ldb,
    aclblasOperation_t trans, const CsyrkxWorkspace& ws, uint8_t* wsBase, uint8_t* dT1, uint8_t* dT2, uint8_t* dT3,
    uint8_t* dT4)
{
    aclrtStream stream = h->stream;

    uint8_t* dAr = wsBase + ws.tempBytes * 4;
    uint8_t* dAi = dAr + ws.realBytes;
    uint8_t* dBr = dAi + ws.realBytes;
    uint8_t* dBi = dBr + ws.realBytes;

    // Phase 0: deinterleave A and B (AIV)
    CsyrkxDeinterleaveTilingData deintTiling{};
    deintTiling.rows = ws.physRows;
    deintTiling.cols = ws.physCols;
    deintTiling.rowsPerCore = CeilDiv<uint32_t>(ws.physRows, ws.aivCoreNum);

    CsyrkxDeinterleaveTilingData deintTilingA = deintTiling;
    deintTilingA.lda = lda;
    csyrkx_deinterleave_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(A)), reinterpret_cast<GM_ADDR>(dAr),
        reinterpret_cast<GM_ADDR>(dAi), deintTilingA, ws.aivCoreNum, stream);

    CsyrkxDeinterleaveTilingData deintTilingB = deintTiling;
    deintTilingB.lda = ldb;
    csyrkx_deinterleave_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<aclblasComplex*>(B)), reinterpret_cast<GM_ADDR>(dBr),
        reinterpret_cast<GM_ADDR>(dBi), deintTilingB, ws.aivCoreNum, stream);

    // Phase 1: four real GEMMs (AIC). Ar/Ai/Br/Bi are tightly packed, so the
    // GEMM leading dimension is physRows for both operands.
    aclblasOperation_t transaGemm = (trans == ACLBLAS_OP_N) ? ACLBLAS_OP_N : ACLBLAS_OP_T;
    aclblasOperation_t transbGemm = (trans == ACLBLAS_OP_N) ? ACLBLAS_OP_T : ACLBLAS_OP_N;
    // Reduction dimension: trans=N -> k (physical cols), trans=T/C -> k
    // (physical rows of the k x n input). Both cases equal the original k.
    const uint32_t reduction = (trans == ACLBLAS_OP_N) ? ws.physCols : ws.physRows;
    GemmTilingData cubeTiling = CalCsyrkxGemmTilingData(
        static_cast<int>(n), static_cast<int>(reduction), static_cast<int>(ws.physRows),
        static_cast<int>(ws.physRows), static_cast<int>(ws.tempLdc), transaGemm, transbGemm);
    PrepareCubeTiling(cubeTiling, ws.aicCoreNum);
    cubeTiling.ldc = static_cast<int32_t>(ws.tempLdc);

    const uint32_t numBlocks = static_cast<uint32_t>(cubeTiling.usedCoreNum);
    OP_LOGD(OP_TAG, "launching 4 gemm kernels: aicBlocks=%u", numBlocks);

    // t1/t2 are identical on both trans paths; only t3/t4 swap operands.
    gemm_kernel_do(numBlocks, stream, dBr, dAr, dT1, cubeTiling); // t1: Ar*Br^T | Ar^T*Br
    gemm_kernel_do(numBlocks, stream, dBi, dAi, dT2, cubeTiling); // t2: Ai*Bi^T | Ai^T*Bi
    if (trans == ACLBLAS_OP_N) {
        gemm_kernel_do(numBlocks, stream, dBr, dAi, dT3, cubeTiling); // t3: Ai*Br^T
        gemm_kernel_do(numBlocks, stream, dBi, dAr, dT4, cubeTiling); // t4: Ar*Bi^T
    } else {
        gemm_kernel_do(numBlocks, stream, dBi, dAr, dT3, cubeTiling); // t3: Ar^T*Bi
        gemm_kernel_do(numBlocks, stream, dBr, dAi, dT4, cubeTiling); // t4: Ai^T*Br
    }
}

// ==========================================================================
//  Launch pipeline
// ==========================================================================
static aclblasStatus_t LaunchCsyrkxKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k, float alphaReal,
    float alphaImag, float betaReal, float betaImag, const aclblasComplex* A, uint32_t lda, const aclblasComplex* B,
    uint32_t ldb, aclblasComplex* C, uint32_t ldc)
{
    // OP_C is equivalent to OP_T for syrkx (no conjugation).
    if (trans == ACLBLAS_OP_C) {
        trans = ACLBLAS_OP_T;
    }

    const bool isAlphaZero = (alphaReal == 0.0f && alphaImag == 0.0f);
    const bool isKZero = (k == 0);
    const bool isBetaZero = (betaReal == 0.0f && betaImag == 0.0f);
    const bool skipTemp = isAlphaZero || isKZero;

    OP_LOGD(
        OP_TAG, "alpha=(%.6f,%.6f) beta=(%.6f,%.6f) isAlphaZero=%d isKZero=%d isBetaZero=%d", alphaReal, alphaImag,
        betaReal, betaImag, isAlphaZero, isKZero, isBetaZero);

    CsyrkxWorkspace ws;
    aclblasStatus_t st = PrepareCsyrkxWorkspace(h, n, k, trans, skipTemp, ws);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint8_t* dT1 = nullptr;
    uint8_t* dT2 = nullptr;
    uint8_t* dT3 = nullptr;
    uint8_t* dT4 = nullptr;
    if (!skipTemp) {
        uint8_t* wsBase = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
        dT1 = wsBase;
        dT2 = dT1 + ws.tempBytes;
        dT3 = dT2 + ws.tempBytes;
        dT4 = dT3 + ws.tempBytes;
        RunCsyrkxGemmPhase(h, n, A, lda, B, ldb, trans, ws, wsBase, dT1, dT2, dT3, dT4);
    }

    CsyrkxCombineTilingData combineTiling{};
    combineTiling.n = n;
    combineTiling.ldc = ldc;
    combineTiling.tempLdc = ws.tempLdc;
    combineTiling.rowsPerCore = CeilDiv<uint32_t>(n, ws.usedAivCoreNum);
    combineTiling.alphaReal = alphaReal;
    combineTiling.alphaImag = alphaImag;
    combineTiling.betaReal = betaReal;
    combineTiling.betaImag = betaImag;
    combineTiling.uploMode = static_cast<uint8_t>(uplo);
    combineTiling.isAlphaZero = static_cast<uint8_t>(isAlphaZero ? 1 : 0);
    combineTiling.isKZero = static_cast<uint8_t>(isKZero ? 1 : 0);
    combineTiling.isBetaZero = static_cast<uint8_t>(isBetaZero ? 1 : 0);

    OP_LOGI(OP_TAG, "launching combine kernel: n=%u aivCores=%u", n, ws.usedAivCoreNum);
    csyrkx_combine_kernel_do(
        reinterpret_cast<GM_ADDR>(dT1), reinterpret_cast<GM_ADDR>(dT2), reinterpret_cast<GM_ADDR>(dT3),
        reinterpret_cast<GM_ADDR>(dT4), reinterpret_cast<GM_ADDR>(C), combineTiling, ws.usedAivCoreNum, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
//  Public API entry
// ==========================================================================
extern "C" aclblasStatus_t aclblasCsyrkx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C,
    int ldc)
{
    // 1. handle / enums / non-negative dimensions
    aclblasStatus_t st = CheckCsyrkxEarlyParams(handle, uplo, trans, n, k);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // 2. Leading dimensions (checked before the n == 0 no-op decision)
    st = ValidateCsyrkxLeadingDims(trans, n, k, lda, ldb, ldc);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // 3. Scalar pointers
    CHECK_RET(alpha != nullptr, OP_LOGE(OP_TAG, "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE(OP_TAG, "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);

    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = static_cast<_aclblas_handle*>(handle);

    // 4. Resolve scalar true values (host dereference or device D2H read-back)
    aclblasComplex alphaVal = {0.0f, 0.0f};
    aclblasComplex betaVal = {0.0f, 0.0f};
    st = ResolveCsyrkxScalars(alpha, beta, alphaVal, betaVal, h->stream);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // 5. Conditional matrix pointer checks on the resolved values.
    //    alpha == 0 or k == 0 leaves A/B unreferenced; C is the output and must
    //    be valid whenever n > 0.
    const bool isAlphaZero = (alphaVal.real == 0.0f && alphaVal.imag == 0.0f);
    if (!isAlphaZero && k > 0) {
        CHECK_RET(A != nullptr, OP_LOGE(OP_TAG, "A must not be nullptr when alpha != 0 and k > 0"); return ACLBLAS_STATUS_INVALID_VALUE);
        CHECK_RET(B != nullptr, OP_LOGE(OP_TAG, "B must not be nullptr when alpha != 0 and k > 0"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    CHECK_RET(C != nullptr, OP_LOGE(OP_TAG, "C must not be nullptr when n > 0"); return ACLBLAS_STATUS_INVALID_VALUE);

    // 6. Pipeline
    return LaunchCsyrkxKernel(
        h, uplo, trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k), alphaVal.real, alphaVal.imag, betaVal.real,
        betaVal.imag, A, static_cast<uint32_t>(lda), B, static_cast<uint32_t>(ldb), C, static_cast<uint32_t>(ldc));
}
