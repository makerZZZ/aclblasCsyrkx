/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "strsm_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/devkit_version_compat.h"

namespace {
constexpr uint32_t SIMT_OPTIMAL_THREADS = 64;
constexpr uint32_t SINGLE_CORE_N_THRESHOLD = 1;
constexpr uint32_t AUX_MIN_ELEMS_PER_BLOCK = 1024;
constexpr uint32_t BLOCKED_THRESHOLD = 128;
constexpr uint32_t SIMT_BLOCKED_N_THRESHOLD = 256;
constexpr uint32_t GEMM_L1_SIZE_BYTES = 512 * 1024;
constexpr uint32_t GEMM_L1_BUF_NUM = 2;
constexpr uint32_t GEMM_FRACTAL = 16;
constexpr uint32_t GEMM_FP32_C0 = 8;
// Reserve 10% L1 headroom for NDExtLayout padding and other non-accounted overhead.
constexpr uint32_t GEMM_L1_SAFE_HEADROOM_PERCENT = 10;
constexpr uint32_t GEMM_L1_FULL_PERCENT = 100;

static uint32_t CalcAuxNumBlocks(uint64_t totalElems, uint32_t aivCoreNum)
{
    if (totalElems == 0) return 1;
    uint64_t blocks = (totalElems + AUX_MIN_ELEMS_PER_BLOCK - 1) / AUX_MIN_ELEMS_PER_BLOCK;
    if (blocks > aivCoreNum) blocks = aivCoreNum;
    if (blocks == 0) blocks = 1;
    return static_cast<uint32_t>(blocks);
}

StrsmTilingData CalcTiling(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, int lda, int ldb,
    float alpha, uint32_t coreNum)
{
    StrsmTilingData tiling{};
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.trans = static_cast<uint32_t>(trans);
    tiling.diag = static_cast<uint32_t>(diag);
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<int32_t>(lda);
    tiling.ldb = static_cast<int32_t>(ldb);
    tiling.alpha = alpha;
    uint32_t numThreads = 8;
    uint32_t mU32 = static_cast<uint32_t>(m);
    while (numThreads < mU32 && numThreads < SIMT_OPTIMAL_THREADS)
        numThreads *= 2;
    tiling.numThreads = numThreads;
    uint32_t nU32 = static_cast<uint32_t>(n);
    uint32_t useCores = (nU32 <= SINGLE_CORE_N_THRESHOLD) ? 1 : ((coreNum > 0) ? std::min(coreNum, nU32) : 1);
    tiling.coreNum = useCores;
    tiling.perCoreN = nU32 / useCores;
    tiling.coreRemainder = nU32 % useCores;
    return tiling;
}
} // namespace

void strsm_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* workSpace, const StrsmTilingData& tiling, uint32_t numBlocks, void* stream);
void strsm_right_kernel_do(
    uint8_t* a, uint8_t* b, uint8_t* workSpace, const StrsmTilingData& tiling, uint32_t numBlocks, void* stream);
void strsm_zero_kernel_do(uint8_t* b, uint32_t m, uint32_t n, int32_t ldb, uint32_t numBlocks, void* stream);
void strsm_transpose_kernel_do(
    uint8_t* in, uint8_t* out, uint32_t rows, uint32_t cols, int32_t ldIn, int32_t ldOut,
    uint32_t numBlocks, void* stream);

#if ASC_DEVKIT_GE_9_1
void strsm_panel_kernel_do(uint8_t* a, uint8_t* b, const StrsmPanelTilingData& tiling, uint32_t numBlocks, void* stream);
void strsm_gemm_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* temp, const StrsmGemmTilingData& tiling, uint32_t numBlocks, void* stream);
void strsm_axpy_kernel_do(
    uint8_t* b, uint8_t* temp, const StrsmAxpyTilingData& tiling, uint32_t numBlocks, void* stream);
void strsm_scale_kernel_do(
    uint8_t* b, float alpha, uint32_t m, uint32_t n, uint32_t ldb, uint32_t numBlocks, void* stream);
void strsm_extract_a_kernel_do(
    uint8_t* a, uint8_t* ws, uint32_t mC, uint32_t bs, uint32_t aWsStride, uint32_t lda, uint64_t aOffset, uint32_t transA,
    uint32_t numBlocks, void* stream);
void strsm_extract_b_kernel_do(
    uint8_t* b, uint8_t* ws, uint32_t bs, uint32_t n, uint32_t bWsStride, uint32_t ldb, uint64_t bOffset,
    uint32_t numBlocks, void* stream);
void strsm_axpy_trans_kernel_do(uint8_t* b, uint8_t* temp,
    const StrsmAxpyTilingData& tiling, uint32_t numBlocks, void* stream);
#endif

#if ASC_DEVKIT_GE_9_1
static uint32_t ChooseTileK(uint32_t k)
{
    if (k <= 8) return std::max<uint32_t>(k, 8);
    if (k <= 32) return 32;
    if (k <= 64) return 64;
    return 128;
}

// Max tileK in {128,64,32,16,8} that fits L1 double-buffer for given (tileM, tileN).
// Returns 0 if even tileK=8 overflows (candidate is invalid).
static uint32_t MaxTileKByL1(uint32_t tileM, uint32_t tileN)
{
    const uint32_t candidates[] = {128, 64, 32, 16, 8};
    constexpr uint32_t numCandidates = sizeof(candidates) / sizeof(candidates[0]);
    for (uint32_t i = 0; i < numCandidates; i++) {
        uint32_t tk = candidates[i];
        uint32_t aSide = CeilAlign<uint32_t>(tileM, GEMM_FRACTAL)
            * CeilAlign<uint32_t>(tk, GEMM_FP32_C0) * sizeof(float);
        uint32_t bSide = CeilAlign<uint32_t>(tk, GEMM_FP32_C0)
            * CeilAlign<uint32_t>(tileN, GEMM_FRACTAL) * sizeof(float);
        if (GEMM_L1_BUF_NUM * (aSide + bSide) <=
            GEMM_L1_SIZE_BYTES * (GEMM_L1_FULL_PERCENT - GEMM_L1_SAFE_HEADROOM_PERCENT) / GEMM_L1_FULL_PERCENT) {
            return tk;
        }
    }
    return 0;
}

static void RelaxTileForOverfill(uint32_t gemmM, uint32_t gemmN, uint32_t aicCoreNum,
    uint32_t& bestM, uint32_t& bestN)
{
    const uint32_t candidates[] = {256, 128, 64, 32, 16};
    constexpr uint32_t numCandidates = sizeof(candidates) / sizeof(candidates[0]);
    uint64_t curTiles = CeilDiv<uint64_t>(gemmM, bestM) * CeilDiv<uint64_t>(gemmN, bestN);
    if (curTiles <= aicCoreNum) {
        return;
    }
    uint32_t halfThresh = aicCoreNum / 2;
    uint64_t curProduct = static_cast<uint64_t>(bestM) * bestN;
    for (uint32_t i = 0; i < numCandidates; i++) {
        for (uint32_t j = 0; j < numCandidates; j++) {
            uint32_t tm = candidates[i];
            uint32_t tn = candidates[j];
            if (MaxTileKByL1(tm, tn) == 0) {
                continue;
            }
            uint64_t tiles = CeilDiv<uint64_t>(gemmM, tm) * CeilDiv<uint64_t>(gemmN, tn);
            if (tiles >= halfThresh && tiles <= aicCoreNum) {
                uint64_t product = static_cast<uint64_t>(tm) * tn;
                if (product > curProduct) {
                    bestM = tm;
                    bestN = tn;
                    curProduct = product;
                }
            }
        }
    }
}

static void ChooseGemmTileMN(uint32_t gemmM, uint32_t gemmN, uint32_t aicCoreNum,
    uint32_t& tileM, uint32_t& tileN)
{
    const uint32_t candidates[] = {256, 128, 64, 32, 16};
    constexpr uint32_t numCandidates = sizeof(candidates) / sizeof(candidates[0]);
    uint32_t bestM = 16;
    uint32_t bestN = 16;
    uint64_t bestTiles = 0;
    bool found = false;
    for (uint32_t i = 0; i < numCandidates; i++) {
        for (uint32_t j = 0; j < numCandidates; j++) {
            uint32_t tm = candidates[i];
            uint32_t tn = candidates[j];
            if (MaxTileKByL1(tm, tn) == 0) {
                continue;
            }
            uint64_t tiles = CeilDiv<uint64_t>(gemmM, tm) * CeilDiv<uint64_t>(gemmN, tn);
            if (tiles >= aicCoreNum) {
                if (!found || static_cast<uint64_t>(tm) * tn > static_cast<uint64_t>(bestM) * bestN) {
                    bestM = tm;
                    bestN = tn;
                    found = true;
                }
            } else if (!found && tiles > bestTiles) {
                bestTiles = tiles;
                bestM = tm;
                bestN = tn;
            }
        }
    }

    if (found) {
        RelaxTileForOverfill(gemmM, gemmN, aicCoreNum, bestM, bestN);
    }

    tileM = bestM;
    tileN = bestN;
}

static StrsmGemmTilingData CalcGemmTiling(uint32_t gemmM, uint32_t gemmN, uint32_t bs,
    uint32_t lda, uint32_t ldb, uint64_t aOffset, uint64_t bOffset, uint32_t tempRowStride,
    uint32_t aicCoreNum)
{
    StrsmGemmTilingData t{};
    t.m = gemmM;
    t.n = gemmN;
    t.k = bs;
    t.lda = lda;
    t.ldb = ldb;
    t.aOffset = aOffset;
    t.bOffset = bOffset;
    ChooseGemmTileMN(gemmM, gemmN, aicCoreNum, t.tileM, t.tileN);
    uint32_t maxKByL1 = MaxTileKByL1(t.tileM, t.tileN);
    t.tileKChunk = ChooseTileK(std::min<uint32_t>(maxKByL1, bs));
    t.tempRowStride = tempRowStride;
    return t;
}
#endif


static aclblasStatus_t StrsmAlphaZeroPath(_aclblas_handle* h, int m, int n, int ldb, float* B, uint32_t aivCoreNum)
{
    uint32_t nU32 = static_cast<uint32_t>(n);
    uint32_t numBlocks = std::min(aivCoreNum, nU32);
    if (numBlocks == 0) numBlocks = 1;
    OP_LOGI("aclblasStrsm", "launching strsm_zero_kernel: m=%d, n=%d, blocks=%u", m, n, numBlocks);
    strsm_zero_kernel_do(reinterpret_cast<uint8_t*>(B), static_cast<uint32_t>(m), nU32,
        static_cast<int32_t>(ldb), numBlocks, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

static size_t CalcRightBlockedWorkspace(uint32_t mU32, uint32_t nU32, uint32_t panelBs, bool isTransAfter)
{
    uint32_t noTransTempStride = CeilAlign<uint32_t>(nU32, 8u);
    uint32_t transTempStride = CeilAlign<uint32_t>(mU32, 8u);
    size_t blockedTempWs = isTransAfter ? static_cast<size_t>(nU32) * transTempStride * sizeof(float) :
                                          static_cast<size_t>(mU32) * noTransTempStride * sizeof(float);
    size_t aWs = static_cast<size_t>(nU32) * panelBs * sizeof(float);
    uint32_t mAligned = CeilAlign<uint32_t>(mU32, 8u);
    size_t bWs = static_cast<size_t>(panelBs) * mAligned * sizeof(float);
    return isTransAfter ? (blockedTempWs + aWs + bWs) : blockedTempWs;
}

static aclblasStatus_t StrsmLeftDispatch(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, int lda,
    int ldb, float alpha, const float* A, float* B, uint32_t aivCoreNum);

static aclblasStatus_t StrsmRightDevicePath(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, int lda,
    int ldb, float alpha, const float* A, float* B, uint32_t aivCoreNum)
{
    uint32_t mU32 = static_cast<uint32_t>(m);
    uint32_t nU32 = static_cast<uint32_t>(n);
    int newLdb = std::max(1, n);
    size_t btBytes = static_cast<size_t>(newLdb) * static_cast<size_t>(m) * sizeof(float);
    bool isTransAfter = (trans == ACLBLAS_OP_N);
    size_t blockedWsMax = CalcRightBlockedWorkspace(mU32, nU32, 128u, isTransAfter);
    size_t totalWsMax = blockedWsMax + btBytes;
    if (totalWsMax > GetEffectiveWorkspaceSize(h)) {
        aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, totalWsMax);
        if (wsRet != ACLBLAS_STATUS_SUCCESS) {
            return wsRet;
        }
    }
    uint8_t* wsBase = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    uint8_t* btDev = wsBase + blockedWsMax;

    uint32_t transBlocks = std::min(aivCoreNum, mU32);
    if (transBlocks == 0) {
        transBlocks = 1;
    }
    OP_LOGI("aclblasStrsm", "launching transpose kernel (B->Bt): m=%d, n=%d, blocks=%u", m, n, transBlocks);
    strsm_transpose_kernel_do(reinterpret_cast<uint8_t*>(B), btDev, mU32, nU32, static_cast<int32_t>(ldb),
        static_cast<int32_t>(newLdb), transBlocks, h->stream);

    aclblasOperation_t fTrans = (trans == ACLBLAS_OP_N) ? ACLBLAS_OP_T : ACLBLAS_OP_N;
    if (mU32 >= SIMT_BLOCKED_N_THRESHOLD) {
        OP_LOGI("aclblasStrsm", "right path: routing to left blocked dispatch (m=%d, n=%d)", m, n);
        aclblasStatus_t solveRet = StrsmLeftDispatch(h, uplo, fTrans, diag, n, m, lda, newLdb, alpha, A,
            reinterpret_cast<float*>(btDev), aivCoreNum);
        if (solveRet != ACLBLAS_STATUS_SUCCESS) {
            return solveRet;
        }
    } else {
        StrsmTilingData tiling = CalcTiling(uplo, fTrans, diag, n, m, lda, newLdb, alpha, aivCoreNum);
        OP_LOGI("aclblasStrsm", "launching kernel(right): blocks=%u, cores=%u", tiling.coreNum, aivCoreNum);
        strsm_kernel_do(reinterpret_cast<uint8_t*>(const_cast<float*>(A)), btDev, nullptr, tiling, tiling.coreNum,
            h->stream);
    }

    uint32_t transBackBlocks = std::min(aivCoreNum, nU32);
    if (transBackBlocks == 0) {
        transBackBlocks = 1;
    }
    OP_LOGI("aclblasStrsm", "launching transpose kernel (Bt->B): n=%d, m=%d, blocks=%u", n, m, transBackBlocks);
    strsm_transpose_kernel_do(btDev, reinterpret_cast<uint8_t*>(B), nU32, mU32, static_cast<int32_t>(newLdb),
        static_cast<int32_t>(ldb), transBackBlocks, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t StrsmRightSimtPath(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, int lda,
    int ldb, float alpha, const float* A, float* B, uint32_t aivCoreNum)
{
    aclblasOperation_t fTrans = (trans == ACLBLAS_OP_N) ? ACLBLAS_OP_T : ACLBLAS_OP_N;
    StrsmTilingData tiling = CalcTiling(uplo, fTrans, diag, n, m, lda, ldb, alpha, aivCoreNum);
    OP_LOGI("aclblasStrsm", "launching right kernel(simt): mEff=%u nEff=%u cores=%u", tiling.m, tiling.n, aivCoreNum);
    strsm_right_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(B), nullptr, tiling,
        tiling.coreNum, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t StrsmLeftSimtPath(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, int lda,
    int ldb, float alpha, const float* A, float* B, uint32_t aivCoreNum)
{
    StrsmTilingData tiling = CalcTiling(uplo, trans, diag, m, n, lda, ldb, alpha, aivCoreNum);
    OP_LOGI("aclblasStrsm", "launching kernel(simt): blocks=%u, cores=%u", tiling.coreNum, aivCoreNum);
    strsm_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(B), nullptr, tiling,
        tiling.coreNum, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

#if ASC_DEVKIT_GE_9_1
static uint32_t ChoosePanelSize(uint32_t m)
{
    if (m <= 64) return 64;
    return 128;
}

static void StrsmTrailingUpdateNoTrans(_aclblas_handle* h, uint32_t nU32, uint32_t k, uint32_t bs,
    uint32_t mC, uint64_t aOffset, uint64_t axpyOffset, uint32_t ldaU32, uint32_t ldbU32,
    uint32_t noTransTempStride, uint32_t aicCoreNum, uint32_t aivCoreNum,
    const float* A, float* B, uint8_t* tempDev)
{
    StrsmGemmTilingData gt = CalcGemmTiling(nU32, mC, bs, ldbU32, ldaU32, k, aOffset,
        noTransTempStride, aicCoreNum);
    uint64_t gemmTileCount = static_cast<uint64_t>((nU32 + gt.tileM - 1) / gt.tileM)
                           * static_cast<uint64_t>((mC + gt.tileN - 1) / gt.tileN);
    uint32_t gemmBlocks = static_cast<uint32_t>(std::min<uint64_t>(gemmTileCount, aicCoreNum));
    if (gemmBlocks == 0) gemmBlocks = 1;
    strsm_gemm_kernel_do(reinterpret_cast<uint8_t*>(B), reinterpret_cast<uint8_t*>(const_cast<float*>(A)),
        tempDev, gt, gemmBlocks, h->stream);
    StrsmAxpyTilingData at{};
    at.m = mC;
    at.n = nU32;
    at.ldb = ldbU32;
    at.tempRowStride = noTransTempStride;
    at.bOffset = axpyOffset;
    strsm_axpy_trans_kernel_do(reinterpret_cast<uint8_t*>(B), tempDev, at,
        CalcAuxNumBlocks(static_cast<uint64_t>(mC) * nU32, aivCoreNum), h->stream);
}

static void StrsmTrailingUpdateTrans(_aclblas_handle* h, uint32_t nU32, uint32_t k, uint32_t bs,
    uint32_t mC, uint64_t aOffset, uint64_t axpyOffset, uint32_t ldaU32, uint32_t ldbU32,
    uint32_t transTempStride, uint32_t nAligned, uint32_t aicCoreNum, uint32_t aivCoreNum,
    const float* A, float* B, uint8_t* tempDev, uint8_t* aWsDev, uint8_t* bWsDev)
{
    uint32_t bsAligned = CeilAlign<uint32_t>(bs, 8u);
    strsm_extract_a_kernel_do(reinterpret_cast<uint8_t*>(const_cast<float*>(A)), aWsDev,
        mC, bs, bsAligned, ldaU32, aOffset, 1,
        CalcAuxNumBlocks(static_cast<uint64_t>(mC) * bs, aivCoreNum), h->stream);
    strsm_extract_b_kernel_do(reinterpret_cast<uint8_t*>(B), bWsDev,
        bs, nU32, nAligned, ldbU32, k,
        CalcAuxNumBlocks(static_cast<uint64_t>(bs) * nU32, aivCoreNum), h->stream);
    StrsmGemmTilingData gt = CalcGemmTiling(mC, nU32, bs, bsAligned, nAligned, 0, 0,
        transTempStride, aicCoreNum);
    uint64_t gemmTileCount = static_cast<uint64_t>((mC + gt.tileM - 1) / gt.tileM)
                           * static_cast<uint64_t>((nU32 + gt.tileN - 1) / gt.tileN);
    uint32_t gemmBlocks = static_cast<uint32_t>(std::min<uint64_t>(gemmTileCount, aicCoreNum));
    if (gemmBlocks == 0) gemmBlocks = 1;
    strsm_gemm_kernel_do(aWsDev, bWsDev, tempDev, gt, gemmBlocks, h->stream);
    StrsmAxpyTilingData at{};
    at.m = mC;
    at.n = nU32;
    at.ldb = ldbU32;
    at.tempRowStride = gt.tempRowStride;
    at.bOffset = axpyOffset;
    strsm_axpy_kernel_do(reinterpret_cast<uint8_t*>(B), tempDev, at,
        CalcAuxNumBlocks(static_cast<uint64_t>(mC) * nU32, aivCoreNum), h->stream);
}

struct StrsmBlockedCtx {
    uint32_t mU32;
    uint32_t nU32;
    uint32_t ldaU32;
    uint32_t ldbU32;
    uint32_t panelBs;
    uint32_t aicCoreNum;
    uint32_t noTransTempStride;
    uint32_t transTempStride;
    uint32_t nAligned;
    uint8_t* tempDev;
    uint8_t* aWsDev;
    uint8_t* bWsDev;
    bool isUpper;
    bool isTrans;
};

static void StrsmBlockedStep(_aclblas_handle* h, const StrsmBlockedCtx& ctx, uint32_t step, uint32_t numSteps,
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int lda, int ldb,
    const float* A, float* B, uint32_t aivCoreNum)
{
    uint32_t k, bs;
    bool forward = (!ctx.isTrans && !ctx.isUpper) || (ctx.isTrans && ctx.isUpper);
    k = forward ? step * ctx.panelBs : (numSteps - 1 - step) * ctx.panelBs;
    bs = std::min(ctx.panelBs, ctx.mU32 - k);

    StrsmPanelTilingData pt{};
    pt.uplo = static_cast<uint32_t>(uplo);
    pt.trans = static_cast<uint32_t>(trans);
    pt.diag = static_cast<uint32_t>(diag);
    pt.m = ctx.mU32;
    pt.n = ctx.nU32;
    pt.lda = static_cast<int32_t>(lda);
    pt.ldb = static_cast<int32_t>(ldb);
    pt.panelStart = k;
    pt.panelSize = bs;
    uint32_t panelBlocks = std::min(aivCoreNum, ctx.nU32);
    if (panelBlocks == 0) panelBlocks = 1;
    strsm_panel_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), reinterpret_cast<uint8_t*>(B), pt, panelBlocks, h->stream);

    uint32_t mC;
    uint64_t aOffset;
    uint64_t axpyOffset;
    if (forward) {
        mC = ctx.mU32 - k - bs;
        if (mC == 0) return;
        aOffset = ctx.isTrans ? (static_cast<uint64_t>(k) + static_cast<uint64_t>(k + bs) * ctx.ldaU32)
                              : (static_cast<uint64_t>(k + bs) + static_cast<uint64_t>(k) * ctx.ldaU32);
        axpyOffset = k + bs;
    } else {
        mC = k;
        if (mC == 0) return;
        aOffset = ctx.isTrans ? static_cast<uint64_t>(k) : static_cast<uint64_t>(k) * ctx.ldaU32;
        axpyOffset = 0;
    }

    if (!ctx.isTrans) {
        StrsmTrailingUpdateNoTrans(h, ctx.nU32, k, bs, mC, aOffset, axpyOffset, ctx.ldaU32, ctx.ldbU32,
            ctx.noTransTempStride, ctx.aicCoreNum, aivCoreNum, A, B, ctx.tempDev);
    } else {
        StrsmTrailingUpdateTrans(h, ctx.nU32, k, bs, mC, aOffset, axpyOffset, ctx.ldaU32, ctx.ldbU32,
            ctx.transTempStride, ctx.nAligned, ctx.aicCoreNum, aivCoreNum, A, B, ctx.tempDev, ctx.aWsDev, ctx.bWsDev);
    }
}

static aclblasStatus_t StrsmLeftBlockedPath(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, int lda,
    int ldb, float alpha, const float* A, float* B, uint32_t aivCoreNum)
{
    uint32_t mU32 = static_cast<uint32_t>(m);
    uint32_t nU32 = static_cast<uint32_t>(n);
    uint32_t panelBs = ChoosePanelSize(mU32);
    if (panelBs == 0) {
        OP_LOGE("aclblasStrsm", "invalid panelBs=0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t scaleBlocks = std::min(aivCoreNum, nU32);
    if (scaleBlocks == 0) scaleBlocks = 1;
    strsm_scale_kernel_do(reinterpret_cast<uint8_t*>(B), alpha, mU32, nU32,
        static_cast<uint32_t>(ldb), scaleBlocks, h->stream);

    StrsmBlockedCtx ctx{};
    ctx.mU32 = mU32;
    ctx.nU32 = nU32;
    ctx.ldaU32 = static_cast<uint32_t>(lda);
    ctx.ldbU32 = static_cast<uint32_t>(ldb);
    ctx.panelBs = panelBs;
    ctx.aicCoreNum = GetAicCoreCount();
    if (ctx.aicCoreNum == 0) ctx.aicCoreNum = 1;
    ctx.isUpper = (uplo == ACLBLAS_UPPER);
    ctx.isTrans = (trans != ACLBLAS_OP_N);
    ctx.noTransTempStride = CeilAlign<uint32_t>(mU32, 8u);
    ctx.transTempStride = CeilAlign<uint32_t>(nU32, 8u);
    ctx.nAligned = CeilAlign<uint32_t>(nU32, 8u);

    size_t tempWs = ctx.isTrans
        ? static_cast<size_t>(mU32) * ctx.transTempStride * sizeof(float)
        : static_cast<size_t>(nU32) * ctx.noTransTempStride * sizeof(float);
    size_t aWs = static_cast<size_t>(mU32) * panelBs * sizeof(float);
    size_t bWs = static_cast<size_t>(panelBs) * ctx.nAligned * sizeof(float);
    size_t totalWsMax = ctx.isTrans ? (tempWs + aWs + bWs) : tempWs;
    if (totalWsMax > GetEffectiveWorkspaceSize(h)) {
        aclblasStatus_t wsRet = EnsureDefaultWorkspace(h, totalWsMax);
        if (wsRet != ACLBLAS_STATUS_SUCCESS) {
            return wsRet;
        }
    }
    ctx.tempDev = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    if (ctx.isTrans) {
        ctx.aWsDev = ctx.tempDev + tempWs;
        ctx.bWsDev = ctx.aWsDev + aWs;
    }

    uint32_t numSteps = CeilDiv<uint32_t>(mU32, panelBs);
    for (uint32_t step = 0; step < numSteps; ++step) {
        StrsmBlockedStep(h, ctx, step, numSteps, uplo, trans, diag, lda, ldb, A, B, aivCoreNum);
    }

    return ACLBLAS_STATUS_SUCCESS;
}
#endif

static aclblasStatus_t ValidateStrsmParams(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const float* alpha, const float* A, const float* B, int lda, int ldb)
{
    CHECK_RET(
        uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
        OP_LOGE("aclblasStrsm", "invalid uplo=%d", static_cast<int>(uplo)); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasStrsm", "invalid trans=%d", static_cast<int>(trans)); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT,
        OP_LOGE("aclblasStrsm", "invalid diag=%d", static_cast<int>(diag)); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(alpha != nullptr, OP_LOGE("aclblasStrsm", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    CHECK_RET(B != nullptr, OP_LOGE("aclblasStrsm", "B must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    if (*alpha != 0.0f) {
        CHECK_RET(A != nullptr, OP_LOGE("aclblasStrsm", "A must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    int k = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    CHECK_RET(
        lda >= std::max(1, k), OP_LOGE("aclblasStrsm", "invalid lda=%d, k=%d", lda, k); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        ldb >= std::max(1, m), OP_LOGE("aclblasStrsm", "invalid ldb=%d, m=%d", ldb, m); return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t StrsmLeftDispatch(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, int lda, int ldb, float alpha, const float* A, float* B, uint32_t aivCoreNum)
{
#if ASC_DEVKIT_GE_9_1
    uint32_t mU32 = static_cast<uint32_t>(m);
    uint32_t nU32 = static_cast<uint32_t>(n);
    bool useBlockedPath = (mU32 > BLOCKED_THRESHOLD) ||
                          (mU32 <= BLOCKED_THRESHOLD && nU32 >= SIMT_BLOCKED_N_THRESHOLD);
    if (!useBlockedPath) {
        return StrsmLeftSimtPath(h, uplo, trans, diag, m, n, lda, ldb, alpha, A, B, aivCoreNum);
    }
    return StrsmLeftBlockedPath(h, uplo, trans, diag, m, n, lda, ldb, alpha, A, B, aivCoreNum);
#else
    return StrsmLeftSimtPath(h, uplo, trans, diag, m, n, lda, ldb, alpha, A, B, aivCoreNum);
#endif
}

aclblasStatus_t aclblasStrsm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasStrsm", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    CHECK_RET(
        side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT,
        OP_LOGE("aclblasStrsm", "invalid side=%d", static_cast<int>(side)); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0 && n >= 0, OP_LOGE("aclblasStrsm", "m and n must be >= 0, got m=%d, n=%d", m, n);
              return ACLBLAS_STATUS_INVALID_VALUE);

    aclblasStatus_t st = ValidateStrsmParams(side, uplo, trans, diag, m, n, alpha, A, B, lda, ldb);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    if (*alpha == 0.0f) {
        OP_LOGI("aclblasStrsm", "alpha==0 fast path, m=%d, n=%d", m, n);
        uint32_t aivCoreNum = GetAivCoreCount();
        CHECK_RET(aivCoreNum > 0, OP_LOGE("aclblasStrsm", "GetAivCoreCount failed"); return ACLBLAS_STATUS_INTERNAL_ERROR);
        return StrsmAlphaZeroPath(h, m, n, ldb, B, aivCoreNum);
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    CHECK_RET(aivCoreNum > 0, OP_LOGE("aclblasStrsm", "GetAivCoreCount failed"); return ACLBLAS_STATUS_INTERNAL_ERROR);

    if (side == ACLBLAS_SIDE_RIGHT) {
        if (static_cast<uint32_t>(n) <= BLOCKED_THRESHOLD) {
            return StrsmRightSimtPath(h, uplo, trans, diag, m, n, lda, ldb, *alpha, A, B, aivCoreNum);
        }
        return StrsmRightDevicePath(h, uplo, trans, diag, m, n, lda, ldb, *alpha, A, B, aivCoreNum);
    }

    return StrsmLeftDispatch(h, uplo, trans, diag, m, n, lda, ldb, *alpha, A, B, aivCoreNum);
}
