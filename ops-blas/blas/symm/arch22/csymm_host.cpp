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
 * \file csymm_host.cpp
 * \brief CSYMM host implementation for Atlas A2 training series (arch22).
 *
 *        side='L': C = alpha*A*B + beta*C   (A is m x m symmetric)
 *        side='R': C = alpha*B*A + beta*C   (A is n x n symmetric)
 *        alpha and beta are both complex; C is a general m x n matrix. All matrices
 *        are column-major complex64 and only A's uplo triangle is referenced.
 *
 *        Pipeline:
 *          Phase 0a (AIV) split   : A -> Ar, Ai and B -> Br, Bi   (two launches)
 *          Phase 0b (AIV) expand  : mirror A's triangle into a full square
 *          Phase 1  (AIC) gemm    : four real GEMMs, both operands plain
 *          Phase 2  (AIV) combine : assemble and scale the complex result
 *
 *        Column-major handling rests on C^T = (A*B)^T = B^T*A^T. A packed
 *        column-major buffer read as row-major is already the transpose of its
 *        logical matrix, so every operand is in the orientation the row-major Matmul
 *        wants and none of the four GEMMs needs a transpose flag. Writing
 *        G(x, y) for that plain product,
 *          side='L': t1=G(Br,Ar) t2=G(Bi,Ai) t3=G(Bi,Ar) t4=G(Br,Ai)
 *          side='R': t1=G(Ar,Br) t2=G(Ai,Bi) t3=G(Ai,Br) t4=G(Ar,Bi)
 *        which is one formula with `side` deciding only which matrix comes first;
 *        the imaginary part is t3 + t4, an addition, so the labelling of the two
 *        cross terms does not matter.
 *
 *        Phase 0b needs A's packed buffers padded to a multiple of the expansion
 *        tile, which is why the workspace helper sizes them from aPadded.
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/complex_blas3_host_utils.h"
#include "csymm_kernel.h"

namespace {
constexpr const char* OP_TAG = "aclblasCsymm";

inline bool IsComplexZero(const aclblasComplex& v) { return v.real == 0.0F && v.imag == 0.0F; }
} // namespace

// ============================================================================
//  Parameter validation
// ============================================================================
// Enum legality and leading dimensions, checked before the m == 0 || n == 0 quick
// return.
static aclblasStatus_t ValidateCsymmEnumsAndDims(
    aclblasSideMode_t side, aclblasFillMode_t uplo, int m, int n, int lda, int ldb, int ldc)
{
    CHECK_RET(
        side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT,
        OP_LOGE(
            OP_TAG, "side must be ACLBLAS_SIDE_LEFT(141) or ACLBLAS_SIDE_RIGHT(142), got %d", static_cast<int>(side));
        return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE(OP_TAG, "uplo must be ACLBLAS_UPPER(121) or ACLBLAS_LOWER(122), got %d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(m >= 0, OP_LOGE(OP_TAG, "m must be >= 0, got %d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);

    // A is m x m for side='L' and n x n for side='R'.
    const int ldaMin = (side == ACLBLAS_SIDE_LEFT) ? std::max(1, m) : std::max(1, n);
    CHECK_RET(lda >= ldaMin,
              OP_LOGE(OP_TAG, "lda must be >= %d for side=%d, got %d", ldaMin, static_cast<int>(side), lda);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldb >= std::max(1, m), OP_LOGE(OP_TAG, "ldb must be >= max(1,m)=%d, got %d", std::max(1, m), ldb);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= std::max(1, m), OP_LOGE(OP_TAG, "ldc must be >= max(1,m)=%d, got %d", std::max(1, m), ldc);
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateCsymmPointers(
    int m, int n, const aclblasComplex* alpha, const aclblasComplex* A, const aclblasComplex* B,
    const aclblasComplex* beta, const aclblasComplex* C)
{
    CHECK_RET(alpha != nullptr, OP_LOGE(OP_TAG, "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE(OP_TAG, "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    const bool hasWork = (m > 0 && n > 0);
    CHECK_RET(!(A == nullptr && hasWork), OP_LOGE(OP_TAG, "A must not be nullptr when m>0 and n>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(B == nullptr && hasWork), OP_LOGE(OP_TAG, "B must not be nullptr when m>0 and n>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(C == nullptr && hasWork), OP_LOGE(OP_TAG, "C must not be nullptr when m>0 and n>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Launch
// ============================================================================
// Phase 0a splits A and B into real halves; Phase 0b mirrors A's stored triangle
// into a full square. imagSign = +1 because A is symmetric rather than Hermitian,
// so Ai is symmetric too and its diagonal carries real information -- the one and
// only difference from CHEMM.
static void RunCsymmSplitAndExpand(
    void* stream, aclblasFillMode_t uplo, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, const aclblasComplex* A,
    const aclblasComplex* B, const CBlas3SymmShape& shape, const CBlas3SymmWorkspace& ws, uint32_t aivCoreNum)
{
    // A is split over its whole d x d extent. The triangle BLAS leaves undefined
    // is read here, which is harmless: it is inside the caller's allocation and
    // Phase 0b overwrites every byte of it.
    CBlas3SplitTilingData splitA{};
    splitA.rows = shape.d;
    splitA.cols = shape.d;
    splitA.lda = lda;
    splitA.packedLd = shape.aPadded;
    splitA.scaleEnabled = 1U;
    splitA.scaleFactor = CBLAS3_RANGE_SCALE_DOWN;
    const uint32_t splitABlocks = std::min(aivCoreNum, std::max(1U, shape.d));
    OP_LOGI(OP_TAG, "phase0a split A: d=%u packedLd=%u aivBlocks=%u", shape.d, shape.aPadded, splitABlocks);
    csymm_split_kernel_do(
        splitABlocks, stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)), ws.ar, ws.ai, splitA);

    CBlas3SplitTilingData splitB{};
    splitB.rows = m;
    splitB.cols = n;
    splitB.lda = ldb;
    splitB.packedLd = m;
    const uint32_t splitBBlocks = std::min(aivCoreNum, std::max(1U, n));
    OP_LOGI(OP_TAG, "phase0a split B: m=%u n=%u aivBlocks=%u", m, n, splitBBlocks);
    csymm_split_kernel_do(
        splitBBlocks, stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(B)), ws.br, ws.bi, splitB);

    CBlas3ExpandTilingData expand{};
    expand.d = shape.d;
    expand.packedLd = shape.aPadded;
    expand.uploMode = static_cast<uint32_t>(uplo);
    expand.tileCount = shape.tileCount;
    expand.imagSign = 1;
    const uint32_t expandBlocks = std::min(aivCoreNum, std::max(1U, shape.tileCount));
    OP_LOGI(OP_TAG, "phase0b expand: tiles=%u aivBlocks=%u", shape.tileCount, expandBlocks);
    csymm_expand_kernel_do(expandBlocks, stream, ws.ar, ws.ai, expand);
}

// Phase 0/0b for the paired-K layout: A is split and mirrored twice, once into
// [Ar | -Ai] and once into [Ai | Ar]. Each buffer holds its two planes one packed
// column apart, so a doubled packedLd plus a base offset is all that separates
// them; the tile mirror runs unchanged on each plane pair. B is not touched --
// the GEMM reads the caller's buffer directly.
static void RunCsymmPairedSplitAndExpand(
    void* stream, aclblasFillMode_t uplo, uint32_t lda, const aclblasComplex* A, const CBlas3SymmShape& shape,
    const CBlas3SymmWorkspace& ws, uint32_t aivCoreNum)
{
    const uint32_t pairLd = shape.aPadded * CBLAS3_CPLX_FLOATS;
    const size_t planeGap = static_cast<size_t>(shape.aPadded) * sizeof(float);
    const uint32_t splitBlocks = std::min(aivCoreNum, std::max(1U, shape.d));
    const uint32_t expandBlocks = std::min(aivCoreNum, std::max(1U, shape.tileCount));
    OP_LOGI(OP_TAG, "phase0 paired: d=%u pairLd=%u aivBlocks=%u/%u", shape.d, pairLd, splitBlocks, expandBlocks);

    CBlas3SplitTilingData split{};
    split.rows = shape.d;
    split.cols = shape.d;
    split.lda = lda;
    split.packedLd = pairLd;
    split.scaleEnabled = 1U;
    split.scaleFactor = CBLAS3_RANGE_SCALE_DOWN;

    CBlas3ExpandTilingData expand{};
    expand.d = shape.d;
    expand.packedLd = pairLd;
    expand.uploMode = static_cast<uint32_t>(uplo);
    expand.tileCount = shape.tileCount;
    expand.imagSign = 1;

    uint8_t* const src = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A));
    // [Ar | -Ai]: real plane first, negated imaginary plane one column on.
    split.negateImag = 1U;
    csymm_split_kernel_do(splitBlocks, stream, src, ws.aPair[0], ws.aPair[0] + planeGap, split);
    csymm_expand_kernel_do(expandBlocks, stream, ws.aPair[0], ws.aPair[0] + planeGap, expand);
    // [Ai | Ar]: the same two planes with the destinations swapped, no negation.
    split.negateImag = 0U;
    csymm_split_kernel_do(splitBlocks, stream, src, ws.aPair[1] + planeGap, ws.aPair[1], split);
    csymm_expand_kernel_do(expandBlocks, stream, ws.aPair[1] + planeGap, ws.aPair[1], expand);
}

// Phase 1 for the paired-K layout: two GEMMs over the doubled contraction.
//   Pr = B * [Ar | -Ai] = sum_j (Ar_j*Br_j - Ai_j*Bi_j)
//   Pi = B * [Ai |  Ar] = sum_j (Ai_j*Br_j + Ar_j*Bi_j)
// Both sums alternate their two contributions per input column, so a term and its
// counterpart meet inside the accumulator while the running sum is still small.
// Summing them as separate products instead lets each reach the fp32 range limit
// on its own, and Inf - Inf is NaN.
static void RunCsymmPairedGemms(
    void* stream, uint32_t ldb, const aclblasComplex* B, const CBlas3SymmShape& shape, const CBlas3SymmWorkspace& ws,
    uint32_t aicCoreNum)
{
    CBlas3GemmTilingData gemm = CBlas3MakeSymmPairedGemmTiling(shape, ldb, ws.tempLdc);
    const uint32_t gemmBlocks = std::min(gemm.mBlocks * gemm.nBlocks, aicCoreNum);
    const uint32_t kChunk = csymm_gemm_single_k();
    OP_LOGI(
        OP_TAG, "phase1 paired gemm: M=%u N=%u K=%u ldA=%u ldB=%u aicBlocks=%u kChunk=%u", gemm.m, gemm.n, gemm.k,
        gemm.ldA, gemm.ldB, gemmBlocks, kChunk);

    uint8_t* const left = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(B));
    CBlas3LaunchGemmChunked(csymm_gemm_kernel_do, gemmBlocks, stream, left, ws.aPair[0], ws.temp[0], gemm, kChunk);
    CBlas3LaunchGemmChunked(csymm_gemm_kernel_do, gemmBlocks, stream, left, ws.aPair[1], ws.temp[1], gemm, kChunk);
}

// Phase 1: four plain GEMMs. `first` is the matrix that leads the operand pair,
// which is the only thing `side` changes.
static void RunCsymmGemms(
    void* stream, bool sideLeft, const CBlas3SymmShape& shape, const CBlas3SymmWorkspace& ws, uint32_t aicCoreNum)
{
    CBlas3GemmTilingData gemm = CBlas3MakeSymmGemmTiling(shape, ws.tempLdc);
    const uint32_t gemmBlocks = std::min(gemm.mBlocks * gemm.nBlocks, aicCoreNum);
    const uint32_t kChunk = csymm_gemm_single_k();
    OP_LOGI(
        OP_TAG, "phase1 gemm: M=%u N=%u K=%u ldA=%u ldB=%u aicBlocks=%u kChunk=%u", gemm.m, gemm.n, gemm.k, gemm.ldA,
        gemm.ldB, gemmBlocks, kChunk);

    uint8_t* firstRe = sideLeft ? ws.br : ws.ar;
    uint8_t* firstIm = sideLeft ? ws.bi : ws.ai;
    uint8_t* secondRe = sideLeft ? ws.ar : ws.br;
    uint8_t* secondIm = sideLeft ? ws.ai : ws.bi;

    auto gemmInto = [&](uint8_t* l, uint8_t* r, uint8_t* out) {
        CBlas3LaunchGemmChunked(csymm_gemm_kernel_do, gemmBlocks, stream, l, r, out, gemm, kChunk);
    };
    gemmInto(firstRe, secondRe, ws.temp[0]); // t1
    gemmInto(firstIm, secondIm, ws.temp[1]); // t2
    gemmInto(firstIm, secondRe, ws.temp[2]); // t3
    gemmInto(firstRe, secondIm, ws.temp[3]); // t4
}

// Phases 0 and 1, in whichever layout the shape selected.
static aclblasStatus_t RunCsymmPhases01(
    void* stream, aclblasFillMode_t uplo, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, const aclblasComplex* A,
    const aclblasComplex* B, const CBlas3SymmShape& shape, const CBlas3SymmWorkspace& ws, bool paired,
    uint32_t aivCoreNum)
{
    const uint32_t aicCoreNum = GetAicCoreCount();
    if (aicCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAicCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (paired) {
        RunCsymmPairedSplitAndExpand(stream, uplo, lda, A, shape, ws, aivCoreNum);
        RunCsymmPairedGemms(stream, ldb, B, shape, ws, aicCoreNum);
    } else {
        RunCsymmSplitAndExpand(stream, uplo, m, n, lda, ldb, A, B, shape, ws, aivCoreNum);
        RunCsymmGemms(stream, shape.sideLeft, shape, ws, aicCoreNum);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Phase 2: Pr = t1 - t2, Pi = t3 + t4, then complex alpha/beta scaling. Under the
// paired-K layout the two sums are already finished, so t1 / t3 are read as-is.
static void RunCsymmCombine(
    void* stream, uint32_t m, uint32_t n, uint32_t ldc, aclblasComplex* C, const CBlas3SymmWorkspace& ws, bool paired,
    bool skipTemp, bool isBetaZero, const aclblasComplex& alphaVal, const aclblasComplex& betaVal, uint32_t aivCoreNum)
{
    CsymmCombineTilingData combine{};
    combine.pairedTemps = static_cast<uint32_t>(paired ? 1 : 0);
    combine.unscaleEnabled = 1U;
    combine.unscaleFactor = CBLAS3_RANGE_SCALE_UP;
    combine.m = m;
    combine.n = n;
    combine.ldc = ldc;
    combine.tempLdc = ws.tempLdc;
    combine.skipAlphaTerm = static_cast<uint32_t>(skipTemp ? 1 : 0);
    combine.isBetaZero = static_cast<uint32_t>(isBetaZero ? 1 : 0);
    combine.alphaRe = alphaVal.real;
    combine.alphaIm = alphaVal.imag;
    combine.betaRe = betaVal.real;
    combine.betaIm = betaVal.imag;
    const uint32_t combineBlocks = std::min(aivCoreNum, std::max(1U, n));
    OP_LOGI(OP_TAG, "phase2 combine: m=%u n=%u paired=%d aivBlocks=%u", m, n, static_cast<int>(paired), combineBlocks);
    // Paired: t1 = Pr, t3 = Pi. The unused slots repeat them so every pointer the
    // kernel binds stays inside the workspace.
    uint8_t* const t2 = paired ? ws.temp[0] : ws.temp[1];
    uint8_t* const t3 = paired ? ws.temp[1] : ws.temp[2];
    uint8_t* const t4 = paired ? ws.temp[1] : ws.temp[3];
    csymm_combine_kernel_do(combineBlocks, stream, ws.temp[0], t2, t3, t4, reinterpret_cast<uint8_t*>(C), combine);
}

static aclblasStatus_t LaunchCsymmKernel(
    _aclblas_handle* h, aclblasSideMode_t side, aclblasFillMode_t uplo, uint32_t m, uint32_t n,
    const aclblasComplex* alpha, const aclblasComplex* A, uint32_t lda, const aclblasComplex* B, uint32_t ldb,
    const aclblasComplex* beta, aclblasComplex* C, uint32_t ldc)
{
    void* stream = h->stream;

    aclblasComplex alphaVal{0.0F, 0.0F};
    aclblasComplex betaVal{0.0F, 0.0F};
    const aclblasStatus_t readRet = CBlas3ReadScalarsFromDevice(alpha, beta, alphaVal, betaVal, stream, OP_TAG);
    if (readRet != ACLBLAS_STATUS_SUCCESS) {
        return readRet;
    }

    const bool skipTemp = IsComplexZero(alphaVal);
    const bool isBetaZero = IsComplexZero(betaVal);
    OP_LOGD(
        OP_TAG, "alpha=(%f,%f) beta=(%f,%f) skipTemp=%d isBetaZero=%d", static_cast<double>(alphaVal.real),
        static_cast<double>(alphaVal.imag), static_cast<double>(betaVal.real), static_cast<double>(betaVal.imag),
        static_cast<int>(skipTemp), static_cast<int>(isBetaZero));

    if (skipTemp && betaVal.real == 1.0F && betaVal.imag == 0.0F) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const bool sideLeft = (side == ACLBLAS_SIDE_LEFT);
    const CBlas3SymmShape shape = CBlas3ResolveSymmShape(sideLeft, m, n);
    CBlas3SymmWorkspace ws{};
    const aclblasStatus_t wsRet = CBlas3PrepareSymmWorkspace(h, shape, skipTemp, ws, OP_TAG);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        return wsRet;
    }

    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    const bool paired = CBlas3SymmUsesPairedK(shape);
    if (!skipTemp) {
        const aclblasStatus_t phaseRet =
            RunCsymmPhases01(stream, uplo, m, n, lda, ldb, A, B, shape, ws, paired, aivCoreNum);
        if (phaseRet != ACLBLAS_STATUS_SUCCESS) {
            return phaseRet;
        }
    }

    RunCsymmCombine(stream, m, n, ldc, C, ws, paired, skipTemp, isBetaZero, alphaVal, betaVal, aivCoreNum);
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Public API entry
// ============================================================================
extern "C" aclblasStatus_t aclblasCsymm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C,
    int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    const aclblasStatus_t enumSt = ValidateCsymmEnumsAndDims(side, uplo, m, n, lda, ldb, ldc);
    if (enumSt != ACLBLAS_STATUS_SUCCESS) {
        return enumSt;
    }

    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const aclblasStatus_t ptrSt = ValidateCsymmPointers(m, n, alpha, A, B, beta, C);
    if (ptrSt != ACLBLAS_STATUS_SUCCESS) {
        return ptrSt;
    }

    return LaunchCsymmKernel(
        static_cast<_aclblas_handle*>(handle), side, uplo, static_cast<uint32_t>(m), static_cast<uint32_t>(n), alpha, A,
        static_cast<uint32_t>(lda), B, static_cast<uint32_t>(ldb), beta, C, static_cast<uint32_t>(ldc));
}
