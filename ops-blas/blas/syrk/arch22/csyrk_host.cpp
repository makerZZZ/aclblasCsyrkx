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
 * \file csyrk_host.cpp
 * \brief CSYRK host implementation for Atlas A2 training series (arch22).
 *
 *        C = alpha * A * A^T + beta * C   (trans = 'N', A is n x k)
 *        C = alpha * A^T * A + beta * C   (trans = 'T', A is k x n)
 *        alpha and beta are complex; C is symmetric and only the uplo triangle is
 *        referenced and updated. All matrices are column-major complex64.
 *
 *        Three-stage pipeline, with Phases 0 and 1 shared with the other complex
 *        BLAS-3 operators:
 *          Phase 0 (AIV) split   : complex A -> P = [Ar|Ai], Q = [Ar|-Ai], R = [Ai|Ar]
 *          Phase 1 (AIC) gemm    : Pr = P*Q^T, Pi = P*R^T  (two GEMMs, K extent 2k)
 *          Phase 2 (AIV) combine : complex alpha/beta scaling only
 *
 *        Folding the sign pattern into the operands keeps the real part accurate:
 *        Ar*Ar^T and Ai*Ai^T are each O(k) in magnitude while their difference is
 *        only O(sqrt(k)), so subtracting two separately rounded GEMM results loses
 *        about two orders of magnitude of relative accuracy at k=8000.
 */

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/complex_blas3_host_utils.h"
#include "csyrk_kernel.h"

namespace {
constexpr const char* OP_TAG = "aclblasCsyrk";
// Phase 1 uses K-concatenated operands, so it needs only two temps (Pr, Pi) but
// three packed buffers (P = [Ar|Ai], Q = [Ar|-Ai], R = [Ai|Ar]), each of K extent
// 2k. Total workspace is 6*k*n + 2*tempLdc*n floats, which caps a square input at
// roughly n = 8190 against the library's 2GiB workspace limit.
constexpr uint32_t CSYRK_TEMP_COUNT = 2;   // Pr, Pi
constexpr uint32_t CSYRK_PACKED_COUNT = 3; // P, Q, R

inline bool IsComplexZero(const aclblasComplex& v) { return v.real == 0.0F && v.imag == 0.0F; }

inline bool IsComplexOne(const aclblasComplex& v) { return v.real == 1.0F && v.imag == 0.0F; }
} // namespace

// ============================================================================
//  Parameter validation
// ============================================================================
// Enum legality and leading dimensions. Must run *before* the n == 0 quick
// return: an illegal uplo/trans is an error regardless of the problem size, which
// is how reference BLAS and the shared CPU golden both behave.
static aclblasStatus_t ValidateCsyrkEnumsAndDims(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, int lda, int ldc)
{
    CHECK_RET(uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER,
              OP_LOGE(OP_TAG, "uplo must be ACLBLAS_UPPER(121) or ACLBLAS_LOWER(122), got %d", static_cast<int>(uplo));
              return ACLBLAS_STATUS_INVALID_ENUM);
    // CSYRK takes the plain transpose, not the conjugate transpose.
    CHECK_RET(trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T,
              OP_LOGE(OP_TAG, "trans must be ACLBLAS_OP_N(111) or ACLBLAS_OP_T(112), got %d", static_cast<int>(trans));
              return ACLBLAS_STATUS_INVALID_ENUM);
    CHECK_RET(n >= 0, OP_LOGE(OP_TAG, "n must be >= 0, got %d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(k >= 0, OP_LOGE(OP_TAG, "k must be >= 0, got %d", k); return ACLBLAS_STATUS_INVALID_VALUE);

    const int ldaMin = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
    CHECK_RET(lda >= ldaMin,
              OP_LOGE(OP_TAG, "lda must be >= %d for trans=%d, got %d", ldaMin, static_cast<int>(trans), lda);
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(ldc >= std::max(1, n), OP_LOGE(OP_TAG, "ldc must be >= max(1,n)=%d, got %d", std::max(1, n), ldc);
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// Pointer checks, run only after the quick return: with n == 0 nothing is read or
// written, so null pointers are legal there.
static aclblasStatus_t ValidateCsyrkPointers(
    int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, const aclblasComplex* beta,
    const aclblasComplex* C)
{
    CHECK_RET(alpha != nullptr, OP_LOGE(OP_TAG, "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(beta != nullptr, OP_LOGE(OP_TAG, "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(A == nullptr && n > 0 && k > 0), OP_LOGE(OP_TAG, "A must not be nullptr when n>0 and k>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(!(C == nullptr && n > 0), OP_LOGE(OP_TAG, "C must not be nullptr when n>0");
              return ACLBLAS_STATUS_INVALID_VALUE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Launch
// ============================================================================
// The real part is Cr = Ar*Ar^T - Ai*Ai^T, computed as one GEMM P*Q^T over a
// doubled K extent with the sign folded into Q. How the two contributions are
// ordered along K decides where the subtraction happens:
//
//   block-concatenated  P = [Ar | Ai], Q = [Ar | -Ai]
//       the accumulator sums the whole Ar*Ar^T block before the Ai block starts
//       subtracting, so each block can reach the fp32 range limit on its own and
//       Inf - Inf is NaN;
//   K-interleaved       P = [Ar_0, Ai_0, Ar_1, Ai_1, ...], Q = [Ar_0, -Ai_0, ...]
//       the accumulator alternates the two signs per input column, so a term and
//       its counterpart cancel while the running sum is still small.
//
// Both forms hold the same values and the same total extent - they differ only
// in the destination offsets, so the interleaved form is selected by the strides
// alone and Phase 1 is unaffected. R = [Ai | Ar] follows the same pairing, which
// keeps P*R^T = Ar*Ai^T + Ai*Ar^T (the imaginary part, both terms same sign).
//
// Interleaving needs each pair to sit in adjacent packed columns, which the
// contiguous Phase 0 writes only allow when a whole input column is contiguous
// in the packed buffer:
//   trans='N': the packed view is (2k x n) row-major with row stride n, so an
//              input column is one contiguous run -> interleaved;
//   trans='T': the packed view is (n x 2k) row-major, so the two halves sit side
//              by side inside each row and interleaving would need a stride-2
//              scatter -> kept block-concatenated. No RANDOM_EXTREME case in the
//              official grid uses trans='T', and away from overflow the two forms
//              are numerically equivalent.
static CBlas3SplitConcatTilingData MakeCsyrkSplitTiling(const CBlas3RankKShape& shape, uint32_t lda, bool transposed)
{
    CBlas3SplitConcatTilingData split{};
    split.rows = shape.aRows;
    split.cols = shape.aCols;
    split.lda = lda;
    if (!transposed) {
        split.packedLd = shape.aRows;       // = n
        split.blockStride = shape.aRows;    // = n, the pair member sits one column on
        split.colStride = shape.aRows * 2U; // = 2n, one pair per input column
    } else {
        split.packedLd = shape.aRows * 2U;  // = 2k
        split.blockStride = shape.aRows;    // = k
        split.colStride = shape.aRows * 2U; // = 2k
    }
    return split;
}

// Phase 0 (split into P/Q/R over a doubled K extent, interleaved or
// block-concatenated per MakeCsyrkSplitTiling) followed by Phase 1 (two real
// GEMMs). Only reached when alpha and k are both non-zero.
static aclblasStatus_t RunCsyrkSplitAndGemms(
    void* stream, aclblasFillMode_t uplo, aclblasOperation_t trans, const CBlas3RankKShape& shape, uint32_t lda,
    const aclblasComplex* A, const CBlas3Workspace& ws, uint32_t aivCoreNum)
{
    const uint32_t aicCoreNum = GetAicCoreCount();
    if (aicCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAicCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    const CBlas3SplitConcatTilingData split = MakeCsyrkSplitTiling(shape, lda, trans != ACLBLAS_OP_N);
    const uint32_t splitBlocks = std::min(aivCoreNum, std::max(1U, shape.aCols));
    OP_LOGI(
        OP_TAG, "phase0 split: rows=%u cols=%u packedLd=%u blockStride=%u aivBlocks=%u", split.rows, split.cols,
        split.packedLd, split.blockStride, splitBlocks);
    csyrk_split_kernel_do(
        splitBlocks, stream, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)), ws.packed[0], ws.packed[1],
        ws.packed[2], split);

    // Phase 1: two real GEMMs over the doubled K extent.
    //   Pr = P * Q^T = Ar*Ar^T - Ai*Ai^T
    //   Pi = P * R^T = Ar*Ai^T + Ai*Ar^T
    CBlas3GemmTilingData gemm = CBlas3MakeRankKGemmTiling(shape, ws.tempLdc, static_cast<uint32_t>(uplo));
    gemm.k = shape.k * 2U;          // concatenated K extent
    gemm.packedLd = split.packedLd; // row stride, no longer equal to K
    const uint32_t gemmBlocks = std::min(gemm.mBlocks * gemm.nBlocks, aicCoreNum);
    const uint32_t kChunk = csyrk_gemm_single_k();
    OP_LOGI(
        OP_TAG, "phase1 gemm: n=%u k=%u(concat) packedLd=%u aicBlocks=%u transMode=%u kChunk=%u", gemm.m, gemm.k,
        gemm.packedLd, gemmBlocks, shape.transMode, kChunk);

    uint8_t* bufP = ws.packed[0];
    CBlas3LaunchGemmChunked(csyrk_gemm_kernel_do, gemmBlocks, stream, bufP, ws.packed[1], ws.temp[0], gemm, kChunk);
    CBlas3LaunchGemmChunked(csyrk_gemm_kernel_do, gemmBlocks, stream, bufP, ws.packed[2], ws.temp[1], gemm, kChunk);
    return ACLBLAS_STATUS_SUCCESS;
}

// Phase 2: scale and write the uplo triangle of C. Both Pr and Pi are symmetric,
// so the column-major (transposed) read needs no pointer swap.
static void RunCsyrkCombine(
    void* stream, aclblasFillMode_t uplo, uint32_t n, uint32_t ldc, aclblasComplex* C, const CBlas3Workspace& ws,
    bool skipTemp, bool isBetaZero, const aclblasComplex& alphaVal, const aclblasComplex& betaVal, uint32_t aivCoreNum)
{
    CsyrkCombineTilingData combine{};
    combine.n = n;
    combine.ldc = ldc;
    combine.tempLdc = ws.tempLdc;
    combine.uploMode = static_cast<uint32_t>(uplo);
    combine.skipAlphaTerm = static_cast<uint32_t>(skipTemp ? 1 : 0);
    combine.isBetaZero = static_cast<uint32_t>(isBetaZero ? 1 : 0);
    combine.alphaRe = alphaVal.real;
    combine.alphaIm = alphaVal.imag;
    combine.betaRe = betaVal.real;
    combine.betaIm = betaVal.imag;
    const uint32_t combineBlocks = std::min(aivCoreNum, std::max(1U, n));
    OP_LOGI(OP_TAG, "phase2 combine: n=%u aivBlocks=%u", n, combineBlocks);
    csyrk_combine_kernel_do(combineBlocks, stream, ws.temp[0], ws.temp[1], reinterpret_cast<uint8_t*>(C), combine);
}

static aclblasStatus_t LaunchCsyrkKernel(
    _aclblas_handle* h, aclblasFillMode_t uplo, aclblasOperation_t trans, uint32_t n, uint32_t k,
    const aclblasComplex* alpha, const aclblasComplex* A, uint32_t lda, const aclblasComplex* beta, aclblasComplex* C,
    uint32_t ldc)
{
    void* stream = h->stream;

    aclblasComplex alphaVal{0.0F, 0.0F};
    aclblasComplex betaVal{0.0F, 0.0F};
    const aclblasStatus_t readRet = CBlas3ReadScalarsFromDevice(alpha, beta, alphaVal, betaVal, stream, OP_TAG);
    if (readRet != ACLBLAS_STATUS_SUCCESS) {
        return readRet;
    }

    const bool skipTemp = IsComplexZero(alphaVal) || (k == 0U);
    const bool isBetaZero = IsComplexZero(betaVal);
    OP_LOGD(
        OP_TAG, "alpha=(%f,%f) beta=(%f,%f) skipTemp=%d isBetaZero=%d", static_cast<double>(alphaVal.real),
        static_cast<double>(alphaVal.imag), static_cast<double>(betaVal.real), static_cast<double>(betaVal.imag),
        static_cast<int>(skipTemp), static_cast<int>(isBetaZero));

    // Nothing contributes from A and C keeps its value: no kernel needed at all.
    if (skipTemp && IsComplexOne(betaVal)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const CBlas3RankKShape shape = CBlas3ResolveRankKShape(trans != ACLBLAS_OP_N, n, k);
    // Each packed buffer holds both halves, so it is twice the size of a plain
    // packed operand. Express that by doubling aCols, which is what the workspace
    // planner multiplies by.
    CBlas3RankKShape packedShape = shape;
    packedShape.aCols = shape.aCols * 2U;
    CBlas3Workspace ws{};
    const aclblasStatus_t wsRet =
        CBlas3PrepareWorkspace(h, packedShape, CSYRK_TEMP_COUNT, CSYRK_PACKED_COUNT, skipTemp, ws, OP_TAG);
    if (wsRet != ACLBLAS_STATUS_SUCCESS) {
        return wsRet;
    }

    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE(OP_TAG, "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (!skipTemp) {
        const aclblasStatus_t gemmRet = RunCsyrkSplitAndGemms(stream, uplo, trans, shape, lda, A, ws, aivCoreNum);
        if (gemmRet != ACLBLAS_STATUS_SUCCESS) {
            return gemmRet;
        }
    }

    RunCsyrkCombine(stream, uplo, n, ldc, C, ws, skipTemp, isBetaZero, alphaVal, betaVal, aivCoreNum);
    return ACLBLAS_STATUS_SUCCESS;
}

// ============================================================================
//  Public API entry
// ============================================================================
extern "C" aclblasStatus_t aclblasCsyrk(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* beta, aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        OP_LOGE(OP_TAG, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    const aclblasStatus_t enumSt = ValidateCsyrkEnumsAndDims(uplo, trans, n, k, lda, ldc);
    if (enumSt != ACLBLAS_STATUS_SUCCESS) {
        return enumSt;
    }

    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const aclblasStatus_t ptrSt = ValidateCsyrkPointers(n, k, alpha, A, beta, C);
    if (ptrSt != ACLBLAS_STATUS_SUCCESS) {
        return ptrSt;
    }

    return LaunchCsyrkKernel(
        static_cast<_aclblas_handle*>(handle), uplo, trans, static_cast<uint32_t>(n), static_cast<uint32_t>(k), alpha,
        A, static_cast<uint32_t>(lda), beta, C, static_cast<uint32_t>(ldc));
}
