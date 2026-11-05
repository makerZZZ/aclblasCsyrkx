/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Blocked, multi-threaded drivers for the CBLAS level-3 reference routines the
// complex BLAS-3 tests use as their golden.
//
// The reference dominates the runtime of these suites: on a 4096x4096 CHERK case
// one cblas_cherk call costs ~290 s against ~0.3 ms of NPU time, and the
// available OpenBLAS is a SINGLE_THREADED build, so openblas_set_num_threads is
// a no-op. The routines here therefore partition the *output* into independent
// column panels and drive one CBLAS call per panel from its own thread. Level-3
// CBLAS is reentrant, so concurrent calls on disjoint output blocks are safe.
//
// Partitioning the output does not reorder any accumulation: every element
// still sums over the full K extent in the same order, so CSYMM comes out
// bit-identical to the single-threaded call. CHERK, CSYRK and CHER2K are the
// exception on their large panels (see DIAG_GEMM_THRESHOLD below): the
// diagonal block there is issued as GEMM call(s) instead of one fused
// cblas_c{herk,syrk,her2k} pass (CherkDiagBlock / CsyrkDiagBlock /
// Cher2kDiagBlock; CHER2K's off-diagonal rectangle always was two GEMMs, see
// Cher2kOffBlock), which rounds the intermediate to fp32 once more and moves
// results by a few ULP (~3e-7 relative, against the 2^-10 relative tolerance
// the accuracy checks use). That is a rounding-path difference of the kind
// any two BLAS builds show, not a change of reference semantics.
//
// The diagonal block of a large panel specifically avoids the single-threaded
// cblas_c{herk,syrk,her2k} call: measured on this OpenBLAS build, one such
// call scales far worse than the equivalent cblas_cgemm for the same output
// size, badly enough that one large panel's diagonal call can dominate the
// whole suite's wall time on its own (a 4096x4096 CHERK case is on the order
// of a minute of that call alone) -- and, transitively, dominate everyone
// else's too: the PreSmoke pod's exit-137 kills have landed on whichever
// operator's large panel happened to still be running when that One Big Call
// finally exhausted the pod's time/resource budget. Expressing the diagonal as
// GEMM(s) instead removes that outlier without touching the actual math.

#ifndef CBLAS_THREADED_H
#define CBLAS_THREADED_H

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

#include <cblas.h>

#ifndef _WIN32
#include <sched.h>
#endif

namespace blas_test {

// Complex float elements are two floats wide throughout.
constexpr int64_t CPLX = 2;
// Panels are rounded up to this to keep each CBLAS call on a sane tile size.
constexpr int PANEL_ALIGN = 64;
constexpr int MAX_GOLDEN_THREADS = 32;

// Usable cores. sched_getaffinity honours a pod's cpuset, which the raw
// hardware_concurrency does not; BLAS_GOLDEN_THREADS overrides both.
inline int GoldenThreads()
{
    static const int cached = [] {
        const char* env = std::getenv("BLAS_GOLDEN_THREADS");
        if (env != nullptr) {
            char* end = nullptr;
            const long v = std::strtol(env, &end, 10);
            if (end != env && v > 0) {
                return std::min(static_cast<int>(v), MAX_GOLDEN_THREADS);
            }
        }
        int cores = 0;
        // CPU_COUNT only exists when sched.h was compiled with _GNU_SOURCE; without
        // it fall through to hardware_concurrency, which ignores the cpuset.
#if defined(CPU_COUNT)
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0) {
            cores = CPU_COUNT(&set);
        }
#endif
        if (cores <= 0) {
            cores = static_cast<int>(std::thread::hardware_concurrency());
        }
        const int used = std::min(std::max(cores, 1), MAX_GOLDEN_THREADS);
        // Logged once: the reference cost scales with this, and it is the only
        // way to tell from a CI log how many cores the runner actually offered.
        std::cout << "[GOLDEN] reference threads=" << used << " (usable cores=" << cores
                  << ", cap=" << MAX_GOLDEN_THREADS << ")" << std::endl;
        return used;
    }();
    return cached;
}

// Smallest multiple of PANEL_ALIGN for which `lanes` chunks cover span. Grown by
// repeated addition rather than a ceiling division, so the expression carries no
// divisor at all; the clamp on lanes also rules out a non-terminating loop.
inline int ChunkFor(int span, int threads)
{
    const int lanes = (threads > 0) ? threads : 1;
    int chunk = PANEL_ALIGN;
    while (static_cast<int64_t>(chunk) * static_cast<int64_t>(lanes) < static_cast<int64_t>(span)) {
        chunk += PANEL_ALIGN;
    }
    return chunk;
}

// Splits [0, span) into one aligned chunk per thread and runs them in parallel.
inline void ForEachChunk(int span, const std::function<void(int, int)>& body)
{
    if (span <= 0) {
        return;
    }
    const int chunk = ChunkFor(span, GoldenThreads());
    if (chunk >= span) {
        body(0, span);
        return;
    }
    std::vector<std::thread> pool;
    for (int start = 0; start < span; start += chunk) {
        const int len = std::min(chunk, span - start);
        pool.emplace_back([&body, start, len] { body(start, len); });
    }
    for (auto& t : pool) {
        t.join();
    }
}

// One column panel of a triangular output: the diagonal block [j0, j0+b) plus
// the rectangle of the same columns that lies inside the stored triangle.
struct TriPanel {
    int j0;
    int b;
    int offRow0;
    int offRows;
};

inline TriPanel MakeTriPanel(CBLAS_UPLO uplo, int n, int j0, int b)
{
    if (uplo == CblasUpper) {
        return TriPanel{j0, b, 0, j0};
    }
    return TriPanel{j0, b, j0 + b, n - (j0 + b)};
}

// Start of a rank-k operand's sub-block: a row offset when the operand is used
// untransposed (it is n-by-k), a column offset otherwise (it is k-by-n).
inline const float* RankKSub(const float* mat, int ld, bool noTrans, int idx)
{
    return noTrans ? mat + CPLX * idx : mat + CPLX * static_cast<int64_t>(idx) * ld;
}

inline float* TriBlock(float* C, int ldc, int col, int row)
{
    return C + CPLX * (static_cast<int64_t>(col) * ldc + row);
}

// Drives one panel per thread over a triangular n-by-n output.
inline void ForEachTriPanel(CBLAS_UPLO uplo, int n, const std::function<void(const TriPanel&)>& body)
{
    ForEachChunk(n, [&](int j0, int b) { body(MakeTriPanel(uplo, n, j0, b)); });
}

// Panels at or above this size use a GEMM-based diagonal block instead of a
// direct call into the single-threaded cblas_cXXXk routine, for CHERK, CSYRK
// and CHER2K alike -- see the block comment above Cher2kDiagBlock for why:
// measured on this OpenBLAS build, one cblas_c{herk,syrk,her2k} call scales far
// worse than the equivalent cblas_cgemm for the same output size, badly enough
// that a single diagonal panel of a large case can dominate the whole suite's
// wall time (and, transitively, everyone else's -- see the PreSmoke pod's
// exit-137 kills, which have landed on cherk/csyrk/chpr's own large panels).
//
// The threshold cannot be tuned against an assumed thread count: panel size is
// span/GoldenThreads() rounded up to PANEL_ALIGN, and GoldenThreads() varies
// with whatever cpuset the runner hands out, so the same n produces different
// panel sizes on different machines. What is fixed is PANEL_ALIGN itself: any
// span that does not fit in one thread's single panel (the ForEachChunk
// short-circuit) gets chunks of at least PANEL_ALIGN, so PANEL_ALIGN is the
// smallest panel size performance ever needed the fast path for. Below it, the
// only way to land there is a span small enough to run as one whole panel --
// e.g. n=32, where cherk/csyrk/cher2k each carry one RANDOM_EXTREME case
// (TC_FL_166/174/182) whose accuracy check leans on the reference overflowing
// to Inf at the same places the operator does. GEMM's one extra fp32 rounding
// step (see Cher2kOffBlock's comment) can shift where that happens, so those
// panels are kept on the exact single-call path where performance was never
// the problem anyway.
constexpr int DIAG_GEMM_THRESHOLD = PANEL_ALIGN;

// GEMM writes the full dense p.b x p.b diagonal square, but cblas_cXXXk's
// contract is to touch only the uplo-referenced triangle and leave the other
// half exactly as the caller had it (callers rely on this to catch
// out-of-footprint writes). CherkDiagBlock, CsyrkDiagBlock and Cher2kDiagBlock
// all share this fixup: save the non-referenced triangle before the GEMM(s),
// let them overwrite the whole square, then restore it. O(b^2) bookkeeping
// against the O(b^2*k) GEMM cost, not a measurable overhead.
inline std::vector<float> SaveOffTriangle(CBLAS_UPLO uplo, const float* dst, int b, int ldc)
{
    const bool upper = (uplo == CblasUpper);
    std::vector<float> saved(static_cast<size_t>(b) * static_cast<size_t>(b) * CPLX);
    size_t count = 0;
    for (int col = 0; col < b; ++col) {
        const int rowStart = upper ? col + 1 : 0;
        const int rowEnd = upper ? b : col;
        for (int row = rowStart; row < rowEnd; ++row) {
            const float* src = dst + CPLX * (static_cast<int64_t>(col) * ldc + row);
            saved[count++] = src[0];
            saved[count++] = src[1];
        }
    }
    return saved;
}

inline void RestoreOffTriangle(CBLAS_UPLO uplo, float* dst, int b, int ldc, const std::vector<float>& saved)
{
    const bool upper = (uplo == CblasUpper);
    size_t count = 0;
    for (int col = 0; col < b; ++col) {
        const int rowStart = upper ? col + 1 : 0;
        const int rowEnd = upper ? b : col;
        for (int row = rowStart; row < rowEnd; ++row) {
            float* d = dst + CPLX * (static_cast<int64_t>(col) * ldc + row);
            d[0] = saved[count++];
            d[1] = saved[count++];
        }
    }
}

// The diagonal block of one CHERK panel, as a single GEMM -- same formula the
// off-diagonal rectangle below already uses, just square. See
// SaveOffTriangle/RestoreOffTriangle for the non-referenced-triangle fixup,
// and why the diagonal's imaginary part is re-zeroed afterwards except at
// cblas_cherk's own alpha == 0 && beta == 1 quick-return, which this mirrors
// (real alpha here, rather than CHER2K's complex one).
inline void CherkDiagBlock(
    CBLAS_UPLO uplo, const TriPanel& p, bool noTrans, int k, float alpha, const float* A, int lda, float beta,
    float* C, int ldc)
{
    const float alphaC[CPLX] = {alpha, 0.0F};
    const float betaC[CPLX] = {beta, 0.0F};
    float* dst = TriBlock(C, ldc, p.j0, p.j0);

    const std::vector<float> saved = SaveOffTriangle(uplo, dst, p.b, ldc);
    cblas_cgemm(
        CblasColMajor, noTrans ? CblasNoTrans : CblasConjTrans, noTrans ? CblasConjTrans : CblasNoTrans, p.b, p.b, k,
        alphaC, RankKSub(A, lda, noTrans, p.j0), lda, RankKSub(A, lda, noTrans, p.j0), lda, betaC, dst, ldc);
    RestoreOffTriangle(uplo, dst, p.b, ldc, saved);

    if (!(alpha == 0.0F && beta == 1.0F)) {
        for (int i = 0; i < p.b; ++i) {
            dst[CPLX * (static_cast<int64_t>(i) * ldc + i) + 1] = 0.0F;
        }
    }
}

// C := alpha*A*A^H + beta*C (trans == CblasNoTrans) or alpha*A^H*A + beta*C.
inline void CherkThreaded(
    CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, float alpha, const float* A, int lda, float beta, float* C,
    int ldc)
{
    const bool noTrans = (trans == CblasNoTrans);
    const float alphaC[CPLX] = {alpha, 0.0F};
    const float betaC[CPLX] = {beta, 0.0F};
    ForEachTriPanel(uplo, n, [&](const TriPanel& p) {
        if (p.offRows > 0) {
            cblas_cgemm(
                CblasColMajor, noTrans ? CblasNoTrans : CblasConjTrans, noTrans ? CblasConjTrans : CblasNoTrans,
                p.offRows, p.b, k, alphaC, RankKSub(A, lda, noTrans, p.offRow0), lda, RankKSub(A, lda, noTrans, p.j0),
                lda, betaC, TriBlock(C, ldc, p.j0, p.offRow0), ldc);
        }
        if (p.b >= DIAG_GEMM_THRESHOLD) {
            CherkDiagBlock(uplo, p, noTrans, k, alpha, A, lda, beta, C, ldc);
        } else {
            cblas_cherk(
                CblasColMajor, uplo, trans, p.b, k, alpha, RankKSub(A, lda, noTrans, p.j0), lda, beta,
                TriBlock(C, ldc, p.j0, p.j0), ldc);
        }
    });
}

// The diagonal block of one CSYRK panel, as a single GEMM -- same shape as
// CherkDiagBlock, but C is complex-*symmetric* here rather than Hermitian, so
// there is no diagonal-imaginary normalization to reapply: the
// SaveOffTriangle/RestoreOffTriangle fixup is the whole difference from a
// plain cblas_csyrk call.
inline void CsyrkDiagBlock(
    CBLAS_UPLO uplo, const TriPanel& p, bool noTrans, int k, const float* alpha, const float* A, int lda,
    const float* beta, float* C, int ldc)
{
    float* dst = TriBlock(C, ldc, p.j0, p.j0);
    const std::vector<float> saved = SaveOffTriangle(uplo, dst, p.b, ldc);
    cblas_cgemm(
        CblasColMajor, noTrans ? CblasNoTrans : CblasTrans, noTrans ? CblasTrans : CblasNoTrans, p.b, p.b, k, alpha,
        RankKSub(A, lda, noTrans, p.j0), lda, RankKSub(A, lda, noTrans, p.j0), lda, beta, dst, ldc);
    RestoreOffTriangle(uplo, dst, p.b, ldc, saved);
}

// C := alpha*A*A^T + beta*C (trans == CblasNoTrans) or alpha*A^T*A + beta*C.
inline void CsyrkThreaded(
    CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, const float* alpha, const float* A, int lda,
    const float* beta, float* C, int ldc)
{
    const bool noTrans = (trans == CblasNoTrans);
    ForEachTriPanel(uplo, n, [&](const TriPanel& p) {
        if (p.offRows > 0) {
            cblas_cgemm(
                CblasColMajor, noTrans ? CblasNoTrans : CblasTrans, noTrans ? CblasTrans : CblasNoTrans, p.offRows, p.b,
                k, alpha, RankKSub(A, lda, noTrans, p.offRow0), lda, RankKSub(A, lda, noTrans, p.j0), lda, beta,
                TriBlock(C, ldc, p.j0, p.offRow0), ldc);
        }
        if (p.b >= DIAG_GEMM_THRESHOLD) {
            CsyrkDiagBlock(uplo, p, noTrans, k, alpha, A, lda, beta, C, ldc);
        } else {
            cblas_csyrk(
                CblasColMajor, uplo, trans, p.b, k, alpha, RankKSub(A, lda, noTrans, p.j0), lda, beta,
                TriBlock(C, ldc, p.j0, p.j0), ldc);
        }
    });
}

// The off-diagonal rectangle of one CHER2K panel, as two accumulating GEMMs.
// beta rides on the first call so it is applied exactly once.
inline void Cher2kOffBlock(
    const TriPanel& p, bool noTrans, int k, const float* alpha, const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc)
{
    const float alphaConj[CPLX] = {alpha[0], -alpha[1]};
    const float betaC[CPLX] = {beta, 0.0F};
    const float one[CPLX] = {1.0F, 0.0F};
    const CBLAS_TRANSPOSE opLeft = noTrans ? CblasNoTrans : CblasConjTrans;
    const CBLAS_TRANSPOSE opRight = noTrans ? CblasConjTrans : CblasNoTrans;
    float* dst = TriBlock(C, ldc, p.j0, p.offRow0);
    cblas_cgemm(
        CblasColMajor, opLeft, opRight, p.offRows, p.b, k, alpha, RankKSub(A, lda, noTrans, p.offRow0), lda,
        RankKSub(B, ldb, noTrans, p.j0), ldb, betaC, dst, ldc);
    cblas_cgemm(
        CblasColMajor, opLeft, opRight, p.offRows, p.b, k, alphaConj, RankKSub(B, ldb, noTrans, p.offRow0), ldb,
        RankKSub(A, lda, noTrans, p.j0), lda, one, dst, ldc);
}

// The diagonal block of one CHER2K panel, as two accumulating GEMMs -- same
// formula as Cher2kOffBlock, just square (both operands' row range is p.j0..
// p.j0+p.b instead of the off-diagonal's offRow0..offRow0+offRows). See
// SaveOffTriangle/RestoreOffTriangle for the non-referenced-triangle fixup GEMM
// needs that a fused cblas_cher2k call would not. The other property GEMM
// cannot give for free is the exact-zero imaginary diagonal cblas_cher2k
// guarantees by construction (C is Hermitian, so its diagonal is real) --
// fixed up by hand afterwards.
inline void Cher2kDiagBlock(
    CBLAS_UPLO uplo, const TriPanel& p, bool noTrans, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, float beta, float* C, int ldc)
{
    const float alphaConj[CPLX] = {alpha[0], -alpha[1]};
    const float betaC[CPLX] = {beta, 0.0F};
    const float one[CPLX] = {1.0F, 0.0F};
    const CBLAS_TRANSPOSE opLeft = noTrans ? CblasNoTrans : CblasConjTrans;
    const CBLAS_TRANSPOSE opRight = noTrans ? CblasConjTrans : CblasNoTrans;
    float* dst = TriBlock(C, ldc, p.j0, p.j0);

    const std::vector<float> saved = SaveOffTriangle(uplo, dst, p.b, ldc);
    cblas_cgemm(
        CblasColMajor, opLeft, opRight, p.b, p.b, k, alpha, RankKSub(A, lda, noTrans, p.j0), lda,
        RankKSub(B, ldb, noTrans, p.j0), ldb, betaC, dst, ldc);
    cblas_cgemm(
        CblasColMajor, opLeft, opRight, p.b, p.b, k, alphaConj, RankKSub(B, ldb, noTrans, p.j0), ldb,
        RankKSub(A, lda, noTrans, p.j0), lda, one, dst, ldc);
    RestoreOffTriangle(uplo, dst, p.b, ldc, saved);
    // cblas_cher2k normalizes the diagonal's imaginary part to zero after doing
    // any actual update. Its only exception is the BLAS quick-return case
    // alpha == 0 && beta == 1, which leaves C byte-for-byte untouched. Preserve
    // that exception while matching the normal update path for every other
    // alpha/beta combination (including alpha == 0 with beta != 1).
    const bool quickReturn = alpha[0] == 0.0F && alpha[1] == 0.0F && beta == 1.0F;
    if (!quickReturn) {
        for (int i = 0; i < p.b; ++i) {
            dst[CPLX * (static_cast<int64_t>(i) * ldc + i) + 1] = 0.0F;
        }
    }
}

// C := alpha*A*B^H + conj(alpha)*B*A^H + beta*C, or the trans == ConjTrans form.
inline void Cher2kThreaded(
    CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, const float* alpha, const float* A, int lda, const float* B,
    int ldb, float beta, float* C, int ldc)
{
    const bool noTrans = (trans == CblasNoTrans);
    ForEachTriPanel(uplo, n, [&](const TriPanel& p) {
        if (p.offRows > 0) {
            Cher2kOffBlock(p, noTrans, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        if (p.b >= DIAG_GEMM_THRESHOLD) {
            Cher2kDiagBlock(uplo, p, noTrans, k, alpha, A, lda, B, ldb, beta, C, ldc);
        } else {
            cblas_cher2k(
                CblasColMajor, uplo, trans, p.b, k, alpha, RankKSub(A, lda, noTrans, p.j0), lda,
                RankKSub(B, ldb, noTrans, p.j0), ldb, beta, TriBlock(C, ldc, p.j0, p.j0), ldc);
        }
    });
}

// C := alpha*A*B + beta*C (side == CblasLeft) or alpha*B*A + beta*C, A symmetric.
// The dense output splits along whichever extent leaves A whole: columns of C for
// a left-side product, rows of C for a right-side one.
inline void CsymmThreaded(
    CBLAS_SIDE side, CBLAS_UPLO uplo, int m, int n, const float* alpha, const float* A, int lda, const float* B,
    int ldb, const float* beta, float* C, int ldc)
{
    const bool left = (side == CblasLeft);
    ForEachChunk(left ? n : m, [&](int start, int len) {
        const int64_t bOff = left ? CPLX * static_cast<int64_t>(start) * ldb : CPLX * start;
        const int64_t cOff = left ? CPLX * static_cast<int64_t>(start) * ldc : CPLX * start;
        cblas_csymm(
            CblasColMajor, side, uplo, left ? m : len, left ? len : n, alpha, A, lda, B + bOff, ldb, beta, C + cOff,
            ldc);
    });
}

} // namespace blas_test

#endif // CBLAS_THREADED_H

