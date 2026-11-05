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
 * \file complex_blas3_tiling_data.h
 * \brief Tiling structures shared by the complex BLAS-3 operators on arch22
 *        (chemm / csymm / cherk / csyrk / cher2k).
 *
 *        All five operators run the same three-stage pipeline:
 *          Phase 0 (AIV) split   : complex input -> packed real Xr, Xi
 *          Phase 1 (AIC) gemm    : real FP32 GEMMs through the Matmul API
 *          Phase 2 (AIV) combine : reassemble the complex result, scale, write back
 *
 *        Phases 0 and 1 are operator-independent and their tiling structs live
 *        here. Phase 2 differs per operator (sign pattern, real vs complex
 *        alpha/beta, whether the diagonal imaginary part is forced to zero), so
 *        each operator declares its own combine struct.
 *
 *        Field layout must stay byte-identical between host and device.
 */

#ifndef COMPLEX_BLAS3_TILING_DATA_ARCH22_H
#define COMPLEX_BLAS3_TILING_DATA_ARCH22_H

#include <cstdint>

// Mirrors aclblasFillMode_t so kernels do not have to include the public host API
// header. Values follow the CBLAS convention.
constexpr uint32_t CBLAS3_UPLO_UPPER = 121;
constexpr uint32_t CBLAS3_UPLO_LOWER = 122;
// Sentinel for operators whose output is a full m x n matrix (chemm / csymm):
// no triangle is skipped.
constexpr uint32_t CBLAS3_UPLO_NONE = 0;

// Cube output tile and GEMM base block. Declared here so that the host tiling and
// the kernel's compile-time MatmulShapeParams read one single definition --
// keeping two copies in sync by hand is a silent-miscompute risk, because
// exceeding the static tiling's singleCoreK/M/N is not diagnosed at runtime.
//
// 128 x 128 x 64 is the largest block that keeps L0A, L0B and L0C double buffered
// for fp32; raising baseK to 128 turns dbL0B/dbL0C off and costs about 30%.
constexpr uint32_t CBLAS3_TILE_M = 128;
constexpr uint32_t CBLAS3_TILE_N = 128;
constexpr uint32_t CBLAS3_BASE_K = 64;

// Phase 0: de-interleave a complex matrix into separate real and imaginary parts.
//
// The input is column-major with leading dimension lda. One logical column is a
// contiguous run of 2*rows floats, so a single GatherMask pass over that run
// yields one whole column of Xr and one of Xi. Columns are handed to cores in a
// round robin, which keeps every core busy even when cols is barely above the
// core count.
//
// Shared unchanged by cherk / csyrk / cher2k. cher2k runs it twice (A and B).
struct CBlas3SplitTilingData {
    uint32_t rows;     // logical rows of the input
    uint32_t cols;     // logical cols of the input
    uint32_t lda;      // leading dimension of the input, in complex elements
    uint32_t packedLd; // row stride of the packed real output, in floats
    // Non-zero emits -Xi instead of Xi. Callers that pair the two halves along K
    // use it to fold a subtraction into the operand; zero (the default) leaves the
    // split unchanged.
    uint32_t negateImag;
    // Non-zero multiplies both halves by `scaleFactor`, which callers set to a
    // power of two to move the operand away from the top of the fp32 range before
    // it reaches the Cube. Zero (the default) leaves the split unchanged.
    uint32_t scaleEnabled;
    float scaleFactor;
};

// Phase 0 variant that emits the three K-concatenated operands the rank-K
// operators need. Writing them here rather than assembling them later matters for
// accuracy, not just for launch count.
//
// With Xr, Xi the real and imaginary parts of the input, define along the K axis
//   P = [Xr | Xi]      Q = [Xr | -Xi]      R = [Xi | Xr]
// each of logical K extent 2k. Then
//   cherk:  Cr = P*P^T   Ci = R*Q^T
//   csyrk:  Cr = P*Q^T   Ci = P*R^T
// so both operators need exactly these three buffers and two GEMMs instead of
// four.
//
// Why this is worth the extra buffer: computing e.g. Ar*Ar^T - Ai*Ai^T as two
// separate GEMMs subtracts two independently rounded fp32 results whose own
// magnitude is O(k) even when their difference is O(sqrt(k)). At k=8000 that
// costs two orders of magnitude of relative accuracy on the difference. Folding
// the sign into the operand makes the cancellation happen inside one Cube
// accumulation chain instead.
//
// Layout: `blockStride` is the distance between the two halves and `packedLd` the
// row stride, both in floats. The two differ per trans mode because Phase 0 must
// keep its writes contiguous:
//   trans='N': the packed view is (k x n) row-major with row stride n, so the two
//              halves stack as whole blocks -> packedLd = n, blockStride = k*n
//   trans='T': the packed view is (n x k) row-major with row stride k, so the two
//              halves sit side by side within each row -> packedLd = 2k,
//              blockStride = k
struct CBlas3SplitConcatTilingData {
    uint32_t rows;        // logical rows of the input
    uint32_t cols;        // logical cols of the input
    uint32_t lda;         // leading dimension of the input, in complex elements
    uint32_t packedLd;    // row stride of each concatenated buffer, in floats
    uint32_t blockStride; // offset from the first half to the second, in floats
    uint32_t colStride;   // offset between consecutive input columns, in floats
};

// Which operand the GEMM must transpose.
//
// Phase 0 always emits tightly packed column-major real matrices. Read as a
// row-major matrix, that packed buffer is the transpose of the logical matrix,
// so a Gram product needs exactly one transposed operand, and which one depends
// on the trans mode:
//
//   trans='N': C = A*A^H, packed = A^T (k x n row-major) -> transpose left
//   trans='C': C = A^H*A, packed = A^T (n x k row-major) -> transpose right
//
// chemm / csymm need neither: their output is assembled through the identity
//   C^T = (A*B)^T = B^T*A^T
// and both packed operands are already the transposes the row-major Matmul wants,
// so all four of their GEMMs run with plain operands.
enum CBlas3GemmTransMode : uint32_t {
    CBLAS3_GEMM_TRANS_LEFT = 0,
    CBLAS3_GEMM_TRANS_RIGHT = 1,
    CBLAS3_GEMM_TRANS_NONE = 2,
};

// Side of the symm-family operators, mirroring aclblasSideMode_t.
constexpr uint32_t CBLAS3_SIDE_LEFT = 141;
constexpr uint32_t CBLAS3_SIDE_RIGHT = 142;

// Edge of the square tile the triangular expansion works in. 64 x 64 fp32 is 16KB,
// so source, destination and the Gather index all fit in UB several times over,
// and a tile row is 256B -- a whole DataCopyPad burst, which a row-at-a-time
// mirror would not achieve (4B per burst).
constexpr uint32_t CBLAS3_EXPAND_TILE = 64;

// Phase 0b for chemm / csymm: mirror a triangular-stored square matrix into a
// full one, in place on the already-split real buffers (Phase 0a already wrote
// every element; this fills the unreferenced triangle from its mirror, since
// the Matmul API needs a dense operand).
//
// Each tile is classified against the uplo triangle: inside (already correct,
// skipped), outside (read + transpose the mirror tile), or on the diagonal
// (I == J; same as outside plus a per-element select of which side to keep).
//
// imagSign is the only difference between the two operators in this stage:
//   csymm: A = A^T -> imagSign = +1 (Ar, Ai both symmetric)
//   chemm: A = A^H -> imagSign = -1 (Ai antisymmetric; its diagonal, unused by
//          BLAS, is forced to zero)
// See the design doc for the full tiling/index derivation.
struct CBlas3ExpandTilingData {
    uint32_t d;         // A is d x d
    uint32_t packedLd;  // row stride of Ar / Ai, in floats
    uint32_t uploMode;  // CBLAS3_UPLO_UPPER / CBLAS3_UPLO_LOWER
    uint32_t tileCount; // tiles per side = ceil(d / CBLAS3_EXPAND_TILE)
    int32_t imagSign;   // +1 symmetric (csymm), -1 Hermitian (chemm)
};

// Phase 1: one real GEMM launch. The host issues it once per product it needs
// (cherk/csyrk: 4 times, cher2k: 8 times, chemm/csymm: 4 times).
//
// Shared unchanged by all five operators; uploMode = CBLAS3_UPLO_NONE disables
// the triangle skip for the symm-family operators.
struct CBlas3GemmTilingData {
    uint32_t m;        // rows of the output
    uint32_t n;        // cols of the output
    uint32_t k;        // full reduction length
    uint32_t packedLd; // row stride of both packed operands, in floats
    uint32_t ldc;      // row stride of the temp output, in floats
    uint32_t mBlocks;
    uint32_t nBlocks;
    uint32_t kBase;     // K-chunk offset
    uint32_t kCount;    // K-chunk length, <= the kernel's compile-time singleK
    uint32_t enAtomic;  // 0 = overwrite temp, 1 = atomically accumulate
    uint32_t uploMode;  // CBLAS3_UPLO_UPPER / _LOWER / _NONE
    uint32_t transMode; // CBlas3GemmTransMode
    // Only read when transMode == CBLAS3_GEMM_TRANS_NONE, where the two operands
    // do not share a row stride: the left one is (m x k) row-major with stride ldA
    // and the right one (k x n) row-major with stride ldB. The Gram products keep
    // using packedLd for both.
    uint32_t ldA;
    uint32_t ldB;
};

#endif // COMPLEX_BLAS3_TILING_DATA_ARCH22_H
