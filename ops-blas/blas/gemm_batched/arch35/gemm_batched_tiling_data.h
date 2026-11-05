/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

constexpr uint32_t GEMM_BATCHED_FP32_C0 = 8;
constexpr uint32_t GEMM_BATCHED_FP16_C0 = 16;
constexpr uint32_t GEMM_BATCHED_FRACTAL = 16;
constexpr uint32_t GEMM_BATCHED_L0C_C0 = 16;
constexpr uint32_t GEMM_BATCHED_L1_BUF_NUM = 2;
constexpr uint32_t GEMM_BATCHED_L1_BUF_MASK = GEMM_BATCHED_L1_BUF_NUM - 1;
constexpr uint32_t GEMM_BATCHED_DEFAULT_TILE_M = 128;
constexpr uint32_t GEMM_BATCHED_DEFAULT_TILE_N = 128;
constexpr uint32_t GEMM_BATCHED_DEFAULT_TILE_K_CHUNK = 128;
constexpr uint32_t GEMM_BATCHED_BASE_M = 16;
constexpr uint32_t GEMM_BATCHED_BASE_N = 16;
// L1 (arch35 DAV_3510) is partitioned into 2 independent banks of 256 KB each.
// Ping/Pong buffers use bank-isolated layout: Ping at offset 0 (bank 0),
// Pong at offset TOTAL_L1_SIZE/2 (bank 1). Within each bank A and B are
// contiguous: [Ping A][Ping B] | [Pong A][Pong B].
// This avoids L1 bank conflicts during MTE1 transfers.
// Reference: cann-samples/memory_optimization/l1_bank_conflict

constexpr uint32_t GEMM_BATCHED_FP32_L0_BASE_K = 16;
constexpr uint32_t GEMM_BATCHED_FP16_L0_BASE_K = 64;
constexpr uint32_t GEMM_BATCHED_UB_ALIGN_BYTES = 32;

enum GemmBatchedDTypeCase : int32_t {
    GEMM_BATCHED_DTYPE_FP32 = 0,
    GEMM_BATCHED_DTYPE_FP16 = 1,
    GEMM_BATCHED_DTYPE_BF16 = 2,
    GEMM_BATCHED_DTYPE_INVALID = -1
};

struct GemmBatchedGemmTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t usedAicCoreNum;
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint32_t mBlocks;
    uint32_t nBlocks;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t tileKChunk;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t isTransA;
    uint32_t isTransB;
    uint32_t batchCount;
    uint32_t totalTasks;
    int32_t dtypeCase;
    uint32_t balancedSplit;
};

struct GemmBatchedAlphaBetaTilingData {
    int32_t m;
    int32_t n;
    int32_t ldc;
    int32_t tempRowStride;
    float alpha;
    float beta;
    int32_t hasBeta;
    int32_t batchCount;
    int32_t dtypeCase;
    int32_t usedAivCoreNum;
    int64_t totalCols;
};

// ============================================================================
// Complex GEMM (cgemm_batched) tiling structures
// ============================================================================

// Deinterleave kernel: split interleaved complex matrix (real,imag,real,imag...)
// into two contiguous float matrices (real part and imag part).
// Layout: complex element [i][j] stored at (i*ld + j) as aclblasComplex.
// Output: realMat[i*ld + j], imagMat[i*ld + j] as float.
struct CgemmBatchedDeinterleaveTilingData {
    int32_t m;           // physical rows of the matrix
    int32_t k;           // physical columns of the matrix
    int32_t lda;         // leading dimension (in complex elements)
    int32_t batchCount;
    int32_t isConjugate; // 1 if imaginary part should be negated (for conj-transpose C)
    int32_t usedAivCoreNum;
};

// Combine kernel: given four real GEMM results T1=Ar*Br, T2=Ai*Bi, T3=Ar*Bi, T4=Ai*Br,
// compute P_r = T1 - T2, P_i = T3 + T4, then
//   C_r = alpha_r*P_r - alpha_i*P_i + beta_r*C_orig_r - beta_i*C_orig_i
//   C_i = alpha_r*P_i + alpha_i*P_r + beta_r*C_orig_i + beta_i*C_orig_r
// and write back interleaved complex C.
// T1..T4 are stored as float matrices with stride = tempRowStride (= m, column-major packed).
// C_orig / C_out are interleaved aclblasComplex with leading dimension ldc.
struct CgemmBatchedCombineTilingData {
    int32_t m;
    int32_t n;
    int32_t ldc;           // leading dimension of C (in complex elements)
    int32_t tempRowStride; // stride of T1..T4 (= ceilAlign(m, L0C_C0))
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    int32_t hasBeta; // 1 if (betaReal!=0 || betaImag!=0)
    int32_t batchCount;
    int32_t usedAivCoreNum;
    int64_t totalCols; // batchCount * n
};

// Direct SIMT complex accumulation for skinny logical outputs.  Keeping the
// complex multiply-add intact avoids the cancellation introduced by the 4M
// decomposition when each batch contains only a few output values.
struct CgemmBatchedDirectTilingData {
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    int32_t transA; // 0: N, 1: T, 2: conjugate transpose
    int32_t transB; // 0: N, 1: T, 2: conjugate transpose
    int32_t batchCount;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
};

// Fast path for the benchmark's compact N/N, alpha=1, beta=0 cases.
// A complex (m x k) is expanded into a strict-FP32 real matrix (2m x 2k):
//   [ ar -ai ]
//   [ ai  ar ]
// while interleaved B is already a real (2k x n) matrix.  One real GEMM then
// writes the interleaved (2m x n) complex result directly.
struct CgemmBatchedRealifyATilingData {
    int32_t m;
    int32_t k;
    int32_t lda;
    int32_t batchCount;
    uint32_t dataOffsetBytes; // 0 selects the legacy one-pointer-array layout
};

struct CgemmBatchedSplitKPointerTilingData {
    uint32_t batchCount;
    uint32_t alignedPtrBytes;
    uint64_t aOffsetBytes;
    uint64_t bOffsetBytes;
    uint64_t tempDataOffsetBytes;
    uint64_t tempPerBatchBytes;
};
// Generic packer used by the one-real-GEMM path when either operand is
// transposed/conjugate-transposed.  It materializes R(op(A)) as (2m x 2k)
// and the interleaved op(B) as a real (2k x n) matrix.
struct CgemmBatchedRealifyGenericTilingData {
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t transA;
    int32_t transB;
    int32_t batchCount;
    int32_t packA;
    int32_t packB;
    int32_t chunkedSpatial;
    int32_t tileTransA;
    int32_t tileTransB;
    int32_t transposeTile;
};
