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
 * \file csyrk_tiling_data.h
 * \brief Tiling data structures for aclblasCsyrk (arch35).
 *        Shared by host side (const ref) and kernel side (by value).
 *
 *        Pipeline:
 *          Phase 0 (AIV): deinterleave complex A -> Ar, Ai
 *          Phase 1 (AIC): 4M GEMM via tensor_api (syrk_gemm_arch35.h)
 *                         temp = [Q0|Q1|Q2|Q3] (full matrix, 4 quads)
 *          Phase 2 (AIV): combine C = alpha*(Cr + i*Ci) + beta*C_old (complex alpha/beta)
 */

#pragma once

#include <cstdint>

// [general] SIMD mode parameters for AIV kernels
constexpr uint32_t CSYRK_ARCH35_SCALE_BLOCK = 64;       // scale block size (rows/cols per tile)
constexpr uint32_t CSYRK_ARCH35_ELEMENTS_PER_BLOCK = 8; // 32 bytes / sizeof(float)

// Direct (single-kernel SIMT) path for tiny shapes: n<=64 && k<=256 (or any
// strided layout). One SIMT kernel computes every uplo element directly from A
// with the Netlib evaluation order (Kahan-compensated, special-value aware),
// avoiding the 3-kernel deint/cube/combine decomposition whose fixed launch
// cost dominates at these sizes (the reference passes TC_PF_1006..1014 this
// way at 2-22us).
struct CsyrkDirectTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldc;
    uint32_t transposed; // 1 = trans=T/C: A is k x n, dot of row-major pairs
    uint32_t upper;      // 1 = UPPER, 0 = LOWER
    uint32_t strided;    // 1 = lda/ldc have padding or k==0: per-element ref
    uint32_t checked;    // 1 = host pre-scanned A (small n*k): the kernel may
                         //     skip the per-element unsafe scan
};

// Phase 0: Deinterleave A (complex) -> Ar, Ai (real) Tiling (AIV-only)
struct CsyrkDeinterleaveTilingData {
    uint32_t nRows;       // A physical rows (elements per column)
    uint32_t nCols;       // A physical columns
    uint32_t lda;         // A leading dimension (in complex elements)
    uint32_t outLdc;      // output Ar/Ai leading dim (32B aligned)
    uint32_t colsPerCore; // columns per AIV core
};

// Phase 1: Fused 4M cube GEMM Tiling (AIC-only, tensor_api)
//   temp (4n x tempLdc) = [Q0 | Q1 | Q2 | Q3] stacked quads:
//     Q0 = Ar*Ar^T  Q1 = Ai*Ai^T  Q2 = Ar*Ai^T  Q3 = Ai*Ar^T
//   (transposed variant A^T*A for trans=T/C; OP_C maps to OP_T, no conjugation)
//   Full-matrix GEMM via SyrkGemmKernelImpl; combine stage consumes the uplo triangle.
struct CsyrkCubeTilingData {
    uint32_t n;           // C order
    uint32_t k;           // inner dimension (original)
    uint32_t arLdc;       // Ar/Ai leading dim (col-major rows)
    uint32_t tempLdc;     // temp quad leading dim (ceil-align n to 8)
    uint32_t usedCoreNum; // cube cores to launch
    uint32_t singleCoreM; // GEMM rows per core
    uint32_t singleCoreN; // GEMM cols per core
    uint32_t tileM;       // GEMM L0C tile rows
    uint32_t tileN;       // GEMM L0C tile cols
    uint32_t tileKChunk;  // GEMM K chunk per L1 stage
    uint8_t isTransN;     // 1 = trans=N (C = A*A^T), 0 = trans=T/C (C = A^T*A)
    uint8_t triangleMode; // 1 = upper triangle only, 2 = lower triangle only (0 = full)
    uint8_t hf32;         // 1 = HF32x3 compensated cube (default large-n path)
};

// Phase 2: Combine/Scale Tiling (AIV-only)
//   C_upper = alpha*(Cr + i*Ci) + beta*C_old  (complex alpha/beta)
struct CsyrkCombineTilingData {
    uint32_t n;            // C matrix order
    uint32_t ldc;          // C matrix leading dimension (in complex elements)
    uint32_t tempLdc;      // temp matrix leading dimension (= n)
    uint32_t k;            // inner dimension (diagonal reduction length)
    uint32_t arLdc;        // Ar/Ai leading dim (col-major rows)
    uint32_t rowsPerCore;  // rows per AIV core
    uint8_t isTransN;      // 1 = trans=N: diag reads Ar/Ai rows; 0 = trans=T/C: reads columns
    float alphaReal;       // alpha (complex) real part
    float alphaImag;       // alpha (complex) imag part
    float betaReal;        // beta (complex) real part
    float betaImag;        // beta (complex) imag part
    uint8_t uploMode;      // ACLBLAS_UPPER / ACLBLAS_LOWER
    uint8_t isAlphaZero;   // alpha == 0 short-circuit
    uint8_t isKZero;       // k == 0 short-circuit
    uint8_t isBetaZero;    // beta == 0 short-circuit
    uint8_t isFastCfg;     // alpha == (1,0) && beta == 0: 64x64 single-tile fast path
    uint8_t useDirectDiag; // 1 = recompute diagonal from Ar/Ai (large-k precision);
                           // 0 = use the quad value (small k, cheaper)
    // SIMT combine for small n: one thread per (i,j) of an n x n row band, no
    // TPipe/TBuf/pipe barriers. Small-n combine is dominated by fixed SIMD
    // overhead, which thread-level parallelism avoids. Only used for the fast
    // config (alpha==(1,0), beta==0) with n <= 256.
    uint8_t useSimt;      // 1 = launch the SIMT combine kernel
    uint32_t simtThreads; // threads per AIV block
};
