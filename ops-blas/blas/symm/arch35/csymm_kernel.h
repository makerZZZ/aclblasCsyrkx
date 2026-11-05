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

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

// FP32 GEMM tiling descriptor shared with the gemm operator (see
// gemm/arch35/gemm_tiling_struct.h). csymm carries four extra fields there
// (kStart / kEnd / kSegmentCount / maxTilesPerCore); the shared struct keeps
// gemm's common fields byte-identical so gemm's by-value kernel serialization
// is unaffected, while csymm only touches its extra fields by name.
#include "gemm/arch35/gemm_tiling_struct.h"

// Small/mid-size default tile (large matrices are dispatched to 128x128x64
// in GemmCubeKernelDispatch based on host-side tiling.baseM/baseK).
constexpr int32_t GEMM_BASE_M = 64;
constexpr int32_t GEMM_BASE_K = 128;
constexpr int32_t GEMM_BASE_N = 64;
constexpr int32_t GEMM_C0_SIZE = 8;
constexpr int32_t GEMM_FRACTAL = 16;
constexpr int32_t GEMM_TILE_K_CHUNK = 512;

// Prep kernel: mirror the symmetric A (half-stored, complex) into a full
// real/imag plane Ar/Ai and de-interleave B into Br/Bi. Also emits the
// 3m sum planes Aplus = Ar+Ai and Bplus = Br+Bi. All on device,
// column-major, coalesced writes.
//   Ar, Ai, Aplus : S x S  (S = M for LEFT, N for RIGHT), leading dim S
//   Br, Bi, Bplus : M x N,                                leading dim M
void csymm_prep_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR A, int32_t lda, int32_t uplo, GM_ADDR B, int32_t ldb,
    GM_ADDR Ar, GM_ADDR Ai, GM_ADDR Br, GM_ADDR Bi,
    GM_ADDR Aplus, GM_ADDR Bplus,
    int32_t M, int32_t N, int32_t S,
    int32_t aStart, int32_t aEnd);

// Cube FP32 GEMM kernel launcher (csymm-owned copy of the proven gemm cube).
// c1 receives K-segment 1 when tiling.kSegmentCount == 2 (otherwise unused).
void csymm_cube_gemm_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR c1,
    const GemmTilingData& tiling);

// Fused 3-GEMM launcher: one launch of 3 * perGemmBlocks blocks runs the three
// 3m GEMMs (w1/w2/w3) concurrently across the AIC cores (see
// csymm_cube_gemm3_kernel). Same tiling for all three; planes differ.
void csymm_cube_gemm3_do(
    uint32_t perGemmBlocks, void* stream,
    GM_ADDR a0, GM_ADDR b0, GM_ADDR c0, GM_ADDR d0,
    GM_ADDR a1, GM_ADDR b1, GM_ADDR c1_, GM_ADDR d1,
    GM_ADDR a2, GM_ADDR b2, GM_ADDR c2, GM_ADDR d2,
    const GemmTilingData& tiling);


// Small-shape single-launch direct kernel launcher (m <= 64 && n <= 64):
// C = alpha*A*B + beta*C computed entirely on AIV in one launch, mirroring the
// half-stored symmetric A on the fly. side: 0 = LEFT, 1 = RIGHT.
//   LEFT : C = A*B, A is SxS (S = m), B is m x n
//   RIGHT: C = B*A, A is SxS (S = n), B is m x n
void csymm_small_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR A, int32_t lda, int32_t uplo, GM_ADDR B, int32_t ldb,
    GM_ADDR C, int32_t ldc,
    int32_t m, int32_t n, int32_t S, int32_t side,
    float ar, float ai, float br, float bi);

// Scale kernel launcher: C = beta * C (in-place), for alpha==0 / k==0 path.
void csymm_scale_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR cInOut, int32_t m, int32_t n, int32_t ldc,
    float betaReal, float betaImag, int32_t isComplex);

// Combine kernel launcher: merge the 3m GEMM results (optionally K-split into
// two segments per plane) into complex C.
//   kSplit==0: Re = w2-w3,             Im = w1-w2-w3
//   kSplit==1: Re = (w2a-w3a)+(w2b-w3b), Im = (w1a-w2a-w3a)+(w1b-w2b-w3b)
// C = alpha*Re + i*alpha*Im + beta*C
void csymm_combine_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR w1a, GM_ADDR w2a, GM_ADDR w3a,
    GM_ADDR w1b, GM_ADDR w2b, GM_ADDR w3b, int32_t kSplit, int32_t tempLdc,
    GM_ADDR cInOut, int32_t m, int32_t n, int32_t ldc,
    float ar, float ai, float br, float bi);
