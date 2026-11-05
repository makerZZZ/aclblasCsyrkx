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
#include "cann_ops_blas_common.h"

// Cube (MMAD) alignment granularity on Ascend 950PR / DAV_3510.
constexpr uint32_t CSYMM_ARCH35_BASE_M = 16;
constexpr uint32_t CSYMM_ARCH35_BASE_N = 16;
constexpr uint32_t CSYMM_ARCH35_BASE_K = 8;
// FixPipe NZ2ND requires the N direction of the destination to be 8-aligned.
constexpr uint32_t CSYMM_ARCH35_FIXPIPE_N_ALIGN = 8;
constexpr uint32_t CSYMM_ARCH35_DEFAULT_TILE_M = 128;
constexpr uint32_t CSYMM_ARCH35_DEFAULT_TILE_N = 128;
constexpr uint32_t CSYMM_ARCH35_DEFAULT_TILE_K_CHUNK = 128;
constexpr uint32_t CSYMM_ARCH35_FP32_SIZE = sizeof(float);
constexpr uint32_t CSYMM_ARCH35_L1_SIZE_BYTES = 512 * 1024;
// C = alpha * (A * B) + beta * C, complex product expanded into 4 real MMADs:
//   term0 = Ar * Br, term1 = Ai * Bi, term2 = Ar * Bi, term3 = Ai * Br
//   prodReal = term0 - term1, prodImag = term2 + term3
constexpr uint32_t CSYMM_ARCH35_TERM_NUM = 4;

// Phase 1 tiling: split the COMPLEX64 operands into real / imaginary float planes.
//   A: symmetric triangle is mirrored into a full dense plane pair (no conjugation).
//   B: plain element-wise de-interleave.
struct CsymmPrepTilingData {
    uint32_t sideMode;
    uint32_t uploMode;
    uint32_t usedAivCoreNum;
    uint32_t dimA;
    uint32_t lda;
    uint32_t aRowsPerCore;
    uint32_t m;
    uint32_t n;
    uint32_t ldb;
    uint32_t bElemTotal;
    uint32_t bElemsPerCore;
    // Both jobs run inside a single asc_vf_call, so they share one thread count.
    uint32_t nthreads;
};

// Phase 2 tiling: one real MMAD (identical layout to the ssymm arch35 GEMM kernel).
struct CsymmGemmTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t sideMode;
    uint32_t usedAicCoreNum;
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t tileKChunk;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t tempRowStride;
};

// Phase 3 tiling: complex epilogue, C = alpha * prod + beta * C.
struct CsymmScaleTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t ldc;
    uint32_t tempRowStride;
    uint32_t usedAivCoreNum;
    // Work is split over the flattened column-major element space of C.
    uint32_t elemTotal;
    uint32_t scaleElemsPerCore;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    // 1: alpha == (0,0) -> C = beta * C only, the 4 temp planes are not read.
    uint32_t skipTemp;
    uint32_t nthreads;
};
