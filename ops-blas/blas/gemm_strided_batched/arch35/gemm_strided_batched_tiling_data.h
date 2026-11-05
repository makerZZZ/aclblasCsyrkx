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
 * \file gemm_strided_batched_tiling_data.h
 * \brief Tiling data structures shared between host and kernel for gemm_strided_batched.
 *
 * Strided Batched GEMM (FP32) on arch35 (DAV_3510) via the Blaze tensor_api path. Two tilings match the
 * two kernels: GemmSbGemmTilingData drives the AIC Blaze GEMM kernel, GemmSbCombineTilingData drives the
 * SIMD-membase AIV alpha/beta combine (and its beta-scale degenerate form). batch/stride are resolved to
 * per-batch GM offsets on the host side, so neither tiling carries a batch dimension. Both are passed to
 * their kernel by value (runtime launch-parameter copy), so no fields are core-indexed arrays (rule R4).
 *
 * Column-major swap (design section 4.3): the row-major engine computes C^T = op(B)^T op(A)^T, so the
 * engine-side dimensions are mEff = original n, nEff = original m, kEff = original k. The left matrix is
 * the original B (stride ldb), the right matrix is the original A (stride lda).
 */

#pragma once

#include <cstdint>

struct GemmSbGemmTilingData {
    // Engine-side logical scale after the column-major swap: mEff = original n, nEff = original m.
    uint32_t mEff;
    uint32_t nEff;
    uint32_t kEff;
    // Leading dimensions of the swapped operands (ldLeft = original ldb, ldRight = original lda).
    uint32_t ldLeft;
    uint32_t ldRight;
    // C / temp write-out row stride. Fast path: user ldc (must be a multiple of 8). General / degraded
    // path: CeilAlign(nEff, 8) = CeilAlign(original m, 8) (design HIGH-1 / MED-3).
    uint32_t ldC;
    // Transpose flags select the GM LayoutPtn (0 -> NDExt, 1 -> DNExt). transLeft derives from transB,
    // transRight derives from transA (design section 4.3).
    uint32_t transLeft;
    uint32_t transRight;
    // Multi-core and tile configuration (serpentine M/N tile split, K-chunk accumulation).
    uint32_t usedAicCoreNum;
    uint32_t singleCoreM;
    uint32_t singleCoreN;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t tileKChunk;
    uint32_t baseK;
};

struct GemmSbCombineTilingData {
    // Original (unswapped) C_i logical dimensions and user leading dimension (column-major).
    uint32_t m;
    uint32_t n;
    uint32_t ldc;
    // temp row stride = CeilAlign(original m, 8); must equal the GEMM ldC (design HIGH-1, section 3.3.2).
    uint32_t tempRowStride;
    float alpha;
    float beta;
    uint32_t hasBeta;
    // Column partitioning: each core owns a contiguous span of C columns; every column of temp (C^T row)
    // and of C is a contiguous length-m vector, so both reads and the write stay contiguous.
    uint32_t usedAivCoreNum;
    uint32_t tileLen;
};
