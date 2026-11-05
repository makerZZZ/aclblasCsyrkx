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

constexpr int32_t GEMM_BASE_M = 32;
constexpr int32_t GEMM_BASE_K = 8;
constexpr int32_t GEMM_BASE_N = 16;
constexpr int32_t GEMM_C0_SIZE = 8;
constexpr int32_t GEMM_FRACTAL = 16;
constexpr int32_t GEMM_TILE_K_CHUNK = 64;

struct GemmTilingData {
    // Dimension fields
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    int32_t cLdc;
    // Tiling fields
    int32_t usedCoreNum;
    int32_t mBlocks;
    int32_t nBlocks;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    int32_t tileKChunk;
    int32_t c0Size;
    int32_t isTransA;
    int32_t isTransB;
    int32_t hasBeta;
    // Float fields grouped together for alignment
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
    // Cgemm extension fields are appended to preserve existing field offsets
    int32_t cgemmKPartitions;
    int32_t cgemmKSpan;
    int32_t cgemmThreeProducts;
};
