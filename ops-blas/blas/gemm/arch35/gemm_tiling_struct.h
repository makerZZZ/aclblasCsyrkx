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

// Unified FP32 GEMM tiling descriptor shared by the gemm operator and the
// csymm operator's cube / 3m GEMM kernels (arch35). The two operators used to
// keep independent copies of this struct, which tripped the duplicate-code
// scan.
//
// csymm needs four extra fields (kStart / kEnd / kSegmentCount /
// maxTilesPerCore) that gemm does not read. They are appended AFTER the common
// float fields so the byte offsets of every field gemm uses stay identical:
// gemm serializes this struct by value into its device kernel, so any shift in
// the common-field layout would break it. csymm recompiles against this same
// definition, so its relocated extra fields remain self-consistent (it only
// ever touches them by name).
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
    // --- csymm (symm arch35) extensions; appended to preserve gemm layout ---
    // Max baseM x baseN tiles executed by the busiest core (host-side estimate
    // used to pick the core grid; 0 when unavailable).
    int32_t maxTilesPerCore;
    // K-range for this GEMM (global K coordinates). Splitting K into two
    // segments halves the magnitude of each partial sum, which cuts the
    // cancellation error of the 3m combine (W1-W2-W3) roughly in half.
    int32_t kStart;
    int32_t kEnd;
    // 1 or 2: when 2, the kernel writes segment 0 to c and segment 1 to c1 in
    // a single launch (saves 3 extra kernel launches across the 3m GEMMs).
    int32_t kSegmentCount;
};
