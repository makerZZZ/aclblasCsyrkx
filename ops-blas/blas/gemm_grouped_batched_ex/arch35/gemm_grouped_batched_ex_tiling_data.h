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

enum GroupedGemmDtypeCase : int32_t {
    GROUPED_GEMM_FP16 = 0,
    GROUPED_GEMM_BF16,
    GROUPED_GEMM_FP8_E4M3_E4M3,
    GROUPED_GEMM_FP8_E5M2_E5M2,
    GROUPED_GEMM_FP8_E4M3_E5M2,
    GROUPED_GEMM_FP8_E5M2_E4M3,
};

// Header followed in GM by groupCount GroupedGemmGroupData records.
struct GroupedGemmTilingHeader {
    uint32_t groupCount;
    uint32_t problemCount;
    uint32_t totalCubeTasks;
    uint32_t totalEpilogueTasks;
    uint32_t epilogueTile;
    uint32_t reserved;
};

struct GroupedGemmGroupData {
    // Cube sees the column-major GEMM as row-major B^T * A^T.
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    int32_t isTransA;
    int32_t isTransB;
    int32_t mBlocks;
    int32_t nBlocks;
    int32_t singleCoreM;
    int32_t singleCoreN;

    uint32_t batchStart;
    uint32_t batchCount;
    uint32_t cubeTaskStart;
    uint32_t cubeTaskCount;
    uint32_t epilogueTaskStart;
    uint32_t epilogueTaskCount;

    // Original column-major C shape/stride used by the AIV epilogue.
    int32_t originalM;
    int32_t originalN;
    int32_t originalLdc;
    int32_t hasGemm;
    uint64_t workspaceOffset;
    float alpha;
    float beta;
};
