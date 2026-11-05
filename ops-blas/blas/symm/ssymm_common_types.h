/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

struct SsymmTileShape {
    uint32_t tm = 0;
    uint32_t tn = 0;
    uint32_t tk = 0;
};

inline constexpr uint32_t SSYMM_MAX_CORE_NUM = 64;
inline constexpr uint32_t SSYMM_RIGHT_TILE_M = 8;
inline constexpr uint32_t SSYMM_RIGHT_TILE_K = 64;
inline constexpr uint32_t SSYMM_RIGHT_DISPATCH_TILE_N = 64;
inline constexpr uint32_t SSYMM_RIGHT_FALLBACK_BUFFER_STRIDE_N = 128;

enum class SsymmBackendKind : uint32_t {
    GenericFallback = 0,
    LeftCube = 1,
    RightCube = 2,
};

struct SsymmProblemSpec {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t lda = 0;
    uint32_t ldb = 0;
    uint32_t ldc = 0;
    float alpha = 0.0f;
    float beta = 0.0f;
};

struct SsymmWorkspaceLayout {
    bool valid = true;
    uint64_t totalBytes = 0;
    uint64_t scratchBytes = 0;
    uint64_t densePackBytes = 0;
    uint64_t configBytes = 0;
};

struct SsymmExecutionPlan {
    SsymmProblemSpec spec{};
    SsymmBackendKind backend = SsymmBackendKind::GenericFallback;
    SsymmTileShape tile{};
    uint32_t coreNum = 1;
    bool regularDense = false;
    bool smallShapeFallback = false;
    SsymmWorkspaceLayout workspace{};
    uint32_t debugFlags = 0;
};

struct SsymmRuntimeTrace {
    SsymmBackendKind backendKind = SsymmBackendKind::GenericFallback;
};

// Host 侧填写并通过 aclrtMemcpy 传递给 Device 的 Tiling 参数结构体。
// Device 侧通过 GM 指针解析该结构体，字段布局 Host/Device 必须严格一致。
struct SsymmTilingData {
    uint32_t side;
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t kDim;
    uint32_t useCoreNum;
    uint32_t rightChunkMask;
    uint32_t rightChunkTn;
    uint32_t rightChunkTk;
    float alpha;
    float beta;
    uint32_t rowsPerCore;   // 每核基础分配量（向下取整）
    uint32_t rowRemainder;  // 剩余行数（前 remainder 个核多分配 1 行）
};

