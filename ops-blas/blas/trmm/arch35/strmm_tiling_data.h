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

constexpr uint32_t STRMM_ARCH35_BASE_M = 16;
constexpr uint32_t STRMM_ARCH35_BASE_N = 16;
constexpr uint32_t STRMM_ARCH35_BASE_K = 8;
constexpr uint32_t STRMM_ARCH35_FIXPIPE_N_ALIGN = 8;
constexpr uint32_t STRMM_ARCH35_DEFAULT_TILE_M = 128;
constexpr uint32_t STRMM_ARCH35_DEFAULT_TILE_N = 128;
constexpr uint32_t STRMM_ARCH35_DEFAULT_TILE_K_CHUNK = 256;
constexpr uint32_t STRMM_ARCH35_FP32_SIZE = sizeof(float);
constexpr uint32_t STRMM_ARCH35_L1_SIZE_BYTES = 512 * 1024;
constexpr uint32_t STRMM_ARCH35_L1_BUF_NUM = 2;

struct StrmmMirrorTilingData {
    uint32_t sideMode;
    uint32_t uploMode;
    uint32_t transMode;
    uint32_t diagMode;
    uint32_t usedAivCoreNum;
    uint32_t mirrorRowsPerCore;
    uint32_t lda;
    uint32_t dimA;
    uint32_t nthreads;
};

struct StrmmGemmTilingData {
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
    uint32_t tempRowStride;
};

struct StrmmScaleTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t ldc;
    uint32_t tempRowStride;
    uint32_t usedAivCoreNum;
    uint32_t scaleRowsPerCore;
    float alphaVal;
    uint32_t nthreads;
};
