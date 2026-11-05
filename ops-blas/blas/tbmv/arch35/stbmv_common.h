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
 * \file stbmv_common.h
 * \brief single-precision tbmv common definitions for ascend950
 */

#pragma once

#include <cstddef>
#include <cstdint>

static constexpr uint32_t STBMV_UB_X_FLOATS = 16384; // 64 KiB UB budget for cached x window.
static constexpr uint32_t STBMV_FAST_TILE_FLOATS = 4096;

struct StbmvTilingData {
    uint32_t numThreads;
    uint32_t numBlocks;
    uint32_t rowsPerBlock;
    uint32_t useUb;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
    int64_t incx;
};

struct StbmvFastTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t uplo;
    uint32_t useCoreNum;
};

void stbmv_arch35_kernel_do(
    const float* a, float* x, uint8_t* workspace, const StbmvTilingData& tilingData, uint32_t numBlocks, void* stream);

int stbmv_arch35_simd_fastpath_kernel_do(
    const float* a, float* x, uint8_t* workspace, size_t workspaceSize, const StbmvTilingData& tilingData,
    const StbmvFastTilingData& fastTilingData, uint32_t numBlocks, void* stream);
