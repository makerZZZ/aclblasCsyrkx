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

inline constexpr uint32_t CGERU_SIMT_THREADS = 512;
inline constexpr uint32_t CGERU_CACHE_COMPLEX = 4096;
inline constexpr uint32_t CGERU_COMPLEX_COMPONENTS = 2;
inline constexpr uint64_t CGERU_MIN_ELEMENTS_PER_BLOCK = 1024;
inline constexpr uint32_t CGERU_VECTOR_MAX_ROWS = 4096;
inline constexpr uint32_t CGERU_CONTIGUOUS_UB_BYTES = 240U * 1024U;
inline constexpr uint32_t CGERU_MAX_DMA_BLOCKS = 4095;
inline constexpr uint32_t CGERU_MAX_STRIDED_DMA_BLOCKS = 32;
inline constexpr uint32_t CGERU_TILING_CACHE_X_AND_P = 0;
inline constexpr uint32_t CGERU_TILING_DIRECT_GM = 1;
inline constexpr uint32_t CGERU_TILING_CONTIGUOUS_REG = 2;

struct CgeruTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t rowBlocks;
    uint32_t colBlocks;
    uint32_t tilingKey;
    float alphaReal;
    float alphaImag;
    int64_t incx;
    int64_t incy;
    uint64_t xStart;
    uint64_t yStart;
};
