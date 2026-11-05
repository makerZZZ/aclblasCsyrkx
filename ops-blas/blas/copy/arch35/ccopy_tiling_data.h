/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCOPY_TILING_DATA_H
#define CCOPY_TILING_DATA_H

#include <cstdint>

struct CcopyTilingData {
    uint32_t totalN;           // logical complex element count
    uint32_t perCoreN;         // base complex elements per core, 32-byte aligned
    uint32_t extraBlockCores;  // cores receiving one additional aligned block
    uint32_t tailElements;     // remaining complex elements assigned to the last core
    uint32_t tileSize;         // UB tile size in logical complex elements
    uint32_t queueBufferCount; // one for a single-tile fast path, otherwise two
    int32_t incx;              // source stride in complex elements
    int32_t incy;              // destination stride in complex elements
};

constexpr uint32_t CCOPY_LANES_PER_COMPLEX = 2;
constexpr uint32_t CCOPY_BYTES_PER_COMPLEX = 2 * sizeof(uint32_t);
constexpr uint32_t CCOPY_LANES_PER_BLOCK = 32 / sizeof(uint32_t);
constexpr uint32_t CCOPY_COMPLEX_PER_BLOCK = 32 / CCOPY_BYTES_PER_COMPLEX;
constexpr uint32_t CCOPY_MAX_COMPACT_BLOCKS = 4092; // largest multiple of 4 not exceeding 4095

#endif                                              // CCOPY_TILING_DATA_H
