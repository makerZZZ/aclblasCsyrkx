/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCOPY_TILING_DATA_H
#define SCOPY_TILING_DATA_H

#include <cstdint>

struct ScopyTilingData {
    uint32_t totalN;          // total element count
    uint32_t perCoreN;        // base elements per core, ELEMENTS_PER_BLOCK aligned
    uint32_t extraBlockCores; // count of cores that get +1 aligned block each
    uint32_t tailElements;    // unaligned elements (0..7), assigned to last core
    uint32_t tileSize;        // UB tile size, aligned to ELEMENTS_PER_BLOCK
    int32_t incx;             // source stride (elements), 1 for continuous path
    int32_t incy;             // destination stride (elements), 1 for continuous path
};

// elements per 32-byte block for FP32 (= 8)
constexpr uint32_t ELEMENTS_PER_BLOCK = 32 / sizeof(float);

#endif // SCOPY_TILING_DATA_H
