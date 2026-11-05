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

// Max elements per UB tile for FP32 ReduceMin (limited by repeatTimes <= 255)
constexpr uint32_t FP32_MAX_DATA_COUNT = 16320;

struct IsaminTilingData {
    uint32_t totalN;     // total element count
    uint32_t perCoreN;   // elements per core (first useCoreNum-1 cores)
    uint32_t lastCoreN;  // elements for the last core
    uint32_t useCoreNum; // actual core count used
    uint32_t tileSize;   // UB tile size per round
    uint32_t nthreads;   // SIMT thread count (used when incx != 1)
    uint32_t incx;       // stride of x (1 = contiguous)
};
