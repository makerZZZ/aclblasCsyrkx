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

// Complex elements per UB tile for the unit-stride (incx==1) fold. The pipeline is
// double-buffered (2 live slots of 2*tileSize floats each) and each tile also needs
// a magnitude scratch (re/im/mask, ~2.25*tileSize*4 B) and a per-tile result slot.
// A small tile keeps many tiles in flight so the MTE2 copy of tile t+1 overlaps the
// V-pipe Abs/DeInterleave/Add/ReduceMin of tile t. 4096 complex = 32 KB per slot,
// ~117 KB total against the 248 KB UB of dav_3510, and 2*4096=8192 floats is well
// under the DataCopy 32768-float cap.
constexpr uint32_t ICAMIN_TILE_COMPLEX = 4096;

struct IcaminTilingData {
    uint32_t totalN;     // total complex element count (logical length of x)
    uint32_t perCoreN;   // complex elements per core (first useCoreNum-1 cores)
    uint32_t lastCoreN;  // complex elements for the last core
    uint32_t useCoreNum; // actual core count used
    uint32_t tileSize;   // complex elements per UB tile round
    uint32_t nthreads;   // SIMT thread count per block (strided scan and NaN fallback)
    uint32_t incx;       // stride of x in complex elements (1 = contiguous)
};
