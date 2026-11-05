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

namespace CgercConfig {

// Ascend 950PR register width; Kernel checks this shared Host contract against the SDK definition.
constexpr uint32_t VECTOR_REGISTER_BYTES = 256U;
constexpr uint32_t FP32_VECTOR_LANES = VECTOR_REGISTER_BYTES / sizeof(float);
constexpr uint32_t UB_BLOCK_BYTES = 32U;
constexpr uint32_t A_BUFFER_COUNT = 3U;
constexpr uint32_t Y_ALIGNMENT_MARGIN = UB_BLOCK_BYTES - 1U;

} // namespace CgercConfig

struct CgercTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t numThreads;
    uint32_t rowBlocks;
    uint32_t colBlocks;
    int32_t incx;
    int32_t incy;
    float alphaReal;
    float alphaImag;
    uint32_t contiguous;
    uint32_t alphaOne;
    uint32_t vectorPath;
    uint32_t vectorAlignedRows;
    uint32_t vectorTileColumns;
};
