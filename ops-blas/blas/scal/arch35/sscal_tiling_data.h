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
#include "common/helper/kernel_constant.h"

constexpr uint32_t ELEMENTS_PER_BLOCK = 32 / sizeof(float);
constexpr uint32_t SSCAL_MAX_CORE_NUM = 64;

struct SscalTilingData {
    uint32_t totalN;
    uint32_t perCoreN;
    uint32_t remainder;
    uint32_t tileSize;
    float alpha;

    int64_t incx;
    uint32_t useCoreNum;
    uint32_t startOffset[SSCAL_MAX_CORE_NUM];
    uint32_t calCount[SSCAL_MAX_CORE_NUM];
    uint32_t nthreads;
};

void sscal_kernel_do(uint8_t* x, uint8_t* workSpace, const SscalTilingData& tiling,
                     uint32_t numBlocks, void *stream);

