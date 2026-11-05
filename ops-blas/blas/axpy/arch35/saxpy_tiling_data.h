/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SAXPY_TILING_DATA_H
#define SAXPY_TILING_DATA_H

#include <cstdint>
#include "common/helper/kernel_constant.h"

constexpr uint32_t SAXPY_ELEMENTS_PER_BLOCK = 32 / sizeof(float);
constexpr uint32_t SAXPY_MAX_CORE_NUM = 64;
constexpr uint32_t SAXPY_PIPE_DEPTH = 2;

struct SaxpyTilingData {
    uint32_t totalN;
    uint32_t perCoreN;
    uint32_t remainder;
    uint32_t tileSize;
    float alpha;

    int64_t incx;
    int64_t incy;
    uint32_t useCoreNum;
    uint32_t startOffset[SAXPY_MAX_CORE_NUM];
    uint32_t calCount[SAXPY_MAX_CORE_NUM];
    uint32_t nthreads;
};

#endif
