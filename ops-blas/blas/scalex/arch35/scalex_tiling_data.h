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

struct ScalexTilingData {
    // SIMD path (incx==1)
    uint32_t totalN;
    uint32_t perCoreN;
    uint32_t remainder;
    uint32_t tileSize;
    float alpha;
    uint32_t alphaIsDevice;

    // SIMT path (incx!=1)
    int64_t incx;
    uint32_t nthreads;
    uint32_t numBlocks;

    // shared
    uint32_t xType;  // aclDataType value
};

