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

struct AxpyExTilingData {
    // SIMD path (incx==1 && incy==1)
    uint32_t totalN;    // total element count
    uint32_t perCoreN;  // per-core base count (aligned to alignUnit)
    uint32_t remainder; // tail core extra elements
    uint32_t tileSize;  // UB tile size in elements

    // SIMT path (incx!=1 || incy!=1)
    int64_t incx;         // x stride (signed, BLAS semantics)
    int64_t incy;         // y stride (signed, BLAS semantics)
    int64_t startOffsetX; // x start offset: (incx>=0)?0:(1-n)*incx
    int64_t startOffsetY; // y start offset: (incy>=0)?0:(1-n)*incy
    uint32_t nthreads;    // SIMT threads per block
    uint32_t numBlocks;   // SIMT block count (also used as launch grid)

    // shared
    float alpha;            // alpha scalar (valid when alphaIsDevice==0)
    uint32_t alphaIsDevice; // 1=alpha in device memory, 0=host
    uint32_t xType;         // aclDataType value (ACL_FLOAT/ACL_FLOAT16/ACL_BF16)
};
