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

struct RotExTilingData {
    // 公共参数
    int32_t n;
    int32_t incx;
    int32_t incy;
    int64_t kx;
    int64_t ky;
    uint32_t tilingKey;         // 0=SIMD S组, 2=SIMT
    uint32_t executionType;     // aclDataType 枚举值
    uint32_t xType;             // x 数据类型 (aclDataType 枚举值)
    uint32_t yType;             // y 数据类型 (aclDataType 枚举值)
    uint32_t csType;            // c/s 数据类型 (aclDataType 枚举值)

    // c/s 标量值 (由 host 读取嵌入，S 组仅实数部分有效)
    float cReal;
    float sReal;

    // SIMD 路径参数 (TilingKey 0)
    uint32_t tileSize;
    uint32_t perCoreN;
    uint32_t remainder;

    // SIMT 路径参数 (TilingKey 2)
    uint32_t nthreads;
};
