/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSPR2_TILING_DATA_H
#define SSPR2_TILING_DATA_H

#include <cstdint>

inline constexpr uint32_t UB_XY_FLOATS = 8192;

inline constexpr uint32_t UB_THRESHOLD = 128;

struct Sspr2TilingData {
    uint32_t n;
    uint32_t uplo;
    float    alpha;
    int64_t  incx;
    int64_t  incy;
};

#endif // SSPR2_TILING_DATA_H
