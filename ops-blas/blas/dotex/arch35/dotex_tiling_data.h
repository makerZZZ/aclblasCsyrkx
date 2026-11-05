/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DOTEX_TILING_DATA_H
#define DOTEX_TILING_DATA_H

#include <cstdint>

constexpr uint32_t DOTEX_XTYPE_FP32 = 0;
constexpr uint32_t DOTEX_XTYPE_FP16 = 1;
constexpr uint32_t DOTEX_XTYPE_BF16 = 2;

struct DotexTilingData {
    int n;
    int incx;
    int incy;
    int64_t kx;              // SIMT stride offset: (incx>=0) ? 0 : (1LL-n)*incx
    int64_t ky;              // SIMT stride offset: (incy>=0) ? 0 : (1LL-n)*incy
    uint32_t srcType;        // 0=FP32, 1=FP16, 2=BF16
    uint32_t useCoreNum;
    uint32_t numThreads;     // SIMT threads per core
};

#endif // DOTEX_TILING_DATA_H
