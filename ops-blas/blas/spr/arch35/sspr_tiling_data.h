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

// UB-x path x vector cache size (32 KB = 8192 floats). 32KB leaves 216KB DCache for AP streaming.
// Ascend950 profiling shows 32KB optimal across all n values (48KB/64KB both underperform).
inline constexpr uint32_t UB_X_FLOATS = 8192;

// n >= UB_THRESHOLD enables UB-x path. Ascend950 profiling (SPR_PROFILE_REPEAT=5, median):
// n<128: DCache natural caching suffices, GM path faster
// n>=128: DCache contention becomes significant, UB-x isolation wins
// Both UPPER and LOWER share the same threshold.
inline constexpr uint32_t UB_THRESHOLD = 128;

struct SsprTilingData {
    uint32_t numThreads;         // threads per block
    uint32_t columnsPerBlock;    // columns per block
    uint32_t n;                  // matrix order
    uint32_t uplo;               // ACLBLAS_UPPER(121) or ACLBLAS_LOWER(122)
    float    alpha;              // scalar alpha (dereferenced from pointer)
    int64_t  incx;               // x vector stride
};
