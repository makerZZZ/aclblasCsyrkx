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

// Maximum AIV cores on a single ascend950 (arch35) chip.
constexpr uint32_t SNRM2_MAX_CORE_NUM = 64;

// Reduce kernel sums partial-sum slots rounded up to one UB block (32B = 8 floats).
// Host-side workspace sizing and the kernel memset must cover this padded count,
// matching ELEMENTS_PER_BLOCK in snrm2_kernel.cpp.
constexpr uint32_t SNRM2_WORKSPACE_ALIGN_FLOATS = 8;

// @brief TilingData for aclblasSnrm2 (arch35).
//        Follows R4: no arrays -- kernel computes per-core offset/count from scalar fields.
struct Snrm2TilingData {
    // === Shared ===
    int32_t n;              ///< total vector elements
    int32_t incx;           ///< stride (incx==1 -> SIMD, incx!=1 -> SIMT)
    uint32_t useCoreNum;    ///< actual AIV core count used

    // === SIMD path (incx==1) per-core calculation ===
    uint32_t perCoreN;      ///< base elements per core = n / useCoreNum
    uint32_t remainder;     ///< first 'remainder' cores get +1 = n % useCoreNum

    // === SIMT path (incx!=1) ===
    uint32_t nthreads;      ///< SIMT threads per block (128-aligned, max 2048)
};

void snrm2_kernel_do(uint8_t* xGm, uint8_t* resultGm, uint8_t* workspace,
                      const Snrm2TilingData& tiling, uint32_t numBlocks, void* stream);
