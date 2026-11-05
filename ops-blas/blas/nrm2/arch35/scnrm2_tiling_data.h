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

constexpr uint32_t SCNRM2_MAX_CORE_NUM = 64;
// Ascend 950 Vector UB budget: input tile + abs scratch + ReduceSum temporary
// must fit together with TPipe metadata.  27392 FP32 values is the largest
// 256B-repeat-aligned tile used by the tuned 950 nrm2 implementation.
constexpr uint32_t SCNRM2_MAX_DATA_COUNT = 27392;

struct Scnrm2TilingData {
    int32_t n;
    int32_t incx;
    uint32_t useCoreNum;
    uint32_t maxDataCount;
    uint32_t batchPerCore;
    uint32_t remain;
    uint32_t nthreads;
    uint32_t checkNormResult;
};

void scnrm2_kernel_do(uint8_t* x, uint8_t* result, uint8_t* workspace,
    const Scnrm2TilingData& tiling, uint32_t numBlocks, void* stream);
