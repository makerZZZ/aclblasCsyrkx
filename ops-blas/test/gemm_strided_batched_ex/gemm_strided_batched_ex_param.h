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

#include <algorithm>
#include <cstdint>
#include <string>

#include "gemm_batched_ex_param.h"

inline int64_t ParseStride(const std::string& value, int64_t defaultValue)
{
    if (value.empty()) {
        return defaultValue;
    }
    try {
        return std::stoll(value);
    } catch (...) {
        return defaultValue;
    }
}

struct GemmStridedBatchedExParam : public GemmBatchedExParam {
    int64_t strideA = 1;
    int64_t strideB = 1;
    int64_t strideC = 1;

    explicit GemmStridedBatchedExParam(const csv_map& map) : GemmBatchedExParam(map)
    {
        const int64_t aFootprint = static_cast<int64_t>(std::max(1, lda)) * std::max(1, physCols(m, k, transA));
        const int64_t bFootprint = static_cast<int64_t>(std::max(1, ldb)) * std::max(1, physCols(k, n, transB));
        const int64_t cFootprint = static_cast<int64_t>(std::max(1, ldc)) * std::max(1, n);
        strideA = ParseStride(ReadMap(map, "strideA", std::to_string(aFootprint)), aFootprint);
        strideB = ParseStride(ReadMap(map, "strideB", std::to_string(bFootprint)), bFootprint);
        strideC = ParseStride(ReadMap(map, "strideC", std::to_string(cFootprint)), cFootprint);
    }
};
