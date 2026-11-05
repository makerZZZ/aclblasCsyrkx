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
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

enum class BlasDataFill { RANDOM, ZEROS, ONES };

inline BlasDataFill parseDataFill(const std::string& s)
{
    std::string upper = s;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (upper == "ZEROS")
        return BlasDataFill::ZEROS;
    if (upper == "ONES")
        return BlasDataFill::ONES;
    return BlasDataFill::RANDOM;
}

struct GelsBatchedParam : public BlasTestParamBase {
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int nrhs = 0;
    BlasDataFill aFill = BlasDataFill::RANDOM;
    int lda = 0;
    BlasDataFill cFill = BlasDataFill::RANDOM;
    int ldc = 0;
    int batchSize = 0;

    GelsBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        nrhs = parseInt(ReadMap(map, "nrhs", "0"));
        aFill = parseDataFill(ReadMap(map, "a_fill", "random"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, m))));
        cFill = parseDataFill(ReadMap(map, "c_fill", "random"));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max({1, m, n}))));
        batchSize = parseInt(ReadMap(map, "batch_size", "1"));
    }
};

