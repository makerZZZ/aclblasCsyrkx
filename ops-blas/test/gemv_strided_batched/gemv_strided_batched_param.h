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

struct GemvStridedBatchedParam : public BlasTestParamBase {
    // 0=HSH(F16→F16) 1=S(F32) 2=HSS(F16→F32) 3=TST(BF16→BF16) 4=TSS(BF16→F32)
    int dtype = 1;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    float alpha = 1.0f;
    BlasFillMode a = parseFill("RANDOM");
    int lda = 0;
    BlasFillMode x = parseFill("RANDOM");
    int incx = 1;
    int64_t stridex = 0;
    float beta = 0.0f;
    BlasFillMode y = parseFill("RANDOM");
    int incy = 1;
    int64_t stridey = 0;
    int batchCount = 0;
    int64_t strideA = 0;

    GemvStridedBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        dtype = parseInt(ReadMap(map, "dtype", "1"));
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        alpha = parseFloat(ReadMap(map, "alpha", "1.0"));
        a = parseFill(ReadMap(map, "A", "RANDOM"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, m))));
        x = parseFill(ReadMap(map, "x", "RANDOM"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        stridex = parseInt64(ReadMap(map, "stridex", "0"));
        beta = parseFloat(ReadMap(map, "beta", "0.0"));
        y = parseFill(ReadMap(map, "y", "RANDOM"));
        incy = parseInt(ReadMap(map, "incy", "1"));
        stridey = parseInt64(ReadMap(map, "stridey", "0"));
        batchCount = parseInt(ReadMap(map, "batchCount", "0"));
        strideA = parseInt64(ReadMap(map, "strideA", "0"));
    }
};


