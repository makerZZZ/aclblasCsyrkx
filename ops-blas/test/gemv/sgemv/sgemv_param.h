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

#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct SgemvParam : public BlasTestParamBase {
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    float alpha = 1.0f;
    BlasFillMode alphaFill = parseFill("RANDOM");
    BlasFillMode a = parseFill("RANDOM");
    int lda = 0;
    BlasFillMode x = parseFill("RANDOM");
    int incx = 1;
    float beta = 0.0f;
    BlasFillMode betaFill = parseFill("RANDOM");
    BlasFillMode y = parseFill("RANDOM");
    int incy = 1;

    SgemvParam(const csv_map& map) : BlasTestParamBase(map)
    {
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        alpha = parseFloat(ReadMap(map, "alpha", "1.0"));
        alphaFill = parseFill(ReadMap(map, "alpha_fill", "RANDOM"));
        a = parseFill(ReadMap(map, "a", "RANDOM"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, m))));
        x = parseFill(ReadMap(map, "x", "RANDOM"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        beta = parseFloat(ReadMap(map, "beta", "0.0"));
        betaFill = parseFill(ReadMap(map, "beta_fill", "RANDOM"));
        y = parseFill(ReadMap(map, "y", "RANDOM"));
        incy = parseInt(ReadMap(map, "incy", "1"));
    }
};

