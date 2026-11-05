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

struct SgerParam : public BlasTestParamBase {
    int m = 0;
    int n = 0;
    BlasFillMode alpha = parseFill("RANDOM");
    float alphaValue = 1.0f;
    BlasFillMode x = parseFill("INDEX");
    int incx = 1;
    BlasFillMode y = parseFill("INDEX");
    int incy = 1;
    BlasFillMode A = parseFill("INDEX");
    int lda = 0;

    SgerParam(const csv_map& map) : BlasTestParamBase(map)
    {
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        alpha = parseFill(ReadMap(map, "alpha", "RANDOM"));
        alphaValue = parseFloat(ReadMap(map, "alpha_value", "1.0"));
        x = parseFill(ReadMap(map, "x", "RANDOM"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        y = parseFill(ReadMap(map, "y", "RANDOM"));
        incy = parseInt(ReadMap(map, "incy", "1"));
        A = parseFill(ReadMap(map, "A", "RANDOM"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, m))));
    }
};

