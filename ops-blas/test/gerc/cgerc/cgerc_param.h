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

struct CgercParam : public BlasTestParamBase {
    int m = 0;
    int n = 0;
    float alphaReal = 1.0F;
    float alphaImag = 0.0F;
    bool nullAlpha = false;
    BlasFillMode x = parseFill("RANDOM_NORM_5_5");
    int incx = 1;
    BlasFillMode y = parseFill("RANDOM_NORM_5_5");
    int incy = 1;
    BlasFillMode A = parseFill("RANDOM_NORM_5_5");
    int lda = 0;

    explicit CgercParam(const csv_map& map) : BlasTestParamBase(map)
    {
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        const std::string alphaRealText = ReadMap(map, "alpha_real", "1");
        const std::string alphaImagText = ReadMap(map, "alpha_imag", "0");
        nullAlpha = alphaRealText == "null" || alphaRealText == "nullptr";
        alphaReal = nullAlpha ? 0.0F : parseFloat(alphaRealText, 1.0F);
        alphaImag = nullAlpha ? 0.0F : parseFloat(alphaImagText, 0.0F);
        x = parseFill(ReadMap(map, "x", "RANDOM_NORM_5_5"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        y = parseFill(ReadMap(map, "y", "RANDOM_NORM_5_5"));
        incy = parseInt(ReadMap(map, "incy", "1"));
        A = parseFill(ReadMap(map, "A", "RANDOM_NORM_5_5"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, m))));
    }
};
