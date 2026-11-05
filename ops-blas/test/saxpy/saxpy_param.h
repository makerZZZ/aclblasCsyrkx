/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SAXPY_PARAM_H
#define SAXPY_PARAM_H

#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

struct SaxpyParam : public BlasTestParamBase {
    int n = 0;
    int incx = 1;
    int incy = 1;
    float alpha = 1.0f;
    BlasFillMode x = parseFill("RANDOM");
    BlasFillMode y = parseFill("RANDOM");

    SaxpyParam(const csv_map& m) : BlasTestParamBase(m)
    {
        n = parseInt(ReadMap(m, "n", "0"));
        incx = parseInt(ReadMap(m, "incx", "1"));
        incy = parseInt(ReadMap(m, "incy", "1"));
        alpha = parseFloat(ReadMap(m, "alpha", "1.0"));
        x = parseFill(ReadMap(m, "x", "RANDOM"));
        y = parseFill(ReadMap(m, "y", "RANDOM"));
    }
};

#endif
