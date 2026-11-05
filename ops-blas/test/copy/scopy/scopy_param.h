/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCOPY_PARAM_H
#define SCOPY_PARAM_H

#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct ScopyParam : public BlasTestParamBase {
    int n = 0;
    int incx = 1;
    int incy = 1;
    BlasFillMode x = BlasFillMode("INDEX");
    BlasFillMode y = BlasFillMode("VALUE_NORM_N999");
    int xAlignOffset = 0;
    int yAlignOffset = 0;

    ScopyParam(const csv_map& m) : BlasTestParamBase(m)
    {
        n = parseInt(ReadMap(m, "n", "0"));
        incx = parseInt(ReadMap(m, "incx", "1"));
        incy = parseInt(ReadMap(m, "incy", "1"));
        x = BlasFillMode(ReadMap(m, "x_fill", "INDEX"));
        y = BlasFillMode(ReadMap(m, "y_fill", "VALUE_NORM_N999"));
        xAlignOffset = parseInt(ReadMap(m, "x_align_offset", "0"));
        yAlignOffset = parseInt(ReadMap(m, "y_align_offset", "0"));
    }
};

#endif // SCOPY_PARAM_H
