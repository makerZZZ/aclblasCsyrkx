/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AXPY_EX_PARAM_H
#define AXPY_EX_PARAM_H

#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

struct AxpyExParam : public BlasTestParamBase {
    int n = 0;
    std::string alphaStr; // raw alpha CSV string (supports "nan"/"inf")
    T_DtypeLocal alphaType = static_cast<int32_t>(ACL_FLOAT);
    bool alphaIsNull = false; // R01: when true, TEST_P sets alphaPtr=nullptr
    BlasFillMode x = parseFill("RANDOM_1_1");
    T_DtypeLocal xType = static_cast<int32_t>(ACL_FLOAT);
    int incx = 1;
    BlasFillMode y = parseFill("RANDOM_1_1");
    T_DtypeLocal yType = static_cast<int32_t>(ACL_FLOAT);
    int incy = 1;
    T_DtypeLocal executionType = static_cast<int32_t>(ACL_FLOAT);
    bool alphaOnDevice = true;

    AxpyExParam(const csv_map& m) : BlasTestParamBase(m)
    {
        n = parseInt(ReadMap(m, "n", "0"));
        alphaStr = ReadMap(m, "alpha", "2.5");
        alphaType = parseDataType(ReadMap(m, "alpha_type", "ACL_FLOAT"));
        alphaIsNull = (ReadMap(m, "alpha_null", "false") == "true");
        x = parseFill(ReadMap(m, "x", "RANDOM_1_1"));
        xType = parseDataType(ReadMap(m, "x_type", "ACL_FLOAT"));
        incx = parseInt(ReadMap(m, "incx", "1"));
        y = parseFill(ReadMap(m, "y", "RANDOM_1_1"));
        yType = parseDataType(ReadMap(m, "y_type", "ACL_FLOAT"));
        incy = parseInt(ReadMap(m, "incy", "1"));
        executionType = parseDataType(ReadMap(m, "execution_type", "ACL_FLOAT"));
        alphaOnDevice = (ReadMap(m, "alpha_on_device", "true") == "true");
    }

    // Lazy parse of alpha string → float. std::stof handles "nan"/"inf".
    float alphaVal() const { return parseFloat(alphaStr, 0.0f); }
};

#endif
