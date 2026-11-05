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
#include <string>
#include <unordered_map>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct ScalexParam : public BlasTestParamBase {
    int n = 0;
    float alphaVal = 2.5f;
    T_DtypeLocal alphaType = static_cast<int32_t>(ACL_FLOAT);
    BlasFillMode x = parseFill("RANDOM");
    T_DtypeLocal xType = static_cast<int32_t>(ACL_FLOAT);
    int incx = 1;
    T_DtypeLocal executionType = static_cast<int32_t>(ACL_FLOAT);
    bool alphaOnDevice = true;

    ScalexParam(const csv_map& m) : BlasTestParamBase(m)
    {
        n = parseInt(ReadMap(m, "n", "0"));
        alphaVal = parseFloat(ReadMap(m, "alpha", "2.5"));
        alphaType = parseDataType(ReadMap(m, "alpha_type", "ACL_FLOAT"));
        x = parseFill(ReadMap(m, "x", "RANDOM"));
        xType = parseDataType(ReadMap(m, "x_type", "ACL_FLOAT"));
        incx = parseInt(ReadMap(m, "incx", "1"));
        executionType = parseDataType(ReadMap(m, "execution_type", "ACL_FLOAT"));
        alphaOnDevice = (ReadMap(m, "alpha_on_device", "true") == "true");
    }
};
