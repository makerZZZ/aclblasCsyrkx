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

#include "csv_loader.h"

struct SdotParam : public BlasTestParamBase {
    int n = 0;
    BlasFillMode x = parseFill("RANDOM");
    int incx = 1;
    BlasFillMode y = parseFill("RANDOM");
    int incy = 1;

    SdotParam(const csv_map& map) : BlasTestParamBase(map)
    {
        n = parseInt(ReadMap(map, "n", "0"));
        x = parseFill(ReadMap(map, "x", "RANDOM"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        y = parseFill(ReadMap(map, "y", "RANDOM"));
        incy = parseInt(ReadMap(map, "incy", "1"));
    }
};

