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

#include "csv_loader.h"

struct SrotmgParam : public BlasTestParamBase {
    float d1 = 1.0f;
    float d2 = 1.0f;
    float x1 = 1.0f;
    float y1 = 1.0f;

    SrotmgParam(const csv_map& map) : BlasTestParamBase(map)
    {
        d1 = parseFloat(ReadMap(map, "d1", "1.0"));
        d2 = parseFloat(ReadMap(map, "d2", "1.0"));
        x1 = parseFloat(ReadMap(map, "x1", "1.0"));
        y1 = parseFloat(ReadMap(map, "y1", "1.0"));
    }
};
