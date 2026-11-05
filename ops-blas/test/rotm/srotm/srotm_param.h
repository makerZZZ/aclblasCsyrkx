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

#include <array>
#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct SrotmParam : public BlasTestParamBase {
    int64_t n = 0;
    int64_t incx = 1;
    int64_t incy = 1;
    std::array<float, 5> sparam = {-1.0f, 1.0f, 0.0f, 0.0f, 1.0f};

    SrotmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        n = static_cast<int64_t>(parseInt(ReadMap(map, "n", "0")));
        incx = static_cast<int64_t>(parseInt(ReadMap(map, "incx", "1")));
        incy = static_cast<int64_t>(parseInt(ReadMap(map, "incy", "1")));
        sparam[0] = parseFloat(ReadMap(map, "sparam0", "-1.0"));
        sparam[1] = parseFloat(ReadMap(map, "sparam1", "1.0"));
        sparam[2] = parseFloat(ReadMap(map, "sparam2", "0.0"));
        sparam[3] = parseFloat(ReadMap(map, "sparam3", "0.0"));
        sparam[4] = parseFloat(ReadMap(map, "sparam4", "1.0"));
    }
};

