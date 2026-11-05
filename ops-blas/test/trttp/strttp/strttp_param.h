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

struct StrttpParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    int n = 0;
    BlasFillMode a = parseFill("INDEX");
    int lda = 0;
    BlasFillMode ap = parseFill("VALUE_NORM_0");

    StrttpParam(const csv_map& map) : BlasTestParamBase(map)
    {
        uplo = parseFillMode(ReadMap(map, "uplo", "LOWER"));
        n = parseInt(ReadMap(map, "n", "0"));
        a = parseFill(ReadMap(map, "a", "INDEX"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, n))));
        ap = parseFill(ReadMap(map, "ap", "VALUE_NORM_0"));
    }
};

