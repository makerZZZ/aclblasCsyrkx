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
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct SgeamParam : public BlasTestParamBase {
    aclblasOperation_t transa = ACLBLAS_OP_N;
    aclblasOperation_t transb = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    BlasFillMode alphaFill = BlasFillMode("VALUE_NORM_1");
    BlasFillMode betaFill = BlasFillMode("VALUE_NORM_1");
    BlasFillMode aFill = BlasFillMode("RANDOM_1_1");
    BlasFillMode bFill = BlasFillMode("RANDOM_1_1");
    int nullA = 0;
    int nullB = 0;
    int nullC = 0;
    int inplace = 0; // 0=out-of-place, 1=C==A, 2=C==B

    SgeamParam(const csv_map& csv) : BlasTestParamBase(csv)
    {
        transa = parseOpTrans(ReadMap(csv, "transa", "N"));
        transb = parseOpTrans(ReadMap(csv, "transb", "N"));
        m = parseInt(ReadMap(csv, "m", "0"));
        n = parseInt(ReadMap(csv, "n", "0"));
        lda = parseInt(ReadMap(csv, "lda", std::to_string((transa == ACLBLAS_OP_N) ? std::max(1, m) : std::max(1, n))));
        ldb = parseInt(ReadMap(csv, "ldb", std::to_string((transb == ACLBLAS_OP_N) ? std::max(1, m) : std::max(1, n))));
        ldc = parseInt(ReadMap(csv, "ldc", std::to_string(std::max(1, m))));
        alphaFill = BlasFillMode(ReadMap(csv, "alpha_fill", "VALUE_NORM_1"));
        betaFill = BlasFillMode(ReadMap(csv, "beta_fill", "VALUE_NORM_1"));
        aFill = BlasFillMode(ReadMap(csv, "a_fill", "RANDOM_1_1"));
        bFill = BlasFillMode(ReadMap(csv, "b_fill", "RANDOM_1_1"));
        nullA = parseInt(ReadMap(csv, "nullA", "0"));
        nullB = parseInt(ReadMap(csv, "nullB", "0"));
        nullC = parseInt(ReadMap(csv, "nullC", "0"));
        inplace = parseInt(ReadMap(csv, "inplace", "0"));
    }
};
