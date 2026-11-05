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
#include "cgemv_fill.h"

// CSV columns:
//   case_name,description,trans,m,n,alpha_real,alpha_imag,a,lda,x,incx,
//   beta_real,beta_imag,y,incy,expect_result,mere_threshold,mare_multiplier,random_seed
// alpha_real/beta_real may hold the literal "null" to request a nullptr scalar (negative cases).
struct CgemvParam : public BlasTestParamBase {
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    float alphaRe = 1.0f;
    float alphaIm = 0.0f;
    bool alphaNull = false;
    cgemv_test::FillMode a = cgemv_test::parseFill("RANDOM_NORM_5_5");
    int lda = 0;
    cgemv_test::FillMode x = cgemv_test::parseFill("RANDOM_NORM_5_5");
    int incx = 1;
    float betaRe = 0.0f;
    float betaIm = 0.0f;
    bool betaNull = false;
    cgemv_test::FillMode y = cgemv_test::parseFill("RANDOM_NORM_5_5");
    int incy = 1;

    CgemvParam(const csv_map& map) : BlasTestParamBase(map)
    {
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));

        std::string alphaReStr = ReadMap(map, "alpha_real", "1.0");
        alphaNull = (alphaReStr == "null");
        alphaRe = alphaNull ? 0.0f : parseFloat(alphaReStr);
        alphaIm = parseFloat(ReadMap(map, "alpha_imag", "0.0"));

        a = cgemv_test::parseFill(ReadMap(map, "a", "RANDOM_NORM_5_5"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, m))));
        x = cgemv_test::parseFill(ReadMap(map, "x", "RANDOM_NORM_5_5"));
        incx = parseInt(ReadMap(map, "incx", "1"));

        std::string betaReStr = ReadMap(map, "beta_real", "0.0");
        betaNull = (betaReStr == "null");
        betaRe = betaNull ? 0.0f : parseFloat(betaReStr);
        betaIm = parseFloat(ReadMap(map, "beta_imag", "0.0"));

        y = cgemv_test::parseFill(ReadMap(map, "y", "RANDOM_NORM_5_5"));
        incy = parseInt(ReadMap(map, "incy", "1"));
    }
};
