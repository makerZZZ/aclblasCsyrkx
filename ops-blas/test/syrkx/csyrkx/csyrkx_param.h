/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSYRKX_PARAM_H
#define CSYRKX_PARAM_H

#include <algorithm>
#include <string>

#include "cann_ops_blas.h"
#include "csv_loader.h"

// CSV schema (task-provided csyrkx_test.csv):
//   case_name,description,uplo,trans,n,k,alpha_real,alpha_imag,
//   a_fill,lda,b_fill,ldb,beta_real,beta_imag,c_fill,ldc,
//   expect_result,mere_threshold,mare_multiplier,random_seed
//
// `null` in alpha_real / beta_real encodes a nullptr scalar pointer; the
// tensor fill token NULLPTR encodes a nullptr matrix pointer. The fill tokens
// (RANDOM_NORM_5_5 / VALUE_NORM_0 / RANDOM_ALTER / RANDOM_EXTREME /
// VALUE_NORM_INF / VALUE_NORM_NAN) are parsed by the shared BlasFillMode.
struct CsyrkxParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int n = 0;
    int k = 0;
    float alphaReal = 1.0f;
    float alphaImag = 0.0f;
    BlasFillMode aFill = BlasFillMode("RANDOM_NORM_5_5");
    int lda = 0;
    BlasFillMode bFill = BlasFillMode("RANDOM_NORM_5_5");
    int ldb = 0;
    float betaReal = 0.0f;
    float betaImag = 0.0f;
    BlasFillMode cFill = BlasFillMode("RANDOM_NORM_5_5");
    int ldc = 0;
    bool nullAlpha = false;
    bool nullBeta = false;
    bool nullA = false;
    bool nullB = false;
    bool nullC = false;

    explicit CsyrkxParam(const csv_map& m) : BlasTestParamBase(m)
    {
        uplo = parseFillMode(ReadMap(m, "uplo", "UPPER"));
        trans = parseOpTrans(ReadMap(m, "trans", "N"));
        n = parseInt(ReadMap(m, "n", "0"));
        k = parseInt(ReadMap(m, "k", "0"));

        const std::string alphaRealStr = ReadMap(m, "alpha_real", "1.0");
        nullAlpha = (alphaRealStr == "null" || alphaRealStr == "nullptr");
        alphaReal = nullAlpha ? 0.0f : parseFloat(alphaRealStr, 1.0f);
        alphaImag = nullAlpha ? 0.0f : parseFloat(ReadMap(m, "alpha_imag", "0.0"), 0.0f);

        const int defaultLdab = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
        aFill = BlasFillMode(ReadMap(m, "a_fill", "RANDOM_NORM_5_5"));
        lda = parseInt(ReadMap(m, "lda", std::to_string(defaultLdab)));
        bFill = BlasFillMode(ReadMap(m, "b_fill", "RANDOM_NORM_5_5"));
        ldb = parseInt(ReadMap(m, "ldb", std::to_string(defaultLdab)));

        const std::string betaRealStr = ReadMap(m, "beta_real", "0.0");
        nullBeta = (betaRealStr == "null" || betaRealStr == "nullptr");
        betaReal = nullBeta ? 0.0f : parseFloat(betaRealStr, 0.0f);
        betaImag = nullBeta ? 0.0f : parseFloat(ReadMap(m, "beta_imag", "0.0"), 0.0f);

        cFill = BlasFillMode(ReadMap(m, "c_fill", "RANDOM_NORM_5_5"));
        ldc = parseInt(ReadMap(m, "ldc", std::to_string(std::max(1, n))));

        nullA = (aFill.method == BlasFillMode::M_NULLPTR);
        nullB = (bFill.method == BlasFillMode::M_NULLPTR);
        nullC = (cFill.method == BlasFillMode::M_NULLPTR);
    }
};

#endif // CSYRKX_PARAM_H
