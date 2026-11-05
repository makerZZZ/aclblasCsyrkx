/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSYRKX_PARAM_H
#define SSYRKX_PARAM_H

#include <string>
#include <algorithm>
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct SsyrkxParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int n = 0;
    int k = 0;
    float alpha = 1.0f;
    BlasFillMode a = BlasFillMode("RANDOM_NORM_P1_P1");
    int lda = 0;
    BlasFillMode b = BlasFillMode("RANDOM_NORM_P1_P1");
    int ldb = 0;
    float beta = 0.0f;
    BlasFillMode c = BlasFillMode("RANDOM_NORM_P1_P1");
    int ldc = 0;
    bool nullAlpha = false;
    bool nullA = false;
    bool nullB = false;
    bool nullBeta = false;
    bool nullC = false;

    explicit SsyrkxParam(const csv_map& m) : BlasTestParamBase(m)
    {
        uplo = parseFillMode(ReadMap(m, "uplo", "UPPER"));
        trans = parseOpTrans(ReadMap(m, "trans", "N"));
        n = parseInt(ReadMap(m, "n", "0"));
        k = parseInt(ReadMap(m, "k", "0"));

        std::string alphaStr = ReadMap(m, "alpha", "1.0");
        nullAlpha = (alphaStr == "null" || alphaStr == "nullptr") || (ReadMap(m, "nullAlpha", "0") == "1");
        alpha = (alphaStr == "null" || alphaStr == "nullptr") ? 0.0f : parseFloat(alphaStr, 1.0f);

        a = BlasFillMode(ReadMap(m, "a", "RANDOM_NORM_P1_P1"));

        lda = parseInt(ReadMap(m, "lda", "0"));

        b = BlasFillMode(ReadMap(m, "b", "RANDOM_NORM_P1_P1"));

        ldb = parseInt(ReadMap(m, "ldb", "0"));

        std::string betaStr = ReadMap(m, "beta", "0.0");
        nullBeta = (betaStr == "null" || betaStr == "nullptr") || (ReadMap(m, "nullBeta", "0") == "1");
        beta = (betaStr == "null" || betaStr == "nullptr") ? 0.0f : parseFloat(betaStr, 0.0f);

        c = BlasFillMode(ReadMap(m, "c", "RANDOM_NORM_P1_P1"));

        ldc = parseInt(ReadMap(m, "ldc", "0"));

        nullA = (ReadMap(m, "nullA", "0") == "1");
        nullB = (ReadMap(m, "nullB", "0") == "1");
        nullC = (ReadMap(m, "nullC", "0") == "1");

        // Auto-compute lda/ldb/ldc when 0
        if (lda == 0) {
            lda = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
        }
        if (ldb == 0) {
            ldb = (trans == ACLBLAS_OP_N) ? std::max(1, n) : std::max(1, k);
        }
        if (ldc == 0) {
            ldc = std::max(1, n);
        }
    }
};

#endif // SSYRKX_PARAM_H
