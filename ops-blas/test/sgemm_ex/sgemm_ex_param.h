/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGEMM_EX_PARAM_H
#define SGEMM_EX_PARAM_H

#include <string>
#include <algorithm>
#include <unordered_map>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

// ── aclblasGemmAlgo_t parser ──
inline aclblasGemmAlgo_t parseGemmAlgo(const std::string& s)
{
    static const std::unordered_map<std::string, aclblasGemmAlgo_t> t = {
        {"ACLBLAS_GEMM_DEFAULT", ACLBLAS_GEMM_DEFAULT},
        {"DEFAULT", ACLBLAS_GEMM_DEFAULT},
        {"ACLBLAS_GEMM_ALGO0", ACLBLAS_GEMM_ALGO0},
        {"ALGO0", ACLBLAS_GEMM_ALGO0},
        {"ACLBLAS_GEMM_ALGO1", ACLBLAS_GEMM_ALGO1},
        {"ALGO1", ACLBLAS_GEMM_ALGO1},
        {"ACLBLAS_GEMM_ALGO2", ACLBLAS_GEMM_ALGO2},
        {"ALGO2", ACLBLAS_GEMM_ALGO2},
        {"ACLBLAS_GEMM_ALGO3", ACLBLAS_GEMM_ALGO3},
        {"ALGO3", ACLBLAS_GEMM_ALGO3},
        {"ACLBLAS_GEMM_ALGO4", ACLBLAS_GEMM_ALGO4},
        {"ALGO4", ACLBLAS_GEMM_ALGO4},
        {"ACLBLAS_GEMM_ALGO5", ACLBLAS_GEMM_ALGO5},
        {"ALGO5", ACLBLAS_GEMM_ALGO5},
        {"ACLBLAS_GEMM_ALGO6", ACLBLAS_GEMM_ALGO6},
        {"ALGO6", ACLBLAS_GEMM_ALGO6},
        {"ACLBLAS_GEMM_ALGO7", ACLBLAS_GEMM_ALGO7},
        {"ALGO7", ACLBLAS_GEMM_ALGO7},
    };
    return parseEnum(s, t, ACLBLAS_GEMM_DEFAULT);
}

struct SgemmExParam : public BlasTestParamBase {
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int k = 0;
    float alpha = 1.0f;
    int lda = 0;
    int ldb = 0;
    float beta = 0.0f;
    int ldc = 0;
    aclblasGemmAlgo_t algo = ACLBLAS_GEMM_DEFAULT;
    BlasFillMode aFill = BlasFillMode("RANDOM_1");
    BlasFillMode bFill = BlasFillMode("RANDOM_1");
    BlasFillMode cFill = BlasFillMode("VALUE_NORM_0");
    bool alphaNull = false;
    bool betaNull = false;
    bool aNull = false;
    bool bNull = false;
    bool cNull = false;

    SgemmExParam(const csv_map& map) : BlasTestParamBase(map)
    {
        transA = parseOpTrans(ReadMap(map, "transA", "N"));
        transB = parseOpTrans(ReadMap(map, "transB", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));
        alpha = parseFloat(ReadMap(map, "alpha", "1.0"));
        beta = parseFloat(ReadMap(map, "beta", "0.0"));

        // Default lda: for column-major, ld >= physical rows of A
        int physRowsA = (transA == ACLBLAS_OP_N) ? m : k;
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, physRowsA))));
        int physRowsB = (transB == ACLBLAS_OP_N) ? k : n;
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(std::max(1, physRowsB))));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, m))));

        algo = parseGemmAlgo(ReadMap(map, "algo", "DEFAULT"));

        aFill = BlasFillMode(ReadMap(map, "a_fill", "RANDOM_1"));
        bFill = BlasFillMode(ReadMap(map, "b_fill", "RANDOM_1"));
        cFill = BlasFillMode(ReadMap(map, "c_fill", "VALUE_NORM_0"));

        alphaNull = (ReadMap(map, "alpha_null", "false") == "true");
        betaNull = (ReadMap(map, "beta_null", "false") == "true");
        aNull = (ReadMap(map, "a_null", "false") == "true");
        bNull = (ReadMap(map, "b_null", "false") == "true");
        cNull = (ReadMap(map, "c_null", "false") == "true");
    }
};

#endif // SGEMM_EX_PARAM_H
