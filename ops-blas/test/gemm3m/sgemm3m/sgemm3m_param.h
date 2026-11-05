/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGEMM3M_PARAM_H
#define SGEMM3M_PARAM_H

#include <string>
#include <algorithm>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

// ── Physical rows/cols of merged A/B based on transposition (column-major) ──
//   A contains A1/A2/A3 merged along K dimension:
//   transA=N: A is M×(3K) (physical rows = M)
//   transA=T/C: A is (3K)×M (physical rows = 3K)
inline int gemm3mPhysRowsA(int m, int k, aclblasOperation_t transA)
{
    return (transA == ACLBLAS_OP_N) ? m : 3 * k;
}
inline int gemm3mPhysColsA(int m, int k, aclblasOperation_t transA)
{
    return (transA == ACLBLAS_OP_N) ? 3 * k : m;
}
//   B contains B1/B2/B3 merged along K dimension:
//   transB=N: B is (3K)×N (physical rows = 3K)
//   transB=T/C: B is N×(3K) (physical rows = N)
inline int gemm3mPhysRowsB(int k, int n, aclblasOperation_t transB)
{
    return (transB == ACLBLAS_OP_N) ? 3 * k : n;
}
inline int gemm3mPhysColsB(int k, int n, aclblasOperation_t transB)
{
    return (transB == ACLBLAS_OP_N) ? n : 3 * k;
}

struct Gemm3mParam : public BlasTestParamBase {
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int k = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    BlasFillMode aFill = parseFill("RANDOM_1");
    BlasFillMode bFill = parseFill("RANDOM_1");
    BlasFillMode cFill = parseFill("VALUE_NORM_0");
    bool alphaNull = false;
    bool betaNull = false;
    bool aNull = false;
    bool bNull = false;
    bool cNull = false;

    Gemm3mParam(const csv_map& map) : BlasTestParamBase(map)
    {
        transA = parseOpTrans(ReadMap(map, "transA", "N"));
        transB = parseOpTrans(ReadMap(map, "transB", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));
        alpha = parseFloat(ReadMap(map, "alpha", "1.0"));
        beta = parseFloat(ReadMap(map, "beta", "0.0"));

        // Default leading dimensions: column-major, ld >= physical rows
        int physRowsA = gemm3mPhysRowsA(std::max(0, m), std::max(0, k), transA);
        int physRowsB = gemm3mPhysRowsB(std::max(0, k), std::max(0, n), transB);
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, physRowsA))));
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(std::max(1, physRowsB))));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, std::max(0, m)))));

        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM_1"));
        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM_1"));
        cFill = parseFill(ReadMap(map, "c_fill", "VALUE_NORM_0"));

        alphaNull = (ReadMap(map, "alpha_null", "false") == "true");
        betaNull = (ReadMap(map, "beta_null", "false") == "true");
        aNull = (ReadMap(map, "a_null", "false") == "true");
        bNull = (ReadMap(map, "b_null", "false") == "true");
        cNull = (ReadMap(map, "c_null", "false") == "true");
    }
};

#endif // SGEMM3M_PARAM_H
