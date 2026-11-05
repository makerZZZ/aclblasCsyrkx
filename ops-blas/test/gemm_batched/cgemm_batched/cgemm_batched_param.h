/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CGEMM_BATCHED_PARAM_H
#define CGEMM_BATCHED_PARAM_H

#include <string>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

inline int cgemmBatchedPhysRows(int logicalRows, int logicalCols, aclblasOperation_t trans)
{
    return (trans == ACLBLAS_OP_N) ? logicalRows : logicalCols;
}

inline int cgemmBatchedPhysCols(int logicalRows, int logicalCols, aclblasOperation_t trans)
{
    return (trans == ACLBLAS_OP_N) ? logicalCols : logicalRows;
}

#define CGEMM_BATCHED_SIGNATURE(name) \
    aclblasStatus_t name( \
        aclblasHandle_t handle, \
        aclblasOperation_t transA, aclblasOperation_t transB, \
        int m, int n, int k, \
        const aclblasComplex* alpha, \
        const aclblasComplex* const Aarray[], int lda, \
        const aclblasComplex* const Barray[], int ldb, \
        const aclblasComplex* beta, \
        aclblasComplex* const Carray[], int ldc, \
        int batchCount)

inline aclblasComplex parseComplex(const std::string& s)
{
    aclblasComplex v{1.0f, 0.0f};
    if (s.empty()) return v;
    size_t pos = s.find(';');
    if (pos == std::string::npos) {
        v.real = parseFloat(s, 1.0f);
        v.imag = 0.0f;
    } else {
        v.real = parseFloat(s.substr(0, pos), 1.0f);
        v.imag = parseFloat(s.substr(pos + 1), 0.0f);
    }
    return v;
}

struct CgemmBatchedParam : public BlasTestParamBase {
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int k = 0;
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    int lda = 0;
    int ldb = 0;
    int ldc = 0;
    int batchCount = 1;
    BlasFillMode aFill = parseFill("RANDOM_NORM_1");
    BlasFillMode bFill = parseFill("RANDOM_NORM_1");
    BlasFillMode cFill = parseFill("VALUE_NORM_0");
    bool alphaNull = false;
    bool betaNull = false;
    bool aarrayNull = false;
    bool barrayNull = false;
    bool carrayNull = false;

    CgemmBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        transA = parseOpTrans(ReadMap(map, "transA", "N"));
        transB = parseOpTrans(ReadMap(map, "transB", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));
        alpha = parseComplex(ReadMap(map, "alpha", "1.0;0.0"));
        beta  = parseComplex(ReadMap(map, "beta", "0.0;0.0"));

        int physRowsA = cgemmBatchedPhysRows(m, k, transA);
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, physRowsA))));
        int physRowsB = cgemmBatchedPhysRows(k, n, transB);
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(std::max(1, physRowsB))));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, m))));

        batchCount = parseInt(ReadMap(map, "batchCount", "1"));

        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM_NORM_1"));
        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM_NORM_1"));
        cFill = parseFill(ReadMap(map, "c_fill", "VALUE_NORM_0"));

        alphaNull = (ReadMap(map, "alpha_null", "false") == "true");
        betaNull  = (ReadMap(map, "beta_null", "false") == "true");
        aarrayNull = (ReadMap(map, "aarray_null", "false") == "true");
        barrayNull = (ReadMap(map, "barray_null", "false") == "true");
        carrayNull = (ReadMap(map, "carray_null", "false") == "true");
    }
};

#endif // CGEMM_BATCHED_PARAM_H
