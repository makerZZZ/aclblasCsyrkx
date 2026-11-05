/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_STRIDED_BATCHED_PARAM_H
#define GEMM_STRIDED_BATCHED_PARAM_H

#include <algorithm>
#include <cstdint>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

// ── Strided-batched matrix buffer (column-major) ──
//   在同一块连续缓冲中放置 batchCount 个 physRows×physCols 列主序子矩阵，
//   相邻 batch 起始元素偏移为 stride（以元素个数计）。
//   - stride == 0     : 广播语义，全批复用同一子矩阵，缓冲仅含单个矩阵；
//   - stride  > 0     : 缓冲大小 = stride*(batchCount-1) + ld*physCols；
//   每个 batch 使用 seed+batchIdx 生成，保证批间数据不同以暴露 stride 偏移错误。
inline std::vector<float> gsbMakeStridedBatched(
    int physRows, int physCols, int ld, int64_t stride, int batchCount, const BlasFillMode& fill, uint32_t seed = 0)
{
    if (fill.method == BlasFillMode::M_NULLPTR)
        return {};
    if (physRows <= 0 || physCols <= 0 || ld <= 0 || batchCount <= 0)
        return {};

    const int64_t matSize = static_cast<int64_t>(ld) * physCols;
    const int64_t total = (stride == 0) ? matSize : (stride * (batchCount - 1) + matSize);
    std::vector<float> data(static_cast<size_t>(total), 0.0f);

    const int effectiveBatches = (stride == 0) ? 1 : batchCount;
    for (int b = 0; b < effectiveBatches; b++) {
        std::vector<float> mat = makeBlasMatrix(physRows, physCols, ld, fill, seed + static_cast<uint32_t>(b));
        const int64_t off = static_cast<int64_t>(b) * stride;
        const size_t copyCount = std::min(mat.size(), static_cast<size_t>(matSize));
        std::copy(mat.begin(), mat.begin() + copyCount, data.begin() + off);
    }
    return data;
}

// ── Physical (stored) dimensions for a column-major matrix under trans ──
//   op(X) shape is logicalRows×logicalCols; physically stored matrix rows/cols
//   depend on trans (N: rows=logicalRows; T/C: rows=logicalCols).
inline int gsbPhysRows(int logicalRows, int logicalCols, aclblasOperation_t trans)
{
    return (trans == ACLBLAS_OP_N) ? logicalRows : logicalCols;
}
inline int gsbPhysCols(int logicalRows, int logicalCols, aclblasOperation_t trans)
{
    return (trans == ACLBLAS_OP_N) ? logicalCols : logicalRows;
}

// ── Parameter struct for aclblasSgemmStridedBatched, fields in API order ──
struct GemmStridedBatchedParam : public BlasTestParamBase {
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    int m = 0;
    int n = 0;
    int k = 0;
    float alpha = 1.0f;
    int lda = 0;
    int64_t strideA = 0;
    int ldb = 0;
    int64_t strideB = 0;
    float beta = 0.0f;
    int ldc = 0;
    int64_t strideC = 0;
    int batchCount = 0;

    BlasFillMode aFill = parseFill("RANDOM_1_1");
    BlasFillMode bFill = parseFill("RANDOM_1_1");
    BlasFillMode cFill = parseFill("RANDOM_1_1");

    // Nullptr flags for error-path testing
    bool alphaNull = false;
    bool betaNull = false;
    bool aNull = false;
    bool bNull = false;
    bool cNull = false;

    // Whether strideA/B/C were explicitly given in CSV (empty → derive compact layout)
    bool strideAGiven = false;
    bool strideBGiven = false;
    bool strideCGiven = false;

    GemmStridedBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        transA = parseOpTrans(ReadMap(map, "transA", "N"));
        transB = parseOpTrans(ReadMap(map, "transB", "N"));
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));
        alpha = parseFloat(ReadMap(map, "alpha", "1.0"));
        beta = parseFloat(ReadMap(map, "beta", "0.0"));
        batchCount = parseInt(ReadMap(map, "batchCount", "1"));

        // Leading dimensions default to minimal legal value (column-major physical rows)
        int physRowsA = gsbPhysRows(m, k, transA);
        int physRowsB = gsbPhysRows(k, n, transB);
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, physRowsA))));
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(std::max(1, physRowsB))));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, m))));

        // Strides: if empty in CSV → derive compact packing later; if "0" → broadcast
        std::string sA = ReadMap(map, "strideA", "");
        std::string sB = ReadMap(map, "strideB", "");
        std::string sC = ReadMap(map, "strideC", "");
        strideAGiven = !sA.empty();
        strideBGiven = !sB.empty();
        strideCGiven = !sC.empty();
        strideA = strideAGiven ? parseInt64(sA) : 0;
        strideB = strideBGiven ? parseInt64(sB) : 0;
        strideC = strideCGiven ? parseInt64(sC) : 0;

        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM_1_1"));
        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM_1_1"));
        cFill = parseFill(ReadMap(map, "c_fill", "RANDOM_1_1"));

        alphaNull = (ReadMap(map, "alpha_null", "") == "NULLPTR");
        betaNull = (ReadMap(map, "beta_null", "") == "NULLPTR");
        aNull = (ReadMap(map, "a_null", "") == "NULLPTR");
        bNull = (ReadMap(map, "b_null", "") == "NULLPTR");
        cNull = (ReadMap(map, "c_null", "") == "NULLPTR");
    }

    // Compact (packed) strides in element units: A_i / B_i / C_i back-to-back.
    int64_t compactStrideA() const { return static_cast<int64_t>(lda) * gsbPhysCols(m, k, transA); }
    int64_t compactStrideB() const { return static_cast<int64_t>(ldb) * gsbPhysCols(k, n, transB); }
    int64_t compactStrideC() const { return static_cast<int64_t>(ldc) * n; }

    // Effective stride used for data layout / golden / npu (compact when not given).
    int64_t effStrideA() const { return strideAGiven ? strideA : compactStrideA(); }
    int64_t effStrideB() const { return strideBGiven ? strideB : compactStrideB(); }
    int64_t effStrideC() const { return strideCGiven ? strideC : compactStrideC(); }
};

#endif // GEMM_STRIDED_BATCHED_PARAM_H
