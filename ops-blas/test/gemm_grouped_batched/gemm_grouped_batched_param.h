/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_GROUPED_BATCHED_PARAM_H
#define GEMM_GROUPED_BATCHED_PARAM_H

#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

// Helper: detect NULLPTR marker for array parameters
inline bool isNullptrMarker(const std::string& s)
{
    return s == "NULLPTR" || s.empty();
}

inline void parseOptionalIntArray(
    const std::string& value, bool& nullFlag, std::vector<int>& out)
{
    if (isNullptrMarker(value)) {
        nullFlag = true;
        return;
    }
    out = parseIntArray(value);
}

inline void parseOptionalTransArray(
    const std::string& value, bool& nullFlag, std::vector<aclblasOperation_t>& out)
{
    if (isNullptrMarker(value)) {
        nullFlag = true;
        return;
    }
    out = parseTransArray(value);
}

inline void markNullIfNeeded(const std::string& value, bool& nullFlag)
{
    if (isNullptrMarker(value)) {
        nullFlag = true;
    }
}

struct GemmGroupedBatchedParam : public BlasTestParamBase {
    // 0=S(FP32)
    int dtype = 0;
    int groupCount = 1;
    std::vector<aclblasOperation_t> transaArray;
    std::vector<aclblasOperation_t> transbArray;
    std::vector<int> mArray;
    std::vector<int> nArray;
    std::vector<int> kArray;
    // Alpha/beta stored as raw CSV strings; parsed in test.cpp based on dtype
    std::string alphaArrayStr;
    std::string betaArrayStr;
    std::vector<int> ldaArray;
    std::vector<int> ldbArray;
    std::vector<int> ldcArray;
    std::vector<int> groupSizeArray;
    BlasFillMode A_fill = parseFill("RANDOM");
    BlasFillMode B_fill = parseFill("RANDOM");
    BlasFillMode C_fill = parseFill("RANDOM");

    // Nullptr flags for host-side array parameters (error-path testing)
    bool transaArrayNull = false;
    bool transbArrayNull = false;
    bool mArrayNull = false;
    bool nArrayNull = false;
    bool kArrayNull = false;
    bool alphaArrayNull = false;
    bool ldaArrayNull = false;
    bool ldbArrayNull = false;
    bool betaArrayNull = false;
    bool ldcArrayNull = false;
    bool groupSizeArrayNull = false;

    GemmGroupedBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        dtype = parseInt(ReadMap(map, "dtype", "0"));
        groupCount = parseInt(ReadMap(map, "groupCount", "1"));

        parseOptionalTransArray(ReadMap(map, "transaArray", "ACLBLAS_OP_N"), transaArrayNull, transaArray);
        parseOptionalTransArray(ReadMap(map, "transbArray", "ACLBLAS_OP_N"), transbArrayNull, transbArray);
        parseOptionalIntArray(ReadMap(map, "mArray", "4"), mArrayNull, mArray);
        parseOptionalIntArray(ReadMap(map, "nArray", "4"), nArrayNull, nArray);
        parseOptionalIntArray(ReadMap(map, "kArray", "4"), kArrayNull, kArray);

        alphaArrayStr = ReadMap(map, "alphaArray", "1.0");
        markNullIfNeeded(alphaArrayStr, alphaArrayNull);

        parseOptionalIntArray(ReadMap(map, "ldaArray", "4"), ldaArrayNull, ldaArray);
        parseOptionalIntArray(ReadMap(map, "ldbArray", "4"), ldbArrayNull, ldbArray);

        betaArrayStr = ReadMap(map, "betaArray", "0.0");
        markNullIfNeeded(betaArrayStr, betaArrayNull);

        parseOptionalIntArray(ReadMap(map, "ldcArray", "4"), ldcArrayNull, ldcArray);
        parseOptionalIntArray(ReadMap(map, "groupSizeArray", "1"), groupSizeArrayNull, groupSizeArray);

        A_fill = parseFill(ReadMap(map, "A_fill", "RANDOM"));
        B_fill = parseFill(ReadMap(map, "B_fill", "RANDOM"));
        C_fill = parseFill(ReadMap(map, "C_fill", "RANDOM"));
    }
};

#endif // GEMM_GROUPED_BATCHED_PARAM_H