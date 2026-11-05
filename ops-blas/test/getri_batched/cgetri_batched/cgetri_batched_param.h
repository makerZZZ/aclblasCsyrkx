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
#include <utility>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

enum class CgetriPivotMode {
    PIVOT,
    NO_PIVOT,
};

inline CgetriPivotMode ParseCgetriPivotMode(const std::string& value)
{
    if (value == "nullptr" || value == "NULLPTR" || value == "NO_PIVOT" || value == "null") {
        return CgetriPivotMode::NO_PIVOT;
    }
    return CgetriPivotMode::PIVOT;
}

enum class CgetriMatrixType {
    RANDOM_NONSINGULAR,
    SINGULAR_ZERO_COL,
    SINGULAR_DEPENDENT_ROW,
    IDENTITY,
    DIAGONALLY_DOMINANT,
    ILL_CONDITIONED,
    MIXED,
    DIAGONAL,
    UPPER_TRIANGULAR,
    LOWER_TRIANGULAR,
    LOWER_BIDIAGONAL,
    PERMUTED_DENSE,
    MIXED_DIAGONAL,
    INF_INPUT,
    NAN_INPUT,
    NULLPTR_AARRAY,
    NULLPTR_CARRAY,
    NULLPTR_INFOARRAY,
    NULLPTR_HANDLE,
};

inline CgetriMatrixType ParseCgetriMatrixType(const std::string& value)
{
    static const std::pair<const char*, CgetriMatrixType> types[] = {
        {"SINGULAR_ZERO_COL", CgetriMatrixType::SINGULAR_ZERO_COL},
        {"SINGULAR_DEPENDENT_ROW", CgetriMatrixType::SINGULAR_DEPENDENT_ROW},
        {"IDENTITY", CgetriMatrixType::IDENTITY},
        {"DIAGONALLY_DOMINANT", CgetriMatrixType::DIAGONALLY_DOMINANT},
        {"ILL_CONDITIONED", CgetriMatrixType::ILL_CONDITIONED},
        {"MIXED", CgetriMatrixType::MIXED},
        {"DIAGONAL", CgetriMatrixType::DIAGONAL},
        {"UPPER_TRIANGULAR", CgetriMatrixType::UPPER_TRIANGULAR},
        {"LOWER_TRIANGULAR", CgetriMatrixType::LOWER_TRIANGULAR},
        {"LOWER_BIDIAGONAL", CgetriMatrixType::LOWER_BIDIAGONAL},
        {"PERMUTED_DENSE", CgetriMatrixType::PERMUTED_DENSE},
        {"MIXED_DIAGONAL", CgetriMatrixType::MIXED_DIAGONAL},
        {"INF_INPUT", CgetriMatrixType::INF_INPUT},
        {"NAN_INPUT", CgetriMatrixType::NAN_INPUT},
        {"NULLPTR_AARRAY", CgetriMatrixType::NULLPTR_AARRAY},
        {"nullptr_Aarray", CgetriMatrixType::NULLPTR_AARRAY},
        {"NULLPTR_CARRAY", CgetriMatrixType::NULLPTR_CARRAY},
        {"nullptr_Carray", CgetriMatrixType::NULLPTR_CARRAY},
        {"NULLPTR_INFOARRAY", CgetriMatrixType::NULLPTR_INFOARRAY},
        {"nullptr_infoarray", CgetriMatrixType::NULLPTR_INFOARRAY},
        {"NULLPTR_HANDLE", CgetriMatrixType::NULLPTR_HANDLE},
        {"nullptr_handle", CgetriMatrixType::NULLPTR_HANDLE},
    };
    for (const auto& entry : types) {
        if (value == entry.first) {
            return entry.second;
        }
    }
    return CgetriMatrixType::RANDOM_NONSINGULAR;
}

struct CgetriBatchedParam : public BlasTestParamBase {
    int n = 0;
    int lda = 0;
    int ldc = 0;
    int batchSize = 0;
    CgetriPivotMode pivotMode = CgetriPivotMode::PIVOT;
    CgetriMatrixType matrixType = CgetriMatrixType::RANDOM_NONSINGULAR;

    explicit CgetriBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        n = parseInt(ReadMap(map, "n", "0"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, n))));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, n))));
        batchSize = parseInt(ReadMap(map, "batch_size", "1"));
        pivotMode = ParseCgetriPivotMode(ReadMap(map, "pivot_mode", "PIVOT"));
        matrixType = ParseCgetriMatrixType(ReadMap(map, "matrix_type", "RANDOM_NONSINGULAR"));
    }

    bool IsPerformanceCase() const { return caseName.rfind("TC_PF_", 0) == 0; }
};
