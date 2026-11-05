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

#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

// Pivot mode: whether PivotArray is allocated or nullptr
enum class GetriPivotMode {
    PIVOT,   // PivotArray is allocated (use partial pivoting)
    NO_PIVOT // PivotArray is nullptr (no pivoting)
};

inline GetriPivotMode parseGetriPivotMode(const std::string& s)
{
    if (s == "nullptr" || s == "NULLPTR" || s == "NO_PIVOT" || s == "null")
        return GetriPivotMode::NO_PIVOT;
    return GetriPivotMode::PIVOT;
}

// Matrix type for input data generation
enum class GetriMatrixType {
    RANDOM_NONSINGULAR,     // Random [0,1] + n on diagonal
    SINGULAR_ZERO_COL,      // First column all zeros
    SINGULAR_DEPENDENT_ROW, // One row = 2x another row
    IDENTITY,               // Identity matrix
    DIAGONALLY_DOMINANT,    // Diagonal = n, off-diagonal = random [0,1]
    ILL_CONDITIONED,        // Hilbert matrix
    MIXED,                  // Mix of nonsingular and singular across batches
    DIAGONAL,               // Diagonal matrix, diagonal elements in [0.1, 10]
    UPPER_TRIANGULAR,       // Upper triangular, diagonal in [0.5, 2]
    LOWER_TRIANGULAR,       // Lower triangular, diagonal in [0.5, 2]
    NULLPTR_AARRAY,         // Error test: pass nullptr as Aarray
    NULLPTR_CARRAY,         // Error test: pass nullptr as Carray
    NULLPTR_INFOARRAY,      // Error test: pass nullptr as infoArray
    NULLPTR_HANDLE          // Error test: pass nullptr as handle
};

inline GetriMatrixType parseGetriMatrixType(const std::string& s)
{
    if (s == "SINGULAR_ZERO_COL")
        return GetriMatrixType::SINGULAR_ZERO_COL;
    if (s == "SINGULAR_DEPENDENT_ROW")
        return GetriMatrixType::SINGULAR_DEPENDENT_ROW;
    if (s == "IDENTITY")
        return GetriMatrixType::IDENTITY;
    if (s == "DIAGONALLY_DOMINANT")
        return GetriMatrixType::DIAGONALLY_DOMINANT;
    if (s == "ILL_CONDITIONED")
        return GetriMatrixType::ILL_CONDITIONED;
    if (s == "MIXED")
        return GetriMatrixType::MIXED;
    if (s == "DIAGONAL")
        return GetriMatrixType::DIAGONAL;
    if (s == "UPPER_TRIANGULAR")
        return GetriMatrixType::UPPER_TRIANGULAR;
    if (s == "LOWER_TRIANGULAR")
        return GetriMatrixType::LOWER_TRIANGULAR;
    if (s == "NULLPTR_AARRAY" || s == "nullptr_Aarray")
        return GetriMatrixType::NULLPTR_AARRAY;
    if (s == "NULLPTR_CARRAY" || s == "nullptr_Carray")
        return GetriMatrixType::NULLPTR_CARRAY;
    if (s == "NULLPTR_INFOARRAY" || s == "nullptr_infoarray")
        return GetriMatrixType::NULLPTR_INFOARRAY;
    if (s == "NULLPTR_HANDLE" || s == "nullptr_handle")
        return GetriMatrixType::NULLPTR_HANDLE;
    return GetriMatrixType::RANDOM_NONSINGULAR;
}

inline BlasLapackMatrixType toBlasLapackMatrixType(GetriMatrixType t)
{
    switch (t) {
        case GetriMatrixType::IDENTITY: return BlasLapackMatrixType::IDENTITY;
        case GetriMatrixType::DIAGONALLY_DOMINANT: return BlasLapackMatrixType::DIAGONALLY_DOMINANT;
        case GetriMatrixType::ILL_CONDITIONED: return BlasLapackMatrixType::ILL_CONDITIONED;
        case GetriMatrixType::DIAGONAL: return BlasLapackMatrixType::DIAGONAL;
        case GetriMatrixType::UPPER_TRIANGULAR: return BlasLapackMatrixType::UPPER_TRIANGULAR;
        case GetriMatrixType::LOWER_TRIANGULAR: return BlasLapackMatrixType::LOWER_TRIANGULAR;
        case GetriMatrixType::SINGULAR_ZERO_COL: return BlasLapackMatrixType::SINGULAR_ZERO_COL;
        case GetriMatrixType::SINGULAR_DEPENDENT_ROW: return BlasLapackMatrixType::SINGULAR_DEPENDENT_ROW;
        case GetriMatrixType::MIXED: return BlasLapackMatrixType::MIXED;
        default: return BlasLapackMatrixType::RANDOM_NONSINGULAR;
    }
}

struct SgetriBatchedParam : public BlasTestParamBase {
    int n = 0;
    int lda = 0;
    int ldc = 0;
    int batchSize = 0;
    GetriPivotMode pivotMode = GetriPivotMode::PIVOT;
    GetriMatrixType matrixType = GetriMatrixType::RANDOM_NONSINGULAR;

    SgetriBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        n = parseInt(ReadMap(map, "n", "0"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, n))));
        ldc = parseInt(ReadMap(map, "ldc", std::to_string(std::max(1, n))));
        batchSize = parseInt(ReadMap(map, "batch_size", "1"));
        pivotMode = parseGetriPivotMode(ReadMap(map, "pivot_mode", "PIVOT"));
        matrixType = parseGetriMatrixType(ReadMap(map, "matrix_type", "RANDOM_NONSINGULAR"));
    }
};

