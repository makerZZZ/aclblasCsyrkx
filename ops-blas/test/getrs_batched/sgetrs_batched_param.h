/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGETRS_BATCHED_PARAM_H
#define SGETRS_BATCHED_PARAM_H

#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

// Pivot mode: whether devIpiv is allocated or nullptr
enum class GetrsPivotMode {
    PIVOT,   // devIpiv is allocated (use partial pivoting from getrf)
    NO_PIVOT // devIpiv is nullptr (no pivoting, P = I)
};

inline GetrsPivotMode parseGetrsPivotMode(const std::string& s)
{
    if (s == "nullptr" || s == "NULLPTR" || s == "NO_PIVOT" || s == "null")
        return GetrsPivotMode::NO_PIVOT;
    return GetrsPivotMode::PIVOT;
}

// Matrix type for input data generation
enum class GetrsMatrixType {
    RANDOM_NONSINGULAR,     // Random [0,1] + n on diagonal (guaranteed nonsingular)
    IDENTITY,               // Identity matrix
    DIAGONALLY_DOMINANT,    // Diagonal = n, off-diagonal = random [0,1]
    ILL_CONDITIONED,        // Hilbert matrix
    DIAGONAL,               // Diagonal matrix, diagonal elements in [0.1, 10]
    UPPER_TRIANGULAR,       // Upper triangular, diagonal in [0.5, 2]
    LOWER_TRIANGULAR,       // Lower triangular, diagonal in [0.5, 2]
    SINGULAR_ZERO_COL,      // First column all zeros
    SINGULAR_DEPENDENT_ROW, // One row = 2x another row
    MIXED,                  // Mix of nonsingular and singular across batches
    NULLPTR_AARRAY,         // Error test: pass nullptr as Aarray
    NULLPTR_BARRAY,         // Error test: pass nullptr as Barray
    NULLPTR_HANDLE          // Error test: pass nullptr as handle
};

inline GetrsMatrixType parseGetrsMatrixType(const std::string& s)
{
    if (s == "IDENTITY")
        return GetrsMatrixType::IDENTITY;
    if (s == "DIAGONALLY_DOMINANT")
        return GetrsMatrixType::DIAGONALLY_DOMINANT;
    if (s == "ILL_CONDITIONED")
        return GetrsMatrixType::ILL_CONDITIONED;
    if (s == "DIAGONAL")
        return GetrsMatrixType::DIAGONAL;
    if (s == "UPPER_TRIANGULAR")
        return GetrsMatrixType::UPPER_TRIANGULAR;
    if (s == "LOWER_TRIANGULAR")
        return GetrsMatrixType::LOWER_TRIANGULAR;
    if (s == "SINGULAR_ZERO_COL")
        return GetrsMatrixType::SINGULAR_ZERO_COL;
    if (s == "SINGULAR_DEPENDENT_ROW")
        return GetrsMatrixType::SINGULAR_DEPENDENT_ROW;
    if (s == "MIXED")
        return GetrsMatrixType::MIXED;
    if (s == "NULLPTR_AARRAY" || s == "nullptr_Aarray")
        return GetrsMatrixType::NULLPTR_AARRAY;
    if (s == "NULLPTR_BARRAY" || s == "nullptr_Barray")
        return GetrsMatrixType::NULLPTR_BARRAY;
    if (s == "NULLPTR_HANDLE" || s == "nullptr_handle")
        return GetrsMatrixType::NULLPTR_HANDLE;
    return GetrsMatrixType::RANDOM_NONSINGULAR;
}

inline BlasLapackMatrixType toBlasLapackMatrixType(GetrsMatrixType t)
{
    switch (t) {
        case GetrsMatrixType::IDENTITY: return BlasLapackMatrixType::IDENTITY;
        case GetrsMatrixType::DIAGONALLY_DOMINANT: return BlasLapackMatrixType::DIAGONALLY_DOMINANT;
        case GetrsMatrixType::ILL_CONDITIONED: return BlasLapackMatrixType::ILL_CONDITIONED;
        case GetrsMatrixType::DIAGONAL: return BlasLapackMatrixType::DIAGONAL;
        case GetrsMatrixType::UPPER_TRIANGULAR: return BlasLapackMatrixType::UPPER_TRIANGULAR;
        case GetrsMatrixType::LOWER_TRIANGULAR: return BlasLapackMatrixType::LOWER_TRIANGULAR;
        case GetrsMatrixType::SINGULAR_ZERO_COL: return BlasLapackMatrixType::SINGULAR_ZERO_COL;
        case GetrsMatrixType::SINGULAR_DEPENDENT_ROW: return BlasLapackMatrixType::SINGULAR_DEPENDENT_ROW;
        case GetrsMatrixType::MIXED: return BlasLapackMatrixType::MIXED;
        default: return BlasLapackMatrixType::RANDOM_NONSINGULAR;
    }
}

// Info pointer mode
enum class GetrsInfoMode {
    NORMAL,  // Pass valid info pointer
    NULLPTR  // Pass nullptr as info
};

inline GetrsInfoMode parseGetrsInfoMode(const std::string& s)
{
    if (s == "NULLPTR" || s == "nullptr")
        return GetrsInfoMode::NULLPTR;
    return GetrsInfoMode::NORMAL;
}

struct SgetrsBatchedParam : public BlasTestParamBase {
    aclblasOperation_t trans = ACLBLAS_OP_N;
    int n = 0;
    int nrhs = 0;
    int lda = 0;
    int ldb = 0;
    int batchCount = 0;
    GetrsPivotMode pivotMode = GetrsPivotMode::PIVOT;
    GetrsMatrixType matrixType = GetrsMatrixType::RANDOM_NONSINGULAR;
    GetrsInfoMode infoMode = GetrsInfoMode::NORMAL;

    SgetrsBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        trans = parseOpTrans(ReadMap(map, "trans", "N"));
        n = parseInt(ReadMap(map, "n", "0"));
        nrhs = parseInt(ReadMap(map, "nrhs", "0"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, n))));
        ldb = parseInt(ReadMap(map, "ldb", std::to_string(std::max(1, n))));
        batchCount = parseInt(ReadMap(map, "batch_count", "1"));
        pivotMode = parseGetrsPivotMode(ReadMap(map, "pivot_mode", "PIVOT"));
        matrixType = parseGetrsMatrixType(ReadMap(map, "matrix_type", "RANDOM_NONSINGULAR"));
        infoMode = parseGetrsInfoMode(ReadMap(map, "info_mode", "NORMAL"));
    }
};

#endif // SGETRS_BATCHED_PARAM_H
