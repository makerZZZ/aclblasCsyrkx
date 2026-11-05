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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

enum class MatinvMatrixType {
    RANDOM_NONSINGULAR,
    IDENTITY,
    DIAGONAL,
    DIAGONALLY_DOMINANT,
    ILL_CONDITIONED,
    UPPER_TRIANGULAR,
    LOWER_TRIANGULAR,
    SINGULAR_ZERO_COL,
    SINGULAR_DEPENDENT_ROW,
    MIXED,
    NULLPTR_A,
    NULLPTR_AINV,
    NULLPTR_INFO,
    NULLPTR_ALL
};

inline MatinvMatrixType parseMatinvMatrixType(const std::string& s)
{
    if (s == "IDENTITY")
        return MatinvMatrixType::IDENTITY;
    if (s == "DIAGONAL")
        return MatinvMatrixType::DIAGONAL;
    if (s == "DIAGONALLY_DOMINANT")
        return MatinvMatrixType::DIAGONALLY_DOMINANT;
    if (s == "ILL_CONDITIONED")
        return MatinvMatrixType::ILL_CONDITIONED;
    if (s == "UPPER_TRIANGULAR")
        return MatinvMatrixType::UPPER_TRIANGULAR;
    if (s == "LOWER_TRIANGULAR")
        return MatinvMatrixType::LOWER_TRIANGULAR;
    if (s == "SINGULAR_ZERO_COL")
        return MatinvMatrixType::SINGULAR_ZERO_COL;
    if (s == "SINGULAR_DEPENDENT_ROW")
        return MatinvMatrixType::SINGULAR_DEPENDENT_ROW;
    if (s == "MIXED")
        return MatinvMatrixType::MIXED;
    if (s == "NULLPTR_A" || s == "nullptr_A")
        return MatinvMatrixType::NULLPTR_A;
    if (s == "NULLPTR_AINV" || s == "nullptr_Ainv")
        return MatinvMatrixType::NULLPTR_AINV;
    if (s == "NULLPTR_INFO" || s == "nullptr_info")
        return MatinvMatrixType::NULLPTR_INFO;
    if (s == "NULLPTR_ALL" || s == "nullptr_all")
        return MatinvMatrixType::NULLPTR_ALL;
    return MatinvMatrixType::RANDOM_NONSINGULAR;
}

inline BlasLapackMatrixType toBlasLapackType(MatinvMatrixType t)
{
    switch (t) {
        case MatinvMatrixType::RANDOM_NONSINGULAR:
            return BlasLapackMatrixType::RANDOM_NONSINGULAR;
        case MatinvMatrixType::IDENTITY:
            return BlasLapackMatrixType::IDENTITY;
        case MatinvMatrixType::DIAGONAL:
            return BlasLapackMatrixType::DIAGONAL;
        case MatinvMatrixType::DIAGONALLY_DOMINANT:
            return BlasLapackMatrixType::DIAGONALLY_DOMINANT;
        case MatinvMatrixType::ILL_CONDITIONED:
            return BlasLapackMatrixType::ILL_CONDITIONED;
        case MatinvMatrixType::UPPER_TRIANGULAR:
            return BlasLapackMatrixType::UPPER_TRIANGULAR;
        case MatinvMatrixType::LOWER_TRIANGULAR:
            return BlasLapackMatrixType::LOWER_TRIANGULAR;
        case MatinvMatrixType::SINGULAR_ZERO_COL:
            return BlasLapackMatrixType::SINGULAR_ZERO_COL;
        case MatinvMatrixType::SINGULAR_DEPENDENT_ROW:
            return BlasLapackMatrixType::SINGULAR_DEPENDENT_ROW;
        case MatinvMatrixType::MIXED:
            return BlasLapackMatrixType::MIXED;
        default:
            return BlasLapackMatrixType::RANDOM_NONSINGULAR;
    }
}

struct MatinvBatchedParam : public BlasTestParamBase {
    int n = 0;
    int lda = 0;
    int ldaInv = 0;
    int batchSize = 0;
    MatinvMatrixType matrixType = MatinvMatrixType::RANDOM_NONSINGULAR;

    MatinvBatchedParam(const csv_map& map) : BlasTestParamBase(map)
    {
        n = parseInt(ReadMap(map, "n", "0"));
        lda = parseInt(ReadMap(map, "lda", std::to_string(std::max(1, n))));
        ldaInv = parseInt(ReadMap(map, "lda_inv", std::to_string(std::max(1, n))));
        batchSize = parseInt(ReadMap(map, "batch_size", "1"));
        matrixType = parseMatinvMatrixType(ReadMap(map, "matrix_type", "RANDOM_NONSINGULAR"));
    }
};
