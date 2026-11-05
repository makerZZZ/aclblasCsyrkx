/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file sgeam_tiling_data.h
 * @brief Tiling data structure for Sgeam operator (host/device shared)
 *        C = alpha * op(A) + beta * op(B)
 */

#pragma once

#include <cstdint>
#include "cann_ops_blas_common.h"

struct SgeamTilingData {
    uint32_t m;             // Number of rows in output matrix C
    uint32_t n;             // Number of columns in output matrix C
    float alpha;            // Scalar alpha
    uint32_t alphaIsZero;   // Optimized flag indicating alpha == 0
    aclblasOperation_t opA; // Transpose type for matrix A
    uint32_t lda;           // Leading dimension of matrix A
    float beta;             // Scalar beta
    uint32_t betaIsZero;    // Optimized flag indicating beta == 0
    aclblasOperation_t opB; // Transpose type for matrix B
    uint32_t ldb;           // Leading dimension of matrix B
    uint32_t ldc;           // Leading dimension of matrix C
    // 2D multi-core tiling: colBlocks x mBlocks
    uint32_t colBlocks;      // Number of column blocks
    uint32_t perCoreN;       // Number of columns processed per core
    uint32_t remainder;      // Column remainder
    uint32_t mBlocks;        // Number of row blocks
    uint32_t perCoreMTile;   // Number of tiles processed per core in m dimension
    uint32_t mTileRemainder; // Tile count remainder in m dimension
    uint32_t tileM;          // Tile size in m dimension (number of elements)
    uint32_t colsIter;       // NN path: columns processed per inner iteration (multi-column)
    uint32_t reserved;       // Reserved for 8-byte alignment
};
