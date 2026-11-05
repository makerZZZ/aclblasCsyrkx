/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file stpsv_tiling_data.h
 * \brief Tiling data for single-precision triangular packed solver (arch22, Atlas A2/A3).
 */

#pragma once

#include <cstdint>

// Largest order this operator accepts.
//
// The whole solution vector x is kept resident in UB (that is what makes the single-core
// dependency chain as short as it is), so the order is bounded by the UB capacity:
// n*4 bytes for x plus ~32 KB of DMA/ReduceSum scratch. 32768 keeps the UB footprint at
// ~160 KB of the 192 KB available per AIV while staying 8x above the largest case in the
// official test set (n = 4096). Requests above this bound are rejected with
// ACLBLAS_STATUS_NOT_SUPPORTED instead of being silently mis-computed.
constexpr uint32_t ACLBLAS_STPSV_MAX_N = 32768U;

struct StpsvTilingData {
    uint64_t ap;    // device pointer to the packed triangular matrix (length n*(n+1)/2)
    uint64_t x;     // device pointer to the right-hand side / solution vector
    uint32_t n;     // order of the triangular system
    uint32_t uplo;  // ACLBLAS_UPPER or ACLBLAS_LOWER
    uint32_t trans; // ACLBLAS_OP_N, ACLBLAS_OP_T or ACLBLAS_OP_C
    uint32_t diag;  // ACLBLAS_NON_UNIT or ACLBLAS_UNIT
    int64_t incx;   // stride of x (must be non-zero)
};
