/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sgemm3m_tiling_data.h
 * \brief Tiling data structure shared between host and kernel for gemm3m (tensor_api).
 *
 * C = alpha * (A1*B1 + A2*B2 + A3*B3) + beta * C (column-major)
 * After column-major swap: C^T = B1^T*A1^T + B2^T*A2^T + B3^T*A3^T (row-major)
 */

#pragma once

#include <cstdint>

constexpr int32_t GEMM3M_BASE_M = 128;
constexpr int32_t GEMM3M_BASE_N = 128;
constexpr int32_t GEMM3M_BASE_K = 64;
constexpr int32_t GEMM3M_K_L1 = 64;
constexpr int32_t GEMM3M_NUM_PAIRS = 3;

struct Gemm3MTilingData {
    // Matrix dimensions (after column-major swap: m=N_orig, n=M_orig)
    int32_t m;       // = N_orig (first operand rows)
    int32_t n;       // = M_orig (second operand cols)
    int32_t k;       // = K (reduction dimension)

    // Leading dimensions (after swap: lda=ldb_orig, ldb=lda_orig)
    int32_t lda;     // = ldb_orig
    int32_t ldb;     // = lda_orig
    int32_t ldc;     // swap before = ldc_orig; needPostProcess => m (temp stride)

    // Original ldc (C matrix column-major stride), unaffected by swap / temp stride
    int32_t ldcOrig; // = user-provided ldc, used by AlphaBetaKernel

    // Multi-core partitioning
    int32_t usedCoreNum;
    int32_t mBlocks;
    int32_t nBlocks;
    int32_t singleCoreM;
    int32_t singleCoreN;

    // Alpha/beta
    float alpha;
    float beta;

    // Transpose/post-process flags (0/1 boolean semantics, packed at end)
    int8_t isTransA;
    int8_t isTransB;
    int8_t hasBeta;          // beta != 0.0f ? 1 : 0
    int8_t needPostProcess;  // (alpha != 1.0f || beta != 0.0f) ? 1 : 0
};
