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
 * \file sgemm_ex_tiling_data.h
 * \brief Tiling data structure shared between host and kernel for sgemm_ex (FP32, SIMD membase).
 *
 * FP32-only GEMM: C = alpha * op(A) * op(B) + beta * C
 * Uses BlockMmad low-level API on arch35 (DAV_3510).
 */

#pragma once

#include <cstdint>

struct SgemmExTilingData {
    // Matrix dimensions (logical: op(A) is M*K, op(B) is K*N, C is M*N)
    int32_t m;
    int32_t n;
    int32_t k;
    // Leading dimensions (physical storage strides, column-major)
    int32_t lda;
    int32_t ldb;
    int32_t ldc;     // Cube kernel: tempAB row stride (= CeilAlign(m,16) when needPostProcess) or original ldc
    int32_t cLdc;    // Vector kernel: original ldc for accessing C matrix (BLAS allows ldc >= max(1,m))
    // Multi-core partitioning
    int32_t usedCoreNum;
    int32_t mBlocks;       // CeilDiv(m, singleCoreM)
    int32_t nBlocks;       // CeilDiv(n, singleCoreN)
    // Per-core workload
    int32_t singleCoreM;
    int32_t singleCoreN;
    // Cube tile sizes (FP32: baseM=32, baseN=16, baseK=8)
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    // Transpose flags
    int32_t isTransA;
    int32_t isTransB;
    // Alpha/beta (for Vector post-processing kernel)
    float alpha;
    float beta;
    int32_t hasBeta;       // 1 if beta != 0.0f
};
