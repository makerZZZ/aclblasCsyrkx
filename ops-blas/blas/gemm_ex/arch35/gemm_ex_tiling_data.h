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
 * \file gemm_ex_tiling_data.h
 * \brief Tiling data structure shared between host and kernel for gemm_ex (SIMD membase).
 *
 * Phase 2: Custom tiling for BlockMmad low-level API (replaces TCubeTiling).
 * The host manually computes all tiling parameters; the kernel uses them directly.
 */

#pragma once

#include <cstdint>

enum GemmDTypeCase : int32_t {
    GEMM_DTYPE_FP16 = 0,
    GEMM_DTYPE_BF16 = 1,
    GEMM_DTYPE_FP32 = 2,
    GEMM_DTYPE_FP8_E4M3 = 3,
    GEMM_DTYPE_FP8_E5M2 = 4,
    GEMM_DTYPE_FP8_E5M2_E4M3 = 5,
    GEMM_DTYPE_FP8_E4M3_E5M2 = 6,
    GEMM_DTYPE_FP16_OUT_F32 = -1,
    GEMM_DTYPE_BF16_OUT_F32 = -2,
    GEMM_DTYPE_INVALID = -3
};

struct GemmExTilingData {
    // Matrix dimensions (logical: op(A) is M×K, op(B) is K×N, C is M×N)
    int32_t m;
    int32_t n;
    int32_t k;
    // Leading dimensions (physical storage strides, column-major)
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    // Multi-core partitioning
    int32_t usedCoreNum;
    int32_t mBlocks; // CeilDiv(m, singleCoreM)
    int32_t nBlocks; // CeilDiv(n, singleCoreN)
    // Per-core workload
    int32_t singleCoreM;
    int32_t singleCoreN;
    // Cube tile sizes (hardware-dependent)
    int32_t baseM;  // 16 for FP16/BF16
    int32_t baseN;  // 16 for FP16/BF16
    int32_t baseK;  // 16 for FP16/BF16, 8 for FP32, 32 for FP8
    int32_t c0Size; // Same as baseK for arch35
    // Transpose flags
    int32_t isTransA;
    int32_t isTransB;
    // Alpha/beta (for kernel-side fusion if needed)
    float alpha;
    float beta;
    int32_t hasBeta;
    // Output control
    int32_t outputFp32; // 1 = L0C→GM as FP32, 0 = quantize to FP16/BF16
};
