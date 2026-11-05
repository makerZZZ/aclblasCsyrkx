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
 * \file gemm_strided_batched_kernel.h
 * \brief kernel_do launcher declarations shared by host.cpp and kernel.cpp.
 *
 * Data GM pointers use GM_ADDR. Tiling is passed by const reference (no H2D copy). Kernels launch
 * asynchronously; the caller is responsible for stream synchronization (hard rule 6).
 */

#pragma once

#include <cstdint>
#include "gemm_strided_batched_tiling_data.h"

// GM_ADDR resolves to the real __gm__ pointer type on the device side (kernel.cpp includes
// kernel_operator.h first); on the host side (host.cpp) this fallback keeps it a plain uint8_t*
// so plan pointers can be reinterpret_cast without the __gm__ address-space qualifier.
#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Blaze GEMM kernel launcher (AIC). Computes C^T = op(B)^T op(A)^T with gmLeft = B_i, gmRight = A_i.
// gmOut is C_i directly (fast path, ldC = user ldc) or a compact workspace temp (general / degraded path).
void gemm_sb_gemm_kernel_do(
    uint32_t numBlocks, void* stream, GM_ADDR gmLeft, GM_ADDR gmRight, GM_ADDR gmOut,
    const GemmSbGemmTilingData& tiling);

// SIMD-membase alpha/beta combine launcher (AIV). Reads the GEMM temp (C^T, row stride tempRowStride) and
// the original C_i (user ldc), writes cOut = alpha * temp + beta * cOrig back to C_i.
void gemm_sb_combine_kernel_do(
    uint32_t numBlocks, void* stream, GM_ADDR gmTemp, GM_ADDR gmCOrig, GM_ADDR gmCOut,
    const GemmSbCombineTilingData& tiling);

// SIMD-membase beta-scale launcher (AIV). In-place cOut = beta * cOut for the only-scale path
// (k = 0 or alpha = 0 with beta not in {0, 1}).
void gemm_sb_beta_scale_kernel_do(
    uint32_t numBlocks, void* stream, GM_ADDR gmC, const GemmSbCombineTilingData& tiling);
