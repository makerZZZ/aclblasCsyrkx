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
 * \file csymm_kernel.h
 * \brief Kernel launcher signatures for CSYMM on arch22, shared by the host and the
 *        kernel translation unit so that the compiler checks both against one
 *        declaration.
 */

#ifndef CSYMM_ARCH22_KERNEL_H
#define CSYMM_ARCH22_KERNEL_H

#include <cstdint>
#include "common/helper/complex_blas3_tiling_data.h"
#include "csymm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Phase 0a (AIV): complex input -> packed real parts. Called twice, for A and B.
void csymm_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR src, GM_ADDR re, GM_ADDR im, CBlas3SplitTilingData tiling);

// Phase 0b (AIV): mirror the triangular-stored A into a full square, in place on
// the packed buffers Phase 0a produced.
void csymm_expand_kernel_do(uint32_t blockDim, void* stream, GM_ADDR re, GM_ADDR im, CBlas3ExpandTilingData tiling);

// Phase 1 (AIC): one real GEMM, both operands plain.
void csymm_gemm_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling);

// Phase 2 (AIV): assemble and scale the complex result.
void csymm_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c,
    CsymmCombineTilingData tiling);

// The compile-time singleK of the Matmul static tiling, so the host can split K
// without duplicating the constant.
uint32_t csymm_gemm_single_k();

#endif // CSYMM_ARCH22_KERNEL_H
