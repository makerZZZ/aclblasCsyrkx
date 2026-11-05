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
 * \file csyrk_kernel.h
 * \brief Kernel launcher signatures for CSYRK on arch22, shared by the host and
 *        the kernel translation unit so that the compiler checks both against one
 *        declaration.
 *
 *        GM_ADDR is guarded: on the host side it falls back to a plain pointer,
 *        while the kernel translation unit includes kernel_operator.h first and
 *        therefore keeps AscendC's __gm__ qualified definition.
 */

#ifndef CSYRK_ARCH22_KERNEL_H
#define CSYRK_ARCH22_KERNEL_H

#include <cstdint>
#include "common/helper/complex_blas3_tiling_data.h"
#include "csyrk_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Phase 0 (AIV): complex A -> the three K-concatenated real operands
// P = [Ar|Ai], Q = [Ar|-Ai], R = [Ai|Ar]. Shared implementation.
void csyrk_split_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR a, GM_ADDR p, GM_ADDR q, GM_ADDR r, CBlas3SplitConcatTilingData tiling);

// Phase 1 (AIC): one real GEMM. Dispatches on tiling.transMode.
void csyrk_gemm_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR left, GM_ADDR right, GM_ADDR out, CBlas3GemmTilingData tiling);

// Phase 2 (AIV): complex alpha/beta scaling of the two GEMM results.
void csyrk_combine_kernel_do(
    uint32_t blockDim, void* stream, GM_ADDR pr, GM_ADDR pi, GM_ADDR c, CsyrkCombineTilingData tiling);

// The compile-time singleK of the Matmul static tiling, so the host can split K
// without duplicating the constant.
uint32_t csyrk_gemm_single_k();

#endif // CSYRK_ARCH22_KERNEL_H
