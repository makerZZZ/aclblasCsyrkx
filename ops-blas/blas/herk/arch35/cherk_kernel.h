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
 * \file cherk_kernel.h
 * \brief Declaration of kernel launchers for aclblasCherk (arch35).
 *        Shared by host.cpp and kernel.cpp.
 *
 *        Phase 1 (AIC GEMM) reuses gemm_kernel_do from blas/gemm/arch35/gemm_kernel.h.
 *        Phase 0 and Phase 2 kernels are declared here.
 */

#pragma once

#include <cstdint>
#include "cherk_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Phase 0: Deinterleave complex A → Ar, Ai (AIV-only)
// Reads complex A from GM, splits into real/imag float matrices in GM.
void cherk_deinterleave_kernel_do(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai,
    const CherkDeinterleaveTilingData& tiling,
    uint32_t numBlocks, void* stream);

// Phase 2: Combine/Scale/Hermitian kernel (AIV-only)
// Merges 4 real GEMM results (t1..t4) into complex C with alpha/beta scaling and Hermitian constraint.
//   Cr = t1 + t2 (symmetric), Ci = t3 - t4 (anti-symmetric, diagonal = 0)
//   C = alpha * (Cr + i*Ci) + beta * C_old
//   Only writes uplo triangle; diagonal imaginary part is forced to zero.
void cherk_combine_kernel_do(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4,
    GM_ADDR c,
    const CherkCombineTilingData& tiling,
    uint32_t numBlocks, void* stream);
