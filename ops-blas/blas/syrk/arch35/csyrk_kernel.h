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
 * \brief Declaration of kernel launchers for aclblasCsyrk (arch35).
 *        Shared by host.cpp and kernel.cpp.
 *
 *        Phase 0 (AIV): deinterleave complex A -> Ar, Ai, AiNeg
 *        Phase 1 (AIC): fused 4M cube GEMM (native BlockMmad API)
 *        Phase 2 (AIV): combine/scale C = alpha*(Cr + i*Ci) + beta*C_old
 */

#pragma once

#include <cstdint>
#include "csyrk_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Phase 0: Deinterleave complex A -> Ar, Ai (AIV-only)
void csyrk_deinterleave_kernel_do(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, const CsyrkDeinterleaveTilingData& tiling, uint32_t numBlocks, void* stream);

// HF32 residual kernel: writes x - HF32(x) into arLow/aiLow (SIMT).
void csyrk_hf32resid_kernel_do(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR arLow, GM_ADDR aiLow, const CsyrkDeinterleaveTilingData& tiling, uint32_t numBlocks,
    void* stream);

// Direct single-kernel SIMT path for tiny shapes (n<=64 && k<=256 or strided).
void csyrk_direct_kernel_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, const CsyrkDirectTilingData& tiling, uint32_t numBlocks,
    void* stream);

// Phase 1: Fused 4M cube GEMM (AIC-only)
//   temp (n x 4n) = [Q0|Q1|Q2|Q3] (triangular tiles only)
//   arLow/aiLow: HF32x3 residual buffers (x - HF32(x)); pass nullptr for fp32.
void csyrk_cube_kernel_do(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR temp, const CsyrkCubeTilingData& tiling, uint32_t numBlocks, void* stream,
    GM_ADDR arLow = nullptr, GM_ADDR aiLow = nullptr);

// Phase 2: Combine/Scale kernel (AIV-only)
void csyrk_combine_kernel_do(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR temp, GM_ADDR c, const CsyrkCombineTilingData& tiling, uint32_t numBlocks,
    void* stream);

// Phase 2 (SIMT): small-n combine, one thread per uplo element (fast config).
void csyrk_combine_simt_kernel_do(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR temp, GM_ADDR c, const CsyrkCombineTilingData& tiling, uint32_t numBlocks,
    void* stream);
