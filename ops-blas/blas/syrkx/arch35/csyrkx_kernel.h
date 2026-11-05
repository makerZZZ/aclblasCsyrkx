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
 * \file csyrkx_kernel.h
 * \brief Kernel launcher declarations for aclblasCsyrkx (arch35).
 *        Shared by csyrkx_host.cpp and csyrkx_kernel.cpp.
 *
 *        Phase 1 (AIC GEMM) reuses gemm_kernel_do from blas/gemm/arch35/gemm_kernel.h.
 */

#pragma once

#include <cstdint>
#include "csyrkx_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Phase 0: Deinterleave complex matrix -> real, imag (AIV-only).
// Called twice by the host: A -> Ar, Ai and B -> Br, Bi.
void csyrkx_deinterleave_kernel_do(
    GM_ADDR a, GM_ADDR ar, GM_ADDR ai, const CsyrkxDeinterleaveTilingData& tiling, uint32_t numBlocks, void* stream);

// Phase 2: Combine/Scale kernel (AIV-only).
//   t1=Ar*Br^T, t2=Ai*Bi^T, t3=Ar*Bi^T, t4=Ai*Br^T (trans=N; trans=T/C: A^T variants)
//   P_r = t1 - t2, P_i = t3 + t4
//   Cr  = alpha_r*P_r - alpha_i*P_i, Ci = alpha_r*P_i + alpha_i*P_r
//   C   = beta*C_old + (Cr + i*Ci) on the uplo triangle; the other triangle is
//   never referenced nor written (except restoring the lower-diagonal column
//   segment that shares a store with the diagonal cell).
void csyrkx_combine_kernel_do(
    GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, GM_ADDR t4, GM_ADDR c, const CsyrkxCombineTilingData& tiling,
    uint32_t numBlocks, void* stream);
