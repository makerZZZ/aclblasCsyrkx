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
 * \file csyrkx_tiling_data.h
 * \brief Tiling data structures for aclblasCsyrkx (arch35).
 *        Shared by host side (const ref) and kernel side (by value).
 *
 *        C = alpha * op(A) * op(B)^T + beta * C, only the uplo triangle is written.
 *        op(A)/op(B) = A/B (trans=N, A/B are n x k) or A^T/B^T (trans=T/C, A/B are k x n).
 *        OP_C is equivalent to OP_T: syrkx produces a symmetric (not Hermitian)
 *        matrix, so no conjugation is applied.
 *
 *        Phase 0 (AIV): deinterleave A -> Ar, Ai and B -> Br, Bi. Called twice.
 *        Phase 1 (AIC): four real GEMMs (4M decomposition), tiling is GemmTilingData.
 *        Phase 2 (AIV): combine P_r = t1 - t2, P_i = t3 + t4, scale by complex
 *                       alpha/beta, store the uplo triangle.
 */

#pragma once

#include <cstdint>

// [general] SIMD mode parameters for AIV kernels
constexpr uint32_t CSYRKX_ARCH35_SCALE_BLOCK = 64;       // rows/cols per scale tile
constexpr uint32_t CSYRKX_ARCH35_ELEMENTS_PER_BLOCK = 8; // 32 bytes / sizeof(float)

// Phase 0: Deinterleave one complex matrix -> real, imag (AIV-only).
// Called twice: A -> Ar, Ai and B -> Br, Bi.
struct CsyrkxDeinterleaveTilingData {
    uint32_t rows;        // physical rows (trans=N: n, trans=T/C: k)
    uint32_t cols;        // physical cols (trans=N: k, trans=T/C: n)
    uint32_t lda;         // leading dimension of the complex input (elements)
    uint32_t rowsPerCore; // rows handled by each AIV core
};

// Phase 2: Combine/Scale tiling (AIV-only).
//   P_r = t1 - t2, P_i = t3 + t4
//   Cr  = alpha_r * P_r - alpha_i * P_i
//   Ci  = alpha_r * P_i + alpha_i * P_r
//   C   = (Cr + i*Ci) + beta * C_old, uplo triangle only
struct CsyrkxCombineTilingData {
    uint32_t n;           // C order
    uint32_t ldc;         // C leading dimension (complex elements)
    uint32_t tempLdc;     // temp row stride (= CeilAlign(n, GEMM_FRACTAL))
    uint32_t rowsPerCore; // rows per AIV core
    float alphaReal;      // alpha real part
    float alphaImag;      // alpha imaginary part
    float betaReal;       // beta real part
    float betaImag;       // beta imaginary part
    uint8_t uploMode;     // ACLBLAS_UPPER / ACLBLAS_LOWER
    uint8_t isAlphaZero;  // alpha == (0, 0) short-circuit
    uint8_t isKZero;      // k == 0 short-circuit
    uint8_t isBetaZero;   // beta == (0, 0) short-circuit
};
