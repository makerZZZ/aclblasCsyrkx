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
 * \file sgemm_ex_kernel.h
 * \brief Kernel launcher declarations for sgemm_ex (host.cpp / kernel.cpp shared).
 *
 * Two kernel launchers:
 *   1. sgemm_ex_kernel_do       — Cube kernel: op(B') * op(A') matrix multiply
 *   2. sgemm_ex_alpha_beta_do   — Vector kernel: C = alpha * tempAB + beta * C
 *
 * Tiling passed by const reference from host; kernel launch copies by value.
 */

#pragma once

#include <cstdint>
#include "sgemm_ex_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Cube kernel launcher: executes op(B') * op(A') on Cube cores
// a/b/c are GM device pointers (after column-major swap: a=B, b=A, c=tempAB or C)
void sgemm_ex_kernel_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR a, GM_ADDR b, GM_ADDR c,
    const SgemmExTilingData& tilingData);

// Vector kernel launcher: post-processing C = alpha * tempAB + beta * C
// tempAB reads from workspace; cOrig/cOut point to the user C matrix
void sgemm_ex_alpha_beta_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR tempAB, GM_ADDR cOrig, GM_ADDR cOut,
    const SgemmExTilingData& tilingData);
