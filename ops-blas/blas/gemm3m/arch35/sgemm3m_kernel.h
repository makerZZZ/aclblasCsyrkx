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
 * \file sgemm3m_kernel.h
 * \brief kernel_do declarations for gemm3m (host and kernel shared).
 */

#pragma once

#include <cstdint>
#include "sgemm3m_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

// Cube kernel launcher: receives swapped tilingData (const ref, launch by value)
// Sub-matrix pointers (a1/a2/a3, b1/b2/b3) are computed inside the kernel from
// the merged A and B matrices.
void gemm3m_kernel_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR a, GM_ADDR b,
    GM_ADDR c, const Gemm3MTilingData& tilingData);

// Alpha/beta vector kernel launcher: receives unswapped abTilingData (const ref)
void gemm3m_alpha_beta_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR tempAB, GM_ADDR cOrig, GM_ADDR cOut,
    const Gemm3MTilingData& abTilingData);
