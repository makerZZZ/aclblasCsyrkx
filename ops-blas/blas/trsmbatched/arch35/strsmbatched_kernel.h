/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "strsmbatched_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

void trsmbatched_scale_kernel_do(
    GM_ADDR b, float alpha, uint32_t m, uint32_t n, int32_t ldb, uint32_t numBlocks, void* stream);

void trsmbatched_zero_kernel_do(
    GM_ADDR b, uint32_t m, uint32_t n, int32_t ldb, uint32_t numBlocks, void* stream);

void trsmbatched_panel_kernel_do(
    GM_ADDR a, GM_ADDR b, const TrsmbatchedPanelTilingData& tiling, uint32_t numBlocks, void* stream);

void trsmbatched_gemm_kernel_do(
    GM_ADDR a, GM_ADDR x, GM_ADDR temp, const TrsmbatchedGemmTilingData& tiling, uint32_t numBlocks, void* stream);

void trsmbatched_axpy_kernel_do(
    GM_ADDR b, GM_ADDR temp, const TrsmbatchedAxpyTilingData& tiling, uint32_t numBlocks, void* stream);

void trsmbatched_axpy_trans_kernel_do(
    GM_ADDR b, GM_ADDR temp, const TrsmbatchedAxpyTilingData& tiling, uint32_t numBlocks, void* stream);

void trsmbatched_extract_a_kernel_do(
    GM_ADDR a, GM_ADDR ws, uint32_t mC, uint32_t bs, uint32_t aWsStride, uint32_t lda,
    uint64_t aOffset, uint32_t numBlocks, void* stream);

void trsmbatched_extract_b_kernel_do(
    GM_ADDR b, GM_ADDR ws, uint32_t bs, uint32_t n, uint32_t bWsStride, uint32_t ldb,
    uint64_t bOffset, uint32_t numBlocks, void* stream);

void trsmbatched_transpose_kernel_do(
    GM_ADDR in, GM_ADDR out, uint32_t rows, uint32_t cols, int32_t ldIn, int32_t ldOut,
    uint32_t numBlocks, void* stream);
