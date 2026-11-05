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
#include "cher2k_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

constexpr uint32_t CHER2K_DIRECT_INTERLEAVE_ROWS = 64U;
constexpr uint32_t CHER2K_DIRECT_INTERLEAVE_COLS = 64U;
constexpr uint32_t CHER2K_DIRECT_INTERLEAVE_COUNT = CHER2K_DIRECT_INTERLEAVE_ROWS * CHER2K_DIRECT_INTERLEAVE_COLS * 2U;
constexpr uint32_t CHER2K_SMALL_INTERLEAVE_ROWS = 128U;
constexpr uint32_t CHER2K_SMALL_INTERLEAVE_COLS = 8U;
constexpr uint32_t CHER2K_SMALL_INTERLEAVE_OFFSET = CHER2K_DIRECT_INTERLEAVE_COUNT;
constexpr uint32_t CHER2K_SMALL_INTERLEAVE_COUNT = CHER2K_SMALL_INTERLEAVE_ROWS * CHER2K_SMALL_INTERLEAVE_COLS * 2U;
constexpr uint32_t CHER2K_TRANSPOSE_OFFSET_ROWS = 32U;
constexpr uint32_t CHER2K_TRANSPOSE_OFFSET_COLS = 64U;
constexpr uint32_t CHER2K_TRANSPOSE_OFFSET = CHER2K_SMALL_INTERLEAVE_OFFSET + CHER2K_SMALL_INTERLEAVE_COUNT;
constexpr uint32_t CHER2K_TRANSPOSE_OFFSET_COUNT = CHER2K_TRANSPOSE_OFFSET_ROWS * CHER2K_TRANSPOSE_OFFSET_COLS;
constexpr uint32_t CHER2K_TINY_INTERLEAVE_ROWS = 32U;
constexpr uint32_t CHER2K_TINY_INTERLEAVE_COLS = 1U;
constexpr uint32_t CHER2K_TINY_INTERLEAVE_OFFSET = CHER2K_TRANSPOSE_OFFSET + CHER2K_TRANSPOSE_OFFSET_COUNT;
constexpr uint32_t CHER2K_TINY_INTERLEAVE_COUNT = CHER2K_TINY_INTERLEAVE_ROWS * CHER2K_TINY_INTERLEAVE_COLS * 2U;
constexpr uint32_t CHER2K_POST_TRANSPOSE_ROWS = 64U;
constexpr uint32_t CHER2K_POST_TRANSPOSE_COLS = 64U;
constexpr uint32_t CHER2K_POST_TRANSPOSE_OFFSET = CHER2K_TINY_INTERLEAVE_OFFSET + CHER2K_TINY_INTERLEAVE_COUNT;
constexpr uint32_t CHER2K_POST_TRANSPOSE_COUNT = CHER2K_POST_TRANSPOSE_ROWS * CHER2K_POST_TRANSPOSE_COLS;
constexpr uint32_t CHER2K_INTERLEAVE_OFFSET_COUNT = CHER2K_POST_TRANSPOSE_OFFSET + CHER2K_POST_TRANSPOSE_COUNT;

void cher2k_kernel_do(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, const Cher2kTilingData& tiling, void* stream,
    bool skipPreprocess = false, GM_ADDR alpha = nullptr, GM_ADDR beta = nullptr, GM_ADDR interleaveOffsets = nullptr);
