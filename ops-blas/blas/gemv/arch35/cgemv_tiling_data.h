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
 * \file cgemv_tiling_data.h
 * \brief Tiling data structure shared between host and kernel for cgemv (SIMT).
 */

#pragma once

#include <cstdint>

// T/C slab: work items per block = columns x row segments; partials meet in a UB array
constexpr uint32_t CGEMV_T_MAX_ITEMS = 64;    // host target is <=40; leave headroom for a full 64-warp block
constexpr uint32_t CGEMV_T_MAX_SEG_SHIFT = 3; // max log2(row segments per column) -> 8 segments
// A 64-row N slab gives one thread per row, i.e. only two resident warps, which cannot
// hide GM latency on this architecture (measured: latency is hidden by warp count, not by
// per-thread unrolling).  When the slab owns enough columns to divide, each row is given
// several column groups so the block runs 2*groups warps; each group reduces its own
// column slice into its own workspace slab and the existing finalize sums the extra slabs.
// The gate is the number of columns available to split, not the shape of the case.
constexpr uint32_t CGEMV_NSMALL_ROWS = 64;
constexpr uint32_t CGEMV_NSMALL_MIN_COLS_PER_GROUP = 8;
constexpr uint32_t CGEMV_NSMALL_MAX_GROUPS = 8;

struct CgemvTilingData {
    uint32_t numThreads;  // threads per block
    uint32_t m;           // matrix A rows
    uint32_t n;           // matrix A columns
    uint32_t lda;         // A leading dimension (in complex elements)
    uint32_t trans;       // 0 = N, 1 = T, 2 = C (conjugate transpose)
    uint32_t alphaIsZero; // alpha == (0,0): scale-only path, A/x are not read
    uint32_t betaIsZero;  // beta == (0,0): y is not read (may be NaN/uninitialized)
    float alphaR;         // scalar alpha real part
    float alphaI;         // scalar alpha imaginary part
    float betaR;          // scalar beta real part
    float betaI;          // scalar beta imaginary part
    int64_t incx;         // x stride (in complex elements, may be negative)
    int64_t incy;         // y stride (in complex elements, may be negative)
    // Slab fast path (incx == 1): each block streams a contiguous column slab.
    uint32_t useSlab;   // 0 = generic paths; 1 = slab kernels
    uint32_t chunkLen;  // columns per slab (ceil(n / numBlocks))
    uint32_t colChunks; // number of column slabs (grid size of the slab compute launch)
    uint32_t segShift;  // T/C slab: log2(row segments per column); 0 = one column per warp
};
