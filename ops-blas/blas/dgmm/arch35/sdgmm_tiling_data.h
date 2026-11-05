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
 * \file sdgmm_tiling_data.h
 * \brief Tiling data structure for aclblasSdgmm (arch35).
 *        Shared by host side (const ref) and kernel side (by value).
 */

#pragma once

#include <cstdint>

// Normalized mode constants shared by Host and Kernel.
// The Host converts aclblasSideMode_t (ACLBLAS_SIDE_LEFT=141 / RIGHT=142)
// to these values before storing in tiling.mode.
static constexpr uint32_t SDGMM_MODE_LEFT = 0;
static constexpr uint32_t SDGMM_MODE_RIGHT = 1;

/*!
 * \brief Tiling data for Sdgmm.
 *
 * Block decomposition: 2D grid of colBlocks × mBlocks.
 *   - colBlocks = min(n, aivCoreNum); each col-block handles perCoreN columns.
 *     The first `remainder` col-blocks get perCoreN+1 columns; the rest get
 *     perCoreN (balanced, not all dumped on the last block).
 *   - mBlocks > 1 only when n < aivCoreNum (tall-skinny case): the m dimension
 *     is split into mTile-sized segments and distributed across m-blocks.
 *     The first `mTileRemainder` m-blocks get perCoreMTile+1 tiles; the rest
 *     get perCoreMTile. When mBlocks==1, every core processes all m-tiles.
 *   - Block index layout: blockIdx = colBlock * mBlocks + mBlock.
 */
struct SdgmmTilingData {
    uint32_t mode;    // normalized: SDGMM_MODE_LEFT or SDGMM_MODE_RIGHT
    uint32_t m;       // number of rows of matrix A/C
    uint32_t n;       // number of columns of matrix A/C
    int32_t  incx;    // stride of vector x (may be negative)
    uint32_t lda;     // leading dimension of A (column-major)
    uint32_t ldc;     // leading dimension of C (column-major)

    uint32_t perCoreN;      // base columns per col-block
    uint32_t remainder;     // first `remainder` col-blocks get +1 column
    uint32_t mBlocks;       // number of m-blocks (1 when no m-split)
    uint32_t perCoreMTile;  // base m-tiles per m-block
    uint32_t mTileRemainder;// first `mTileRemainder` m-blocks get +1 tile
    uint32_t tileM;         // row-segment tile size in m dimension (element count)
};
