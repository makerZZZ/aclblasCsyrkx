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
 * \file matrix_transform_perm_table.h
 * \brief Complex-layout permutation table builder (single data source shared by the
 *        operator device path and the test golden), see design 1.3.A section 3.6.
 *        Used by the iteration-two complex-layout branches (COL4_4R2_8C / COL32_2R_4R4).
 */

#pragma once

#include <cstdint>

#include "matrix_transform_tiling_data.h"

// These offset helpers are the single permutation data source shared by the operator Host
// routing (which uploads the in-tile offset table to GM) and the test golden de-layout, so the
// device write placement and the golden agree byte-for-byte (review gate I-01). They are plain
// host inline functions; the device kernel consumes the Host-built GM table rather than calling
// these directly, keeping one authoritative permutation body.
#define MT_PERM_FN inline

// Composite tile geometry per complex order.
constexpr uint32_t MT_COL4_4R2_8C_ROWS = 8;
constexpr uint32_t MT_COL4_4R2_8C_COLS = 32;
constexpr uint32_t MT_COL32_2R_4R4_ROWS = 32;
constexpr uint32_t MT_COL32_2R_4R4_COLS = 32;

// COL32_2R_4R4 in-tile element offset: offset(row,col) = rowPerm*32 + col,
// rowPerm = ((row % 8) / 2 * 4 + row / 8) * 2 + row % 2 (design 1.3.A section 3.6).
MT_PERM_FN uint32_t MatCol32R2R4R4Offset(uint32_t row, uint32_t col)
{
    const uint32_t rowPerm = ((row % 8U) / 2U * 4U + row / 8U) * 2U + row % 2U;
    return rowPerm * MT_COL32_2R_4R4_COLS + col;
}

// COL4_4R2_8C in-tile element offset (design 1.3.A section 3.6).
MT_PERM_FN uint32_t MatCol44R28COffset(uint32_t row, uint32_t col)
{
    const uint32_t colOuter = col / 4U;
    const uint32_t colInner = col % 4U;
    const uint32_t rowPair = row / 2U;
    const uint32_t rowInPair = row % 2U;
    return ((rowPair * 8U + colOuter) * 8U) + (rowInPair * 4U + colInner);
}

// Number of rows in one composite tile for a complex order (0 for non-complex orders).
MT_PERM_FN uint32_t MatComplexTileRows(uint8_t order)
{
    if (order == MT_ORDER_COL4_4R2_8C) {
        return MT_COL4_4R2_8C_ROWS;
    }
    if (order == MT_ORDER_COL32_2R_4R4) {
        return MT_COL32_2R_4R4_ROWS;
    }
    return 0U;
}

// In-tile element offset for a complex order: maps (rowInTile, colInTile) to the element offset
// inside one composite tile (size = tileRows * 32). Both complex orders share colspan 32.
MT_PERM_FN uint32_t MatComplexTileOffset(uint8_t order, uint32_t rowInTile, uint32_t colInTile)
{
    if (order == MT_ORDER_COL4_4R2_8C) {
        return MatCol44R28COffset(rowInTile, colInTile);
    }
    return MatCol32R2R4R4Offset(rowInTile, colInTile);
}

// Build the column-major in-tile element-offset table consumed by the device group path.
// Layout: table[colInTile * tileRows + rowInTile] = in-tile element offset, so that a
// column-major-within-tile UB buffer (the kernel's accTile) maps element k to its physical
// slot via Gather/Scatter. Returns tileRows*32, or 0 for non-complex orders.
inline uint32_t MatBuildTileColMajorTable(uint8_t order, uint32_t* table, uint32_t tableCapacity)
{
    if (table == nullptr) {
        return 0U;
    }
    const uint32_t tileRows = MatComplexTileRows(order);
    if (tileRows == 0U) {
        return 0U;
    }
    // The device tile uses a fixed 32-element column stride so that the per-column INT8 load lands
    // on a 32-byte-aligned UB slot. Valid (rowInTile < tileRows) entries carry the real in-tile
    // offset; padding rows are routed to unique scratch slots in [tileSize, 1024) (never copied to
    // GM) to keep the Scatter a clean bijection over the full 1024-element column-major tile.
    const uint32_t fullCount = 32U * 32U;
    const uint32_t tileSize = tileRows * 32U;
    if (fullCount > tableCapacity) {
        return 0U;
    }
    uint32_t padSlot = tileSize;
    for (uint32_t colInTile = 0U; colInTile < 32U; ++colInTile) {
        for (uint32_t rowInTile = 0U; rowInTile < 32U; ++rowInTile) {
            const uint32_t idx = colInTile * 32U + rowInTile;
            if (rowInTile < tileRows) {
                table[idx] = MatComplexTileOffset(order, rowInTile, colInTile);
            } else {
                table[idx] = padSlot++;
            }
        }
    }
    return fullCount;
}
