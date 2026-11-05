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
 * \file cgetri_batched_tiling_data.h
 * \brief Tiling data for batched single-precision complex matrix inversion.
 */

#pragma once

#include <cstdint>

constexpr uint32_t CGETRI_BLOCKED_TILE_N = 64;
constexpr uint32_t CGETRI_BLOCKED_THRESHOLD_N = 321;

struct CgetriBatchedTilingData {
    uint32_t n;
    uint32_t lda;
    uint32_t ldc;
    uint32_t usedCoreNum;
    uint32_t batchPerCore;
    uint32_t batchTail;
    uint32_t usePivot;
};

struct CgetriBlockedInitTilingData {
    uint32_t n;
    uint32_t lda;
    uint32_t batchSize;
    uint32_t matrixStride;
    uint32_t usedCoreNum;
    uint32_t batchPerCore;
    uint32_t batchTail;
    uint32_t usePivot;
};

struct CgetriBlockedSolveTilingData {
    uint32_t n;
    uint32_t batchSize;
    uint32_t matrixStride;
    uint32_t rowStart;
    uint32_t blockSize;
    uint32_t isLower;
    uint32_t usedCoreNum;
};

struct CgetriBlockedCombineTilingData {
    uint32_t n;
    uint32_t updateRows;
    uint32_t batchSize;
    uint32_t matrixStride;
    uint32_t tempRowStride;
    uint32_t tempMatrixStride;
    uint32_t targetRow;
    uint32_t isImag;
    uint32_t usedCoreNum;
    uint64_t totalElements;
};

struct CgetriBlockedFinalizeTilingData {
    uint32_t n;
    uint32_t ldc;
    uint32_t batchSize;
    uint32_t matrixStride;
    uint32_t usedCoreNum;
    uint64_t totalElements;
};
