/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trsm_batched_tiling_data.h
 * \brief Tiling data struct and shared BLAS enum constants for the StrsmBatched operator
 *        (included by both host and kernel).
 */

#pragma once

#include <cstdint>

struct TrsmBatchedTilingData {
    int32_t m;
    int32_t n;
    int32_t lda;
    int32_t ldb;
    int32_t batchCount;
    int32_t nb;
    int32_t totalCores;
    int32_t side;
    int32_t uplo;
    int32_t transa;
    int32_t diag;
    float alphaReal;
    int64_t workspaceOffset;
    int32_t coopMode;
    int32_t splitFactor;
    int32_t splitNcolsMax;
};

constexpr int32_t SIDE_LEFT = 0;
constexpr int32_t SIDE_RIGHT = 1;
constexpr int32_t UPLO_UPPER = 0;
constexpr int32_t UPLO_LOWER = 1;
constexpr int32_t TRANS_N = 0;
constexpr int32_t TRANS_T = 1;
constexpr int32_t DIAG_NONUNIT = 0;
constexpr int32_t DIAG_UNIT = 1;

constexpr int32_t FLOAT_ALIGN = 8;

#define CEIL_ALIGN(val, align) (((val) + (align) - 1) / (align) * (align))
