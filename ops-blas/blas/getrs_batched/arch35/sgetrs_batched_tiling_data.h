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
 * \file sgetrs_batched_tiling_data.h
 * \brief Tiling data for batched single-precision triangular solve (SgetrsBatched).
 */

#pragma once

#include <cstdint>

inline constexpr uint32_t GETRS_TRANS_N = 0;
inline constexpr uint32_t GETRS_TRANS_T = 1;
inline constexpr uint32_t GETRS_TRANS_C = 2;

struct SgetrsBatchedTilingData {
    uint32_t n;            // matrix dimension (square matrix side length)
    uint32_t nrhs;         // number of right-hand side columns
    uint32_t lda;          // leading dimension of Aarray[i]
    uint32_t ldb;          // leading dimension of Barray[i]
    uint32_t usedCoreNum;  // actually used Core count
    uint32_t batchPerCore; // batches per core (except last core)
    uint32_t batchTail;    // batches for the last core
    uint32_t usePivot;     // 1 = use pivoting, 0 = no pivoting (devIpiv == NULL)
    uint32_t trans;        // 0 = OP_N, 1 = OP_T, 2 = OP_C
};
