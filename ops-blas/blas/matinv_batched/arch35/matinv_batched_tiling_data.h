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
 * \file matinv_batched_tiling_data.h
 * \brief Tiling data for batched single-precision matrix inversion (SmatinvBatched).
 */

#pragma once

#include <cstdint>

struct SmatinvBatchedTilingData {
    uint32_t n;            // matrix dimension (square matrix side length)
    uint32_t lda;          // leading dimension of A[i] (input matrix)
    uint32_t ldaInv;       // leading dimension of Ainv[i] (output matrix)
    uint32_t usedCoreNum;  // number of AI cores actually used
    uint32_t batchPerCore; // batches per core (non-tail cores)
    uint32_t batchTail;    // batches for the last (tail) core
    uint32_t batchSize;    // total number of batches
};
