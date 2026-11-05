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
 * \file sgels_batched_tiling_data.h
 * \brief Tiling data for batched least-squares solver (aclblasSgelsBatched).
 *        SIMT model: single kernel launch, all batches processed in-kernel loop.
 */

#pragma once

#include <cstdint>

// Batch distribution across AI Cores (R4 compliant: no arrays):
//   startBatchId = blockIdx * batchPerCore
//   calBatchNum  = (blockIdx == usedCoreNum - 1) ? batchTail : batchPerCore
struct SgelsBatchedTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t nrhs;
    int32_t lda;
    int32_t ldc;
    uint32_t batchSize;
    uint32_t batchPerCore;
    uint32_t batchTail;
    uint32_t usedCoreNum;
    uint32_t numThreads;
    uint32_t minMN;
    uint32_t maxMN;
    uint32_t trans;
    int32_t origLda;
};
