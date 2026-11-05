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
 * \file matmul_fp32_host.cpp
 * \brief Host-side FP32 tiling helper for ascend950 (arch35).
 */

#include <algorithm>
#include <cstdint>

#include "matmul_fp32_tiling_data.h"
#include "matmul_fp32_host.h"

void matmul_fp32_get_tiling(
    uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, uint32_t lda, uint32_t ldb, uint32_t numBlocks,
    MatmulFp32TilingData& tilingData)
{
    tilingData = {};
    tilingData.m = static_cast<uint32_t>(m);
    tilingData.n = static_cast<uint32_t>(n);
    tilingData.k = static_cast<uint32_t>(k);

    tilingData.baseM = static_cast<uint32_t>(std::min<uint64_t>(m, 128ULL));
    tilingData.baseN = static_cast<uint32_t>(std::min<uint64_t>(n, 256ULL));
    tilingData.baseK = static_cast<uint32_t>(std::min<uint64_t>(k, 128ULL / sizeof(float)));
    tilingData.kL1 = static_cast<uint32_t>(std::min<uint64_t>(k, 512ULL / sizeof(float)));

    tilingData.usedCoreNum = numBlocks;
    tilingData.lda = lda;
    tilingData.ldb = ldb;
    tilingData.transA = transA ? 1U : 0U;
    tilingData.transB = transB ? 1U : 0U;
}
