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
 * \file sgemv_tiling_data.h
 * \brief Tiling data structure shared between host and kernel for sgemv (SIMT).
 */

#pragma once

#include <cstdint>

struct SgemvTilingData {
    uint32_t numThreads;   // threads per block
    uint32_t rowsPerBlock; // ceil(outDim / numBlocks)
    uint32_t m;            // matrix A rows
    uint32_t n;            // matrix A columns
    uint32_t lda;          // A leading dimension
    uint32_t trans;        // 0 = N, 1 = T/C
    float alpha;           // scalar alpha
    float beta;            // scalar beta
    int64_t incx;          // x stride
    int64_t incy;          // y stride
};
