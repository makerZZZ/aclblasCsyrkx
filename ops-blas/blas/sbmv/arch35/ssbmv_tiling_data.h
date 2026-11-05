/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ssbmv_tiling_data.h
 * \brief
 */

#pragma once

#include <cstdint>

struct SsbmvTilingData {
    uint32_t numThreads;    // threads per block
    uint32_t numBlocks;     // blocks launched
    uint32_t rowsPerBlock;  // ceil(n / numBlocks), pre-computed by host
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t uplo;          // ACLBLAS_UPPER or ACLBLAS_LOWER
    float alpha;
    float beta;
    int64_t incx;
    int64_t incy;
};
