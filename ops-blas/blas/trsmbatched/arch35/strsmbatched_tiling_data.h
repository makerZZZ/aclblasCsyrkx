/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "cann_ops_blas_common.h"

struct TrsmbatchedPanelTilingData {
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
    uint32_t m;
    uint32_t n;
    int32_t lda;
    int32_t ldb;
    uint32_t panelStart;
    uint32_t panelSize;
};

struct TrsmbatchedGemmTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint64_t aOffset;
    uint64_t bOffset;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t tileKChunk;
    uint32_t tempRowStride;
};

struct TrsmbatchedAxpyTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t ldb;
    uint32_t tempRowStride;
    uint64_t bOffset;
};

