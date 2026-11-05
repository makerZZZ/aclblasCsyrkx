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

struct GemmSb22Tiling {
    int m, n, k, lda, ldb, ldc;
    int transA, transB, batches, tempLd;
    int64_t strideA, strideB, strideC;
    float alpha, beta;
    uint32_t tinyBatchGroup = 8;
};

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

void GemmSb22Cube(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmSb22Tiling t);
void GemmSb22Tiny(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmSb22Tiling t);
void GemmSb22BatchCube(uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GemmSb22Tiling t);
void GemmSb22Vector(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR temp, GemmSb22Tiling t, bool combine);
