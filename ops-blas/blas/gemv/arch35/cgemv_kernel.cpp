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
 * \file cgemv_kernel.cpp
 * \brief CGEMV path dispatch; numerical kernels live in separate translation units.
 */

#include "cgemv_kernel_common.h"

void cgemv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    // The ordered N512 kernel needs no workspace, including when slab setup falls back.
    bool useOrderedN512 = tiling.trans == 0 && tiling.m == CGEMV_N512_DIM && tiling.n == CGEMV_N512_DIM &&
                          tiling.lda == CGEMV_N512_DIM && tiling.alphaR == 1.0F && tiling.alphaI == 0.0F &&
                          tiling.betaIsZero != 0 && tiling.incx == 1 && tiling.incy == 1;
    if (useOrderedN512) {
        CgemvLaunchOrderedN512(a, x, y, workSpace, tiling, numBlocks, stream);
    } else if (tiling.useSlab != 0 && tiling.trans == 0 && tiling.alphaIsZero == 0) {
        CgemvLaunchNSlab(a, x, y, workSpace, tiling, numBlocks, stream);
    } else if (tiling.useSlab != 0 && tiling.trans != 0 && tiling.alphaIsZero == 0) {
        CgemvLaunchTSlab(a, x, y, workSpace, tiling, numBlocks, stream);
    } else if (tiling.alphaIsZero != 0) {
        CgemvLaunchScale(a, x, y, workSpace, tiling, numBlocks, stream);
    } else {
        CgemvLaunchOrdered(a, x, y, workSpace, tiling, numBlocks, stream);
    }
}
