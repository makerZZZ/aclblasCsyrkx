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
#include "cgetri_batched_tiling_data.h"

void cgetri_batched_kernel_do(
    uint8_t* aarray, uint8_t* pivotArray, uint8_t* carray, uint8_t* infoArray, const CgetriBatchedTilingData& tiling,
    uint32_t numBlocks, void* stream);

void cgetri_blocked_init_kernel_do(
    uint8_t* aarray, uint8_t* pivotArray, uint8_t* infoArray, uint8_t* aReal, uint8_t* aImag, uint8_t* cReal,
    uint8_t* cImag, const CgetriBlockedInitTilingData& tiling, uint32_t numBlocks, void* stream);

void cgetri_blocked_solve_kernel_do(
    uint8_t* aReal, uint8_t* aImag, uint8_t* cReal, uint8_t* cImag, uint8_t* infoArray,
    const CgetriBlockedSolveTilingData& tiling, uint32_t numBlocks, void* stream);

void cgetri_blocked_combine_kernel_do(
    uint8_t* temp1, uint8_t* temp2, uint8_t* target, uint8_t* infoArray, const CgetriBlockedCombineTilingData& tiling,
    uint32_t numBlocks, void* stream);

void cgetri_blocked_finalize_kernel_do(
    uint8_t* cReal, uint8_t* cImag, uint8_t* carray, uint8_t* infoArray, const CgetriBlockedFinalizeTilingData& tiling,
    uint32_t numBlocks, void* stream);
