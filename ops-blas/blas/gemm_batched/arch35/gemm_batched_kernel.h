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
 * \file gemm_batched_kernel.h
 * \brief 供 Cgetri 分块更新调用的 GEMM 发射接口。
 */

#pragma once

#include <cstdint>

#include "gemm_batched_tiling_data.h"

void cgetri_strided_gemm_kernel_do(
    uint32_t numBlocks, void* stream, const uint8_t* a, const uint8_t* b, uint8_t* c, uint8_t* infoArray,
    uint64_t strideA, uint64_t strideB, uint64_t strideC, const GemmBatchedGemmTilingData& tilingData);
