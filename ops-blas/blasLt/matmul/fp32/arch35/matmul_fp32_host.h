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
 * \file matmul_fp32_host.h
 * \brief Host-side FP32 matmul tiling API (arch35).
 */

#pragma once

#include <cstdint>

struct MatmulFp32TilingData;

void matmul_fp32_get_tiling(
    uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, uint32_t lda, uint32_t ldb, uint32_t numBlocks,
    MatmulFp32TilingData& tilingData);
