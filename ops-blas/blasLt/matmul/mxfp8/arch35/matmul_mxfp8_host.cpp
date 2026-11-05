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
 * \file matmul_mxfp8_host.cpp
 * \brief Host-side MXFP8 tiling helper.
 */

#include "version/asc_devkit_version.h"

#if ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0

#include <cstdint>

#include "matmul_mxfp8_host.h"
#include "quant_matmul_mx_tiling_swat_host.h"
#include "quant_matmul_tiling_data.h"

void matmul_mxfp8_get_tiling(
    uint64_t m, uint64_t n, uint64_t k, bool transA, bool transB, uint32_t numBlocks,
    QuantMatmulTilingData& tilingData)
{
    const uint64_t aicNum = numBlocks > 0U ? static_cast<uint64_t>(numBlocks) :
                                             quant_matmul_mx_tiling::DEFAULT_AIC_NUM;
    quant_matmul_mx_tiling::quant_matmul_mxfp8_swat_get_tiling(m, n, k, transA, transB, aicNum, tilingData);
}

#endif // ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0
