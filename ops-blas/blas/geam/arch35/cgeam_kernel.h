/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file cgeam_kernel.h
 * @brief Kernel launcher declaration for Cgeam operator (complex)
 */

#pragma once

#include <cstdint>
#include "acl/acl_base_rt.h"
#include "cgeam_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

/**
 * @brief Launch Cgeam kernel asynchronously
 *
 * @param A GM address of matrix A (complex)
 * @param B GM address of matrix B (complex)
 * @param C GM address of matrix C (complex, output)
 * @param tiling Host-computed tiling data (copied to device by value)
 * @param numBlocks Block count for the <<<>>> launch
 * @param stream aclrtStream handle
 */
void cgeam_kernel_do(
    GM_ADDR A, GM_ADDR B, GM_ADDR C, const CgeamTilingData& tiling, uint32_t numBlocks, aclrtStream stream);
