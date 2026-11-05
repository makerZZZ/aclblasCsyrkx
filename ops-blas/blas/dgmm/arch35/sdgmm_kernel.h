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
 * \file sdgmm_kernel.h
 * \brief Declaration of the kernel launcher for aclblasSdgmm (arch35).
 *        Uses AscendC APIs: GlobalTensor / TBuf / DataCopyPad / Mul / Muls.
 *        Shared by host.cpp and kernel.cpp.
 */

#pragma once

#include <cstdint>
#include "sdgmm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

/*!
 * \brief Kernel launcher: asynchronously launches the kernel.
 *
 * The kernel handles negative incx internally by computing non-negative
 * indices (same formula as the CPU golden), so the Host passes the original
 * x pointer without pre-offsetting. Tiling is passed by value through the
 * <<<>>> launch (small H2D copy). The launch is asynchronous: the caller
 * must not call aclrtSynchronizeStream.
 *
 * \param x         GM address of vector x (original pointer, not pre-offset)
 * \param A         GM address of matrix A (column-major, lda x n)
 * \param C         GM address of output matrix C (column-major, ldc x n)
 * \param tiling    host-computed tiling data (copied to device by value)
 * \param numBlocks block count for the <<<>>> launch
 * \param stream    aclrtStream handle
 */
void sdgmm_kernel_do(const GM_ADDR x, const GM_ADDR A, GM_ADDR C,
                     const SdgmmTilingData& tiling,
                     uint32_t numBlocks, void* stream);
