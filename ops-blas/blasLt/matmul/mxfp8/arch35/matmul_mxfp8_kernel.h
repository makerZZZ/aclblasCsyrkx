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
 * \file matmul_mxfp8_kernel.h
 * \brief Host-side launch entry for the MXFP8 (BLAZE) device kernel (arch35).
 */

#pragma once

#include <cstdint>

struct QuantMatmulTilingData;

// MXFP8 (BLAZE) launch entry.
void matmul_mxfp8_do(
    uint8_t* dA, uint8_t* dB, uint8_t* dScaleA, uint8_t* dScaleB, uint8_t* dC, const QuantMatmulTilingData& tiling,
    bool transA, bool transB, void* stream);
