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
 * \file matrix_transform_kernel.h
 * \brief Host-side launch declaration for the aclblasLtMatrixTransform device kernel.
 */

#pragma once

#include <cstdint>

#include "matrix_transform_tiling_data.h"

// dtype codes: 0=FP32, 1=FP16, 2=BF16, 3=INT8, 4=INT32, 5=FP8_E4M3FN, 6=FP8_E5M2, 7=FP4_E2M1.
// idxGm:  GM workspace holding the pivot complex order column-major in-tile element-offset table
//         (nullptr for linear-only cases that never use the group path).
// idxAGm / idxBGm: per-operand de-layout offset tables (orderA / orderB column-major tables),
//         used by the complex-input + op=T de-layout pass; nullptr when that operand is not staged.
// ndAGm / ndBGm:   column-major ND GM workspaces receiving the de-layouted complex input before
//         the op=T main pass reads them as COL linear inputs; nullptr when not staged.
void matrix_transform_kernel_do(
    uint8_t* aGm, uint8_t* bGm, uint8_t* cGm, uint8_t* idxGm, uint8_t* idxAGm, uint8_t* idxBGm, uint8_t* ndAGm,
    uint8_t* ndBGm, uint8_t dtypeA, uint8_t dtypeB, uint8_t dtypeC, const MatrixTransformTilingData& tiling,
    uint32_t numBlocks, void* stream);
