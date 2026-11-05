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
 * \file matrix_transform_get_tiling.h
 * \brief Host-side tiling API declaration for aclblasLtMatrixTransform.
 */

#pragma once

#include <cstdint>

#include "matrix_transform_tiling_data.h"

void matrix_transform_get_tiling(
    uint32_t rows, uint32_t cols, uint32_t lda, uint32_t ldb, uint32_t ldc, uint8_t orderA, uint8_t orderB,
    uint8_t orderC, uint8_t opA, uint8_t opB, uint8_t hasB, uint32_t alphaBits, uint32_t betaBits,
    MatrixTransformTilingData& tilingData);
