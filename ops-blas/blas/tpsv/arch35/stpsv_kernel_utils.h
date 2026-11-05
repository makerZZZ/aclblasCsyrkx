/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file stpsv_kernel_utils.h
 * \brief Shared helpers for tpsv kernel files. Included only from kernel compilation units.
 */

#pragma once

#include <cstdint>

__aicore__ inline uint32_t TpsvPackedUpperIdx(uint32_t i, uint32_t j)
{
    return i + j * (j + 1) / 2;
}

__aicore__ inline uint32_t TpsvPackedLowerIdx(uint32_t i, uint32_t j, uint32_t n)
{
    return i + (2 * n - j - 1) * j / 2;
}

__simt_callee__ inline uint32_t TpsvPackedUpperIdxSimt(uint32_t i, uint32_t j)
{
    return i + j * (j + 1) / 2;
}

__simt_callee__ inline uint32_t TpsvPackedLowerIdxSimt(uint32_t i, uint32_t j, uint32_t n)
{
    return i + (2 * n - j - 1) * j / 2;
}
