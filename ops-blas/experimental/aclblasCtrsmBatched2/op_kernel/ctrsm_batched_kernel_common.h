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

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ctrsm_batched_tiling_data.h"
#include "../../trsm_mix_common.h"

using namespace matmul;

namespace {
__aicore__ inline bool NeedTransA(const __gm__ CtrsmBatchedTilingData* t)
{
    bool right = (t->side == SIDE_RIGHT);
    bool isT = (t->transa == TRANS_T);
    bool isC = (t->transa == TRANS_C);
    return ((isT || isC) && !right) || (t->transa == TRANS_N && right);
}

__aicore__ inline bool IsConjTransA(const __gm__ CtrsmBatchedTilingData* t)
{
    return (t->transa == TRANS_C);
}
}  // namespace
