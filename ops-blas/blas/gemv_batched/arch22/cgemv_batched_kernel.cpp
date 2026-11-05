/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace CgemvBatched {

#include "cgemv_batched_kernel_impl.h"

}  // namespace CgemvBatched

extern "C" __global__ __aicore__ __vector__ void cgemv_batched(
    GM_ADDR A, GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);
    uint32_t dtype = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf));
    uint32_t transVal = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + sizeof(uint32_t)));

    if (dtype > 0) {
        if (transVal > 0) {
            CgemvBatched::CgemvBatched::CgemvBatchedAIV<float, true> op;
            op.Init(A, x, y, mask, transVal, workSpace, tilingGm);
            op.Process();
        } else {
            CgemvBatched::CgemvBatched::CgemvBatchedAIV<float, false> op;
            op.Init(A, x, y, mask, transVal, workSpace, tilingGm);
            op.Process();
        }
    } else {
        if (transVal > 0) {
            CgemvBatched::CgemvBatched::CgemvBatchedAIV<half, true> op;
            op.Init(A, x, y, mask, transVal, workSpace, tilingGm);
            op.Process();
        } else {
            CgemvBatched::CgemvBatched::CgemvBatchedAIV<half, false> op;
            op.Init(A, x, y, mask, transVal, workSpace, tilingGm);
            op.Process();
        }
    }
}

void cgemv_batched_kernel_do(GM_ADDR A, GM_ADDR x, GM_ADDR mask, GM_ADDR y,
                             GM_ADDR workSpace, GM_ADDR tilingGm,
                             uint32_t numBlocks, void *stream)
{
    cgemv_batched<<<numBlocks, nullptr, stream>>>(A, x, mask, y, workSpace, tilingGm);
}
