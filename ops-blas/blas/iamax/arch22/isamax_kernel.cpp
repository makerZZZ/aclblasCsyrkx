/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ISAMAX_KERNEL_H
#define ISAMAX_KERNEL_H

#include "isamax_kernel_impl.h"

// Kernel entry point
__global__ __aicore__ __vector__ void isamax_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = workspace;
    Isamax::Isamax<float> op;
    op.Init(x, y, userWS, tiling);
    op.Process();
}

// Wrapper function for host to call
void isamax_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling,
                      uint32_t numBlocks, void *stream)
{
    isamax_kernel<<<numBlocks, nullptr, stream>>>(x, y, workspace, tiling);
}

#endif  // ISAMAX_KERNEL_H
