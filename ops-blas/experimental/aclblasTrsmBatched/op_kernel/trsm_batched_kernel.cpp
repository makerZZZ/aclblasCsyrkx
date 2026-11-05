/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trsm_batched_kernel.cpp
 * \brief Kernel entry point and launch wrapper for batched triangular solve (StrsmBatched).
 *        Uses KERNEL_TYPE_MIX_AIC_1_2 with dual-mode dispatch:
 *        - Independent mode (BC >= AIC cores): each AIV handles a different batch
 *        - Cooperative mode (BC < AIC cores): two AIVs split columns of the same batch
 */

#ifndef TRSM_BATCHED_KERNEL_H
#define TRSM_BATCHED_KERNEL_H

#define ASCENDC_CUBE_ONLY
#include "trsm_batched_aiv.h"
#include "trsm_batched_aic.h"

extern "C" __global__ __aicore__ void trsm_batched_mix12_kernel(
    GM_ADDR aArrayGm, GM_ADDR bArrayGm, GM_ADDR trsmTilingGm,
    GM_ADDR cubeTilingGm, GM_ADDR gemmWsGm, GM_ADDR sysWorkspace, GM_ADDR idGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;

    auto trsmTiling = (__gm__ TrsmBatchedTilingData*)trsmTilingGm;
    TCubeTiling cubeTiling;
    CopyCubeTiling(&cubeTiling, cubeTilingGm);

    if ASCEND_IS_AIC {
        TrsmMixAic aic;
        aic.Init(aArrayGm, bArrayGm, gemmWsGm, trsmTiling, cubeTiling, idGm);
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), aic.mm, &aic.cubeTiling);
        aic.Process12();
        aic.mm.End();
    }
    if ASCEND_IS_AIV {
        TrsmMixAiv aiv;
        aiv.Init(aArrayGm, bArrayGm, gemmWsGm, trsmTiling, &pipe, true);
        aiv.Process12();
    }
}

void trsm_batched_mix12_kernel_do(GM_ADDR aArrayGm, GM_ADDR bArrayGm, GM_ADDR trsmTilingGm,
                                  GM_ADDR cubeTilingGm, GM_ADDR gemmWsGm, GM_ADDR sysWorkspace,
                                  GM_ADDR idGm, uint32_t numBlocks, void* stream)
{
    trsm_batched_mix12_kernel<<<numBlocks, nullptr, stream>>>(
        aArrayGm, bArrayGm, trsmTilingGm, cubeTilingGm, gemmWsGm, sysWorkspace, idGm);
}

#endif  // TRSM_BATCHED_KERNEL_H
