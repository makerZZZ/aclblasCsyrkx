/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CTRSM_BATCHED_KERNEL_H
#define CTRSM_BATCHED_KERNEL_H

#define ASCENDC_CUBE_ONLY
#include "ctrsm_batched_kernel_aiv.h"
#include "ctrsm_batched_kernel_aic.h"

extern "C" __global__ __aicore__ void ctrsm_batched_mix12_kernel(
    GM_ADDR aArrayGm, GM_ADDR bArrayGm, GM_ADDR trsmTilingGm,
    GM_ADDR cubeTilingGm, GM_ADDR gemmWsGm, GM_ADDR sysWorkspace)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    auto trsmTiling = (__gm__ CtrsmBatchedTilingData*)trsmTilingGm;
    TCubeTiling cubeTiling;
    CopyCubeTiling(&cubeTiling, cubeTilingGm);
    if ASCEND_IS_AIC {
        CtrsmMixAic aic;
        aic.Init(aArrayGm, bArrayGm, gemmWsGm, trsmTiling, cubeTiling);
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), aic.mm, &aic.cubeTiling);
        aic.Process12();
        aic.mm.End();
    }
    if ASCEND_IS_AIV {
        // 按 side/uplo/transa 计算计算路径，分派到对应的模板路径类型
        bool right = (trsmTiling->side == SIDE_RIGHT);
        bool needTA = NeedTransA(trsmTiling);
        int32_t effUplo = needTA ? (trsmTiling->uplo == UPLO_UPPER ? UPLO_LOWER : UPLO_UPPER)
                                 : trsmTiling->uplo;
        bool forward = (effUplo == UPLO_LOWER);
        if (forward && !right) {
            CtrsmLowerLeft aiv;
            aiv.Init(aArrayGm, bArrayGm, gemmWsGm, trsmTiling, &pipe);
            aiv.Process12();
        } else if (forward && right) {
            CtrsmLowerRight aiv;
            aiv.Init(aArrayGm, bArrayGm, gemmWsGm, trsmTiling, &pipe);
            aiv.Process12();
        } else if (!forward && !right) {
            CtrsmUpperLeft aiv;
            aiv.Init(aArrayGm, bArrayGm, gemmWsGm, trsmTiling, &pipe);
            aiv.Process12();
        } else {
            CtrsmUpperRight aiv;
            aiv.Init(aArrayGm, bArrayGm, gemmWsGm, trsmTiling, &pipe);
            aiv.Process12();
        }
    }
}

void ctrsm_batched_mix12_kernel_do(GM_ADDR aArrayGm, GM_ADDR bArrayGm, GM_ADDR trsmTilingGm,
                                    GM_ADDR cubeTilingGm, GM_ADDR gemmWsGm, GM_ADDR sysWorkspace,
                                    uint32_t numBlocks, void* stream)
{
    ctrsm_batched_mix12_kernel<<<numBlocks, nullptr, stream>>>(
        aArrayGm, bArrayGm, trsmTilingGm, cubeTilingGm, gemmWsGm, sysWorkspace);
}

#endif
