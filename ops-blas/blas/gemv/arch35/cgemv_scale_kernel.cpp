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
 * \file cgemv_scale_kernel.cpp
 * \brief Scale-only CGEMV path; A and x are never read.
 */

#include "cgemv_kernel_common.h"

// ==========================================================================
//  Scale path — alpha == (0,0) and beta != (1,0): y = beta * y
//  A and x are never touched. When betaIsZero != 0, writes zeros without reading y.
// ==========================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgemvScale(
    uint32_t outDim, float betaR, float betaI, uint32_t betaIsZero, int64_t incy, __gm__ float* yGm)
{
    int64_t len64 = static_cast<int64_t>(outDim);
    bool betaIsOne = (betaR == 1.0f) && (betaI == 0.0f);

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < outDim; i += gridDim.x * blockDim.x) {
        int64_t yIdx = CgemvStridedIdx(static_cast<int64_t>(i), len64, incy) * 2;
        if (betaIsZero != 0) {
            yGm[yIdx] = 0.0f;
            yGm[yIdx + 1] = 0.0f;
        } else if (!betaIsOne) {
            float yR = yGm[yIdx];
            float yI = yGm[yIdx + 1];
            float outR = CgemvRoundedMul(betaR, yR);
            outR -= CgemvRoundedMul(betaI, yI);

            float outI = CgemvRoundedMul(betaR, yI);
            outI += CgemvRoundedMul(betaI, yR);

            yGm[yIdx] = outR;
            yGm[yIdx + 1] = outI;
        }
    }
}

__global__ __aicore__ void cgemv_scale_kernel(GM_ADDR y, CgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t outDim = (tiling.trans == 0) ? tiling.m : tiling.n;
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    asc_vf_call<CgemvScale>(
        dim3{tiling.numThreads, 1, 1}, outDim, tiling.betaR, tiling.betaI, tiling.betaIsZero, tiling.incy, yGm);
}

void CgemvLaunchScale(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    cgemv_scale_kernel<<<(numBlocks), nullptr, stream>>>(y, tiling);
}
