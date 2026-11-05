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
 * \file ssymv_kernel.cpp
 * \brief SSYMV Kernel for ascend950 (DAV_3510)
 */

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "ssymv_tiling_data.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"

template<bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SsymvSimtCompute(
    uint32_t n, uint32_t lda, float alpha, float beta, int64_t incx, int64_t incy,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    int64_t n64 = static_cast<int64_t>(n);
    int64_t lda64 = static_cast<int64_t>(lda);

    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
         row < n; row += gridDim.x * blockDim.x) {

        int64_t row64 = static_cast<int64_t>(row);
        float acc = 0.0f;

        for (uint32_t jCol = 0; jCol < n; ++jCol) {
            int64_t col64 = static_cast<int64_t>(jCol);

            int64_t aIdx;
            if constexpr (UPLO_IS_UPPER) {
                aIdx = (row <= jCol) ? (row64 + col64 * lda64) : (col64 + row64 * lda64);
            } else {
                aIdx = (row >= jCol) ? (row64 + col64 * lda64) : (col64 + row64 * lda64);
            }
            float aVal = aGm[aIdx];

            int64_t xIdx = (incx >= 0) ? (col64 * incx) : ((n64 - 1 - col64) * (-incx));
            float xVal = xGm[xIdx];

            acc += aVal * xVal;
        }

        int64_t yIdx = (incy >= 0) ? (row64 * incy) : ((n64 - 1 - row64) * (-incy));
        float yVal = yGm[yIdx];
        yGm[yIdx] = alpha * acc + beta * yVal;
    }
}

__global__ __aicore__ void ssymv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace,
                                        const SsymvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<SsymvSimtCompute<true>>(
            dim3{tiling.nthreads, 1, 1}, tiling.n, tiling.lda,
            tiling.alpha, tiling.beta, tiling.incx, tiling.incy,
            reinterpret_cast<__gm__ const float*>(a),
            reinterpret_cast<__gm__ const float*>(x),
            reinterpret_cast<__gm__ float*>(y));
    } else {
        asc_vf_call<SsymvSimtCompute<false>>(
            dim3{tiling.nthreads, 1, 1}, tiling.n, tiling.lda,
            tiling.alpha, tiling.beta, tiling.incx, tiling.incy,
            reinterpret_cast<__gm__ const float*>(a),
            reinterpret_cast<__gm__ const float*>(x),
            reinterpret_cast<__gm__ float*>(y));
    }
}

void ssymv_kernel_do(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace,
    uint32_t numBlocks, const SsymvTilingData& tiling, void* stream)
{
    ssymv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
}
