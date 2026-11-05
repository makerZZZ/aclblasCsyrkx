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
 * \file matmul_mxfp8_kernel.cpp
 * \brief MXFP8 matmul kernel.
 */

#if ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0

#include <cstdint>

#include "aux_get_c0_size.h"
#include "integral_constant.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_without_batch.h"
#include "matmul_mxfp8_kernel.h"
#include "quant_matmul_tiling_data.h"

template <bool TransA, bool TransB>
__global__ __aicore__ __cube__ void QuantMatmulMxfp8Kernel(
    GM_ADDR dA, GM_ADDR dB, GM_ADDR dScaleA, GM_ADDR dScaleB, GM_ADDR dC,
    const QuantMatmulTilingData quantMatmulTilingData)
{
    using namespace AscendC::Std;
    using namespace AscendC::Te;
    using namespace Blaze::Gemm;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using AType = fp8_e4m3fn_t;
    using BType = fp8_e4m3fn_t;
    using CType = bfloat16_t;
    using BiasType = float;
    using DispatchPolicy = MatmulWithScaleMx<0UL, false, KernelMmadWithScaleMxWithoutBatch>;
    using LayoutA = conditional_t<TransA, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutB = conditional_t<TransB, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0UL, LayoutA, LayoutB, AType>;
    using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename KernelImpl::Params;
    using BlockMmadParams = typename BlockMmad::Params;
    using L1Params = typename BlockMmad::L1Params;
    using BlockSchedulerParams = typename BlockScheduler::Params;
    using QBMMTiling = typename KernelImpl::QBMMTiling;

    ProblemShape problemShape{
        static_cast<int64_t>(quantMatmulTilingData.m), static_cast<int64_t>(quantMatmulTilingData.n),
        static_cast<int64_t>(quantMatmulTilingData.k), 1L};
    BlockMmadParams mmadParams{dA, dB, dC, nullptr, dScaleA, dScaleB};
    L1Params l1Params{
        static_cast<uint64_t>(quantMatmulTilingData.stepK) * quantMatmulTilingData.baseK,
        quantMatmulTilingData.scaleKL1, static_cast<uint64_t>(quantMatmulTilingData.nBufferNum)};
    BlockSchedulerParams schedulerParams{
        static_cast<int64_t>(quantMatmulTilingData.baseM),
        static_cast<int64_t>(quantMatmulTilingData.baseN),
        static_cast<int64_t>(quantMatmulTilingData.mTailTile),
        static_cast<int64_t>(quantMatmulTilingData.nTailTile),
        static_cast<int64_t>(quantMatmulTilingData.mBaseTailSplitCnt),
        static_cast<int64_t>(quantMatmulTilingData.nBaseTailSplitCnt),
        static_cast<int64_t>(quantMatmulTilingData.mTailMain),
        static_cast<int64_t>(quantMatmulTilingData.nTailMain)};
    QBMMTiling qbmmParams{
        quantMatmulTilingData.baseM, quantMatmulTilingData.baseN, quantMatmulTilingData.baseK, 0U,
        quantMatmulTilingData.dbL0c};
    Params params{problemShape, mmadParams, l1Params, schedulerParams, qbmmParams};
    KernelImpl kernel;
    kernel(params);
}

void matmul_mxfp8_do(
    uint8_t* dA, uint8_t* dB, uint8_t* dScaleA, uint8_t* dScaleB, uint8_t* dC, const QuantMatmulTilingData& tiling,
    bool transA, bool transB, void* stream)
{
    QuantMatmulTilingData tilingCopy = tiling;

    if (transA && transB) {
        QuantMatmulMxfp8Kernel<true, true>
            <<<tilingCopy.usedCoreNum, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dC, tilingCopy);
    } else if (transA && !transB) {
        QuantMatmulMxfp8Kernel<true, false>
            <<<tilingCopy.usedCoreNum, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dC, tilingCopy);
    } else if (!transA && transB) {
        QuantMatmulMxfp8Kernel<false, true>
            <<<tilingCopy.usedCoreNum, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dC, tilingCopy);
    } else {
        QuantMatmulMxfp8Kernel<false, false>
            <<<tilingCopy.usedCoreNum, nullptr, stream>>>(dA, dB, dScaleA, dScaleB, dC, tilingCopy);
    }
}

#endif // ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR > 0
