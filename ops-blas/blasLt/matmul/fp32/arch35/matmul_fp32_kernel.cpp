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
 * \file ltmatmul_fp32_kernel.cpp
 * \brief FP32 matmul kernel with host-provided tiling for ltmatmul.
 */

#include <cstdint>

#include "matmul_fp32_kernel.h"
#include "matmul_fp32_tiling_data.h"
#include "matmul_fp32_kernel_utils.h"

template <typename T>
__aicore__ inline void MatmulCopyInA1(
    const AscendC::GlobalTensor<T>& aGlobal, const AscendC::LocalTensor<T>& al1Local, uint64_t curML1, uint64_t curKL1,
    uint64_t k)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = curML1;
    nd2nzParams.dValue = curKL1;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = k;
    nd2nzParams.dstNzC0Stride = CeilAlign(curML1, AscendC::BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    AscendC::DataCopy(al1Local, aGlobal, nd2nzParams);
}

template <typename T>
__aicore__ inline void MatmulCopyInB1(
    const AscendC::GlobalTensor<T>& bGlobal, const AscendC::LocalTensor<T>& bl1Local, uint64_t curNL1, uint64_t curKL1,
    uint64_t n)
{
    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = curKL1;
    nd2nzParams.dValue = curNL1;
    nd2nzParams.srcNdMatrixStride = 1;
    nd2nzParams.srcDValue = n;
    nd2nzParams.dstNzC0Stride = CeilAlign(curKL1, AscendC::BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 1;
    AscendC::DataCopy(bl1Local, bGlobal, nd2nzParams);
}

template <typename T>
__aicore__ inline void MatmulCopyInA2(
    const AscendC::LocalTensor<T>& al0Local, const AscendC::LocalTensor<T>& al1Local, uint64_t curML1, uint64_t curKL1,
    uint64_t mL0, uint64_t kL0, bool transA)
{
    AscendC::LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;
    loadDataParams.mStep = CeilDiv(mL0, AscendC::BLOCK_CUBE);
    loadDataParams.kStep = CeilDiv(kL0, BlasLtGetC0Size<T>());
    loadDataParams.srcStride = CeilDiv(curML1, AscendC::BLOCK_CUBE);
    loadDataParams.dstStride = loadDataParams.mStep;
    loadDataParams.ifTranspose = transA;
    AscendC::LoadData<T>(al0Local, al1Local, loadDataParams);
}

template <typename T>
__aicore__ inline void MatmulCopyInB2(
    const AscendC::LocalTensor<T>& bl0Local, const AscendC::LocalTensor<T>& bl1Local, uint64_t curNL1, uint64_t curKL1,
    uint64_t nL0, uint64_t kL0, bool transB)
{
    AscendC::LoadData2DParamsV2 loadDataParams;
    loadDataParams.mStartPosition = 0;
    loadDataParams.kStartPosition = 0;
    loadDataParams.mStep = CeilDiv(kL0, AscendC::BLOCK_CUBE);
    loadDataParams.kStep = CeilDiv(nL0, AscendC::BLOCK_CUBE) * TWO_ALIGN;
    loadDataParams.dstStride = loadDataParams.kStep >> 1;
    loadDataParams.srcStride = CeilDiv(curKL1, AscendC::BLOCK_CUBE);
    loadDataParams.ifTranspose = !transB;
    AscendC::LoadData<T>(bl0Local, bl1Local, loadDataParams);
}

template <typename T>
__aicore__ inline void LtMmad(
    const AscendC::LocalTensor<float>& c1Local, const AscendC::LocalTensor<T>& al0Local,
    const AscendC::LocalTensor<T>& bl0Local, uint64_t m, uint64_t n, uint64_t k, bool isFirstLoop)
{
    AscendC::MmadParams mmadParams;
    mmadParams.m = m;
    mmadParams.n = n;
    mmadParams.k = k;
    mmadParams.cmatrixSource = false;
    mmadParams.cmatrixInitVal = isFirstLoop;
    mmadParams.unitFlag = 0;
    AscendC::Mmad(c1Local, al0Local, bl0Local, mmadParams);
}

template <typename T>
__aicore__ inline void MatmulCopyOut(
    const AscendC::GlobalTensor<T>& cGlobal, const AscendC::LocalTensor<float>& c1Local, uint64_t baseM, uint64_t baseN,
    uint64_t n)
{
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.nSize = baseN;
    intriParams.mSize = baseM;
    intriParams.dstStride = n;
    intriParams.srcStride = CeilAlign(baseM, AscendC::BLOCK_CUBE);
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.reluPre = 0;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    AscendC::DataCopy(cGlobal, c1Local, intriParams);
}

__global__ __aicore__ void MatmulFp32Kernel(GM_ADDR aGm, GM_ADDR bGm, GM_ADDR dGm, const MatmulFp32TilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    const uint32_t m = tiling.m;
    const uint32_t n = tiling.n;
    const uint32_t k = tiling.k;
    const uint64_t baseM = tiling.baseM;
    const uint64_t baseN = tiling.baseN;
    const uint64_t baseK = tiling.baseK;
    const uint64_t kL1 = tiling.kL1;
    const uint32_t lda = tiling.lda > 0 ? tiling.lda : k;
    const uint32_t ldb = tiling.ldb > 0 ? tiling.ldb : n;
    const bool transA = tiling.transA != 0;
    const bool transB = tiling.transB != 0;

    AscendC::GlobalTensor<float> aGlobal;
    AscendC::GlobalTensor<float> bGlobal;
    AscendC::GlobalTensor<float> dGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aGm));
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bGm));
    dGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dGm));

    AscendC::LocalTensor<float> al0Local{AscendC::TPosition::A2, 0, L0A_SIZE};
    AscendC::LocalTensor<float> bl0Local{AscendC::TPosition::B2, 0, L0B_SIZE};
    AscendC::LocalTensor<float> l0cLocal{AscendC::TPosition::CO1, 0, L0C_SIZE};
    AscendC::LocalTensor<float> l1Local{AscendC::TPosition::A1, 0, L1_SIZE};

    constexpr static uint64_t HALF_L0_SIZE = L0A_SIZE / DOUBLE_BUFFER_COUNT / sizeof(float);

    const uint64_t mTileNum = CeilDiv(m, baseM);
    const uint64_t nTileNum = CeilDiv(n, baseN);
    const uint64_t tileNum = mTileNum * nTileNum;
    const uint64_t tailBaseM = m - (mTileNum - 1) * baseM;
    const uint64_t tailBaseN = n - (nTileNum - 1) * baseN;
    const uint64_t kL1TileNum = CeilDiv(k, kL1);
    const uint64_t tailKL1 = k - (kL1TileNum - 1) * kL1;

    uint64_t l1PingPong = 0;
    uint64_t l0PingPong = 0;

    const uint64_t curBlockIdx = AscendC::GetBlockIdx();
    const uint64_t blockNum = AscendC::GetBlockNum();

    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);

    for (uint64_t tileIdx = curBlockIdx; tileIdx < tileNum; tileIdx += blockNum) {
        const uint64_t mTileIdx = tileIdx / nTileNum;
        const uint64_t nTileIdx = tileIdx % nTileNum;
        const uint64_t mL1 = mTileIdx == (mTileNum - 1) ? tailBaseM : baseM;
        const uint64_t nL1 = nTileIdx == (nTileNum - 1) ? tailBaseN : baseN;
        const uint64_t mL0 = mL1;
        const uint64_t nL0 = nL1;
        const uint64_t mOffset = mTileIdx * baseM;
        const uint64_t nOffset = nTileIdx * baseN;
        uint64_t offsetA = transA ? mOffset : (mOffset * lda);
        uint64_t offsetB = transB ? (nOffset * ldb) : nOffset;
        const uint64_t offsetD = mOffset * n + nOffset;

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);

        for (uint64_t iter0 = 0; iter0 < kL1TileNum; ++iter0) {
            const uint64_t curKL1 = (iter0 + 1 == kL1TileNum) ? tailKL1 : kL1;
            const uint64_t l1BufId = l1PingPong & 0x1;

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            const uint64_t offsetAL1 = baseM * kL1 * l1BufId;
            if (transA) {
                MatmulCopyInA1<float>(aGlobal[offsetA], l1Local[offsetAL1], curKL1, mL1, lda);
                offsetA += curKL1 * lda;
            } else {
                MatmulCopyInA1<float>(aGlobal[offsetA], l1Local[offsetAL1], mL1, curKL1, lda);
                offsetA += curKL1;
            }
            const uint64_t offsetBL1 = baseM * kL1 * 2 + baseN * kL1 * l1BufId;
            if (transB) {
                MatmulCopyInB1<float>(bGlobal[offsetB], l1Local[offsetBL1], curKL1, nL1, ldb);
                offsetB += curKL1;
            } else {
                MatmulCopyInB1<float>(bGlobal[offsetB], l1Local[offsetBL1], nL1, curKL1, ldb);
                offsetB += curKL1 * ldb;
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BufId);

            const uint64_t kL0TileNum = CeilDiv(curKL1, baseK);
            const uint64_t tailKL0 = curKL1 - (kL0TileNum - 1) * baseK;
            uint64_t offsetAL0 = offsetAL1;
            uint64_t offsetBL0 = offsetBL1;
            for (uint64_t iter1 = 0; iter1 < kL0TileNum; ++iter1) {
                const uint64_t curKL0 = (iter1 + 1 == kL0TileNum) ? tailKL0 : baseK;
                const uint64_t l0BufId = l0PingPong & 0x1;

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                const uint64_t l0Offset = HALF_L0_SIZE * l0BufId;
                MatmulCopyInA2<float>(al0Local[l0Offset], l1Local[offsetAL0], mL1, curKL1, mL0, curKL0, transA);
                MatmulCopyInB2<float>(bl0Local[l0Offset], l1Local[offsetBL0], nL1, curKL1, nL0, curKL0, transB);
                offsetAL0 += CeilAlign(mL1, AscendC::BLOCK_CUBE) * baseK;
                offsetBL0 += baseK * BlasLtGetC0Size<float>();

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0BufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0BufId);

                const bool isFirstLoop = iter0 == 0 && iter1 == 0;
                LtMmad<float>(l0cLocal, al0Local[l0Offset], bl0Local[l0Offset], mL0, nL0, curKL0, isFirstLoop);

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BufId);
                l0PingPong++;
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BufId);
            l1PingPong++;
        }

        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ZERO_FLAG);
        MatmulCopyOut<float>(dGlobal[offsetD], l0cLocal, mL0, nL0, n);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ZERO_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(FIRST_FLAG);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ZERO_FLAG);
}

void matmul_fp32_do(
    uint8_t* a, uint8_t* b, uint8_t* dRaw, const MatmulFp32TilingData& tiling, uint32_t numBlocks, void* stream)
{
    MatmulFp32TilingData tilingCopy = tiling;
    tilingCopy.usedCoreNum = numBlocks;
    MatmulFp32Kernel<<<tilingCopy.usedCoreNum, nullptr, stream>>>(a, b, dRaw, tilingCopy);
}
