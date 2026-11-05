/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cgeru_tiling_data.h"
#include "common/helper/kernel_constant.h"

namespace {

constexpr uint32_t CGERU_FLOAT_EXPONENT_SHIFT = 23U;
constexpr uint32_t CGERU_FLOAT_EXPONENT_MASK = 0xffU;
constexpr uint32_t CGERU_FLOAT_PRODUCT_OVERFLOW_EXPONENT_SUM = 381U;

struct CgeruTile {
    uint32_t rowStart;
    uint32_t rowCount;
    uint32_t colStart;
    uint32_t colCount;
};

__aicore__ __inline__ CgeruTile CgeruGetTile(const CgeruTilingData& tiling)
{
    const uint32_t blockIndex = AscendC::GetBlockIdx();
    const uint32_t rowBlock = blockIndex % tiling.rowBlocks;
    const uint32_t colBlock = blockIndex / tiling.rowBlocks;
    const uint32_t baseRows = tiling.m / tiling.rowBlocks;
    const uint32_t extraRows = tiling.m - baseRows * tiling.rowBlocks;
    const uint32_t baseCols = tiling.n / tiling.colBlocks;
    const uint32_t extraCols = tiling.n - baseCols * tiling.colBlocks;
    return CgeruTile{
        rowBlock * baseRows + ((rowBlock < extraRows) ? rowBlock : extraRows),
        baseRows + ((rowBlock < extraRows) ? 1U : 0U),
        colBlock * baseCols + ((colBlock < extraCols) ? colBlock : extraCols),
        baseCols + ((colBlock < extraCols) ? 1U : 0U)};
}

struct CgeruRegBufferLayout {
    uint32_t xBufferBytes;
    uint32_t aBufferBytes;
    uint32_t pBufferBytes;
    uint32_t tileCapacity;
};

struct CgeruRegEvents {
    event_t mte2ToV;
    event_t vToMte3;
    event_t mte3ToMte2;
    event_t scalarToV;
};

__aicore__ __inline__ CgeruRegEvents CgeruAcquireRegEvents()
{
    return CgeruRegEvents{
        static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V))};
}

__aicore__ __inline__ CgeruRegBufferLayout CgeruPlanRegBuffers(uint32_t rowCount, uint32_t colCount)
{
    if (rowCount == 0U || colCount == 0U) {
        return {};
    }
    constexpr uint32_t complexBytes = CGERU_COMPLEX_COMPONENTS * sizeof(float);
    const uint64_t rowBytes = static_cast<uint64_t>(rowCount) * complexBytes;
    constexpr uint32_t complexVectorBytes = 2U * AscendC::VECTOR_REG_WIDTH;
    const uint64_t alignedRowBytes = ((rowBytes + complexVectorBytes - 1U) / complexVectorBytes) * complexVectorBytes;
    const uint64_t xBufferBytes = alignedRowBytes + complexVectorBytes;
    const uint64_t reservedBytes = xBufferBytes + complexVectorBytes + 32U;
    if (reservedBytes >= CGERU_CONTIGUOUS_UB_BYTES) {
        return {};
    }
    const uint64_t availableColumnBytes = CGERU_CONTIGUOUS_UB_BYTES - reservedBytes;
    uint32_t tileCapacity = static_cast<uint32_t>(availableColumnBytes / (alignedRowBytes + complexBytes));
    tileCapacity = (tileCapacity < colCount) ? tileCapacity : colCount;
    tileCapacity = (tileCapacity < CGERU_MAX_DMA_BLOCKS) ? tileCapacity : CGERU_MAX_DMA_BLOCKS;
    if (alignedRowBytes != rowBytes && tileCapacity > CGERU_MAX_STRIDED_DMA_BLOCKS) {
        tileCapacity = CGERU_MAX_STRIDED_DMA_BLOCKS;
    }
    // Include every allocation's alignment padding before accepting the plan.
    while (tileCapacity != 0U) {
        const uint64_t aBufferBytes = alignedRowBytes * tileCapacity + complexVectorBytes;
        const uint64_t pBufferBytes = 32U + ((static_cast<uint64_t>(tileCapacity) * complexBytes + 31U) / 32U) * 32U;
        if (xBufferBytes + aBufferBytes + pBufferBytes <= CGERU_CONTIGUOUS_UB_BYTES) {
            return CgeruRegBufferLayout{
                static_cast<uint32_t>(xBufferBytes), static_cast<uint32_t>(aBufferBytes),
                static_cast<uint32_t>(pBufferBytes), tileCapacity};
        }
        --tileCapacity;
    }
    return {};
}

__simt_callee__ __aicore__ __inline__ bool CgeruProductNeedsSeparateEvaluation(float lhs, float rhs)
{
    const uint32_t lhsExponent = (__float_as_uint(lhs) >> CGERU_FLOAT_EXPONENT_SHIFT) & CGERU_FLOAT_EXPONENT_MASK;
    const uint32_t rhsExponent = (__float_as_uint(rhs) >> CGERU_FLOAT_EXPONENT_SHIFT) & CGERU_FLOAT_EXPONENT_MASK;
    if (lhsExponent == CGERU_FLOAT_EXPONENT_MASK || rhsExponent == CGERU_FLOAT_EXPONENT_MASK) {
        return true;
    }
    return lhsExponent + rhsExponent >= CGERU_FLOAT_PRODUCT_OVERFLOW_EXPONENT_SUM;
}

__simt_callee__ __aicore__ __inline__ void CgeruComplexMultiply(
    float lhsReal, float lhsImag, float rhsReal, float rhsImag, float& resultReal, float& resultImag)
{
    if (lhsImag == 0.0f) {
        resultReal = lhsReal * rhsReal;
        resultImag = lhsReal * rhsImag;
        return;
    }
    if (lhsReal == 0.0f) {
        resultReal = -(lhsImag * rhsImag);
        resultImag = lhsImag * rhsReal;
        return;
    }
    resultReal = lhsReal * rhsReal - lhsImag * rhsImag;
    resultImag = lhsReal * rhsImag + lhsImag * rhsReal;
}

__simt_callee__ __aicore__ __inline__ void CgeruAccumulate(
    float xReal, float xImag, float pReal, float pImag, uint64_t aFloat, __gm__ float* aGm)
{
    if (CgeruProductNeedsSeparateEvaluation(xReal, pReal) ||
        CgeruProductNeedsSeparateEvaluation(xImag, pImag) ||
        CgeruProductNeedsSeparateEvaluation(xReal, pImag) ||
        CgeruProductNeedsSeparateEvaluation(xImag, pReal)) {
        // Materialize the four FP32 products whenever a product can overflow or
        // contains a non-finite operand. This preserves BLAS Inf/NaN semantics
        // without disabling contraction for the normal path.
        volatile float realProduct0 = xReal * pReal;
        volatile float realProduct1 = xImag * pImag;
        volatile float imagProduct0 = xReal * pImag;
        volatile float imagProduct1 = xImag * pReal;
        const float realUpdate = realProduct0 - realProduct1;
        const float imagUpdate = imagProduct0 + imagProduct1;
        aGm[aFloat] = aGm[aFloat] + realUpdate;
        aGm[aFloat + 1] = aGm[aFloat + 1] + imagUpdate;
        return;
    }

    const float realUpdate = xReal * pReal - xImag * pImag;
    const float imagUpdate = xReal * pImag + xImag * pReal;
    aGm[aFloat] = aGm[aFloat] + realUpdate;
    aGm[aFloat + 1] = aGm[aFloat + 1] + imagUpdate;
}

__simd_callee__ inline void CgeruContiguousRegColumn(
    __ubuf__ float* xAddr, __ubuf__ float* aColAddr, uint32_t rowCount, AscendC::Reg::RegTensor<float>& pReal,
    AscendC::Reg::RegTensor<float>& pImag, AscendC::Reg::MaskReg& yNonzeroMask,
    AscendC::Reg::MaskReg& fullMask)
{
    constexpr uint32_t vectorLength = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    AscendC::Reg::RegTensor<float> xReal;
    AscendC::Reg::RegTensor<float> xImag;
    AscendC::Reg::RegTensor<float> aReal;
    AscendC::Reg::RegTensor<float> aImag;
    AscendC::Reg::RegTensor<float> product0;
    AscendC::Reg::RegTensor<float> product1;
    AscendC::Reg::MaskReg tailMask;
    AscendC::Reg::MaskReg activeMask;

    uint32_t remaining = rowCount;
    for (uint32_t row = 0; row < rowCount; row += vectorLength) {
        tailMask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::And(activeMask, yNonzeroMask, tailMask, fullMask);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(
            xReal, xImag, xAddr + row * CGERU_COMPLEX_COMPONENTS);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_DINTLV_B32>(
            aReal, aImag, aColAddr + row * CGERU_COMPLEX_COMPONENTS);

        AscendC::Reg::Mul(product0, xReal, pReal, activeMask);
        AscendC::Reg::Mul(product1, xImag, pImag, activeMask);
        AscendC::Reg::Sub(product0, product0, product1, activeMask);
        AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::MERGING>(aReal, aReal, product0, activeMask);

        AscendC::Reg::Mul(product0, xReal, pImag, activeMask);
        AscendC::Reg::Mul(product1, xImag, pReal, activeMask);
        AscendC::Reg::Add(product0, product0, product1, activeMask);
        AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::MERGING>(aImag, aImag, product0, activeMask);
        AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_INTLV_B32>(
            aColAddr + row * CGERU_COMPLEX_COMPONENTS, aReal, aImag, tailMask);
    }
}

__simd_vf__ inline void CgeruContiguousRegVf(
    __ubuf__ float* xAddr, __ubuf__ float* aAddr, __ubuf__ float* paramsAddr, uint32_t rowCount, uint32_t colCount,
    uint32_t alphaMode)
{
    constexpr uint32_t yBase = 32U / sizeof(float);
    const uint32_t rowBytes = rowCount * CGERU_COMPLEX_COMPONENTS * sizeof(float);
    constexpr uint32_t complexVectorBytes = 2U * AscendC::VECTOR_REG_WIDTH;
    const uint32_t aColStride =
        ((rowBytes + complexVectorBytes - 1U) / complexVectorBytes) * (complexVectorBytes / sizeof(float));
    AscendC::Reg::RegTensor<float> alphaReal;
    AscendC::Reg::RegTensor<float> alphaImag;
    AscendC::Reg::RegTensor<float> yReal;
    AscendC::Reg::RegTensor<float> yImag;
    AscendC::Reg::RegTensor<float> pReal;
    AscendC::Reg::RegTensor<float> pImag;
    AscendC::Reg::RegTensor<float> product0;
    AscendC::Reg::RegTensor<float> product1;
    AscendC::Reg::MaskReg yRealNonzero;
    AscendC::Reg::MaskReg yImagNonzero;
    AscendC::Reg::MaskReg yNonzero;
    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(alphaReal, paramsAddr);
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(alphaImag, paramsAddr + 1U);
    for (uint32_t localCol = 0; localCol < colCount; ++localCol) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
            yReal, paramsAddr + yBase + localCol * CGERU_COMPLEX_COMPONENTS);
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(
            yImag, paramsAddr + yBase + localCol * CGERU_COMPLEX_COMPONENTS + 1U);
        if (alphaMode == 1U) {
            AscendC::Reg::Mul(pReal, alphaReal, yReal, fullMask);
            AscendC::Reg::Mul(pImag, alphaReal, yImag, fullMask);
        } else if (alphaMode == 2U) {
            AscendC::Reg::Mul(pReal, alphaImag, yImag, fullMask);
            AscendC::Reg::Muls(pReal, pReal, -1.0f, fullMask);
            AscendC::Reg::Mul(pImag, alphaImag, yReal, fullMask);
        } else {
            AscendC::Reg::Mul(product0, alphaReal, yReal, fullMask);
            AscendC::Reg::Mul(product1, alphaImag, yImag, fullMask);
            AscendC::Reg::Sub(pReal, product0, product1, fullMask);
            AscendC::Reg::Mul(product0, alphaReal, yImag, fullMask);
            AscendC::Reg::Mul(product1, alphaImag, yReal, fullMask);
            AscendC::Reg::Add(pImag, product0, product1, fullMask);
        }
        AscendC::Reg::Compares<float, AscendC::CMPMODE::NE>(yRealNonzero, yReal, 0.0f, fullMask);
        AscendC::Reg::Compares<float, AscendC::CMPMODE::NE>(yImagNonzero, yImag, 0.0f, fullMask);
        AscendC::Reg::Or(yNonzero, yRealNonzero, yImagNonzero, fullMask);
        CgeruContiguousRegColumn(
            xAddr, aAddr + localCol * aColStride, rowCount, pReal, pImag, yNonzero, fullMask);
    }
}

__aicore__ __inline__ void CgeruProcessRegTile(
    AscendC::GlobalTensor<float>& yGm, AscendC::GlobalTensor<float>& aGm, AscendC::LocalTensor<float>& aLocal,
    AscendC::LocalTensor<float>& paramsLocal, __ubuf__ float* xAddr, uint32_t lda, uint32_t rowStart, uint32_t rowCount,
    uint32_t colStart, uint32_t colCount, uint32_t alphaMode, event_t eventMte2ToV, event_t eventVToMte3,
    event_t eventMte3ToMte2)
{
    const uint32_t rowBytes = rowCount * CGERU_COMPLEX_COMPONENTS * sizeof(float);
    const uint32_t alignedRowBytes = ((rowBytes + 31U) / 32U) * 32U;
    constexpr uint32_t complexVectorBytes = 2U * AscendC::VECTOR_REG_WIDTH;
    const uint32_t aColBytes = ((rowBytes + complexVectorBytes - 1U) / complexVectorBytes) * complexVectorBytes;
    const uint32_t aUbGapBlocks = (aColBytes - alignedRowBytes) / 32U;
    const uint32_t gmColumnGap = (lda - rowCount) * CGERU_COMPLEX_COMPONENTS * sizeof(float);
    const uint64_t aOffset = (static_cast<uint64_t>(colStart) * lda + rowStart) * CGERU_COMPLEX_COMPONENTS;
    const AscendC::DataCopyPadExtParams<float> noPadding{false, 0, 0, 0.0f};
    if (gmColumnGap == 0U && rowBytes == aColBytes) {
        const AscendC::DataCopyExtParams copyIn{1, rowBytes * colCount, 0, 0, 0};
        AscendC::DataCopyPad<float>(aLocal, aGm[aOffset], copyIn, noPadding);
    } else {
        const AscendC::DataCopyExtParams copyIn{
            static_cast<uint16_t>(colCount), rowBytes, gmColumnGap, aUbGapBlocks, 0};
        AscendC::DataCopyPad<float>(aLocal, aGm[aOffset], copyIn, noPadding);
    }
    constexpr uint32_t yBase = 32U / sizeof(float);
    const AscendC::DataCopyExtParams yCopy{
        1, static_cast<uint32_t>(colCount * CGERU_COMPLEX_COMPONENTS * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad<float>(
        paramsLocal[yBase], yGm[static_cast<uint64_t>(colStart) * CGERU_COMPLEX_COMPONENTS], yCopy, noPadding);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventMte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventMte2ToV);

    auto* aAddr = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
    auto* paramsAddr = reinterpret_cast<__ubuf__ float*>(paramsLocal.GetPhyAddr());
    asc_vf_call<CgeruContiguousRegVf>(xAddr, aAddr, paramsAddr, rowCount, colCount, alphaMode);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventVToMte3);

    if (gmColumnGap == 0U && rowBytes == aColBytes) {
        const AscendC::DataCopyExtParams copyOut{1, rowBytes * colCount, 0, 0, 0};
        AscendC::DataCopyPad<float>(aGm[aOffset], aLocal, copyOut);
    } else {
        const AscendC::DataCopyExtParams copyOut{
            static_cast<uint16_t>(colCount), rowBytes, aUbGapBlocks, gmColumnGap, 0};
        AscendC::DataCopyPad<float>(aGm[aOffset], aLocal, copyOut);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventMte3ToMte2);
}

__simt_callee__ __aicore__ __inline__ void CgeruLoadCacheInputs(
    uint64_t xStart, uint64_t yStart, uint32_t rowStart, uint32_t rowCount, uint32_t colStart, uint32_t colCount,
    int64_t incx, int64_t incy, float alphaReal, float alphaImag, __gm__ const float* xGm, __gm__ const float* yGm,
    __ubuf__ float* xRealUb, __ubuf__ float* xImagUb, __ubuf__ float* pRealUb, __ubuf__ float* pImagUb,
    __ubuf__ uint32_t* yFlagUb)
{
    for (uint32_t localRow = threadIdx.x; localRow < rowCount; localRow += CGERU_SIMT_THREADS) {
        const int64_t xElement = static_cast<int64_t>(xStart) + static_cast<int64_t>(rowStart + localRow) * incx;
        const int64_t xFloat = xElement * CGERU_COMPLEX_COMPONENTS;
        xRealUb[localRow] = xGm[xFloat];
        xImagUb[localRow] = xGm[xFloat + 1];
    }
    for (uint32_t localCol = threadIdx.x; localCol < colCount; localCol += CGERU_SIMT_THREADS) {
        const int64_t yElement = static_cast<int64_t>(yStart) + static_cast<int64_t>(colStart + localCol) * incy;
        const int64_t yFloat = yElement * CGERU_COMPLEX_COMPONENTS;
        const float yReal = yGm[yFloat];
        const float yImag = yGm[yFloat + 1];
        const uint32_t yFlag = (yReal != 0.0f || yImag != 0.0f) ? 1U : 0U;
        yFlagUb[localCol] = yFlag;
        if (yFlag != 0U) {
            float pReal = 0.0f;
            float pImag = 0.0f;
            CgeruComplexMultiply(alphaReal, alphaImag, yReal, yImag, pReal, pImag);
            pRealUb[localCol] = pReal;
            pImagUb[localCol] = pImag;
        } else {
            pRealUb[localCol] = 0.0f;
            pImagUb[localCol] = 0.0f;
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGERU_SIMT_THREADS) inline void CgeruCache(
    uint32_t lda, float alphaReal, float alphaImag, int64_t incx, int64_t incy, uint64_t xStart, uint64_t yStart,
    uint32_t rowStart, uint32_t rowCount, uint32_t colStart, uint32_t colCount, __gm__ const float* xGm,
    __gm__ const float* yGm, __gm__ float* aGm)
{
    __ubuf__ float xRealUb[CGERU_CACHE_COMPLEX];
    __ubuf__ float xImagUb[CGERU_CACHE_COMPLEX];
    __ubuf__ float pRealUb[CGERU_CACHE_COMPLEX];
    __ubuf__ float pImagUb[CGERU_CACHE_COMPLEX];
    __ubuf__ uint32_t yFlagUb[CGERU_CACHE_COMPLEX];

    CgeruLoadCacheInputs(
        xStart, yStart, rowStart, rowCount, colStart, colCount, incx, incy, alphaReal, alphaImag, xGm, yGm, xRealUb,
        xImagUb, pRealUb, pImagUb, yFlagUb);
    asc_syncthreads();

    if (rowCount >= (CGERU_SIMT_THREADS / 2U)) {
        for (uint32_t localRow = threadIdx.x; localRow < rowCount; localRow += CGERU_SIMT_THREADS) {
            const float xReal = xRealUb[localRow];
            const float xImag = xImagUb[localRow];
            for (uint32_t localCol = 0; localCol < colCount; ++localCol) {
                if (yFlagUb[localCol] == 0U) {
                    continue;
                }
                const float pReal = pRealUb[localCol];
                const float pImag = pImagUb[localCol];
                const uint64_t aElement = static_cast<uint64_t>(colStart + localCol) * lda + rowStart + localRow;
                const uint64_t aFloat = aElement * CGERU_COMPLEX_COMPONENTS;
                CgeruAccumulate(xReal, xImag, pReal, pImag, aFloat, aGm);
            }
        }
        return;
    }

    const uint64_t tileElements = static_cast<uint64_t>(rowCount) * colCount;
    for (uint64_t local = threadIdx.x; local < tileElements; local += CGERU_SIMT_THREADS) {
        const uint32_t localCol = static_cast<uint32_t>(local / rowCount);
        if (yFlagUb[localCol] == 0U) {
            continue;
        }
        const uint32_t localRow = static_cast<uint32_t>(local - static_cast<uint64_t>(localCol) * rowCount);
        const float xReal = xRealUb[localRow];
        const float xImag = xImagUb[localRow];
        const float pReal = pRealUb[localCol];
        const float pImag = pImagUb[localCol];
        const uint64_t aElement = static_cast<uint64_t>(colStart + localCol) * lda + rowStart + localRow;
        const uint64_t aFloat = aElement * CGERU_COMPLEX_COMPONENTS;
        CgeruAccumulate(xReal, xImag, pReal, pImag, aFloat, aGm);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGERU_SIMT_THREADS) inline void CgeruDirectGm(
    uint32_t lda, float alphaReal, float alphaImag, int64_t incx, int64_t incy, uint64_t xStart, uint64_t yStart,
    uint32_t rowStart, uint32_t rowCount, uint32_t colStart, uint32_t colCount, __gm__ const float* xGm,
    __gm__ const float* yGm, __gm__ float* aGm)
{
    const uint64_t tileElements = static_cast<uint64_t>(rowCount) * colCount;
    for (uint64_t local = threadIdx.x; local < tileElements; local += CGERU_SIMT_THREADS) {
        const uint32_t localCol = static_cast<uint32_t>(local / rowCount);
        const uint32_t localRow = static_cast<uint32_t>(local - static_cast<uint64_t>(localCol) * rowCount);
        const uint32_t col = colStart + localCol;
        const int64_t yElement = static_cast<int64_t>(yStart) + static_cast<int64_t>(col) * incy;
        const int64_t yFloat = yElement * CGERU_COMPLEX_COMPONENTS;
        const float yReal = yGm[yFloat];
        const float yImag = yGm[yFloat + 1];
        if (yReal == 0.0f && yImag == 0.0f) {
            continue;
        }

        const uint32_t row = rowStart + localRow;
        const int64_t xElement = static_cast<int64_t>(xStart) + static_cast<int64_t>(row) * incx;
        const int64_t xFloat = xElement * CGERU_COMPLEX_COMPONENTS;
        const float xReal = xGm[xFloat];
        const float xImag = xGm[xFloat + 1];
        float pReal = 0.0f;
        float pImag = 0.0f;
        CgeruComplexMultiply(alphaReal, alphaImag, yReal, yImag, pReal, pImag);
        const uint64_t aElement = static_cast<uint64_t>(col) * lda + row;
        const uint64_t aFloat = aElement * CGERU_COMPLEX_COMPONENTS;
        CgeruAccumulate(xReal, xImag, pReal, pImag, aFloat, aGm);
    }
}

__aicore__ __inline__ void CgeruRunDirectGm(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, const CgeruTilingData& tiling, const CgeruTile& tile)
{
    const dim3 threadCount{CGERU_SIMT_THREADS, 1, 1};
    asc_vf_call<CgeruDirectGm>(
        threadCount, tiling.lda, tiling.alphaReal, tiling.alphaImag, tiling.incx, tiling.incy, tiling.xStart,
        tiling.yStart, tile.rowStart, tile.rowCount, tile.colStart, tile.colCount,
        reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ const float*>(y),
        reinterpret_cast<__gm__ float*>(A));
}

__aicore__ __inline__ void CgeruRunReg(
    GM_ADDR x, GM_ADDR y, GM_ADDR A, const CgeruTilingData& tiling, const CgeruTile& tile,
    const CgeruRegBufferLayout& layout)
{
    const uint32_t rowStart = tile.rowStart;
    const uint32_t rowCount = tile.rowCount;
    const uint32_t colStart = tile.colStart;
    const uint32_t colCount = tile.colCount;

    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> aGm;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));

    const uint32_t rowBytes = rowCount * CGERU_COMPLEX_COMPONENTS * sizeof(float);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> xBuffer;
    AscendC::TBuf<AscendC::TPosition::VECIN> aBuffer;
    AscendC::TBuf<AscendC::TPosition::VECIN> pBuffer;
    pipe.InitBuffer(xBuffer, layout.xBufferBytes);
    pipe.InitBuffer(aBuffer, layout.aBufferBytes);
    pipe.InitBuffer(pBuffer, layout.pBufferBytes);
    AscendC::LocalTensor<float> xLocal = xBuffer.Get<float>();
    AscendC::LocalTensor<float> aLocal = aBuffer.Get<float>();
    AscendC::LocalTensor<float> pLocal = pBuffer.Get<float>();
    const CgeruRegEvents events = CgeruAcquireRegEvents();
    const uint64_t xOffset = static_cast<uint64_t>(rowStart) * CGERU_COMPLEX_COMPONENTS;
    const AscendC::DataCopyExtParams xCopy{1, rowBytes, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<float> noPadding{false, 0, 0, 0.0f};
    AscendC::DataCopyPad<float, AscendC::PaddingMode::Compact>(xLocal, xGm[xOffset], xCopy, noPadding);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(events.mte2ToV);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(events.mte2ToV);
    auto* xAddr = reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr());
    // Scalar stores initialize UB consumed by VF loads; volatile preserves the
    // memory writes, while S_V supplies pipeline ordering. This is not an FMA guard.
    auto* pScalarAddr = reinterpret_cast<__ubuf__ volatile float*>(pLocal.GetPhyAddr());
    pScalarAddr[0] = tiling.alphaReal;
    pScalarAddr[1] = tiling.alphaImag;
    const uint32_t alphaMode = (tiling.alphaImag == 0.0f) ? 1U : ((tiling.alphaReal == 0.0f) ? 2U : 0U);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(events.scalarToV);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(events.scalarToV);

    for (uint32_t colOffset = 0; colOffset < colCount; colOffset += layout.tileCapacity) {
        const uint32_t tileCols =
            (colCount - colOffset < layout.tileCapacity) ? (colCount - colOffset) : layout.tileCapacity;
        CgeruProcessRegTile(
            yGm, aGm, aLocal, pLocal, xAddr, tiling.lda, rowStart, rowCount, colStart + colOffset, tileCols, alphaMode,
            events.mte2ToV, events.vToMte3, events.mte3ToMte2);
    }
}

} // namespace

__global__ __aicore__ void cgeru_contiguous_reg_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR A, const CgeruTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const CgeruTile tile = CgeruGetTile(tiling);
    if (tile.rowCount == 0U || tile.colCount == 0U) {
        return;
    }
    const CgeruRegBufferLayout layout = CgeruPlanRegBuffers(tile.rowCount, tile.colCount);
    if (layout.tileCapacity == 0U) {
        // No valid UB plan: compute the same tile directly, without InitBuffer.
        CgeruRunDirectGm(x, y, A, tiling, tile);
        return;
    }
    CgeruRunReg(x, y, A, tiling, tile, layout);
}

__global__ __aicore__ void cgeru_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR A, const CgeruTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const CgeruTile tile = CgeruGetTile(tiling);

    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ const float*>(y);
    auto* aGm = reinterpret_cast<__gm__ float*>(A);
    const dim3 threadCount{CGERU_SIMT_THREADS, 1, 1};
    if (tiling.tilingKey == CGERU_TILING_CACHE_X_AND_P) {
        asc_vf_call<CgeruCache>(
            threadCount, tiling.lda, tiling.alphaReal, tiling.alphaImag, tiling.incx, tiling.incy, tiling.xStart,
            tiling.yStart, tile.rowStart, tile.rowCount, tile.colStart, tile.colCount, xGm, yGm, aGm);
        return;
    }
    CgeruRunDirectGm(x, y, A, tiling, tile);
}

void cgeru_arch35_kernel_do(
    const uint8_t* x, const uint8_t* y, uint8_t* A, const CgeruTilingData& tiling, uint32_t numBlocks, void* stream)
{
    GM_ADDR kernelX = const_cast<uint8_t*>(x);
    GM_ADDR kernelY = const_cast<uint8_t*>(y);
    if (tiling.tilingKey == CGERU_TILING_CONTIGUOUS_REG) {
        cgeru_contiguous_reg_kernel<<<numBlocks, nullptr, stream>>>(kernelX, kernelY, A, tiling);
        return;
    }
    cgeru_kernel<<<numBlocks, nullptr, stream>>>(kernelX, kernelY, A, tiling);
}
