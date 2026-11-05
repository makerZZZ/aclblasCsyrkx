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
 * \file csymv_kernel.cpp
 * \brief CSYMV Kernel for ascend950 (DAV_3510)
 */

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "csymv_tiling_data.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"

using namespace AscendC;

namespace {

constexpr uint32_t CSYMV_REG_TILE = 64U;
constexpr uint32_t CSYMV_REG_COMPLEX_FLOATS = CSYMV_REG_TILE * 2U;

__simd_vf__ inline void CsymvReduceRegTilesVf(
    __ubuf__ float* partialAddr, __ubuf__ float* outAddr, uint32_t tileCount, float alphaReal, float alphaImag)
{
    using namespace AscendC::Reg;
    RegTensor<float> sumReal, sumImag, partialReal, partialImag, outReal, outImag, temp;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    Duplicate(sumReal, 0.0f, all);
    Duplicate(sumImag, 0.0f, all);
    for (uint16_t tile = 0U; tile < tileCount; ++tile) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
            partialReal, partialImag, partialAddr + static_cast<uint32_t>(tile) * CSYMV_REG_COMPLEX_FLOATS);
        Add(sumReal, sumReal, partialReal, all);
        Add(sumImag, sumImag, partialImag, all);
    }
    Muls(outReal, sumReal, alphaReal, all);
    Muls(temp, sumImag, alphaImag, all);
    Sub(outReal, outReal, temp, all);
    Muls(outImag, sumImag, alphaReal, all);
    Muls(temp, sumReal, alphaImag, all);
    Add(outImag, outImag, temp, all);
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(outAddr, outReal, outImag, all);
}

constexpr uint32_t CSYMV_REG_ROW_CHUNKS = 4U;
constexpr uint32_t CSYMV_REG_RECT_ROWS = CSYMV_REG_ROW_CHUNKS * CSYMV_REG_TILE;
constexpr uint32_t CSYMV_REG_RECT_MATRIX_FLOATS = CSYMV_REG_RECT_ROWS * CSYMV_REG_COMPLEX_FLOATS;

template <bool UPLO_IS_UPPER, bool IS_DIAGONAL>
__simd_callee__ inline void CsymvRegRowProductsVf(
    __ubuf__ float* aAddr, __ubuf__ float* xColAddr, __ubuf__ float* rowOutAddr, uint32_t rowCount,
    uint32_t colCount)
{
    using namespace AscendC::Reg;
    RegTensor<float> aReal, aImag, xColReal, xColImag, rowReal, rowImag, temp, zero;
    RegTensor<int32_t> rowIndex;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg directMask;
    MaskReg rowMask = UpdateMask<float>(rowCount);
    Duplicate(zero, 0.0f, all);
    if constexpr (IS_DIAGONAL) {
        Arange(rowIndex, 0);
    }
    Duplicate(rowReal, 0.0f, all);
    Duplicate(rowImag, 0.0f, all);
    __ubuf__ float* aRead = aAddr;
    __ubuf__ float* xColRealRead = xColAddr;
    __ubuf__ float* xColImagRead = xColAddr + 1U;
    for (uint16_t col = 0U; col < colCount; ++col) {
        LoadAlign<float, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B32>(
            aReal, aImag, aRead, CSYMV_REG_RECT_ROWS * 2U);
        Select(aReal, aReal, zero, rowMask);
        Select(aImag, aImag, zero, rowMask);
        LoadAlign<float, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_BRC_B32>(xColReal, xColRealRead, 2U);
        LoadAlign<float, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_BRC_B32>(xColImag, xColImagRead, 2U);
        if constexpr (IS_DIAGONAL) {
            if constexpr (UPLO_IS_UPPER) {
                CompareScalar<int32_t, CMPMODE::LE>(directMask, rowIndex, static_cast<int32_t>(col), all);
            } else {
                CompareScalar<int32_t, CMPMODE::GE>(directMask, rowIndex, static_cast<int32_t>(col), all);
            }
            Select(aReal, aReal, zero, directMask);
            Select(aImag, aImag, zero, directMask);
        }
        Muls(temp, aImag, -1.0f, all);
        MulAddDst(rowReal, aReal, xColReal, all);
        MulAddDst(rowReal, temp, xColImag, all);
        MulAddDst(rowImag, aReal, xColImag, all);
        MulAddDst(rowImag, aImag, xColReal, all);
    }
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(rowOutAddr, rowReal, rowImag, all);
}

template <bool UPLO_IS_UPPER, bool IS_DIAGONAL>
__simd_callee__ inline void CsymvRegColProductsVf(
    __ubuf__ float* aAddr, __ubuf__ float* xRowAddr, __ubuf__ float* colOutAddr, uint32_t rowCount,
    uint32_t colCount)
{
    using namespace AscendC::Reg;
    RegTensor<float> aReal, aImag, xRowReal, xRowImag, productReal, productImag, temp, zero;
    RegTensor<int32_t> rowIndex;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    MaskReg mirrorMask;
    MaskReg rowMask = UpdateMask<float>(rowCount);
    Duplicate(zero, 0.0f, all);
    if constexpr (IS_DIAGONAL) {
        Arange(rowIndex, 0);
    }
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(xRowReal, xRowImag, xRowAddr);
    Select(xRowReal, xRowReal, zero, rowMask);
    Select(xRowImag, xRowImag, zero, rowMask);
    __ubuf__ float* aRead = aAddr;
    __ubuf__ float* colWrite = colOutAddr;
    UnalignRegForStore colStore;
    for (uint16_t col = 0U; col < colCount; ++col) {
        LoadAlign<float, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B32>(
            aReal, aImag, aRead, CSYMV_REG_RECT_ROWS * 2U);
        Select(aReal, aReal, zero, rowMask);
        Select(aImag, aImag, zero, rowMask);
        if constexpr (IS_DIAGONAL) {
            if constexpr (UPLO_IS_UPPER) {
                CompareScalar<int32_t, CMPMODE::LT>(mirrorMask, rowIndex, static_cast<int32_t>(col), all);
            } else {
                CompareScalar<int32_t, CMPMODE::GT>(mirrorMask, rowIndex, static_cast<int32_t>(col), all);
            }
            Select(aReal, aReal, zero, mirrorMask);
            Select(aImag, aImag, zero, mirrorMask);
        }
        Muls(temp, aImag, -1.0f, all);
        Duplicate(productReal, 0.0f, all);
        Duplicate(productImag, 0.0f, all);
        MulAddDst(productReal, aReal, xRowReal, all);
        MulAddDst(productReal, temp, xRowImag, all);
        MulAddDst(productImag, aReal, xRowImag, all);
        MulAddDst(productImag, aImag, xRowReal, all);
        ReduceSum(productReal, productReal, all);
        ReduceSum(productImag, productImag, all);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(colWrite, productReal, colStore, 1U);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(colWrite, productImag, colStore, 1U);
    }
    StoreUnAlignPost(colWrite, colStore, 0U);
}

template <bool UPLO_IS_UPPER, bool IS_DIAGONAL, uint16_t CHUNK_COUNT>
__simd_vf__ inline void CsymvRegFixedTileVf(
    __ubuf__ float* aAddr, __ubuf__ float* xRowAddr, __ubuf__ float* xColAddr, __ubuf__ float* rowOutAddr,
    __ubuf__ float* colOutAddr, uint32_t rowCount, uint32_t colCount)
{
    static_assert(!IS_DIAGONAL || CHUNK_COUNT == 1U, "a diagonal vector function must cover exactly one 64x64 tile");
    for (uint16_t chunk = 0U; chunk < CHUNK_COUNT; ++chunk) {
        const uint32_t offset = static_cast<uint32_t>(chunk) * CSYMV_REG_COMPLEX_FLOATS;
        const uint32_t chunkBase = static_cast<uint32_t>(chunk) * CSYMV_REG_TILE;
        const uint32_t chunkRowCount =
            rowCount > chunkBase ? (rowCount - chunkBase < CSYMV_REG_TILE ? rowCount - chunkBase : CSYMV_REG_TILE) : 0U;
        CsymvRegRowProductsVf<UPLO_IS_UPPER, IS_DIAGONAL>(
            aAddr + offset, xColAddr, rowOutAddr + offset, chunkRowCount, colCount);
        CsymvRegColProductsVf<UPLO_IS_UPPER, IS_DIAGONAL>(
            aAddr + offset, xRowAddr + offset, colOutAddr + offset, chunkRowCount, colCount);
    }
}

template <bool UPLO_IS_UPPER>
class CsymvRegRectAiv {
public:
    __aicore__ inline void Init(
        __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* workspaceGm, const CsymvTilingData& tiling,
        TPipe* pipe)
    {
        blockIdx_ = GetBlockIdx();
        n_ = tiling.n;
        lda_ = tiling.lda;
        useCoreNum_ = tiling.useCoreNum;
        tileCount_ = (n_ + CSYMV_REG_TILE - 1U) / CSYMV_REG_TILE;
        rowBlockCount_ = (tileCount_ + CSYMV_REG_ROW_CHUNKS - 1U) / CSYMV_REG_ROW_CHUNKS;
        totalTasks_ = 0U;
        for (uint32_t colTile = 0U; colTile < tileCount_; ++colTile) {
            totalTasks_ +=
                UPLO_IS_UPPER ? colTile / CSYMV_REG_ROW_CHUNKS + 1U : rowBlockCount_ - colTile / CSYMV_REG_ROW_CHUNKS;
        }
        aGlobal_.SetGlobalBuffer(const_cast<__gm__ float*>(aGm), static_cast<uint64_t>(lda_) * n_ * 2U);
        xGlobal_.SetGlobalBuffer(const_cast<__gm__ float*>(xGm), static_cast<uint64_t>(n_) * 2U);
        workspaceGlobal_.SetGlobalBuffer(
            workspaceGm, static_cast<uint64_t>(tileCount_) * tileCount_ * CSYMV_REG_COMPLEX_FLOATS);

        pipe->InitBuffer(aQueue_, 1, CSYMV_REG_RECT_MATRIX_FLOATS * sizeof(float));
        pipe->InitBuffer(xRowQueue_, 1, CSYMV_REG_RECT_ROWS * 2U * sizeof(float));
        pipe->InitBuffer(xColQueue_, 1, CSYMV_REG_COMPLEX_FLOATS * sizeof(float));
        pipe->InitBuffer(rowOutQueue_, 1, CSYMV_REG_RECT_ROWS * 2U * sizeof(float));
        pipe->InitBuffer(colOutQueue_, 1, CSYMV_REG_RECT_ROWS * 2U * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (useCoreNum_ == 0U || blockIdx_ >= useCoreNum_) {
            return;
        }
        for (uint32_t task = blockIdx_; task < totalTasks_; task += useCoreNum_) {
            uint32_t rowBlock;
            uint32_t colTile;
            DecodeTask(task, rowBlock, colTile);
            ProcessTile(rowBlock, colTile);
        }
    }

private:
    template <bool IS_DIAGONAL, uint16_t CHUNK_COUNT>
    __aicore__ inline void CallFixedTile(
        __ubuf__ float* aAddr, __ubuf__ float* xRowAddr, __ubuf__ float* xColAddr, __ubuf__ float* rowOutAddr,
        __ubuf__ float* colOutAddr, uint16_t firstChunk, uint32_t rowCount, uint32_t colCount)
    {
        const uint32_t offset = static_cast<uint32_t>(firstChunk) * CSYMV_REG_COMPLEX_FLOATS;
        const uint32_t firstRow = static_cast<uint32_t>(firstChunk) * CSYMV_REG_TILE;
        const uint32_t remainingRows = rowCount > firstRow ? rowCount - firstRow : 0U;
        if constexpr (UPLO_IS_UPPER) {
            asc_vf_call<CsymvRegFixedTileVf<true, IS_DIAGONAL, CHUNK_COUNT>>(
                aAddr + offset, xRowAddr + offset, xColAddr, rowOutAddr + offset, colOutAddr + offset,
                remainingRows, colCount);
        } else {
            asc_vf_call<CsymvRegFixedTileVf<false, IS_DIAGONAL, CHUNK_COUNT>>(
                aAddr + offset, xRowAddr + offset, xColAddr, rowOutAddr + offset, colOutAddr + offset,
                remainingRows, colCount);
        }
    }

    __aicore__ inline void DecodeTask(uint32_t task, uint32_t& rowBlock, uint32_t& colTile) const
    {
        uint32_t offset = task;
        colTile = 0U;
        for (; colTile < tileCount_; ++colTile) {
            const uint32_t diagonalBlock = colTile / CSYMV_REG_ROW_CHUNKS;
            const uint32_t count = UPLO_IS_UPPER ? diagonalBlock + 1U : rowBlockCount_ - diagonalBlock;
            if (offset < count) {
                break;
            }
            offset -= count;
        }
        const uint32_t diagonalBlock = colTile / CSYMV_REG_ROW_CHUNKS;
        rowBlock = UPLO_IS_UPPER ? offset : diagonalBlock + offset;
    }

    __aicore__ inline void CopyInputs(uint32_t rowBase, uint32_t colBase, uint32_t rowCount, uint32_t colCount)
    {
        LocalTensor<float> aLocal = aQueue_.AllocTensor<float>();
        const uint32_t rowFloats = rowCount * 2U;
        const uint32_t alignedRowFloats = (rowFloats + 7U) & ~7U;
        const uint32_t rowPaddingFloats = alignedRowFloats - rowFloats;
        const int64_t srcStride =
            static_cast<int64_t>(lda_ - rowCount) * 2L * static_cast<int64_t>(sizeof(float));
        constexpr uint32_t floatsPerBlock = 32U / sizeof(float);
        const int64_t dstStride =
            static_cast<int64_t>((CSYMV_REG_RECT_ROWS * 2U - alignedRowFloats) / floatsPerBlock);
        DataCopyExtParams aParams{
            static_cast<uint16_t>(colCount), rowCount * 2U * static_cast<uint32_t>(sizeof(float)), srcStride,
            dstStride, 0};
        DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(rowPaddingFloats), 0.0f};
        const uint64_t aOffset = (static_cast<uint64_t>(colBase) * lda_ + rowBase) * 2U;
        DataCopyPad(aLocal, aGlobal_[aOffset], aParams, padParams);
        aQueue_.EnQue(aLocal);

        LocalTensor<float> xRowLocal = xRowQueue_.AllocTensor<float>();
        DataCopyExtParams rowParams{1U, rowCount * 2U * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(xRowLocal, xGlobal_[static_cast<uint64_t>(rowBase) * 2U], rowParams, padParams);
        xRowQueue_.EnQue(xRowLocal);
        LocalTensor<float> xColLocal = xColQueue_.AllocTensor<float>();
        const uint32_t colFloats = colCount * 2U;
        const uint32_t colPaddingFloats = ((colFloats + 7U) & ~7U) - colFloats;
        DataCopyExtParams colParams{1U, colCount * 2U * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> colPad{true, 0, static_cast<uint8_t>(colPaddingFloats), 0.0f};
        DataCopyPad(xColLocal, xGlobal_[static_cast<uint64_t>(colBase) * 2U], colParams, colPad);
        xColQueue_.EnQue(xColLocal);
    }

    __aicore__ inline void ComputeUpperDiagonal(
        uint16_t diagonalChunk, __ubuf__ float* aAddr, __ubuf__ float* xRowAddr, __ubuf__ float* xColAddr,
        __ubuf__ float* rowOutAddr, __ubuf__ float* colOutAddr, uint32_t rowCount, uint32_t colCount)
    {
        switch (diagonalChunk) {
            case 0U:
                CallFixedTile<true, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 0U, rowCount, colCount);
                break;
            case 1U:
                CallFixedTile<false, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 0U, rowCount, colCount);
                CallFixedTile<true, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 1U, rowCount, colCount);
                break;
            case 2U:
                CallFixedTile<false, 2U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 0U, rowCount, colCount);
                CallFixedTile<true, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 2U, rowCount, colCount);
                break;
            default:
                CallFixedTile<false, 3U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 0U, rowCount, colCount);
                CallFixedTile<true, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 3U, rowCount, colCount);
                break;
        }
    }

    __aicore__ inline void ComputeLowerDiagonal(
        uint16_t diagonalChunk, __ubuf__ float* aAddr, __ubuf__ float* xRowAddr, __ubuf__ float* xColAddr,
        __ubuf__ float* rowOutAddr, __ubuf__ float* colOutAddr, uint32_t rowCount, uint32_t colCount)
    {
        switch (diagonalChunk) {
            case 0U:
                CallFixedTile<true, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 0U, rowCount, colCount);
                CallFixedTile<false, 3U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 1U, rowCount, colCount);
                break;
            case 1U:
                CallFixedTile<true, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 1U, rowCount, colCount);
                CallFixedTile<false, 2U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 2U, rowCount, colCount);
                break;
            case 2U:
                CallFixedTile<true, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 2U, rowCount, colCount);
                CallFixedTile<false, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 3U, rowCount, colCount);
                break;
            default:
                CallFixedTile<true, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 3U, rowCount, colCount);
                break;
        }
    }

    __aicore__ inline void DispatchFixedTiles(
        uint16_t diagonalChunk, bool hasDiagonal, __ubuf__ float* aAddr, __ubuf__ float* xRowAddr,
        __ubuf__ float* xColAddr, __ubuf__ float* rowOutAddr, __ubuf__ float* colOutAddr, uint32_t rowCount,
        uint32_t colCount)
    {
        if (!hasDiagonal) {
            CallFixedTile<false, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 0U, rowCount, colCount);
            CallFixedTile<false, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 1U, rowCount, colCount);
            CallFixedTile<false, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 2U, rowCount, colCount);
            CallFixedTile<false, 1U>(aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, 3U, rowCount, colCount);
        } else if constexpr (UPLO_IS_UPPER) {
            ComputeUpperDiagonal(diagonalChunk, aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, rowCount, colCount);
        } else {
            ComputeLowerDiagonal(diagonalChunk, aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, rowCount, colCount);
        }
    }

    __aicore__ inline void Compute(uint16_t diagonalChunk, bool hasDiagonal, uint32_t rowCount, uint32_t colCount)
    {
        LocalTensor<float> aLocal = aQueue_.DeQue<float>();
        LocalTensor<float> xRowLocal = xRowQueue_.DeQue<float>();
        LocalTensor<float> xColLocal = xColQueue_.DeQue<float>();
        LocalTensor<float> rowOutLocal = rowOutQueue_.AllocTensor<float>();
        LocalTensor<float> colOutLocal = colOutQueue_.AllocTensor<float>();
        auto* aAddr = reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr());
        auto* xRowAddr = reinterpret_cast<__ubuf__ float*>(xRowLocal.GetPhyAddr());
        auto* xColAddr = reinterpret_cast<__ubuf__ float*>(xColLocal.GetPhyAddr());
        auto* rowOutAddr = reinterpret_cast<__ubuf__ float*>(rowOutLocal.GetPhyAddr());
        auto* colOutAddr = reinterpret_cast<__ubuf__ float*>(colOutLocal.GetPhyAddr());
        DispatchFixedTiles(
            diagonalChunk, hasDiagonal, aAddr, xRowAddr, xColAddr, rowOutAddr, colOutAddr, rowCount, colCount);
        rowOutQueue_.EnQue(rowOutLocal);
        colOutQueue_.EnQue(colOutLocal);
        aQueue_.FreeTensor(aLocal);
        xRowQueue_.FreeTensor(xRowLocal);
        xColQueue_.FreeTensor(xColLocal);
    }

    __aicore__ inline void CopyOutputs(uint32_t rowBlock, uint32_t colTile, uint16_t firstChunk, uint16_t chunkCount)
    {
        LocalTensor<float> rowOutLocal = rowOutQueue_.DeQue<float>();
        LocalTensor<float> colOutLocal = colOutQueue_.DeQue<float>();
        for (uint16_t chunkOffset = 0U; chunkOffset < chunkCount; ++chunkOffset) {
            const uint16_t chunk = firstChunk + chunkOffset;
            const uint32_t rowTile = rowBlock * CSYMV_REG_ROW_CHUNKS + chunk;
            const uint64_t rowOffset =
                (static_cast<uint64_t>(rowTile) * tileCount_ + colTile) * CSYMV_REG_COMPLEX_FLOATS;
            LocalTensor<float> rowPart = rowOutLocal[static_cast<uint32_t>(chunk) * CSYMV_REG_COMPLEX_FLOATS];
            LocalTensor<float> colPart = colOutLocal[static_cast<uint32_t>(chunk) * CSYMV_REG_COMPLEX_FLOATS];
            if (rowTile == colTile) {
                Add(rowPart, rowPart, colPart, CSYMV_REG_COMPLEX_FLOATS);
                PipeBarrier<PIPE_V>();
                DataCopy(workspaceGlobal_[rowOffset], rowPart, CSYMV_REG_COMPLEX_FLOATS);
            } else {
                const uint64_t colOffset =
                    (static_cast<uint64_t>(colTile) * tileCount_ + rowTile) * CSYMV_REG_COMPLEX_FLOATS;
                DataCopy(workspaceGlobal_[rowOffset], rowPart, CSYMV_REG_COMPLEX_FLOATS);
                DataCopy(workspaceGlobal_[colOffset], colPart, CSYMV_REG_COMPLEX_FLOATS);
            }
        }
        rowOutQueue_.FreeTensor(rowOutLocal);
        colOutQueue_.FreeTensor(colOutLocal);
    }

    __aicore__ inline void ProcessTile(uint32_t rowBlock, uint32_t colTile)
    {
        const uint32_t diagonalBlock = colTile / CSYMV_REG_ROW_CHUNKS;
        const bool hasDiagonal = rowBlock == diagonalBlock;
        const uint16_t diagonalChunk = static_cast<uint16_t>(colTile % CSYMV_REG_ROW_CHUNKS);
        const uint32_t rowCount =
            n_ - rowBlock * CSYMV_REG_RECT_ROWS < CSYMV_REG_RECT_ROWS ?
                n_ - rowBlock * CSYMV_REG_RECT_ROWS :
                CSYMV_REG_RECT_ROWS;
        const uint16_t validChunks = static_cast<uint16_t>((rowCount + CSYMV_REG_TILE - 1U) / CSYMV_REG_TILE);
        const uint32_t colCount =
            n_ - colTile * CSYMV_REG_TILE < CSYMV_REG_TILE ? n_ - colTile * CSYMV_REG_TILE : CSYMV_REG_TILE;
        const uint16_t firstChunk = hasDiagonal && !UPLO_IS_UPPER ? diagonalChunk : 0U;
        const uint16_t chunkCount = hasDiagonal ?
                                        (UPLO_IS_UPPER ? diagonalChunk + 1U : validChunks - diagonalChunk) :
                                        validChunks;
        const uint32_t rowBase = rowBlock * CSYMV_REG_RECT_ROWS;
        const uint32_t colBase = colTile * CSYMV_REG_TILE;
        CopyInputs(rowBase, colBase, rowCount, colCount);
        Compute(diagonalChunk, hasDiagonal, rowCount, colCount);
        CopyOutputs(rowBlock, colTile, firstChunk, chunkCount);
    }

    TQue<TPosition::VECIN, 1> aQueue_;
    TQue<TPosition::VECIN, 1> xRowQueue_;
    TQue<TPosition::VECIN, 1> xColQueue_;
    TQue<TPosition::VECOUT, 1> rowOutQueue_;
    TQue<TPosition::VECOUT, 1> colOutQueue_;
    GlobalTensor<float> aGlobal_;
    GlobalTensor<float> xGlobal_;
    GlobalTensor<float> workspaceGlobal_;
    uint32_t blockIdx_ = 0U;
    uint32_t n_ = 0U;
    uint32_t lda_ = 0U;
    uint32_t useCoreNum_ = 0U;
    uint32_t tileCount_ = 0U;
    uint32_t rowBlockCount_ = 0U;
    uint32_t totalTasks_ = 0U;
};

} // namespace

__simt_callee__ __aicore__ inline void CsymvStore(
    int64_t yIdx, float alphaReal, float alphaImag, float betaReal, float betaImag, float accReal, float accImag,
    float yReal, float yImag, bool betaIsOne, __gm__ float* yGm)
{
    const float productReal = alphaReal * accReal - alphaImag * accImag;
    const float productImag = alphaReal * accImag + alphaImag * accReal;
    if (betaIsOne) {
        yGm[2 * yIdx] = productReal + yReal;
        yGm[2 * yIdx + 1] = productImag + yImag;
    } else {
        yGm[2 * yIdx] = productReal + betaReal * yReal - betaImag * yImag;
        yGm[2 * yIdx + 1] = productImag + betaReal * yImag + betaImag * yReal;
    }
}

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CsymvSimtCompute(
    uint32_t n, uint32_t lda, float alphaReal, float alphaImag, float betaReal, float betaImag, int64_t incx,
    int64_t incy, uint32_t alphaIsDevice, uint32_t betaIsDevice, __gm__ const float* alphaGm, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ const float* betaGm, __gm__ float* yGm)
{
    if (alphaIsDevice != 0U) {
        alphaReal = alphaGm[0];
        alphaImag = alphaGm[1];
    }
    if (betaIsDevice != 0U) {
        betaReal = betaGm[0];
        betaImag = betaGm[1];
    }

    const bool alphaIsZero = alphaReal == 0.0f && alphaImag == 0.0f;
    const bool betaIsZero = betaReal == 0.0f && betaImag == 0.0f;
    const bool betaIsOne = betaReal == 1.0f && betaImag == 0.0f;
    if (alphaIsZero && betaIsOne) {
        return;
    }

    const int64_t n64 = static_cast<int64_t>(n);
    const int64_t lda64 = static_cast<int64_t>(lda);

    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        const int64_t row64 = static_cast<int64_t>(row);
        const int64_t yIdx = (incy >= 0) ? (row64 * incy) : ((n64 - 1 - row64) * (-incy));

        float yReal = 0.0f;
        float yImag = 0.0f;
        if (!betaIsZero) {
            yReal = yGm[2 * yIdx];
            yImag = yGm[2 * yIdx + 1];
        }

        float accReal = 0.0f;
        float accImag = 0.0f;
        if (!alphaIsZero) {
            for (uint32_t col = 0; col < n; ++col) {
                const int64_t col64 = static_cast<int64_t>(col);
                int64_t aIdx;
                if constexpr (UPLO_IS_UPPER) {
                    aIdx = (row <= col) ? (row64 + col64 * lda64) : (col64 + row64 * lda64);
                } else {
                    aIdx = (row >= col) ? (row64 + col64 * lda64) : (col64 + row64 * lda64);
                }
                const int64_t xIdx = (incx >= 0) ? (col64 * incx) : ((n64 - 1 - col64) * (-incx));
                const float aReal = aGm[2 * aIdx];
                const float aImag = aGm[2 * aIdx + 1];
                const float xReal = xGm[2 * xIdx];
                const float xImag = xGm[2 * xIdx + 1];
                accReal += aReal * xReal - aImag * xImag;
                accImag += aReal * xImag + aImag * xReal;
            }
        }

        CsymvStore(yIdx, alphaReal, alphaImag, betaReal, betaImag, accReal, accImag, yReal, yImag, betaIsOne, yGm);
    }
}

__simt_callee__ __aicore__ inline float CsymvWarpReduce(float value)
{
    for (uint32_t offset = 16U; offset > 0U; offset >>= 1U) {
        value += asc_shfl_down(value, offset, 32U);
    }
    return value;
}

template <bool UPLO_IS_UPPER>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CsymvSimtCooperative(
    uint32_t n, uint32_t lda, float alphaReal, float alphaImag, __gm__ const float* aGm, __gm__ const float* xGm,
    __gm__ float* yGm)
{
    const uint32_t laneId = threadIdx.x & 31U;
    const uint32_t warpId = threadIdx.x >> 5U;
    const uint32_t warpsPerBlock = blockDim.x >> 5U;
    const uint64_t lda64 = static_cast<uint64_t>(lda);

    for (uint32_t row = blockIdx.x * warpsPerBlock + warpId; row < n; row += gridDim.x * warpsPerBlock) {
        float localReal = 0.0f;
        float localImag = 0.0f;
        for (uint32_t col = laneId; col < n; col += 32U) {
            uint64_t aIdx;
            if constexpr (UPLO_IS_UPPER) {
                aIdx = (row <= col) ? static_cast<uint64_t>(row) + static_cast<uint64_t>(col) * lda64 :
                                      static_cast<uint64_t>(col) + static_cast<uint64_t>(row) * lda64;
            } else {
                aIdx = (row >= col) ? static_cast<uint64_t>(row) + static_cast<uint64_t>(col) * lda64 :
                                      static_cast<uint64_t>(col) + static_cast<uint64_t>(row) * lda64;
            }
            const float aReal = aGm[2U * aIdx];
            const float aImag = aGm[2U * aIdx + 1U];
            const float xReal = xGm[2U * col];
            const float xImag = xGm[2U * col + 1U];
            localReal += aReal * xReal - aImag * xImag;
            localImag += aReal * xImag + aImag * xReal;
        }

        localReal = CsymvWarpReduce(localReal);
        localImag = CsymvWarpReduce(localImag);
        if (laneId == 0U) {
            const float resultReal = alphaReal * localReal - alphaImag * localImag;
            const float resultImag = alphaReal * localImag + alphaImag * localReal;
            yGm[2U * row] = resultReal;
            yGm[2U * row + 1U] = resultImag;
        }
    }
}

__global__ __aicore__ void csymv_kernel(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y, GM_ADDR workSpace, const CsymvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<CsymvSimtCompute<true>>(
            dim3{tiling.nthreads, 1, 1}, tiling.n, tiling.lda, tiling.alphaReal, tiling.alphaImag, tiling.betaReal,
            tiling.betaImag, tiling.incx, tiling.incy, tiling.alphaIsDevice, tiling.betaIsDevice,
            reinterpret_cast<__gm__ const float*>(alpha), reinterpret_cast<__gm__ const float*>(a),
            reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ const float*>(beta),
            reinterpret_cast<__gm__ float*>(y));
    } else {
        asc_vf_call<CsymvSimtCompute<false>>(
            dim3{tiling.nthreads, 1, 1}, tiling.n, tiling.lda, tiling.alphaReal, tiling.alphaImag, tiling.betaReal,
            tiling.betaImag, tiling.incx, tiling.incy, tiling.alphaIsDevice, tiling.betaIsDevice,
            reinterpret_cast<__gm__ const float*>(alpha), reinterpret_cast<__gm__ const float*>(a),
            reinterpret_cast<__gm__ const float*>(x), reinterpret_cast<__gm__ const float*>(beta),
            reinterpret_cast<__gm__ float*>(y));
    }
}

__global__ __aicore__ void csymv_regtile_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR workspace, const CsymvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if (tiling.uplo == ACLBLAS_UPPER) {
        CsymvRegRectAiv<true> op;
        op.Init(
            reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(x),
            reinterpret_cast<__gm__ float*>(workspace), tiling, &pipe);
        op.Process();
    } else {
        CsymvRegRectAiv<false> op;
        op.Init(
            reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(x),
            reinterpret_cast<__gm__ float*>(workspace), tiling, &pipe);
        op.Process();
    }
}

__global__ __aicore__ void csymv_regtile_reduce_kernel(GM_ADDR workspace, GM_ADDR y, const CsymvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t tileCount = (tiling.n + CSYMV_REG_TILE - 1U) / CSYMV_REG_TILE;
    const uint32_t outputTile = GetBlockIdx();
    if (outputTile >= tileCount) {
        return;
    }

    TPipe pipe;
    TQue<TPosition::VECIN, 1> partialQueue;
    TQue<TPosition::VECOUT, 1> outQueue;
    const uint32_t partialFloats = tileCount * CSYMV_REG_COMPLEX_FLOATS;
    pipe.InitBuffer(partialQueue, 1, partialFloats * sizeof(float));
    pipe.InitBuffer(outQueue, 1, CSYMV_REG_COMPLEX_FLOATS * sizeof(float));
    GlobalTensor<float> workspaceGlobal;
    GlobalTensor<float> yGlobal;
    workspaceGlobal.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspace), static_cast<uint64_t>(tileCount) * partialFloats);
    yGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), static_cast<uint64_t>(tiling.n) * 2U);

    LocalTensor<float> partialLocal = partialQueue.AllocTensor<float>();
    DataCopy(partialLocal, workspaceGlobal[static_cast<uint64_t>(outputTile) * partialFloats], partialFloats);
    partialQueue.EnQue(partialLocal);
    partialLocal = partialQueue.DeQue<float>();
    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
    asc_vf_call<CsymvReduceRegTilesVf>(
        reinterpret_cast<__ubuf__ float*>(partialLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(outLocal.GetPhyAddr()), tileCount, tiling.alphaReal, tiling.alphaImag);
    outQueue.EnQue(outLocal);
    partialQueue.FreeTensor(partialLocal);
    outLocal = outQueue.DeQue<float>();
    const uint32_t outputCount =
        tiling.n - outputTile * CSYMV_REG_TILE < CSYMV_REG_TILE ?
            tiling.n - outputTile * CSYMV_REG_TILE :
            CSYMV_REG_TILE;
    DataCopyExtParams outputParams{
        1U, outputCount * 2U * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(yGlobal[static_cast<uint64_t>(outputTile) * CSYMV_REG_COMPLEX_FLOATS], outLocal, outputParams);
    outQueue.FreeTensor(outLocal);
}

__global__ __aicore__ void csymv_cooperative_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, const CsymvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.uplo == ACLBLAS_UPPER) {
        asc_vf_call<CsymvSimtCooperative<true>>(
            dim3{tiling.nthreads, 1, 1}, tiling.n, tiling.lda, tiling.alphaReal, tiling.alphaImag,
            reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(x),
            reinterpret_cast<__gm__ float*>(y));
    } else {
        asc_vf_call<CsymvSimtCooperative<false>>(
            dim3{tiling.nthreads, 1, 1}, tiling.n, tiling.lda, tiling.alphaReal, tiling.alphaImag,
            reinterpret_cast<__gm__ const float*>(a), reinterpret_cast<__gm__ const float*>(x),
            reinterpret_cast<__gm__ float*>(y));
    }
}

void csymv_kernel_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR x, GM_ADDR beta, GM_ADDR y, GM_ADDR workSpace, uint32_t numBlocks,
    const CsymvTilingData& tiling, void* stream)
{
    if (tiling.fastPath != 0U) {
        if (tiling.fastPath == 2U) {
            csymv_cooperative_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, tiling);
            return;
        }
        csymv_regtile_kernel<<<numBlocks, nullptr, stream>>>(a, x, workSpace, tiling);
        const uint32_t tileCount = (tiling.n + CSYMV_REG_TILE - 1U) / CSYMV_REG_TILE;
        csymv_regtile_reduce_kernel<<<tileCount, nullptr, stream>>>(workSpace, y, tiling);
        return;
    }
    csymv_kernel<<<numBlocks, nullptr, stream>>>(alpha, a, x, beta, y, workSpace, tiling);
}
