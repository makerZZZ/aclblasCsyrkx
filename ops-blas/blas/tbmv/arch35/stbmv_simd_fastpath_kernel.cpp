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
 * \file stbmv_simd_fastpath_kernel.cpp
 * \brief single-precision triangular band matrix-vector multiply SIMD fastpath kernels for ascend950
 */

#include <cstddef>
#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "stbmv_common.h"

using namespace AscendC;

static constexpr uint32_t STBMV_FAST_BUFFER_NUM = 1;
static constexpr uint32_t STBMV_BYTES_PER_FLOAT = 4;
static constexpr uint32_t STBMV_ELEMENTS_PER_BLOCK = 8;

class StbmvColumnSimdFastAiv {
public:
    __aicore__ inline void Init(
        __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm, const StbmvFastTilingData& tdata)
    {
        blockIdx = GetBlockIdx();
        n = tdata.n;
        k = tdata.k;
        lda = tdata.lda;
        uplo = tdata.uplo;
        useCoreNum = tdata.useCoreNum;

        aGlobal.SetGlobalBuffer(const_cast<__gm__ float*>(aGm), static_cast<uint64_t>(lda) * n);
        xGlobal.SetGlobalBuffer(const_cast<__gm__ float*>(xGm), n);
        yGlobal.SetGlobalBuffer(yGm, n);

        pipe.InitBuffer(aQueue, STBMV_FAST_BUFFER_NUM, STBMV_FAST_TILE_FLOATS * sizeof(float));
        pipe.InitBuffer(yQueue, STBMV_FAST_BUFFER_NUM, STBMV_FAST_TILE_FLOATS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (useCoreNum == 0 || blockIdx >= useCoreNum) {
            return;
        }

        SetAtomicAdd<float>();
        for (uint32_t col = blockIdx; col < n; col += useCoreNum) {
            uint32_t rowStart = 0;
            uint32_t bandStart = 0;
            uint32_t segmentLen = 0;
            GetColumnRange(col, rowStart, bandStart, segmentLen);
            float xValue = xGlobal.GetValue(col);

            for (uint32_t offset = 0; offset < segmentLen; offset += STBMV_FAST_TILE_FLOATS) {
                uint32_t dataCount = STBMV_FAST_TILE_FLOATS;
                if (offset + dataCount > segmentLen) {
                    dataCount = segmentLen - offset;
                }
                CopyAIn(static_cast<uint64_t>(col) * lda + bandStart + offset, dataCount);
                Compute(xValue, dataCount);
                CopyYOut(rowStart + offset, dataCount);
            }
        }
        SetAtomicNone();
    }

private:
    __aicore__ inline bool IsBlockAligned(uint32_t dataCount) const
    {
        return (dataCount % STBMV_ELEMENTS_PER_BLOCK) == 0;
    }

    __aicore__ inline void GetColumnRange(
        uint32_t col, uint32_t& rowStart, uint32_t& bandStart, uint32_t& segmentLen) const
    {
        if (uplo == ACLBLAS_LOWER) {
            rowStart = col;
            bandStart = 0;
            uint64_t end = static_cast<uint64_t>(col) + k + 1U;
            uint32_t rowEnd = (end < n) ? static_cast<uint32_t>(end) : n;
            segmentLen = rowEnd - rowStart;
            return;
        }

        rowStart = (col >= k) ? (col - k) : 0U;
        uint32_t rowEnd = col + 1U;
        segmentLen = rowEnd - rowStart;
        bandStart = k - (col - rowStart);
    }

    __aicore__ inline void CopyAIn(uint64_t aOffset, uint32_t dataCount)
    {
        LocalTensor<float> aLocal = aQueue.AllocTensor<float>();
        if (IsBlockAligned(dataCount)) {
            DataCopy(aLocal, aGlobal[aOffset], dataCount);
        } else {
            uint8_t paddingNum = static_cast<uint8_t>(STBMV_ELEMENTS_PER_BLOCK - dataCount % STBMV_ELEMENTS_PER_BLOCK);
            DataCopyExtParams copyParams{1, dataCount * STBMV_BYTES_PER_FLOAT, 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, paddingNum, 0};
            DataCopyPad(aLocal, aGlobal[aOffset], copyParams, padParams);
        }
        aQueue.EnQue<float>(aLocal);
    }

    __aicore__ inline void Compute(float xValue, uint32_t dataCount)
    {
        LocalTensor<float> aLocal = aQueue.DeQue<float>();
        LocalTensor<float> yLocal = yQueue.AllocTensor<float>();

        Muls(yLocal, aLocal, xValue, dataCount);

        yQueue.EnQue<float>(yLocal);
        aQueue.FreeTensor(aLocal);
    }

    __aicore__ inline void CopyYOut(uint32_t row, uint32_t dataCount)
    {
        LocalTensor<float> yLocal = yQueue.DeQue<float>();
        if (IsBlockAligned(dataCount)) {
            DataCopy(yGlobal[row], yLocal, dataCount);
        } else {
            DataCopyExtParams copyParams{1, dataCount * STBMV_BYTES_PER_FLOAT, 0, 0, 0};
            DataCopyPad(yGlobal[row], yLocal, copyParams);
        }
        yQueue.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, STBMV_FAST_BUFFER_NUM> aQueue;
    TQue<QuePosition::VECOUT, STBMV_FAST_BUFFER_NUM> yQueue;
    GlobalTensor<float> aGlobal;
    GlobalTensor<float> xGlobal;
    GlobalTensor<float> yGlobal;
    uint32_t blockIdx = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t lda = 0;
    uint32_t uplo = 0;
    uint32_t useCoreNum = 0;
};

__simt_callee__ __aicore__ __inline__ int64_t StbmvFastVectorIndex(uint32_t idx, uint32_t n, int64_t incx)
{
    int64_t stride = incx;
    return (stride > 0) ? static_cast<int64_t>(idx) * stride :
                          static_cast<int64_t>(n - 1U - idx) * (-stride);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void StbmvFastCopyBack(
    uint32_t n, int64_t incx, __gm__ const float* midOutGm, __gm__ float* xGm)
{
    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {
        xGm[StbmvFastVectorIndex(row, n, incx)] = midOutGm[row];
    }
}

__global__ __aicore__ void stbmv_column_simd_fast_kernel(
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ uint8_t* workspaceGm, StbmvFastTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ float* yGm = reinterpret_cast<__gm__ float*>(workspaceGm);
    StbmvColumnSimdFastAiv op;
    op.Init(aGm, xGm, yGm, tdata);
    op.Process();
}

__global__ __aicore__ void stbmv_simd_fast_copy_kernel(
    __gm__ float* xGm, __gm__ const uint8_t* workspaceGm, StbmvTilingData tdata)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ const float* midOutGm = reinterpret_cast<__gm__ const float*>(workspaceGm);
    asc_vf_call<StbmvFastCopyBack>(dim3{tdata.numThreads, 1, 1}, tdata.n, tdata.incx, midOutGm, xGm);
}

int stbmv_arch35_simd_fastpath_kernel_do(
    const float* a, float* x, uint8_t* workspace, size_t workspaceSize, const StbmvTilingData& tilingData,
    const StbmvFastTilingData& fastTilingData, uint32_t numBlocks, void* stream)
{
    aclError aclRet = aclrtMemsetAsync(workspace, workspaceSize, 0, workspaceSize, static_cast<aclrtStream>(stream));
    if (aclRet != ACL_SUCCESS) {
        return static_cast<int>(aclRet);
    }
    stbmv_column_simd_fast_kernel<<<numBlocks, nullptr, stream>>>(a, x, workspace, fastTilingData);
    stbmv_simd_fast_copy_kernel<<<numBlocks, nullptr, stream>>>(x, workspace, tilingData);
    return ACL_SUCCESS;
}
