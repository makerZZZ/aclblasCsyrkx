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
#include <type_traits>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "rotex_kernel.h"

using namespace AscendC;

// BUFFER_NUM=1: 单缓冲流水线
// Givens 旋转为 SCALAR bound (SCALAR ~47%), VEC 计算密度低 (仅 Muls+Axpy),
// 双缓冲需减半 tileSize 增加 tile 数, 反而增大 SCALAR 开销, 实测无收益.
constexpr uint32_t ROTEX_BUFFER_NUM = 1;

// ==========================================================================
// SIMT VF path — for discrete strides (TilingKey 2)
// 通用模板: 适用于 float/half/bfloat16 对称类型 (executionType=FP32, c/s 统一切换到 float)
// ==========================================================================
template <typename XType>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void RotExSimt(
    uint32_t n, int64_t incx, int64_t incy, int64_t kx, int64_t ky,
    float c, float s,
    __gm__ XType* xGm, __gm__ XType* yGm)
{
    for (uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += gridDim.x * blockDim.x) {
        int64_t xi = kx + static_cast<int64_t>(idx) * incx;
        int64_t yi = ky + static_cast<int64_t>(idx) * incy;
        float xVal = static_cast<float>(xGm[xi]);
        float yVal = static_cast<float>(yGm[yi]);
        xGm[xi] = static_cast<XType>(c * xVal + s * yVal);
        yGm[yi] = static_cast<XType>((-s) * xVal + c * yVal);
    }
}

// Not supported on arch35: D (ACL_DOUBLE), C (ACL_COMPLEX64), Z (ACL_COMPLEX128) groups.
// Host-side validation returns ACLBLAS_STATUS_NOT_SUPPORTED for these types.
// arch35 AI Core does not support double precision or complex arithmetic.

// ==========================================================================
// SIMD S组 path — for continuous stride with DT_FLOAT executionType (Key 0)
// ==========================================================================
template <typename XType>
class RotExSimdKernelS {
public:
    __aicore__ inline RotExSimdKernelS() {}
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR x, GM_ADDR y,
                                 const RotExTilingData& t);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count);
    __aicore__ inline void Compute(uint32_t count);
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count);

    TPipe* pipe_;
    TQue<QuePosition::VECIN, ROTEX_BUFFER_NUM> xQueue;
    TQue<QuePosition::VECIN, ROTEX_BUFFER_NUM> yQueue;
    TQue<QuePosition::VECOUT, ROTEX_BUFFER_NUM> outXQueue;
    TQue<QuePosition::VECOUT, ROTEX_BUFFER_NUM> outYQueue;
    // 混合精度使用的 float 缓冲区
    TBuf<QuePosition::VECCALC> xFloatBuf_;
    TBuf<QuePosition::VECCALC> yFloatBuf_;
    TBuf<QuePosition::VECCALC> outXFloatBuf_;
    TBuf<QuePosition::VECCALC> outYFloatBuf_;

    GlobalTensor<XType> xGM;
    GlobalTensor<XType> yGM;
    float cVal;
    float sVal;

    uint32_t startOffset;
    uint32_t calCount;
    uint32_t tileSize;
    uint32_t tileNum;
    uint32_t remainderCount;
};

template <typename XType>
__aicore__ inline void RotExSimdKernelS<XType>::Init(
    TPipe* pipe, GM_ADDR x, GM_ADDR y, const RotExTilingData& t)
{
    pipe_ = pipe;
    cVal = t.cReal;
    sVal = t.sReal;
    tileSize = t.tileSize;

    // 模分发负载均衡 (同 rotm), 使用 host 预计算的 perCoreN/remainder
    uint32_t blockIdx = GetBlockIdx();
    uint32_t perCore = t.perCoreN;
    uint32_t remain = t.remainder;
    startOffset = blockIdx * perCore;
    if (blockIdx < remain) {
        startOffset += blockIdx;
        perCore += 1;
    } else {
        startOffset += remain;
    }
    calCount = (perCore > 0) ? perCore : 0;
    if (calCount == 0) return;

    // 设置 GM 缓存
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(x));
    yGM.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(y));

    // UB 队列初始化, 使用 host 预计算的 tileSize 计算队列大小
    if constexpr (std::is_same<XType, float>::value) {
        // 纯 FP32: 4 个等分 UB
        uint32_t ubPerQueue = tileSize * sizeof(float);
        pipe_->InitBuffer(xQueue, ROTEX_BUFFER_NUM, ubPerQueue);
        pipe_->InitBuffer(yQueue, ROTEX_BUFFER_NUM, ubPerQueue);
        pipe_->InitBuffer(outXQueue, ROTEX_BUFFER_NUM, ubPerQueue);
        pipe_->InitBuffer(outYQueue, ROTEX_BUFFER_NUM, ubPerQueue);
    } else {
        // 混合精度: 24B/element (xIn 2B + yIn 2B + 4*floatBuf 16B + outX 2B + outY 2B)
        uint32_t qSize = tileSize * sizeof(XType);
        uint32_t fSize = tileSize * sizeof(float);
        pipe_->InitBuffer(xQueue, ROTEX_BUFFER_NUM, qSize);
        pipe_->InitBuffer(yQueue, ROTEX_BUFFER_NUM, qSize);
        pipe_->InitBuffer(outXQueue, ROTEX_BUFFER_NUM, qSize);
        pipe_->InitBuffer(outYQueue, ROTEX_BUFFER_NUM, qSize);
        pipe_->InitBuffer(xFloatBuf_, fSize);
        pipe_->InitBuffer(yFloatBuf_, fSize);
        pipe_->InitBuffer(outXFloatBuf_, fSize);
        pipe_->InitBuffer(outYFloatBuf_, fSize);
    }

    tileNum = calCount / tileSize;
    remainderCount = calCount % tileSize;
}

template <typename XType>
__aicore__ inline void RotExSimdKernelS<XType>::CopyIn(uint32_t offset, uint32_t count)
{
    uint32_t nbytes = count * sizeof(XType);
    DataCopyExtParams ext{1, nbytes, 0, 0, 0};
    DataCopyPadExtParams<XType> padParams{false, 0, 0, 0};

    LocalTensor<XType> xLocal = xQueue.AllocTensor<XType>();
    DataCopyPad(xLocal, xGM[startOffset + offset], ext, padParams);
    xQueue.EnQue(xLocal);

    LocalTensor<XType> yLocal = yQueue.AllocTensor<XType>();
    DataCopyPad(yLocal, yGM[startOffset + offset], ext, padParams);
    yQueue.EnQue(yLocal);
}

template <typename XType>
__aicore__ inline void RotExSimdKernelS<XType>::Compute(uint32_t count)
{
    LocalTensor<XType> xLocal = xQueue.DeQue<XType>();
    LocalTensor<XType> yLocal = yQueue.DeQue<XType>();

    if constexpr (std::is_same<XType, float>::value) {
        // 纯 FP32: 直接使用 Muls + Axpy
        // x_out = c * x + s * y
        LocalTensor<float> outXLocal = outXQueue.AllocTensor<float>();
        Muls(outXLocal, xLocal, cVal, count);
        Axpy(outXLocal, yLocal, sVal, count);

        // y_out = -s * x + c * y (使用原始 xLocal 值)
        LocalTensor<float> outYLocal = outYQueue.AllocTensor<float>();
        Muls(outYLocal, xLocal, -sVal, count);
        Axpy(outYLocal, yLocal, cVal, count);

        outXQueue.EnQue(outXLocal);
        outYQueue.EnQue(outYLocal);
    } else {
        // 混合精度: XType -> FP32 -> 计算 -> XType
        // 使用 4 个独立 float 缓冲区
        LocalTensor<float> xFloat = xFloatBuf_.Get<float>();
        LocalTensor<float> yFloat = yFloatBuf_.Get<float>();
        LocalTensor<float> outXFloat = outXFloatBuf_.Get<float>();
        LocalTensor<float> outYFloat = outYFloatBuf_.Get<float>();

        // XType -> FP32 (CAST_NONE: 精度无损升转换)
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, count);
        Cast(yFloat, yLocal, RoundMode::CAST_NONE, count);

        // x_out = c * x + s * y (FP32 精度)
        Muls(outXFloat, xFloat, cVal, count);
        Axpy(outXFloat, yFloat, sVal, count);

        // y_out = -s * x + c * y (FP32 精度, 使用原始 xFloat)
        Muls(outYFloat, xFloat, -sVal, count);
        Axpy(outYFloat, yFloat, cVal, count);

        // FP32 -> XType (CAST_RINT: 四舍六入五成双)
        LocalTensor<XType> outXLocal = outXQueue.AllocTensor<XType>();
        LocalTensor<XType> outYLocal = outYQueue.AllocTensor<XType>();
        Cast(outXLocal, outXFloat, RoundMode::CAST_RINT, count);
        Cast(outYLocal, outYFloat, RoundMode::CAST_RINT, count);

        outXQueue.EnQue(outXLocal);
        outYQueue.EnQue(outYLocal);
    }

    xQueue.FreeTensor(xLocal);
    yQueue.FreeTensor(yLocal);
}

template <typename XType>
__aicore__ inline void RotExSimdKernelS<XType>::CopyOut(uint32_t offset, uint32_t count)
{
    uint32_t nbytes = count * sizeof(XType);
    DataCopyExtParams ext{1, nbytes, 0, 0, 0};

    LocalTensor<XType> outXLocal = outXQueue.DeQue<XType>();
    DataCopyPad(xGM[startOffset + offset], outXLocal, ext);
    outXQueue.FreeTensor(outXLocal);

    LocalTensor<XType> outYLocal = outYQueue.DeQue<XType>();
    DataCopyPad(yGM[startOffset + offset], outYLocal, ext);
    outYQueue.FreeTensor(outYLocal);
}

template <typename XType>
__aicore__ inline void RotExSimdKernelS<XType>::Process()
{
    if (calCount == 0) return;

    for (uint32_t i = 0; i < tileNum; i++) {
        uint32_t offset = i * tileSize;
        CopyIn(offset, tileSize);
        Compute(tileSize);
        CopyOut(offset, tileSize);
    }
    if (remainderCount > 0) {
        uint32_t offset = tileNum * tileSize;
        CopyIn(offset, remainderCount);
        Compute(remainderCount);
        CopyOut(offset, remainderCount);
    }
}

// ==========================================================================
// Kernel main entry
// ==========================================================================
extern "C" __global__ __aicore__ void rotex_kernel(GM_ADDR x, GM_ADDR y, RotExTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    uint32_t key = t.tilingKey;

    if (key == 0) {
        // SIMD S 组: DT_FLOAT executionType, 连续步长
        TPipe pipe;

        if (t.xType == DT_FLOAT) {
            RotExSimdKernelS<float> op;
            op.Init(&pipe, x, y, t);
            op.Process();
        } else if (t.xType == DT_FLOAT16) {
            RotExSimdKernelS<half> op;
            op.Init(&pipe, x, y, t);
            op.Process();
        } else {
            RotExSimdKernelS<bfloat16_t> op;
            op.Init(&pipe, x, y, t);
            op.Process();
        }
    } else {
        // SIMT 路径 (TilingKey 2): 离散步长, 对称类型 (xType==yType==csType)
        float cVal = t.cReal;
        float sVal = t.sReal;

        if (t.xType == DT_FLOAT) {
            asc_vf_call<RotExSimt<float>>(
                dim3{t.nthreads, 1, 1},
                static_cast<uint32_t>(t.n), static_cast<int64_t>(t.incx),
                static_cast<int64_t>(t.incy), t.kx, t.ky,
                cVal, sVal,
                reinterpret_cast<__gm__ float*>(x),
                reinterpret_cast<__gm__ float*>(y));
        } else if (t.xType == DT_FLOAT16) {
            asc_vf_call<RotExSimt<half>>(
                dim3{t.nthreads, 1, 1},
                static_cast<uint32_t>(t.n), static_cast<int64_t>(t.incx),
                static_cast<int64_t>(t.incy), t.kx, t.ky,
                cVal, sVal,
                reinterpret_cast<__gm__ half*>(x),
                reinterpret_cast<__gm__ half*>(y));
        } else {
            // xType = BF16
            asc_vf_call<RotExSimt<bfloat16_t>>(
                dim3{t.nthreads, 1, 1},
                static_cast<uint32_t>(t.n), static_cast<int64_t>(t.incx),
                static_cast<int64_t>(t.incy), t.kx, t.ky,
                cVal, sVal,
                reinterpret_cast<__gm__ bfloat16_t*>(x),
                reinterpret_cast<__gm__ bfloat16_t*>(y));
        }
    }
}

void rotex_kernel_do(GM_ADDR x, GM_ADDR y, const RotExTilingData& tilingData,
                     uint32_t numBlocks, void* stream)
{
    rotex_kernel<<<numBlocks, nullptr, stream>>>(x, y, tilingData);
}
