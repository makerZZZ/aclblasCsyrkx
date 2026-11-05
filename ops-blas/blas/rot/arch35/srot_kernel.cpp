/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


// Two separate kernel entries selected by tilingKey in srot_kernel_do:
//   tilingKey == 0: srot_aiv_kernel  — contiguous path (incx==1 && incy==1) -> SIMD membase
//                   DataCopy/DataCopyPad GM<->UB, paired vector compute in UB
//   tilingKey == 1: srot_simt_kernel — strided path (any other incx/incy)   -> SIMT
//                   multi-core grid-stride loop on GM (numBlocks cores in parallel)
// In-place overwrite avoidance:
//   - contiguous: original x/y are held in the VECIN input queues during the
//     whole Compute stage (only freed at the end of Compute), so both new_x and
//     new_y can still read the originals. The new values are written to separate
//     VECOUT queues (outNewX/outNewY) and copied back to the original GM in
//     CopyOut. No staging buffer is overwritten before its consumer finishes.
//   - strided: each thread loads xi/yi into registers, computes both new values,
//     then writes back; the originals survive in registers until both stores issue.

#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "srot_kernel.h"

using namespace AscendC;

// ==========================================================================
// Contiguous path: SIMD membase operator class
// ==========================================================================
class SrotAIV {
public:
    __aicore__ inline SrotAIV() {}
    __aicore__ inline void Init(__gm__ float* x, __gm__ float* y, __gm__ float* cPtr, __gm__ float* sPtr,
                                const SrotTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t curOffset, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t dataCount);
    __aicore__ inline void CopyOut(uint32_t curOffset, uint32_t dataCount);

    GlobalTensor<float> xGM_;
    GlobalTensor<float> yGM_;
    TQue<TPosition::VECIN, 1> inQueueX_;    // MTE2 input for x (EnQue signals MTE2 done)
    TQue<TPosition::VECIN, 1> inQueueY_;    // MTE2 input for y (EnQue signals MTE2 done)
    TQue<TPosition::VECOUT, 1> outNewX_;    // Vector output for new x (EnQue signals Vector done)
    TQue<TPosition::VECOUT, 1> outNewY_;    // Vector output for new y (EnQue signals Vector done)
    TBuf<TPosition::VECCALC> workBuf_;      // pure-Vector scratch (round(c*y)), no MTE traffic
    SrotTilingData tiling_;
    float cosValue_;
    float sinValue_;
    uint32_t blockIdx_;
    uint32_t myOffset_;
    uint32_t myCount_;
};

__aicore__ inline void SrotAIV::Init(__gm__ float* x, __gm__ float* y, __gm__ float* cPtr, __gm__ float* sPtr,
                                     const SrotTilingData& tiling, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    tiling_ = tiling;
    // c/s source: device pointer -> read scalar from GM; host pointer -> use tiling scalar.
    // Device path avoids any host dereference / D2H sync; host path keeps scalar broadcast.
    cosValue_ = tiling.cIsDevice ? *cPtr : tiling.cosValue;
    sinValue_ = tiling.sIsDevice ? *sPtr : tiling.sinValue;

    // Even split: base perCoreN elements, front `remainder` cores each get one extra.
    // Offset = blockIdx*perCoreN + (extra ones already issued by earlier cores), so the
    // per-core ranges butt exactly with no gap/overlap regardless of alignment.
    uint32_t perCore = tiling_.perCoreN;
    myOffset_ = blockIdx_ * perCore;
    if (blockIdx_ < tiling_.remainder) {
        myOffset_ += blockIdx_;
        perCore += 1;
    } else {
        myOffset_ += tiling_.remainder;
    }
    myCount_ = perCore;

    xGM_.SetGlobalBuffer(x, tiling_.totalN);
    yGM_.SetGlobalBuffer(y, tiling_.totalN);

    uint32_t bufSize = tiling_.tileSize * sizeof(float);
    pipe->InitBuffer(inQueueX_, 1, bufSize);
    pipe->InitBuffer(inQueueY_, 1, bufSize);
    pipe->InitBuffer(outNewX_, 1, bufSize);
    pipe->InitBuffer(outNewY_, 1, bufSize);
    pipe->InitBuffer(workBuf_, bufSize);
}

// ==========================================================================
// CopyIn: MTE2 GM -> UB. EnQue at the end is the MTE2 -> Vector sync point.
// ==========================================================================
__aicore__ inline void SrotAIV::CopyIn(uint32_t curOffset, uint32_t dataCount)
{
    // Pure DataCopyPad: the GM side needs only 1-byte alignment, so the per-core start
    // offset (non-block-aligned under even split) and any element count are both fine.
    // isPad=false lets the framework auto-pad the tail in UB; the Vector stage consumes
    // only `dataCount` elements, so the dummy tail is never read.
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

    LocalTensor<float> xLocal = inQueueX_.AllocTensor<float>();
    LocalTensor<float> yLocal = inQueueY_.AllocTensor<float>();
    DataCopyPad(xLocal, xGM_[curOffset], copyParams, padParams);
    DataCopyPad(yLocal, yGM_[curOffset], copyParams, padParams);
    // EnQue marks both tiles ready and signals Vector that MTE2 has finished.
    inQueueX_.EnQue(xLocal);
    inQueueY_.EnQue(yLocal);
}

// ==========================================================================
// Compute: Vector stage. DeQue at the start waits for MTE2; EnQue at the end
// signals MTE3. Original x/y are held for the whole stage (freed only at the
// bottom) so both new_x and new_y can read the originals -> in-place safe.
// ==========================================================================
__aicore__ inline void SrotAIV::Compute(uint32_t dataCount)
{
    // DeQue blocks until the MTE2 copies from CopyIn have landed in UB.
    LocalTensor<float> xLocal = inQueueX_.DeQue<float>();
    LocalTensor<float> yLocal = inQueueY_.DeQue<float>();
    LocalTensor<float> newX = outNewX_.AllocTensor<float>();
    LocalTensor<float> newY = outNewY_.AllocTensor<float>();
    LocalTensor<float> work = workBuf_.Get<float>();
    int32_t count = static_cast<int32_t>(dataCount);

    // Givens rotation, two separate legs using the ORIGINAL x and y:
    //   new_x = c*x + s*y
    //   new_y = c*y - s*x
    // Both legs read xLocal/yLocal, which stay live until the end of Compute.

    // ----- new_x = c*x + s*y (Axpy FMA: tmp = c*x; tmp += s*y) -----
    Muls(newX, xLocal, cosValue_, count);    // newX = round(c*x)
    Axpy(newX, yLocal, sinValue_, count);    // newX = round(c*x) + s*y (FMA, single rounding on the addend)

    // ----- new_y = c*y - s*x (per-step rounding, Muls + Muls + Add) -----
    // Use per-step rounding (Muls + Muls + Add) instead of Axpy's FMA: when c==s
    // (45° rotation) and x≈y the result cancels toward 0, and FMA's unrounded c*y
    // leg diverges from the OpenBLAS reference (which rounds each step), inflating
    // the relative error past the FP32 gate. workBuf holds the rounded c*y; the two
    // rounded legs are then added.
    Muls(newY, xLocal, -sinValue_, count);   // newY = round(-s*x)
    Muls(work, yLocal, cosValue_, count);    // work = round(c*y)
    Add(newY, newY, work, count);            // newY = round(c*y - s*x)

    // EnQue signals MTE3 that Vector is done with these outputs.
    outNewX_.EnQue(newX);
    outNewY_.EnQue(newY);
    // Original x/y released only after both new values are computed -> in-place safe.
    inQueueX_.FreeTensor(xLocal);
    inQueueY_.FreeTensor(yLocal);
}

// ==========================================================================
// CopyOut: MTE3 UB -> GM. DeQue at the start waits for Vector; in-place write
// back to the original xGM_/yGM_.
// ==========================================================================
__aicore__ inline void SrotAIV::CopyOut(uint32_t curOffset, uint32_t dataCount)
{
    // Pure DataCopyPad (UB->GM): writes exactly dataCount floats to GM and auto-strips
    // the in-UB dummy tail. GM side needs only 1-byte alignment.
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(float)), 0, 0, 0};

    // DeQue blocks until the Vector compute on these outputs has finished.
    LocalTensor<float> newX = outNewX_.DeQue<float>();
    LocalTensor<float> newY = outNewY_.DeQue<float>();
    DataCopyPad(xGM_[curOffset], newX, copyParams);
    DataCopyPad(yGM_[curOffset], newY, copyParams);
    outNewX_.FreeTensor(newX);
    outNewY_.FreeTensor(newY);
}

__aicore__ inline void SrotAIV::Process()
{
    if (myCount_ == 0) {
        return;
    }

    uint32_t tileLoop = myCount_ / tiling_.tileSize;
    uint32_t tileTail = myCount_ % tiling_.tileSize;
    uint32_t curOffset = myOffset_;

    for (uint32_t i = 0; i < tileLoop; i++) {
        CopyIn(curOffset, tiling_.tileSize);
        Compute(tiling_.tileSize);
        CopyOut(curOffset, tiling_.tileSize);
        curOffset += tiling_.tileSize;
    }
    if (tileTail > 0) {
        CopyIn(curOffset, tileTail);
        Compute(tileTail);
        CopyOut(curOffset, tileTail);
    }
}

// ==========================================================================
// Strided path: SIMT compute function
//
// Multi-core grid-stride loop: the host launches the SIMT kernel on numBlocks
// cores (>= 1). Every thread in every block walks the N elements with a
// grid-stride of (block_num * blockDim.x), starting at
// (blockIdx.x * blockDim.x + threadIdx.x). blockDim.x = tiling.nthreads, sized by
// the host to the per-core element count (rounded up to SIMT_MIN_THREAD_NUM,
// capped at SIMT_MAX_THREAD_NUM) so a small n no longer launches the full
// SIMT_MAX_THREAD_NUM only to retire most threads. LAUNCH_BOUND stays at the max
// for register allocation; the runtime launch may use any blockDim.x <= it.
// ==========================================================================

// Zero-stride serial fallback (R2): netlib srot.f with incx==0 holds IX constant across the
// whole loop, so each of the N iterations reuses the element just written by the previous
// iteration (serial accumulation). A multi-thread grid-stride loop would read the same
// original value concurrently and race on the single address, diverging from netlib.
// When zeroIncX or zeroIncY is set, only block0.thread0 runs the N iterations in strict
// netlib order (IX/IY recomputed exactly as the reference: start anchor + i*inc, inc
// possibly 0), matching the serial accumulation bit-for-bit. All other threads/blocks
// return immediately so the reused address is touched by exactly one program-order stream.
// The non-zero stride stays on the parallel grid-stride path; the zero-stride side is
// simply re-read/re-written each iteration.
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM)
inline void srot_simt_compute(uint32_t n, uint32_t absIncX, uint32_t absIncY, int32_t negX, int32_t negY,
                              int32_t zeroIncX, int32_t zeroIncY, float c, float s, __gm__ float* xGm,
                              __gm__ float* yGm)
{
    // Zero stride (either side) forces single-thread serial execution to reproduce netlib's
    // serial accumulation over the reused address. absInc is set to 1 for the zero side so the
    // address math stays valid (the index is forced to element 0 by jx/jy below).
    // Multi-core guard: with numBlocks cores in flight, the race would span blocks too, so the
    // guard must retire every thread except block0.thread0.
    if ((zeroIncX != 0) || (zeroIncY != 0)) {
        if ((blockIdx.x != 0) || (threadIdx.x != 0)) {
            return;
        }
        for (uint32_t i = 0; i < n; i++) {
            uint32_t jx = (zeroIncX != 0) ? 0U : ((negX != 0) ? (n - 1 - i) : i);
            uint32_t jy = (zeroIncY != 0) ? 0U : ((negY != 0) ? (n - 1 - i) : i);
            uint32_t xIdx = jx * absIncX;
            uint32_t yIdx = jy * absIncY;
            float xi = xGm[xIdx];
            float yi = yGm[yIdx];
            float newX = c * xi + s * yi;
            float newY = c * yi - s * xi;
            xGm[xIdx] = newX;
            yGm[yIdx] = newY;
        }
        return;
    }

    // Grid-stride loop: thread (blockIdx.x*blockDim.x + threadIdx.x) starts at its global
    // index and strides by the total number of threads in the grid (block_num*blockDim.x).
    // Each thread handles disjoint elements across the whole N range; with numBlocks cores
    // this is genuine multi-core parallelism (vs the old single-block threadIdx.x loop).
    uint32_t tidGlobal = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t gridStride = block_num * blockDim.x;
    for (uint32_t i = tidGlobal; i < n; i += gridStride) {
        uint32_t jx = (negX != 0) ? (n - 1 - i) : i;
        uint32_t jy = (negY != 0) ? (n - 1 - i) : i;
        uint32_t xIdx = jx * absIncX;
        uint32_t yIdx = jy * absIncY;
        float xi = xGm[xIdx];
        float yi = yGm[yIdx];
        float newX = c * xi + s * yi;
        float newY = c * yi - s * xi;
        xGm[xIdx] = newX;
        yGm[yIdx] = newY;
    }
}

// ==========================================================================
// Kernel entries — two separate binaries so SIMD (TPipe/DataCopy) and SIMT
// (asc_vf_call) code never coexist in one kernel image. 
// ==========================================================================

// Contiguous path: SIMD membase (incx==1 && incy==1)
extern "C" __global__ __aicore__ void srot_aiv_kernel(__gm__ float* x, __gm__ float* y, __gm__ float* cPtr,
                                                      __gm__ float* sPtr, SrotTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SrotAIV op;
    op.Init(x, y, cPtr, sPtr, tiling, &pipe);
    op.Process();
}

// Strided path: SIMT (any other incx/incy combination)
extern "C" __global__ __aicore__ void srot_simt_kernel(__gm__ float* x, __gm__ float* y, __gm__ float* cPtr,
                                                       __gm__ float* sPtr, SrotTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ float* xGm = x;
    __gm__ float* yGm = y;
    uint32_t n = tiling.totalN;
    int32_t incx = tiling.incx;
    int32_t incy = tiling.incy;
    int32_t zeroIncX = (incx == 0) ? 1 : 0;
    int32_t zeroIncY = (incy == 0) ? 1 : 0;
    // |incx|/|incy| computed via int64 to avoid signed-negation UB when incx == INT32_MIN.
    uint32_t absIncX = static_cast<uint32_t>(incx < 0 ? -static_cast<int64_t>(incx) : static_cast<int64_t>(incx));
    uint32_t absIncY = static_cast<uint32_t>(incy < 0 ? -static_cast<int64_t>(incy) : static_cast<int64_t>(incy));
    if (absIncX == 0) {
        absIncX = 1; // inc==0 reuses element 0; a single stride covers it
    }
    if (absIncY == 0) {
        absIncY = 1;
    }
    int32_t negX = (incx < 0) ? 1 : 0;
    int32_t negY = (incy < 0) ? 1 : 0;
    // c/s source: device pointer -> read scalar from GM; host pointer -> use tiling scalar.
    float c = tiling.cIsDevice ? *cPtr : tiling.cosValue;
    float s = tiling.sIsDevice ? *sPtr : tiling.sinValue;
    asc_vf_call<srot_simt_compute>(dim3{tiling.nthreads, 1, 1}, n, absIncX, absIncY, negX, negY, zeroIncX,
                                   zeroIncY, c, s, xGm, yGm);
}

// ==========================================================================
// Host-side kernel launcher
// ==========================================================================
void srot_kernel_do(float* x, float* y, float* cPtr, float* sPtr, uint32_t numBlocks,
                    const SrotTilingData& tiling, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    if (tiling.tilingKey == 0) {
        srot_aiv_kernel<<<numBlocks, nullptr, aclStream>>>(x, y, cPtr, sPtr, tiling);
    } else {
        srot_simt_kernel<<<numBlocks, nullptr, aclStream>>>(x, y, cPtr, sPtr, tiling);
    }
}