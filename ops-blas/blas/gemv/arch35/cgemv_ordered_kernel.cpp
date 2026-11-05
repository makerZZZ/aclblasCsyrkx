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
 * \file cgemv_ordered_kernel.cpp
 * \brief Ordered generic and N512 CGEMV paths.
 */

#include "cgemv_kernel_common.h"

using namespace AscendC;

// One full 64-row N512 tile.  All active SIMD lanes execute the same column
// sequence, while each lane owns one output row.  Keeping product formation
// and the sole accumulator update as separate register operations mirrors the
// ordered scalar recurrence without a split-K/finalize reduction.
__simd_vf__ inline void CgemvN512OrderedTileVf(
    uint32_t cols, uint32_t reset, __ubuf__ float* a, __ubuf__ float* x, __ubuf__ float* acc)
{
    Reg::RegTensor<float> ar, ai, xr, xi, tmp;
    Reg::RegTensor<float> accR, accI, productR, productI;
    Reg::MaskReg all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    if (reset != 0U) {
        Reg::Duplicate(accR, 0.0F, all);
        Reg::Duplicate(accI, 0.0F, all);
    } else {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(accR, accI, acc);
    }

    for (uint16_t col = 0; col < static_cast<uint16_t>(cols); ++col) {
        uint32_t aOffset = static_cast<uint32_t>(col) * CGEMV_N512_ROW_SLOT_FLOATS;
        uint32_t xOffset = static_cast<uint32_t>(col) * CGEMV_N512_COMPLEX_FLOATS;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(ar, ai, a + aOffset);
        Reg::Duplicate(xr, x[xOffset], all);
        Reg::Duplicate(xi, x[xOffset + 1U], all);

        Reg::Mul(productR, ar, xr, all);
        Reg::Mul(tmp, ai, xi, all);
        Reg::Sub(productR, productR, tmp, all);
        Reg::Add(accR, accR, productR, all);

        Reg::Mul(productI, ar, xi, all);
        Reg::Mul(tmp, ai, xr, all);
        Reg::Add(productI, productI, tmp, all);
        Reg::Add(accI, accI, productI, all);
    }

    Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(acc, accR, accI, all);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__aicore__ inline void CgemvN512OrderedRow(
    uint32_t row0, GlobalTensor<float>& aGm, GlobalTensor<float>& xGm, GlobalTensor<float>& yGm,
    LocalTensor<float>& aLocal, LocalTensor<float>& xLocal, LocalTensor<float>& accLocal)
{
    constexpr uint32_t aSrcStrideBytes = (CGEMV_N512_DIM - CGEMV_N512_ROWS_PER_BLOCK) * CGEMV_N512_COMPLEX_BYTES;
    uint32_t reset = 1U;
    for (uint32_t col = 0; col < CGEMV_N512_DIM; col += CGEMV_N512_TILE_COLS) {
        uint64_t aOffset = CGEMV_N512_COMPLEX_FLOATS * (static_cast<uint64_t>(col) * CGEMV_N512_DIM + row0);
        DataCopyPad(
            aLocal, aGm[aOffset],
            DataCopyExtParams{
                static_cast<uint16_t>(CGEMV_N512_TILE_COLS), CGEMV_N512_ROW_SLOT_BYTES,
                static_cast<int64_t>(aSrcStrideBytes), 0, 0},
            DataCopyPadExtParams<float>{false, 0, 0, 0});
        DataCopyPad(
            xLocal, xGm[static_cast<uint64_t>(col) * CGEMV_N512_COMPLEX_FLOATS],
            DataCopyExtParams{1, CGEMV_N512_X_TILE_BYTES, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        asc_vf_call<CgemvN512OrderedTileVf>(
            CGEMV_N512_TILE_COLS, reset, reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(accLocal.GetPhyAddr()));
        reset = 0U;

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(
        yGm[static_cast<uint64_t>(row0) * CGEMV_N512_COMPLEX_FLOATS], accLocal,
        DataCopyExtParams{1, CGEMV_N512_ROW_SLOT_BYTES, 0, 0, 0});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

__aicore__ inline void CgemvN512OrderedRun(GM_ADDR a, GM_ADDR x, GM_ADDR y, TPipe& pipe)
{
    uint32_t rowTile = GetBlockIdx();
    if (rowTile >= CGEMV_N512_ROW_BLOCKS) {
        return;
    }
    uint32_t row0 = rowTile * CGEMV_N512_ROWS_PER_BLOCK;
    GlobalTensor<float> aGm;
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    TBuf<QuePosition::VECCALC> aBuf;
    TBuf<QuePosition::VECCALC> xBuf;
    TBuf<QuePosition::VECCALC> accBuf;
    pipe.InitBuffer(aBuf, CGEMV_N512_A_TILE_BYTES);
    pipe.InitBuffer(xBuf, CGEMV_N512_X_TILE_BYTES);
    pipe.InitBuffer(accBuf, CGEMV_N512_ACC_BYTES);
    LocalTensor<float> aLocal = aBuf.Get<float>();
    LocalTensor<float> xLocal = xBuf.Get<float>();
    LocalTensor<float> accLocal = accBuf.Get<float>();

    CgemvN512OrderedRow(row0, aGm, xGm, yGm, aLocal, xLocal, accLocal);
}

// ==========================================================================
//  GM path — trans=N: grid-stride over output rows (general strides fallback)
// ==========================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgemvNGm(
    uint32_t m, uint32_t n, uint32_t lda, float alphaR, float alphaI, float betaR, float betaI, uint32_t betaIsZero,
    int64_t incx, int64_t incy, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    int64_t lda64 = static_cast<int64_t>(lda);
    int64_t n64 = static_cast<int64_t>(n);
    int64_t m64 = static_cast<int64_t>(m);

    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < m; row += gridDim.x * blockDim.x) {
        int64_t row64 = static_cast<int64_t>(row);
        int64_t yIdx = CgemvStridedIdx(row64, m64, incy) * 2;
        float accR;
        float accI;
        CgemvNInitialValue(yGm, yIdx, betaR, betaI, betaIsZero, accR, accI);

        for (uint32_t col = 0; col < n; ++col) {
            int64_t col64 = static_cast<int64_t>(col);
            // A stored column-major: A[row,col] = a[row + col*lda] (complex units)
            int64_t aIdx = (row64 + col64 * lda64) * 2;
            float2 av = make_float2(aGm[aIdx], aGm[aIdx + 1]);

            int64_t xIdx = CgemvStridedIdx(col64, n64, incx) * 2;
            float2 xv = make_float2(xGm[xIdx], xGm[xIdx + 1]);

            // Netlib order: temp = alpha*x(j); y(i) += temp*A(i,j).
            float2 temp = CgemvNScaledX(xv, alphaR, alphaI);
            CgemvCmla<false>(temp, av, accR, accI);
        }

        yGm[yIdx] = accR;
        yGm[yIdx + 1] = accI;
    }
}

// ==========================================================================
//  GM path — trans=T/C: grid-stride over output columns (general strides fallback)
//  IS_CONJ=true conjugates A elements (A^H path)
// ==========================================================================
template <bool IS_CONJ>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgemvTGm(
    uint32_t m, uint32_t n, uint32_t lda, float alphaR, float alphaI, float betaR, float betaI, uint32_t betaIsZero,
    int64_t incx, int64_t incy, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    int64_t lda64 = static_cast<int64_t>(lda);
    int64_t n64 = static_cast<int64_t>(n);
    int64_t m64 = static_cast<int64_t>(m);

    for (uint32_t col = blockIdx.x * blockDim.x + threadIdx.x; col < n; col += gridDim.x * blockDim.x) {
        float accR = 0.0f;
        float accI = 0.0f;
        int64_t col64 = static_cast<int64_t>(col);

        for (uint32_t row = 0; row < m; ++row) {
            int64_t row64 = static_cast<int64_t>(row);
            int64_t aIdx = (row64 + col64 * lda64) * 2;
            float2 av = make_float2(aGm[aIdx], aGm[aIdx + 1]);

            // x has m elements for trans=T/C
            int64_t xIdx = CgemvStridedIdx(row64, m64, incx) * 2;
            float2 xv = make_float2(xGm[xIdx], xGm[xIdx + 1]);

            CgemvCmla<IS_CONJ>(av, xv, accR, accI);
        }

        // y has n elements for trans=T/C
        int64_t yIdx = CgemvStridedIdx(col64, n64, incy) * 2;
        CgemvWriteBack(yGm, yIdx, accR, accI, alphaR, alphaI, betaR, betaI, betaIsZero);
    }
}

// ==========================================================================
//  UB path — trans=N (incx == 1): x cached in UB, per-thread row walk.
//  Fallback for shapes outside the slab kernels' coverage: the handle workspace
//  is too small for colChunks*m partials (e.g. m >= ~75k with the default 32 MiB),
//  or lda*n >= 2^32. Exercised by the WorkspaceFallbackN test.
// ==========================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgemvNUb(
    uint32_t m, uint32_t n, uint32_t lda, float alphaR, float alphaI, float betaR, float betaI, uint32_t betaIsZero,
    int64_t incy, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    __ubuf__ float xUb[CGEMV_UB_X_FLOATS];
    int64_t m64 = static_cast<int64_t>(m);
    int64_t stepF = static_cast<int64_t>(lda) * 2;

    uint32_t xFloats = n * 2;
    for (uint32_t i = threadIdx.x; i < xFloats; i += blockDim.x) {
        xUb[i] = xGm[i];
    }
    asc_syncthreads();
    __ubuf__ const float2* xUb2 = reinterpret_cast<__ubuf__ const float2*>(xUb);

    for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < m; row += gridDim.x * blockDim.x) {
        int64_t yIdx = CgemvStridedIdx(static_cast<int64_t>(row), m64, incy) * 2;
        float accR;
        float accI;
        CgemvNInitialValue(yGm, yIdx, betaR, betaI, betaIsZero, accR, accI);
        int64_t aOff = static_cast<int64_t>(row) * 2;

        for (uint32_t col = 0; col < n; ++col) {
            float2 av = *reinterpret_cast<__gm__ const float2*>(aGm + aOff);
            float2 temp = CgemvNScaledX(xUb2[col], alphaR, alphaI);
            CgemvCmla<false>(temp, av, accR, accI);
            aOff += stepF;
        }

        yGm[yIdx] = accR;
        yGm[yIdx + 1] = accI;
    }
}

// ==========================================================================
//  UB path — trans=T/C (incx == 1): x cached in UB, per-thread column walk.
//  Fallback for shapes outside the slab kernels' coverage: only lda*n >= 2^32
//  (>= 32 GiB of A) reaches it with incx == 1, so it is kept simple and
//  mirrors the tested CgemvNUb / CgemvTGm code.
// ==========================================================================
template <bool IS_CONJ>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgemvTUb(
    uint32_t m, uint32_t n, uint32_t lda, float alphaR, float alphaI, float betaR, float betaI, uint32_t betaIsZero,
    int64_t incy, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    __ubuf__ float xUb[CGEMV_UB_X_FLOATS];
    int64_t n64 = static_cast<int64_t>(n);
    int64_t lda64 = static_cast<int64_t>(lda);

    uint32_t xFloats = m * 2;
    for (uint32_t i = threadIdx.x; i < xFloats; i += blockDim.x) {
        xUb[i] = xGm[i];
    }
    asc_syncthreads();
    __ubuf__ const float2* xUb2 = reinterpret_cast<__ubuf__ const float2*>(xUb);

    for (uint32_t col = blockIdx.x * blockDim.x + threadIdx.x; col < n; col += gridDim.x * blockDim.x) {
        float accR = 0.0f;
        float accI = 0.0f;
        int64_t aOff = static_cast<int64_t>(col) * lda64 * 2;

        for (uint32_t row = 0; row < m; ++row) {
            float2 av = *reinterpret_cast<__gm__ const float2*>(aGm + aOff);
            CgemvCmla<IS_CONJ>(av, xUb2[row], accR, accI);
            aOff += 2;
        }

        int64_t yIdx = CgemvStridedIdx(static_cast<int64_t>(col), n64, incy) * 2;
        CgemvWriteBack(yGm, yIdx, accR, accI, alphaR, alphaI, betaR, betaI, betaIsZero);
    }
}

__global__ __aicore__ void cgemv_n512_ordered_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgemvN512OrderedRun(a, x, y, pipe);
}

__global__ __aicore__ void cgemv_kernel(GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, CgemvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);

    uint32_t outDim = (tiling.trans == 0) ? tiling.m : tiling.n;
    uint32_t xDim = (tiling.trans == 0) ? tiling.n : tiling.m;

    // UB path requires incx==1 and x vector fits in UB and enough work
    bool useUb = (tiling.incx == 1) && (xDim * 2 <= CGEMV_UB_X_FLOATS) && (outDim >= CGEMV_UB_MIN_OUT);

    if (tiling.trans == 0) {
        if (useUb) {
            asc_vf_call<CgemvNUb>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alphaR, tiling.alphaI,
                tiling.betaR, tiling.betaI, tiling.betaIsZero, tiling.incy, aGm, xGm, yGm);
        } else {
            asc_vf_call<CgemvNGm>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alphaR, tiling.alphaI,
                tiling.betaR, tiling.betaI, tiling.betaIsZero, tiling.incx, tiling.incy, aGm, xGm, yGm);
        }
    } else if (tiling.trans == 1) {
        if (useUb) {
            asc_vf_call<CgemvTUb<false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alphaR, tiling.alphaI,
                tiling.betaR, tiling.betaI, tiling.betaIsZero, tiling.incy, aGm, xGm, yGm);
        } else {
            asc_vf_call<CgemvTGm<false>>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alphaR, tiling.alphaI,
                tiling.betaR, tiling.betaI, tiling.betaIsZero, tiling.incx, tiling.incy, aGm, xGm, yGm);
        }
    } else {
        if (useUb) {
            asc_vf_call<CgemvTUb<true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alphaR, tiling.alphaI,
                tiling.betaR, tiling.betaI, tiling.betaIsZero, tiling.incy, aGm, xGm, yGm);
        } else {
            asc_vf_call<CgemvTGm<true>>(
                dim3{tiling.numThreads, 1, 1}, tiling.m, tiling.n, tiling.lda, tiling.alphaR, tiling.alphaI,
                tiling.betaR, tiling.betaI, tiling.betaIsZero, tiling.incx, tiling.incy, aGm, xGm, yGm);
        }
    }
}

void CgemvLaunchOrderedN512(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    cgemv_n512_ordered_kernel<<<(CGEMV_N512_ROW_BLOCKS), nullptr, stream>>>(a, x, y);
}

void CgemvLaunchOrdered(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const CgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    cgemv_kernel<<<(numBlocks), nullptr, stream>>>(a, x, y, workSpace, tiling);
}
