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
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "ctbsv_tiling_data.h"
#include "common/helper/kernel_constant.h"

#define CTBSV_ARITH_ATTR __simt_callee__ inline
#include "ctbsv_complex_arith.h"

using namespace AscendC;

constexpr uint32_t CTBSV_UB_X_COMPLEX = 4096;
constexpr uint32_t CTBSV_UB_PARTIALS = 64;

template <bool UPLO_IS_UPPER>
__simt_callee__ inline int64_t CtbsvSimtAColBase(uint32_t col, uint32_t k, uint32_t lda)
{
    const int64_t col64 = static_cast<int64_t>(col);
    const int64_t lda64 = static_cast<int64_t>(lda);
    if constexpr (UPLO_IS_UPPER) {
        return static_cast<int64_t>(k) - col64 + col64 * lda64;
    }
    return col64 * (lda64 - 1);
}

__simt_callee__ inline int64_t CtbsvSimtXOffset(uint32_t idx, uint32_t n, int32_t incx)
{
    if (incx >= 0) {
        return static_cast<int64_t>(idx) * static_cast<int64_t>(incx);
    }
    return static_cast<int64_t>(n - 1 - idx) * static_cast<int64_t>(-incx);
}

__simt_callee__ inline void CtbsvSimtLoadX(
    uint32_t n, int32_t incx, __gm__ aclblasComplex* xGm, __ubuf__ float* xUb)
{
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
        int64_t off = CtbsvSimtXOffset(j, n, incx);
        const uint32_t dst = j * 2U;
        xUb[dst] = xGm[off].real;
        xUb[dst + 1U] = xGm[off].imag;
    }
    asc_syncthreads();
}

__simt_callee__ inline void CtbsvSimtStoreX(
    uint32_t n, int32_t incx, __gm__ aclblasComplex* xGm, __ubuf__ float* xUb)
{
    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
        int64_t off = CtbsvSimtXOffset(j, n, incx);
        const uint32_t src = j * 2U;
        xGm[off].real = xUb[src];
        xGm[off].imag = xUb[src + 1U];
    }
    asc_syncthreads();
}

__simt_callee__ inline void CtbsvSimtReduceComplex(
    float partialRe, float partialIm, __ubuf__ float* bufRe, __ubuf__ float* bufIm)
{
    float warpRe = asc_reduce_add(partialRe);
    float warpIm = asc_reduce_add(partialIm);
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5U;
    if (lane == 0U) {
        bufRe[warp] = warpRe;
        bufIm[warp] = warpIm;
    }
    asc_syncthreads();
    if (threadIdx.x < 32U) {
        const uint32_t nwarps = (blockDim.x + 31U) >> 5U;
        float sumRe = (threadIdx.x < nwarps) ? bufRe[threadIdx.x] : 0.0f;
        float sumIm = (threadIdx.x < nwarps) ? bufIm[threadIdx.x] : 0.0f;
        sumRe = asc_reduce_add(sumRe);
        sumIm = asc_reduce_add(sumIm);
        if (threadIdx.x == 0) {
            bufRe[0] = sumRe;
            bufIm[0] = sumIm;
        }
    }
}

template <bool TILED>
__simt_callee__ inline void CtbsvSimtReadX(
    uint32_t idx, uint32_t n, int32_t incx, __gm__ aclblasComplex* xGm,
    __ubuf__ float* xUb, float& re, float& im)
{
    if constexpr (TILED) {
        const int64_t off = CtbsvSimtXOffset(idx, n, incx);
        re = xGm[off].real;
        im = xGm[off].imag;
    } else {
        const uint32_t packed = idx * 2U;
        re = xUb[packed];
        im = xUb[packed + 1U];
    }
}

template <bool TILED>
__simt_callee__ inline void CtbsvSimtWriteX(
    uint32_t idx, uint32_t n, int32_t incx, __gm__ aclblasComplex* xGm,
    __ubuf__ float* xUb, float re, float im)
{
    if constexpr (TILED) {
        const int64_t off = CtbsvSimtXOffset(idx, n, incx);
        xGm[off].real = re;
        xGm[off].imag = im;
    } else {
        const uint32_t packed = idx * 2U;
        xUb[packed] = re;
        xUb[packed + 1U] = im;
    }
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool CONJ_ELEM>
__simt_callee__ inline void CtbsvSimtDivDiag(
    uint32_t j, uint32_t k, uint32_t lda, __gm__ const aclblasComplex* aGm, float& tRe, float& tIm)
{
    if constexpr (DIAG_IS_UNIT) {
        return;
    }
    const int64_t aIdx = CtbsvSimtAColBase<UPLO_IS_UPPER>(j, k, lda) + static_cast<int64_t>(j);
    float diagRe = aGm[aIdx].real;
    float diagIm = aGm[aIdx].imag;
    float outRe = 0.0f;
    float outIm = 0.0f;
    if constexpr (CONJ_ELEM) {
        CtbsvComplexDivByConj(tRe, tIm, diagRe, diagIm, outRe, outIm);
    } else {
        CtbsvComplexDiv(tRe, tIm, diagRe, diagIm, outRe, outIm);
    }
    tRe = outRe;
    tIm = outIm;
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool TILED>
__simt_callee__ inline bool CtbsvSimtScaleNoTransCol(
    uint32_t j, uint32_t n, uint32_t k, uint32_t lda, int32_t incx, __gm__ const aclblasComplex* aGm,
    __gm__ aclblasComplex* xGm, __ubuf__ float* xUb, __ubuf__ float* tBuf, __ubuf__ uint32_t* skipBuf)
{
    if (threadIdx.x == 0) {
        float tRe = 0.0f;
        float tIm = 0.0f;
        CtbsvSimtReadX<TILED>(j, n, incx, xGm, xUb, tRe, tIm);
        uint32_t skip = CtbsvIsCzero(tRe, tIm) ? 1U : 0U;
        if (skip == 0U) {
            CtbsvSimtDivDiag<UPLO_IS_UPPER, DIAG_IS_UNIT, false>(j, k, lda, aGm, tRe, tIm);
            CtbsvSimtWriteX<TILED>(j, n, incx, xGm, xUb, tRe, tIm);
        }
        tBuf[0] = tRe;
        tBuf[1] = tIm;
        skipBuf[0] = skip;
    }
    asc_syncthreads();
    return skipBuf[0] == 0U;
}

template <bool UPLO_IS_UPPER, bool TILED>
__simt_callee__ inline void CtbsvSimtScatterRange(
    uint32_t iStart, uint32_t iEnd, uint32_t j, uint32_t n, uint32_t k, uint32_t lda, int32_t incx,
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm,
    __ubuf__ float* xUb, float tRe, float tIm)
{
    const int64_t aBase = CtbsvSimtAColBase<UPLO_IS_UPPER>(j, k, lda);
    for (uint32_t i = iStart + threadIdx.x; i < iEnd; i += blockDim.x) {
        const int64_t aIdx = aBase + static_cast<int64_t>(i);
        float aRe = aGm[aIdx].real;
        float aIm = aGm[aIdx].imag;
        float xRe = 0.0f;
        float xIm = 0.0f;
        float pRe = 0.0f;
        float pIm = 0.0f;
        CtbsvSimtReadX<TILED>(i, n, incx, xGm, xUb, xRe, xIm);
        CtbsvCmulFast(tRe, tIm, aRe, aIm, pRe, pIm);
        CtbsvSimtWriteX<TILED>(i, n, incx, xGm, xUb, xRe - pRe, xIm - pIm);
    }
    asc_syncthreads();
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool TILED>
__simt_callee__ inline void CtbsvSimtNoTrans(
    uint32_t n, uint32_t k, uint32_t lda, int32_t incx, __gm__ const aclblasComplex* aGm,
    __gm__ aclblasComplex* xGm, __ubuf__ float* xUb)
{
    __ubuf__ float tBuf[2];
    __ubuf__ uint32_t skipBuf[1];
    if constexpr (UPLO_IS_UPPER) {
        for (uint32_t j = n; j-- > 0;) {
            if (!CtbsvSimtScaleNoTransCol<UPLO_IS_UPPER, DIAG_IS_UNIT, TILED>(
                    j, n, k, lda, incx, aGm, xGm, xUb, tBuf, skipBuf)) {
                continue;
            }
            const uint32_t iStart = (j >= k) ? (j - k) : 0;
            CtbsvSimtScatterRange<UPLO_IS_UPPER, TILED>(
                iStart, j, j, n, k, lda, incx, aGm, xGm, xUb, tBuf[0], tBuf[1]);
        }
    } else {
        for (uint32_t j = 0; j < n; ++j) {
            if (!CtbsvSimtScaleNoTransCol<UPLO_IS_UPPER, DIAG_IS_UNIT, TILED>(
                    j, n, k, lda, incx, aGm, xGm, xUb, tBuf, skipBuf)) {
                continue;
            }
            const uint32_t iEnd = (j + k + 1 < n) ? (j + k + 1) : n;
            CtbsvSimtScatterRange<UPLO_IS_UPPER, TILED>(
                j + 1, iEnd, j, n, k, lda, incx, aGm, xGm, xUb, tBuf[0], tBuf[1]);
        }
    }
}

template <bool CONJ_ELEM, bool TILED>
__simt_callee__ inline void CtbsvSimtTransMulPair(
    uint32_t i, uint32_t step, uint32_t n, int32_t incx, int64_t aBase,
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm, __ubuf__ float* xUb,
    float& pRe, float& pIm)
{
    const int64_t aIdx0 = aBase + static_cast<int64_t>(i);
    const int64_t aIdx1 = aBase + static_cast<int64_t>(i + step);
    float a0Re = aGm[aIdx0].real;
    float a0Im = aGm[aIdx0].imag;
    float a1Re = aGm[aIdx1].real;
    float a1Im = aGm[aIdx1].imag;
    float x0Re = 0.0f;
    float x0Im = 0.0f;
    float x1Re = 0.0f;
    float x1Im = 0.0f;
    float m0Re = 0.0f;
    float m0Im = 0.0f;
    float m1Re = 0.0f;
    float m1Im = 0.0f;
    CtbsvSimtReadX<TILED>(i, n, incx, xGm, xUb, x0Re, x0Im);
    CtbsvSimtReadX<TILED>(i + step, n, incx, xGm, xUb, x1Re, x1Im);
    if constexpr (CONJ_ELEM) {
        a0Im = -a0Im;
        a1Im = -a1Im;
    }
    CtbsvCmulFast(a0Re, a0Im, x0Re, x0Im, m0Re, m0Im);
    CtbsvCmulFast(a1Re, a1Im, x1Re, x1Im, m1Re, m1Im);
    pRe += m0Re + m1Re;
    pIm += m0Im + m1Im;
}

template <bool CONJ_ELEM, bool TILED>
__simt_callee__ inline void CtbsvSimtTransMulOne(
    uint32_t i, uint32_t n, int32_t incx, int64_t aBase,
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm, __ubuf__ float* xUb,
    float& pRe, float& pIm)
{
    const int64_t aIdx = aBase + static_cast<int64_t>(i);
    float aRe = aGm[aIdx].real;
    float aIm = aGm[aIdx].imag;
    float xRe = 0.0f;
    float xIm = 0.0f;
    float mRe = 0.0f;
    float mIm = 0.0f;
    if constexpr (CONJ_ELEM) {
        aIm = -aIm;
    }
    CtbsvSimtReadX<TILED>(i, n, incx, xGm, xUb, xRe, xIm);
    CtbsvCmulFast(aRe, aIm, xRe, xIm, mRe, mIm);
    pRe += mRe;
    pIm += mIm;
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool CONJ_ELEM, bool TILED>
__simt_callee__ inline void CtbsvSimtTransScaleCol(
    uint32_t j, uint32_t n, uint32_t k, uint32_t lda, int32_t incx,
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm,
    __ubuf__ float* xUb, float subRe, float subIm)
{
    if (threadIdx.x == 0) {
        float tRe = 0.0f;
        float tIm = 0.0f;
        CtbsvSimtReadX<TILED>(j, n, incx, xGm, xUb, tRe, tIm);
        tRe -= subRe;
        tIm -= subIm;
        CtbsvSimtDivDiag<UPLO_IS_UPPER, DIAG_IS_UNIT, CONJ_ELEM>(j, k, lda, aGm, tRe, tIm);
        CtbsvSimtWriteX<TILED>(j, n, incx, xGm, xUb, tRe, tIm);
    }
    asc_syncthreads();
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool CONJ_ELEM, bool TILED>
__simt_callee__ inline void CtbsvSimtTransCol(
    uint32_t j, uint32_t iStart, uint32_t iEnd, uint32_t n, uint32_t k, uint32_t lda, int32_t incx,
    __gm__ const aclblasComplex* aGm, __gm__ aclblasComplex* xGm,
    __ubuf__ float* xUb, __ubuf__ float* partRe, __ubuf__ float* partIm)
{
    if (iStart >= iEnd) {
        CtbsvSimtTransScaleCol<UPLO_IS_UPPER, DIAG_IS_UNIT, CONJ_ELEM, TILED>(
            j, n, k, lda, incx, aGm, xGm, xUb, 0.0f, 0.0f);
        return;
    }
    const int64_t aBase = CtbsvSimtAColBase<UPLO_IS_UPPER>(j, k, lda);
    float pRe = 0.0f;
    float pIm = 0.0f;
    uint32_t i = iStart + threadIdx.x;
    const uint32_t step = blockDim.x;
    for (; i + step < iEnd; i += step * 2U) {
        CtbsvSimtTransMulPair<CONJ_ELEM, TILED>(i, step, n, incx, aBase, aGm, xGm, xUb, pRe, pIm);
    }
    if (i < iEnd) {
        CtbsvSimtTransMulOne<CONJ_ELEM, TILED>(i, n, incx, aBase, aGm, xGm, xUb, pRe, pIm);
    }
    if (step <= 32U) {
        pRe = asc_reduce_add(pRe);
        pIm = asc_reduce_add(pIm);
        if (threadIdx.x == 0) {
            float tRe = 0.0f;
            float tIm = 0.0f;
            CtbsvSimtReadX<TILED>(j, n, incx, xGm, xUb, tRe, tIm);
            tRe -= pRe;
            tIm -= pIm;
            CtbsvSimtDivDiag<UPLO_IS_UPPER, DIAG_IS_UNIT, CONJ_ELEM>(j, k, lda, aGm, tRe, tIm);
            CtbsvSimtWriteX<TILED>(j, n, incx, xGm, xUb, tRe, tIm);
        }
        asc_syncthreads();
        return;
    }
    CtbsvSimtReduceComplex(pRe, pIm, partRe, partIm);
    CtbsvSimtTransScaleCol<UPLO_IS_UPPER, DIAG_IS_UNIT, CONJ_ELEM, TILED>(
        j, n, k, lda, incx, aGm, xGm, xUb, partRe[0], partIm[0]);
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool CONJ_ELEM, bool TILED>
__simt_callee__ inline void CtbsvSimtTrans(
    uint32_t n, uint32_t k, uint32_t lda, int32_t incx, __gm__ const aclblasComplex* aGm,
    __gm__ aclblasComplex* xGm, __ubuf__ float* xUb, __ubuf__ float* partRe, __ubuf__ float* partIm)
{
    if constexpr (UPLO_IS_UPPER) {
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t iStart = (j >= k) ? (j - k) : 0;
            CtbsvSimtTransCol<UPLO_IS_UPPER, DIAG_IS_UNIT, CONJ_ELEM, TILED>(
                j, iStart, j, n, k, lda, incx, aGm, xGm, xUb, partRe, partIm);
        }
    } else {
        for (uint32_t j = n; j-- > 0;) {
            const uint32_t iEnd = (j + k + 1 < n) ? (j + k + 1) : n;
            CtbsvSimtTransCol<UPLO_IS_UPPER, DIAG_IS_UNIT, CONJ_ELEM, TILED>(
                j, j + 1, iEnd, n, k, lda, incx, aGm, xGm, xUb, partRe, partIm);
        }
    }
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool TRANSPOSED, bool CONJ_ELEM, bool TILED>
__simt_callee__ inline void CtbsvSimtProcess(
    uint32_t n, uint32_t k, uint32_t lda, int32_t incx, __gm__ const aclblasComplex* aGm,
    __gm__ aclblasComplex* xGm, __ubuf__ float* xUb, __ubuf__ float* partRe, __ubuf__ float* partIm)
{
    if constexpr (!TILED) {
        CtbsvSimtLoadX(n, incx, xGm, xUb);
    }
    if constexpr (!TRANSPOSED) {
        CtbsvSimtNoTrans<UPLO_IS_UPPER, DIAG_IS_UNIT, TILED>(n, k, lda, incx, aGm, xGm, xUb);
    } else {
        CtbsvSimtTrans<UPLO_IS_UPPER, DIAG_IS_UNIT, CONJ_ELEM, TILED>(
            n, k, lda, incx, aGm, xGm, xUb, partRe, partIm);
    }
    if constexpr (!TILED) {
        CtbsvSimtStoreX(n, incx, xGm, xUb);
    }
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool TRANSPOSED, bool CONJ_ELEM>
__simt_callee__ inline void CtbsvSimtTiled(
    uint32_t n, uint32_t k, uint32_t lda, int32_t incx, __gm__ const aclblasComplex* aGm,
    __gm__ aclblasComplex* xGm)
{
    __ubuf__ float dummyXUb[2];
    __ubuf__ float partRe[CTBSV_UB_PARTIALS];
    __ubuf__ float partIm[CTBSV_UB_PARTIALS];
    CtbsvSimtProcess<UPLO_IS_UPPER, DIAG_IS_UNIT, TRANSPOSED, CONJ_ELEM, true>(
        n, k, lda, incx, aGm, xGm, dummyXUb, partRe, partIm);
}

template <bool UPLO_IS_UPPER, bool DIAG_IS_UNIT, bool TRANSPOSED, bool CONJ_ELEM>
__simt_callee__ inline void CtbsvSimtUb(
    uint32_t n, uint32_t k, uint32_t lda, int32_t incx, __gm__ const aclblasComplex* aGm,
    __gm__ aclblasComplex* xGm)
{
    __ubuf__ float xUb[CTBSV_UB_X_COMPLEX * 2U];
    __ubuf__ float partRe[CTBSV_UB_PARTIALS];
    __ubuf__ float partIm[CTBSV_UB_PARTIALS];
    CtbsvSimtProcess<UPLO_IS_UPPER, DIAG_IS_UNIT, TRANSPOSED, CONJ_ELEM, false>(
        n, k, lda, incx, aGm, xGm, xUb, partRe, partIm);
}

template <bool UPLO_IS_UPPER, bool TRANSPOSED, bool CONJ_ELEM, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtbsvSimt(
    uint32_t n, uint32_t k, uint32_t lda, int32_t incx, __gm__ const aclblasComplex* aGm,
    __gm__ aclblasComplex* xGm)
{
    if (n <= CTBSV_UB_X_COMPLEX) {
        CtbsvSimtUb<UPLO_IS_UPPER, DIAG_IS_UNIT, TRANSPOSED, CONJ_ELEM>(n, k, lda, incx, aGm, xGm);
    } else {
        CtbsvSimtTiled<UPLO_IS_UPPER, DIAG_IS_UNIT, TRANSPOSED, CONJ_ELEM>(n, k, lda, incx, aGm, xGm);
    }
}

template <bool UPLO_IS_UPPER, bool TRANSPOSED, bool CONJ_ELEM>
__aicore__ inline void CtbsvSimtLaunchDiag(
    const CtbsvTilingData& tiling, uint32_t ldaU32, __gm__ aclblasComplex* aGm, __gm__ aclblasComplex* xGm)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        asc_vf_call<CtbsvSimt<UPLO_IS_UPPER, TRANSPOSED, CONJ_ELEM, false>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, ldaU32, tiling.incx, aGm, xGm);
    } else {
        asc_vf_call<CtbsvSimt<UPLO_IS_UPPER, TRANSPOSED, CONJ_ELEM, true>>(
            dim3{tiling.numThreads, 1, 1}, tiling.n, tiling.k, ldaU32, tiling.incx, aGm, xGm);
    }
}

template <bool UPLO_IS_UPPER>
__aicore__ inline void CtbsvSimtLaunchTrans(
    const CtbsvTilingData& tiling, uint32_t ldaU32, __gm__ aclblasComplex* aGm, __gm__ aclblasComplex* xGm)
{
    if (tiling.trans == ACLBLAS_OP_N) {
        CtbsvSimtLaunchDiag<UPLO_IS_UPPER, false, false>(tiling, ldaU32, aGm, xGm);
    } else if (tiling.trans == ACLBLAS_OP_T) {
        CtbsvSimtLaunchDiag<UPLO_IS_UPPER, true, false>(tiling, ldaU32, aGm, xGm);
    } else {
        CtbsvSimtLaunchDiag<UPLO_IS_UPPER, true, true>(tiling, ldaU32, aGm, xGm);
    }
}

__global__ __aicore__ void ctbsv_kernel_simt(CtbsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* aGm = reinterpret_cast<__gm__ aclblasComplex*>(tiling.a);
    auto* xGm = reinterpret_cast<__gm__ aclblasComplex*>(tiling.x);
    const uint32_t ldaU32 = tiling.lda;
    if (tiling.uplo == ACLBLAS_LOWER) {
        CtbsvSimtLaunchTrans<false>(tiling, ldaU32, aGm, xGm);
    } else {
        CtbsvSimtLaunchTrans<true>(tiling, ldaU32, aGm, xGm);
    }
}

void ctbsv_simt_kernel_do(const CtbsvTilingData& tiling, void* stream)
{
    ctbsv_kernel_simt<<<1, nullptr, stream>>>(tiling);
}
