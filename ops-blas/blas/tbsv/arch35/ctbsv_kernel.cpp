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
#include "ctbsv_tiling_data.h"
#include "common/helper/kernel_constant.h"
#include "ctbsv_complex_arith.h"

using namespace AscendC;

enum class CtbsvUplo { UPPER, LOWER };
enum class CtbsvTrans { NO_TRANS, TRANS, CONJ };
enum class CtbsvDiag { UNIT, NON_UNIT };

constexpr uint32_t CTBSV_FLOATS_PER_BLOCK = 8;

__aicore__ inline uint32_t CtbsvAlignBlock(uint32_t n)
{
    return (n + CTBSV_FLOATS_PER_BLOCK - 1U) / CTBSV_FLOATS_PER_BLOCK * CTBSV_FLOATS_PER_BLOCK;
}

__aicore__ inline void CtbsvCopyGm2UbIssue(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, int64_t gmOff, uint32_t count)
{
    const uint32_t aligned = (count / CTBSV_FLOATS_PER_BLOCK) * CTBSV_FLOATS_PER_BLOCK;
    const uint32_t tail = count - aligned;
    if (aligned > 0U) {
        DataCopy(dst, src[gmOff], aligned);
    }
    if (tail > 0U) {
        const uint8_t pad = static_cast<uint8_t>(CTBSV_FLOATS_PER_BLOCK - tail);
        DataCopyExtParams params{1, static_cast<uint32_t>(tail * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, pad, 0.0f};
        DataCopyPad(dst[aligned], src[gmOff + static_cast<int64_t>(aligned)], params, padParams);
    }
}

__aicore__ inline void CtbsvCopyGm2Ub(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, int64_t gmOff, uint32_t count)
{
    CtbsvCopyGm2UbIssue(dst, src, gmOff, count);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

__aicore__ inline void CtbsvCopyUb2Gm(
    const GlobalTensor<float>& dst, int64_t gmOff, const LocalTensor<float>& src, uint32_t count)
{
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    const uint32_t aligned = (count / CTBSV_FLOATS_PER_BLOCK) * CTBSV_FLOATS_PER_BLOCK;
    const uint32_t tail = count - aligned;
    if (aligned > 0U) {
        DataCopy(dst[gmOff], src, aligned);
    }
    if (tail > 0U) {
        DataCopyExtParams params{1, static_cast<uint32_t>(tail * sizeof(float)), 0, 0, 0};
        DataCopyPad(dst[gmOff + static_cast<int64_t>(aligned)], src[aligned], params);
    }
}

__aicore__ inline int64_t CtbsvXOffset(uint32_t idx, uint32_t n, int32_t incx)
{
    if (incx >= 0) {
        return static_cast<int64_t>(idx) * static_cast<int64_t>(incx);
    } else {
        return static_cast<int64_t>(n - 1 - idx) * static_cast<int64_t>(-incx);
    }
}

__aicore__ inline int64_t CtbsvComplexFloatIdx(int64_t complexIdx)
{
    return complexIdx * 2;
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
class CtbsvKernel {
public:
    __aicore__ inline CtbsvKernel() {}
    __aicore__ inline void Init(const CtbsvTilingData& tiling);
    __aicore__ inline void Process(TPipe& pipe);

private:
    __aicore__ inline void ProcessRows(TPipe& pipe);
    __aicore__ inline void LoadXUb(const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline void StoreXUb(const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline void GetDiag(uint32_t row, float& diagRe, float& diagIm) const;
    __aicore__ inline void DivByDiag(uint32_t j, float& tRe, float& tIm) const;
    __aicore__ inline bool ScaleNoTransCol(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm,
        uint32_t j, float& tRe, float& tIm);
    __aicore__ inline void ScatterCol(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm,
        uint32_t i, uint32_t j, float tRe, float tIm);
    __aicore__ inline void GatherTransTerm(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm,
        uint32_t i, uint32_t j, float& tRe, float& tIm);
    __aicore__ inline void StoreTransCol(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm,
        uint32_t j, float tRe, float tIm);
    __aicore__ inline void ProcessNoTransUpper(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline void ProcessNoTransLower(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline void ProcessTransUpper(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline void ProcessTransLower(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm);
    __aicore__ inline int64_t XOffset(uint32_t idx) const;
    __aicore__ inline void IssueACol(const LocalTensor<float>& dst, uint32_t j, bool ping);
    __aicore__ inline void WaitACol(bool ping);
    __aicore__ inline void LoadACol(uint32_t j);
    __aicore__ inline void BindACol(bool usePing);
    __aicore__ inline void BindAColVec(bool usePing);
    __aicore__ inline void PrefetchACol(uint32_t j, bool usePing);
    __aicore__ inline void GetAFromCol(uint32_t row, uint32_t col, float& aRe, float& aIm) const;
    __aicore__ inline void GatherTransVec(
        const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm,
        uint32_t iStart, uint32_t iEnd, uint32_t j, float& tRe, float& tIm);

    GlobalTensor<float> aGM;
    GlobalTensor<float> xGM;
    LocalTensor<float> aColUb;
    LocalTensor<float> aColPing;
    LocalTensor<float> aColPong;
    LocalTensor<float> xPackUb;
    LocalTensor<float> aReUb;
    LocalTensor<float> aImUb;
    LocalTensor<float> prodReUb;
    LocalTensor<float> prodImUb;
    LocalTensor<float> vecTmpUb;
    LocalTensor<float> vecWorkUb;
    LocalTensor<float> vecOutUb;
    LocalTensor<uint32_t> gatherOffUb;
    uint32_t aColAlign;

    uint32_t n;
    uint32_t k;
    uint32_t lda;
    int32_t incx;
};

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline int64_t CtbsvKernel<UPLO, TRANS, DIAG>::XOffset(uint32_t idx) const
{
    return CtbsvXOffset(idx, n, incx);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::Init(const CtbsvTilingData& tiling)
{
    this->n = tiling.n;
    this->k = tiling.k;
    this->lda = tiling.lda;
    this->incx = tiling.incx;

    int64_t aCount = static_cast<int64_t>(tiling.lda) * static_cast<int64_t>(n) * 2;
    int64_t absIncx = incx >= 0 ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    int64_t xCount = (n > 0) ? (absIncx * static_cast<int64_t>(n - 1) + 1) * 2 : 0;
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.a), aCount);
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.x), xCount);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::IssueACol(
    const LocalTensor<float>& dst, uint32_t j, bool ping)
{
    const int64_t gmOff = static_cast<int64_t>(j) * static_cast<int64_t>(lda) * 2;
    CtbsvCopyGm2UbIssue(dst, aGM, gmOff, (k + 1U) * 2U);
    if (ping) {
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
    } else {
        SetFlag<HardEvent::MTE2_S>(EVENT_ID1);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::WaitACol(bool ping)
{
    if (ping) {
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    } else {
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::LoadACol(uint32_t j)
{
    IssueACol(aColUb, j, true);
    WaitACol(true);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::BindACol(bool usePing)
{
    WaitACol(usePing);
    aColUb = usePing ? aColPing : aColPong;
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::BindAColVec(bool usePing)
{
    BindACol(usePing);
    const int32_t srcCount = static_cast<int32_t>(aColAlign);
    DeInterleave(aReUb, aImUb, aColUb, srcCount);
    if constexpr (TRANS == CtbsvTrans::CONJ) {
        Muls(aImUb, aImUb, -1.0f, srcCount / 2);
    }
    PipeBarrier<PIPE_V>();
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::PrefetchACol(uint32_t j, bool usePing)
{
    IssueACol(usePing ? aColPing : aColPong, j, usePing);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::GatherTransVec(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm,
    uint32_t iStart, uint32_t iEnd, uint32_t j, float& tRe, float& tIm)
{
    const uint32_t count = iEnd - iStart;
    if (count == 0U) {
        return;
    }
    if (count < 8U) {
        for (uint32_t i = iStart; i < iEnd; ++i) {
            GatherTransTerm(xUbRe, xUbIm, i, j, tRe, tIm);
        }
        return;
    }
    const uint32_t bandOff = (UPLO == CtbsvUplo::UPPER) ? (k + iStart - j) : (iStart - j);
    const uint32_t aBase = bandOff * static_cast<uint32_t>(sizeof(float));
    const uint32_t xBase = iStart * static_cast<uint32_t>(sizeof(float));
    Gather(prodReUb, aReUb, gatherOffUb, aBase, count);
    Gather(prodImUb, aImUb, gatherOffUb, aBase, count);
    Gather(vecTmpUb, xUbRe, gatherOffUb, xBase, count);
    Gather(aReUb, xUbIm, gatherOffUb, xBase, count);
    PipeBarrier<PIPE_V>();
    const int32_t nElem = static_cast<int32_t>(count);
    Mul(vecWorkUb, prodReUb, vecTmpUb, nElem);
    Mul(aImUb, prodImUb, aReUb, nElem);
    Sub(vecWorkUb, vecWorkUb, aImUb, nElem);
    Mul(prodReUb, prodReUb, aReUb, nElem);
    Mul(prodImUb, prodImUb, vecTmpUb, nElem);
    Add(prodImUb, prodReUb, prodImUb, nElem);
    ReduceSum<float>(vecOutUb, vecWorkUb, aReUb, nElem);
    ReduceSum<float>(vecOutUb[8], prodImUb, aReUb, nElem);
    SetFlag<HardEvent::V_S>(EVENT_ID4);
    WaitFlag<HardEvent::V_S>(EVENT_ID4);
    tRe -= vecOutUb.GetValue(0);
    tIm -= vecOutUb.GetValue(8);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::GetAFromCol(
    uint32_t row, uint32_t col, float& aRe, float& aIm) const
{
    const uint32_t bandRow = (UPLO == CtbsvUplo::UPPER) ? (k + row - col) : (row - col);
    const uint32_t fIdx = bandRow * 2U;
    aRe = aColUb.GetValue(fIdx);
    aIm = aColUb.GetValue(fIdx + 1U);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::GetDiag(uint32_t row, float& diagRe, float& diagIm) const
{
    (void)row;
    const uint32_t bandRow = (UPLO == CtbsvUplo::UPPER) ? k : 0U;
    const uint32_t fIdx = bandRow * 2U;
    diagRe = aColUb.GetValue(fIdx);
    diagIm = aColUb.GetValue(fIdx + 1U);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::DivByDiag(uint32_t j, float& tRe, float& tIm) const
{
    float diagRe = 0.0f;
    float diagIm = 0.0f;
    GetDiag(j, diagRe, diagIm);
    float outRe = 0.0f;
    float outIm = 0.0f;
    if constexpr (TRANS == CtbsvTrans::CONJ) {
        CtbsvComplexDivByConj(tRe, tIm, diagRe, diagIm, outRe, outIm);
    } else {
        CtbsvComplexDiv(tRe, tIm, diagRe, diagIm, outRe, outIm);
    }
    tRe = outRe;
    tIm = outIm;
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline bool CtbsvKernel<UPLO, TRANS, DIAG>::ScaleNoTransCol(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm,
    uint32_t j, float& tRe, float& tIm)
{
    tRe = xUbRe.GetValue(j);
    tIm = xUbIm.GetValue(j);
    if (CtbsvIsCzero(tRe, tIm)) {
        return false;
    }
    if constexpr (DIAG == CtbsvDiag::NON_UNIT) {
        DivByDiag(j, tRe, tIm);
        xUbRe.SetValue(j, tRe);
        xUbIm.SetValue(j, tIm);
    }
    return true;
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::ScatterCol(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm,
    uint32_t i, uint32_t j, float tRe, float tIm)
{
    float aRe = 0.0f;
    float aIm = 0.0f;
    float pRe = 0.0f;
    float pIm = 0.0f;
    GetAFromCol(i, j, aRe, aIm);
    CtbsvCmulFast(tRe, tIm, aRe, aIm, pRe, pIm);
    xUbRe.SetValue(i, xUbRe.GetValue(i) - pRe);
    xUbIm.SetValue(i, xUbIm.GetValue(i) - pIm);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::GatherTransTerm(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm,
    uint32_t i, uint32_t j, float& tRe, float& tIm)
{
    float aRe = 0.0f;
    float aIm = 0.0f;
    float pRe = 0.0f;
    float pIm = 0.0f;
    GetAFromCol(i, j, aRe, aIm);
    if constexpr (TRANS == CtbsvTrans::CONJ) {
        aIm = -aIm;
    }
    CtbsvCmulFast(aRe, aIm, xUbRe.GetValue(i), xUbIm.GetValue(i), pRe, pIm);
    tRe -= pRe;
    tIm -= pIm;
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::StoreTransCol(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm,
    uint32_t j, float tRe, float tIm)
{
    if constexpr (DIAG == CtbsvDiag::NON_UNIT) {
        DivByDiag(j, tRe, tIm);
    }
    xUbRe.SetValue(j, tRe);
    xUbIm.SetValue(j, tIm);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::LoadXUb(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    if (incx == 1) {
        CtbsvCopyGm2Ub(xPackUb, xGM, 0, n * 2U);
        for (uint32_t i = 0; i < n; ++i) {
            xUbRe.SetValue(i, xPackUb.GetValue(i * 2U));
            xUbIm.SetValue(i, xPackUb.GetValue(i * 2U + 1U));
        }
        return;
    }
    for (uint32_t i = 0; i < n; ++i) {
        int64_t off = CtbsvComplexFloatIdx(XOffset(i));
        xUbRe.SetValue(i, xGM.GetValue(off));
        xUbIm.SetValue(i, xGM.GetValue(off + 1));
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::StoreXUb(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    if (incx == 1) {
        for (uint32_t i = 0; i < n; ++i) {
            xPackUb.SetValue(i * 2U, xUbRe.GetValue(i));
            xPackUb.SetValue(i * 2U + 1U, xUbIm.GetValue(i));
        }
        CtbsvCopyUb2Gm(xGM, 0, xPackUb, n * 2U);
        return;
    }
    for (uint32_t i = 0; i < n; ++i) {
        int64_t off = CtbsvComplexFloatIdx(XOffset(i));
        xGM.SetValue(off, xUbRe.GetValue(i));
        xGM.SetValue(off + 1, xUbIm.GetValue(i));
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::ProcessNoTransUpper(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    if (n == 0U) {
        return;
    }
    bool usePing = true;
    PrefetchACol(n - 1U, true);
    for (uint32_t j = n; j-- > 0;) {
        BindACol(usePing);
        if (j > 0U) {
            PrefetchACol(j - 1U, !usePing);
        }
        usePing = !usePing;
        float tRe = 0.0f;
        float tIm = 0.0f;
        if (!ScaleNoTransCol(xUbRe, xUbIm, j, tRe, tIm)) {
            continue;
        }
        const uint32_t iStart = (j >= k) ? (j - k) : 0;
        for (uint32_t i = j; i-- > iStart;) {
            ScatterCol(xUbRe, xUbIm, i, j, tRe, tIm);
        }
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::ProcessNoTransLower(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    if (n == 0U) {
        return;
    }
    bool usePing = true;
    PrefetchACol(0U, true);
    for (uint32_t j = 0; j < n; ++j) {
        BindACol(usePing);
        if (j + 1U < n) {
            PrefetchACol(j + 1U, !usePing);
        }
        usePing = !usePing;
        float tRe = 0.0f;
        float tIm = 0.0f;
        if (!ScaleNoTransCol(xUbRe, xUbIm, j, tRe, tIm)) {
            continue;
        }
        const uint32_t iEnd = (j + k + 1 < n) ? (j + k + 1) : n;
        for (uint32_t i = j + 1; i < iEnd; ++i) {
            ScatterCol(xUbRe, xUbIm, i, j, tRe, tIm);
        }
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::ProcessTransUpper(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    if (n == 0U) {
        return;
    }
    bool usePing = true;
    PrefetchACol(0U, true);
    for (uint32_t j = 0; j < n; ++j) {
        BindAColVec(usePing);
        if (j + 1U < n) {
            PrefetchACol(j + 1U, !usePing);
        }
        usePing = !usePing;
        float tRe = xUbRe.GetValue(j);
        float tIm = xUbIm.GetValue(j);
        const uint32_t iStart = (j >= k) ? (j - k) : 0;
        GatherTransVec(xUbRe, xUbIm, iStart, j, j, tRe, tIm);
        StoreTransCol(xUbRe, xUbIm, j, tRe, tIm);
        SetFlag<HardEvent::S_V>(EVENT_ID4);
        WaitFlag<HardEvent::S_V>(EVENT_ID4);
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::ProcessTransLower(
    const LocalTensor<float>& xUbRe, const LocalTensor<float>& xUbIm)
{
    if (n == 0U) {
        return;
    }
    bool usePing = true;
    PrefetchACol(n - 1U, true);
    for (uint32_t j = n; j-- > 0;) {
        BindAColVec(usePing);
        if (j > 0U) {
            PrefetchACol(j - 1U, !usePing);
        }
        usePing = !usePing;
        float tRe = xUbRe.GetValue(j);
        float tIm = xUbIm.GetValue(j);
        const uint32_t iEnd = (j + k + 1 < n) ? (j + k + 1) : n;
        GatherTransVec(xUbRe, xUbIm, j + 1U, iEnd, j, tRe, tIm);
        StoreTransCol(xUbRe, xUbIm, j, tRe, tIm);
        SetFlag<HardEvent::S_V>(EVENT_ID4);
        WaitFlag<HardEvent::S_V>(EVENT_ID4);
    }
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::ProcessRows(TPipe& pipe)
{
    TBuf<TPosition::VECCALC> xUbReBuf;
    TBuf<TPosition::VECCALC> xUbImBuf;
    TBuf<TPosition::VECCALC> xPackBuf;
    TBuf<TPosition::VECCALC> aColPingBuf;
    TBuf<TPosition::VECCALC> aColPongBuf;
    TBuf<TPosition::VECCALC> aReBuf;
    TBuf<TPosition::VECCALC> aImBuf;
    TBuf<TPosition::VECCALC> prodReBuf;
    TBuf<TPosition::VECCALC> prodImBuf;
    TBuf<TPosition::VECCALC> vecTmpBuf;
    TBuf<TPosition::VECCALC> vecWorkBuf;
    TBuf<TPosition::VECCALC> vecOutBuf;
    TBuf<TPosition::VECCALC> gatherOffBuf;
    const uint32_t xAlign = CtbsvAlignBlock(n);
    const uint32_t packAlign = CtbsvAlignBlock(n * 2U);
    aColAlign = CtbsvAlignBlock((k + 1U) * 2U);
    constexpr uint32_t kVecPad = 256 / sizeof(float);
    const uint32_t xPad = (TRANS != CtbsvTrans::NO_TRANS) ? kVecPad : 0U;
    pipe.InitBuffer(xUbReBuf, (xAlign + xPad) * sizeof(float));
    pipe.InitBuffer(xUbImBuf, (xAlign + xPad) * sizeof(float));
    pipe.InitBuffer(xPackBuf, packAlign * sizeof(float));
    pipe.InitBuffer(aColPingBuf, aColAlign * sizeof(float));
    pipe.InitBuffer(aColPongBuf, aColAlign * sizeof(float));
    LocalTensor<float> xUbRe = xUbReBuf.Get<float>();
    LocalTensor<float> xUbIm = xUbImBuf.Get<float>();
    xPackUb = xPackBuf.Get<float>();
    aColPing = aColPingBuf.Get<float>();
    aColPong = aColPongBuf.Get<float>();
    aColUb = aColPing;
    if constexpr (TRANS != CtbsvTrans::NO_TRANS) {
        const uint32_t pairAlign = CtbsvAlignBlock(aColAlign / 2U);
        const uint32_t workAlign = CtbsvAlignBlock(k + 1U + 256U);
        const uint32_t pairPad = pairAlign + kVecPad;
        pipe.InitBuffer(aReBuf, pairPad * sizeof(float));
        pipe.InitBuffer(aImBuf, pairPad * sizeof(float));
        pipe.InitBuffer(prodReBuf, pairPad * sizeof(float));
        pipe.InitBuffer(prodImBuf, pairPad * sizeof(float));
        pipe.InitBuffer(vecTmpBuf, pairPad * sizeof(float));
        pipe.InitBuffer(vecWorkBuf, workAlign * sizeof(float));
        pipe.InitBuffer(vecOutBuf, 32U * sizeof(float));
        pipe.InitBuffer(gatherOffBuf, pairPad * sizeof(uint32_t));
        aReUb = aReBuf.Get<float>();
        aImUb = aImBuf.Get<float>();
        prodReUb = prodReBuf.Get<float>();
        prodImUb = prodImBuf.Get<float>();
        vecTmpUb = vecTmpBuf.Get<float>();
        vecWorkUb = vecWorkBuf.Get<float>();
        vecOutUb = vecOutBuf.Get<float>();
        gatherOffUb = gatherOffBuf.Get<uint32_t>();
        Duplicate(aReUb, 0.0f, static_cast<int32_t>(pairPad));
        Duplicate(aImUb, 0.0f, static_cast<int32_t>(pairPad));
        for (uint32_t i = 0; i < pairPad; ++i) {
            gatherOffUb.SetValue(i, i * static_cast<uint32_t>(sizeof(float)));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID5);
        WaitFlag<HardEvent::S_V>(EVENT_ID5);
    }
    LoadXUb(xUbRe, xUbIm);
    if constexpr (TRANS != CtbsvTrans::NO_TRANS) {
        for (uint32_t i = n; i < n + kVecPad; ++i) {
            xUbRe.SetValue(i, 0.0f);
            xUbIm.SetValue(i, 0.0f);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID4);
        WaitFlag<HardEvent::S_V>(EVENT_ID4);
    }
    if constexpr (TRANS == CtbsvTrans::NO_TRANS) {
        if constexpr (UPLO == CtbsvUplo::UPPER) {
            ProcessNoTransUpper(xUbRe, xUbIm);
        } else {
            ProcessNoTransLower(xUbRe, xUbIm);
        }
    } else {
        if constexpr (UPLO == CtbsvUplo::UPPER) {
            ProcessTransUpper(xUbRe, xUbIm);
        } else {
            ProcessTransLower(xUbRe, xUbIm);
        }
    }
    StoreXUb(xUbRe, xUbIm);
}

template <CtbsvUplo UPLO, CtbsvTrans TRANS, CtbsvDiag DIAG>
__aicore__ inline void CtbsvKernel<UPLO, TRANS, DIAG>::Process(TPipe& pipe)
{
    ProcessRows(pipe);
}

#define DEFINE_CTBSV_KERNEL(uplo, trans, diag, name)                                        \
    __global__ __aicore__ void name(CtbsvTilingData tiling)                                 \
    {                                                                                       \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                     \
        TPipe pipe;                                                                         \
        CtbsvKernel<CtbsvUplo::uplo, CtbsvTrans::trans, CtbsvDiag::diag> op;                \
        op.Init(tiling);                                                                    \
        op.Process(pipe);                                                                   \
    }

DEFINE_CTBSV_KERNEL(LOWER, NO_TRANS, NON_UNIT, ctbsv_kernel_lower_no_trans_non_unit)
DEFINE_CTBSV_KERNEL(LOWER, NO_TRANS, UNIT, ctbsv_kernel_lower_no_trans_unit)
DEFINE_CTBSV_KERNEL(UPPER, NO_TRANS, NON_UNIT, ctbsv_kernel_upper_no_trans_non_unit)
DEFINE_CTBSV_KERNEL(UPPER, NO_TRANS, UNIT, ctbsv_kernel_upper_no_trans_unit)
DEFINE_CTBSV_KERNEL(LOWER, TRANS, NON_UNIT, ctbsv_kernel_lower_trans_non_unit)
DEFINE_CTBSV_KERNEL(LOWER, TRANS, UNIT, ctbsv_kernel_lower_trans_unit)
DEFINE_CTBSV_KERNEL(UPPER, TRANS, NON_UNIT, ctbsv_kernel_upper_trans_non_unit)
DEFINE_CTBSV_KERNEL(UPPER, TRANS, UNIT, ctbsv_kernel_upper_trans_unit)
DEFINE_CTBSV_KERNEL(LOWER, CONJ, NON_UNIT, ctbsv_kernel_lower_conj_non_unit)
DEFINE_CTBSV_KERNEL(LOWER, CONJ, UNIT, ctbsv_kernel_lower_conj_unit)
DEFINE_CTBSV_KERNEL(UPPER, CONJ, NON_UNIT, ctbsv_kernel_upper_conj_non_unit)
DEFINE_CTBSV_KERNEL(UPPER, CONJ, UNIT, ctbsv_kernel_upper_conj_unit)

#undef DEFINE_CTBSV_KERNEL

void ctbsv_simt_kernel_do(const CtbsvTilingData &tiling, void* stream);

namespace {
void CtbsvLaunchLowerNoTrans(const CtbsvTilingData &tiling, void* stream)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        ctbsv_kernel_lower_no_trans_non_unit<<<1, nullptr, stream>>>(tiling);
    } else {
        ctbsv_kernel_lower_no_trans_unit<<<1, nullptr, stream>>>(tiling);
    }
}

void CtbsvLaunchLowerTrans(const CtbsvTilingData &tiling, void* stream)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        ctbsv_kernel_lower_trans_non_unit<<<1, nullptr, stream>>>(tiling);
    } else {
        ctbsv_kernel_lower_trans_unit<<<1, nullptr, stream>>>(tiling);
    }
}

void CtbsvLaunchLowerConj(const CtbsvTilingData &tiling, void* stream)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        ctbsv_kernel_lower_conj_non_unit<<<1, nullptr, stream>>>(tiling);
    } else {
        ctbsv_kernel_lower_conj_unit<<<1, nullptr, stream>>>(tiling);
    }
}

void CtbsvLaunchUpperNoTrans(const CtbsvTilingData &tiling, void* stream)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        ctbsv_kernel_upper_no_trans_non_unit<<<1, nullptr, stream>>>(tiling);
    } else {
        ctbsv_kernel_upper_no_trans_unit<<<1, nullptr, stream>>>(tiling);
    }
}

void CtbsvLaunchUpperTrans(const CtbsvTilingData &tiling, void* stream)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        ctbsv_kernel_upper_trans_non_unit<<<1, nullptr, stream>>>(tiling);
    } else {
        ctbsv_kernel_upper_trans_unit<<<1, nullptr, stream>>>(tiling);
    }
}

void CtbsvLaunchUpperConj(const CtbsvTilingData &tiling, void* stream)
{
    if (tiling.diag == ACLBLAS_NON_UNIT) {
        ctbsv_kernel_upper_conj_non_unit<<<1, nullptr, stream>>>(tiling);
    } else {
        ctbsv_kernel_upper_conj_unit<<<1, nullptr, stream>>>(tiling);
    }
}

void CtbsvLaunchLower(const CtbsvTilingData &tiling, void* stream)
{
    if (tiling.trans == ACLBLAS_OP_N) {
        CtbsvLaunchLowerNoTrans(tiling, stream);
    } else if (tiling.trans == ACLBLAS_OP_T) {
        CtbsvLaunchLowerTrans(tiling, stream);
    } else {
        CtbsvLaunchLowerConj(tiling, stream);
    }
}

void CtbsvLaunchUpper(const CtbsvTilingData &tiling, void* stream)
{
    if (tiling.trans == ACLBLAS_OP_N) {
        CtbsvLaunchUpperNoTrans(tiling, stream);
    } else if (tiling.trans == ACLBLAS_OP_T) {
        CtbsvLaunchUpperTrans(tiling, stream);
    } else {
        CtbsvLaunchUpperConj(tiling, stream);
    }
}
}  // namespace

void ctbsv_kernel_do(const CtbsvTilingData &tiling, void* stream)
{
    if (tiling.numThreads > 0) {
        ctbsv_simt_kernel_do(tiling, stream);
        return;
    }
    if (tiling.uplo == ACLBLAS_LOWER) {
        CtbsvLaunchLower(tiling, stream);
    } else {
        CtbsvLaunchUpper(tiling, stream);
    }
}
