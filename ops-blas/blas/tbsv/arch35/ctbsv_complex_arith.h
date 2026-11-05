/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CTBSV_COMPLEX_ARITH_H
#define CTBSV_COMPLEX_ARITH_H

#ifndef CTBSV_ARITH_ATTR
#define CTBSV_ARITH_ATTR __aicore__ inline
#endif

CTBSV_ARITH_ATTR float CtbsvAbsf(float x)
{
    return (x >= 0.0f) ? x : -x;
}

CTBSV_ARITH_ATTR bool CtbsvIsNan(float x)
{
    return x != x;
}

CTBSV_ARITH_ATTR bool CtbsvIsInf(float x)
{
    return CtbsvAbsf(x) > 3.402823466e+38f;
}

CTBSV_ARITH_ATTR bool CtbsvIsFinite(float x)
{
    return (!CtbsvIsNan(x)) && (!CtbsvIsInf(x));
}

CTBSV_ARITH_ATTR float CtbsvCopysign(float mag, float signSrc)
{
    float absMag = CtbsvAbsf(mag);
    return (signSrc < 0.0f) ? -absMag : absMag;
}

CTBSV_ARITH_ATTR float CtbsvInf()
{
    return 3.402823466e+38f * 3.402823466e+38f;
}

CTBSV_ARITH_ATTR bool CtbsvIsCzero(float re, float im)
{
    return (re == 0.0f) && (im == 0.0f);
}

CTBSV_ARITH_ATTR void CtbsvRecenterInfOperand(float& xRe, float& xIm, float& yRe, float& yIm, bool& recalc)
{
    if (!(CtbsvIsInf(xRe) || CtbsvIsInf(xIm))) {
        return;
    }
    xRe = CtbsvCopysign(CtbsvIsInf(xRe) ? 1.0f : 0.0f, xRe);
    xIm = CtbsvCopysign(CtbsvIsInf(xIm) ? 1.0f : 0.0f, xIm);
    if (CtbsvIsNan(yRe)) {
        yRe = CtbsvCopysign(0.0f, yRe);
    }
    if (CtbsvIsNan(yIm)) {
        yIm = CtbsvCopysign(0.0f, yIm);
    }
    recalc = true;
}

CTBSV_ARITH_ATTR void CtbsvCmulRecover(
    float aRe, float aIm, float bRe, float bIm, float ac, float bd, float ad, float bc, float& pRe, float& pIm)
{
    bool recalc = false;
    CtbsvRecenterInfOperand(aRe, aIm, bRe, bIm, recalc);
    CtbsvRecenterInfOperand(bRe, bIm, aRe, aIm, recalc);
    if (!recalc && (CtbsvIsInf(ac) || CtbsvIsInf(bd) || CtbsvIsInf(ad) || CtbsvIsInf(bc))) {
        const float tiny = 5.421010862427522e-20f; // 2^-64
        aRe *= tiny;
        aIm *= tiny;
        bRe *= tiny;
        bIm *= tiny;
        recalc = true;
    }
    if (recalc) {
        const float inf = CtbsvInf();
        pRe = inf * (aRe * bRe - aIm * bIm);
        pIm = inf * (aRe * bIm + aIm * bRe);
    }
}

// libgcc __mulsc3: recover Inf-Inf NaN so overflow matches OpenBLAS/compiler complex mul.
CTBSV_ARITH_ATTR void CtbsvCmul(
    float aRe, float aIm, float bRe, float bIm, float& pRe, float& pIm)
{
    float ac = aRe * bRe;
    float bd = aIm * bIm;
    float ad = aRe * bIm;
    float bc = aIm * bRe;
    pRe = ac - bd;
    pIm = ad + bc;
    if (CtbsvIsNan(pRe) && CtbsvIsNan(pIm)) {
        CtbsvCmulRecover(aRe, aIm, bRe, bIm, ac, bd, ad, bc, pRe, pIm);
    }
}

// Finite hot path: 4 muls, no NaN recover. Perf cases stay in range; overflow EX still
// has special-value align + residual fallback.
CTBSV_ARITH_ATTR void CtbsvCmulFast(
    float aRe, float aIm, float bRe, float bIm, float& pRe, float& pIm)
{
    pRe = aRe * bRe - aIm * bIm;
    pIm = aRe * bIm + aIm * bRe;
}

CTBSV_ARITH_ATTR void CtbsvDivZeroDiag(
    float sumRe, float sumIm, float diagRe, float& outRe, float& outIm)
{
    float signedInf = CtbsvCopysign(CtbsvInf(), diagRe);
    outRe = signedInf * sumRe;
    outIm = signedInf * sumIm;
}

CTBSV_ARITH_ATTR bool CtbsvSmithDiv(
    float sumRe, float sumIm, float diagRe, float diagIm, float& outRe, float& outIm, float& denom)
{
    denom = 0.0f;
    if (CtbsvAbsf(diagRe) < CtbsvAbsf(diagIm)) {
        if (diagIm == 0.0f) {
            CtbsvDivZeroDiag(sumRe, sumIm, diagRe, outRe, outIm);
            return false;
        }
        float ratio = diagRe / diagIm;
        denom = (diagRe * ratio) + diagIm;
        if (denom == 0.0f) {
            CtbsvDivZeroDiag(sumRe, sumIm, diagRe, outRe, outIm);
            return false;
        }
        outRe = (sumRe * ratio + sumIm) / denom;
        outIm = (sumIm * ratio - sumRe) / denom;
        return true;
    }
    if (diagRe == 0.0f) {
        CtbsvDivZeroDiag(sumRe, sumIm, diagRe, outRe, outIm);
        return false;
    }
    float ratio = diagIm / diagRe;
    denom = (diagIm * ratio) + diagRe;
    if (denom == 0.0f) {
        CtbsvDivZeroDiag(sumRe, sumIm, diagRe, outRe, outIm);
        return false;
    }
    outRe = (sumRe + sumIm * ratio) / denom;
    outIm = (sumIm - sumRe * ratio) / denom;
    return true;
}

CTBSV_ARITH_ATTR void CtbsvComplexDivRecover(
    float sumRe, float sumIm, float diagRe, float diagIm, float denom, float& outRe, float& outIm)
{
    if (denom == 0.0f && (!CtbsvIsNan(sumRe) || !CtbsvIsNan(sumIm))) {
        CtbsvDivZeroDiag(sumRe, sumIm, diagRe, outRe, outIm);
        return;
    }
    if ((CtbsvIsInf(sumRe) || CtbsvIsInf(sumIm)) && CtbsvIsFinite(diagRe) && CtbsvIsFinite(diagIm)) {
        float a = CtbsvCopysign(CtbsvIsInf(sumRe) ? 1.0f : 0.0f, sumRe);
        float b = CtbsvCopysign(CtbsvIsInf(sumIm) ? 1.0f : 0.0f, sumIm);
        const float inf = CtbsvInf();
        outRe = inf * (a * diagRe + b * diagIm);
        outIm = inf * (b * diagRe - a * diagIm);
        return;
    }
    if ((CtbsvIsInf(diagRe) || CtbsvIsInf(diagIm)) && CtbsvIsFinite(sumRe) && CtbsvIsFinite(sumIm)) {
        float c = CtbsvCopysign(CtbsvIsInf(diagRe) ? 1.0f : 0.0f, diagRe);
        float d = CtbsvCopysign(CtbsvIsInf(diagIm) ? 1.0f : 0.0f, diagIm);
        outRe = 0.0f * (sumRe * c + sumIm * d);
        outIm = 0.0f * (sumIm * c - sumRe * d);
    }
}

// libgcc __divsc3 / Smith cdiv, then recover 0-diag and overflow.
CTBSV_ARITH_ATTR void CtbsvComplexDiv(
    float sumRe, float sumIm, float diagRe, float diagIm, float& outRe, float& outIm)
{
    float denom = 0.0f;
    if (!CtbsvSmithDiv(sumRe, sumIm, diagRe, diagIm, outRe, outIm, denom)) {
        return;
    }
    if (CtbsvIsNan(outRe) && CtbsvIsNan(outIm)) {
        CtbsvComplexDivRecover(sumRe, sumIm, diagRe, diagIm, denom, outRe, outIm);
    }
}

CTBSV_ARITH_ATTR void CtbsvComplexDivByConj(
    float sumRe, float sumIm, float diagRe, float diagIm, float& outRe, float& outIm)
{
    CtbsvComplexDiv(sumRe, sumIm, diagRe, -diagIm, outRe, outIm);
}

#endif // CTBSV_COMPLEX_ARITH_H
