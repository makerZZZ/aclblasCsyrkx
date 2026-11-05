/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

// General strided and non-finite fallback follows the reference's FP32 evaluation order.
// Conservative range for reordered products; includes non-finite bit patterns.
__simt_callee__ __aicore__ inline bool CsyrkUnsafeValue(float value)
{
    union {
        float value;
        uint32_t bits;
    } part{value};
    const uint32_t magnitude = part.bits & 0x7fffffffU;
    return magnitude > 0x5a0e1bcaU || (magnitude != 0 && magnitude < 0x24e69595U);
}

__simt_callee__ __aicore__ inline float CsyrkCopySign(float magnitude, float sign)
{
    union FloatBits {
        float value;
        uint32_t bits;
    };
    FloatBits mag{magnitude};
    FloatBits source{sign};
    // The SIMT backend does not lower fcopysign; keep the sign operation integral.
    volatile uint32_t signBit = source.bits & 0x80000000U;
    mag.bits = (mag.bits & 0x7fffffffU) | signBit;
    return mag.value;
}

struct CsyrkValue {
    float real;
    float imag;
};

// C complex multiplication recovers infinities when both provisional components are NaN.
// See the __mulsc3 semantics in compiler-rt; the finite path keeps FP32 intermediates.
__simt_callee__ __aicore__ inline CsyrkValue CsyrkMultiply(float ar, float ai, float br, float bi)
{
#pragma clang fp contract(off)
    volatile float rr = ar * br;
    volatile float ii = ai * bi;
    volatile float ri = ar * bi;
    volatile float ir = ai * br;
    CsyrkValue result{rr - ii, ri + ir};
    if (__builtin_isnan(result.real) && __builtin_isnan(result.imag)) {
        const bool leftInfinite = __builtin_isinf(ar) || __builtin_isinf(ai);
        const bool rightInfinite = __builtin_isinf(br) || __builtin_isinf(bi);
        const bool productInfinite =
            __builtin_isinf(rr) || __builtin_isinf(ii) || __builtin_isinf(ri) || __builtin_isinf(ir);
        if (leftInfinite || rightInfinite || productInfinite) {
            ar = leftInfinite ? CsyrkCopySign(__builtin_isinf(ar) ? 1.0f : 0.0f, ar) :
                                (__builtin_isnan(ar) ? CsyrkCopySign(0.0f, ar) : ar);
            ai = leftInfinite ? CsyrkCopySign(__builtin_isinf(ai) ? 1.0f : 0.0f, ai) :
                                (__builtin_isnan(ai) ? CsyrkCopySign(0.0f, ai) : ai);
            br = rightInfinite ? CsyrkCopySign(__builtin_isinf(br) ? 1.0f : 0.0f, br) :
                                 (__builtin_isnan(br) ? CsyrkCopySign(0.0f, br) : br);
            bi = rightInfinite ? CsyrkCopySign(__builtin_isinf(bi) ? 1.0f : 0.0f, bi) :
                                 (__builtin_isnan(bi) ? CsyrkCopySign(0.0f, bi) : bi);
            volatile float r = ar * br - ai * bi;
            volatile float i = ar * bi + ai * br;
            result.real = __builtin_inff() * r;
            result.imag = __builtin_inff() * i;
        }
    }
    return result;
}

__simt_callee__ __aicore__ inline CsyrkValue CsyrkReferenceTranspose(
    __gm__ float* a, uint32_t row, uint32_t col, uint32_t k, uint32_t lda, float ar, float ai)
{
#pragma clang fp contract(off)
    float pr = 0.0f;
    float pi = 0.0f;
    for (uint32_t l = 0; l < k; ++l) {
        const uint64_t x = (static_cast<uint64_t>(row) * lda + l) * 2;
        const uint64_t y = (static_cast<uint64_t>(col) * lda + l) * 2;
        const float xr = a[x];
        const float xi = a[x + 1];
        const float yr = a[y];
        const float yi = a[y + 1];
        const auto product = CsyrkMultiply(xr, xi, yr, yi);
        pr += product.real;
        pi += product.imag;
    }
    const auto scaled = CsyrkMultiply(ar, ai, pr, pi);
    return scaled;
}

__simt_callee__ __aicore__ inline void CsyrkReferenceElement(
    __gm__ float* a, __gm__ float* c, uint32_t row, uint32_t col, uint32_t k, uint32_t lda, uint32_t ldc,
    uint32_t transposed, float ar, float ai, float br, float bi)
{
#pragma clang fp contract(off)
    const bool skip = k == 0 || (ar == 0.0f && ai == 0.0f);
    const bool zeroBeta = br == 0.0f && bi == 0.0f;
    const bool oneBeta = br == 1.0f && bi == 0.0f;
    if (skip && oneBeta)
        return;
    const uint64_t output = (static_cast<uint64_t>(col) * ldc + row) * 2;
    float real = 0.0f;
    float imag = 0.0f;
    if (!zeroBeta) {
        const float cr = c[output];
        const float ci = c[output + 1];
        const auto scaled = oneBeta && transposed == 0 ? CsyrkValue{cr, ci} : CsyrkMultiply(br, bi, cr, ci);
        real = scaled.real;
        imag = scaled.imag;
    }
    if (!skip && transposed == 0) {
        for (uint32_t l = 0; l < k; ++l) {
            const uint64_t y = (static_cast<uint64_t>(l) * lda + col) * 2;
            const float yr = a[y];
            const float yi = a[y + 1];
            if (yr == 0.0f && yi == 0.0f)
                continue;
            const bool oneAlpha = ar == 1.0f && ai == 0.0f;
            const auto scaled = oneAlpha ? CsyrkValue{yr, yi} : CsyrkMultiply(ar, ai, yr, yi);
            const uint64_t x = (static_cast<uint64_t>(l) * lda + row) * 2;
            const float xr = a[x];
            const float xi = a[x + 1];
            const auto product = CsyrkMultiply(scaled.real, scaled.imag, xr, xi);
            real += product.real;
            imag += product.imag;
        }
    } else if (!skip) {
        const auto scaled = CsyrkReferenceTranspose(a, row, col, k, lda, ar, ai);
        real += scaled.real;
        imag += scaled.imag;
    }
    c[output] = real;
    c[output + 1] = imag;
}
