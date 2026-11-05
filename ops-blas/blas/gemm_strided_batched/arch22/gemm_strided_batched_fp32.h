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
#include <cstdint>

namespace GemmSbFp32 {
union Bits {
    float value;
    uint32_t bits;
};

__aicore__ inline uint32_t HighestBit(uint64_t value)
{
    uint32_t result = 0;
    for (uint32_t step = 32; step != 0; step >>= 1) {
        if (value >> step) {
            value >>= step;
            result += step;
        }
    }
    return result;
}

__aicore__ inline uint64_t ShiftJam(uint64_t value, uint32_t shift)
{
    if (shift == 0)
        return value;
    if (shift >= 64)
        return value != 0;
    return (value >> shift) | ((value & ((uint64_t(1) << shift) - 1)) != 0);
}

__aicore__ inline uint64_t RoundEven(uint64_t value, int shift)
{
    if (shift <= 0)
        return value << -shift;
    if (shift >= 63)
        return 0;
    const uint64_t result = value >> shift;
    const uint64_t remainder = value & ((uint64_t(1) << shift) - 1);
    const uint64_t halfway = uint64_t(1) << (shift - 1);
    return result + (remainder > halfway || (remainder == halfway && (result & 1)));
}

// Rare extreme-input path: retain the full 48-bit product before the single FP32 rounding.
// Bit 60 normalization leaves guard precision even when nearly equal terms cancel.
__aicore__ inline uint32_t FmaBits(uint32_t a, uint32_t b, uint32_t c)
{
    const uint32_t aa = a & 0x7fffffffU, bb = b & 0x7fffffffU, cc = c & 0x7fffffffU;
    const uint32_t sign = (a ^ b) & 0x80000000U, signC = c & 0x80000000U;
    if (aa > 0x7f800000U || bb > 0x7f800000U || cc > 0x7f800000U)
        return 0x7fc00000U;
    if (aa == 0x7f800000U || bb == 0x7f800000U) {
        if (aa == 0 || bb == 0 || (cc == 0x7f800000U && sign != signC))
            return 0x7fc00000U;
        return sign | 0x7f800000U;
    }
    if (cc == 0x7f800000U)
        return c;
    if (aa == 0 || bb == 0)
        return cc == 0 ? sign & signC : c;
    const uint32_t ea = aa >> 23, eb = bb >> 23, ec = cc >> 23;
    const uint64_t ma = (aa & 0x7fffffU) | (ea ? 0x800000U : 0U);
    const uint64_t mb = (bb & 0x7fffffU) | (eb ? 0x800000U : 0U);
    const uint64_t mc = (cc & 0x7fffffU) | (ec ? 0x800000U : 0U);
    const uint64_t product = ma * mb;
    const uint32_t hp = HighestBit(product), hc = HighestBit(mc);
    const int ep = static_cast<int>((ea ? ea : 1) + (eb ? eb : 1)) - 300 + hp;
    const int eC = static_cast<int>(ec ? ec : 1) - 150 + hc;
    const int exponent = mc != 0 && eC > ep ? eC : ep;
    const uint64_t vp = ShiftJam(product << (60 - hp), exponent - ep);
    const uint64_t vc = mc == 0 ? 0 : ShiftJam(mc << (60 - hc), exponent - eC);
    uint32_t resultSign = sign;
    uint64_t magnitude;
    if (sign == signC) {
        magnitude = vp + vc;
    } else if (vp >= vc) {
        magnitude = vp - vc;
    } else {
        magnitude = vc - vp;
        resultSign = signC;
    }
    if (magnitude == 0)
        return 0;
    const int high = HighestBit(magnitude);
    int resultExponent = exponent - 60 + high;
    if (resultExponent < -126) {
        const uint32_t fraction = static_cast<uint32_t>(RoundEven(magnitude, -exponent - 89));
        return resultSign | fraction;
    }
    uint64_t mantissa = RoundEven(magnitude, high - 23);
    if (mantissa == 0x1000000U) {
        mantissa >>= 1;
        ++resultExponent;
    }
    if (resultExponent > 127)
        return resultSign | 0x7f800000U;
    return resultSign | (static_cast<uint32_t>(resultExponent + 127) << 23) |
           (static_cast<uint32_t>(mantissa) & 0x7fffffU);
}

__aicore__ inline float Fma(float a, float b, float c)
{
    Bits av{a}, bv{b}, cv{c};
    Bits result;
    result.bits = FmaBits(av.bits, bv.bits, cv.bits);
    return result.value;
}
} // namespace GemmSbFp32
