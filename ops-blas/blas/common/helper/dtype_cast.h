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
#include <cstddef>

namespace blas_common {

template <typename To, typename From>
inline To BitCast(const From &value)
{
    static_assert(sizeof(To) == sizeof(From), "BitCast requires same-sized types");
    To result{};
    auto *dst = reinterpret_cast<uint8_t *>(&result);
    const auto *src = reinterpret_cast<const uint8_t *>(&value);
    for (size_t i = 0; i < sizeof(To); ++i) {
        dst[i] = src[i];
    }
    return result;
}

inline float HalfToFloat(uint16_t value)
{
    uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            bits = sign | static_cast<uint32_t>(127 - 14 - shift) << 23 | mantissa << 13;
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | mantissa << 13;
    } else {
        bits = sign | (exponent + 112u) << 23 | mantissa << 13;
    }
    return BitCast<float>(bits);
}

inline uint16_t FloatToHalf(float value)
{
    uint32_t bits = BitCast<uint32_t>(value);
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t abs = bits & 0x7fffffffu;
    if (abs >= 0x7f800000u) {
        uint16_t payload = static_cast<uint16_t>((abs & 0x007fffffu) >> 13);
        return static_cast<uint16_t>(sign | 0x7c00u | payload | (payload == 0 ? 0 : 1));
    }
    int32_t exponent = static_cast<int32_t>((abs >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = abs & 0x007fffffu;
    if (exponent >= 31) { return static_cast<uint16_t>(sign | 0x7c00u); }
    if (exponent <= 0) {
        if (exponent < -10) { return static_cast<uint16_t>(sign); }
        mantissa |= 0x00800000u;
        uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t rounded = (mantissa + ((1u << (shift - 1)) - 1u) + ((mantissa >> shift) & 1u)) >> shift;
        return static_cast<uint16_t>(sign | rounded);
    }
    uint32_t roundedMantissa = mantissa + 0x00000fffu + ((mantissa >> 13) & 1u);
    if ((roundedMantissa & 0x00800000u) != 0) {
        roundedMantissa = 0;
        ++exponent;
        if (exponent >= 31) { return static_cast<uint16_t>(sign | 0x7c00u); }
    }
    return static_cast<uint16_t>(sign | static_cast<uint32_t>(exponent) << 10 |
                                 (roundedMantissa >> 13));
}

inline float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    return BitCast<float>(bits);
}

inline uint16_t FloatToBf16(float value)
{
    uint32_t bits = BitCast<uint32_t>(value);
    const uint32_t absolute = bits & 0x7fffffffu;
    if (absolute >= 0x7f800000u) {
        uint16_t result = static_cast<uint16_t>(bits >> 16);
        // Preserve NaN when its payload only occupies the truncated low bits.
        if (absolute > 0x7f800000u) {
            result = static_cast<uint16_t>(result | 1u);
        }
        return result;
    }
    uint32_t roundingBias = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + roundingBias) >> 16);
}

} // namespace blas_common
