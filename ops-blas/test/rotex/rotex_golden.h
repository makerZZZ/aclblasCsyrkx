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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// ─────────────────────────────────────────────────────────────────────────────
// FP16 <-> float conversion
// ─────────────────────────────────────────────────────────────────────────────
inline uint16_t rotExFloatToFp16(float v)
{
    uint32_t x = *reinterpret_cast<const uint32_t*>(&v);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu)
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));
    if (exp >= 0x1F)
        return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t halfMant = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t halfWay = 1u << (shift - 1u);
        if (rem > halfWay || (rem == halfWay && (halfMant & 1u))) halfMant++;
        return static_cast<uint16_t>(sign | halfMant);
    }
    uint32_t halfMant = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (halfMant & 1u))) {
        halfMant++;
        if (halfMant == 0x400u) { halfMant = 0; exp++; }
    }
    if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | halfMant);
}

inline float rotExFp16ToFloat(uint16_t h)
{
    uint32_t hsign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t hexp = (h >> 10) & 0x1Fu;
    uint32_t hmant = h & 0x3FFu;
    uint32_t f;
    if (hexp == 0) {
        if (hmant == 0) { f = hsign; }
        else {
            int e = -1;
            do { hmant <<= 1; e++; } while ((hmant & 0x400u) == 0);
            hmant &= 0x3FFu;
            f = hsign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (hmant << 13);
        }
    } else if (hexp == 0x1Fu) {
        f = hsign | 0x7F800000u | (hmant << 13);
    } else {
        f = hsign | (static_cast<uint32_t>(hexp - 15 + 127) << 23) | (hmant << 13);
    }
    float out = *reinterpret_cast<float*>(&f);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// BF16 <-> float conversion (round-to-nearest-even)
// ─────────────────────────────────────────────────────────────────────────────
inline uint16_t rotExFloatToBf16(float v)
{
    uint32_t bits = *reinterpret_cast<const uint32_t*>(&v);
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

inline float rotExBf16ToFloat(uint16_t h)
{
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    return *reinterpret_cast<float*>(&bits);
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantisation round-trip: float -> target dtype -> float
// ─────────────────────────────────────────────────────────────────────────────
inline float rotExCastToDtype(float v, int32_t dtype)
{
    if (dtype == static_cast<int32_t>(ACL_FLOAT16))
        return rotExFp16ToFloat(rotExFloatToFp16(v));
    else if (dtype == static_cast<int32_t>(ACL_BF16))
        return rotExBf16ToFloat(rotExFloatToBf16(v));
    return v; // FP32: no-op
}

// ─────────────────────────────────────────────────────────────────────────────
// Element size helpers (in bytes and in floats)
// ─────────────────────────────────────────────────────────────────────────────
inline size_t rotExTypeSize(int32_t dtype)
{
    switch (static_cast<aclDataType>(dtype)) {
        case ACL_FLOAT:      return 4;
        case ACL_FLOAT16:    return 2;
        case ACL_BF16:       return 2;
        default:             return 4;
    }
}

inline int rotExElemFloats(int32_t dtype)
{
    switch (static_cast<aclDataType>(dtype)) {
        case ACL_FLOAT:      return 1;
        case ACL_FLOAT16:    return 1;  // host stores as float
        case ACL_BF16:       return 1;  // host stores as float
        default:             return 1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// aclblasRotEx_cpu -- reference implementation
//
// S group (FP32/FP16/BF16) only.  Direct float-precision implementation.
// Signature matches the NPU API exactly (void* pointers for type flexibility).
// ─────────────────────────────────────────────────────────────────────────────
inline aclblasStatus_t aclblasRotEx_cpu(
    aclblasHandle_t handle,
    int n,
    void *x,
    aclDataType xType,
    int incx,
    void *y,
    aclDataType yType,
    int incy,
    const void *c,
    const void *s,
    aclDataType csType,
    aclDataType executionType)
{
    // ── Parameter validation ──
    if (handle == nullptr) return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    {
        aclblasStatus_t _st = RotExValidateCommonParams(n, x, y, c, s, incx, incy);
        if (_st != ACLBLAS_STATUS_SUCCESS) return _st;
    }

    // ── Read c, s values based on csType (S group only: FP32, FP16, BF16) ──
    double cReal = 0.0, sReal = 0.0;

    auto cs = static_cast<aclDataType>(csType);
    if (cs == ACL_FLOAT) {
        cReal = static_cast<double>(*static_cast<const float*>(c));
        sReal = static_cast<double>(*static_cast<const float*>(s));
    } else if (cs == ACL_FLOAT16) {
        uint16_t hc = *static_cast<const uint16_t*>(c);
        uint16_t hs = *static_cast<const uint16_t*>(s);
        cReal = static_cast<double>(rotExFp16ToFloat(hc));
        sReal = static_cast<double>(rotExFp16ToFloat(hs));
    } else if (cs == ACL_BF16) {
        uint16_t hc = *static_cast<const uint16_t*>(c);
        uint16_t hs = *static_cast<const uint16_t*>(s);
        cReal = static_cast<double>(rotExBf16ToFloat(hc));
        sReal = static_cast<double>(rotExBf16ToFloat(hs));
    }

    float* xFloat = static_cast<float*>(x);
    float* yFloat = static_cast<float*>(y);

    int absIncX = std::abs(incx);
    int absIncY = std::abs(incy);
    int xEF = rotExElemFloats(static_cast<int32_t>(xType));
    int yEF = rotExElemFloats(static_cast<int32_t>(yType));
    int xStride = absIncX * xEF;
    int yStride = absIncY * yEF;

    // ── Float path (S group: FP32, BF16, FP16) ──
    // Quantise input to target dtype (simulates NPU input path)
    int32_t xDtype = static_cast<int32_t>(xType);
    int32_t yDtype = static_cast<int32_t>(yType);
    int32_t csDtype = static_cast<int32_t>(csType);

    // Pre-quantize c and s
    double cQ = static_cast<double>(rotExCastToDtype(static_cast<float>(cReal), csDtype));
    double sQ = static_cast<double>(rotExCastToDtype(static_cast<float>(sReal), csDtype));

    for (int i = 0; i < n; i++) {
        int xPos = (incx > 0) ? i * xStride : (n - 1 - i) * xStride;
        int yPos = (incy > 0) ? i * yStride : (n - 1 - i) * yStride;

        // Read and quantise x and y to target dtype
        double xVal = static_cast<double>(rotExCastToDtype(xFloat[xPos], xDtype));
        double yVal = static_cast<double>(rotExCastToDtype(yFloat[yPos], yDtype));

        // Compute rotation in double precision
        double xNew = cQ * xVal + sQ * yVal;
        double yNew = (-sQ) * xVal + cQ * yVal;  // use original xVal

        // Quantise result and write back as float
        xFloat[xPos] = rotExCastToDtype(static_cast<float>(xNew), xDtype);
        yFloat[yPos] = rotExCastToDtype(static_cast<float>(yNew), yDtype);
    }

    return ACLBLAS_STATUS_SUCCESS;
}
