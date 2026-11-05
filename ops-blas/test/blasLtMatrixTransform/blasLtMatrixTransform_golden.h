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
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blasLt.h"
#include "blasLtMatrixTransform_param.h"
// ── Complex-layout in-tile permutation offsets (test-side, self-contained) ──
// Golden owns its complex-layout (COL4_4R2_8C / COL32_2R_4R4) in-tile offset derivation locally
// (matching the blasLtMatmul golden范式: golden relies only on test-local code, never a device-side
// header). The two formulas below are the design 1.3.A §3.6 specification verbatim; they are an
// INDEPENDENT re-derivation of the device placement, NOT a copy of the operator header. The
// byte-for-byte agreement with the device is guaranteed by the shared spec and is cross-checked by
// the AnchorCol4_4R2_8C / AnchorCol32_2R_4R4 TEST_F anchors (hand-verified offsets) — so a
// divergence between golden and device cannot pass silently as a "dual error" (closes the I-01
// "dual-error" hole without a single data source).

// Composite tile geometry per complex order (design 1.3.A §3.6).
constexpr uint32_t MT_COL4_4R2_8C_ROWS = 8;
constexpr uint32_t MT_COL4_4R2_8C_COLS = 32;
constexpr uint32_t MT_COL32_2R_4R4_ROWS = 32;
constexpr uint32_t MT_COL32_2R_4R4_COLS = 32;

// COL32_2R_4R4 in-tile element offset: offset(row,col) = rowPerm*32 + col,
// rowPerm = ((row % 8) / 2 * 4 + row / 8) * 2 + row % 2 (design 1.3.A §3.6).
inline uint32_t MtCol32R2R4R4Offset(uint32_t row, uint32_t col)
{
    const uint32_t rowPerm = ((row % 8U) / 2U * 4U + row / 8U) * 2U + row % 2U;
    return rowPerm * MT_COL32_2R_4R4_COLS + col;
}

// COL4_4R2_8C in-tile element offset (design 1.3.A §3.6).
inline uint32_t MtCol44R28COffset(uint32_t row, uint32_t col)
{
    const uint32_t colOuter = col / 4U;
    const uint32_t colInner = col % 4U;
    const uint32_t rowPair = row / 2U;
    const uint32_t rowInPair = row % 2U;
    return ((rowPair * 8U + colOuter) * 8U) + (rowInPair * 4U + colInner);
}

// ── BF16 / FP16 round-trip (match device store/load semantics) ──
inline float ltTransformBf16RoundTrip(float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(float));
    // round-to-nearest-even on the truncated 16 low bits
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t roundBias = 0x7FFFu + lsb;
    bits += roundBias;
    uint16_t bf16 = static_cast<uint16_t>(bits >> 16);
    uint32_t back = static_cast<uint32_t>(bf16) << 16;
    float out;
    std::memcpy(&out, &back, sizeof(float));
    return out;
}

inline float ltTransformFp16RoundTrip(float v)
{
    // IEEE754 binary16 round-trip via bit manipulation (round-to-nearest-even).
    uint32_t x;
    std::memcpy(&x, &v, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    uint16_t h;
    if (((x >> 23) & 0xFF) == 0xFF) {
        h = static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));  // inf/nan
    } else if (exp >= 0x1F) {
        h = static_cast<uint16_t>(sign | 0x7C00u);  // overflow → inf
    } else if (exp <= 0) {
        if (exp < -10) {
            h = static_cast<uint16_t>(sign);
        } else {
            mant |= 0x800000u;
            int shift = 14 - exp;
            uint32_t halfMant = mant >> shift;
            uint32_t rem = mant & ((1u << shift) - 1);
            uint32_t halfWay = 1u << (shift - 1);
            if (rem > halfWay || (rem == halfWay && (halfMant & 1))) halfMant++;
            h = static_cast<uint16_t>(sign | halfMant);
        }
    } else {
        uint32_t halfMant = mant >> 13;
        uint32_t rem = mant & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (halfMant & 1))) {
            halfMant++;
            if (halfMant == 0x400u) { halfMant = 0; exp++; }
        }
        if (exp >= 0x1F) h = static_cast<uint16_t>(sign | 0x7C00u);
        else h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | halfMant);
    }
    // decode back to float
    uint32_t hsign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t hexp = (h >> 10) & 0x1F;
    uint32_t hmant = h & 0x3FFu;
    uint32_t f;
    if (hexp == 0) {
        if (hmant == 0) {
            f = hsign;
        } else {
            int e = -1;
            do { hmant <<= 1; e++; } while ((hmant & 0x400u) == 0);
            hmant &= 0x3FFu;
            f = hsign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (hmant << 13);
        }
    } else if (hexp == 0x1F) {
        f = hsign | 0x7F800000u | (hmant << 13);
    } else {
        f = hsign | (static_cast<uint32_t>(hexp - 15 + 127) << 23) | (hmant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

// ── FP8 E4M3FN (1-5-2 sign-exp-mant? no: 1 sign, 4 exp, 3 mant; bias=7; finite-only, no inf) ──
// Device CAST_RINT semantics: round-to-nearest-even, saturate to ±448 (max finite E4M3FN value).
// Decode the 8-bit pattern back to float so golden starts from the same already-quantised value
// the device reads (mirrors the FP16/BF16 round-trip path in ltTransformQuantizeInput).
inline float ltFp8E4m3ToFloat(uint8_t v)
{
    uint32_t sign = (v >> 7) & 1u;
    uint32_t exp = (v >> 3) & 0xFu;
    uint32_t mant = v & 0x7u;
    float value;
    if (exp == 0u) {
        value = static_cast<float>(mant) / 8.0f * std::pow(2.0f, -6.0f);  // subnormal, 2^(1-bias)
    } else if (exp == 0xFu && mant == 0x7u) {
        value = std::numeric_limits<float>::quiet_NaN();                  // E4M3FN: only S.1111.111 is NaN
    } else {
        value = (1.0f + static_cast<float>(mant) / 8.0f) * std::pow(2.0f, static_cast<float>(exp) - 7.0f);
    }
    return sign ? -value : value;
}

inline uint8_t ltFloatToFp8E4m3(float f)
{
    if (std::isnan(f)) return 0x7Fu;
    uint32_t sign = std::signbit(f) ? 0x80u : 0x00u;
    float a = std::fabs(f);
    const float maxFinite = 448.0f;                 // largest E4M3FN magnitude
    if (a >= maxFinite) {
        // saturate to max finite (matches device CAST_RINT saturating cast, no inf in E4M3FN)
        if (a > maxFinite) return static_cast<uint8_t>(sign | 0x7Eu);  // S.1111.110 = 448
    }
    if (a == 0.0f) return static_cast<uint8_t>(sign);
    // search the nearest representable magnitude by RINT in the (exp,mant) lattice.
    // smallest subnormal = 2^-9; representable mantissa step depends on exponent.
    int e = static_cast<int>(std::floor(std::log2(a)));
    int unbExp = e + 7;                              // biased exponent candidate
    uint32_t expBits, mantBits;
    if (unbExp <= 0) {
        // subnormal: value = mant/8 * 2^-6, step = 2^-9
        float scaled = a / std::pow(2.0f, -9.0f);    // in units of the subnormal LSB
        float r = std::nearbyint(scaled);
        if (r > 7.0f) {                              // rounds up into the smallest normal
            expBits = 1u; mantBits = 0u;
        } else {
            expBits = 0u; mantBits = static_cast<uint32_t>(r);
        }
    } else {
        if (unbExp > 15) unbExp = 15;
        float frac = a / std::pow(2.0f, static_cast<float>(unbExp - 7)) - 1.0f;  // in [0,1)
        float r = std::nearbyint(frac * 8.0f);
        if (r > 8.0f) r = 8.0f;
        if (r == 8.0f) { unbExp += 1; r = 0.0f; }    // mantissa carry
        if (unbExp > 15 || (unbExp == 15 && r > 6.0f)) return static_cast<uint8_t>(sign | 0x7Eu);  // saturate
        expBits = static_cast<uint32_t>(unbExp);
        mantBits = static_cast<uint32_t>(r);
    }
    return static_cast<uint8_t>(sign | (expBits << 3) | (mantBits & 0x7u));
}

// ── FP8 E5M2 (1 sign, 5 exp, 2 mant; bias=15; IEEE-style with inf/nan) ──
inline float ltFp8E5m2ToFloat(uint8_t v)
{
    uint32_t sign = (v >> 7) & 1u;
    uint32_t exp = (v >> 2) & 0x1Fu;
    uint32_t mant = v & 0x3u;
    float value;
    if (exp == 0u) {
        value = static_cast<float>(mant) / 4.0f * std::pow(2.0f, -14.0f);  // subnormal
    } else if (exp == 0x1Fu) {
        value = mant ? std::numeric_limits<float>::quiet_NaN()
                     : std::numeric_limits<float>::infinity();
    } else {
        value = (1.0f + static_cast<float>(mant) / 4.0f) * std::pow(2.0f, static_cast<float>(exp) - 15.0f);
    }
    return sign ? -value : value;
}

inline uint8_t ltFloatToFp8E5m2(float f)
{
    if (std::isnan(f)) return 0x7Fu;
    uint32_t sign = std::signbit(f) ? 0x80u : 0x00u;
    float a = std::fabs(f);
    const float maxFinite = 57344.0f;               // largest E5M2 finite magnitude (1.75 * 2^15)
    if (std::isinf(a) || a > maxFinite) return static_cast<uint8_t>(sign | 0x7Cu);  // overflow → inf
    if (a == 0.0f) return static_cast<uint8_t>(sign);
    int e = static_cast<int>(std::floor(std::log2(a)));
    int unbExp = e + 15;
    uint32_t expBits, mantBits;
    if (unbExp <= 0) {
        float scaled = a / std::pow(2.0f, -16.0f);   // subnormal LSB = 2^-16
        float r = std::nearbyint(scaled);
        if (r > 3.0f) { expBits = 1u; mantBits = 0u; }
        else { expBits = 0u; mantBits = static_cast<uint32_t>(r); }
    } else {
        if (unbExp > 30) unbExp = 30;
        float frac = a / std::pow(2.0f, static_cast<float>(unbExp - 15)) - 1.0f;
        float r = std::nearbyint(frac * 4.0f);
        if (r > 4.0f) r = 4.0f;
        if (r == 4.0f) { unbExp += 1; r = 0.0f; }
        if (unbExp >= 31) return static_cast<uint8_t>(sign | 0x7Cu);  // overflow → inf
        expBits = static_cast<uint32_t>(unbExp);
        mantBits = static_cast<uint32_t>(r);
    }
    return static_cast<uint8_t>(sign | (expBits << 2) | (mantBits & 0x3u));
}

// ── FP4 E2M1 (1 sign, 2 exp, 1 mant; bias=1; OCP MXFP4 lattice) ──
// 4-bit nibble decode. Representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
inline float ltFp4E2m1ToFloat(uint8_t nib)
{
    uint32_t sign = (nib >> 3) & 1u;
    uint32_t exp = (nib >> 1) & 0x3u;
    uint32_t mant = nib & 0x1u;
    float value;
    if (exp == 0u) {
        value = (mant == 0u) ? 0.0f : 0.5f;          // subnormal: 0 or 0.5
    } else {
        value = (1.0f + static_cast<float>(mant) * 0.5f) * std::pow(2.0f, static_cast<float>(exp) - 1.0f);
    }
    return sign ? -value : value;
}

inline uint8_t ltFloatToFp4E2m1(float f)
{
    uint32_t sign = std::signbit(f) ? 0x8u : 0x0u;
    float a = std::fabs(f);
    // RINT to nearest representable magnitude (round-half-to-even on the {0,0.5,1,1.5,2,3,4,6} set).
    static const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    static const uint8_t codes[8] = {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u};
    if (a >= 6.0f) return static_cast<uint8_t>(sign | 0x7u);  // saturate to ±6
    int best = 0;
    float bestErr = std::fabs(a - levels[0]);
    for (int i = 1; i < 8; i++) {
        float err = std::fabs(a - levels[i]);
        if (err < bestErr - 1e-12f) { bestErr = err; best = i; }
        else if (std::fabs(err - bestErr) <= 1e-12f) {
            // tie → round to even code (matches round-to-nearest-even quantisation)
            if ((codes[i] & 1u) == 0u) { best = i; }
        }
    }
    return static_cast<uint8_t>(sign | codes[best]);
}

// FP8 round-trip (float → fp8 RINT → float), per variant. Mirrors device store/load quantisation.
inline float ltTransformFp8E4m3RoundTrip(float v) { return ltFp8E4m3ToFloat(ltFloatToFp8E4m3(v)); }
inline float ltTransformFp8E5m2RoundTrip(float v) { return ltFp8E5m2ToFloat(ltFloatToFp8E5m2(v)); }

// FP4 round-trip: float → bf16 → fp4x2 (RINT) → bf16 → float (cast唯一对端 bf16 中转, §4.2.6).
inline float ltTransformFp4E2m1RoundTrip(float v)
{
    float bf = ltTransformBf16RoundTrip(v);                 // device casts via bf16 first
    return ltFp4E2m1ToFloat(ltFloatToFp4E2m1(bf));
}

// ── physical offset for a logical column-major (row,col) element under a linear order ──
// COL : col*ld + row     (ld >= rows)
// ROW : row*ld + col     (ld >= cols)
// COL32: group*ld + row*32 + (col%32), group = col/32  (ld >= rows*32)
inline int64_t ltTransformLinearOffset(
    aclblasLtOrder_t order, int row, int col, int rows, int ld)
{
    switch (order) {
        case ACLBLASLT_ORDER_COL:
            return static_cast<int64_t>(col) * ld + row;
        case ACLBLASLT_ORDER_ROW:
            return static_cast<int64_t>(row) * ld + col;
        case ACLBLASLT_ORDER_COL32: {
            int group = col / 32;
            int colInTile = col % 32;
            return static_cast<int64_t>(group) * ld + static_cast<int64_t>(row) * 32 + colInTile;
        }
        case ACLBLASLT_ORDER_COL4_4R2_8C: {
            // 32-col group × 8-row composite tile. group steps by ld; each 8-row block steps by
            // (32*8); in-tile offset from the test-local perm derivation (Anchor TEST_F cross-checks it).
            int group = col / 32;
            int rowBlock = row / static_cast<int>(MT_COL4_4R2_8C_ROWS);
            uint32_t tileOff =
                MtCol44R28COffset(static_cast<uint32_t>(row % static_cast<int>(MT_COL4_4R2_8C_ROWS)),
                                  static_cast<uint32_t>(col % 32));
            int64_t tileSize = static_cast<int64_t>(MT_COL4_4R2_8C_ROWS) * MT_COL4_4R2_8C_COLS;
            return static_cast<int64_t>(group) * ld + static_cast<int64_t>(rowBlock) * tileSize +
                   static_cast<int64_t>(tileOff);
        }
        case ACLBLASLT_ORDER_COL32_2R_4R4: {
            // 32-col group × 32-row composite tile. In-tile offset = MtCol32R2R4R4Offset (test-local).
            int group = col / 32;
            int rowBlock = row / static_cast<int>(MT_COL32_2R_4R4_ROWS);
            uint32_t tileOff =
                MtCol32R2R4R4Offset(static_cast<uint32_t>(row % static_cast<int>(MT_COL32_2R_4R4_ROWS)),
                                    static_cast<uint32_t>(col % 32));
            int64_t tileSize = static_cast<int64_t>(MT_COL32_2R_4R4_ROWS) * MT_COL32_2R_4R4_COLS;
            return static_cast<int64_t>(group) * ld + static_cast<int64_t>(rowBlock) * tileSize +
                   static_cast<int64_t>(tileOff);
        }
        default:
            return static_cast<int64_t>(col) * ld + row;
    }
}

// quantize each element to its storage dtype (round-trip), matching device store/load semantics.
// cuBLASLt semantics: inputs are physically stored as their declared dtype, so the kernel reads
// already-quantized values before promoting to the scaleType (FP32) compute domain. FP32 / INT8 /
// INT32 inputs are already exactly representable as float, so they pass through unchanged.
inline void ltTransformQuantizeInput(std::vector<float>& nd, aclDataType dtype)
{
    if (dtype == ACL_FLOAT16) {
        for (float& v : nd) v = ltTransformFp16RoundTrip(v);
    } else if (dtype == ACL_BF16) {
        for (float& v : nd) v = ltTransformBf16RoundTrip(v);
    } else if (dtype == ACL_FLOAT8_E4M3FN) {
        for (float& v : nd) v = ltTransformFp8E4m3RoundTrip(v);
    } else if (dtype == ACL_FLOAT8_E5M2) {
        for (float& v : nd) v = ltTransformFp8E5m2RoundTrip(v);
    } else if (dtype == ACL_FLOAT4_E2M1) {
        for (float& v : nd) v = ltTransformFp4E2m1RoundTrip(v);
    }
    // ACL_FLOAT / ACL_INT8 / ACL_INT32: exactly representable, no quantization needed.
}

// de-layout: physical float buffer (in input order) → logical ND (rows×cols, col-major index col*rows+row)
inline std::vector<float> ltTransformDeLayout(
    const std::vector<float>& phys, aclblasLtOrder_t order, int rows, int cols, int ld)
{
    std::vector<float> nd(static_cast<size_t>(rows) * cols, 0.0f);
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            int64_t off = ltTransformLinearOffset(order, r, c, rows, ld);
            if (off >= 0 && off < static_cast<int64_t>(phys.size()))
                nd[static_cast<size_t>(c) * rows + r] = phys[static_cast<size_t>(off)];
        }
    }
    return nd;
}

// re-layout: logical ND (rows×cols, col-major index col*rows+row) → physical buffer (in `order`).
// Inverse of ltTransformDeLayout; used by the I-01 anchor test to materialise the expected
// complex-layout physical placement from the shared permutation header (no device dependency).
inline std::vector<float> ltTransformReLayout(
    const std::vector<float>& nd, aclblasLtOrder_t order, int rows, int cols, int ld)
{
    int64_t physCount = 0;
    for (int c = 0; c < cols; c++)
        for (int r = 0; r < rows; r++)
            physCount = std::max(physCount, ltTransformLinearOffset(order, r, c, rows, ld) + 1);
    std::vector<float> phys(static_cast<size_t>(std::max<int64_t>(physCount, 1)), 0.0f);
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            int64_t off = ltTransformLinearOffset(order, r, c, rows, ld);
            if (off >= 0 && off < static_cast<int64_t>(phys.size()))
                phys[static_cast<size_t>(off)] = nd[static_cast<size_t>(c) * rows + r];
        }
    }
    return phys;
}

// physical (pre-op) rows/cols given the logical op-applied dims.
// Inputs are physically stored in pre-op shape; op=N keeps dims, op=T/C swaps them.
// Mirrors ltPhysDims in the npu wrapper so de-layout walks the real physical buffer extent.
inline void ltTransformPhysDims(int opRows, int opCols, aclblasOperation_t op, int& physRows, int& physCols)
{
    if (op == ACLBLAS_OP_N) { physRows = opRows; physCols = opCols; }
    else { physRows = opCols; physCols = opRows; }
}

// apply op (N / T / C≡T) to a logical ND matrix (col-major index)
inline std::vector<float> ltTransformApplyOp(
    const std::vector<float>& nd, aclblasOperation_t op, int rows, int cols, int& outRows, int& outCols)
{
    if (op == ACLBLAS_OP_N) {
        outRows = rows; outCols = cols;
        return nd;
    }
    // transpose a col-major (rows×cols) matrix `nd` (index = col*rows+row, per ltTransformDeLayout)
    // into a col-major (cols×rows) matrix `out` (index = outCol*outRows+outRow).
    // out[newRow=origCol, newCol=origRow] = nd[origRow, origCol]; outRows = cols.
    // NOTE: the source stride MUST be `rows` (nd's col-major row count), not `cols`. Using `cols`
    // only happens to be correct for square matrices and is the non-square op=T de-layout bug.
    outRows = cols; outCols = rows;
    std::vector<float> out(static_cast<size_t>(outRows) * outCols, 0.0f);
    for (int origCol = 0; origCol < cols; origCol++)
        for (int origRow = 0; origRow < rows; origRow++)
            out[static_cast<size_t>(origRow) * outRows + origCol] =
                nd[static_cast<size_t>(origCol) * rows + origRow];
    return out;
}

// cast a logical value to output dtype storage semantics, returning the comparable float.
inline float ltTransformCastToOut(double v, aclDataType outDtype)
{
    switch (outDtype) {
        case ACL_FLOAT:
            return static_cast<float>(v);
        case ACL_FLOAT16:
            return ltTransformFp16RoundTrip(static_cast<float>(v));
        case ACL_BF16:
            return ltTransformBf16RoundTrip(static_cast<float>(v));
        case ACL_INT32: {
            double r = std::nearbyint(v);
            if (r > 2147483647.0) r = 2147483647.0;
            if (r < -2147483648.0) r = -2147483648.0;
            return static_cast<float>(static_cast<int32_t>(r));
        }
        case ACL_INT8: {
            double r = std::nearbyint(v);
            if (r > 127.0) r = 127.0;     // saturate (§4.2.4)
            if (r < -128.0) r = -128.0;
            return static_cast<float>(static_cast<int8_t>(static_cast<int>(r)));
        }
        case ACL_FLOAT8_E4M3FN:
            return ltTransformFp8E4m3RoundTrip(static_cast<float>(v));   // float→fp8(RINT)→float
        case ACL_FLOAT8_E5M2:
            return ltTransformFp8E5m2RoundTrip(static_cast<float>(v));
        case ACL_FLOAT4_E2M1:
            return ltTransformFp4E2m1RoundTrip(static_cast<float>(v));   // bf16→fp4x2(RINT) write-back
        default:
            return static_cast<float>(v);
    }
}

// ── CPU golden (host, column-major basis). Mirrors aclblasLtMatrixTransform semantics. ──
// out: logical column-major (rowsC×colsC), index col*rowsC+row. Caller compares against device
//      result that has been de-layouted to the same logical ND order.
inline aclblasStatus_t aclblasLtMatrixTransform_cpu(
    aclblasLtHandle_t lightHandle,
    aclDataType dtypeA, aclblasLtOrder_t orderA, aclblasOperation_t transA,
    int rowsA, int colsA, int lda, const std::vector<float>& A,
    aclDataType dtypeB, aclblasLtOrder_t orderB, aclblasOperation_t transB,
    int rowsB, int colsB, int ldb, const std::vector<float>& B, bool hasB,
    aclDataType dtypeC, int rowsC, int colsC,
    aclDataType scaleType, float alpha, float beta,
    std::vector<float>& out)
{
    if (lightHandle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (rowsC == 0 || colsC == 0) { out.clear(); return ACLBLAS_STATUS_SUCCESS; }

    const bool intPath = (scaleType == ACL_INT32);

    // de-layout → quantize input to storage dtype → op(A)
    // Inputs are physically stored in their PRE-OP physical shape (op=T/C swaps rows/cols vs the
    // logical op-applied dims). De-layout MUST walk that physical extent — using the logical dims
    // for a non-square op=T input would index past the real buffer (the device reads physical dims,
    // then op transposes). The square op=T case hides this because physical == logical dims.
    // Inputs are physically stored as their declared dtype (FP16/BF16 lose precision on store),
    // so golden must quantize the de-layouted elements before op / scale-add to match device.
    int physRowsA = 0, physColsA = 0;
    ltTransformPhysDims(rowsA, colsA, transA, physRowsA, physColsA);
    std::vector<float> ndA = ltTransformDeLayout(A, orderA, physRowsA, physColsA, lda);
    ltTransformQuantizeInput(ndA, dtypeA);
    int opRowsA = 0, opColsA = 0;
    std::vector<float> opA = ltTransformApplyOp(ndA, transA, physRowsA, physColsA, opRowsA, opColsA);

    std::vector<float> opB;
    int opRowsB = 0, opColsB = 0;
    if (hasB) {
        int physRowsB = 0, physColsB = 0;
        ltTransformPhysDims(rowsB, colsB, transB, physRowsB, physColsB);
        std::vector<float> ndB = ltTransformDeLayout(B, orderB, physRowsB, physColsB, ldb);
        ltTransformQuantizeInput(ndB, dtypeB);
        opB = ltTransformApplyOp(ndB, transB, physRowsB, physColsB, opRowsB, opColsB);
    }

    out.assign(static_cast<size_t>(rowsC) * colsC, 0.0f);
    for (int c = 0; c < colsC; c++) {
        for (int r = 0; r < rowsC; r++) {
            double a = static_cast<double>(opA[static_cast<size_t>(c) * opRowsA + r]);
            double tmp;
            if (intPath) {
                // integer scale-add in INT32 domain (alpha/beta as int32)
                long long ai = static_cast<long long>(std::llround(a));
                long long alphaI = static_cast<long long>(std::llround(static_cast<double>(alpha)));
                long long acc = alphaI * ai;
                if (hasB) {
                    long long bi = static_cast<long long>(std::llround(
                        static_cast<double>(opB[static_cast<size_t>(c) * opRowsB + r])));
                    long long betaI = static_cast<long long>(std::llround(static_cast<double>(beta)));
                    acc += betaI * bi;
                }
                tmp = static_cast<double>(acc);
            } else {
                tmp = static_cast<double>(alpha) * a;
                if (hasB)
                    tmp += static_cast<double>(beta) *
                           static_cast<double>(opB[static_cast<size_t>(c) * opRowsB + r]);
            }
            out[static_cast<size_t>(c) * rowsC + r] = ltTransformCastToOut(tmp, dtypeC);
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

