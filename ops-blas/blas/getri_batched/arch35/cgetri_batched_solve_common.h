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
 * \file cgetri_batched_solve_common.h
 * \brief 直接求解和分块求解共用的 SIMD 三角求解实现。
 */

#pragma once

#include "cgetri_batched_kernel_common.h"

namespace CgetriBatched {

template <bool Packed>
__simd_callee__ __attribute__((always_inline)) inline void CgetriReciprocalsVf(
    __ubuf__ float* ar, __ubuf__ float* ai, uint32_t n)
{
    Reg::RegTensor<uint32_t> indices, lanes, rows, lowBits, seven;
    Reg::RegTensor<float> real, imag, absReal, absImag, large, small, ratio, scale, one, inverse, product;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    uint32_t count = Packed ? n * 8 : n;
    uint32_t luLd = (n + 7) / 8 * 8;
    Reg::Duplicate(one, 1.0f);
    Reg::Duplicate(seven, 7u);
    for (uint16_t first = 0; first < (Packed ? n * 8 : n); first += 64) {
        auto mask = Reg::UpdateMask<float>(count);
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lanes), static_cast<int32_t>(first));
        if constexpr (Packed) {
            Reg::ShiftRights(rows, lanes, static_cast<int16_t>(3), all);
            Reg::And(lowBits, lanes, seven, all);
            Reg::Muls(indices, rows, (luLd + 1u) * 8u, all);
            Reg::Add(indices, indices, lowBits, all);
        } else {
            Reg::Muls(indices, lanes, luLd + 1, all);
        }
        Reg::Gather(real, ar, indices, mask);
        Reg::Gather(imag, ai, indices, mask);
        Reg::Abs(absReal, real, mask);
        Reg::Abs(absImag, imag, mask);
        Reg::MaskReg realMajor;
        Reg::Compare<float, CMPMODE::GE>(realMajor, absReal, absImag, mask);
        Reg::Select(large, real, imag, realMajor);
        Reg::Select(small, imag, real, realMajor);
        Reg::Div(ratio, small, large, mask);
        Reg::Mul(scale, small, ratio, mask);
        Reg::Add(scale, large, scale, mask);
        Reg::Div(inverse, one, scale, mask);
        Reg::Mul(product, ratio, inverse, mask);
        Reg::Select(real, inverse, product, realMajor);
        Reg::Select(imag, product, inverse, realMajor);
        Reg::Neg(real, real, mask);
        Reg::Scatter(ar, real, indices, mask);
        Reg::Scatter(ai, imag, indices, mask);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename FloatRegister, typename MaskRegister>
__simd_callee__ __attribute__((always_inline)) inline void CgetriUpdateRhsVf(
    __ubuf__ float* xr, __ubuf__ float* xi, uint32_t offset, FloatRegister& fr, FloatRegister& fi, FloatRegister& jr,
    FloatRegister& ji, FloatRegister& negativeJi, MaskRegister& mask)
{
    Reg::RegTensor<float> ir, ii;
    Reg::LoadAlign(ir, xr + offset);
    Reg::LoadAlign(ii, xi + offset);
    Reg::MulAddDst(ir, fr, jr, mask);
    Reg::MulAddDst(ir, fi, negativeJi, mask);
    Reg::MulAddDst(ii, fr, ji, mask);
    Reg::MulAddDst(ii, fi, jr, mask);
    Reg::StoreAlign(xr + offset, ir, mask);
    Reg::StoreAlign(xi + offset, ii, mask);
}

template <bool Packed>
__simd_callee__ __attribute__((always_inline)) inline void CgetriSolveLowerVf(
    __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi, uint32_t n, uint32_t lowerStart)
{
    uint32_t luLd = (n + 7) / 8 * 8;
    Reg::RegTensor<float> jr, ji, fr, fi, p3;
    auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    // Full-width induction variables also preserve the nested forward
    // solve with the CANN Debug (-O0) code generator.
    for (uint32_t j = lowerStart; j < n; j++) {
        Reg::LoadAlign(jr, xr + j * 72);
        Reg::LoadAlign(ji, xi + j * 72);
        Reg::Neg(p3, ji, mask);
        for (uint32_t row = j + 1; row < n; row++) {
            Reg::LoadAlign<float, Packed ? Reg::LoadDist::DIST_BLK : Reg::LoadDist::DIST_BRC_B32>(
                fr, ar + (row + j * luLd) * (Packed ? 8 : 1));
            Reg::LoadAlign<float, Packed ? Reg::LoadDist::DIST_BLK : Reg::LoadDist::DIST_BRC_B32>(
                fi, ai + (row + j * luLd) * (Packed ? 8 : 1));
            CgetriUpdateRhsVf(xr, xi, row * 72, fr, fi, jr, ji, p3, mask);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

template <uint32_t FixedN, bool Packed = false, uint32_t Phase = 0, bool KeepReciprocals = false>
__simd_vf__ inline void CgetriSolveVf(
    __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi, uint32_t dynamicN,
    uint32_t lowerStart)
{
    uint32_t n = FixedN == 0 ? dynamicN : FixedN;
    uint32_t luLd = (n + 7) / 8 * 8;
    if constexpr (Phase != 1 && !KeepReciprocals) {
        CgetriReciprocalsVf<Packed>(ar, ai, n);
    }
    Reg::RegTensor<float> jr, ji, fr, fi, p0, p1, p2, p3;
    Reg::MaskReg mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    if constexpr (Phase != 2) {
        constexpr bool useRhsStart = !Packed && Phase == 0 && FixedN != 64;
        CgetriSolveLowerVf<Packed>(ar, ai, xr, xi, n, useRhsStart ? lowerStart : 0);
    }
    if constexpr (Phase != 1) {
        for (uint16_t reverse = 0; reverse < n; reverse++) {
            uint16_t j = n - 1 - reverse;
            Reg::LoadAlign(jr, xr + j * 72);
            Reg::LoadAlign(ji, xi + j * 72);
            Reg::LoadAlign<float, Packed ? Reg::LoadDist::DIST_BLK : Reg::LoadDist::DIST_BRC_B32>(
                fr, ar + (j + j * luLd) * (Packed ? 8 : 1));
            Reg::LoadAlign<float, Packed ? Reg::LoadDist::DIST_BLK : Reg::LoadDist::DIST_BRC_B32>(
                fi, ai + (j + j * luLd) * (Packed ? 8 : 1));
            Reg::Mul(p0, fr, jr, mask);
            Reg::Mul(p1, fi, ji, mask);
            Reg::Mul(p2, fr, ji, mask);
            Reg::Mul(p3, fi, jr, mask);
            Reg::Sub(jr, p0, p1, mask);
            Reg::Add(ji, p2, p3, mask);
            Reg::StoreAlign(xr + j * 72, jr, mask);
            Reg::StoreAlign(xi + j * 72, ji, mask);
            Reg::Neg(p3, ji, mask);
            for (uint16_t row = 0; row < j; row++) {
                Reg::LoadAlign<float, Packed ? Reg::LoadDist::DIST_BLK : Reg::LoadDist::DIST_BRC_B32>(
                    fr, ar + (row + j * luLd) * (Packed ? 8 : 1));
                Reg::LoadAlign<float, Packed ? Reg::LoadDist::DIST_BLK : Reg::LoadDist::DIST_BRC_B32>(
                    fi, ai + (row + j * luLd) * (Packed ? 8 : 1));
                CgetriUpdateRhsVf(xr, xi, row * 72, fr, fi, jr, ji, p3, mask);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
        }
    }
}

} // namespace CgetriBatched
