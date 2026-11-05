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
 * \file cgetri_batched_kernel_common.h
 * \brief Cgetri 内核共用常量、复数除法和主元元数据。
 */

#pragma once

#include <cstdint>

#include "cann_ops_blas_common.h"
#include "cgetri_batched_tiling_data.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "c_api/asc_simd.h"

namespace CgetriBatched {

namespace Reg = AscendC::Reg;
using AscendC::CMPMODE;

constexpr uint32_t CGETRI_SIMT_MAX_THREADS = 256;
// Bound the small-matrix SIMT launch to leave room for each lane's complex RHS.
constexpr uint32_t CGETRI_SMALL_N = 8;
constexpr uint32_t CGETRI_SMALL_GROUP_SIZE = 32;
constexpr uint32_t CGETRI_SMALL_GROUP_COUNT = CGETRI_SIMT_MAX_THREADS / CGETRI_SMALL_GROUP_SIZE;
constexpr uint32_t CGETRI_SMALL_MATRIX_ELEMENTS = CGETRI_SMALL_N * CGETRI_SMALL_N;
// The blocked initializer marks proven diagonal matrices until finalization.
// Keep this internal state distinct from both public info and an untouched slot.
constexpr int CGETRI_DIAGONAL_FAST_PATH_INFO = -2;

constexpr uint32_t CGETRI_STREAM_MAX_N = CGETRI_BLOCKED_THRESHOLD_N - 1;
constexpr uint32_t CGETRI_METADATA_PERMUTATION = (CGETRI_STREAM_MAX_N + 63) / 64 * 64;
constexpr uint32_t CGETRI_METADATA_STATUS = CGETRI_METADATA_PERMUTATION + 64;
constexpr uint32_t CGETRI_METADATA_AUX = CGETRI_METADATA_STATUS + 8;
constexpr uint32_t CGETRI_METADATA_ELEMENTS = CGETRI_METADATA_AUX + 8;
static_assert(CGETRI_STREAM_MAX_N % 8 == 0, "Stream LU columns must be aligned");

static_assert(
    (CGETRI_SIMT_MAX_THREADS & (CGETRI_SIMT_MAX_THREADS - 1)) == 0, "CGETRI_SIMT_MAX_THREADS must be a power of two");

__simt_callee__ __aicore__ inline float AbsFloat(float value) { return value < 0.0f ? -value : value; }

__simt_callee__ __aicore__ inline aclblasComplex ComplexDiv(aclblasComplex numerator, aclblasComplex denominator)
{
    float absReal = AbsFloat(denominator.real);
    float absImag = AbsFloat(denominator.imag);
    if (absReal >= absImag) {
        float ratio = denominator.imag / denominator.real;
        float scale = denominator.real + denominator.imag * ratio;
        return {
            (numerator.real + numerator.imag * ratio) / scale,
            (numerator.imag - numerator.real * ratio) / scale,
        };
    }

    float ratio = denominator.real / denominator.imag;
    float scale = denominator.imag + denominator.real * ratio;
    return {
        (numerator.real * ratio + numerator.imag) / scale,
        (numerator.imag * ratio - numerator.real) / scale,
    };
}

__simd_vf__ inline void CgetriInitMetadataVf(__ubuf__ int* metadata, uint32_t n, uint32_t firstCol)
{
    Reg::RegTensor<int32_t> value;
    auto all = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::Arange(value, static_cast<int32_t>(firstCol));
    Reg::StoreAlign(metadata + CGETRI_METADATA_PERMUTATION, value, all);
    if (firstCol == 0) {
        uint32_t one = 1;
        auto scalar = Reg::UpdateMask<int32_t>(one);
        Reg::Duplicate(value, static_cast<int32_t>(n + 1));
        Reg::StoreAlign(metadata + CGETRI_METADATA_STATUS, value, scalar);
        Reg::Duplicate(value, 0);
        Reg::StoreAlign(metadata + CGETRI_METADATA_AUX, value, scalar);
    }
}

__simd_vf__ inline void CgetriInitPermutationVf(__ubuf__ int* metadata, uint32_t n, uint32_t firstPivot, uint32_t count)
{
    Reg::RegTensor<int32_t> permutation, pivot, current, nonDiagonal, one;
    auto all = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg valid, nonnegative, equalK, equalPivot, moved;
    Reg::LoadAlign(permutation, metadata + CGETRI_METADATA_PERMUTATION);
    Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_BRC_B32>(nonDiagonal, metadata + CGETRI_METADATA_AUX);
    Reg::Duplicate(one, 1);
    for (uint16_t k = 0; k < count; k++) {
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_BRC_B32>(pivot, metadata + k);
        Reg::Adds(pivot, pivot, -1, all);
        Reg::Duplicate(current, static_cast<int32_t>(firstPivot + k));
        Reg::Compares<int32_t, CMPMODE::GE>(nonnegative, pivot, 0, all);
        Reg::Compares<int32_t, CMPMODE::LT>(valid, pivot, static_cast<int32_t>(n), nonnegative);
        Reg::Select(pivot, pivot, current, valid);
        Reg::Compare<int32_t, CMPMODE::NE>(moved, pivot, current, all);
        Reg::Select(nonDiagonal, one, nonDiagonal, moved);
        Reg::Compare<int32_t, CMPMODE::EQ>(equalK, permutation, current, all);
        Reg::Compare<int32_t, CMPMODE::EQ>(equalPivot, permutation, pivot, all);
        Reg::Select(permutation, current, permutation, equalPivot);
        Reg::Select(permutation, pivot, permutation, equalK);
    }
    Reg::StoreAlign(metadata + CGETRI_METADATA_PERMUTATION, permutation, all);
    uint32_t oneElement = 1;
    auto scalar = Reg::UpdateMask<int32_t>(oneElement);
    Reg::StoreAlign(metadata + CGETRI_METADATA_AUX, nonDiagonal, scalar);
}

} // namespace CgetriBatched
