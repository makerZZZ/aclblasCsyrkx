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
 * \file cgetri_batched_kernel.cpp
 * \brief 小矩阵 SIMT 求解，以及中等矩阵 SIMD 和流式三角求解。
 */

#include <cstdint>

#include "cann_ops_blas_common.h"
#include "cgetri_batched_kernel.h"
#include "cgetri_batched_solve_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "c_api/asc_simd.h"

using namespace AscendC;
using namespace CgetriBatched;

namespace {

__simt_callee__ __aicore__ inline aclblasComplex ComplexMul(aclblasComplex lhs, aclblasComplex rhs)
{
    return {
        lhs.real * rhs.real - lhs.imag * rhs.imag,
        lhs.real * rhs.imag + lhs.imag * rhs.real,
    };
}

__simt_callee__ __aicore__ inline aclblasComplex ComplexSubMul(
    aclblasComplex value, aclblasComplex lhs, aclblasComplex rhs)
{
    aclblasComplex product = ComplexMul(lhs, rhs);
    return {value.real - product.real, value.imag - product.imag};
}

__simt_callee__ __aicore__ inline bool IsZero(aclblasComplex value) { return value.real == 0.0f && value.imag == 0.0f; }

union CgetriComplexBits {
    aclblasComplex value;
    uint64_t packed;
};

static_assert(sizeof(aclblasComplex) == sizeof(uint64_t), "aclblasComplex must occupy one 64-bit GM word");

// Move one COMPLEX64 value as one aligned scalar word.  Copying the aggregate itself across
// address spaces is unsupported, while the equivalent uint64_t GM transaction is supported.
__simt_callee__ __aicore__ inline aclblasComplex LoadComplex(__gm__ aclblasComplex* base, uint32_t index)
{
    CgetriComplexBits bits;
    bits.packed = reinterpret_cast<__gm__ uint64_t*>(base)[index];
    return bits.value;
}

__simt_callee__ __aicore__ inline void StoreComplex(__gm__ aclblasComplex* base, uint32_t index, aclblasComplex value)
{
    CgetriComplexBits bits;
    bits.value = value;
    reinterpret_cast<__gm__ uint64_t*>(base)[index] = bits.packed;
}

__simt_callee__ __aicore__ inline aclblasComplex LoadSmallComplex(
    __ubuf__ float* realBase, __ubuf__ float* imagBase, uint32_t index)
{
    return {realBase[index], imagBase[index]};
}

__simt_callee__ __aicore__ inline __gm__ aclblasComplex* ReadComplexPtrFromArray(GM_ADDR arrayBase, uint32_t batchIdx)
{
    __gm__ uint64_t* addrSlot = reinterpret_cast<__gm__ uint64_t*>(arrayBase) + batchIdx;
    return reinterpret_cast<__gm__ aclblasComplex*>(*addrSlot);
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void CgetriLoadSmallLu(
    GM_ADDR aarrayBase, __gm__ int* pivotArray, __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ int* pivots,
    __ubuf__ int* fastPath, uint32_t n, uint32_t lda, uint32_t batchIdx, uint32_t lane, uint32_t usePivot)
{
    uint32_t matrixElements = n * n;
    __gm__ aclblasComplex* A = ReadComplexPtrFromArray(aarrayBase, batchIdx);
    for (uint32_t linear = lane; linear < matrixElements; linear += CGETRI_SMALL_GROUP_SIZE) {
        uint32_t row = linear % n;
        uint32_t col = linear / n;
        uint32_t sourceIndex = row + col * lda;
        aclblasComplex value = LoadComplex(A, sourceIndex);
        float real = value.real;
        float imag = value.imag;
        ar[linear] = real;
        ai[linear] = imag;
        if (row != col && (real != 0.0f || imag != 0.0f)) {
            fastPath[0] = 0;
        }
    }
    if (usePivot != 0 && lane < n) {
        int pivot = pivotArray[batchIdx * n + lane];
        pivots[lane] = pivot;
        if (pivot != static_cast<int>(lane + 1)) {
            fastPath[0] = 0;
        }
    }
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void CgetriSolveSmallColumn(
    __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ int* pivots, __gm__ aclblasComplex* C, uint32_t n, uint32_t ldc,
    uint32_t col, uint32_t usePivot)
{
    aclblasComplex values[CGETRI_SMALL_N] = {};
    values[col].real = 1.0f;

    if (usePivot != 0) {
        for (uint32_t k = 0; k < n; k++) {
            int pivotRow = pivots[k] - 1;
            if (pivotRow >= 0 && pivotRow < static_cast<int>(n) && pivotRow != static_cast<int>(k)) {
                aclblasComplex temp = values[k];
                values[k] = values[static_cast<uint32_t>(pivotRow)];
                values[static_cast<uint32_t>(pivotRow)] = temp;
            }
        }
    }

    for (uint32_t j = 0; j < n; j++) {
        aclblasComplex cjc = values[j];
        for (uint32_t row = j + 1; row < n; row++) {
            aclblasComplex factor = LoadSmallComplex(ar, ai, row + j * n);
            values[row] = ComplexSubMul(values[row], factor, cjc);
        }
    }

    for (int32_t j = static_cast<int32_t>(n) - 1; j >= 0; j--) {
        uint32_t row = static_cast<uint32_t>(j);
        aclblasComplex diagonal = LoadSmallComplex(ar, ai, row + row * n);
        aclblasComplex cjc = ComplexDiv(values[row], diagonal);
        values[row] = cjc;
        for (uint32_t i = 0; i < row; i++) {
            aclblasComplex factor = LoadSmallComplex(ar, ai, i + row * n);
            values[i] = ComplexSubMul(values[i], factor, cjc);
        }
    }

    for (uint32_t row = 0; row < n; row++) {
        StoreComplex(C, row + col * ldc, values[row]);
    }
}

__simt_callee__ __aicore__ __attribute__((always_inline)) inline void CgetriInvertSmallDiagonal(
    __ubuf__ float* ar, __ubuf__ float* ai, __gm__ aclblasComplex* C, uint32_t n, uint32_t ldc, uint32_t row)
{
    aclblasComplex diagonal = LoadSmallComplex(ar, ai, row + row * n);
    aclblasComplex reciprocal = ComplexDiv({1.0f, 0.0f}, diagonal);
    for (uint32_t col = 0; col < n; col++) {
        aclblasComplex value = row == col ? reciprocal : aclblasComplex{0.0f, 0.0f};
        StoreComplex(C, row + col * ldc, value);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGETRI_SIMT_MAX_THREADS) inline void CgetriSmallSimt(
    uint32_t n, uint32_t lda, uint32_t ldc, uint32_t numBatch, uint32_t startBatch, GM_ADDR aarrayBase,
    GM_ADDR carrayBase, __gm__ int* pivotArray, uint32_t usePivot, __gm__ int* infoArray)
{
    __ubuf__ float sharedAReal[CGETRI_SMALL_GROUP_COUNT * CGETRI_SMALL_MATRIX_ELEMENTS];
    __ubuf__ float sharedAImag[CGETRI_SMALL_GROUP_COUNT * CGETRI_SMALL_MATRIX_ELEMENTS];
    __ubuf__ int sharedPivots[CGETRI_SMALL_GROUP_COUNT * CGETRI_SMALL_N];
    __ubuf__ int sharedInfo[CGETRI_SMALL_GROUP_COUNT];
    __ubuf__ int sharedFastPath[CGETRI_SMALL_GROUP_COUNT];

    uint32_t group = threadIdx.x / CGETRI_SMALL_GROUP_SIZE;
    uint32_t lane = threadIdx.x % CGETRI_SMALL_GROUP_SIZE;
    uint32_t groupMatrixOffset = group * CGETRI_SMALL_MATRIX_ELEMENTS;
    uint32_t groupPivotOffset = group * CGETRI_SMALL_N;

    for (uint32_t wave = 0; wave < numBatch; wave += CGETRI_SMALL_GROUP_COUNT) {
        uint32_t batchOffset = wave + group;
        bool active = batchOffset < numBatch;
        uint32_t batchIdx = startBatch + batchOffset;

        if (lane == 0) {
            sharedInfo[group] = 0;
            sharedFastPath[group] = 1;
        }
        asc_syncthreads();
        bool needsGeneral = active && sharedInfo[group] == 0;
        if (needsGeneral) {
            CgetriLoadSmallLu(
                aarrayBase, pivotArray, sharedAReal + groupMatrixOffset, sharedAImag + groupMatrixOffset,
                sharedPivots + groupPivotOffset, sharedFastPath + group, n, lda, batchIdx, lane, usePivot);
        }
        asc_syncthreads();

        if (needsGeneral && lane == 0) {
            for (uint32_t k = 0; k < n; k++) {
                if (IsZero(LoadSmallComplex(sharedAReal, sharedAImag, groupMatrixOffset + k + k * n))) {
                    sharedInfo[group] = static_cast<int>(k + 1);
                    break;
                }
            }
            infoArray[batchIdx] = sharedInfo[group];
        }
        asc_syncthreads();

        if (needsGeneral && sharedInfo[group] == 0 && sharedFastPath[group] != 0 && lane < n) {
            CgetriInvertSmallDiagonal(
                sharedAReal + groupMatrixOffset, sharedAImag + groupMatrixOffset,
                ReadComplexPtrFromArray(carrayBase, batchIdx), n, ldc, lane);
        } else if (needsGeneral && sharedInfo[group] == 0 && lane < n) {
            CgetriSolveSmallColumn(
                sharedAReal + groupMatrixOffset, sharedAImag + groupMatrixOffset, sharedPivots + groupPivotOffset,
                ReadComplexPtrFromArray(carrayBase, batchIdx), n, ldc, lane, usePivot);
        }
        asc_syncthreads();
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGETRI_SIMT_MAX_THREADS) inline void CgetriUnpackPackedSimt(
    uint32_t n, uint32_t ldc, uint32_t batch, uint32_t count, GM_ADDR carrayBase, __gm__ int* infoArray,
    __ubuf__ float* xr, __ubuf__ float* xi, __ubuf__ int* metadata)
{
    uint32_t groups = n <= 8 ? 8u : (n <= 16 ? 4u : 2u);
    uint32_t slot = threadIdx.x % groups;
    int status = metadata[128 + slot * 8];
    if (threadIdx.x < count) {
        infoArray[batch + slot] = status;
    }
    if (slot >= count || status != 0) {
        return;
    }
    __gm__ aclblasComplex* C = ReadComplexPtrFromArray(carrayBase, batch + slot);
    for (uint32_t k = threadIdx.x / groups; k < n * n; k += CGETRI_SIMT_MAX_THREADS / groups) {
        uint32_t row = k % n;
        uint32_t col = k / n;
        uint32_t offset = row * 72 + col * groups + slot;
        StoreComplex(C, row + col * ldc, {xr[offset], xi[offset]});
    }
}

template <bool Packed>
__simd_vf__ inline void CgetriStageLuVf(__ubuf__ float* ar, __ubuf__ float* ai, uint32_t n)
{
    uint32_t luLd = (n + 7) / 8 * 8;
    Reg::RegTensor<float> real, imag;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    if constexpr (Packed) {
        uint32_t groups = n <= 8 ? 8u : (n <= 16 ? 4u : 2u);
        Reg::RegTensor<uint32_t> lanes, row, slot, index, one;
        Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lanes), 0);
        Reg::Duplicate(one, groups - 1u);
        Reg::ShiftRights(row, lanes, static_cast<int16_t>(3), all);
        Reg::Muls(row, row, 2u, all);
        Reg::And(slot, lanes, one, all);
        Reg::Muls(slot, slot, n * luLd * 2u, all);
        Reg::Add(index, row, slot, all);
        for (uint16_t first = 0; first < n * luLd * 8; first += 64) {
            Reg::Gather(real, ar + 8192 + first / 4, index, all);
            Reg::Gather(imag, ar + 8193 + first / 4, index, all);
            Reg::Neg(real, real, all);
            Reg::Neg(imag, imag, all);
            Reg::StoreAlign(ar + first, real, all);
            Reg::StoreAlign(ai + first, imag, all);
        }
    } else {
        // The raw complex input occupies ai and the unused RHS arena. Reading
        // forward is safe: each imaginary output is behind its raw source.
        for (uint16_t first = 0; first < n * luLd; first += 64) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(real, imag, ai + first * 2);
            Reg::Neg(real, real, all);
            Reg::Neg(imag, imag, all);
            Reg::StoreAlign(ar + first, real, all);
            Reg::StoreAlign(ai + first, imag, all);
        }
    }
}

// 仅在所有有效主元都不交换行时跳过置换；非法主元仍按已有恒等处理规则忽略。
__simd_vf__ inline void CgetriDetectPivotSwapsVf(__ubuf__ int* metadata, uint32_t n)
{
    Reg::RegTensor<int32_t> pivot, rows, swaps, one, reduced;
    Reg::MaskReg inRange, upper, changed;
    auto all = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(swaps, 0);
    Reg::Duplicate(one, 1);
    uint32_t remaining = n;
    for (uint32_t first = 0; first < n; first += 64) {
        auto valid = Reg::UpdateMask<int32_t>(remaining);
        Reg::LoadAlign(pivot, metadata + first);
        Reg::Arange(rows, static_cast<int32_t>(first + 1));
        Reg::Compares<int32_t, CMPMODE::GE>(inRange, pivot, 1, valid);
        Reg::Compares<int32_t, CMPMODE::LE>(upper, pivot, static_cast<int32_t>(n), inRange);
        Reg::Compare<int32_t, CMPMODE::NE>(changed, pivot, rows, upper);
        Reg::Select(swaps, one, swaps, changed);
    }
    Reg::Reduce<Reg::ReduceType::MAX>(reduced, swaps, all);
    uint32_t count = 1;
    auto scalar = Reg::UpdateMask<int32_t>(count);
    Reg::StoreAlign(metadata + 184, reduced, scalar);
}

// Each lane tracks the final row of one identity column. Pivot swaps remain
// in registers, avoiding a serial SIMT shared-memory read/modify/write loop.
// 寄存器参数由 VF 编译阶段推导，避免 Host 编译阶段依赖设备专用类型。
template <bool Packed, typename IntRegister, typename UintRegister>
__simd_callee__ __attribute__((always_inline)) inline void CgetriPrepareStatusVf(
    __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ int* metadata, uint32_t n, IntRegister& columns,
    UintRegister& slots, uint32_t groups)
{
    uint32_t luLd = (n + 7) / 8 * 8;
    Reg::RegTensor<int32_t> diagonalRows, current, status, sentinel, zeros;
    Reg::RegTensor<uint32_t> indices;
    Reg::RegTensor<float> real, imag;
    Reg::MaskReg realZero, imagZero, singular, selectSlot, valid;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zeros, 0);
    Reg::Duplicate(sentinel, static_cast<int32_t>(n + 1));
    Reg::Duplicate(status, static_cast<int32_t>(n + 1));
    uint32_t scalarCount = 1;
    auto scalarMask = Reg::UpdateMask<int32_t>(scalarCount);
    for (uint16_t first = 0; first < n; first += 64 / groups) {
        Reg::Adds(diagonalRows, columns, static_cast<int32_t>(first), all);
        Reg::Muls(
            indices, reinterpret_cast<Reg::RegTensor<uint32_t>&>(diagonalRows), (luLd + 1u) * (Packed ? 8u : 1u), all);
        if constexpr (Packed) {
            Reg::Add(indices, indices, slots, all);
        }
        Reg::Compares<int32_t, CMPMODE::LT>(valid, diagonalRows, static_cast<int32_t>(n), all);
        Reg::Gather(real, ar, indices, valid);
        Reg::Gather(imag, ai, indices, valid);
        Reg::Compares<float, CMPMODE::EQ>(realZero, real, 0.0f, valid);
        Reg::Compares<float, CMPMODE::EQ>(imagZero, imag, 0.0f, valid);
        Reg::And(singular, realZero, imagZero, all);
        Reg::Adds(current, diagonalRows, 1, all);
        Reg::Select(current, current, sentinel, singular);
        Reg::Min(status, status, current, all);
    }
    for (uint16_t slot = 0; slot < groups; slot++) {
        Reg::Compares<uint32_t, CMPMODE::EQ>(selectSlot, slots, static_cast<uint32_t>(slot), all);
        Reg::RegTensor<int32_t> selected, reduced;
        Reg::Select(selected, status, sentinel, selectSlot);
        Reg::Reduce<Reg::ReduceType::MIN>(reduced, selected, all);
        Reg::Compares<int32_t, CMPMODE::EQ>(valid, reduced, static_cast<int32_t>(n + 1), all);
        Reg::Select(reduced, zeros, reduced, valid);
        Reg::StoreAlign(metadata + 128 + slot * 8, reduced, scalarMask);
    }
}

template <bool Packed, bool CacheNext, typename IntRegister, typename UintRegister>
__simd_callee__ __attribute__((always_inline)) inline void CgetriPreparePermutationVf(
    __ubuf__ int* metadata, uint32_t n, uint32_t firstCol, uint32_t usePivot, IntRegister& columns,
    UintRegister& pivotIndices, IntRegister& permutation)
{
    Reg::RegTensor<int32_t> current, pivot;
    Reg::MaskReg nonnegative, valid, equalK, equalPivot;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::RegTensor<int32_t> nextPermutation;
    Reg::Adds(permutation, columns, static_cast<int32_t>(firstCol), all);
    if constexpr (CacheNext) {
        Reg::Adds(nextPermutation, columns, 64, all);
    }
    if (usePivot != 0) {
        for (uint16_t k = 0; k < n; k++) {
            Reg::Duplicate(current, static_cast<int32_t>(k));
            if constexpr (Packed) {
                Reg::Gather(pivot, metadata + k, pivotIndices, all);
            } else {
                Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_BRC_B32>(pivot, metadata + k);
            }
            Reg::Adds(pivot, pivot, -1, all);
            Reg::Compares<int32_t, CMPMODE::GE>(nonnegative, pivot, 0, all);
            Reg::Compares<int32_t, CMPMODE::LT>(valid, pivot, static_cast<int32_t>(n), nonnegative);
            Reg::Select(pivot, pivot, current, valid);
            Reg::Compare<int32_t, CMPMODE::EQ>(equalK, permutation, current, all);
            Reg::Compare<int32_t, CMPMODE::EQ>(equalPivot, permutation, pivot, all);
            Reg::Select(permutation, current, permutation, equalPivot);
            Reg::Select(permutation, pivot, permutation, equalK);
            if constexpr (CacheNext) {
                Reg::Compare<int32_t, CMPMODE::EQ>(equalK, nextPermutation, current, all);
                Reg::Compare<int32_t, CMPMODE::EQ>(equalPivot, nextPermutation, pivot, all);
                Reg::Select(nextPermutation, current, nextPermutation, equalPivot);
                Reg::Select(nextPermutation, pivot, nextPermutation, equalK);
            }
        }
    }
    if constexpr (CacheNext) {
        Reg::StoreAlign(metadata + 192, nextPermutation, all);
    }
}

template <uint32_t FixedN, bool Packed, bool KeepStatus = false, bool CacheNext = false>
__simd_vf__ inline void CgetriPrepareVf(
    __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi, __ubuf__ int* metadata,
    uint32_t dynamicN, uint32_t firstCol, uint32_t usePivot, uint32_t count)
{
    uint32_t n = FixedN == 0 ? dynamicN : FixedN;
    Reg::RegTensor<int32_t> columns, permutation, current, sentinel, zeros;
    Reg::RegTensor<uint32_t> lanes, slots, one, pivotIndices;
    Reg::RegTensor<float> rhs, zeroFloat;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::MaskReg equalK, valid;
    Reg::Arange(columns, 0);
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(lanes), 0);
    uint32_t groups = Packed ? (n <= 8 ? 8u : (n <= 16 ? 4u : 2u)) : 1u;
    Reg::Duplicate(one, groups - 1u);
    Reg::Duplicate(zeros, 0);
    Reg::Duplicate(zeroFloat, 0.0f);
    if constexpr (Packed) {
        Reg::And(slots, lanes, one, all);
        Reg::ShiftRights(columns, columns, static_cast<int16_t>(n <= 8 ? 3 : (n <= 16 ? 2 : 1)), all);
        Reg::Compares<uint32_t, CMPMODE::LT>(valid, slots, count, all);
        Reg::Select(pivotIndices, slots, reinterpret_cast<Reg::RegTensor<uint32_t>&>(zeros), valid);
        Reg::Muls(pivotIndices, pivotIndices, n, all);
    } else {
        Reg::Duplicate(slots, 0u);
    }
    Reg::Duplicate(sentinel, static_cast<int32_t>(n + 1));
    uint32_t scalarCount = 1;
    auto scalarMask = Reg::UpdateMask<int32_t>(scalarCount);
    if constexpr (!KeepStatus) {
        CgetriPrepareStatusVf<Packed>(ar, ai, metadata, n, columns, slots, groups);
    }
    if constexpr (KeepStatus) {
        // 后续块复用首次准备时保存的 64..127 列置换。
        Reg::LoadAlign(permutation, metadata + 192);
    } else {
        CgetriPreparePermutationVf<Packed, CacheNext>(
            metadata, n, firstCol, usePivot, columns, pivotIndices, permutation);
    }
    if constexpr (!Packed && FixedN != 64) {
        Reg::Adds(current, columns, static_cast<int32_t>(firstCol), all);
        Reg::Compares<int32_t, CMPMODE::LT>(valid, current, static_cast<int32_t>(n), all);
        Reg::Select(current, permutation, sentinel, valid);
        Reg::Reduce<Reg::ReduceType::MIN>(current, current, all);
        Reg::StoreAlign(metadata + 136, current, scalarMask);
    }
    for (uint16_t row = 0; row < n; row++) {
        Reg::Compares<int32_t, CMPMODE::EQ>(equalK, permutation, static_cast<int32_t>(row), all);
        Reg::Duplicate(rhs, 1.0f, equalK);
        Reg::StoreAlign(xr + row * 72, rhs, all);
        Reg::StoreAlign(xi + row * 72, zeroFloat, all);
    }
}

// 尾部最多 8 列时，将行间距收紧到一个 DataBlock，供一次并行更新 8 行。
template <bool Pack>
__simd_vf__ inline void CgetriRepackTailVf(__ubuf__ float* xr, __ubuf__ float* xi, uint32_t n)
{
    Reg::RegTensor<float> real, imag;
    uint32_t count = 8;
    auto mask = Reg::UpdateMask<float>(count);
    for (uint32_t step = 0; step < n; step++) {
        uint32_t row = Pack ? step : n - 1 - step;
        Reg::LoadAlign(real, xr + row * (Pack ? 72 : 8));
        Reg::LoadAlign(imag, xi + row * (Pack ? 72 : 8));
        Reg::StoreAlign(xr + row * (Pack ? 8 : 72), real, mask);
        Reg::StoreAlign(xi + row * (Pack ? 8 : 72), imag, mask);
    }
}

// 每个 DataBlock 保存同一行的 8 个 RHS，E2B 广播 LU 系数，BLK 广播已求出的行。
// LU 和对角倒数由第一组 64 列保留，运算顺序与普通三角求解一致。
__simd_vf__ inline void CgetriSolveTail8Vf(
    __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi, uint32_t n, uint32_t lowerStart)
{
    uint32_t luLd = (n + 7) / 8 * 8;
    Reg::RegTensor<float> jr, ji, fr, fi, p0, p1, p2, p3;
    Reg::RegTensor<int32_t> localRows, rows;
    Reg::MaskReg afterPivot, validRows;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    uint32_t count = 8;
    auto oneRow = Reg::UpdateMask<float>(count);
    Reg::Arange(localRows, 0);
    Reg::ShiftRights(localRows, localRows, static_cast<int16_t>(3), all);
    for (uint32_t j = lowerStart; j < n; j++) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BLK>(jr, xr + j * 8);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BLK>(ji, xi + j * 8);
        Reg::Neg(p3, ji, all);
        for (uint32_t row = (j + 1) / 8 * 8; row < n; row += 8) {
            Reg::Adds(rows, localRows, static_cast<int32_t>(row), all);
            Reg::Compares<int32_t, CMPMODE::GT>(afterPivot, rows, static_cast<int32_t>(j), all);
            Reg::Compares<int32_t, CMPMODE::LT>(validRows, rows, static_cast<int32_t>(n), afterPivot);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_E2B_B32>(fr, ar + j * luLd + row);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_E2B_B32>(fi, ai + j * luLd + row);
            CgetriUpdateRhsVf(xr, xi, row * 8, fr, fi, jr, ji, p3, validRows);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
    for (uint16_t reverse = 0; reverse < n; reverse++) {
        uint16_t j = n - 1 - reverse;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BLK>(jr, xr + j * 8);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BLK>(ji, xi + j * 8);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(fr, ar + j * (luLd + 1));
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(fi, ai + j * (luLd + 1));
        Reg::Mul(p0, fr, jr, all);
        Reg::Mul(p1, fi, ji, all);
        Reg::Mul(p2, fr, ji, all);
        Reg::Mul(p3, fi, jr, all);
        Reg::Sub(jr, p0, p1, all);
        Reg::Add(ji, p2, p3, all);
        Reg::StoreAlign(xr + j * 8, jr, oneRow);
        Reg::StoreAlign(xi + j * 8, ji, oneRow);
        Reg::Neg(p3, ji, all);
        uint32_t remaining = j * 8;
        for (uint16_t row = 0; row < j; row += 8) {
            auto mask = Reg::UpdateMask<float>(remaining);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_E2B_B32>(fr, ar + j * luLd + row);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_E2B_B32>(fi, ai + j * luLd + row);
            CgetriUpdateRhsVf(xr, xi, row * 8, fr, fi, jr, ji, p3, mask);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

template <uint32_t FixedN>
__simt_vf__ __aicore__ LAUNCH_BOUND(CGETRI_SIMT_MAX_THREADS) inline void CgetriUnpackSimt(
    uint32_t dynamicN, uint32_t ldc, uint32_t batch, uint32_t firstCol, GM_ADDR carrayBase, __gm__ int* infoArray,
    __ubuf__ float* xr, __ubuf__ float* xi, __ubuf__ int* metadata)
{
    uint32_t n = FixedN == 0 ? dynamicN : FixedN;
    int status = metadata[128];
    if (threadIdx.x == 0) {
        infoArray[batch] = status;
    }
    if (status != 0) {
        return;
    }
    __gm__ aclblasComplex* C = ReadComplexPtrFromArray(carrayBase, batch);
    uint32_t columns = n - firstCol < 64 ? n - firstCol : 64;
    for (uint32_t k = threadIdx.x; k < n * columns; k += blockDim.x) {
        uint32_t row = k % n;
        uint32_t col = k / n;
        StoreComplex(C, row + (firstCol + col) * ldc, {xr[row * 72 + col], xi[row * 72 + col]});
    }
}

__simd_vf__ inline void CgetriOutputTileVf(
    __ubuf__ float* xr, __ubuf__ float* xi, __ubuf__ float* output, uint32_t firstRow, uint32_t firstCol, uint32_t rows,
    uint32_t columns)
{
    Reg::RegTensor<uint32_t> indices;
    Reg::RegTensor<float> real, imag;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    auto mask = Reg::UpdateMask<float>(rows);
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(indices), 0);
    Reg::Muls(indices, indices, 72u, all);
    for (uint16_t col = 0; col < columns; col++) {
        Reg::Gather(real, xr + firstRow * 72 + firstCol + col, indices, mask);
        Reg::Gather(imag, xi + firstRow * 72 + firstCol + col, indices, mask);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(output + col * 128, real, imag, mask);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGETRI_SIMT_MAX_THREADS) inline void CgetriStoreInfoSimt(
    uint32_t n, uint32_t batch, __ubuf__ int* metadata, uint32_t statusOffset, __gm__ int* infos)
{
    if (threadIdx.x == 0) {
        int status = metadata[statusOffset];
        infos[batch] = status == static_cast<int>(n + 1) ? 0 : status;
    }
}

__aicore__ inline void CgetriWriteRhs(
    __ubuf__ float* xr, __ubuf__ float* xi, __ubuf__ float* output, uint32_t n, uint32_t ldc, uint32_t firstCol,
    __gm__ float* destination)
{
    uint32_t rhsColumns = n - firstCol < 64 ? n - firstCol : 64;
    for (uint32_t col = 0; col < rhsColumns; col += 32) {
        uint32_t columns = rhsColumns - col < 32 ? rhsColumns - col : 32;
        for (uint32_t row = 0; row < n; row += 64) {
            uint32_t rows = n - row < 64 ? n - row : 64;
            asc_vf_call<CgetriOutputTileVf>(xr, xi, output, row, col, rows, columns);
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::UB>();
            asc_copy_ub2gm_align(
                destination + (static_cast<uint64_t>(firstCol + col) * ldc + row) * 2, output, columns, rows * 8, 0,
                ldc * 8, 64 * 8);
            PipeBarrier<PIPE_ALL>();
        }
    }
}

__simd_vf__ inline void CgetriStreamIdentityVf(
    __ubuf__ float* xr, __ubuf__ float* xi, __ubuf__ int* metadata, uint32_t n, uint32_t firstCol)
{
    Reg::RegTensor<int32_t> permutation, columns, sentinel, selected, minimum;
    Reg::RegTensor<float> rhs, zero;
    Reg::MaskReg identity, valid;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::LoadAlign(permutation, metadata + CGETRI_METADATA_PERMUTATION);
    Reg::Arange(columns, static_cast<int32_t>(firstCol));
    Reg::Duplicate(sentinel, static_cast<int32_t>(n + 1));
    Reg::Compares<int32_t, CMPMODE::LT>(valid, columns, static_cast<int32_t>(n), all);
    Reg::Select(selected, permutation, sentinel, valid);
    Reg::Reduce<Reg::ReduceType::MIN>(minimum, selected, all);
    uint32_t one = 1;
    auto scalar = Reg::UpdateMask<int32_t>(one);
    Reg::StoreAlign(metadata + CGETRI_METADATA_STATUS, minimum, scalar);
    Reg::StoreAlign(metadata + CGETRI_METADATA_AUX, sentinel, scalar);
    Reg::Duplicate(zero, 0.0f);
    for (uint16_t row = 0; row < n; row++) {
        Reg::Compares<int32_t, CMPMODE::EQ>(identity, permutation, static_cast<int32_t>(row), valid);
        Reg::Duplicate(rhs, 1.0f, identity);
        Reg::StoreAlign(xr + row * 72, rhs, all);
        Reg::StoreAlign(xi + row * 72, zero, all);
    }
}

__simd_vf__ inline void CgetriStreamStageVf(
    __ubuf__ float* raw, __ubuf__ float* ar, __ubuf__ float* ai, uint32_t elements)
{
    Reg::RegTensor<float> real, imag;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t offset = 0; offset < elements; offset += 64) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(real, imag, raw + offset * 2);
        Reg::Neg(real, real, all);
        Reg::Neg(imag, imag, all);
        Reg::StoreAlign(ar + offset, real, all);
        Reg::StoreAlign(ai + offset, imag, all);
    }
}

__simd_callee__ __attribute__((always_inline)) inline void CgetriStreamReciprocals(
    __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ int* metadata, uint32_t luLd, uint32_t first, uint32_t count)
{
    Reg::RegTensor<int32_t> rows, status, sentinel, reduced;
    Reg::RegTensor<uint32_t> indices;
    Reg::RegTensor<float> real, imag, absReal, absImag, large, small, ratio, scale, one, inverse, product;
    Reg::MaskReg realZero, imagZero, singular, realMajor;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    auto mask = Reg::UpdateMask<float>(count);
    Reg::Arange(rows, 0);
    Reg::Muls(indices, reinterpret_cast<Reg::RegTensor<uint32_t>&>(rows), luLd + 1, all);
    Reg::Adds(indices, indices, first, all);
    Reg::Gather(real, ar, indices, mask);
    Reg::Gather(imag, ai, indices, mask);
    Reg::Compares<float, CMPMODE::EQ>(realZero, real, 0.0f, mask);
    Reg::Compares<float, CMPMODE::EQ>(imagZero, imag, 0.0f, mask);
    Reg::And(singular, realZero, imagZero, all);
    Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_BRC_B32>(sentinel, metadata + CGETRI_METADATA_AUX);
    Reg::Adds(rows, rows, static_cast<int32_t>(first + 1), all);
    Reg::Select(status, rows, sentinel, singular);
    Reg::Min(status, status, sentinel, all);
    Reg::Reduce<Reg::ReduceType::MIN>(reduced, status, all);
    uint32_t oneElement = 1;
    auto scalar = Reg::UpdateMask<int32_t>(oneElement);
    Reg::StoreAlign(metadata + CGETRI_METADATA_AUX, reduced, scalar);
    Reg::Abs(absReal, real, mask);
    Reg::Abs(absImag, imag, mask);
    Reg::Compare<float, CMPMODE::GE>(realMajor, absReal, absImag, mask);
    Reg::Select(large, real, imag, realMajor);
    Reg::Select(small, imag, real, realMajor);
    Reg::Div(ratio, small, large, mask);
    Reg::Mul(scale, small, ratio, mask);
    Reg::Add(scale, large, scale, mask);
    Reg::Duplicate(one, 1.0f);
    Reg::Div(inverse, one, scale, mask);
    Reg::Mul(product, ratio, inverse, mask);
    Reg::Select(real, inverse, product, realMajor);
    Reg::Select(imag, product, inverse, realMajor);
    Reg::Neg(real, real, mask);
    Reg::Scatter(ar, real, indices, mask);
    Reg::Scatter(ai, imag, indices, mask);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <bool Upper>
__simd_vf__ inline void CgetriStreamPanelVf(
    __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi, __ubuf__ int* metadata, uint32_t n,
    uint32_t first, uint32_t count, uint32_t lowerStart)
{
    uint32_t luLd = (n + 7) / 8 * 8;
    if constexpr (Upper) {
        CgetriStreamReciprocals(ar, ai, metadata, luLd, first, count);
    }
    Reg::RegTensor<float> jr, ji, fr, fi, p0, p1, p2, p3;
    auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t step = 0; step < count; step++) {
        uint32_t local = Upper ? count - 1 - step : step;
        uint32_t j = first + local;
        if (!Upper && j < lowerStart) {
            continue;
        }
        Reg::LoadAlign(jr, xr + j * 72);
        Reg::LoadAlign(ji, xi + j * 72);
        if constexpr (Upper) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(fr, ar + local * luLd + j);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(fi, ai + local * luLd + j);
            Reg::Mul(p0, fr, jr, mask);
            Reg::Mul(p1, fi, ji, mask);
            Reg::Mul(p2, fr, ji, mask);
            Reg::Mul(p3, fi, jr, mask);
            Reg::Sub(jr, p0, p1, mask);
            Reg::Add(ji, p2, p3, mask);
            Reg::StoreAlign(xr + j * 72, jr, mask);
            Reg::StoreAlign(xi + j * 72, ji, mask);
        }
        Reg::Neg(p3, ji, mask);
        uint32_t begin = Upper ? 0 : j + 1;
        uint32_t end = Upper ? j : n;
        for (uint16_t row = begin; row < end; row++) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(fr, ar + local * luLd + row);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(fi, ai + local * luLd + row);
            CgetriUpdateRhsVf(xr, xi, row * 72, fr, fi, jr, ji, p3, mask);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

} // namespace

extern "C" __global__ __aicore__ void cgetri_batched_kernel(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR carray, GM_ADDR infoArray, const CgetriBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.usedCoreNum == 0) {
        return;
    }

    uint32_t blockId = GetBlockIdx();
    if (blockId >= tiling.usedCoreNum) {
        return;
    }

    uint32_t startBatch = blockId * tiling.batchPerCore;
    uint32_t numBatch = blockId == tiling.usedCoreNum - 1 ? tiling.batchTail : tiling.batchPerCore;
    if (numBatch == 0) {
        return;
    }

    auto* pivots = reinterpret_cast<__gm__ int*>(pivotArray);
    auto* infos = reinterpret_cast<__gm__ int*>(infoArray);
    asc_vf_call<CgetriSmallSimt>(
        dim3{CGETRI_SIMT_MAX_THREADS, 1, 1}, tiling.n, tiling.lda, tiling.ldc, numBatch, startBatch, aarray, carray,
        pivots, tiling.usePivot, infos);
}

__aicore__ __attribute__((always_inline)) inline uint32_t CgetriLoadVectorLu(
    GM_ADDR aarray, __gm__ int* pivots, const CgetriBatchedTilingData& tiling, __ubuf__ float* ar, __ubuf__ float* ai,
    __ubuf__ int* metadata, uint32_t batch, bool reuseLu, uint32_t usePivot)
{
    uint32_t luLd = (tiling.n + 7) / 8 * 8;
    auto address = reinterpret_cast<__gm__ uint64_t*>(aarray)[batch];
    asc_copy_gm2ub_align(
        ai, reinterpret_cast<__gm__ float*>(address), tiling.n, tiling.n * 8, 0, 0, false, 0, tiling.lda * 8, luLd * 8);
    if (tiling.usePivot != 0) {
        asc_copy_gm2ub_align(
            metadata, pivots + batch * tiling.n, 1, tiling.n * 4, 0, 0, false, 0, tiling.n * 4, tiling.n * 4);
    }
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();
    if (reuseLu && usePivot != 0) {
        asc_vf_call<CgetriDetectPivotSwapsVf>(metadata, tiling.n);
    }
    asc_vf_call<CgetriStageLuVf<false>>(ar, ai, tiling.n);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();
    if (reuseLu && usePivot != 0) {
        usePivot = static_cast<uint32_t>(metadata[184]);
    }
    return usePivot;
}

template <uint32_t FixedN>
__aicore__ __attribute__((always_inline)) inline void CgetriSolveVectorRhs(
    const CgetriBatchedTilingData& tiling, bool keepLu, __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ float* xr,
    __ubuf__ float* xi, __ubuf__ int* metadata)
{
    if constexpr (FixedN != 64) {
        uint32_t lowerStart = static_cast<uint32_t>(metadata[136]);
        if (keepLu && tiling.n <= 72) {
            asc_vf_call<CgetriRepackTailVf<true>>(xr, xi, tiling.n);
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::UB>();
            asc_vf_call<CgetriSolveTail8Vf>(ar, ai, xr, xi, tiling.n, lowerStart);
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::UB>();
            asc_vf_call<CgetriRepackTailVf<false>>(xr, xi, tiling.n);
        } else if (keepLu) {
            asc_vf_call<CgetriSolveVf<FixedN, false, 0, true>>(ar, ai, xr, xi, tiling.n, lowerStart);
        } else {
            asc_vf_call<CgetriSolveVf<FixedN>>(ar, ai, xr, xi, tiling.n, lowerStart);
        }
    } else {
        asc_vf_call<CgetriSolveVf<FixedN>>(ar, ai, xr, xi, tiling.n, 0u);
    }
}

template <uint32_t FixedN>
__aicore__ inline void CgetriVectorProcess(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR carray, GM_ADDR infoArray, const CgetriBatchedTilingData tiling,
    __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi, __ubuf__ int* metadata)
{
    uint32_t blockId = GetBlockIdx();
    uint32_t startBatch = blockId * tiling.batchPerCore;
    uint32_t numBatch = blockId == tiling.usedCoreNum - 1 ? tiling.batchTail : tiling.batchPerCore;
    uint32_t luLd = (tiling.n + 7) / 8 * 8;
    // n<=96 时，LU 实部平面末尾可容纳 4096 个 float 的输出块。
    bool reuseLu = tiling.n > 64 && tiling.n <= 96;
    uint32_t outputOffset = (tiling.n * luLd + 63) / 64 * 64;
    auto* pivots = reinterpret_cast<__gm__ int*>(pivotArray);
    auto* infos = reinterpret_cast<__gm__ int*>(infoArray);
    for (uint32_t batch = startBatch; batch < startBatch + numBatch; batch++) {
        uint32_t usePivot = tiling.usePivot;
        for (uint32_t firstCol = 0; firstCol < tiling.n; firstCol += 64) {
            bool keepLu = reuseLu && firstCol != 0;
            if (!keepLu) {
                usePivot = CgetriLoadVectorLu(aarray, pivots, tiling, ar, ai, metadata, batch, reuseLu, usePivot);
            }
            if (keepLu) {
                asc_vf_call<CgetriPrepareVf<FixedN, false, true>>(
                    ar, ai, xr, xi, metadata, tiling.n, firstCol, usePivot, 1);
            } else if (reuseLu) {
                asc_vf_call<CgetriPrepareVf<FixedN, false, false, true>>(
                    ar, ai, xr, xi, metadata, tiling.n, firstCol, usePivot, 1);
            } else {
                asc_vf_call<CgetriPrepareVf<FixedN, false>>(ar, ai, xr, xi, metadata, tiling.n, firstCol, usePivot, 1);
            }
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::UB>();
            CgetriSolveVectorRhs<FixedN>(tiling, keepLu, ar, ai, xr, xi, metadata);
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::UB>();
            if (tiling.n > 64) {
                // 非整齐尺寸同样使用向量转置和 DMA 写回，减少逐元素 GM 事务。
                asc_vf_call<CgetriStoreInfoSimt>(
                    dim3{CGETRI_SIMT_MAX_THREADS, 1, 1}, tiling.n, batch, metadata, 128u, infos);
                // SIMT 状态写回完成后再进入 SIMD 转置。
                PipeBarrier<PIPE_ALL>();
                DataSyncBarrier<MemDsbT::UB>();
                auto outputAddress = reinterpret_cast<__gm__ uint64_t*>(carray)[batch];
                CgetriWriteRhs(
                    xr, xi, reuseLu ? ar + outputOffset : ar, tiling.n, tiling.ldc, firstCol,
                    reinterpret_cast<__gm__ float*>(outputAddress));
            } else {
                asc_vf_call<CgetriUnpackSimt<FixedN>>(
                    dim3{CGETRI_SIMT_MAX_THREADS, 1, 1}, tiling.n, tiling.ldc, batch, firstCol, carray, infos, xr, xi,
                    metadata);
            }
            DataSyncBarrier<MemDsbT::UB>();
            PipeBarrier<PIPE_ALL>();
        }
    }
}

__aicore__ __attribute__((always_inline)) inline void CgetriPackedVectorProcess(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR carray, GM_ADDR infoArray, const CgetriBatchedTilingData& tiling,
    __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi, __ubuf__ int* metadata)
{
    uint32_t blockId = GetBlockIdx();
    uint32_t startBatch = blockId * tiling.batchPerCore;
    uint32_t numBatch = blockId == tiling.usedCoreNum - 1 ? tiling.batchTail : tiling.batchPerCore;
    uint32_t luLd = (tiling.n + 7) / 8 * 8;
    auto* pivots = reinterpret_cast<__gm__ int*>(pivotArray);
    auto* infos = reinterpret_cast<__gm__ int*>(infoArray);
    uint32_t groups = tiling.n <= 8 ? 8u : (tiling.n <= 16 ? 4u : 2u);
    for (uint32_t offset = 0; offset < numBatch; offset += groups) {
        uint32_t count = numBatch - offset < groups ? numBatch - offset : groups;
        for (uint32_t slot = 0; slot < groups; slot++) {
            auto address = reinterpret_cast<__gm__ uint64_t*>(aarray)[startBatch + offset + (slot < count ? slot : 0)];
            asc_copy_gm2ub_align(
                ar + 8192 + slot * tiling.n * luLd * 2, reinterpret_cast<__gm__ float*>(address), tiling.n,
                tiling.n * 8, 0, 0, false, 0, tiling.lda * 8, luLd * 8);
        }
        if (tiling.usePivot != 0) {
            asc_copy_gm2ub_align(
                metadata, pivots + (startBatch + offset) * tiling.n, 1, count * tiling.n * 4, 0, 0, false, 0,
                count * tiling.n * 4, count * tiling.n * 4);
        }
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_vf_call<CgetriStageLuVf<true>>(ar, ai, tiling.n);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        if (tiling.n == 32) {
            asc_vf_call<CgetriPrepareVf<32, true>>(ar, ai, xr, xi, metadata, tiling.n, 0, tiling.usePivot, count);
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::UB>();
            asc_vf_call<CgetriSolveVf<32, true>>(ar, ai, xr, xi, tiling.n, 0u);
        } else {
            asc_vf_call<CgetriPrepareVf<0, true>>(ar, ai, xr, xi, metadata, tiling.n, 0, tiling.usePivot, count);
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::UB>();
            asc_vf_call<CgetriSolveVf<0, true>>(ar, ai, xr, xi, tiling.n, 0u);
        }
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_vf_call<CgetriUnpackPackedSimt>(
            dim3{CGETRI_SIMT_MAX_THREADS, 1, 1}, tiling.n, tiling.ldc, startBatch + offset, count, carray, infos, xr,
            xi, metadata);
        DataSyncBarrier<MemDsbT::UB>();
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" __global__ __aicore__ void cgetri_vector_kernel(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR carray, GM_ADDR infoArray, const CgetriBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // One explicit 200 KiB arena also holds raw LU while RHS storage is unused.
    // Aligned LU columns and a 72-float RHS stride avoid UB bank conflicts.
    __ubuf__ __align__(32) float arena[51200];
    __ubuf__ __align__(32) int metadata[256];
    __ubuf__ float* ar = arena;
    __ubuf__ float* ai = arena + 16384;
    __ubuf__ float* xr = arena + 32768;
    __ubuf__ float* xi = arena + 41984;
    if (tiling.n <= 32) {
        CgetriPackedVectorProcess(aarray, pivotArray, carray, infoArray, tiling, ar, ai, xr, xi, metadata);
    } else if (tiling.n == 64) {
        CgetriVectorProcess<64>(aarray, pivotArray, carray, infoArray, tiling, ar, ai, xr, xi, metadata);
    } else if (tiling.n == 128) {
        CgetriVectorProcess<128>(aarray, pivotArray, carray, infoArray, tiling, ar, ai, xr, xi, metadata);
    } else {
        CgetriVectorProcess<0>(aarray, pivotArray, carray, infoArray, tiling, ar, ai, xr, xi, metadata);
    }
}

__aicore__ __attribute__((always_inline)) inline void CgetriSolveStreamRhs(
    const CgetriBatchedTilingData& tiling, __gm__ float* input, __ubuf__ float* raw, __ubuf__ float* ar,
    __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi, __ubuf__ int* metadata, uint32_t panelColumns,
    uint32_t luLd)
{
    // The permuted identity has no nonzero RHS above this row.
    uint32_t lowerStart = static_cast<uint32_t>(metadata[CGETRI_METADATA_STATUS]);
    for (uint32_t first = lowerStart / panelColumns * panelColumns; first < tiling.n; first += panelColumns) {
        uint32_t count = tiling.n - first < panelColumns ? tiling.n - first : panelColumns;
        asc_copy_gm2ub_align(
            raw, input + static_cast<uint64_t>(first) * tiling.lda * 2, count, tiling.n * 8, 0, 0, false, 0,
            tiling.lda * 8, luLd * 8);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_vf_call<CgetriStreamStageVf>(raw, ar, ai, count * luLd);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_vf_call<CgetriStreamPanelVf<false>>(ar, ai, xr, xi, metadata, tiling.n, first, count, lowerStart);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
    }
    uint32_t panels = (tiling.n + panelColumns - 1) / panelColumns;
    for (uint32_t reverse = 0; reverse < panels; reverse++) {
        uint32_t first = (panels - 1 - reverse) * panelColumns;
        uint32_t count = tiling.n - first < panelColumns ? tiling.n - first : panelColumns;
        asc_copy_gm2ub_align(
            raw, input + static_cast<uint64_t>(first) * tiling.lda * 2, count, tiling.n * 8, 0, 0, false, 0,
            tiling.lda * 8, luLd * 8);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_vf_call<CgetriStreamStageVf>(raw, ar, ai, count * luLd);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_vf_call<CgetriStreamPanelVf<true>>(ar, ai, xr, xi, metadata, tiling.n, first, count, 0);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
    }
}

extern "C" __global__ __aicore__ void cgetri_stream_kernel(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR carray, GM_ADDR infoArray, const CgetriBatchedTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // 保留 64 列 RHS；随矩阵增大缩减 LU 面板，避免中等尺寸过早切换到多 kernel 路径。
    __ubuf__ __align__(32) float arena[53248];
    __ubuf__ __align__(32) int metadata[CGETRI_METADATA_ELEMENTS];
    uint32_t luLd = (tiling.n + 7) / 8 * 8;
    uint32_t panelColumns = tiling.n <= 256 ? 16u : (tiling.n <= 288 ? 8u : 4u);
    uint32_t rhsElements = tiling.n * 72;
    // SIMD 按 64 个 float 处理尾部；4 列面板也必须为每个平面保留完整的末尾向量。
    uint32_t luElements = (luLd * panelColumns + 63) / 64 * 64;
    auto* xr = arena;
    auto* xi = arena + rhsElements;
    auto* ar = arena + 2 * rhsElements;
    auto* ai = arena + 2 * rhsElements + luElements;
    auto* raw = arena + 2 * rhsElements + 2 * luElements;
    uint32_t startBatch = GetBlockIdx() * tiling.batchPerCore;
    uint32_t numBatch = GetBlockIdx() == tiling.usedCoreNum - 1 ? tiling.batchTail : tiling.batchPerCore;
    auto* pivots = reinterpret_cast<__gm__ int*>(pivotArray);
    auto* infos = reinterpret_cast<__gm__ int*>(infoArray);
    for (uint32_t batch = startBatch; batch < startBatch + numBatch; batch++) {
        auto inputAddress = reinterpret_cast<__gm__ uint64_t*>(aarray)[batch];
        auto outputAddress = reinterpret_cast<__gm__ uint64_t*>(carray)[batch];
        auto* input = reinterpret_cast<__gm__ float*>(inputAddress);
        if (tiling.usePivot != 0) {
            asc_copy_gm2ub_align(
                metadata, pivots + batch * tiling.n, 1, tiling.n * 4, 0, 0, false, 0, tiling.n * 4, tiling.n * 4);
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::UB>();
        }
        for (uint32_t firstCol = 0; firstCol < tiling.n; firstCol += 64) {
            asc_vf_call<CgetriInitMetadataVf>(metadata, tiling.n, firstCol);
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::UB>();
            if (tiling.usePivot != 0) {
                asc_vf_call<CgetriInitPermutationVf>(metadata, tiling.n, 0, tiling.n);
                PipeBarrier<PIPE_ALL>();
                DataSyncBarrier<MemDsbT::UB>();
            }
            asc_vf_call<CgetriStreamIdentityVf>(xr, xi, metadata, tiling.n, firstCol);
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::UB>();
            CgetriSolveStreamRhs(tiling, input, raw, ar, ai, xr, xi, metadata, panelColumns, luLd);
            asc_vf_call<CgetriStoreInfoSimt>(
                dim3{CGETRI_SIMT_MAX_THREADS, 1, 1}, tiling.n, batch, metadata, CGETRI_METADATA_AUX, infos);
            CgetriWriteRhs(xr, xi, raw, tiling.n, tiling.ldc, firstCol, reinterpret_cast<__gm__ float*>(outputAddress));
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::UB>();
        }
    }
}

void cgetri_batched_kernel_do(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR carray, GM_ADDR infoArray, const CgetriBatchedTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    if (tiling.n < CGETRI_SMALL_N) {
        cgetri_batched_kernel<<<numBlocks, nullptr, stream>>>(aarray, pivotArray, carray, infoArray, tiling);
    } else if (tiling.n > 128) {
        cgetri_stream_kernel<<<numBlocks, nullptr, stream>>>(aarray, pivotArray, carray, infoArray, tiling);
    } else {
        cgetri_vector_kernel<<<numBlocks, nullptr, stream>>>(aarray, pivotArray, carray, infoArray, tiling);
    }
}
