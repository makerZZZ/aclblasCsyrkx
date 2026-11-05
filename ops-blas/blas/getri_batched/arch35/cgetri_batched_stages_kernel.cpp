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
 * \file cgetri_batched_stages_kernel.cpp
 * \brief 分块求逆的 init/solve/combine/finalize 阶段 kernel。
 */

#include "cgetri_batched_kernel.h"
#include "cgetri_batched_kernel_common.h"
#include "cgetri_batched_solve_common.h"

using namespace AscendC;
using namespace CgetriBatched;

// ============================================================================
// init: 分块求逆的 LU 拆分、主元初始化和奇异性检查。
// ============================================================================
namespace {

__simd_vf__ inline void CgetriInitializeTileVf(
    __ubuf__ float* raw, __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* cr, __ubuf__ float* ci,
    __ubuf__ int* metadata, uint32_t n, uint32_t firstRow, uint32_t firstCol, uint32_t rows, uint32_t columns)
{
    Reg::RegTensor<float> re, im, rhs, zero;
    Reg::RegTensor<int32_t> row, column, permutation, status, nonDiagonal, sentinel, candidate, one, reduced;
    Reg::MaskReg valid, diagonal, zeroReal, zeroImag, singular, offDiagonal, nonzeroReal, nonzeroImag, nonzero,
        identity;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Duplicate(zero, 0.0f);
    Reg::Duplicate(one, 1);
    Reg::Duplicate(sentinel, static_cast<int32_t>(n + 1));
    Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_BRC_B32>(status, metadata + CGETRI_METADATA_STATUS);
    Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_BRC_B32>(nonDiagonal, metadata + CGETRI_METADATA_AUX);
    for (uint16_t offset = 0; offset < columns * 128; offset += 64) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_DINTLV_B32>(re, im, raw + offset * 2);
        Reg::StoreAlign(real + offset, re, all);
        Reg::StoreAlign(imag + offset, im, all);
        Reg::Arange(row, static_cast<int32_t>(firstRow + offset % 128));
        Reg::Duplicate(column, static_cast<int32_t>(firstCol + offset / 128));
        Reg::Compares<int32_t, CMPMODE::LT>(valid, row, static_cast<int32_t>(firstRow + rows), all);
        Reg::Compare<int32_t, CMPMODE::EQ>(diagonal, row, column, valid);
        Reg::Compares<float, CMPMODE::EQ>(zeroReal, re, 0.0f, diagonal);
        Reg::Compares<float, CMPMODE::EQ>(zeroImag, im, 0.0f, diagonal);
        Reg::And(singular, zeroReal, zeroImag, all);
        Reg::Adds(candidate, row, 1, all);
        Reg::Select(candidate, candidate, sentinel, singular);
        Reg::Min(status, status, candidate, all);
        Reg::Compare<int32_t, CMPMODE::NE>(offDiagonal, row, column, valid);
        Reg::Compares<float, CMPMODE::NE>(nonzeroReal, re, 0.0f, offDiagonal);
        Reg::Compares<float, CMPMODE::NE>(nonzeroImag, im, 0.0f, offDiagonal);
        Reg::Or(nonzero, nonzeroReal, nonzeroImag, all);
        Reg::Select(nonDiagonal, one, nonDiagonal, nonzero);
        Reg::LoadAlign<int32_t, Reg::LoadDist::DIST_BRC_B32>(
            permutation, metadata + CGETRI_METADATA_PERMUTATION + offset / 128);
        Reg::Compare<int32_t, CMPMODE::EQ>(identity, row, permutation, valid);
        Reg::Duplicate(rhs, 1.0f, identity);
        Reg::StoreAlign(cr + offset, rhs, all);
        Reg::StoreAlign(ci + offset, zero, all);
    }
    uint32_t oneElement = 1;
    auto scalar = Reg::UpdateMask<int32_t>(oneElement);
    Reg::Reduce<Reg::ReduceType::MIN>(reduced, status, all);
    Reg::StoreAlign(metadata + CGETRI_METADATA_STATUS, reduced, scalar);
    Reg::Reduce<Reg::ReduceType::MAX>(reduced, nonDiagonal, all);
    Reg::StoreAlign(metadata + CGETRI_METADATA_AUX, reduced, scalar);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(CGETRI_SIMT_MAX_THREADS) inline void CgetriInitInfoSimt(
    uint32_t n, uint32_t batch, uint64_t matrixBase, __ubuf__ int* metadata, __gm__ int* infos, __gm__ float* ar,
    __gm__ float* ai, __gm__ float* cr, __gm__ float* ci)
{
    int status = metadata[CGETRI_METADATA_STATUS] == static_cast<int>(n + 1) ? 0 : metadata[CGETRI_METADATA_STATUS];
    if (status == 0 && metadata[CGETRI_METADATA_AUX] == 0) {
        status = CGETRI_DIAGONAL_FAST_PATH_INFO;
        for (uint32_t row = threadIdx.x; row < n; row += blockDim.x) {
            uint64_t index = matrixBase + static_cast<uint64_t>(row) * (n + 1);
            aclblasComplex inverse = ComplexDiv({1.0f, 0.0f}, {ar[index], ai[index]});
            cr[index] = inverse.real;
            ci[index] = inverse.imag;
        }
    }
    if (threadIdx.x == 0) {
        infos[batch] = status;
    }
}

__aicore__ __attribute__((always_inline)) inline void CgetriInitializeBlockedColumn(
    const CgetriBlockedInitTilingData& tiling, __ubuf__ float* arena, __ubuf__ int* metadata, __gm__ float* input,
    __gm__ float* realGm, __gm__ float* imagGm, __gm__ float* crGm, __gm__ float* ciGm, uint64_t matrixBase,
    uint32_t col, uint32_t columns)
{
    auto* raw = arena;
    auto* real = arena + 8192;
    auto* imag = arena + 12288;
    auto* cr = arena + 16384;
    auto* ci = arena + 20480;
    for (uint32_t row = 0; row < tiling.n; row += 128) {
        uint32_t rows = tiling.n - row < 128 ? tiling.n - row : 128;
        uint64_t offset = matrixBase + static_cast<uint64_t>(col) * tiling.n + row;
        asc_copy_gm2ub_align(
            raw, input + (static_cast<uint64_t>(col) * tiling.lda + row) * 2, columns, rows * 8, 0, 0, false, 0,
            tiling.lda * 8, 128 * 8);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_vf_call<CgetriInitializeTileVf>(raw, real, imag, cr, ci, metadata, tiling.n, row, col, rows, columns);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_copy_ub2gm_align(realGm + offset, real, columns, rows * 4, 0, tiling.n * 4, 128 * 4);
        asc_copy_ub2gm_align(imagGm + offset, imag, columns, rows * 4, 0, tiling.n * 4, 128 * 4);
        asc_copy_ub2gm_align(crGm + offset, cr, columns, rows * 4, 0, tiling.n * 4, 128 * 4);
        asc_copy_ub2gm_align(ciGm + offset, ci, columns, rows * 4, 0, tiling.n * 4, 128 * 4);
        PipeBarrier<PIPE_ALL>();
    }
}

} // namespace

extern "C" __global__ __aicore__ void cgetri_blocked_init_kernel(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR infoArray, GM_ADDR aReal, GM_ADDR aImag, GM_ADDR cReal, GM_ADDR cImag,
    const CgetriBlockedInitTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ __align__(32) float arena[24576];
    __ubuf__ __align__(32) int metadata[CGETRI_METADATA_ELEMENTS];
    auto* realGm = reinterpret_cast<__gm__ float*>(aReal);
    auto* imagGm = reinterpret_cast<__gm__ float*>(aImag);
    auto* crGm = reinterpret_cast<__gm__ float*>(cReal);
    auto* ciGm = reinterpret_cast<__gm__ float*>(cImag);
    auto* pivots = reinterpret_cast<__gm__ int*>(pivotArray);
    uint32_t blockId = GetBlockIdx();
    uint32_t startBatch = blockId * tiling.batchPerCore;
    uint32_t numBatch = blockId == tiling.usedCoreNum - 1 ? tiling.batchTail : tiling.batchPerCore;
    for (uint32_t batch = startBatch; batch < startBatch + numBatch; batch++) {
        auto address = reinterpret_cast<__gm__ uint64_t*>(aarray)[batch];
        auto* input = reinterpret_cast<__gm__ float*>(address);
        uint64_t matrixBase = static_cast<uint64_t>(batch) * tiling.matrixStride;
        for (uint32_t col = 0; col < tiling.n; col += 32) {
            uint32_t columns = tiling.n - col < 32 ? tiling.n - col : 32;
            asc_vf_call<CgetriInitMetadataVf>(metadata, tiling.n, col);
            PipeBarrier<PIPE_ALL>();
            DataSyncBarrier<MemDsbT::UB>();
            if (tiling.usePivot != 0) {
                for (uint32_t first = 0; first < tiling.n; first += 256) {
                    uint32_t count = tiling.n - first < 256 ? tiling.n - first : 256;
                    asc_copy_gm2ub_align(
                        metadata, pivots + batch * tiling.n + first, 1, count * 4, 0, 0, false, 0, count * 4,
                        count * 4);
                    PipeBarrier<PIPE_ALL>();
                    DataSyncBarrier<MemDsbT::UB>();
                    asc_vf_call<CgetriInitPermutationVf>(metadata, tiling.n, first, count);
                    PipeBarrier<PIPE_ALL>();
                    DataSyncBarrier<MemDsbT::UB>();
                }
            }
            CgetriInitializeBlockedColumn(
                tiling, arena, metadata, input, realGm, imagGm, crGm, ciGm, matrixBase, col, columns);
        }
        asc_vf_call<CgetriInitInfoSimt>(
            dim3{CGETRI_SIMT_MAX_THREADS, 1, 1}, tiling.n, batch, matrixBase, metadata,
            reinterpret_cast<__gm__ int*>(infoArray), realGm, imagGm, crGm, ciGm);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
    }
}

void cgetri_blocked_init_kernel_do(
    GM_ADDR aarray, GM_ADDR pivotArray, GM_ADDR infoArray, GM_ADDR aReal, GM_ADDR aImag, GM_ADDR cReal, GM_ADDR cImag,
    const CgetriBlockedInitTilingData& tiling, uint32_t numBlocks, void* stream)
{
    cgetri_blocked_init_kernel<<<numBlocks, nullptr, stream>>>(
        aarray, pivotArray, infoArray, aReal, aImag, cReal, cImag, tiling);
}

// ============================================================================
// solve: 分块三角求解。
// ============================================================================
namespace {

template <bool Write>
__simd_vf__ inline void CgetriTransposeRhsVf(
    __ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi, uint32_t rows, uint32_t columns)
{
    uint32_t luLd = (rows + 7) / 8 * 8;
    Reg::RegTensor<uint32_t> indices;
    Reg::RegTensor<float> real, imag;
    auto all = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(indices), 0);
    if constexpr (Write) {
        Reg::Muls(indices, indices, 72u, all);
        for (uint16_t col = 0; col < columns; col++) {
            uint32_t remaining = rows;
            for (uint16_t first = 0; first < rows; first += 64) {
                auto mask = Reg::UpdateMask<float>(remaining);
                Reg::Gather(real, xr + first * 72 + col, indices, mask);
                Reg::Gather(imag, xi + first * 72 + col, indices, mask);
                Reg::StoreAlign(ar + col * luLd + first, real, mask);
                Reg::StoreAlign(ai + col * luLd + first, imag, mask);
            }
        }
    } else {
        Reg::Muls(indices, indices, luLd, all);
        auto mask = Reg::UpdateMask<float>(columns);
        for (uint16_t row = 0; row < rows; row++) {
            Reg::Gather(real, ar + row, indices, mask);
            Reg::Gather(imag, ai + row, indices, mask);
            Reg::StoreAlign(xr + row * 72, real, all);
            Reg::StoreAlign(xi + row * 72, imag, all);
        }
    }
}

__simd_vf__ inline void CgetriNegateLuVf(__ubuf__ float* ar, __ubuf__ float* ai, uint32_t elements)
{
    Reg::RegTensor<float> real, imag;
    auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t first = 0; first < elements; first += 64) {
        Reg::LoadAlign(real, ar + first);
        Reg::LoadAlign(imag, ai + first);
        Reg::Neg(real, real, mask);
        Reg::Neg(imag, imag, mask);
        Reg::StoreAlign(ar + first, real, mask);
        Reg::StoreAlign(ai + first, imag, mask);
    }
}

__aicore__ __attribute__((always_inline)) inline void CgetriLoadBlockedTile(
    const CgetriBlockedSolveTilingData& tiling, __ubuf__ float* arena, __gm__ float* aRealGm, __gm__ float* aImagGm,
    __gm__ float* cRealGm, __gm__ float* cImagGm, uint32_t columns, uint64_t luOffset, uint64_t rhsOffset)
{
    auto* ar = arena;
    auto* ai = arena + 16384;
    auto* xr = arena + 32768;
    auto* xi = arena + 41984;
    uint32_t luLd = (tiling.blockSize + 7) / 8 * 8;
    asc_copy_gm2ub_align(
        ar, cRealGm + rhsOffset, columns, tiling.blockSize * 4, 0, 0, false, 0, tiling.n * 4, luLd * 4);
    asc_copy_gm2ub_align(
        ai, cImagGm + rhsOffset, columns, tiling.blockSize * 4, 0, 0, false, 0, tiling.n * 4, luLd * 4);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();
    asc_vf_call<CgetriTransposeRhsVf<false>>(ar, ai, xr, xi, tiling.blockSize, columns);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();
    asc_copy_gm2ub_align(
        ar, aRealGm + luOffset, tiling.blockSize, tiling.blockSize * 4, 0, 0, false, 0, tiling.n * 4, luLd * 4);
    asc_copy_gm2ub_align(
        ai, aImagGm + luOffset, tiling.blockSize, tiling.blockSize * 4, 0, 0, false, 0, tiling.n * 4, luLd * 4);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();
    asc_vf_call<CgetriNegateLuVf>(ar, ai, tiling.blockSize * luLd);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::UB>();
}

} // namespace

extern "C" __global__ __aicore__ void cgetri_blocked_solve_kernel(
    GM_ADDR aReal, GM_ADDR aImag, GM_ADDR cReal, GM_ADDR cImag, GM_ADDR infoArray,
    const CgetriBlockedSolveTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ __align__(32) float arena[51200];
    __ubuf__ float* ar = arena;
    __ubuf__ float* ai = arena + 16384;
    __ubuf__ float* xr = arena + 32768;
    __ubuf__ float* xi = arena + 41984;
    uint32_t rhsTiles = (tiling.n + 63) / 64;
    uint64_t totalTiles = static_cast<uint64_t>(tiling.batchSize) * rhsTiles;
    uint32_t luLd = (tiling.blockSize + 7) / 8 * 8;
    auto* infos = reinterpret_cast<__gm__ int*>(infoArray);
    auto* aRealGm = reinterpret_cast<__gm__ float*>(aReal);
    auto* aImagGm = reinterpret_cast<__gm__ float*>(aImag);
    auto* cRealGm = reinterpret_cast<__gm__ float*>(cReal);
    auto* cImagGm = reinterpret_cast<__gm__ float*>(cImag);
    for (uint64_t tile = GetBlockIdx(); tile < totalTiles; tile += tiling.usedCoreNum) {
        uint32_t batch = static_cast<uint32_t>(tile / rhsTiles);
        if (infos[batch] != 0) {
            continue;
        }
        uint32_t firstCol = static_cast<uint32_t>(tile % rhsTiles) * 64;
        uint32_t columns = tiling.n - firstCol < 64 ? tiling.n - firstCol : 64;
        uint64_t matrixBase = static_cast<uint64_t>(batch) * tiling.matrixStride;
        uint64_t luOffset = matrixBase + static_cast<uint64_t>(tiling.rowStart) * (tiling.n + 1);
        uint64_t rhsOffset = matrixBase + tiling.rowStart + static_cast<uint64_t>(firstCol) * tiling.n;
        CgetriLoadBlockedTile(tiling, arena, aRealGm, aImagGm, cRealGm, cImagGm, columns, luOffset, rhsOffset);
        if (tiling.isLower != 0) {
            asc_vf_call<CgetriSolveVf<0, false, 1>>(ar, ai, xr, xi, tiling.blockSize, 0u);
        } else {
            asc_vf_call<CgetriSolveVf<0, false, 2>>(ar, ai, xr, xi, tiling.blockSize, 0u);
        }
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_vf_call<CgetriTransposeRhsVf<true>>(ar, ai, xr, xi, tiling.blockSize, columns);
        DataSyncBarrier<MemDsbT::UB>();
        PipeBarrier<PIPE_ALL>();
        asc_copy_ub2gm_align(cRealGm + rhsOffset, ar, columns, tiling.blockSize * 4, 0, tiling.n * 4, luLd * 4);
        asc_copy_ub2gm_align(cImagGm + rhsOffset, ai, columns, tiling.blockSize * 4, 0, tiling.n * 4, luLd * 4);
        PipeBarrier<PIPE_ALL>();
    }
}

void cgetri_blocked_solve_kernel_do(
    GM_ADDR aReal, GM_ADDR aImag, GM_ADDR cReal, GM_ADDR cImag, GM_ADDR infoArray,
    const CgetriBlockedSolveTilingData& tiling, uint32_t numBlocks, void* stream)
{
    cgetri_blocked_solve_kernel<<<numBlocks, nullptr, stream>>>(aReal, aImag, cReal, cImag, infoArray, tiling);
}

// ============================================================================
// combine: 合并 Cube 计算的复数更新结果。
// ============================================================================
namespace {

__simd_vf__ inline void CgetriCombineVf(
    __ubuf__ float* first, __ubuf__ float* second, __ubuf__ float* target, uint32_t elements, uint32_t isImag)
{
    Reg::RegTensor<float> lhs, rhs, value;
    auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t offset = 0; offset < elements; offset += 64) {
        Reg::LoadAlign(lhs, first + offset);
        Reg::LoadAlign(rhs, second + offset);
        Reg::LoadAlign(value, target + offset);
        if (isImag != 0) {
            Reg::Add(lhs, lhs, rhs, mask);
        } else {
            Reg::Sub(lhs, lhs, rhs, mask);
        }
        Reg::Sub(value, value, lhs, mask);
        Reg::StoreAlign(target + offset, value, mask);
    }
}

} // namespace

extern "C" __global__ __aicore__ void cgetri_blocked_combine_kernel(
    GM_ADDR temp1, GM_ADDR temp2, GM_ADDR target, GM_ADDR infoArray, const CgetriBlockedCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ __align__(32) float arena[12288];
    auto* first = arena;
    auto* second = arena + 4096;
    auto* value = arena + 8192;
    auto* firstGm = reinterpret_cast<__gm__ float*>(temp1);
    auto* secondGm = reinterpret_cast<__gm__ float*>(temp2);
    auto* targetGm = reinterpret_cast<__gm__ float*>(target);
    auto* infos = reinterpret_cast<__gm__ int*>(infoArray);
    uint32_t rowTiles = (tiling.updateRows + 127) / 128;
    uint32_t colTiles = (tiling.n + 31) / 32;
    uint64_t tilesPerBatch = static_cast<uint64_t>(rowTiles) * colTiles;
    for (uint64_t tile = GetBlockIdx(); tile < tilesPerBatch * tiling.batchSize; tile += tiling.usedCoreNum) {
        uint32_t batch = static_cast<uint32_t>(tile / tilesPerBatch);
        if (infos[batch] != 0) {
            continue;
        }
        uint32_t row = static_cast<uint32_t>((tile % tilesPerBatch) / colTiles) * 128;
        uint32_t col = static_cast<uint32_t>(tile % colTiles) * 32;
        uint32_t rows = tiling.updateRows - row < 128 ? tiling.updateRows - row : 128;
        uint32_t columns = tiling.n - col < 32 ? tiling.n - col : 32;
        uint32_t stride = (rows + 7) / 8 * 8;
        uint64_t tempOffset = static_cast<uint64_t>(batch) * tiling.tempMatrixStride + col * tiling.tempRowStride + row;
        uint64_t targetOffset =
            static_cast<uint64_t>(batch) * tiling.matrixStride + col * tiling.n + tiling.targetRow + row;
        asc_copy_gm2ub_align(
            first, firstGm + tempOffset, columns, rows * 4, 0, 0, false, 0, tiling.tempRowStride * 4, stride * 4);
        asc_copy_gm2ub_align(
            second, secondGm + tempOffset, columns, rows * 4, 0, 0, false, 0, tiling.tempRowStride * 4, stride * 4);
        asc_copy_gm2ub_align(
            value, targetGm + targetOffset, columns, rows * 4, 0, 0, false, 0, tiling.n * 4, stride * 4);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_vf_call<CgetriCombineVf>(first, second, value, stride * columns, tiling.isImag);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_copy_ub2gm_align(targetGm + targetOffset, value, columns, rows * 4, 0, tiling.n * 4, stride * 4);
        PipeBarrier<PIPE_ALL>();
    }
}

void cgetri_blocked_combine_kernel_do(
    GM_ADDR temp1, GM_ADDR temp2, GM_ADDR target, GM_ADDR infoArray, const CgetriBlockedCombineTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    cgetri_blocked_combine_kernel<<<numBlocks, nullptr, stream>>>(temp1, temp2, target, infoArray, tiling);
}

// ============================================================================
// finalize: 回写复数矩阵并完成状态码转换。
// ============================================================================
namespace {

__simd_vf__ inline void CgetriInterleaveVf(
    __ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* output, uint32_t elements)
{
    Reg::RegTensor<float> re, im;
    auto mask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    for (uint16_t offset = 0; offset < elements; offset += 64) {
        Reg::LoadAlign(re, real + offset);
        Reg::LoadAlign(im, imag + offset);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_INTLV_B32>(output + offset * 2, re, im, mask);
    }
}

} // namespace

extern "C" __global__ __aicore__ void cgetri_blocked_finalize_kernel(
    GM_ADDR cReal, GM_ADDR cImag, GM_ADDR carray, GM_ADDR infoArray, const CgetriBlockedFinalizeTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ __align__(32) float arena[16384];
    auto* real = arena;
    auto* imag = arena + 4096;
    auto* output = arena + 8192;
    auto* realGm = reinterpret_cast<__gm__ float*>(cReal);
    auto* imagGm = reinterpret_cast<__gm__ float*>(cImag);
    auto* infos = reinterpret_cast<__gm__ int*>(infoArray);
    uint32_t rowTiles = (tiling.n + 127) / 128;
    uint32_t colTiles = (tiling.n + 31) / 32;
    uint64_t tilesPerBatch = static_cast<uint64_t>(rowTiles) * colTiles;
    for (uint64_t tile = GetBlockIdx(); tile < tilesPerBatch * tiling.batchSize; tile += tiling.usedCoreNum) {
        uint32_t batch = static_cast<uint32_t>(tile / tilesPerBatch);
        if (infos[batch] > 0) {
            continue;
        }
        uint32_t row = static_cast<uint32_t>((tile % tilesPerBatch) / colTiles) * 128;
        uint32_t col = static_cast<uint32_t>(tile % colTiles) * 32;
        uint32_t rows = tiling.n - row < 128 ? tiling.n - row : 128;
        uint32_t columns = tiling.n - col < 32 ? tiling.n - col : 32;
        uint32_t stride = (rows + 7) / 8 * 8;
        uint64_t sourceOffset = static_cast<uint64_t>(batch) * tiling.matrixStride + col * tiling.n + row;
        auto address = reinterpret_cast<__gm__ uint64_t*>(carray)[batch];
        auto* target = reinterpret_cast<__gm__ float*>(address) + (static_cast<uint64_t>(col) * tiling.ldc + row) * 2;
        asc_copy_gm2ub_align(real, realGm + sourceOffset, columns, rows * 4, 0, 0, false, 0, tiling.n * 4, stride * 4);
        asc_copy_gm2ub_align(imag, imagGm + sourceOffset, columns, rows * 4, 0, 0, false, 0, tiling.n * 4, stride * 4);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_vf_call<CgetriInterleaveVf>(real, imag, output, columns * stride);
        PipeBarrier<PIPE_ALL>();
        DataSyncBarrier<MemDsbT::UB>();
        asc_copy_ub2gm_align(target, output, columns, rows * 8, 0, tiling.ldc * 8, stride * 8);
        PipeBarrier<PIPE_ALL>();
    }
    // One scalar core owns the packed info cache lines.
    if (GetBlockIdx() == 0) {
        for (uint32_t batch = 0; batch < tiling.batchSize; batch++) {
            if (infos[batch] == CGETRI_DIAGONAL_FAST_PATH_INFO) {
                infos[batch] = 0;
            }
        }
    }
}

void cgetri_blocked_finalize_kernel_do(
    GM_ADDR cReal, GM_ADDR cImag, GM_ADDR carray, GM_ADDR infoArray, const CgetriBlockedFinalizeTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    cgetri_blocked_finalize_kernel<<<numBlocks, nullptr, stream>>>(cReal, cImag, carray, infoArray, tiling);
}
