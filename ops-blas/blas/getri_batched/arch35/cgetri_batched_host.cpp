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
 * \file cgetri_batched_host.cpp
 * \brief Host-side API for batched single-precision complex matrix inversion.
 */

#include <algorithm>
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "cgetri_batched_kernel.h"
#include "cgetri_batched_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "gemm_batched/arch35/gemm_batched_kernel.h"

namespace {

constexpr size_t CGETRI_WORKSPACE_ALIGN = 256;

struct CgetriBlockedWorkspace {
    uint8_t* base = nullptr;
    uint8_t* aReal = nullptr;
    uint8_t* aImag = nullptr;
    uint8_t* cReal = nullptr;
    uint8_t* cImag = nullptr;
    uint8_t* temp1 = nullptr;
    uint8_t* temp2 = nullptr;
    uint32_t matrixStride = 0;
    uint32_t tempRowStride = 0;
    uint32_t tempMatrixStride = 0;
    size_t requiredBytes = 0;
};

size_t AlignUpSize(size_t value)
{
    return (value + CGETRI_WORKSPACE_ALIGN - 1) / CGETRI_WORKSPACE_ALIGN * CGETRI_WORKSPACE_ALIGN;
}

bool CheckedAddSize(size_t lhs, size_t rhs, size_t& result)
{
    if (lhs > std::numeric_limits<size_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool CheckedMulSize(size_t lhs, size_t rhs, size_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

aclblasStatus_t ValidateCgetriBatchedParams(
    int n, int lda, int ldc, int batchSize, const aclblasComplex* const Aarray[], aclblasComplex* const Carray[],
    int* infoArray)
{
    if (n < 0) {
        OP_LOGE("aclblasCgetriBatched", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (batchSize < 0) {
        OP_LOGE("aclblasCgetriBatched", "batchSize must be >= 0, got %d", batchSize);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasCgetriBatched", "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (ldc < std::max(1, n)) {
        OP_LOGE("aclblasCgetriBatched", "ldc must be >= max(1, n), got ldc=%d, n=%d", ldc, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || batchSize == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (Aarray == nullptr) {
        OP_LOGE("aclblasCgetriBatched", "Aarray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (Carray == nullptr) {
        OP_LOGE("aclblasCgetriBatched", "Carray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (infoArray == nullptr) {
        OP_LOGE("aclblasCgetriBatched", "infoArray must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

uint32_t CalcCgetriAivCores(uint32_t availableCores, uint64_t totalTasks)
{
    constexpr uint32_t tasksPerCore = 16384;
    uint64_t requested = (totalTasks + tasksPerCore - 1) / tasksPerCore;
    requested = std::max<uint64_t>(1, requested);
    return static_cast<uint32_t>(std::min<uint64_t>(availableCores, requested));
}

aclblasStatus_t CalcCgetriBlockedWorkspaceLayout(
    size_t matrixBytes, size_t tempBytes, size_t (&offsets)[6], size_t& requiredBytes)
{
    size_t cursor = 0;
    auto reserve = [&](size_t bytes, size_t& offset) -> bool {
        cursor = AlignUpSize(cursor);
        offset = cursor;
        return CheckedAddSize(cursor, bytes, cursor);
    };
    if (!reserve(matrixBytes, offsets[0]) || !reserve(matrixBytes, offsets[1]) || !reserve(matrixBytes, offsets[2]) ||
        !reserve(matrixBytes, offsets[3]) || !reserve(tempBytes, offsets[4]) || !reserve(tempBytes, offsets[5])) {
        OP_LOGE(
            "aclblasCgetriBatched", "blocked workspace layout overflow, matrixBytes=%zu tempBytes=%zu", matrixBytes,
            tempBytes);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    requiredBytes = cursor;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t PrepareCgetriBlockedWorkspace(
    _aclblas_handle* handle, int n, int batchSize, CgetriBlockedWorkspace& workspace)
{
    // SIMT diagonal stores and DMA initialization from adjacent batch owners
    // must not share a GM cache line, including when n is odd.
    constexpr uint64_t matrixAlign = CGETRI_WORKSPACE_ALIGN / sizeof(float);
    uint64_t matrixElements = static_cast<uint64_t>(n) * static_cast<uint64_t>(n);
    uint64_t matrixStride64 = (matrixElements + matrixAlign - 1) / matrixAlign * matrixAlign;
    uint32_t tempRowStride = CeilAlign<uint32_t>(static_cast<uint32_t>(n), GEMM_BATCHED_L0C_C0);
    uint64_t tempMatrixStride64 = static_cast<uint64_t>(tempRowStride) * static_cast<uint64_t>(n);
    if (matrixStride64 > std::numeric_limits<uint32_t>::max() ||
        tempMatrixStride64 > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE("aclblasCgetriBatched", "blocked workspace stride overflow, n=%d", n);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    size_t matrixBytes = 0;
    size_t tempBytes = 0;
    if (!CheckedMulSize(static_cast<size_t>(matrixStride64), static_cast<size_t>(batchSize), matrixBytes) ||
        !CheckedMulSize(matrixBytes, sizeof(float), matrixBytes) ||
        !CheckedMulSize(static_cast<size_t>(tempMatrixStride64), static_cast<size_t>(batchSize), tempBytes) ||
        !CheckedMulSize(tempBytes, sizeof(float), tempBytes)) {
        OP_LOGE("aclblasCgetriBatched", "blocked workspace byte count overflow, n=%d batch=%d", n, batchSize);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    size_t cursor = 0;
    size_t offsets[6] = {};
    aclblasStatus_t layoutRet = CalcCgetriBlockedWorkspaceLayout(matrixBytes, tempBytes, offsets, cursor);
    if (layoutRet != ACLBLAS_STATUS_SUCCESS) {
        return layoutRet;
    }

    aclblasStatus_t workspaceRet = EnsureDefaultWorkspace(handle, cursor);
    if (workspaceRet != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasCgetriBatched", "failed to reserve blocked workspace, required=%zu", cursor);
        return workspaceRet;
    }

    workspace.base = static_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    if (workspace.base == nullptr) {
        OP_LOGE("aclblasCgetriBatched", "blocked workspace is nullptr after allocation, required=%zu", cursor);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    workspace.aReal = workspace.base + offsets[0];
    workspace.aImag = workspace.base + offsets[1];
    workspace.cReal = workspace.base + offsets[2];
    workspace.cImag = workspace.base + offsets[3];
    workspace.temp1 = workspace.base + offsets[4];
    workspace.temp2 = workspace.base + offsets[5];
    workspace.matrixStride = static_cast<uint32_t>(matrixStride64);
    workspace.tempRowStride = tempRowStride;
    workspace.tempMatrixStride = static_cast<uint32_t>(tempMatrixStride64);
    workspace.requiredBytes = cursor;
    return ACLBLAS_STATUS_SUCCESS;
}

void CalcCgetriMnBlockSplit(int m, int n, uint32_t cubeCoreNum, int32_t& bestMBlocks, int32_t& bestNBlocks)
{
    int32_t mTiles = (m + static_cast<int>(GEMM_BATCHED_BASE_M) - 1) / static_cast<int>(GEMM_BATCHED_BASE_M);
    int32_t nTiles = (n + static_cast<int>(GEMM_BATCHED_BASE_N) - 1) / static_cast<int>(GEMM_BATCHED_BASE_N);
    bestMBlocks = 1;
    bestNBlocks = 1;
    int32_t bestUtilization = 0;
    for (int32_t mBlocks = 1; mBlocks <= mTiles && mBlocks <= static_cast<int32_t>(cubeCoreNum); mBlocks++) {
        int32_t nBlocks = std::min(nTiles, static_cast<int32_t>(cubeCoreNum) / mBlocks);
        nBlocks = std::max(1, nBlocks);
        int32_t utilization = mBlocks * nBlocks;
        if (utilization > bestUtilization && utilization <= static_cast<int32_t>(cubeCoreNum)) {
            bestUtilization = utilization;
            bestMBlocks = mBlocks;
            bestNBlocks = nBlocks;
        }
    }
}

aclblasStatus_t CalcCgetriGemmTiling(
    int m, int n, int k, int batchSize, int lda, int ldb, int ldc, uint32_t cubeCoreNum,
    GemmBatchedGemmTilingData& tiling)
{
    tiling = GemmBatchedGemmTilingData{};
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.k = static_cast<uint32_t>(k);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.ldb = static_cast<uint32_t>(ldb);
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.batchCount = static_cast<uint32_t>(batchSize);
    tiling.dtypeCase = GEMM_BATCHED_DTYPE_FP32;
    tiling.tileM = GEMM_BATCHED_DEFAULT_TILE_M;
    tiling.tileN = GEMM_BATCHED_DEFAULT_TILE_N;
    tiling.tileKChunk = GEMM_BATCHED_DEFAULT_TILE_K_CHUNK;

    // The Tensor API GEMM implementation is reliable only when a physical AIC
    // executes one logical task in a launch.  Split the spatial work within the
    // cores left after assigning one concurrent matrix to each batch item.
    uint32_t coresPerBatch = std::max<uint32_t>(1, cubeCoreNum / static_cast<uint32_t>(batchSize));
    int32_t mBlocks = 1;
    int32_t nBlocks = 1;
    CalcCgetriMnBlockSplit(m, n, coresPerBatch, mBlocks, nBlocks);
    tiling.mBlocks = static_cast<uint32_t>(mBlocks);
    tiling.nBlocks = static_cast<uint32_t>(nBlocks);
    tiling.singleCoreM = static_cast<uint32_t>(
        ((m + mBlocks - 1) / mBlocks + static_cast<int>(GEMM_BATCHED_BASE_M) - 1) /
        static_cast<int>(GEMM_BATCHED_BASE_M) * static_cast<int>(GEMM_BATCHED_BASE_M));
    tiling.singleCoreN = static_cast<uint32_t>(
        ((n + nBlocks - 1) / nBlocks + static_cast<int>(GEMM_BATCHED_BASE_N) - 1) /
        static_cast<int>(GEMM_BATCHED_BASE_N) * static_cast<int>(GEMM_BATCHED_BASE_N));
    tiling.singleCoreM = std::min(tiling.singleCoreM, tiling.m);
    tiling.singleCoreN = std::min(tiling.singleCoreN, tiling.n);

    uint64_t totalTasks = static_cast<uint64_t>(batchSize) * tiling.mBlocks * tiling.nBlocks;
    if (totalTasks > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(
            "aclblasCgetriBatched", "blocked GEMM task count overflow, totalTasks=%llu",
            static_cast<unsigned long long>(totalTasks));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    tiling.totalTasks = static_cast<uint32_t>(totalTasks);
    tiling.usedAicCoreNum = static_cast<uint32_t>(totalTasks);

    std::swap(tiling.m, tiling.n);
    std::swap(tiling.lda, tiling.ldb);
    std::swap(tiling.isTransA, tiling.isTransB);
    std::swap(tiling.singleCoreM, tiling.singleCoreN);
    std::swap(tiling.mBlocks, tiling.nBlocks);
    return ACLBLAS_STATUS_SUCCESS;
}

void LaunchCgetriBlockedSolve(
    _aclblas_handle* handle, const CgetriBlockedWorkspace& workspace, uint8_t* infoArray, int n, int batchSize,
    uint32_t rowStart, uint32_t blockSize, bool isLower, uint32_t aivCoreNum)
{
    CgetriBlockedSolveTilingData tiling{};
    tiling.n = static_cast<uint32_t>(n);
    tiling.batchSize = static_cast<uint32_t>(batchSize);
    tiling.matrixStride = workspace.matrixStride;
    tiling.rowStart = rowStart;
    tiling.blockSize = blockSize;
    tiling.isLower = isLower ? 1u : 0u;
    uint64_t rhsTiles = static_cast<uint64_t>(batchSize) * ((static_cast<uint32_t>(n) + 63) / 64);
    tiling.usedCoreNum = static_cast<uint32_t>(std::min<uint64_t>(aivCoreNum, rhsTiles));
    cgetri_blocked_solve_kernel_do(
        workspace.aReal, workspace.aImag, workspace.cReal, workspace.cImag, infoArray, tiling, tiling.usedCoreNum,
        handle->stream);
}

aclblasStatus_t LaunchCgetriBlockedUpdate(
    _aclblas_handle* handle, const CgetriBlockedWorkspace& workspace, int n, int batchSize, uint32_t aRow,
    uint32_t aCol, uint32_t bRow, uint32_t targetRow, uint32_t updateRows, uint32_t blockSize, uint32_t aivCoreNum,
    uint32_t aicCoreNum, uint8_t* infoArray)
{
    const uint32_t aOffset = aRow + aCol * static_cast<uint32_t>(n);
    const uint32_t bOffset = bRow;
    const uint8_t* aReal = workspace.aReal + static_cast<size_t>(aOffset) * sizeof(float);
    const uint8_t* aImag = workspace.aImag + static_cast<size_t>(aOffset) * sizeof(float);
    const uint8_t* cReal = workspace.cReal + static_cast<size_t>(bOffset) * sizeof(float);
    const uint8_t* cImag = workspace.cImag + static_cast<size_t>(bOffset) * sizeof(float);

    GemmBatchedGemmTilingData gemmTiling{};
    aclblasStatus_t tilingRet = CalcCgetriGemmTiling(
        static_cast<int>(updateRows), n, static_cast<int>(blockSize), batchSize, n, n,
        static_cast<int>(workspace.tempRowStride), aicCoreNum, gemmTiling);
    if (tilingRet != ACLBLAS_STATUS_SUCCESS) {
        return tilingRet;
    }

    CgetriBlockedCombineTilingData combineTiling{};
    combineTiling.n = static_cast<uint32_t>(n);
    combineTiling.updateRows = updateRows;
    combineTiling.batchSize = static_cast<uint32_t>(batchSize);
    combineTiling.matrixStride = workspace.matrixStride;
    combineTiling.tempRowStride = workspace.tempRowStride;
    combineTiling.tempMatrixStride = workspace.tempMatrixStride;
    combineTiling.targetRow = targetRow;
    combineTiling.totalElements = static_cast<uint64_t>(batchSize) * updateRows * static_cast<uint64_t>(n);
    combineTiling.usedCoreNum = CalcCgetriAivCores(aivCoreNum, combineTiling.totalElements);

    auto launchRealGemm = [&](const uint8_t* lhs, const uint8_t* rhs, uint8_t* output) {
        cgetri_strided_gemm_kernel_do(
            gemmTiling.usedAicCoreNum, handle->stream, rhs, lhs, output, infoArray, workspace.matrixStride,
            workspace.matrixStride, workspace.tempMatrixStride, gemmTiling);
    };

    launchRealGemm(aReal, cReal, workspace.temp1);
    launchRealGemm(aImag, cImag, workspace.temp2);
    combineTiling.isImag = 0;
    cgetri_blocked_combine_kernel_do(
        workspace.temp1, workspace.temp2, workspace.cReal, infoArray, combineTiling, combineTiling.usedCoreNum,
        handle->stream);

    launchRealGemm(aReal, cImag, workspace.temp1);
    launchRealGemm(aImag, cReal, workspace.temp2);
    combineTiling.isImag = 1;
    cgetri_blocked_combine_kernel_do(
        workspace.temp1, workspace.temp2, workspace.cImag, infoArray, combineTiling, combineTiling.usedCoreNum,
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchCgetriBlockedFactors(
    _aclblas_handle* internalHandle, const CgetriBlockedWorkspace& workspace, int* infoArray, int n, int batchSize,
    uint32_t aivCoreNum, uint32_t aicCoreNum)
{
    for (uint32_t rowStart = 0; rowStart < static_cast<uint32_t>(n); rowStart += CGETRI_BLOCKED_TILE_N) {
        uint32_t blockSize = std::min<uint32_t>(CGETRI_BLOCKED_TILE_N, static_cast<uint32_t>(n) - rowStart);
        LaunchCgetriBlockedSolve(
            internalHandle, workspace, reinterpret_cast<uint8_t*>(infoArray), n, batchSize, rowStart, blockSize, true,
            aivCoreNum);
        uint32_t rowEnd = rowStart + blockSize;
        if (rowEnd < static_cast<uint32_t>(n)) {
            aclblasStatus_t updateRet = LaunchCgetriBlockedUpdate(
                internalHandle, workspace, n, batchSize, rowEnd, rowStart, rowStart, rowEnd,
                static_cast<uint32_t>(n) - rowEnd, blockSize, aivCoreNum, aicCoreNum,
                reinterpret_cast<uint8_t*>(infoArray));
            if (updateRet != ACLBLAS_STATUS_SUCCESS) {
                return updateRet;
            }
        }
    }

    uint32_t rowEnd = static_cast<uint32_t>(n);
    while (rowEnd > 0) {
        uint32_t rowStart = rowEnd > CGETRI_BLOCKED_TILE_N ? rowEnd - CGETRI_BLOCKED_TILE_N : 0;
        uint32_t blockSize = rowEnd - rowStart;
        LaunchCgetriBlockedSolve(
            internalHandle, workspace, reinterpret_cast<uint8_t*>(infoArray), n, batchSize, rowStart, blockSize, false,
            aivCoreNum);
        if (rowStart > 0) {
            aclblasStatus_t updateRet = LaunchCgetriBlockedUpdate(
                internalHandle, workspace, n, batchSize, 0, rowStart, rowStart, 0, rowStart, blockSize, aivCoreNum,
                aicCoreNum, reinterpret_cast<uint8_t*>(infoArray));
            if (updateRet != ACLBLAS_STATUS_SUCCESS) {
                return updateRet;
            }
        }
        rowEnd = rowStart;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchCgetriBlockedKernel(
    aclblasHandle_t handle, int n, int lda, int ldc, int batchSize, const int* pivotArray,
    const aclblasComplex* const aarray[], aclblasComplex* const carray[], int* infoArray)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    uint32_t aicCoreNum = GetAicCoreCount();
    if (aivCoreNum == 0 || aicCoreNum == 0) {
        OP_LOGE("aclblasCgetriBatched", "invalid core count, AIV=%u AIC=%u", aivCoreNum, aicCoreNum);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    CgetriBlockedWorkspace workspace{};
    aclblasStatus_t workspaceRet = PrepareCgetriBlockedWorkspace(handle, n, batchSize, workspace);
    if (workspaceRet != ACLBLAS_STATUS_SUCCESS) {
        return workspaceRet;
    }

    uint32_t initCores = std::min<uint32_t>(aivCoreNum, static_cast<uint32_t>(batchSize));
    uint32_t batchPerCore = (static_cast<uint32_t>(batchSize) - 1) / initCores + 1;
    uint32_t usedInitCores = (static_cast<uint32_t>(batchSize) - 1) / batchPerCore + 1;
    CgetriBlockedInitTilingData initTiling{};
    initTiling.n = static_cast<uint32_t>(n);
    initTiling.lda = static_cast<uint32_t>(lda);
    initTiling.batchSize = static_cast<uint32_t>(batchSize);
    initTiling.matrixStride = workspace.matrixStride;
    initTiling.usedCoreNum = usedInitCores;
    initTiling.batchPerCore = batchPerCore;
    initTiling.batchTail = static_cast<uint32_t>(batchSize) - (usedInitCores - 1) * batchPerCore;
    initTiling.usePivot = pivotArray == nullptr ? 0u : 1u;
    cgetri_blocked_init_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<const aclblasComplex**>(aarray)),
        reinterpret_cast<uint8_t*>(const_cast<int*>(pivotArray)), reinterpret_cast<uint8_t*>(infoArray),
        workspace.aReal, workspace.aImag, workspace.cReal, workspace.cImag, initTiling, usedInitCores, handle->stream);

    aclblasStatus_t solveRet =
        LaunchCgetriBlockedFactors(handle, workspace, infoArray, n, batchSize, aivCoreNum, aicCoreNum);
    if (solveRet != ACLBLAS_STATUS_SUCCESS) {
        return solveRet;
    }

    CgetriBlockedFinalizeTilingData finalizeTiling{};
    finalizeTiling.n = static_cast<uint32_t>(n);
    finalizeTiling.ldc = static_cast<uint32_t>(ldc);
    finalizeTiling.batchSize = static_cast<uint32_t>(batchSize);
    finalizeTiling.matrixStride = workspace.matrixStride;
    finalizeTiling.totalElements = static_cast<uint64_t>(batchSize) * workspace.matrixStride;
    finalizeTiling.usedCoreNum = CalcCgetriAivCores(aivCoreNum, finalizeTiling.totalElements);
    cgetri_blocked_finalize_kernel_do(
        workspace.cReal, workspace.cImag, reinterpret_cast<uint8_t*>(const_cast<aclblasComplex**>(carray)),
        reinterpret_cast<uint8_t*>(infoArray), finalizeTiling, finalizeTiling.usedCoreNum, handle->stream);

    OP_LOGD(
        "aclblasCgetriBatched", "blocked path: n=%d batch=%d block=%u workspace=%zu", n, batchSize,
        CGETRI_BLOCKED_TILE_N, workspace.requiredBytes);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchCgetriBatchedKernel(
    aclblasHandle_t handle, int n, int lda, int ldc, int batchSize, const int* PivotArray,
    const aclblasComplex* const Aarray[], aclblasComplex* const Carray[], int* infoArray)
{
    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE("aclblasCgetriBatched", "vector core count is 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t uBatchSize = static_cast<uint32_t>(batchSize);
    uint32_t batchPerCore = (uBatchSize - 1) / coreNum + 1;
    uint32_t usedCoreNum = (uBatchSize - 1) / batchPerCore + 1;

    CgetriBatchedTilingData tiling = {};
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.ldc = static_cast<uint32_t>(ldc);
    tiling.usedCoreNum = usedCoreNum;
    tiling.batchPerCore = batchPerCore;
    tiling.batchTail = uBatchSize - (usedCoreNum - 1) * batchPerCore;
    tiling.usePivot = PivotArray == nullptr ? 0u : 1u;

    OP_LOGD(
        "aclblasCgetriBatched", "tiling: n=%u lda=%u ldc=%u usedCoreNum=%u batchPerCore=%u batchTail=%u usePivot=%u",
        tiling.n, tiling.lda, tiling.ldc, tiling.usedCoreNum, tiling.batchPerCore, tiling.batchTail, tiling.usePivot);
    OP_LOGI("aclblasCgetriBatched", "launching kernel: blocks=%u, cores=%u", usedCoreNum, coreNum);

    cgetri_batched_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<const aclblasComplex**>(Aarray)),
        reinterpret_cast<uint8_t*>(const_cast<int*>(PivotArray)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex**>(Carray)), reinterpret_cast<uint8_t*>(infoArray), tiling,
        usedCoreNum, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasCgetriBatched(
    aclblasHandle_t handle, int n, const aclblasComplex* const Aarray[], int lda, const int* PivotArray,
    aclblasComplex* const Carray[], int ldc, int* infoArray, int batchSize)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCgetriBatched", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t validRet = ValidateCgetriBatchedParams(n, lda, ldc, batchSize, Aarray, Carray, infoArray);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }
    if (n == 0 || batchSize == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    OP_LOGI(
        "aclblasCgetriBatched", "n=%d, lda=%d, ldc=%d, batchSize=%d, usePivot=%d", n, lda, ldc, batchSize,
        PivotArray == nullptr ? 0 : 1);
    if (n >= static_cast<int>(CGETRI_BLOCKED_THRESHOLD_N)) {
        return LaunchCgetriBlockedKernel(handle, n, lda, ldc, batchSize, PivotArray, Aarray, Carray, infoArray);
    }
    return LaunchCgetriBatchedKernel(handle, n, lda, ldc, batchSize, PivotArray, Aarray, Carray, infoArray);
}
