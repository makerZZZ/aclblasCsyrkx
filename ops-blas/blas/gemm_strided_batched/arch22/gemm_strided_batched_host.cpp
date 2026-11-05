/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>

#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "gemm_strided_batched_tiling.h"

namespace {

uint8_t* AsGm(const float* pointer) { return reinterpret_cast<uint8_t*>(const_cast<float*>(pointer)); }

aclblasStatus_t CheckShape(aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n,
    int k, int64_t strideA, int64_t strideB, int64_t strideC, int batchCount, bool& ta, bool& tb)
{
    if (!handle) {
        OP_LOGE("aclblasSgemmStridedBatched", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    const auto validOp = [](aclblasOperation_t op) {
        return op == ACLBLAS_OP_N || op == ACLBLAS_OP_T || op == ACLBLAS_OP_C;
    };
    if (!validOp(transA) || !validOp(transB)) {
        OP_LOGE("aclblasSgemmStridedBatched", "invalid transA=%d or transB=%d", static_cast<int>(transA),
            static_cast<int>(transB));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0 || k < 0 || batchCount < 0 || strideA < 0 || strideB < 0 || strideC < 0) {
        OP_LOGE("aclblasSgemmStridedBatched",
            "invalid shape or stride: m=%d, n=%d, k=%d, batchCount=%d, strideA=%ld, strideB=%ld, strideC=%ld", m,
            n, k, batchCount, strideA, strideB, strideC);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    ta = transA != ACLBLAS_OP_N;
    tb = transB != ACLBLAS_OP_N;
    return ACLBLAS_STATUS_SUCCESS;
}

// Sets done=true when the call is a legitimate no-op that must return SUCCESS without launching.
aclblasStatus_t CheckLayout(const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta,
    float* C, int ldc, bool ta, bool tb, int m, int n, int k, int batchCount, bool& done)
{
    if (lda < std::max(1, ta ? k : m) || ldb < std::max(1, tb ? n : k) || ldc < std::max(1, m)) {
        OP_LOGE("aclblasSgemmStridedBatched",
            "invalid leading dimension: lda=%d (min %d), ldb=%d (min %d), ldc=%d (min %d)", lda,
            std::max(1, ta ? k : m), ldb, std::max(1, tb ? n : k), ldc, std::max(1, m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!alpha || !beta) {
        OP_LOGE("aclblasSgemmStridedBatched", "alpha or beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    done = m == 0 || n == 0 || batchCount == 0;
    if (done)
        return ACLBLAS_STATUS_SUCCESS;
    // beta=0 suppresses reading C; an output allocation is still required.
    if (!C) {
        OP_LOGE("aclblasSgemmStridedBatched", "C must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (k > 0 && (!A || !B)) {
        OP_LOGE("aclblasSgemmStridedBatched", "A and B must not be nullptr when k=%d", k);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    done = (k == 0 || *alpha == 0.0f) && *beta == 1.0f;
    return ACLBLAS_STATUS_SUCCESS;
}

bool IsTinyCase(const GemmSb22Tiling& t)
{
    return t.alpha == 1.0f && t.beta == 0.0f && !t.transA && !t.transB && t.m == t.n && t.n == t.k && t.m == 8;
}

bool IsBatchCubeCase(const GemmSb22Tiling& t)
{
    return t.alpha == 1.0f && t.beta == 0.0f && !t.transA && !t.transB && t.m == t.n && t.n == t.k &&
           (t.m == 16 || t.m == 32) && t.batches >= 128;
}

bool HasUnitLayout(const GemmSb22Tiling& t)
{
    const int64_t plane = static_cast<int64_t>(t.m) * t.m;
    return t.lda == t.m && t.ldb == t.m && t.ldc == t.m && t.strideA == plane && t.strideB == plane &&
           t.strideC == plane;
}

void RunVectorPath(uint32_t vectors, aclblasHandle_t handle, const GemmSb22Tiling& t, const float* A, const float* B,
    float* C)
{
    GemmSb22Vector(vectors, handle->stream, AsGm(A), AsGm(B), AsGm(C), nullptr, t, false);
}

void RunTinyPath(uint32_t vectors, aclblasHandle_t handle, GemmSb22Tiling t, const float* A, const float* B, float* C,
    int batchCount)
{
    // Larger groups reduce repeated gather setup for high batch counts.
    if (batchCount >= 512)
        t.tinyBatchGroup = 32;
    const uint32_t group = batchCount <= 64 ? 2 : t.tinyBatchGroup;
    const uint32_t blocks = std::min(vectors, (static_cast<uint32_t>(batchCount) + group - 1) / group);
    GemmSb22Tiny(blocks, handle->stream, AsGm(A), AsGm(B), AsGm(C), t);
}

void RunBatchCubePath(uint32_t cubes, aclblasHandle_t handle, const GemmSb22Tiling& t, const float* A, const float* B,
    float* C)
{
    GemmSb22BatchCube(cubes, handle->stream, AsGm(A), AsGm(B), AsGm(C), t);
}

void RunCubePath(uint32_t cubes, aclblasHandle_t handle, const GemmSb22Tiling& t, const float* A, const float* B,
    float* C)
{
    GemmSb22Cube(cubes, handle->stream, AsGm(A), AsGm(B), AsGm(C), t);
}

// Groups whole C planes into the workspace to avoid two host launches per matrix.
void RunStridedBatchGroups(uint32_t cubes, uint32_t vectors, aclblasHandle_t handle, GemmSb22Tiling t, const float* A,
    const float* B, float* C, int batchCount)
{
    auto* temp = static_cast<float*>(GetEffectiveWorkspace(handle));
    const uint64_t planeElements = static_cast<uint64_t>(t.tempLd) * t.n;
    const size_t batchCapacity = GetEffectiveWorkspaceSize(handle) / sizeof(float) / planeElements;
    const int batchStep = static_cast<int>(std::min<size_t>(batchCount, batchCapacity));
    for (int batch = 0; batch < batchCount; batch += t.batches) {
        t.batches = std::min(batchStep, batchCount - batch);
        const float* a = A + static_cast<int64_t>(batch) * t.strideA;
        const float* b = B + static_cast<int64_t>(batch) * t.strideB;
        float* c = C + static_cast<int64_t>(batch) * t.strideC;
        auto cubeTiling = t;
        cubeTiling.strideC = static_cast<int64_t>(planeElements);
        GemmSb22Cube(cubes, handle->stream, AsGm(a), AsGm(b), AsGm(temp), cubeTiling);
        GemmSb22Vector(vectors, handle->stream, AsGm(a), AsGm(b), AsGm(c), AsGm(temp), t, true);
    }
}

// Falls back to per-matrix column strips when even one padded C plane exceeds the workspace.
void RunColumnStrips(uint32_t cubes, uint32_t vectors, aclblasHandle_t handle, const GemmSb22Tiling& t,
    const float* A, const float* B, float* C, int batchCount, int strip)
{
    auto* temp = static_cast<float*>(GetEffectiveWorkspace(handle));
    for (int batch = 0; batch < batchCount; ++batch) {
        for (int col = 0; col < t.n; col += strip) {
            auto slice = t;
            slice.batches = 1;
            slice.n = std::min(strip, t.n - col);
            const float* a = A + static_cast<int64_t>(batch) * t.strideA;
            const float* b = B + static_cast<int64_t>(batch) * t.strideB +
                             (t.transB ? col : static_cast<int64_t>(col) * t.ldb);
            float* c = C + static_cast<int64_t>(batch) * t.strideC + static_cast<int64_t>(col) * t.ldc;
            GemmSb22Cube(cubes, handle->stream, AsGm(a), AsGm(b), AsGm(temp), slice);
            GemmSb22Vector(vectors, handle->stream, AsGm(a), AsGm(b), AsGm(c), AsGm(temp), slice, true);
        }
    }
}

// Reuses the handle workspace in stream order, bounded to one column strip.
aclblasStatus_t RunWorkspacePath(
    uint32_t cubes, uint32_t vectors, aclblasHandle_t handle, const GemmSb22Tiling& t, const float* A, const float* B,
    float* C, int batchCount)
{
    const int64_t paddedM = (static_cast<int64_t>(t.m) + 7) / 8 * 8;
    if (paddedM > INT32_MAX) {
        OP_LOGE("aclblasSgemmStridedBatched", "padded m exceeds INT32_MAX: paddedM=%ld", paddedM);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const size_t columnBytes = static_cast<size_t>(paddedM) * sizeof(float);
    if (!GetEffectiveWorkspace(handle) || GetEffectiveWorkspaceSize(handle) < columnBytes) {
        OP_LOGE("aclblasSgemmStridedBatched", "workspace is unavailable or too small: need=%zu, have=%zu",
            columnBytes, GetEffectiveWorkspaceSize(handle));
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const int strip = static_cast<int>(std::min<size_t>(t.n, GetEffectiveWorkspaceSize(handle) / columnBytes));
    GemmSb22Tiling padded = t;
    padded.tempLd = static_cast<int>(paddedM);
    if (GetEffectiveWorkspaceSize(handle) / sizeof(float) >= static_cast<uint64_t>(paddedM) * t.n) {
        RunStridedBatchGroups(cubes, vectors, handle, padded, A, B, C, batchCount);
    } else {
        RunColumnStrips(cubes, vectors, handle, padded, A, B, C, batchCount, strip);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasSgemmStridedBatched(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda, int64_t strideA, const float* B, int ldb, int64_t strideB,
    const float* beta, float* C, int ldc, int64_t strideC, int batchCount)
{
    bool ta = false, tb = false;
    const aclblasStatus_t shape = CheckShape(handle, transA, transB, m, n, k, strideA, strideB, strideC, batchCount, ta, tb);
    if (shape != ACLBLAS_STATUS_SUCCESS)
        return shape;
    bool done = false;
    const aclblasStatus_t layout = CheckLayout(alpha, A, lda, B, ldb, beta, C, ldc, ta, tb, m, n, k, batchCount, done);
    if (layout != ACLBLAS_STATUS_SUCCESS || done)
        return layout;
    GemmSb22Tiling t{m, n, k, lda, ldb, ldc, ta, tb, batchCount, ldc, strideA, strideB, strideC, *alpha, *beta};
    const uint32_t vectors = GetAivCoreCount();
    if (vectors == 0) {
        OP_LOGE("aclblasSgemmStridedBatched", "GetAivCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (k == 0 || *alpha == 0.0f || (batchCount < 32 && static_cast<int64_t>(m) * n <= 4096 / std::max(1, k))) {
        RunVectorPath(vectors, handle, t, A, B, C);
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint32_t cubes = GetAicCoreCount();
    if (cubes == 0) {
        OP_LOGE("aclblasSgemmStridedBatched", "GetAicCoreCount returned 0");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (IsTinyCase(t) && HasUnitLayout(t)) {
        RunTinyPath(vectors, handle, t, A, B, C, batchCount);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (IsBatchCubeCase(t) && HasUnitLayout(t)) {
        RunBatchCubePath(cubes, handle, t, A, B, C);
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (t.alpha == 1.0f && t.beta == 0.0f) {
        RunCubePath(cubes, handle, t, A, B, C);
        return ACLBLAS_STATUS_SUCCESS;
    }
    return RunWorkspacePath(cubes, vectors, handle, t, A, B, C, batchCount);
}
