/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_STRIDED_BATCHED_NPU_WRAPPER_H
#define GEMM_STRIDED_BATCHED_NPU_WRAPPER_H

#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// ── Helper: allocate device memory + H2D copy. ──
// On success *devPtr holds the device buffer; on failure *devPtr = nullptr with error status.
// hostPtr == nullptr or bytes == 0 → succeeds immediately with *devPtr = nullptr (pass-through).
inline aclblasStatus_t GsbAllocAndCopy(void** devPtr, const void* hostPtr, size_t bytes)
{
    *devPtr = nullptr;
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMalloc(devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        *devPtr = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMemcpy(*devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*devPtr);
        *devPtr = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline void GsbCleanup(void* dA, void* dB, void* dC)
{
    if (dA)
        aclrtFree(dA);
    if (dB)
        aclrtFree(dB);
    if (dC)
        aclrtFree(dC);
}

inline aclblasStatus_t GsbAllocAll(
    void** dA, void** dB, void** dC, const void* A, const void* B, const void* C, int k,
    size_t aBytes, size_t bBytes, size_t cBytes)
{
    *dA = nullptr;
    *dB = nullptr;
    *dC = nullptr;
    if (A != nullptr && k > 0) {
        aclblasStatus_t r = GsbAllocAndCopy(dA, A, aBytes);
        if (r != ACLBLAS_STATUS_SUCCESS)
            return r;
    }
    if (B != nullptr && k > 0) {
        aclblasStatus_t r = GsbAllocAndCopy(dB, B, bBytes);
        if (r != ACLBLAS_STATUS_SUCCESS)
            return r;
    }
    if (C != nullptr)
        return GsbAllocAndCopy(dC, C, cBytes);
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t GsbInvokeAndSync(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, void* dA, const float* A, int lda, int64_t strideA, void* dB, const float* B, int ldb,
    int64_t strideB, const float* beta, void* dC, float* C, int ldc, int64_t strideC, int batchCount,
    size_t cBytes)
{
    aclblasStatus_t ret = aclblasSgemmStridedBatched(
        handle, transA, transB, m, n, k, alpha, dA ? static_cast<const float*>(dA) : A, lda, strideA,
        dB ? static_cast<const float*>(dB) : B, ldb, strideB, beta, dC ? static_cast<float*>(dC) : C, ldc, strideC,
        batchCount);
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return ret;

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_EXECUTION_FAILED;

    if (C != nullptr && dC != nullptr) {
        aclError d2h = aclrtMemcpy(C, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (d2h != ACL_SUCCESS)
            return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ret;
}

// ═══════════════════════════════════════════════════════════════════════════════
// NPU wrapper for aclblasSgemmStridedBatched.
//   Test side prepares host std::vector<float> spanning ALL batches (already laid
//   out with the given strides). This wrapper H2D-copies the whole span, invokes the
//   operator on device pointers, synchronizes, and D2H-copies C back.
//   NULLPTR inputs pass through untouched (error-path testing).
//   aBytes/bBytes/cBytes are the total span sizes (in bytes) of each host buffer.
//   Every ACL call is checked; device memory is freed on any failure.
// ═══════════════════════════════════════════════════════════════════════════════
inline aclblasStatus_t aclblasSgemmStridedBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda, int64_t strideA, const float* B, int ldb, int64_t strideB,
    const float* beta, float* C, int ldc, int64_t strideC, int batchCount, size_t aBytes, size_t bBytes,
    size_t cBytes)
{
    if (handle == nullptr || m <= 0 || n <= 0 || batchCount <= 0) {
        return aclblasSgemmStridedBatched(
            handle, transA, transB, m, n, k, alpha, A, lda, strideA, B, ldb, strideB,
            beta, C, ldc, strideC, batchCount);
    }

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;

    aclblasStatus_t allocRet = GsbAllocAll(&dA, &dB, &dC, A, B, C, k, aBytes, bBytes, cBytes);
    if (allocRet != ACLBLAS_STATUS_SUCCESS) {
        GsbCleanup(dA, dB, dC);
        return allocRet;
    }

    aclblasStatus_t ret = GsbInvokeAndSync(
        handle, transA, transB, m, n, k, alpha, dA, A, lda, strideA, dB, B, ldb, strideB,
        beta, dC, C, ldc, strideC, batchCount, cBytes);
    GsbCleanup(dA, dB, dC);
    return ret;
}

#endif // GEMM_STRIDED_BATCHED_NPU_WRAPPER_H
