/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGEMM3M_NPU_WRAPPER_H
#define SGEMM3M_NPU_WRAPPER_H

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "sgemm3m_param.h"

// ── Helper: allocate device memory and copy host→device ──
// On success *devPtr points to allocated device memory.
// On failure *devPtr is set to nullptr and an error status is returned.
// If hostPtr is nullptr or bytes is 0, succeeds immediately with *devPtr = nullptr.
static inline aclblasStatus_t Gemm3mCopyToDevice(void** devPtr, const void* hostPtr, size_t bytes)
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

// ── Helper: copy result from device back to host ──
static inline aclblasStatus_t Gemm3mCopyFromDevice(void* hostPtr, const void* devPtr, size_t bytes)
{
    if (hostPtr == nullptr || devPtr == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMemcpy(hostPtr, bytes, devPtr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return (ret == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

// ── Helper: free up to 3 device buffers (A, B, C) ──
static inline void Gemm3mFreeAll(void* dA, void* dB, void* dC)
{
    if (dA) aclrtFree(dA);
    if (dB) aclrtFree(dB);
    if (dC) aclrtFree(dC);
}

// ═══════════════════════════════════════════════════════════════════════════════
// NPU wrapper for aclblasSgemm3m
//
// Handles device memory allocation, H2D/D2H transfers, and kernel invocation.
// nullptr inputs are passed through without device allocation (for error-path testing).
// Each ACL call (malloc, memcpy, sync, free) is checked for errors.
// On any ACL failure, all allocated device buffers are freed before returning.
// ═══════════════════════════════════════════════════════════════════════════════

inline aclblasStatus_t aclblasSgemm3m_npu(
    aclblasHandle handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta, float* C, int ldc)
{
    // Fast path: handle==nullptr or m<=0 or n<=0 → pass through directly
    // (operator handles validation and short-circuit return)
    if (handle == nullptr || m <= 0 || n <= 0) {
        return aclblasSgemm3m(
            handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    // Compute device buffer sizes (column-major storage)
    const int physColsA = gemm3mPhysColsA(m, k, transA);
    const int physColsB = gemm3mPhysColsB(k, n, transB);
    const int allocLda = std::max(1, lda);
    const int allocLdb = std::max(1, ldb);
    const int allocLdc = std::max(1, ldc);
    const size_t aBytes = static_cast<size_t>(allocLda) * std::max(1, physColsA) * sizeof(float);
    const size_t bBytes = static_cast<size_t>(allocLdb) * std::max(1, physColsB) * sizeof(float);
    const size_t cBytes = static_cast<size_t>(allocLdc) * std::max(1, n) * sizeof(float);

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;

    // Allocate and copy A (skip when nullptr — e.g. alpha=0 short-circuit)
    aclblasStatus_t allocRet;
    if (A != nullptr) {
        allocRet = Gemm3mCopyToDevice(&dA, A, aBytes);
        if (allocRet != ACLBLAS_STATUS_SUCCESS) { Gemm3mFreeAll(dA, dB, dC); return allocRet; }
    }
    if (B != nullptr) {
        allocRet = Gemm3mCopyToDevice(&dB, B, bBytes);
        if (allocRet != ACLBLAS_STATUS_SUCCESS) { Gemm3mFreeAll(dA, dB, dC); return allocRet; }
    }

    // Allocate and copy C (input/output — needs H2D for beta accumulation)
    if (C != nullptr) {
        allocRet = Gemm3mCopyToDevice(&dC, C, cBytes);
        if (allocRet != ACLBLAS_STATUS_SUCCESS) { Gemm3mFreeAll(dA, dB, dC); return allocRet; }
    }

    // Execute kernel — pass device pointers (nullptr passthrough for alpha=0 / error paths)
    aclblasStatus_t ret = aclblasSgemm3m(
        handle, transA, transB, m, n, k, alpha,
        dA ? static_cast<const float*>(dA) : A, lda,
        dB ? static_cast<const float*>(dB) : B, ldb,
        beta, dC ? static_cast<float*>(dC) : C, ldc);

    // Synchronize device (must check return value)
    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        Gemm3mFreeAll(dA, dB, dC);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    // Copy result back from device to host (must check return value)
    if (ret == ACLBLAS_STATUS_SUCCESS && C != nullptr && dC != nullptr) {
        aclblasStatus_t copyRet = Gemm3mCopyFromDevice(C, dC, cBytes);
        if (copyRet != ACLBLAS_STATUS_SUCCESS) {
            Gemm3mFreeAll(dA, dB, dC);
            return copyRet;
        }
    }

    Gemm3mFreeAll(dA, dB, dC);
    return ret;
}

#endif // SGEMM3M_NPU_WRAPPER_H
