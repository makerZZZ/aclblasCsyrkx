/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_NPU_WRAPPER_H
#define GEMM_NPU_WRAPPER_H

#include <algorithm>
#include <complex>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "gemm_param.h"

inline aclblasStatus_t GemmAllocAndCopy(void** devPtr, const void* hostPtr, size_t bytes)
{
    *devPtr = nullptr;
    if (hostPtr == nullptr || bytes == 0)
        return ACLBLAS_STATUS_SUCCESS;
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

inline void GemmFreeBuffers(void* dA, void* dB, void* dC)
{
    if (dA) aclrtFree(dA);
    if (dB) aclrtFree(dB);
    if (dC) aclrtFree(dC);
}

struct GemmDeviceBuf {
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
};

inline aclblasStatus_t GemmAllocDevice(GemmDeviceBuf& dev, int k,
    const void* A, const void* B, void* C,
    size_t aBytes, size_t bBytes, size_t cBytes)
{
    if (A != nullptr && k > 0) {
        aclblasStatus_t ret = GemmAllocAndCopy(&dev.dA, A, aBytes);
        if (ret != ACLBLAS_STATUS_SUCCESS) { GemmFreeBuffers(dev.dA, dev.dB, dev.dC); return ret; }
    }
    if (B != nullptr && k > 0) {
        aclblasStatus_t ret = GemmAllocAndCopy(&dev.dB, B, bBytes);
        if (ret != ACLBLAS_STATUS_SUCCESS) { GemmFreeBuffers(dev.dA, dev.dB, dev.dC); return ret; }
    }
    if (C != nullptr) {
        aclblasStatus_t ret = GemmAllocAndCopy(&dev.dC, C, cBytes);
        if (ret != ACLBLAS_STATUS_SUCCESS) { GemmFreeBuffers(dev.dA, dev.dB, dev.dC); return ret; }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t GemmSyncAndCopyBack(GemmDeviceBuf& dev, void* C, size_t cBytes, aclblasStatus_t apiRet)
{
    if (apiRet != ACLBLAS_STATUS_SUCCESS) {
        GemmFreeBuffers(dev.dA, dev.dB, dev.dC);
        return apiRet;
    }
    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        GemmFreeBuffers(dev.dA, dev.dB, dev.dC);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (dev.dC && C) {
        aclError aclRet = aclrtMemcpy(C, cBytes, dev.dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            GemmFreeBuffers(dev.dA, dev.dB, dev.dC);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    GemmFreeBuffers(dev.dA, dev.dB, dev.dC);
    return apiRet;
}

inline aclblasStatus_t aclblasSgemm_npu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc)
{
    if (handle == nullptr || m <= 0 || n <= 0) {
        return aclblasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    size_t physColsA = (transA == ACLBLAS_OP_N) ? k : m;
    size_t physColsB = (transB == ACLBLAS_OP_N) ? n : k;
    size_t aBytes = static_cast<size_t>(std::max(1, lda)) * std::max(size_t(1), physColsA) * sizeof(float);
    size_t bBytes = static_cast<size_t>(std::max(1, ldb)) * std::max(size_t(1), physColsB) * sizeof(float);
    size_t cBytes = static_cast<size_t>(std::max(1, ldc)) * std::max(1, n) * sizeof(float);

    GemmDeviceBuf dev;
    aclblasStatus_t st = GemmAllocDevice(dev, k, A, B, C, aBytes, bBytes, cBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    aclblasStatus_t ret = aclblasSgemm(
        handle, transA, transB, m, n, k, alpha,
        dev.dA ? static_cast<const float*>(dev.dA) : A, lda,
        dev.dB ? static_cast<const float*>(dev.dB) : B, ldb,
        beta, dev.dC ? static_cast<float*>(dev.dC) : C, ldc);

    return GemmSyncAndCopyBack(dev, C, cBytes, ret);
}

inline aclblasStatus_t aclblasCgemm_npu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc)
{
    if (handle == nullptr || m <= 0 || n <= 0) {
        return aclblasCgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    const size_t elemSize = sizeof(std::complex<float>);
    size_t physColsA = (transA == ACLBLAS_OP_N) ? k : m;
    size_t physColsB = (transB == ACLBLAS_OP_N) ? n : k;
    size_t aBytes = static_cast<size_t>(std::max(1, lda)) * std::max(size_t(1), physColsA) * elemSize;
    size_t bBytes = static_cast<size_t>(std::max(1, ldb)) * std::max(size_t(1), physColsB) * elemSize;
    size_t cBytes = static_cast<size_t>(std::max(1, ldc)) * std::max(1, n) * elemSize;

    GemmDeviceBuf dev;
    aclblasStatus_t st = GemmAllocDevice(dev, k, A, B, C, aBytes, bBytes, cBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    aclblasStatus_t ret = aclblasCgemm(
        handle, transA, transB, m, n, k, alpha,
        dev.dA ? static_cast<const aclblasComplex*>(dev.dA) : A, lda,
        dev.dB ? static_cast<const aclblasComplex*>(dev.dB) : B, ldb, beta,
        dev.dC ? static_cast<aclblasComplex*>(dev.dC) : C, ldc);

    return GemmSyncAndCopyBack(dev, C, cBytes, ret);
}

#endif // GEMM_NPU_WRAPPER_H
