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
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclError CopyToDevice(void** devPtr, const void* hostPtr, size_t bytes)
{
    aclError ret = aclrtMalloc(devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ret;
    aclError copyRet = aclrtMemcpy(*devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (copyRet != ACL_SUCCESS) {
        aclrtFree(*devPtr);
        *devPtr = nullptr;
        return copyRet;
    }
    return ACL_SUCCESS;
}

inline void CleanupDeviceBuffers(void* dA, void* dX, void* dY)
{
    if (dA)
        aclrtFree(dA);
    if (dX)
        aclrtFree(dX);
    if (dY)
        aclrtFree(dY);
}

// ============================================================
// Template implementation for StridedBatched GEMV NPU wrapper
// ============================================================
template <typename T_IN, typename T_OUT, typename ApiFn>
inline aclblasStatus_t GemvStridedBatchedNpuImpl(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const T_IN* A, int lda, int64_t strideA, const T_IN* x, int incx, int64_t stridex,
    const float* beta, T_OUT* y, int incy, int64_t stridey, int batchCount, ApiFn apiFn)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;

    // m == 0 / n == 0 / batchCount == 0 are no-op success paths; let the op decide.
    if (m <= 0 || n <= 0 || batchCount <= 0) {
        return apiFn(handle, trans, m, n, alpha,
                     static_cast<const T_IN*>(nullptr), lda, strideA,
                     static_cast<const T_IN*>(nullptr), incx, stridex,
                     beta, static_cast<T_OUT*>(nullptr), incy, stridey, batchCount);
    }

    if (A == nullptr || x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (strideA <= 0 || stridex <= 0 || stridey <= 0)
        return ACLBLAS_STATUS_INVALID_VALUE;

    const size_t aBytes = static_cast<size_t>(batchCount) * static_cast<size_t>(strideA) * sizeof(T_IN);
    const size_t xBytes = static_cast<size_t>(batchCount) * static_cast<size_t>(stridex) * sizeof(T_IN);
    const size_t yBytes = static_cast<size_t>(batchCount) * static_cast<size_t>(stridey) * sizeof(T_OUT);

    void *dA = nullptr, *dX = nullptr, *dY = nullptr;
    if (CopyToDevice(&dA, A, aBytes) != ACL_SUCCESS) {
        CleanupDeviceBuffers(dA, dX, dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (CopyToDevice(&dX, x, xBytes) != ACL_SUCCESS) {
        CleanupDeviceBuffers(dA, dX, dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (CopyToDevice(&dY, y, yBytes) != ACL_SUCCESS) {
        CleanupDeviceBuffers(dA, dX, dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclblasStatus_t ret = apiFn(
        handle, trans, m, n, alpha,
        static_cast<const T_IN*>(dA), lda, strideA,
        static_cast<const T_IN*>(dX), incx, stridex,
        beta, static_cast<T_OUT*>(dY), incy, stridey, batchCount);

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        CleanupDeviceBuffers(dA, dX, dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        aclError copyRet = aclrtMemcpy(y, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (copyRet != ACL_SUCCESS) {
            CleanupDeviceBuffers(dA, dX, dY);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    CleanupDeviceBuffers(dA, dX, dY);
    return ret;
}

// ---- dtype 1 (S): float in, float out ----
inline aclblasStatus_t aclblasSgemvStridedBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const float* A, int lda, int64_t strideA, const float* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount)
{
    return GemvStridedBatchedNpuImpl<float, float>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex,
        beta, y, incy, stridey, batchCount, aclblasSgemvStridedBatched);
}

// ---- dtype 0 (HSH): uint16_t in, uint16_t out ----
inline aclblasStatus_t aclblasHSHgemvStridedBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount)
{
    return GemvStridedBatchedNpuImpl<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex,
        beta, y, incy, stridey, batchCount, aclblasHSHgemvStridedBatched);
}

// ---- dtype 2 (HSS): uint16_t in, float out ----
inline aclblasStatus_t aclblasHSSgemvStridedBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount)
{
    return GemvStridedBatchedNpuImpl<uint16_t, float>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex,
        beta, y, incy, stridey, batchCount, aclblasHSSgemvStridedBatched);
}

// ---- dtype 3 (TST): uint16_t in, uint16_t out (bf16) ----
inline aclblasStatus_t aclblasTSTgemvStridedBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, uint16_t* y, int incy, int64_t stridey, int batchCount)
{
    return GemvStridedBatchedNpuImpl<uint16_t, uint16_t>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex,
        beta, y, incy, stridey, batchCount, aclblasTSTgemvStridedBatched);
}

// ---- dtype 4 (TSS): uint16_t in, float out (bf16) ----
inline aclblasStatus_t aclblasTSSgemvStridedBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha,
    const uint16_t* A, int lda, int64_t strideA, const uint16_t* x, int incx, int64_t stridex,
    const float* beta, float* y, int incy, int64_t stridey, int batchCount)
{
    return GemvStridedBatchedNpuImpl<uint16_t, float>(
        handle, trans, m, n, alpha, A, lda, strideA, x, incx, stridex,
        beta, y, incy, stridey, batchCount, aclblasTSSgemvStridedBatched);
}
