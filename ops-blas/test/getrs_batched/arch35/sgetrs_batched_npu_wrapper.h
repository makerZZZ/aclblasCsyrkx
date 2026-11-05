/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SGETRS_BATCHED_NPU_H
#define SGETRS_BATCHED_NPU_H

#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "aclblas_handle_internal.h"

/**
 * NPU wrapper for aclblasSgetrsBatched.
 *
 * The API expects device-side pointer arrays and device memory.
 * This wrapper handles:
 *   1. Allocate device memory for each A matrix (input, LU-factored) and copy H2D
 *   2. Allocate device memory for each B matrix (input/output) and copy H2D
 *   3. Create device pointer arrays for A and B
 *   4. Allocate device memory for devIpiv and copy H2D (if not nullptr)
 *   5. Call aclblasSgetrsBatched (info is host-side, passed directly)
 *   6. Sync and copy B results D2H
 *   7. Free all device memory
 *
 * If handle == nullptr or n <= 0 or nrhs <= 0 or batchCount <= 0, pass through directly.
 * If Aarray/Barray == nullptr, pass through for error-path testing.
 */

struct GetrsDeviceBuffers {
    std::vector<void*> dAMatrices;
    std::vector<void*> dBMatrices;
    void* dAPtrArray = nullptr;
    void* dBPtrArray = nullptr;
    void* dIpiv = nullptr;

    ~GetrsDeviceBuffers() { Cleanup(); }

    void Cleanup()
    {
        for (auto& dm : dAMatrices) {
            if (dm)
                aclrtFree(dm);
        }
        for (auto& dm : dBMatrices) {
            if (dm)
                aclrtFree(dm);
        }
        if (dAPtrArray)
            aclrtFree(dAPtrArray);
        if (dBPtrArray)
            aclrtFree(dBPtrArray);
        if (dIpiv)
            aclrtFree(dIpiv);
        dAMatrices.clear();
        dBMatrices.clear();
        dAPtrArray = nullptr;
        dBPtrArray = nullptr;
        dIpiv = nullptr;
    }
};

static aclblasStatus_t GetrsAllocateAndCopyAMatrices(
    GetrsDeviceBuffers& bufs, const float* const Aarray[], int batchCount, size_t matBytes)
{
    bufs.dAMatrices.resize(batchCount, nullptr);
    for (int b = 0; b < batchCount; b++) {
        aclError aclRet = aclrtMalloc(&bufs.dAMatrices[b], matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        if (Aarray[b] != nullptr) {
            aclRet = aclrtMemcpy(bufs.dAMatrices[b], matBytes, Aarray[b], matBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (aclRet != ACL_SUCCESS) {
                bufs.Cleanup();
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t GetrsAllocateAndCopyBMatrices(
    GetrsDeviceBuffers& bufs, float* const Barray[], int batchCount, size_t matBytes)
{
    bufs.dBMatrices.resize(batchCount, nullptr);
    for (int b = 0; b < batchCount; b++) {
        aclError aclRet = aclrtMalloc(&bufs.dBMatrices[b], matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        if (Barray[b] != nullptr) {
            aclRet = aclrtMemcpy(bufs.dBMatrices[b], matBytes, Barray[b], matBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (aclRet != ACL_SUCCESS) {
                bufs.Cleanup();
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t GetrsCreateDevicePtrArrays(
    GetrsDeviceBuffers& bufs, int batchCount, size_t ptrArrayBytes)
{
    // A pointer array
    std::vector<float*> hAPtrArray(batchCount);
    for (int b = 0; b < batchCount; b++) {
        hAPtrArray[b] = static_cast<float*>(bufs.dAMatrices[b]);
    }
    aclError aclRet = aclrtMalloc(&bufs.dAPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(bufs.dAPtrArray, ptrArrayBytes, hAPtrArray.data(), ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // B pointer array
    std::vector<float*> hBPtrArray(batchCount);
    for (int b = 0; b < batchCount; b++) {
        hBPtrArray[b] = static_cast<float*>(bufs.dBMatrices[b]);
    }
    aclRet = aclrtMalloc(&bufs.dBPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(bufs.dBPtrArray, ptrArrayBytes, hBPtrArray.data(), ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t GetrsAllocateIpiv(
    GetrsDeviceBuffers& bufs, const int* devIpiv, size_t ipivBytes)
{
    if (devIpiv != nullptr) {
        aclError aclRet = aclrtMalloc(&bufs.dIpiv, ipivBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclRet = aclrtMemcpy(bufs.dIpiv, ipivBytes, devIpiv, ipivBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t GetrsCopyResultsD2H(
    float* const Barray[], const GetrsDeviceBuffers& bufs, int batchCount, size_t bMatBytes)
{
    for (int b = 0; b < batchCount; b++) {
        if (Barray[b] != nullptr) {
            aclError aclRet = aclrtMemcpy(Barray[b], bMatBytes, bufs.dBMatrices[b], bMatBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (aclRet != ACL_SUCCESS) {
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

/**
 * NPU wrapper entry point.
 *
 * Aarray: host-side array of pointers to LU-factored matrices (input, read-only)
 * Barray: host-side array of pointers to right-hand side / solution matrices (input/output)
 * devIpiv: host-side pivot array from getrf (input, may be nullptr)
 * info: host-side info pointer (output, may be nullptr)
 */
inline aclblasStatus_t aclblasSgetrsBatched_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int n, int nrhs,
    const float* const Aarray[], int lda, const int* devIpiv,
    float* const Barray[], int ldb, int* info, int batchCount)
{
    if (handle == nullptr || n <= 0 || nrhs <= 0 || batchCount <= 0 ||
        Aarray == nullptr || Barray == nullptr) {
        return aclblasSgetrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchCount);
    }

    const size_t aMatBytes = static_cast<size_t>(lda) * n * sizeof(float);
    const size_t bMatBytes = static_cast<size_t>(ldb) * nrhs * sizeof(float);
    const size_t ipivBytes = (devIpiv != nullptr) ? static_cast<size_t>(n) * batchCount * sizeof(int) : 0;
    const size_t ptrArrayBytes = static_cast<size_t>(batchCount) * sizeof(float*);

    GetrsDeviceBuffers bufs;

    // Allocate and copy A matrices
    aclblasStatus_t status = GetrsAllocateAndCopyAMatrices(bufs, Aarray, batchCount, aMatBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    // Allocate and copy B matrices
    status = GetrsAllocateAndCopyBMatrices(bufs, Barray, batchCount, bMatBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    // Create device pointer arrays
    status = GetrsCreateDevicePtrArrays(bufs, batchCount, ptrArrayBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    // Allocate and copy devIpiv
    status = GetrsAllocateIpiv(bufs, devIpiv, ipivBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    // Call the actual API
    aclblasStatus_t ret = aclblasSgetrsBatched(
        handle, trans, n, nrhs,
        reinterpret_cast<const float* const*>(bufs.dAPtrArray), lda,
        static_cast<const int*>(bufs.dIpiv),
        reinterpret_cast<float* const*>(bufs.dBPtrArray), ldb,
        info, batchCount);

    // Sync
    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    // Copy results back
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t copyRet = GetrsCopyResultsD2H(Barray, bufs, batchCount, bMatBytes);
        if (copyRet != ACLBLAS_STATUS_SUCCESS) {
            bufs.Cleanup();
            return copyRet;
        }
    }

    bufs.Cleanup();
    return ret;
}

#endif // SGETRS_BATCHED_NPU_H
