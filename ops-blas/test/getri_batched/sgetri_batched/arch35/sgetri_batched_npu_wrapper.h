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

#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

/**
 * NPU wrapper for aclblasSgetriBatched.
 *
 * The API expects device-side pointer arrays and device memory.
 * This wrapper handles:
 *   1. Allocate device memory for each A matrix (input) and copy H2D
 *   2. Allocate device memory for each C matrix (output)
 *   3. Create device pointer arrays for A and C
 *   4. Allocate device memory for PivotArray and infoArray
 *   5. Call aclblasSgetriBatched
 *   6. Sync and copy C results + infoArray D2H
 *   7. Free all device memory
 *
 * If handle == nullptr or n <= 0 or batchSize <= 0, pass through directly.
 * If Aarray/Carray/infoArray == nullptr, pass through for error-path testing.
 */

struct GetriDeviceBuffers {
    std::vector<void*> dAMatrices;
    std::vector<void*> dCMatrices;
    void* dAPtrArray = nullptr;
    void* dCPtrArray = nullptr;
    void* dPivot = nullptr;
    void* dInfo = nullptr;

    ~GetriDeviceBuffers() { Cleanup(); }

    void Cleanup()
    {
        for (auto& dm : dAMatrices) {
            if (dm)
                aclrtFree(dm);
        }
        for (auto& dm : dCMatrices) {
            if (dm)
                aclrtFree(dm);
        }
        if (dAPtrArray)
            aclrtFree(dAPtrArray);
        if (dCPtrArray)
            aclrtFree(dCPtrArray);
        if (dPivot)
            aclrtFree(dPivot);
        if (dInfo)
            aclrtFree(dInfo);
        dAMatrices.clear();
        dCMatrices.clear();
        dAPtrArray = nullptr;
        dCPtrArray = nullptr;
        dPivot = nullptr;
        dInfo = nullptr;
    }
};

static aclblasStatus_t GetriAllocateAndCopyAMatrices(
    GetriDeviceBuffers& bufs, const float* const Aarray[], int batchSize, size_t matBytes)
{
    bufs.dAMatrices.resize(batchSize, nullptr);
    for (int b = 0; b < batchSize; b++) {
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

static aclblasStatus_t GetriAllocateCMatrices(GetriDeviceBuffers& bufs, int batchSize, size_t matBytes)
{
    bufs.dCMatrices.resize(batchSize, nullptr);
    for (int b = 0; b < batchSize; b++) {
        aclError aclRet = aclrtMalloc(&bufs.dCMatrices[b], matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t AllocateAndCopyPtrArray(
    void*& dPtrArray, const std::vector<void*>& dMatrices, int batchSize, size_t ptrArrayBytes,
    GetriDeviceBuffers& bufs)
{
    std::vector<float*> hPtrArray(batchSize);
    for (int b = 0; b < batchSize; b++) {
        hPtrArray[b] = static_cast<float*>(dMatrices[b]);
    }
    aclError aclRet = aclrtMalloc(&dPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(dPtrArray, ptrArrayBytes, hPtrArray.data(), ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t GetriCreateDevicePtrArrays(GetriDeviceBuffers& bufs, int batchSize, size_t ptrArrayBytes)
{
    aclblasStatus_t status = AllocateAndCopyPtrArray(bufs.dAPtrArray, bufs.dAMatrices, batchSize, ptrArrayBytes, bufs);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    return AllocateAndCopyPtrArray(bufs.dCPtrArray, bufs.dCMatrices, batchSize, ptrArrayBytes, bufs);
}

static aclblasStatus_t GetriAllocatePivotAndInfo(
    GetriDeviceBuffers& bufs, const int* PivotArray, size_t pivotBytes, size_t infoBytes)
{
    if (PivotArray != nullptr) {
        aclError aclRet = aclrtMalloc(&bufs.dPivot, pivotBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclRet = aclrtMemcpy(bufs.dPivot, pivotBytes, PivotArray, pivotBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    aclError aclRet = aclrtMalloc(&bufs.dInfo, infoBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    // Initialize info to 0 on device
    std::vector<int> zeroInfo(infoBytes / sizeof(int), 0);
    aclRet = aclrtMemcpy(bufs.dInfo, infoBytes, zeroInfo.data(), infoBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    return ACLBLAS_STATUS_SUCCESS;
}

static void GetriCopyResultsD2H(
    float* const Carray[], int* infoArray, const GetriDeviceBuffers& bufs, int batchSize, size_t cMatBytes,
    size_t infoBytes)
{
    for (int b = 0; b < batchSize; b++) {
        if (Carray[b] != nullptr) {
            aclrtMemcpy(Carray[b], cMatBytes, bufs.dCMatrices[b], cMatBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }
    if (infoArray != nullptr) {
        aclrtMemcpy(infoArray, infoBytes, bufs.dInfo, infoBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
}

/**
 * NPU wrapper entry point.
 *
 * Aarray: host-side array of pointers to LU-factored matrices (input, read-only)
 * Carray: host-side array of pointers to output inverse matrices (output)
 * PivotArray: host-side pivot array from getrf (input, may be nullptr)
 * infoArray: host-side info array (output)
 */
static aclblasStatus_t AllocateAllDeviceBuffers(
    GetriDeviceBuffers& bufs, const float* const Aarray[], const int* PivotArray, int batchSize, size_t aMatBytes,
    size_t cMatBytes, size_t ptrArrayBytes, size_t pivotBytes, size_t infoBytes)
{
    aclblasStatus_t status = GetriAllocateAndCopyAMatrices(bufs, Aarray, batchSize, aMatBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = GetriAllocateCMatrices(bufs, batchSize, cMatBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = GetriCreateDevicePtrArrays(bufs, batchSize, ptrArrayBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    return GetriAllocatePivotAndInfo(bufs, PivotArray, pivotBytes, infoBytes);
}

inline aclblasStatus_t aclblasSgetriBatched_npu(
    aclblasHandle_t handle, int n, const float* const Aarray[], int lda, const int* PivotArray, float* const Carray[],
    int ldc, int* infoArray, int batchSize, aclrtStream stream)
{
    if (handle == nullptr || n <= 0 || batchSize <= 0 || Aarray == nullptr || Carray == nullptr ||
        infoArray == nullptr) {
        return aclblasSgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize);
    }

    const size_t aMatBytes = static_cast<size_t>(lda) * n * sizeof(float);
    const size_t cMatBytes = static_cast<size_t>(ldc) * n * sizeof(float);
    const size_t pivotBytes = (PivotArray != nullptr) ? static_cast<size_t>(n) * batchSize * sizeof(int) : 0;
    const size_t infoBytes = static_cast<size_t>(batchSize) * sizeof(int);
    const size_t ptrArrayBytes = static_cast<size_t>(batchSize) * sizeof(float*);

    GetriDeviceBuffers bufs;
    aclblasStatus_t status = AllocateAllDeviceBuffers(
        bufs, Aarray, PivotArray, batchSize, aMatBytes, cMatBytes, ptrArrayBytes, pivotBytes, infoBytes);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    aclblasStatus_t ret = aclblasSgetriBatched(
        handle, n, reinterpret_cast<const float* const*>(bufs.dAPtrArray), lda, static_cast<const int*>(bufs.dPivot),
        reinterpret_cast<float* const*>(bufs.dCPtrArray), ldc, static_cast<int*>(bufs.dInfo), batchSize);

    aclrtSynchronizeStream(stream);

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        GetriCopyResultsD2H(Carray, infoArray, bufs, batchSize, cMatBytes, infoBytes);
    }

    bufs.Cleanup();
    return ret;
}

