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
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "aclblas_handle_internal.h"

/**
 * NPU wrapper for aclblasSmatinvBatched.
 *
 * The API expects Device-side pointer arrays and Device memory for matrices.
 * This wrapper accepts HOST-side pointer arrays containing HOST matrix data, and manages:
 *   1. Allocate Device memory for each A matrix (input) and copy H2D
 *   2. Allocate Device memory for each Ainv matrix (output)
 *   3. Create Device pointer arrays for A and Ainv
 *   4. Allocate Device memory for info array
 *   5. Call aclblasSmatinvBatched
 *   6. Sync stream and copy Ainv + info D2H
 *   7. Free all Device memory
 *
 * If handle == nullptr or n <= 0 / n > 32 / batchSize < 0 / lda < max(1,n) / lda_inv < max(1,n),
 * pass through directly to the host-side API (which returns the correct error code).
 * If A == nullptr / Ainv == nullptr / info == nullptr (n > 0, batchSize > 0), pass through.
 */

struct MatinvDeviceBuffers {
    std::vector<void*> dAMatrices;
    std::vector<void*> dAinvMatrices;
    void* dAPtrArray = nullptr;
    void* dAinvPtrArray = nullptr;
    void* dInfo = nullptr;

    ~MatinvDeviceBuffers() { Cleanup(); }

    void Cleanup()
    {
        for (auto& dm : dAMatrices) {
            if (dm)
                aclrtFree(dm);
        }
        for (auto& dm : dAinvMatrices) {
            if (dm)
                aclrtFree(dm);
        }
        if (dAPtrArray)
            aclrtFree(dAPtrArray);
        if (dAinvPtrArray)
            aclrtFree(dAinvPtrArray);
        if (dInfo)
            aclrtFree(dInfo);
        dAMatrices.clear();
        dAinvMatrices.clear();
        dAPtrArray = nullptr;
        dAinvPtrArray = nullptr;
        dInfo = nullptr;
    }
};

inline aclblasStatus_t MatinvAllocAndCopyAMatrices(
    MatinvDeviceBuffers& bufs, const float* const A[], int batchSize, size_t matBytes)
{
    bufs.dAMatrices.resize(batchSize, nullptr);
    for (int b = 0; b < batchSize; b++) {
        aclError aclRet = aclrtMalloc(&bufs.dAMatrices[b], matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        if (A[b] != nullptr) {
            aclRet = aclrtMemcpy(bufs.dAMatrices[b], matBytes, A[b], matBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (aclRet != ACL_SUCCESS) {
                bufs.Cleanup();
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t MatinvAllocAinvMatrices(MatinvDeviceBuffers& bufs, int batchSize, size_t invBytes)
{
    bufs.dAinvMatrices.resize(batchSize, nullptr);
    for (int b = 0; b < batchSize; b++) {
        aclError aclRet = aclrtMalloc(&bufs.dAinvMatrices[b], invBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t MatinvCreatePtrArraysAndInfo(
    MatinvDeviceBuffers& bufs, int batchSize, size_t ptrArrayBytes, size_t infoBytes)
{
    std::vector<float*> hAPtrs(batchSize);
    std::vector<float*> hAinvPtrs(batchSize);
    for (int b = 0; b < batchSize; b++) {
        hAPtrs[b] = static_cast<float*>(bufs.dAMatrices[b]);
        hAinvPtrs[b] = static_cast<float*>(bufs.dAinvMatrices[b]);
    }

    aclError aclRet = aclrtMalloc(&bufs.dAPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(bufs.dAPtrArray, ptrArrayBytes, hAPtrs.data(), ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMalloc(&bufs.dAinvPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(
        bufs.dAinvPtrArray, ptrArrayBytes, hAinvPtrs.data(), ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMalloc(&bufs.dInfo, infoBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    std::vector<int> zeroInfo(batchSize, 0);
    aclRet = aclrtMemcpy(bufs.dInfo, infoBytes, zeroInfo.data(), infoBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t MatinvEnsureWorkspace(
    aclblasHandle_t handle, aclblasHandle_t internalHandleRef, size_t requiredWs,
    void*& expandedWs, bool& wsExpanded)
{
    wsExpanded = false;
    expandedWs = nullptr;
    const size_t alignedWs = (requiredWs + 4095U) & ~static_cast<size_t>(4095U);
    auto* h = internalHandleRef;
    size_t currentWs = GetEffectiveWorkspaceSize(h);

    if (alignedWs > currentWs) {
        aclError aclRet = aclrtMalloc(&expandedWs, alignedWs, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclblasStatus_t wsRet = aclblasSetWorkspace(handle, expandedWs, alignedWs);
        if (wsRet != ACLBLAS_STATUS_SUCCESS) {
            aclrtFree(expandedWs);
            expandedWs = nullptr;
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        wsExpanded = true;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t MatinvRestoreWorkspace(aclblasHandle_t handle, void* expandedWs, bool wsExpanded)
{
    if (!wsExpanded) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclrtStream stream = nullptr;
    aclblasStatus_t getStreamStatus = aclblasGetStream(handle, &stream);
    if (getStreamStatus != ACLBLAS_STATUS_SUCCESS) {
        return getStreamStatus;
    }

    aclblasStatus_t setStreamStatus = aclblasSetStream(handle, stream);
    if (setStreamStatus != ACLBLAS_STATUS_SUCCESS) {
        return setStreamStatus;
    }

    aclrtFree(expandedWs);
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t MatinvCopyResultsD2H(
    const MatinvDeviceBuffers& bufs, float* const Ainv[], int* info, int batchSize,
    size_t invBytes, size_t infoBytes)
{
    for (int b = 0; b < batchSize; b++) {
        if (Ainv[b] != nullptr) {
            aclError aclRet = aclrtMemcpy(
                Ainv[b], invBytes, bufs.dAinvMatrices[b], invBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (aclRet != ACL_SUCCESS) {
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    if (info != nullptr) {
        aclError aclRet = aclrtMemcpy(info, infoBytes, bufs.dInfo, infoBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t MatinvExecuteKernelAndSync(
    aclblasHandle_t handle, MatinvDeviceBuffers& bufs,
    int n, int lda, int lda_inv, int batchSize,
    float* const Ainv[], int* info, size_t invBytes, size_t infoBytes,
    void* expandedWs, bool wsExpanded)
{
    aclblasStatus_t ret = aclblasSmatinvBatched(
        handle, n,
        reinterpret_cast<const float* const*>(bufs.dAPtrArray), lda,
        reinterpret_cast<float* const*>(bufs.dAinvPtrArray), lda_inv,
        static_cast<int*>(bufs.dInfo), batchSize);

    aclError aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) {
        const aclblasStatus_t restoreStatus = MatinvRestoreWorkspace(handle, expandedWs, wsExpanded);
        bufs.Cleanup();
        if (restoreStatus != ACLBLAS_STATUS_SUCCESS) {
            return restoreStatus;
        }
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    const aclblasStatus_t restoreStatus = MatinvRestoreWorkspace(handle, expandedWs, wsExpanded);
    if (restoreStatus != ACLBLAS_STATUS_SUCCESS) {
        bufs.Cleanup();
        return restoreStatus;
    }

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t d2hStatus = MatinvCopyResultsD2H(bufs, Ainv, info, batchSize, invBytes, infoBytes);
        if (d2hStatus != ACLBLAS_STATUS_SUCCESS) {
            bufs.Cleanup();
            return d2hStatus;
        }
    }

    bufs.Cleanup();
    return ret;
}

static inline bool MatinvShouldUseFastPath(
    aclblasHandle_t handle, int n, int batchSize, int lda, int lda_inv,
    const float* const A[], float* const Ainv[], int* info)
{
    return (handle == nullptr || n < 0 || n > 32 || n == 0 || batchSize < 0 || batchSize == 0 ||
            lda < std::max(1, n) || lda_inv < std::max(1, n) ||
            A == nullptr || Ainv == nullptr || info == nullptr);
}

inline aclblasStatus_t MatinvAllocateAllMatrices(MatinvDeviceBuffers& bufs, const float* const A[], int batchSize,
                                                 size_t matBytes, size_t invBytes)
{
    aclblasStatus_t status = MatinvAllocAndCopyAMatrices(bufs, A, batchSize, matBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = MatinvAllocAinvMatrices(bufs, batchSize, invBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        bufs.Cleanup();
    }
    return status;
}

inline aclblasStatus_t aclblasSmatinvBatched_npu(
    aclblasHandle_t handle, int n, const float* const A[], int lda,
    float* const Ainv[], int lda_inv, int* info, int batchSize)
{
    if (MatinvShouldUseFastPath(handle, n, batchSize, lda, lda_inv, A, Ainv, info)) {
        return aclblasSmatinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
    }
    const size_t matBytes = static_cast<size_t>(lda) * n * sizeof(float);
    const size_t invBytes = static_cast<size_t>(lda_inv) * n * sizeof(float);
    const size_t ptrArrayBytes = static_cast<size_t>(batchSize) * sizeof(float*);
    const size_t infoBytes = static_cast<size_t>(batchSize) * sizeof(int);

    MatinvDeviceBuffers bufs;
    aclblasStatus_t status = MatinvAllocateAllMatrices(bufs, A, batchSize, matBytes, invBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = MatinvCreatePtrArraysAndInfo(bufs, batchSize, ptrArrayBytes, infoBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        bufs.Cleanup();
        return status;
    }
    const size_t requiredWs = static_cast<size_t>(batchSize) * sizeof(void*) +
                              static_cast<size_t>(n) * n * batchSize * sizeof(float) +
                              static_cast<size_t>(n) * batchSize * sizeof(int);
    void* expandedWs = nullptr;
    bool wsExpanded = false;
    status = MatinvEnsureWorkspace(handle, handle, requiredWs, expandedWs, wsExpanded);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        bufs.Cleanup();
        return status;
    }
    return MatinvExecuteKernelAndSync(
        handle, bufs, n, lda, lda_inv, batchSize, Ainv, info, invBytes, infoBytes,
        expandedWs, wsExpanded);
}
