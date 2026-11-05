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
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct GeqrfDeviceBuffers {
    std::vector<float*> dA;
    std::vector<float*> dTau;
    float** dAarray = nullptr;
    float** dTauArray = nullptr;
};

inline void FreeGeqrfDeviceBuffers(GeqrfDeviceBuffers& bufs, int batchSize)
{
    for (int b = 0; b < batchSize; b++) {
        if (bufs.dA[b]) {
            aclrtFree(bufs.dA[b]);
            bufs.dA[b] = nullptr;
        }
        if (bufs.dTau[b]) {
            aclrtFree(bufs.dTau[b]);
            bufs.dTau[b] = nullptr;
        }
    }
    if (bufs.dAarray) {
        aclrtFree(bufs.dAarray);
        bufs.dAarray = nullptr;
    }
    if (bufs.dTauArray) {
        aclrtFree(bufs.dTauArray);
        bufs.dTauArray = nullptr;
    }
}

inline aclblasStatus_t AllocGeqrfBatchBuffers(
    GeqrfDeviceBuffers& bufs, float* const Aarray[], int batchSize, size_t matBytes, size_t tauBytes)
{
    const size_t ptrArrayBytes = static_cast<size_t>(batchSize) * sizeof(float*);

    for (int b = 0; b < batchSize; b++) {
        if (Aarray[b] != nullptr) {
            aclError ret = aclrtMalloc(reinterpret_cast<void**>(&bufs.dA[b]), matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                return ACLBLAS_STATUS_ALLOC_FAILED;
            ret = aclrtMemcpy(bufs.dA[b], matBytes, Aarray[b], matBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS)
                return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        aclError ret = aclrtMalloc(reinterpret_cast<void**>(&bufs.dTau[b]), tauBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
            return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclError ret = aclrtMalloc(reinterpret_cast<void**>(&bufs.dAarray), ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    ret = aclrtMemcpy(bufs.dAarray, ptrArrayBytes, bufs.dA.data(), ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;

    ret = aclrtMalloc(reinterpret_cast<void**>(&bufs.dTauArray), ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    ret = aclrtMemcpy(bufs.dTauArray, ptrArrayBytes, bufs.dTau.data(), ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;

    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t CopyGeqrfResultsBack(
    const GeqrfDeviceBuffers& bufs, float* const Aarray[], float* const TauArray[], int batchSize, size_t matBytes,
    size_t tauBytes)
{
    for (int b = 0; b < batchSize; b++) {
        if (Aarray[b] != nullptr && bufs.dA[b] != nullptr) {
            aclError ret = aclrtMemcpy(Aarray[b], matBytes, bufs.dA[b], matBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
        if (TauArray[b] != nullptr && bufs.dTau[b] != nullptr) {
            aclError ret = aclrtMemcpy(TauArray[b], tauBytes, bufs.dTau[b], tauBytes, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasSgeqrfBatched_npu(
    aclblasHandle_t handle, int m, int n, float* const Aarray[], int lda, float* const TauArray[], int* info,
    int batchSize)
{
    if (handle == nullptr || batchSize <= 0 || m <= 0 || n <= 0 || Aarray == nullptr || TauArray == nullptr) {
        return aclblasSgeqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize);
    }

    const int k = std::min(m, n);
    const size_t matBytes = static_cast<size_t>(lda) * n * sizeof(float);
    const size_t tauBytes = static_cast<size_t>(k) * sizeof(float);

    GeqrfDeviceBuffers bufs;
    bufs.dA.resize(batchSize, nullptr);
    bufs.dTau.resize(batchSize, nullptr);

    aclblasStatus_t allocRet = AllocGeqrfBatchBuffers(bufs, Aarray, batchSize, matBytes, tauBytes);
    if (allocRet != ACLBLAS_STATUS_SUCCESS) {
        FreeGeqrfDeviceBuffers(bufs, batchSize);
        return allocRet;
    }

    aclblasStatus_t ret = aclblasSgeqrfBatched(handle, m, n, bufs.dAarray, lda, bufs.dTauArray, info, batchSize);
    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        FreeGeqrfDeviceBuffers(bufs, batchSize);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    aclblasStatus_t copyRet = CopyGeqrfResultsBack(bufs, Aarray, TauArray, batchSize, matBytes, tauBytes);
    FreeGeqrfDeviceBuffers(bufs, batchSize);
    return (ret != ACLBLAS_STATUS_SUCCESS) ? ret : copyRet;
}

