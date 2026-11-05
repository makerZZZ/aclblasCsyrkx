/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_GROUPED_BATCHED_NPU_WRAPPER_H
#define GEMM_GROUPED_BATCHED_NPU_WRAPPER_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "gemm_grouped_batched_golden.h"

// ── Helper: compute matrix element count for GEMM ──
// (Defined in golden.h; forward-declared here to avoid circular include)

// ── S type NPU wrapper ──

inline aclblasStatus_t PassThroughSgemmGroupedBatched(
    aclblasHandle_t handle, int groupCount,
    const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray, const int* ldaArray, const int* ldbArray,
    const float* betaArray, const int* ldcArray, const int* groupSizeArray)
{
    return aclblasSgemmGroupedBatched(handle, groupCount,
        transaArray, transbArray, mArray, nArray, kArray,
        alphaArray, nullptr, ldaArray, nullptr, ldbArray,
        betaArray, nullptr, ldcArray, groupSizeArray);
}

inline bool HasNullNpuWrapperArrays(
    const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray, const float* betaArray,
    const int* ldaArray, const int* ldbArray, const int* ldcArray,
    const int* groupSizeArray,
    const float* const* Aarray_host, const float* const* Barray_host,
    float* const* Carray_host)
{
    return transaArray == nullptr || transbArray == nullptr ||
        mArray == nullptr || nArray == nullptr || kArray == nullptr ||
        alphaArray == nullptr || betaArray == nullptr ||
        ldaArray == nullptr || ldbArray == nullptr || ldcArray == nullptr ||
        groupSizeArray == nullptr || Aarray_host == nullptr ||
        Barray_host == nullptr || Carray_host == nullptr;
}

inline bool HasInvalidGroupDimensions(
    int groupCount, const int* mArray, const int* nArray,
    const int* kArray, const int* groupSizeArray)
{
    for (int g = 0; g < groupCount; g++) {
        if (mArray[g] < 0 || nArray[g] < 0 || kArray[g] < 0 || groupSizeArray[g] < 0) {
            return true;
        }
    }
    return false;
}

inline int ComputeTotalBatchCount(int groupCount, const int* groupSizeArray)
{
    int totalBatchCount = 0;
    for (int g = 0; g < groupCount; g++) {
        totalBatchCount += groupSizeArray[g];
    }
    return totalBatchCount;
}

struct GemmGroupedDeviceBuffers {
    std::vector<void*> dA;
    std::vector<void*> dB;
    std::vector<void*> dC;

    explicit GemmGroupedDeviceBuffers(int totalBatchCount)
        : dA(static_cast<size_t>(totalBatchCount), nullptr),
          dB(static_cast<size_t>(totalBatchCount), nullptr),
          dC(static_cast<size_t>(totalBatchCount), nullptr)
    {
    }

    void Release()
    {
        for (size_t i = 0; i < dA.size(); i++) {
            if (dA[i] != nullptr) {
                aclrtFree(dA[i]);
                dA[i] = nullptr;
            }
            if (dB[i] != nullptr) {
                aclrtFree(dB[i]);
                dB[i] = nullptr;
            }
            if (dC[i] != nullptr) {
                aclrtFree(dC[i]);
                dC[i] = nullptr;
            }
        }
    }
};

inline aclblasStatus_t AllocAndCopyHostToDevice(void** dst, size_t bytes, const void* src)
{
    if (aclrtMalloc(dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtMemcpy(*dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(*dst);
        *dst = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t UploadOneBatchSlot(
    int batchIdx, int m, int n, int k,
    size_t aBytes, size_t bBytes, size_t cBytes,
    const float* const* Aarray_host, const float* const* Barray_host,
    const float* const* Carray_host, GemmGroupedDeviceBuffers& buffers)
{
    if (m <= 0 || n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (k > 0) {
        aclblasStatus_t ret = AllocAndCopyHostToDevice(
            &buffers.dA[batchIdx], aBytes, Aarray_host[batchIdx]);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
        ret = AllocAndCopyHostToDevice(&buffers.dB[batchIdx], bBytes, Barray_host[batchIdx]);
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
    }
    return AllocAndCopyHostToDevice(&buffers.dC[batchIdx], cBytes, Carray_host[batchIdx]);
}

inline aclblasStatus_t UploadGroupedBatchToDevice(
    int groupCount,
    const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const int* ldaArray, const int* ldbArray, const int* ldcArray,
    const int* groupSizeArray,
    const float* const* Aarray_host, const float* const* Barray_host,
    const float* const* Carray_host, GemmGroupedDeviceBuffers& buffers)
{
    const size_t elemSize = sizeof(float);
    int batchIdx = 0;
    for (int g = 0; g < groupCount; g++) {
        int m = mArray[g];
        int n = nArray[g];
        int k = kArray[g];
        int aCols = getACols(transaArray[g], m, k);
        int bCols = getBCols(transbArray[g], k, n);
        size_t aBytes = static_cast<size_t>(ldaArray[g]) * aCols * elemSize;
        size_t bBytes = static_cast<size_t>(ldbArray[g]) * bCols * elemSize;
        size_t cBytes = static_cast<size_t>(ldcArray[g]) * n * elemSize;

        for (int i = 0; i < groupSizeArray[g]; i++) {
            aclblasStatus_t ret = UploadOneBatchSlot(
                batchIdx, m, n, k, aBytes, bBytes, cBytes,
                Aarray_host, Barray_host, Carray_host, buffers);
            if (ret != ACLBLAS_STATUS_SUCCESS) {
                return ret;
            }
            batchIdx++;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline void BuildHostDevicePtrArrays(
    const GemmGroupedDeviceBuffers& buffers,
    std::vector<const float*>& hAPtrs, std::vector<const float*>& hBPtrs,
    std::vector<float*>& hCPtrs)
{
    const int totalBatchCount = static_cast<int>(buffers.dA.size());
    hAPtrs.resize(static_cast<size_t>(totalBatchCount));
    hBPtrs.resize(static_cast<size_t>(totalBatchCount));
    hCPtrs.resize(static_cast<size_t>(totalBatchCount));
    for (int i = 0; i < totalBatchCount; i++) {
        hAPtrs[i] = static_cast<const float*>(buffers.dA[i]);
        hBPtrs[i] = static_cast<const float*>(buffers.dB[i]);
        hCPtrs[i] = static_cast<float*>(buffers.dC[i]);
    }
}

inline aclblasStatus_t CopyOneBatchResultToHost(
    int batchIdx, int m, int n, size_t cBytes,
    float* const* Carray_host, void* dC)
{
    if (m <= 0 || n <= 0 || dC == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMemcpy(Carray_host[batchIdx], cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t SyncAndCopyGroupedBatchResultsToHost(
    int groupCount, const int* mArray, const int* nArray,
    const int* ldcArray, const int* groupSizeArray,
    float* const* Carray_host, const GemmGroupedDeviceBuffers& buffers)
{
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    const size_t elemSize = sizeof(float);
    int batchIdx = 0;
    for (int g = 0; g < groupCount; g++) {
        int m = mArray[g];
        int n = nArray[g];
        size_t cBytes = static_cast<size_t>(ldcArray[g]) * n * elemSize;
        for (int i = 0; i < groupSizeArray[g]; i++) {
            aclblasStatus_t ret = CopyOneBatchResultToHost(
                batchIdx, m, n, cBytes, Carray_host, buffers.dC[batchIdx]);
            if (ret != ACLBLAS_STATUS_SUCCESS) {
                return ret;
            }
            batchIdx++;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline bool NeedsNpuDeviceBuffers(
    int groupCount,
    const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray, const float* betaArray,
    const int* ldaArray, const int* ldbArray, const int* ldcArray,
    const int* groupSizeArray,
    const float* const* Aarray_host, const float* const* Barray_host,
    float* const* Carray_host, int& totalBatchCount)
{
    if (groupCount <= 0) {
        return false;
    }
    if (HasNullNpuWrapperArrays(
            transaArray, transbArray, mArray, nArray, kArray,
            alphaArray, betaArray, ldaArray, ldbArray, ldcArray, groupSizeArray,
            Aarray_host, Barray_host, Carray_host)) {
        return false;
    }
    if (HasInvalidGroupDimensions(groupCount, mArray, nArray, kArray, groupSizeArray)) {
        return false;
    }
    totalBatchCount = ComputeTotalBatchCount(groupCount, groupSizeArray);
    return totalBatchCount > 0;
}

inline aclblasStatus_t aclblasSgemmGroupedBatched_npu(
    aclblasHandle_t handle, int groupCount,
    const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray,
    const int* mArray, const int* nArray, const int* kArray,
    const float* alphaArray,
    const float* const* Aarray_host, const int* ldaArray,
    const float* const* Barray_host, const int* ldbArray,
    const float* betaArray,
    float* const* Carray_host, const int* ldcArray,
    const int* groupSizeArray)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    int totalBatchCount = 0;
    if (!NeedsNpuDeviceBuffers(
            groupCount, transaArray, transbArray, mArray, nArray, kArray,
            alphaArray, betaArray, ldaArray, ldbArray, ldcArray, groupSizeArray,
            Aarray_host, Barray_host, Carray_host, totalBatchCount)) {
        return PassThroughSgemmGroupedBatched(
            handle, groupCount, transaArray, transbArray,
            mArray, nArray, kArray, alphaArray, ldaArray, ldbArray,
            betaArray, ldcArray, groupSizeArray);
    }

    GemmGroupedDeviceBuffers buffers(totalBatchCount);
    aclblasStatus_t ret = UploadGroupedBatchToDevice(
        groupCount, transaArray, transbArray, mArray, nArray, kArray,
        ldaArray, ldbArray, ldcArray, groupSizeArray,
        Aarray_host, Barray_host, Carray_host, buffers);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        buffers.Release();
        return ret;
    }

    std::vector<const float*> hAPtrs;
    std::vector<const float*> hBPtrs;
    std::vector<float*> hCPtrs;
    BuildHostDevicePtrArrays(buffers, hAPtrs, hBPtrs, hCPtrs);

    ret = aclblasSgemmGroupedBatched(handle, groupCount,
        transaArray, transbArray, mArray, nArray, kArray,
        alphaArray, hAPtrs.data(), ldaArray,
        hBPtrs.data(), ldbArray,
        betaArray, hCPtrs.data(), ldcArray,
        groupSizeArray);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        buffers.Release();
        return ret;
    }

    ret = SyncAndCopyGroupedBatchResultsToHost(
        groupCount, mArray, nArray, ldcArray, groupSizeArray, Carray_host, buffers);
    buffers.Release();
    return ret;
}

#endif // GEMM_GROUPED_BATCHED_NPU_WRAPPER_H
