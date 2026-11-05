/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STRSMBATCHED_NPU_WRAPPER_H
#define STRSMBATCHED_NPU_WRAPPER_H

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct TrsmbatchedDeviceBuffers {
    std::vector<void*> dA;
    std::vector<void*> dB;
    void* dAPtrArray = nullptr;
    void* dBPtrArray = nullptr;

    void Cleanup()
    {
        for (auto& p : dA) {
            if (p) aclrtFree(p);
        }
        for (auto& p : dB) {
            if (p) aclrtFree(p);
        }
        if (dAPtrArray) aclrtFree(dAPtrArray);
        if (dBPtrArray) aclrtFree(dBPtrArray);
        dA.clear();
        dB.clear();
        dAPtrArray = nullptr;
        dBPtrArray = nullptr;
    }
};

inline aclblasStatus_t TrsmbatchedAllocAndCopyA(
    TrsmbatchedDeviceBuffers& bufs, const float* const A[],
    int batchCount, size_t aBytes)
{
    bufs.dA.resize(batchCount, nullptr);
    for (int b = 0; b < batchCount; b++) {
        aclError aclRet = aclrtMalloc(&bufs.dA[b], aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        if (A[b] != nullptr) {
            aclRet = aclrtMemcpy(bufs.dA[b], aBytes, A[b], aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (aclRet != ACL_SUCCESS) {
                bufs.Cleanup();
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t TrsmbatchedAllocAndCopyB(
    TrsmbatchedDeviceBuffers& bufs, float* const B[],
    int batchCount, size_t bBytes)
{
    bufs.dB.resize(batchCount, nullptr);
    for (int b = 0; b < batchCount; b++) {
        aclError aclRet = aclrtMalloc(&bufs.dB[b], bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        if (B[b] != nullptr) {
            aclRet = aclrtMemcpy(bufs.dB[b], bBytes, B[b], bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (aclRet != ACL_SUCCESS) {
                bufs.Cleanup();
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t TrsmbatchedCreatePtrArrays(
    TrsmbatchedDeviceBuffers& bufs, int batchCount, bool needA, size_t ptrArrayBytes)
{
    if (needA) {
        std::vector<float*> hAPtrArray(batchCount);
        for (int b = 0; b < batchCount; b++) {
            hAPtrArray[b] = static_cast<float*>(bufs.dA[b]);
        }
        aclError aclRet = aclrtMalloc(&bufs.dAPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclRet = aclrtMemcpy(bufs.dAPtrArray, ptrArrayBytes, hAPtrArray.data(), ptrArrayBytes,
                             ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    std::vector<float*> hBPtrArray(batchCount);
    for (int b = 0; b < batchCount; b++) {
        hBPtrArray[b] = static_cast<float*>(bufs.dB[b]);
    }
    aclError aclRet = aclrtMalloc(&bufs.dBPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(bufs.dBPtrArray, ptrArrayBytes, hBPtrArray.data(), ptrArrayBytes,
                         ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        bufs.Cleanup();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t TrsmbatchedCopyResultsD2H(
    TrsmbatchedDeviceBuffers& bufs, float* const B[], int batchCount, size_t bBytes)
{
    for (int b = 0; b < batchCount; b++) {
        if (B[b] != nullptr && bufs.dB[b] != nullptr) {
            aclError d2hRet = aclrtMemcpy(B[b], bBytes, bufs.dB[b], bBytes,
                                          ACL_MEMCPY_DEVICE_TO_HOST);
            if (d2hRet != ACL_SUCCESS) {
                bufs.Cleanup();
                return ACLBLAS_STATUS_INTERNAL_ERROR;
            }
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasStrsmBatched_npu(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    aclblasOperation_t trans, aclblasDiagType_t diag,
    int m, int n, const float* alpha,
    const float* const A[], int lda,
    float* const B[], int ldb,
    int batchCount)
{
    if (handle == nullptr || m <= 0 || n <= 0 || batchCount <= 0 ||
        alpha == nullptr || B == nullptr) {
        return aclblasStrsmBatched(handle, side, uplo, trans, diag,
                                   m, n, alpha, A, lda, B, ldb, batchCount);
    }

    const int aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(aDim) * sizeof(float);
    const size_t bBytes = static_cast<size_t>(ldb) * static_cast<size_t>(n) * sizeof(float);
    const size_t ptrArrayBytes = static_cast<size_t>(batchCount) * sizeof(float*);

    TrsmbatchedDeviceBuffers bufs;
    bool needA = (A != nullptr && *alpha != 0.0f);

    if (needA) {
        aclblasStatus_t st = TrsmbatchedAllocAndCopyA(bufs, A, batchCount, aBytes);
        if (st != ACLBLAS_STATUS_SUCCESS) return st;
    }

    aclblasStatus_t st = TrsmbatchedAllocAndCopyB(bufs, B, batchCount, bBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    st = TrsmbatchedCreatePtrArrays(bufs, batchCount, needA, ptrArrayBytes);
    if (st != ACLBLAS_STATUS_SUCCESS) return st;

    const float* const* dAPtrs = needA ? static_cast<const float* const*>(bufs.dAPtrArray) : nullptr;
    float* const* dBPtrs = static_cast<float* const*>(bufs.dBPtrArray);

    aclblasStatus_t ret = aclblasStrsmBatched(
        handle, side, uplo, trans, diag, m, n, alpha,
        dAPtrs, lda, dBPtrs, ldb, batchCount);

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        aclrtStream stream = nullptr;
        aclblasGetStream(handle, &stream);
        aclError syncRet = aclrtSynchronizeStream(stream);
        if (syncRet != ACL_SUCCESS) {
            bufs.Cleanup();
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        st = TrsmbatchedCopyResultsD2H(bufs, B, batchCount, bBytes);
        if (st != ACLBLAS_STATUS_SUCCESS) return st;
    }

    bufs.Cleanup();
    return ret;
}

#endif // STRSMBATCHED_NPU_WRAPPER_H
