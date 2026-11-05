/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BATCHED_NPU_COMMON_H
#define GEMM_BATCHED_NPU_COMMON_H

#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct GemmBatchedDeviceCtx {
    std::vector<void*> dAarray;
    std::vector<void*> dBarray;
    std::vector<void*> dCarray;
    void* dAPtrArray = nullptr;
    void* dBPtrArray = nullptr;
    void* dCPtrArray = nullptr;

    void Init(int batchCount)
    {
        dAarray.assign(batchCount, nullptr);
        dBarray.assign(batchCount, nullptr);
        dCarray.assign(batchCount, nullptr);
    }

    void Cleanup()
    {
        for (size_t i = 0; i < dAarray.size(); i++) {
            if (dAarray[i]) aclrtFree(dAarray[i]);
            if (dBarray[i]) aclrtFree(dBarray[i]);
            if (dCarray[i]) aclrtFree(dCarray[i]);
        }
        if (dAPtrArray) aclrtFree(dAPtrArray);
        if (dBPtrArray) aclrtFree(dBPtrArray);
        if (dCPtrArray) aclrtFree(dCPtrArray);
    }
};

struct GemmBatchedBufferSizes {
    size_t aBytes;
    size_t bBytes;
    size_t cBytes;
};

inline aclblasStatus_t AllocAndCopyBuffer(void** devPtr, size_t bytes, const void* hostPtr)
{
    aclError ret = aclrtMalloc(devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { return ACLBLAS_STATUS_ALLOC_FAILED; }
    ret = aclrtMemcpy(*devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { return ACLBLAS_STATUS_INTERNAL_ERROR; }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t AllocAndCopyPtrArray(void** devPtrArray, const void* hostPtrs, size_t bytes)
{
    aclError ret = aclrtMalloc(devPtrArray, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { return ACLBLAS_STATUS_ALLOC_FAILED; }
    ret = aclrtMemcpy(*devPtrArray, bytes, hostPtrs, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { return ACLBLAS_STATUS_INTERNAL_ERROR; }
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename T>
inline aclblasStatus_t AllocateBatchBuffersTpl(
    GemmBatchedDeviceCtx& ctx, int i,
    const T* const Aarray[], const T* const Barray[], T* const Carray[],
    bool needAB, const GemmBatchedBufferSizes& sizes)
{
    if (needAB && Aarray && Aarray[i]) {
        aclblasStatus_t s = AllocAndCopyBuffer(&ctx.dAarray[i], sizes.aBytes, Aarray[i]);
        if (s != ACLBLAS_STATUS_SUCCESS) { return s; }
    }
    if (needAB && Barray && Barray[i]) {
        aclblasStatus_t s = AllocAndCopyBuffer(&ctx.dBarray[i], sizes.bBytes, Barray[i]);
        if (s != ACLBLAS_STATUS_SUCCESS) { return s; }
    }
    if (Carray && Carray[i]) {
        aclblasStatus_t s = AllocAndCopyBuffer(&ctx.dCarray[i], sizes.cBytes, Carray[i]);
        if (s != ACLBLAS_STATUS_SUCCESS) { return s; }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t AllocatePtrArrays(
    GemmBatchedDeviceCtx& ctx, bool needAB,
    bool hasA, bool hasB, bool hasC, int batchCount)
{
    size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    if (needAB && hasA) {
        aclblasStatus_t s = AllocAndCopyPtrArray(&ctx.dAPtrArray, ctx.dAarray.data(), ptrBytes);
        if (s != ACLBLAS_STATUS_SUCCESS) { return s; }
    }
    if (needAB && hasB) {
        aclblasStatus_t s = AllocAndCopyPtrArray(&ctx.dBPtrArray, ctx.dBarray.data(), ptrBytes);
        if (s != ACLBLAS_STATUS_SUCCESS) { return s; }
    }
    if (hasC) {
        aclblasStatus_t s = AllocAndCopyPtrArray(&ctx.dCPtrArray, ctx.dCarray.data(), ptrBytes);
        if (s != ACLBLAS_STATUS_SUCCESS) { return s; }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename T>
inline aclblasStatus_t CopyCarrayFromDevice(
    GemmBatchedDeviceCtx& ctx, T* const Carray[], int batchCount, size_t cBytes)
{
    if (Carray == nullptr) { return ACLBLAS_STATUS_SUCCESS; }
    for (int i = 0; i < batchCount; i++) {
        if (Carray[i] == nullptr || ctx.dCarray[i] == nullptr) { continue; }
        aclError cpyRet = aclrtMemcpy(Carray[i], cBytes, ctx.dCarray[i], cBytes,
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (cpyRet != ACL_SUCCESS) { return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

template <typename TestData, typename Param, typename PtrArrays>
inline PtrArrays BuildGemmBatchedPtrsTpl(TestData& data, const Param& p, int safeBatch)
{
    PtrArrays ptrs;
    ptrs.aPtrs.resize(safeBatch, nullptr);
    ptrs.bPtrs.resize(safeBatch, nullptr);
    ptrs.cPtrs.resize(safeBatch, nullptr);
    for (int b = 0; b < safeBatch; b++) {
        if (!data.aBatch[b].empty() && !p.aarrayNull) {
            ptrs.aPtrs[b] = data.aBatch[b].data();
        }
        if (!data.bBatch[b].empty() && !p.barrayNull) {
            ptrs.bPtrs[b] = data.bBatch[b].data();
        }
        if (!data.cBatch[b].empty() && !p.carrayNull) {
            ptrs.cPtrs[b] = data.cBatch[b].data();
        }
    }
    return ptrs;
}

template <typename T, typename BatchedFn>
inline aclblasStatus_t RunBatchedSyncAndCopy(
    GemmBatchedDeviceCtx& ctx,
    T* const Carray[], int batchCount, size_t cBytes,
    BatchedFn batchedFn)
{
    aclblasStatus_t ret = batchedFn();
    if (ret != ACLBLAS_STATUS_SUCCESS) { ctx.Cleanup(); return ret; }
    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) { ctx.Cleanup(); return ACLBLAS_STATUS_EXECUTION_FAILED; }
    aclblasStatus_t cpyRet = CopyCarrayFromDevice(ctx, Carray, batchCount, cBytes);
    ctx.Cleanup();
    return (cpyRet == ACLBLAS_STATUS_SUCCESS) ? ret : cpyRet;
}

#endif // GEMM_BATCHED_NPU_COMMON_H
