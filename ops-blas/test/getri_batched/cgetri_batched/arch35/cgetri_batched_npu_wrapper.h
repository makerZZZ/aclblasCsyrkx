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

#include <cstddef>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

class CgetriBatchedNpuWrapper {
public:
    ~CgetriBatchedNpuWrapper() { Cleanup(); }

    CgetriBatchedNpuWrapper(const CgetriBatchedNpuWrapper&) = delete;
    CgetriBatchedNpuWrapper& operator=(const CgetriBatchedNpuWrapper&) = delete;
    CgetriBatchedNpuWrapper() = default;

    aclblasStatus_t Prepare(
        const aclblasComplex* const Aarray[], aclblasComplex* const Carray[], const int* PivotArray, int n, int lda,
        int ldc, int batchSize)
    {
        n_ = n;
        lda_ = lda;
        ldc_ = ldc;
        batchSize_ = batchSize;
        aBytes_ = static_cast<size_t>(lda) * n * sizeof(aclblasComplex);
        cBytes_ = static_cast<size_t>(ldc) * n * sizeof(aclblasComplex);
        infoBytes_ = static_cast<size_t>(batchSize) * sizeof(int);

        dAMatrices_.resize(static_cast<size_t>(batchSize), nullptr);
        dCMatrices_.resize(static_cast<size_t>(batchSize), nullptr);
        for (int batch = 0; batch < batchSize; batch++) {
            aclblasStatus_t status =
                AllocateAndCopy(&dAMatrices_[batch], Aarray[batch], aBytes_, ACL_MEMCPY_HOST_TO_DEVICE);
            if (status != ACLBLAS_STATUS_SUCCESS)
                return status;
            status = AllocateAndCopy(&dCMatrices_[batch], Carray[batch], cBytes_, ACL_MEMCPY_HOST_TO_DEVICE);
            if (status != ACLBLAS_STATUS_SUCCESS)
                return status;
        }

        aclblasStatus_t status = CreatePointerArray(dAPtrArray_, dAMatrices_);
        if (status != ACLBLAS_STATUS_SUCCESS)
            return status;
        status = CreatePointerArray(dCPtrArray_, dCMatrices_);
        if (status != ACLBLAS_STATUS_SUCCESS)
            return status;

        if (PivotArray != nullptr) {
            size_t pivotBytes = static_cast<size_t>(n) * batchSize * sizeof(int);
            status = AllocateAndCopy(&dPivot_, PivotArray, pivotBytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (status != ACLBLAS_STATUS_SUCCESS)
                return status;
        }

        std::vector<int> info(static_cast<size_t>(batchSize), -1);
        return AllocateAndCopy(&dInfo_, info.data(), infoBytes_, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    aclblasStatus_t Run(aclblasHandle_t handle) const
    {
        return aclblasCgetriBatched(
            handle, n_, reinterpret_cast<const aclblasComplex* const*>(dAPtrArray_), lda_,
            static_cast<const int*>(dPivot_), reinterpret_cast<aclblasComplex* const*>(dCPtrArray_), ldc_,
            static_cast<int*>(dInfo_), batchSize_);
    }

    aclblasStatus_t Synchronize(aclrtStream stream) const
    {
        return aclrtSynchronizeStream(stream) == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    aclblasStatus_t CopyResults(aclblasComplex* const Carray[], int* infoArray) const
    {
        for (int batch = 0; batch < batchSize_; batch++) {
            aclError ret = aclrtMemcpy(Carray[batch], cBytes_, dCMatrices_[batch], cBytes_, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS)
                return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        aclError ret = aclrtMemcpy(infoArray, infoBytes_, dInfo_, infoBytes_, ACL_MEMCPY_DEVICE_TO_HOST);
        return ret == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
    }

private:
    static aclblasStatus_t AllocateAndCopy(void** device, const void* host, size_t bytes, aclrtMemcpyKind kind)
    {
        aclError ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
            return ACLBLAS_STATUS_ALLOC_FAILED;
        ret = aclrtMemcpy(*device, bytes, host, bytes, kind);
        return ret == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclblasStatus_t CreatePointerArray(void*& deviceArray, const std::vector<void*>& matrices)
    {
        size_t bytes = matrices.size() * sizeof(aclblasComplex*);
        aclError ret = aclrtMalloc(&deviceArray, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS)
            return ACLBLAS_STATUS_ALLOC_FAILED;
        ret = aclrtMemcpy(deviceArray, bytes, matrices.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        return ret == ACL_SUCCESS ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    void Cleanup()
    {
        for (void* pointer : dAMatrices_) {
            if (pointer != nullptr)
                aclrtFree(pointer);
        }
        for (void* pointer : dCMatrices_) {
            if (pointer != nullptr)
                aclrtFree(pointer);
        }
        if (dAPtrArray_ != nullptr)
            aclrtFree(dAPtrArray_);
        if (dCPtrArray_ != nullptr)
            aclrtFree(dCPtrArray_);
        if (dPivot_ != nullptr)
            aclrtFree(dPivot_);
        if (dInfo_ != nullptr)
            aclrtFree(dInfo_);
        dAMatrices_.clear();
        dCMatrices_.clear();
        dAPtrArray_ = nullptr;
        dCPtrArray_ = nullptr;
        dPivot_ = nullptr;
        dInfo_ = nullptr;
    }

    int n_ = 0;
    int lda_ = 0;
    int ldc_ = 0;
    int batchSize_ = 0;
    size_t aBytes_ = 0;
    size_t cBytes_ = 0;
    size_t infoBytes_ = 0;
    std::vector<void*> dAMatrices_;
    std::vector<void*> dCMatrices_;
    void* dAPtrArray_ = nullptr;
    void* dCPtrArray_ = nullptr;
    void* dPivot_ = nullptr;
    void* dInfo_ = nullptr;
};

inline aclblasStatus_t aclblasCgetriBatchedNpu(
    aclblasHandle_t handle, int n, const aclblasComplex* const Aarray[], int lda, const int* PivotArray,
    aclblasComplex* const Carray[], int ldc, int* infoArray, int batchSize, aclrtStream stream)
{
    if (handle == nullptr || n <= 0 || batchSize <= 0 || Aarray == nullptr || Carray == nullptr ||
        infoArray == nullptr) {
        return aclblasCgetriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize);
    }

    CgetriBatchedNpuWrapper buffers;
    aclblasStatus_t status = buffers.Prepare(Aarray, Carray, PivotArray, n, lda, ldc, batchSize);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;

    status = buffers.Run(handle);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    status = buffers.Synchronize(stream);
    if (status != ACLBLAS_STATUS_SUCCESS)
        return status;
    return buffers.CopyResults(Carray, infoArray);
}
