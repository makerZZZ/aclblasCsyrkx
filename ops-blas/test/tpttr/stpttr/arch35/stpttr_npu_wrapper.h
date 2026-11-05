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

#include "acl/acl.h"
#include "cann_ops_blas.h"

// NPU wrapper — same signature as aclblasStpttr.
// nullptr inputs are passed through without device allocation (for error-path testing).

static inline aclblasStatus_t CopyToDevice(void** dPtr, const void* hPtr, size_t bytes)
{
    if (hPtr == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMalloc(dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMemcpy(*dPtr, bytes, hPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*dPtr);
        *dPtr = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static inline aclblasStatus_t CopyFromDevice(void* hPtr, const void* dPtr, size_t bytes)
{
    if (hPtr == nullptr || dPtr == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMemcpy(hPtr, bytes, dPtr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return (ret == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

static inline void FreeDevice(void* dP, void* dA)
{
    if (dP)
        aclrtFree(dP);
    if (dA)
        aclrtFree(dA);
}

inline aclblasStatus_t aclblasStpttr_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* ap, float* a, int lda)
{
    if (handle == nullptr || n <= 0) {
        return aclblasStpttr(handle, uplo, n, ap, a, lda);
    }

    const int allocN = std::max(1, n);
    const int allocLda = std::max(allocN, lda);
    const size_t apBytes = static_cast<size_t>(allocN) * (allocN + 1) / 2 * sizeof(float);
    const size_t aBytes = static_cast<size_t>(allocLda) * allocN * sizeof(float);

    void* dP = nullptr;
    void* dA = nullptr;

    aclblasStatus_t status = CopyToDevice(&dP, ap, apBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = CopyToDevice(&dA, a, aBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        FreeDevice(dP, dA);
        return status;
    }

    aclblasStatus_t ret = aclblasStpttr(handle, uplo, n, static_cast<const float*>(dP), static_cast<float*>(dA), lda);

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        FreeDevice(dP, dA);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    status = CopyFromDevice(a, dA, aBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        FreeDevice(dP, dA);
        return status;
    }

    FreeDevice(dP, dA);
    return ret;
}
