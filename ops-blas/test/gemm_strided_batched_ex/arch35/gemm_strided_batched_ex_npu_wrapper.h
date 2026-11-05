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

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct StridedGemmDeviceBuffers {
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;

    ~StridedGemmDeviceBuffers() { Cleanup(); }

    void Cleanup()
    {
        if (a != nullptr)
            aclrtFree(a);
        if (b != nullptr)
            aclrtFree(b);
        if (c != nullptr)
            aclrtFree(c);
        a = nullptr;
        b = nullptr;
        c = nullptr;
    }
};

inline aclblasStatus_t AllocateAndCopyStrided(void** device, const void* host, size_t bytes)
{
    if (host == nullptr || bytes == 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        *device = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(*device);
        *device = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasGemmStridedBatchedEx_npu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const void* alpha, const void* a, size_t aBytes, aclDataType aType, int lda, int64_t strideA, const void* b,
    size_t bBytes, aclDataType bType, int ldb, int64_t strideB, const void* beta, void* c, size_t cBytes,
    aclDataType cType, int ldc, int64_t strideC, int batchCount, aclblasComputeType_t computeType,
    aclblasGemmAlgo_t algo, size_t aBaseElements = 0, size_t bBaseElements = 0, size_t cBaseElements = 0)
{
    StridedGemmDeviceBuffers device;
    aclblasStatus_t status = AllocateAndCopyStrided(&device.a, a, aBytes);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = AllocateAndCopyStrided(&device.b, b, bBytes);
    }
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = AllocateAndCopyStrided(&device.c, c, cBytes);
    }
    if (status != ACLBLAS_STATUS_SUCCESS) {
        device.Cleanup();
        return status;
    }

    status = aclblasGemmStridedBatchedEx(
        handle, transA, transB, m, n, k, alpha,
        device.a != nullptr ? static_cast<uint8_t*>(device.a) + aBaseElements * aclDataTypeSize(aType) : a, aType, lda,
        strideA, device.b != nullptr ? static_cast<uint8_t*>(device.b) + bBaseElements * aclDataTypeSize(bType) : b,
        bType, ldb, strideB, beta,
        device.c != nullptr ? static_cast<uint8_t*>(device.c) + cBaseElements * aclDataTypeSize(cType) : c, cType, ldc,
        strideC, batchCount, computeType, algo);

    aclError syncStatus = aclrtSynchronizeDevice();
    if (status == ACLBLAS_STATUS_SUCCESS && syncStatus != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (status == ACLBLAS_STATUS_SUCCESS && c != nullptr && device.c != nullptr &&
        aclrtMemcpy(c, cBytes, device.c, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    device.Cleanup();
    return status;
}
