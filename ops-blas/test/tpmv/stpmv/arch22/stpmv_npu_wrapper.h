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

#include <cstdlib>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblasStpmv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* AP, float* x, int incx)
{
    if (handle == nullptr || n <= 0 || AP == nullptr || x == nullptr) {
        return aclblasStpmv_legacy(
            handle, static_cast<aclblasFillMode>(uplo), static_cast<aclblasOperation>(trans),
            static_cast<aclblasDiagType>(diag), static_cast<int64_t>(n), AP, x, x, static_cast<int64_t>(incx));
    }

    const int absInc = std::abs(incx);
    const size_t packedEleNum = static_cast<size_t>(n) * (static_cast<size_t>(n) + 1U) / 2U;
    const size_t packedBytes = packedEleNum * sizeof(float);
    const size_t xBytes = static_cast<size_t>(static_cast<size_t>(n - 1) * absInc + 1) * sizeof(float);
    const int64_t n64 = static_cast<int64_t>(n);
    const int64_t incx64 = static_cast<int64_t>(incx);

    void* dA = nullptr;
    void* dX = nullptr;
    void* dY = nullptr;
    aclError aclRet;

    aclRet = aclrtMalloc(&dA, packedBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(dA, packedBytes, AP, packedBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMalloc(&dY, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasStatus_t ret = aclblasStpmv_legacy(
        handle, static_cast<aclblasFillMode>(uplo), static_cast<aclblasOperation>(trans),
        static_cast<aclblasDiagType>(diag), n64, static_cast<const float*>(dA), static_cast<const float*>(dX),
        static_cast<float*>(dY), incx64);

    if (ret != ACLBLAS_STATUS_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        aclrtFree(dY);
        return ret;
    }

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        aclrtFree(dY);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    aclRet = aclrtMemcpy(x, xBytes, dY, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        aclrtFree(dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclrtFree(dA);
    aclrtFree(dX);
    aclrtFree(dY);

    return ret;
}
