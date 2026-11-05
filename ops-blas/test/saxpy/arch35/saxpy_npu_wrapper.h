/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SAXPY_NPU_H
#define SAXPY_NPU_H

#include <cstdint>
#include <algorithm>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblasSaxpy_npu(
    aclblasHandle_t handle, int n, const float* alpha, float* x, int incx, float* y, int incy)
{
    if (handle == nullptr || n <= 0 || x == nullptr || y == nullptr) {
        return aclblasSaxpy(handle, n, alpha, x, incx, y, incy);
    }

    int absIncX = std::abs(incx);
    int absIncY = std::abs(incy);
    size_t xElements = static_cast<size_t>((n - 1) * absIncX + 1);
    size_t yElements = static_cast<size_t>((n - 1) * absIncY + 1);
    const size_t xBytes = xElements * sizeof(float);
    const size_t yBytes = yElements * sizeof(float);

    void* dX = nullptr;
    void* dY = nullptr;
    aclError aclRet;

    aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    aclRet = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dX);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dX);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(dY, yBytes, y, yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dX);
        aclrtFree(dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclblasStatus_t ret = aclblasSaxpy(handle, n, alpha, static_cast<float*>(dX), incx, static_cast<float*>(dY), incy);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        aclrtFree(dX);
        aclrtFree(dY);
        return ret;
    }

    aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dX);
        aclrtFree(dY);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    aclRet = aclrtMemcpy(y, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dX);
        aclrtFree(dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclrtFree(dX);
    aclrtFree(dY);

    return ret;
}

#endif
