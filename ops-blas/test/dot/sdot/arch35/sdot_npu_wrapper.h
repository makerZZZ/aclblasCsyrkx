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

inline aclblasStatus_t aclblasSdot_npu(
    aclblasHandle_t handle, int n,
    const float* x, int incx, const float* y, int incy, float* result)
{
    if (handle == nullptr || n <= 0) {
        return aclblasSdot(handle, n, x, incx, y, incy, result);
    }

    const int absIncx = std::abs(incx);
    const int absIncy = std::abs(incy);
    const size_t xBytes = (n > 0) ? static_cast<size_t>((n - 1) * absIncx + 1) * sizeof(float) : sizeof(float);
    const size_t yBytes = (n > 0) ? static_cast<size_t>((n - 1) * absIncy + 1) * sizeof(float) : sizeof(float);

    void* dX = nullptr;
    void* dY = nullptr;
    void* dResult = nullptr;
    aclError aclRet;

    if (x != nullptr) {
        aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
        aclRet = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { aclrtFree(dX); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }
    if (y != nullptr) {
        aclRet = aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) { if (dX) aclrtFree(dX); return ACLBLAS_STATUS_ALLOC_FAILED; }
        aclRet = aclrtMemcpy(dY, yBytes, y, yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) { aclrtFree(dX); aclrtFree(dY); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }
    aclRet = aclrtMalloc(&dResult, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        if (dX) aclrtFree(dX);
        if (dY) aclrtFree(dY);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasStatus_t ret = aclblasSdot(
        handle, n, static_cast<const float*>(dX), incx,
        static_cast<const float*>(dY), incy, static_cast<float*>(dResult));

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        aclrtFree(dX);
        aclrtFree(dY);
        aclrtFree(dResult);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (ret == ACLBLAS_STATUS_SUCCESS && result != nullptr) {
        aclrtMemcpy(result, sizeof(float), dResult, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    }

    aclrtFree(dX);
    aclrtFree(dY);
    aclrtFree(dResult);
    return ret;
}

