/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCOPY_NPU_H
#define SCOPY_NPU_H

#include <cstdint>
#include <cstdlib>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// H2D alloc + memcpy for one buffer. On failure, frees *otherPtr for rollback.
inline aclblasStatus_t scopy_h2d_copy(const void* hostPtr, size_t bytes, void** dPtr, void** otherPtr)
{
    if (hostPtr == nullptr)
        return ACLBLAS_STATUS_SUCCESS;
    aclError aclRet = aclrtMalloc(dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        if (*otherPtr)
            aclrtFree(*otherPtr);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(*dPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(*dPtr);
        *dPtr = nullptr;
        if (*otherPtr)
            aclrtFree(*otherPtr);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// NPU wrapper — same signature as aclblasScopy.
// nullptr inputs are passed through without device allocation (for error-path testing).
// n<=0 cases are passed through directly to the API (returns SUCCESS without further validation).
inline aclblasStatus_t aclblasScopy_npu(aclblasHandle_t handle, int n, const float* x, int incx, float* y, int incy)
{
    if (handle == nullptr || n <= 0) {
        return aclblasScopy(handle, n, x, incx, y, incy);
    }

    const int absIncX = std::abs(incx);
    const int absIncY = std::abs(incy);
    const size_t xBytes = (static_cast<size_t>(absIncX) * (n - 1) + 1) * sizeof(float);
    const size_t yBytes = (static_cast<size_t>(absIncY) * (n - 1) + 1) * sizeof(float);

    void* dX = nullptr;
    void* dY = nullptr;
    aclblasStatus_t st = scopy_h2d_copy(x, xBytes, &dX, &dY);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    st = scopy_h2d_copy(y, yBytes, &dY, &dX);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;

    aclblasStatus_t ret = aclblasScopy(handle, n, static_cast<const float*>(dX), incx, static_cast<float*>(dY), incy);

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        if (dX)
            aclrtFree(dX);
        if (dY)
            aclrtFree(dY);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    // x is read-only (const input), no need to copy back
    if (y != nullptr) {
        aclError copyRet = aclrtMemcpy(y, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (copyRet != ACL_SUCCESS) {
            if (dX)
                aclrtFree(dX);
            if (dY)
                aclrtFree(dY);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    if (dX)
        aclrtFree(dX);
    if (dY)
        aclrtFree(dY);
    return ret;
}

#endif // SCOPY_NPU_H
