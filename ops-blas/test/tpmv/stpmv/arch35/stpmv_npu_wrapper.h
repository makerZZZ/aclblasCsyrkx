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

// NPU wrapper — same signature as aclblasStpmv (in-place: x is both input and output).
// nullptr inputs are passed through without device allocation (for error-path testing).
// n <= 0: fast return via direct API call.
inline aclblasStatus_t aclblasStpmv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* AP, float* x, int incx)
{
    // Fast path: null handle or n <= 0, pass through directly
    if (handle == nullptr || n <= 0) {
        return aclblasStpmv(handle, uplo, trans, diag, n, AP, x, incx);
    }

    const int absIncx = std::abs(incx);
    const size_t apSize = static_cast<size_t>(n) * (n + 1) / 2;
    const size_t xSize = static_cast<size_t>((n - 1) * absIncx + 1);

    const size_t apBytes = apSize * sizeof(float);
    const size_t xBytes = xSize * sizeof(float);

    void* dAP = nullptr;
    void* dX = nullptr;
    aclError aclRet;

    // Allocate and copy AP
    if (AP != nullptr) {
        aclRet = aclrtMalloc(&dAP, apBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS)
            return ACLBLAS_STATUS_ALLOC_FAILED;
        aclRet = aclrtMemcpy(dAP, apBytes, AP, apBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            aclrtFree(dAP);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    // Allocate and copy x
    if (x != nullptr) {
        aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            if (dAP)
                aclrtFree(dAP);
            return ACLBLAS_STATUS_ALLOC_FAILED;
        }
        aclRet = aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            if (dAP)
                aclrtFree(dAP);
            aclrtFree(dX);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    // Call kernel (in-place: x is both input and output)
    aclblasStatus_t ret =
        aclblasStpmv(handle, uplo, trans, diag, n, static_cast<const float*>(dAP), static_cast<float*>(dX), incx);

    // Synchronize and copy back x
    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        if (dAP)
            aclrtFree(dAP);
        if (dX)
            aclrtFree(dX);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (x != nullptr && dX != nullptr) {
        aclError d2hRet = aclrtMemcpy(x, xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (d2hRet != ACL_SUCCESS) {
            if (dAP)
                aclrtFree(dAP);
            if (dX)
                aclrtFree(dX);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    // Free device memory
    if (dAP)
        aclrtFree(dAP);
    if (dX)
        aclrtFree(dX);

    return ret;
}

