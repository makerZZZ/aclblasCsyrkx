/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSPMV_NPU_H
#define SSPMV_NPU_H

#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblasSspmv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x,
    int incx, const float* beta, float* y, int incy)
{
    if (handle == nullptr || n <= 0 || AP == nullptr || x == nullptr || y == nullptr || alpha == nullptr ||
        beta == nullptr) {
        return aclblasSspmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
    }

    const size_t nSize = static_cast<size_t>(n);
    const size_t packedEleNum = (nSize * (nSize + 3U) - 2U) / 2U;
    const size_t packedBytes = packedEleNum * sizeof(float);
    const size_t vecBytes = nSize * sizeof(float);

    void* dAP = nullptr;
    void* dX = nullptr;
    void* dY = nullptr;
    aclError aclRet;

    aclRet = aclrtMalloc(&dAP, packedBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) return ACLBLAS_STATUS_ALLOC_FAILED;
    aclRet = aclrtMemcpy(dAP, packedBytes, AP, packedBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { aclrtFree(dAP); return ACLBLAS_STATUS_INTERNAL_ERROR; }

    aclRet = aclrtMalloc(&dX, vecBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) { aclrtFree(dAP); return ACLBLAS_STATUS_ALLOC_FAILED; }
    aclRet = aclrtMemcpy(dX, vecBytes, x, vecBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { aclrtFree(dAP); aclrtFree(dX); return ACLBLAS_STATUS_INTERNAL_ERROR; }

    aclRet = aclrtMalloc(&dY, vecBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) { aclrtFree(dAP); aclrtFree(dX); return ACLBLAS_STATUS_ALLOC_FAILED; }
    aclRet = aclrtMemcpy(dY, vecBytes, y, vecBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) { aclrtFree(dAP); aclrtFree(dX); aclrtFree(dY); return ACLBLAS_STATUS_INTERNAL_ERROR; }

    aclblasStatus_t ret = aclblasSspmv(
        handle, uplo, n, alpha, static_cast<const float*>(dAP), static_cast<const float*>(dX), incx, beta,
        static_cast<float*>(dY), incy);

    if (ret != ACLBLAS_STATUS_SUCCESS) { aclrtFree(dAP); aclrtFree(dX); aclrtFree(dY); return ret; }

    aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) { aclrtFree(dAP); aclrtFree(dX); aclrtFree(dY); return ACLBLAS_STATUS_EXECUTION_FAILED; }

    aclRet = aclrtMemcpy(y, vecBytes, dY, vecBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) { aclrtFree(dAP); aclrtFree(dX); aclrtFree(dY); return ACLBLAS_STATUS_INTERNAL_ERROR; }

    aclrtFree(dAP);
    aclrtFree(dX);
    aclrtFree(dY);

    return ret;
}

#endif
