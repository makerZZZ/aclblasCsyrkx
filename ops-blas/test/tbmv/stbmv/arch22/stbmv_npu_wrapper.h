/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STBMV_NPU_H
#define STBMV_NPU_H

#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblasStbmv_npu(
    aclblasHandle_t handle, aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag, const float* a,
    const int64_t lda, const float* x, float* y, const int64_t n, const int64_t k, const int64_t incx)
{
    if (handle == nullptr || n <= 0 || a == nullptr || x == nullptr || y == nullptr) {
        return aclblasStbmv_legacy(handle, uplo, trans, diag, a, lda, x, y, n, k, incx);
    }

    const size_t matBytes = static_cast<size_t>(k + 1) * static_cast<size_t>(lda) * sizeof(float);
    const size_t vecBytes = static_cast<size_t>(n) * sizeof(float);

    void* dA = nullptr;
    void* dX = nullptr;
    void* dY = nullptr;
    aclError aclRet;

    aclRet = aclrtMalloc(&dA, matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_ALLOC_FAILED;
    aclRet = aclrtMemcpy(dA, matBytes, a, matBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMalloc(&dX, vecBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(dX, vecBytes, x, vecBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMalloc(&dY, vecBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    {
        std::vector<float> yZero(static_cast<size_t>(n), 0.0f);
        aclRet = aclrtMemcpy(dY, vecBytes, yZero.data(), vecBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        aclrtFree(dY);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclblasStatus_t ret = aclblasStbmv_legacy(
        handle, uplo, trans, diag, static_cast<const float*>(dA), lda, static_cast<const float*>(dX),
        static_cast<float*>(dY), n, k, incx);

    if (ret != ACLBLAS_STATUS_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        aclrtFree(dY);
        return ret;
    }

    aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        aclrtFree(dY);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    aclRet = aclrtMemcpy(y, vecBytes, dY, vecBytes, ACL_MEMCPY_DEVICE_TO_HOST);
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

#endif
