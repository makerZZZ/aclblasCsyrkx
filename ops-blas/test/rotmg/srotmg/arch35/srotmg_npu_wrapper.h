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

/*!
 * \brief NPU wrapper for aclblasSrotmg — device pointer path.
 *
 * Allocates device memory for all five scalar arguments, copies input values
 * from host to device, calls aclblasSrotmg with device pointers, then copies
 * results back to host.
 */
inline aclblasStatus_t aclblasSrotmg_npu(
    aclblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param)
{
    if (handle == nullptr || d1 == nullptr || d2 == nullptr ||
        x1 == nullptr || y1 == nullptr || param == nullptr) {
        return aclblasSrotmg(handle, d1, d2, x1, y1, param);
    }

    float hD1 = *d1, hD2 = *d2, hX1 = *x1, hY1 = *y1;

    void* dD1 = nullptr;
    void* dD2 = nullptr;
    void* dX1 = nullptr;
    void* dY1 = nullptr;
    void* dParam = nullptr;

    auto freeAll = [&]() {
        if (dD1) aclrtFree(dD1);
        if (dD2) aclrtFree(dD2);
        if (dX1) aclrtFree(dX1);
        if (dY1) aclrtFree(dY1);
        if (dParam) aclrtFree(dParam);
    };

    if (aclrtMalloc(&dD1, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { freeAll(); return ACLBLAS_STATUS_ALLOC_FAILED; }
    if (aclrtMalloc(&dD2, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { freeAll(); return ACLBLAS_STATUS_ALLOC_FAILED; }
    if (aclrtMalloc(&dX1, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { freeAll(); return ACLBLAS_STATUS_ALLOC_FAILED; }
    if (aclrtMalloc(&dY1, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { freeAll(); return ACLBLAS_STATUS_ALLOC_FAILED; }
    if (aclrtMalloc(&dParam, 5 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { freeAll(); return ACLBLAS_STATUS_ALLOC_FAILED; }

    aclError aclRet;
    aclRet  = aclrtMemcpy(dD1, sizeof(float), &hD1, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclRet |= aclrtMemcpy(dD2, sizeof(float), &hD2, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclRet |= aclrtMemcpy(dX1, sizeof(float), &hX1, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclRet |= aclrtMemcpy(dY1, sizeof(float), &hY1, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasStatus_t ret = ACLBLAS_STATUS_INTERNAL_ERROR;
    if (aclRet == ACL_SUCCESS) {
        ret = aclblasSrotmg(
            handle,
            static_cast<float*>(dD1), static_cast<float*>(dD2),
            static_cast<float*>(dX1), static_cast<const float*>(dY1),
            static_cast<float*>(dParam));

        if (ret == ACLBLAS_STATUS_SUCCESS) {
            if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
                ret = ACLBLAS_STATUS_INTERNAL_ERROR;
            } else {
                aclError d2hRet  = aclrtMemcpy(d1, sizeof(float), dD1, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
                d2hRet |= aclrtMemcpy(d2, sizeof(float), dD2, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
                d2hRet |= aclrtMemcpy(x1, sizeof(float), dX1, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
                d2hRet |= aclrtMemcpy(param, 5 * sizeof(float), dParam, 5 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
                if (d2hRet != ACL_SUCCESS) {
                    ret = ACLBLAS_STATUS_INTERNAL_ERROR;
                }
            }
        }
    }

    freeAll();
    return ret;
}
