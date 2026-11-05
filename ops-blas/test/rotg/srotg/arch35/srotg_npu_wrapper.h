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

// NPU wrapper — 与 aclblasSrotg 签名一致。
// 任一指针为 nullptr 时直接透传给 API（用于 NullHandle / NullA/B/C/S 错误路径测试）；
// 全部非空时：分配 device 内存 → H2D 拷贝 a/b → 调用 ACL → D2H 拷贝 a/b/c/s。
// 资源清理统一收敛到 freeAll lambda，任意失败路径都不会泄漏 device 内存。
inline aclblasStatus_t aclblasSrotg_npu(
    aclblasHandle_t handle, float* a, float* b, float* c, float* s)
{
    if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        return aclblasSrotg(handle, a, b, c, s);
    }

    const size_t scalarBytes = sizeof(float);
    float hA = *a;
    float hB = *b;

    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    void* dS = nullptr;

    auto freeAll = [&]() {
        if (dA) aclrtFree(dA);
        if (dB) aclrtFree(dB);
        if (dC) aclrtFree(dC);
        if (dS) aclrtFree(dS);
    };

    if (aclrtMalloc(&dA, scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { freeAll(); return ACLBLAS_STATUS_ALLOC_FAILED; }
    if (aclrtMalloc(&dB, scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { freeAll(); return ACLBLAS_STATUS_ALLOC_FAILED; }
    if (aclrtMalloc(&dC, scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { freeAll(); return ACLBLAS_STATUS_ALLOC_FAILED; }
    if (aclrtMalloc(&dS, scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { freeAll(); return ACLBLAS_STATUS_ALLOC_FAILED; }

    if (aclrtMemcpy(dA, scalarBytes, &hA, scalarBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) { freeAll(); return ACLBLAS_STATUS_INTERNAL_ERROR; }
    if (aclrtMemcpy(dB, scalarBytes, &hB, scalarBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) { freeAll(); return ACLBLAS_STATUS_INTERNAL_ERROR; }

    aclblasStatus_t ret = aclblasSrotg(handle,
                                       static_cast<float*>(dA), static_cast<float*>(dB),
                                       static_cast<float*>(dC), static_cast<float*>(dS));
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        } else if (aclrtMemcpy(a, scalarBytes, dA, scalarBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                   aclrtMemcpy(b, scalarBytes, dB, scalarBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                   aclrtMemcpy(c, scalarBytes, dC, scalarBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                   aclrtMemcpy(s, scalarBytes, dS, scalarBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            ret = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    freeAll();
    return ret;
}
