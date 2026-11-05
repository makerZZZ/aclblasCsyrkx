/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCOPY_NPU_WRAPPER_H
#define CCOPY_NPU_WRAPPER_H

#include <cstddef>
#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t ccopy_h2d_copy(const void* host, size_t bytes, void** device)
{
    if (host == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    ret = aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*device);
        *device = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline bool ccopy_get_buffer_layout(
    int n, int increment, size_t offsetElements, size_t& offsetBytes, size_t& totalBytes)
{
    uint64_t absIncrement =
        increment > 0 ? static_cast<uint64_t>(increment) : static_cast<uint64_t>(-static_cast<int64_t>(increment));
    uint64_t spanElements = absIncrement * (static_cast<uint64_t>(n) - 1) + 1;
    constexpr size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(aclblasComplex);
    if (spanElements > maxElements) {
        return false;
    }
    size_t span = static_cast<size_t>(spanElements);
    if (offsetElements > maxElements - span) {
        return false;
    }
    offsetBytes = offsetElements * sizeof(aclblasComplex);
    totalBytes = (span + offsetElements) * sizeof(aclblasComplex);
    return true;
}

inline void ccopy_free_device_buffer(void*& device)
{
    if (device != nullptr) {
        aclrtFree(device);
        device = nullptr;
    }
}

inline aclblasStatus_t ccopy_execute_and_copy_back(
    aclblasHandle_t handle, int n, void* deviceX, size_t xOffsetBytes, int incx, void* deviceY, size_t yOffsetBytes,
    int incy, void* yHostBase, size_t yBytes)
{
    const auto* logicalDeviceX =
        deviceX == nullptr ? nullptr :
                             reinterpret_cast<const aclblasComplex*>(static_cast<uint8_t*>(deviceX) + xOffsetBytes);
    auto* logicalDeviceY =
        deviceY == nullptr ? nullptr : reinterpret_cast<aclblasComplex*>(static_cast<uint8_t*>(deviceY) + yOffsetBytes);
    aclblasStatus_t status = aclblasCcopy(handle, n, logicalDeviceX, incx, logicalDeviceY, incy);
    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (status != ACLBLAS_STATUS_SUCCESS || yHostBase == nullptr) {
        return status;
    }
    if (aclrtMemcpy(yHostBase, yBytes, deviceY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return status;
}

inline aclblasStatus_t aclblasCcopy_npu(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, aclblasComplex* y, int incy,
    size_t xOffsetElements = 0, size_t yOffsetElements = 0)
{
    if (handle == nullptr || n <= 0) {
        return aclblasCcopy(handle, n, x, incx, y, incy);
    }

    size_t xOffsetBytes = 0;
    size_t yOffsetBytes = 0;
    size_t xBytes = 0;
    size_t yBytes = 0;
    if (!ccopy_get_buffer_layout(n, incx, xOffsetElements, xOffsetBytes, xBytes)) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (!ccopy_get_buffer_layout(n, incy, yOffsetElements, yOffsetBytes, yBytes)) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    const void* xHostBase = x == nullptr ? nullptr : reinterpret_cast<const uint8_t*>(x) - xOffsetBytes;
    void* yHostBase = y == nullptr ? nullptr : reinterpret_cast<uint8_t*>(y) - yOffsetBytes;
    void* deviceX = nullptr;
    void* deviceY = nullptr;
    aclblasStatus_t status = ccopy_h2d_copy(xHostBase, xBytes, &deviceX);
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = ccopy_h2d_copy(yHostBase, yBytes, &deviceY);
    }
    if (status == ACLBLAS_STATUS_SUCCESS) {
        status = ccopy_execute_and_copy_back(
            handle, n, deviceX, xOffsetBytes, incx, deviceY, yOffsetBytes, incy, yHostBase, yBytes);
    }
    ccopy_free_device_buffer(deviceX);
    ccopy_free_device_buffer(deviceY);
    return status;
}

#endif // CCOPY_NPU_WRAPPER_H
