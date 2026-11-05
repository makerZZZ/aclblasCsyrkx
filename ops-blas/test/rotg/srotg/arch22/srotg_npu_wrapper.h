/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SROTG_NPU_WRAPPER_H
#define SROTG_NPU_WRAPPER_H

#include <cstddef>
#include <cstdint>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

constexpr std::size_t SROTG_VALUE_COUNT = 4;
constexpr std::size_t SROTG_VALUE_BYTES = SROTG_VALUE_COUNT * sizeof(float);

inline float *GetSrotgDeviceValue(const DeviceBuffer &buffer, std::size_t index)
{
    return buffer.floatPtr() + index;
}

inline aclblasStatus_t CopySrotgInputsToDevice(
    float *a, float *b, float *c, float *s, DeviceBuffer &buffer)
{
    if (a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const float hostValues[SROTG_VALUE_COUNT] = {*a, *b, *c, *s};
    try {
        buffer.copyFromHost(hostValues, SROTG_VALUE_BYTES);
    } catch (...) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t CopySrotgOutputsToHost(
    float *a, float *b, float *c, float *s, DeviceBuffer &buffer)
{
    float hostValues[SROTG_VALUE_COUNT] = {};
    try {
        buffer.copyToHost(hostValues, SROTG_VALUE_BYTES);
    } catch (...) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    *a = hostValues[0];
    *b = hostValues[1];
    *c = hostValues[2];
    *s = hostValues[3];
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t SynchronizeSrotgStream(aclblasHandle_t handle)
{
    aclrtStream stream = nullptr;
    aclblasStatus_t ret = aclblasGetStream(handle, &stream);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    const aclError aclRet = aclrtSynchronizeStream(stream);
    return (aclRet == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

inline aclblasStatus_t aclblasSrotg_npu(
    aclblasHandle_t handle, float *a, float *b, float *c, float *s)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    std::unique_ptr<DeviceBuffer> values;
    try {
        values = std::make_unique<DeviceBuffer>(SROTG_VALUE_BYTES);
    } catch (...) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclblasStatus_t ret = CopySrotgInputsToDevice(a, b, c, s, *values);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }

    ret = aclblasSrotg(
        handle, GetSrotgDeviceValue(*values, 0), GetSrotgDeviceValue(*values, 1),
        GetSrotgDeviceValue(*values, 2), GetSrotgDeviceValue(*values, 3));

    const aclblasStatus_t syncRet = SynchronizeSrotgStream(handle);
    if (syncRet != ACLBLAS_STATUS_SUCCESS) {
        return syncRet;
    }
    const aclblasStatus_t copyRet = CopySrotgOutputsToHost(a, b, c, s, *values);
    return (ret == ACLBLAS_STATUS_SUCCESS) ? copyRet : ret;
}

#endif // SROTG_NPU_WRAPPER_H
