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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

inline std::unique_ptr<DeviceBuffer> tryAllocAndCopyIsamax(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr)
        return nullptr;
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

inline aclblasStatus_t aclblasIsamax_npu(aclblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n <= 0 || incx <= 0) {
        return aclblasIsamax(handle, n, x, incx, result);
    }

    const size_t dataBytes = (static_cast<size_t>(n - 1) * static_cast<size_t>(std::abs(incx)) + 1) * sizeof(float);
    const size_t resultBytes = sizeof(int32_t);

    auto dX = tryAllocAndCopyIsamax(x, dataBytes);
    auto dResult = std::make_unique<DeviceBuffer>(resultBytes);

    aclblasStatus_t ret = aclblasIsamax(
        handle, n, dX ? static_cast<const float*>(dX->ptr()) : nullptr, incx,
        dResult ? static_cast<int32_t*>(dResult->ptr()) : nullptr);

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;

    if (ret == ACLBLAS_STATUS_SUCCESS && result != nullptr && dResult) {
        dResult->copyToHost(result, resultBytes);
    }

    return ret;
}
