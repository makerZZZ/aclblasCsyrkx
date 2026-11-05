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

inline std::unique_ptr<DeviceBuffer> ssymvTryAllocAndCopy(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr)
        return nullptr;
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

inline aclblasStatus_t aclblasSsymv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n <= 0) {
        return aclblasSsymv(handle, uplo, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    const size_t aBytes = static_cast<size_t>(lda) * n * sizeof(float);
    const size_t xBytes = static_cast<size_t>((n - 1) * std::abs(incx) + 1) * sizeof(float);
    const size_t yBytes = static_cast<size_t>((n - 1) * std::abs(incy) + 1) * sizeof(float);

    auto dA = ssymvTryAllocAndCopy(a, aBytes);
    auto dX = ssymvTryAllocAndCopy(x, xBytes);
    auto dY = ssymvTryAllocAndCopy(y, yBytes);

    aclblasStatus_t ret = aclblasSsymv(
        handle, uplo, n, alpha, dA ? static_cast<const float*>(dA->ptr()) : nullptr, lda,
        dX ? static_cast<const float*>(dX->ptr()) : nullptr, incx, beta,
        dY ? static_cast<float*>(dY->ptr()) : nullptr, incy);

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;

    if (y != nullptr && dY) {
        dY->copyToHost(y, yBytes);
    }

    return ret;
}
