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

inline std::unique_ptr<DeviceBuffer> sspmvTryAllocAndCopy(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr)
        return nullptr;
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

inline aclblasStatus_t aclblasSspmv_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* ap,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n <= 0) {
        return aclblasSspmv(handle, uplo, n, alpha, ap, x, incx, beta, y, incy);
    }

    const size_t apBytes = static_cast<size_t>(n) * (n + 1) / 2 * sizeof(float);
    const size_t xBytes = static_cast<size_t>((n - 1) * std::abs(incx) + 1) * sizeof(float);
    const size_t yBytes = static_cast<size_t>((n - 1) * std::abs(incy) + 1) * sizeof(float);

    auto dAP = sspmvTryAllocAndCopy(ap, apBytes);
    auto dX = sspmvTryAllocAndCopy(x, xBytes);
    auto dY = sspmvTryAllocAndCopy(y, yBytes);

    aclblasStatus_t ret = aclblasSspmv(
        handle, uplo, n, alpha, dAP ? static_cast<const float*>(dAP->ptr()) : nullptr,
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
