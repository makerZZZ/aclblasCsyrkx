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
#include <cstdint>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

inline std::unique_ptr<DeviceBuffer> tryAllocAndCopyIcamin(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr)
        return nullptr;
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

inline aclblasStatus_t aclblasIcamin_npu(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n <= 0 || incx <= 0) {
        return aclblasIcamin(handle, n, x, incx, result);
    }

    const size_t dataBytes =
        (static_cast<size_t>(n - 1) * static_cast<size_t>(std::abs(incx)) + 1) * sizeof(aclblasComplex);
    const size_t resultBytes = sizeof(int32_t);

    auto dX = tryAllocAndCopyIcamin(x, dataBytes);
    auto dResult = std::make_unique<DeviceBuffer>(resultBytes);

    aclblasStatus_t ret = aclblasIcamin(
        handle, n, dX ? static_cast<const aclblasComplex*>(dX->ptr()) : nullptr, incx,
        dResult ? static_cast<int32_t*>(dResult->ptr()) : nullptr);

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS)
        return ACLBLAS_STATUS_INTERNAL_ERROR;

    if (ret == ACLBLAS_STATUS_SUCCESS && result != nullptr && dResult) {
        dResult->copyToHost(result, resultBytes);
    }

    return ret;
}

namespace {

inline aclblasStatus_t IcaminRepeatLaunches(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result, int times)
{
    for (int i = 0; i < times; ++i) {
        const aclblasStatus_t st = aclblasIcamin(handle, n, x, incx, result);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            return st;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline void IcaminDestroyEvents(aclrtEvent evStart, aclrtEvent evEnd)
{
    if (evStart != nullptr) {
        aclrtDestroyEvent(evStart);
    }
    if (evEnd != nullptr) {
        aclrtDestroyEvent(evEnd);
    }
}

inline aclblasStatus_t IcaminTimeLaunches(
    aclblasHandle_t handle, aclrtStream stream, int n, const aclblasComplex* x, int incx, int* result, int iters,
    double* avgUs)
{
    aclrtEvent evStart = nullptr;
    aclrtEvent evEnd = nullptr;
    if (aclrtCreateEvent(&evStart) != ACL_SUCCESS || aclrtCreateEvent(&evEnd) != ACL_SUCCESS) {
        IcaminDestroyEvents(evStart, evEnd);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclrtRecordEvent(evStart, stream);
    const aclblasStatus_t st = IcaminRepeatLaunches(handle, n, x, incx, result, iters);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        IcaminDestroyEvents(evStart, evEnd);
        return st;
    }
    aclrtRecordEvent(evEnd, stream);
    if (aclrtSynchronizeEvent(evEnd) != ACL_SUCCESS) {
        IcaminDestroyEvents(evStart, evEnd);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    float ms = 0.0f;
    aclrtEventElapsedTime(&ms, evStart, evEnd);
    IcaminDestroyEvents(evStart, evEnd);
    *avgUs = (static_cast<double>(ms) * 1000.0) / static_cast<double>(iters);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

// Perf helper for TC_PF_* only: time repeated aclblasIcamin on device-resident buffers.
inline aclblasStatus_t aclblasIcamin_npu_bench(
    aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result, int warmup, int iters, double* avgUs)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n <= 0 || incx <= 0 || warmup < 0 || iters <= 0 || avgUs == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    const size_t dataBytes =
        (static_cast<size_t>(n - 1) * static_cast<size_t>(std::abs(incx)) + 1) * sizeof(aclblasComplex);
    auto dX = tryAllocAndCopyIcamin(x, dataBytes);
    auto dResult = std::make_unique<DeviceBuffer>(sizeof(int32_t));
    if (!dX || !dResult) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    const aclblasComplex* dxPtr = static_cast<const aclblasComplex*>(dX->ptr());
    int32_t* drPtr = static_cast<int32_t*>(dResult->ptr());
    const aclblasStatus_t warmSt = IcaminRepeatLaunches(handle, n, dxPtr, incx, drPtr, warmup);
    if (warmSt != ACLBLAS_STATUS_SUCCESS) {
        return warmSt;
    }

    aclrtStream stream = nullptr;
    if (aclblasGetStream(handle, &stream) != ACLBLAS_STATUS_SUCCESS || stream == nullptr) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    const aclblasStatus_t timed = IcaminTimeLaunches(handle, stream, n, dxPtr, incx, drPtr, iters, avgUs);
    if (timed != ACLBLAS_STATUS_SUCCESS) {
        return timed;
    }
    if (result != nullptr) {
        dResult->copyToHost(result, sizeof(int32_t));
    }
    return ACLBLAS_STATUS_SUCCESS;
}
