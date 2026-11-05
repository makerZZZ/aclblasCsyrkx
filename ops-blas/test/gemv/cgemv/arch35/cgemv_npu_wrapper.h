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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "device.h"

// cgemv npu wrapper: allocate device buffer and copy host data (nullable)
inline std::unique_ptr<DeviceBuffer> cgemvTryAllocAndCopy(const void* hostPtr, size_t bytes)
{
    if (hostPtr == nullptr) {
        return nullptr;
    }
    auto buf = std::make_unique<DeviceBuffer>(bytes);
    buf->copyFromHost(hostPtr, bytes);
    return buf;
}

inline uint64_t cgemvAbsStride(int inc)
{
    return (inc < 0) ? static_cast<uint64_t>(-static_cast<int64_t>(inc)) : static_cast<uint64_t>(inc);
}

struct CgemvPerfMeasurement {
    aclblasStatus_t status = ACLBLAS_STATUS_INTERNAL_ERROR;
    double avgUs = 0.0;
};

template <typename Launch>
inline aclblasStatus_t cgemvLaunchAndSynchronize(const Launch& launch, aclrtStream stream, int iterations)
{
    for (int i = 0; i < iterations; ++i) {
        aclblasStatus_t status = launch();
        if (status != ACLBLAS_STATUS_SUCCESS) {
            (void)aclrtSynchronizeStream(stream);
            return status;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline CgemvPerfMeasurement cgemvAverageDuration(std::chrono::steady_clock::duration elapsed, int timedIters)
{
    CgemvPerfMeasurement result;
    const double iterations = static_cast<double>(timedIters);
    if (iterations <= 0.0) {
        result.status = ACLBLAS_STATUS_INVALID_VALUE;
        return result;
    }
    result.avgUs = std::chrono::duration<double, std::micro>(elapsed).count() / iterations;
    result.status = ACLBLAS_STATUS_SUCCESS;
    return result;
}

// Allocate and copy once, then reuse the same device buffers for every warmup
// and timed launch. Buffer construction/destruction and H2D/D2H copies stay
// outside the timed interval.
inline CgemvPerfMeasurement aclblasCgemv_npu_benchmark(
    aclblasHandle_t handle, aclrtStream stream, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta,
    const aclblasComplex* y, int incy, int warmup, int timedIters)
{
    CgemvPerfMeasurement result;
    if (timedIters <= 0) {
        result.status = ACLBLAS_STATUS_INVALID_VALUE;
        return result;
    }
    if (handle == nullptr || stream == nullptr || m <= 0 || n <= 0 || warmup < 0) {
        result.status = ACLBLAS_STATUS_INVALID_VALUE;
        return result;
    }

    const int allocLda = std::max(lda, 1);
    const int xCount = (trans == ACLBLAS_OP_N) ? n : m;
    const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
    const size_t aBytes = static_cast<size_t>(allocLda) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    const size_t xBytes = (static_cast<size_t>(xCount - 1) * cgemvAbsStride(incx) + 1) * sizeof(aclblasComplex);
    const size_t yBytes = (static_cast<size_t>(yCount - 1) * cgemvAbsStride(incy) + 1) * sizeof(aclblasComplex);

    auto dA = cgemvTryAllocAndCopy(a, aBytes);
    auto dX = cgemvTryAllocAndCopy(x, xBytes);
    auto dY = cgemvTryAllocAndCopy(y, yBytes);
    const auto* dAPtr = dA ? static_cast<const aclblasComplex*>(dA->ptr()) : nullptr;
    const auto* dXPtr = dX ? static_cast<const aclblasComplex*>(dX->ptr()) : nullptr;
    auto* dYPtr = dY ? static_cast<aclblasComplex*>(dY->ptr()) : nullptr;

    auto launch = [&]() {
        return aclblasCgemv(handle, trans, m, n, alpha, dAPtr, lda, dXPtr, incx, beta, dYPtr, incy);
    };
    result.status = cgemvLaunchAndSynchronize(launch, stream, warmup);
    if (result.status != ACLBLAS_STATUS_SUCCESS) {
        return result;
    }

    const auto start = std::chrono::steady_clock::now();
    result.status = cgemvLaunchAndSynchronize(launch, stream, timedIters);
    if (result.status != ACLBLAS_STATUS_SUCCESS) {
        return result;
    }
    const auto end = std::chrono::steady_clock::now();
    return cgemvAverageDuration(end - start, timedIters);
}

inline aclblasStatus_t aclblasCgemv_npu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (m <= 0 || n <= 0) {
        // Validation / quick-return path: no kernel launch, host pointers are safe
        return aclblasCgemv(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    const int allocLda = std::max(lda, 1);
    const int xCount = (trans == ACLBLAS_OP_N) ? n : m;
    const int yCount = (trans == ACLBLAS_OP_N) ? m : n;
    const size_t aBytes = static_cast<size_t>(allocLda) * std::max(1, n) * sizeof(aclblasComplex);
    const size_t xBytes = (static_cast<size_t>(xCount - 1) * cgemvAbsStride(incx) + 1) * sizeof(aclblasComplex);
    const size_t yBytes = (static_cast<size_t>(yCount - 1) * cgemvAbsStride(incy) + 1) * sizeof(aclblasComplex);

    auto dA = cgemvTryAllocAndCopy(a, aBytes);
    auto dX = cgemvTryAllocAndCopy(x, xBytes);
    auto dY = cgemvTryAllocAndCopy(y, yBytes);

    aclblasStatus_t ret = aclblasCgemv(
        handle, trans, m, n, alpha, dA ? static_cast<const aclblasComplex*>(dA->ptr()) : nullptr, lda,
        dX ? static_cast<const aclblasComplex*>(dX->ptr()) : nullptr, incx, beta,
        dY ? static_cast<aclblasComplex*>(dY->ptr()) : nullptr, incy);

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    if (y != nullptr && dY) {
        dY->copyToHost(y, yBytes);
    }

    return ret;
}
