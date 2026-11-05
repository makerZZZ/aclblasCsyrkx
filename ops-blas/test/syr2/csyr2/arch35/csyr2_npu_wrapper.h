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

#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct Csyr2DeviceBuffers {
    void* x = nullptr;
    void* y = nullptr;
    void* a = nullptr;

    ~Csyr2DeviceBuffers()
    {
        if (x != nullptr) {
            aclrtFree(x);
        }
        if (y != nullptr) {
            aclrtFree(y);
        }
        if (a != nullptr) {
            aclrtFree(a);
        }
    }
};

inline aclblasStatus_t Csyr2AllocAndCopy(const void* host, size_t bytes, void*& device)
{
    if (host == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        device = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline bool Csyr2CanExecute(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* x,
    int incx, const aclblasComplex* y, int incy, const aclblasComplex* a, int lda)
{
    return handle != nullptr && (uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER) && n > 0 && alpha != nullptr &&
           !(alpha->real == 0.0f && alpha->imag == 0.0f) && incx != 0 && incy != 0 && lda >= n && x != nullptr &&
           y != nullptr && a != nullptr;
}

// Host alpha is intentionally forwarded as host memory.  The public API reads
// this scalar during validation/tiling; only x, y and A are device buffers.
// Invalid and quick-return calls are passed through without any allocation so
// their precise API status and null-pointer contract remain observable.
inline aclblasStatus_t aclblasCsyr2_npu(
    aclblasHandle_t handle, aclrtStream stream, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha,
    const aclblasComplex* x, int incx, const aclblasComplex* y, int incy, aclblasComplex* a, int lda)
{
    if (!Csyr2CanExecute(handle, uplo, n, alpha, x, incx, y, incy, a, lda)) {
        return aclblasCsyr2(handle, uplo, n, alpha, x, incx, y, incy, a, lda);
    }

    const uint64_t xElements =
        1ULL + static_cast<uint64_t>(n - 1) * static_cast<uint64_t>(incx < 0 ? -static_cast<int64_t>(incx) : incx);
    const uint64_t yElements =
        1ULL + static_cast<uint64_t>(n - 1) * static_cast<uint64_t>(incy < 0 ? -static_cast<int64_t>(incy) : incy);
    const uint64_t aElements = static_cast<uint64_t>(lda) * static_cast<uint64_t>(n);
    const size_t xBytes = static_cast<size_t>(xElements * sizeof(aclblasComplex));
    const size_t yBytes = static_cast<size_t>(yElements * sizeof(aclblasComplex));
    const size_t aBytes = static_cast<size_t>(aElements * sizeof(aclblasComplex));

    Csyr2DeviceBuffers buffers;
    aclblasStatus_t status = Csyr2AllocAndCopy(x, xBytes, buffers.x);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = Csyr2AllocAndCopy(y, yBytes, buffers.y);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = Csyr2AllocAndCopy(a, aBytes, buffers.a);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    status = aclblasCsyr2(
        handle, uplo, n, alpha, static_cast<const aclblasComplex*>(buffers.x), incx,
        static_cast<const aclblasComplex*>(buffers.y), incy, static_cast<aclblasComplex*>(buffers.a), lda);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    if (aclrtMemcpy(a, aBytes, buffers.a, aBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
