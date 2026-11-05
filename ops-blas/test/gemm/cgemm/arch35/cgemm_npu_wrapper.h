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
#include <cstddef>

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct CgemmDeviceBuffers {
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
};

inline void FreeCgemmDeviceBuffers(CgemmDeviceBuffers& buffers)
{
    if (buffers.a != nullptr) {
        aclrtFree(buffers.a);
    }
    if (buffers.b != nullptr) {
        aclrtFree(buffers.b);
    }
    if (buffers.c != nullptr) {
        aclrtFree(buffers.c);
    }
}

inline aclblasStatus_t CgemmAllocAndCopy(void** device, const void* host, size_t bytes)
{
    *device = nullptr;
    if (host == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(*device);
        *device = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t AllocateCgemmDeviceBuffers(
    CgemmDeviceBuffers& buffers, int k, const aclblasComplex* a, const aclblasComplex* b, aclblasComplex* c,
    size_t aBytes, size_t bBytes, size_t cBytes)
{
    aclblasStatus_t status = ACLBLAS_STATUS_SUCCESS;
    if (a != nullptr && k > 0) {
        status = CgemmAllocAndCopy(&buffers.a, a, aBytes);
    }
    if (status == ACLBLAS_STATUS_SUCCESS && b != nullptr && k > 0) {
        status = CgemmAllocAndCopy(&buffers.b, b, bBytes);
    }
    if (status == ACLBLAS_STATUS_SUCCESS && c != nullptr) {
        status = CgemmAllocAndCopy(&buffers.c, c, cBytes);
    }
    if (status != ACLBLAS_STATUS_SUCCESS) {
        FreeCgemmDeviceBuffers(buffers);
    }
    return status;
}

inline aclblasStatus_t CgemmSyncAndCopyBack(
    CgemmDeviceBuffers& buffers, aclblasComplex* c, size_t cBytes, aclblasStatus_t status)
{
    if (status == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeDevice() != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (status == ACLBLAS_STATUS_SUCCESS && buffers.c != nullptr && c != nullptr &&
        aclrtMemcpy(c, cBytes, buffers.c, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    FreeCgemmDeviceBuffers(buffers);
    return status;
}

inline bool CgemmHasInvalidParameters(
    aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb, const aclblasComplex* beta,
    const aclblasComplex* c, int ldc)
{
    const bool validTransA = transA == ACLBLAS_OP_N || transA == ACLBLAS_OP_T || transA == ACLBLAS_OP_C;
    const bool validTransB = transB == ACLBLAS_OP_N || transB == ACLBLAS_OP_T || transB == ACLBLAS_OP_C;
    if (!validTransA || !validTransB || m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr) {
        return true;
    }
    const int minimumLda = transA == ACLBLAS_OP_N ? std::max(1, m) : std::max(1, k);
    const int minimumLdb = transB == ACLBLAS_OP_N ? std::max(1, k) : std::max(1, n);
    if (lda < minimumLda || ldb < minimumLdb || ldc < std::max(1, m)) {
        return true;
    }
    return c == nullptr || (k > 0 && (a == nullptr || b == nullptr));
}

inline aclblasStatus_t aclblasCgemm_npu(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* a, int lda, const aclblasComplex* b, int ldb,
    const aclblasComplex* beta, aclblasComplex* c, int ldc)
{
    if (handle == nullptr || m <= 0 || n <= 0 ||
        CgemmHasInvalidParameters(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)) {
        return aclblasCgemm(handle, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    const int aColumns = transA == ACLBLAS_OP_N ? k : m;
    const int bColumns = transB == ACLBLAS_OP_N ? n : k;
    const size_t aBytes =
        static_cast<size_t>(std::max(1, lda)) * static_cast<size_t>(std::max(1, aColumns)) * sizeof(*a);
    const size_t bBytes =
        static_cast<size_t>(std::max(1, ldb)) * static_cast<size_t>(std::max(1, bColumns)) * sizeof(*b);
    const size_t cBytes = static_cast<size_t>(std::max(1, ldc)) * std::max(1, n) * sizeof(*c);

    CgemmDeviceBuffers buffers;
    aclblasStatus_t status = AllocateCgemmDeviceBuffers(buffers, k, a, b, c, aBytes, bBytes, cBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    status = aclblasCgemm(
        handle, transA, transB, m, n, k, alpha,
        buffers.a != nullptr ? static_cast<const aclblasComplex*>(buffers.a) : a, lda,
        buffers.b != nullptr ? static_cast<const aclblasComplex*>(buffers.b) : b, ldb, beta,
        buffers.c != nullptr ? static_cast<aclblasComplex*>(buffers.c) : c, ldc);
    return CgemmSyncAndCopyBack(buffers, c, cBytes, status);
}
