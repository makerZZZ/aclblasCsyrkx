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

#include "acl/acl.h"
#include "cann_ops_blas.h"

struct CsyrkDeviceBuffers {
    void* dAlpha = nullptr;
    void* dA = nullptr;
    void* dBeta = nullptr;
    void* dC = nullptr;

    ~CsyrkDeviceBuffers()
    {
        if (dAlpha)
            aclrtFree(dAlpha);
        if (dA)
            aclrtFree(dA);
        if (dBeta)
            aclrtFree(dBeta);
        if (dC)
            aclrtFree(dC);
    }
};

static aclblasStatus_t CsyrkAllocAndCopy(void*& devPtr, const void* hostPtr, size_t bytes)
{
    aclError aclRet = aclrtMalloc(&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        devPtr = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Allocate+copy only when hostPtr is non-null and bytes > 0; leaves devPtr
// null otherwise so the caller can pass nullptr straight through to the API
// (null_alpha / null_beta / nullA / nullC cases exercise INVALID_VALUE paths).
static aclblasStatus_t CsyrkTryAllocAndCopy(const void* hostPtr, size_t bytes, void*& devPtr)
{
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return CsyrkAllocAndCopy(devPtr, hostPtr, bytes);
}

// NPU wrapper for aclblasCsyrk (complex64). alpha/beta are complex scalars
// (8 bytes each on device). When handle is null or n<=0 it forwards directly
// so the operator's own validation paths are exercised.
inline aclblasStatus_t aclblasCsyrk_npu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* beta, aclblasComplex* C, int ldc, bool nullAlpha = false,
    bool nullBeta = false)
{
    const aclblasComplex* effAlpha = nullAlpha ? nullptr : alpha;
    const aclblasComplex* effBeta = nullBeta ? nullptr : beta;

    // Forward invalid shapes to the operator's own validation. k<0 must not reach
    // the allocation below: a negative aCols wraps to a huge size_t request.
    if (handle == nullptr || n <= 0 || k < 0) {
        return aclblasCsyrk(handle, uplo, trans, n, k, effAlpha, A, lda, effBeta, C, ldc);
    }

    const int aCols = (trans == ACLBLAS_OP_N) ? k : n;
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(aCols) * sizeof(aclblasComplex);
    const size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    constexpr size_t scalarBytes = sizeof(aclblasComplex);

    CsyrkDeviceBuffers bufs;
    aclblasStatus_t st;

    st = CsyrkTryAllocAndCopy(effAlpha, scalarBytes, bufs.dAlpha);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    st = CsyrkTryAllocAndCopy(A, aBytes, bufs.dA);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    st = CsyrkTryAllocAndCopy(effBeta, scalarBytes, bufs.dBeta);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;
    st = CsyrkTryAllocAndCopy(C, cBytes, bufs.dC);
    if (st != ACLBLAS_STATUS_SUCCESS)
        return st;

    aclblasStatus_t ret = aclblasCsyrk(
        handle, uplo, trans, n, k, bufs.dAlpha ? static_cast<const aclblasComplex*>(bufs.dAlpha) : effAlpha,
        bufs.dA ? static_cast<const aclblasComplex*>(bufs.dA) : A, lda,
        bufs.dBeta ? static_cast<const aclblasComplex*>(bufs.dBeta) : effBeta,
        bufs.dC ? static_cast<aclblasComplex*>(bufs.dC) : C, ldc);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }

    aclError aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    if (C != nullptr && bufs.dC != nullptr) {
        aclRet = aclrtMemcpy(C, cBytes, bufs.dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    return ret;
}
