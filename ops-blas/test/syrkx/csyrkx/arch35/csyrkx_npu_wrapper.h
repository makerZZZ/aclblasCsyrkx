/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSYRKX_NPU_WRAPPER_H
#define CSYRKX_NPU_WRAPPER_H

#include <cstddef>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Device buffer RAII holder for the complex dual-matrix csyrkx wrapper.
struct CsyrkxDeviceBuffers {
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    void* dAlpha = nullptr;
    void* dBeta = nullptr;

    ~CsyrkxDeviceBuffers()
    {
        if (dA != nullptr) { aclrtFree(dA); }
        if (dB != nullptr) { aclrtFree(dB); }
        if (dC != nullptr) { aclrtFree(dC); }
        if (dAlpha != nullptr) { aclrtFree(dAlpha); }
        if (dBeta != nullptr) { aclrtFree(dBeta); }
    }
};

static inline aclblasStatus_t CsyrkxAllocAndCopyOne(void*& devPtr, const void* hostPtr, size_t bytes, bool required)
{
    if (!required || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclError ret = aclrtMalloc(&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (hostPtr != nullptr) {
        ret = aclrtMemcpy(devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Validation-only invocations pass straight through so the caller's pointers
// (including nullptr scalars) reach the operator unchanged. A/B null pointers
// do NOT force passthrough: the operator accepts them when alpha == 0 or k == 0,
// and device C is still required for a legal call.
static inline bool CsyrkxNeedsPassthrough(
    aclblasHandle handle, int n, const aclblasComplex* alpha, const aclblasComplex* beta, aclblasComplex* C)
{
    return handle == nullptr || n <= 0 || alpha == nullptr || beta == nullptr || C == nullptr;
}

inline aclblasStatus_t aclblasCsyrkx_npu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k,
    const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb,
    const aclblasComplex* beta, aclblasComplex* C, int ldc)
{
    if (CsyrkxNeedsPassthrough(handle, n, alpha, beta, C)) {
        return aclblasCsyrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    const int abCols = (k > 0) ? ((trans == ACLBLAS_OP_N) ? k : n) : 0;
    const size_t aBytes =
        (k > 0) ? static_cast<size_t>(lda) * static_cast<size_t>(abCols) * sizeof(aclblasComplex) : 0;
    const size_t bBytes =
        (k > 0) ? static_cast<size_t>(ldb) * static_cast<size_t>(abCols) * sizeof(aclblasComplex) : 0;
    const size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    const size_t scalarBytes = sizeof(aclblasComplex);

    CsyrkxDeviceBuffers buf;

    aclblasStatus_t st = CsyrkxAllocAndCopyOne(buf.dA, A, aBytes, A != nullptr && aBytes > 0);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }
    st = CsyrkxAllocAndCopyOne(buf.dB, B, bBytes, B != nullptr && bBytes > 0);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }
    st = CsyrkxAllocAndCopyOne(buf.dC, C, cBytes, true);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }
    st = CsyrkxAllocAndCopyOne(buf.dAlpha, alpha, scalarBytes, true);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }
    st = CsyrkxAllocAndCopyOne(buf.dBeta, beta, scalarBytes, true);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }

    aclblasStatus_t ret = aclblasCsyrkx(
        handle, uplo, trans, n, k, static_cast<const aclblasComplex*>(buf.dAlpha),
        static_cast<const aclblasComplex*>(buf.dA), lda, static_cast<const aclblasComplex*>(buf.dB), ldb,
        static_cast<const aclblasComplex*>(buf.dBeta), static_cast<aclblasComplex*>(buf.dC), ldc);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }

    aclError aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    aclRet = aclrtMemcpy(C, cBytes, buf.dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ret;
}

#endif // CSYRKX_NPU_WRAPPER_H
