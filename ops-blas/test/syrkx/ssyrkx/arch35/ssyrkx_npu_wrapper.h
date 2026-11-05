/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSYRKX_NPU_H
#define SSYRKX_NPU_H

#include <cstdint>
#include "acl/acl.h"
#include "cann_ops_blas.h"

struct SsyrkxDeviceBuffers {
    void* dA = nullptr;
    void* dB = nullptr;
    void* dC = nullptr;
    void* dAlpha = nullptr;
    void* dBeta = nullptr;

    ~SsyrkxDeviceBuffers()
    {
        if (dA != nullptr) { aclrtFree(dA); }
        if (dB != nullptr) { aclrtFree(dB); }
        if (dC != nullptr) { aclrtFree(dC); }
        if (dAlpha != nullptr) { aclrtFree(dAlpha); }
        if (dBeta != nullptr) { aclrtFree(dBeta); }
    }
};

static inline aclblasStatus_t SsyrkxAllocAndCopyOne(
    void*& devPtr, const void* hostPtr, size_t bytes, bool required)
{
    if (!required || bytes == 0) { return ACLBLAS_STATUS_SUCCESS; }
    aclError ret = aclrtMalloc(&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) { return ACLBLAS_STATUS_ALLOC_FAILED; }
    if (hostPtr != nullptr) {
        ret = aclrtMemcpy(devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) { return ACLBLAS_STATUS_INTERNAL_ERROR; }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static inline bool SsyrkxNeedsPassthrough(
    aclblasHandle handle, int n, int k,
    const float* alpha, const float* beta, float* C,
    const float* A, const float* B)
{
    return handle == nullptr || n <= 0 ||
        alpha == nullptr || beta == nullptr || C == nullptr ||
        (k > 0 && (A == nullptr || B == nullptr));
}

inline aclblasStatus_t aclblasSsyrkx_npu(
    aclblasHandle handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc)
{
    if (SsyrkxNeedsPassthrough(handle, n, k, alpha, beta, C, A, B)) {
        return aclblasSsyrkx(handle, uplo, trans, n, k,
            alpha, A, lda, B, ldb, beta, C, ldc);
    }

    int aCols = (trans == ACLBLAS_OP_N) ? k : n;
    int bCols = (trans == ACLBLAS_OP_N) ? k : n;

    const size_t aBytes = (k > 0) ? static_cast<size_t>(lda) * static_cast<size_t>(aCols) * sizeof(float) : 0;
    const size_t bBytes = (k > 0) ? static_cast<size_t>(ldb) * static_cast<size_t>(bCols) * sizeof(float) : 0;
    const size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(float);
    const size_t scalarBytes = sizeof(float);

    SsyrkxDeviceBuffers buf;

    aclblasStatus_t st = ACLBLAS_STATUS_SUCCESS;
    st = SsyrkxAllocAndCopyOne(buf.dA, A, aBytes, aBytes > 0);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }
    st = SsyrkxAllocAndCopyOne(buf.dB, B, bBytes, bBytes > 0);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }
    st = SsyrkxAllocAndCopyOne(buf.dC, C, cBytes, true);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }
    st = SsyrkxAllocAndCopyOne(buf.dAlpha, alpha, scalarBytes, true);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }
    st = SsyrkxAllocAndCopyOne(buf.dBeta, beta, scalarBytes, true);
    if (st != ACLBLAS_STATUS_SUCCESS) { return st; }

    aclblasStatus_t ret = aclblasSsyrkx(handle, uplo, trans, n, k,
        static_cast<const float*>(buf.dAlpha),
        (aBytes > 0) ? static_cast<const float*>(buf.dA) : nullptr, lda,
        (bBytes > 0) ? static_cast<const float*>(buf.dB) : nullptr, ldb,
        static_cast<const float*>(buf.dBeta),
        static_cast<float*>(buf.dC), ldc);
    if (ret != ACLBLAS_STATUS_SUCCESS) { return ret; }

    aclError aclRet = aclrtSynchronizeDevice();
    if (aclRet != ACL_SUCCESS) { return ACLBLAS_STATUS_EXECUTION_FAILED; }

    aclRet = aclrtMemcpy(C, cBytes, buf.dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) { return ACLBLAS_STATUS_INTERNAL_ERROR; }

    return ret;
}

#endif // SSYRKX_NPU_H
