/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sgemv_host.cpp
 * \brief Single-precision sgemv host-side implementation (SIMT).
 *
 * y = alpha * op(A) * x + beta * y
 */

#include <algorithm>
#include <cstdint>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "sgemv_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/kernel_constant.h"
#include "common/helper/host_utils.h"

void sgemv_kernel_do(
    uint8_t* A, uint8_t* x, uint8_t* y, uint8_t* workSpace, const SgemvTilingData& tiling, uint32_t numBlocks,
    void* stream);

static aclblasStatus_t ValidateSgemvParams(
    aclblasOperation_t trans, int m, int n, int lda, int incx, int incy, const float* alpha, const float* beta,
    const float* a, const float* x, const float* y)
{
    CHECK_RET(
        trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C,
        OP_LOGE("aclblasSgemv", "invalid trans=%d", static_cast<int>(trans));
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(m >= 0, OP_LOGE("aclblasSgemv", "invalid m=%d", m); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(n >= 0, OP_LOGE("aclblasSgemv", "invalid n=%d", n); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        lda >= std::max(1, m), OP_LOGE("aclblasSgemv", "invalid lda=%d, m=%d", lda, m);
        return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incx != 0, OP_LOGE("aclblasSgemv", "incx must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        incx != INT32_MIN, OP_LOGE("aclblasSgemv", "incx must not be INT32_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(incy != 0, OP_LOGE("aclblasSgemv", "incy must not be zero"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        incy != INT32_MIN, OP_LOGE("aclblasSgemv", "incy must not be INT32_MIN"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        alpha != nullptr, OP_LOGE("aclblasSgemv", "alpha must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    CHECK_RET(
        beta != nullptr, OP_LOGE("aclblasSgemv", "beta must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    if (m > 0 && n > 0) {
        CHECK_RET(a != nullptr, OP_LOGE("aclblasSgemv", "a must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    // x dimension: trans=N ? n : m
    uint32_t xDim = (trans == ACLBLAS_OP_N) ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);
    if (xDim > 0) {
        CHECK_RET(x != nullptr, OP_LOGE("aclblasSgemv", "x must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    // y dimension: trans=N ? m : n
    uint32_t yDim = (trans == ACLBLAS_OP_N) ? static_cast<uint32_t>(m) : static_cast<uint32_t>(n);
    if (yDim > 0) {
        CHECK_RET(y != nullptr, OP_LOGE("aclblasSgemv", "y must not be nullptr"); return ACLBLAS_STATUS_INVALID_VALUE);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static SgemvTilingData CalcSgemvTiling(
    uint32_t useNumBlocks, int m, int n, int lda, aclblasOperation_t trans, float alpha, float beta, int incx, int incy)
{
    SgemvTilingData tiling{};
    bool isTrans = (trans != ACLBLAS_OP_N);
    uint32_t outDim = isTrans ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);

    tiling.numThreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(outDim, useNumBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tiling.rowsPerBlock = CeilDiv<uint32_t>(outDim, useNumBlocks);
    tiling.m = static_cast<uint32_t>(m);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.trans = isTrans ? 1U : 0U;
    tiling.alpha = alpha;
    tiling.beta = beta;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);

    return tiling;
}

aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* a, int lda,
    const float* x, int incx, const float* beta, float* y, int incy)
{
    auto* h = handle;
    CHECK_RET(h != nullptr, OP_LOGE("aclblasSgemv", "handle is nullptr"); return ACLBLAS_STATUS_HANDLE_IS_NULLPTR);

    // Validate parameters
    aclblasStatus_t st = ValidateSgemvParams(trans, m, n, lda, incx, incy, alpha, beta, a, x, y);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // Early exit for empty matrix
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSgemv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    bool isTrans = (trans != ACLBLAS_OP_N);
    uint32_t outDim = isTrans ? static_cast<uint32_t>(n) : static_cast<uint32_t>(m);
    uint32_t useNumBlocks = std::min(CeilDiv<uint32_t>(outDim, SIMT_MIN_THREAD_NUM), aivCoreNum);
    if (useNumBlocks == 0) {
        useNumBlocks = 1;
    }

    SgemvTilingData tiling = CalcSgemvTiling(useNumBlocks, m, n, lda, trans, *alpha, *beta, incx, incy);

    OP_LOGD(
        "aclblasSgemv", "tiling: m=%u n=%u lda=%u trans=%u numBlocks=%u numThreads=%u rowsPerBlock=%u", tiling.m,
        tiling.n, tiling.lda, tiling.trans, useNumBlocks, tiling.numThreads, tiling.rowsPerBlock);
    OP_LOGI("aclblasSgemv", "launching kernel");

    sgemv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(a)), reinterpret_cast<uint8_t*>(const_cast<float*>(x)),
        reinterpret_cast<uint8_t*>(y), nullptr, tiling, useNumBlocks, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}
