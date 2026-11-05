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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline bool cgemvIsZero(const aclblasComplex& v) { return v.real == 0.0f && v.imag == 0.0f; }

inline bool cgemvIsOne(const aclblasComplex& v) { return v.real == 1.0f && v.imag == 0.0f; }

inline aclblasStatus_t validateCgemvCpuParams(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha, int lda, int incx,
    const aclblasComplex* a, const aclblasComplex* x, const aclblasComplex* beta, const aclblasComplex* y, int incy)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (m < 0 || n < 0 || lda < std::max(1, m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0) {
        if (!cgemvIsZero(*alpha) && (a == nullptr || x == nullptr)) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (!(cgemvIsZero(*alpha) && cgemvIsOne(*beta)) && y == nullptr) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline int64_t cgemvCpuVectorIndex(int64_t logicalIndex, int64_t logicalLength, int64_t stride)
{
    return (stride >= 0) ? logicalIndex * stride : (logicalLength - 1 - logicalIndex) * (-stride);
}

inline float cgemvCpuRoundedMul(float lhs, float rhs) { return std::fma(lhs, rhs, -0.0F); }

inline void cgemvCpuAccumulate(
    const aclblasComplex& matrixValue, const aclblasComplex& vectorValue, bool conjugate, float& accReal,
    float& accImag)
{
    const float matrixImag = conjugate ? -matrixValue.imag : matrixValue.imag;
    float term = cgemvCpuRoundedMul(matrixValue.real, vectorValue.real);
    term -= cgemvCpuRoundedMul(matrixImag, vectorValue.imag);
    accReal += term;

    term = cgemvCpuRoundedMul(matrixValue.real, vectorValue.imag);
    term += cgemvCpuRoundedMul(matrixImag, vectorValue.real);
    accImag += term;
}

inline void cgemvCpuWriteBack(
    aclblasComplex& y, float accReal, float accImag, const aclblasComplex& alpha, const aclblasComplex& beta)
{
    if (cgemvIsZero(alpha)) {
        if (cgemvIsZero(beta)) {
            y.real = 0.0F;
            y.imag = 0.0F;
        } else if (!cgemvIsOne(beta)) {
            const float yReal = y.real;
            const float yImag = y.imag;
            y.real = cgemvCpuRoundedMul(beta.real, yReal) - cgemvCpuRoundedMul(beta.imag, yImag);
            y.imag = cgemvCpuRoundedMul(beta.real, yImag) + cgemvCpuRoundedMul(beta.imag, yReal);
        }
        return;
    }

    float outReal;
    float outImag;
    if (cgemvIsOne(alpha)) {
        outReal = accReal;
        outImag = accImag;
    } else {
        outReal = cgemvCpuRoundedMul(alpha.real, accReal);
        outReal -= cgemvCpuRoundedMul(alpha.imag, accImag);
        outImag = cgemvCpuRoundedMul(alpha.real, accImag);
        outImag += cgemvCpuRoundedMul(alpha.imag, accReal);
    }

    if (!cgemvIsZero(beta)) {
        if (cgemvIsOne(beta)) {
            outReal += y.real;
            outImag += y.imag;
        } else {
            float betaReal = cgemvCpuRoundedMul(beta.real, y.real);
            betaReal -= cgemvCpuRoundedMul(beta.imag, y.imag);
            float betaImag = cgemvCpuRoundedMul(beta.real, y.imag);
            betaImag += cgemvCpuRoundedMul(beta.imag, y.real);
            outReal += betaReal;
            outImag += betaImag;
        }
    }
    y.real = outReal;
    y.imag = outImag;
}

// Deterministic scalar oracle for arithmetic-order and non-finite boundary tests.
inline aclblasStatus_t aclblasCgemv_scalar(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy)
{
    const aclblasStatus_t validRet = validateCgemvCpuParams(handle, trans, m, n, alpha, lda, incx, a, x, beta, y, incy);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }
    if (m == 0 || n == 0 || (cgemvIsZero(*alpha) && cgemvIsOne(*beta))) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const int64_t outputLength = (trans == ACLBLAS_OP_N) ? m : n;
    const int64_t dotLength = (trans == ACLBLAS_OP_N) ? n : m;
    for (int64_t output = 0; output < outputLength; ++output) {
        const int64_t yIndex = cgemvCpuVectorIndex(output, outputLength, incy);
        if (cgemvIsZero(*alpha)) {
            cgemvCpuWriteBack(y[yIndex], 0.0F, 0.0F, *alpha, *beta);
            continue;
        }

        float accReal = 0.0F;
        float accImag = 0.0F;
        for (int64_t k = 0; k < dotLength; ++k) {
            const int64_t aIndex = (trans == ACLBLAS_OP_N) ? output + k * lda : k + output * lda;
            const int64_t xIndex = cgemvCpuVectorIndex(k, dotLength, incx);
            cgemvCpuAccumulate(a[aIndex], x[xIndex], trans == ACLBLAS_OP_C, accReal, accImag);
        }
        cgemvCpuWriteBack(y[yIndex], accReal, accImag, *alpha, *beta);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// CPU golden reference for the CSV suite: cblas_cgemv.
inline aclblasStatus_t aclblasCgemv_cpu(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy)
{
    const aclblasStatus_t validRet = validateCgemvCpuParams(handle, trans, m, n, alpha, lda, incx, a, x, beta, y, incy);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (cgemvIsZero(*alpha) && (a == nullptr || x == nullptr)) {
        return aclblasCgemv_scalar(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }

    cblas_cgemv(CblasColMajor, ToCblasOp(trans), m, n, alpha, a, lda, x, incx, beta, y, incy);
    return ACLBLAS_STATUS_SUCCESS;
}
