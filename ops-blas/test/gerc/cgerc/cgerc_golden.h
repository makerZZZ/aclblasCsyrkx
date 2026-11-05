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
#include <limits>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

constexpr float CGERC_SPECIAL_VALUE_RANGE_DIVISOR = 4.0F;

inline uint64_t CgercAbsStride(int stride)
{
    return stride > 0 ? static_cast<uint64_t>(stride) : static_cast<uint64_t>(-static_cast<int64_t>(stride));
}

inline bool CgercComplexIndexRangeSafe(uint64_t maxIndex)
{
    constexpr uint64_t elementBytes = sizeof(aclblasComplex);
    constexpr uint64_t indexLimit = (std::numeric_limits<uint64_t>::max() - (elementBytes - 1U)) / elementBytes;
    return maxIndex <= indexLimit;
}

inline bool CgercAddressRangesSafe(int m, int n, int lda, int incx, int incy)
{
    const uint64_t xMaxIndex = static_cast<uint64_t>(m - 1) * CgercAbsStride(incx);
    const uint64_t yMaxIndex = static_cast<uint64_t>(n - 1) * CgercAbsStride(incy);
    const uint64_t aMaxIndex = static_cast<uint64_t>(n - 1) * static_cast<uint64_t>(lda) + static_cast<uint64_t>(m - 1);
    return CgercComplexIndexRangeSafe(xMaxIndex) && CgercComplexIndexRangeSafe(yMaxIndex) &&
           CgercComplexIndexRangeSafe(aMaxIndex);
}

inline bool CgercParametersInvalid(int m, int n, const aclblasComplex* alpha, int incx, int incy, int lda)
{
    return m < 0 || n < 0 || alpha == nullptr || incx == 0 || incy == 0 || lda < std::max(1, m);
}

inline bool CgercNeedsPortableAlphaOneGolden(
    int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
    const aclblasComplex* A, int lda)
{
    if (alpha->real != 1.0F || alpha->imag != 0.0F) {
        return false;
    }
    constexpr float largeValue = std::numeric_limits<float>::max() / CGERC_SPECIAL_VALUE_RANGE_DIVISOR;
    const auto isSpecial = [=](const aclblasComplex& value) {
        return !std::isfinite(value.real) || !std::isfinite(value.imag) || std::fabs(value.real) > largeValue ||
               std::fabs(value.imag) > largeValue;
    };
    const uint64_t absIncx = CgercAbsStride(incx);
    const uint64_t absIncy = CgercAbsStride(incy);
    for (int row = 0; row < m; ++row) {
        const uint64_t index =
            incx > 0 ? static_cast<uint64_t>(row) * absIncx : static_cast<uint64_t>(m - 1 - row) * absIncx;
        if (isSpecial(x[index])) {
            return true;
        }
    }
    for (int col = 0; col < n; ++col) {
        const uint64_t index =
            incy > 0 ? static_cast<uint64_t>(col) * absIncy : static_cast<uint64_t>(n - 1 - col) * absIncy;
        if (isSpecial(y[index])) {
            return true;
        }
    }
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            if (isSpecial(A[static_cast<uint64_t>(row) + static_cast<uint64_t>(col) * static_cast<uint64_t>(lda)])) {
                return true;
            }
        }
    }
    return false;
}

inline void CgercPortableAlphaOneGolden(
    int m, int n, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    const uint64_t absIncx = CgercAbsStride(incx);
    const uint64_t absIncy = CgercAbsStride(incy);
    for (int col = 0; col < n; ++col) {
        const uint64_t yIndex =
            incy > 0 ? static_cast<uint64_t>(col) * absIncy : static_cast<uint64_t>(n - 1 - col) * absIncy;
        const float yReal = y[yIndex].real;
        const float yImag = y[yIndex].imag;
        for (int row = 0; row < m; ++row) {
            const uint64_t xIndex =
                incx > 0 ? static_cast<uint64_t>(row) * absIncx : static_cast<uint64_t>(m - 1 - row) * absIncx;
            const uint64_t aIndex =
                static_cast<uint64_t>(row) + static_cast<uint64_t>(col) * static_cast<uint64_t>(lda);
            volatile float term = x[xIndex].real * yReal;
            A[aIndex].real += term;
            term = x[xIndex].imag * yImag;
            A[aIndex].real += term;
            term = x[xIndex].imag * yReal;
            A[aIndex].imag += term;
            term = x[xIndex].real * -yImag;
            A[aIndex].imag += term;
        }
    }
}

inline aclblasStatus_t aclblasCgercCpu(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (CgercParametersInvalid(m, n, alpha, incx, incy, lda)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0 && (x == nullptr || y == nullptr || A == nullptr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m > 0 && n > 0) {
        if (!CgercAddressRangesSafe(m, n, lda, incx, incy)) {
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (m == 0 || n == 0 || (alpha->real == 0.0F && alpha->imag == 0.0F)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // CBLAS vendors use different fast paths for Inf/NaN and overflow. Keep the
    // attachment's alpha-one special-value oracle deterministic across hosts.
    if (CgercNeedsPortableAlphaOneGolden(m, n, alpha, x, incx, y, incy, A, lda)) {
        CgercPortableAlphaOneGolden(m, n, x, incx, y, incy, A, lda);
    } else {
        cblas_cgerc(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
    }
    return ACLBLAS_STATUS_SUCCESS;
}
