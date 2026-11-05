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

#include <cmath>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace {

inline float IcaminAbs1Host(float re, float im)
{
    float absRe = (re >= 0.0f) ? re : -re;
    float absIm = (im >= 0.0f) ? im : -im;
    return absRe + absIm;
}

} // namespace

inline aclblasStatus_t aclblasIcamin_cpu(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, int* result)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (result == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0 || incx < 1) {
        *result = 0;
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    // cblas-style icamin: update only when abs1 is strictly less (NaN never updates; ties keep first).
    float bestVal = IcaminAbs1Host(x[0].real, x[0].imag);
    int bestIdx = 0;
    for (int i = 1; i < n; ++i) {
        const aclblasComplex& c = x[static_cast<int64_t>(i) * incx];
        float absVal = IcaminAbs1Host(c.real, c.imag);
        if (absVal < bestVal) {
            bestVal = absVal;
            bestIdx = i;
        }
    }
    *result = bestIdx + 1;
    return ACLBLAS_STATUS_SUCCESS;
}
