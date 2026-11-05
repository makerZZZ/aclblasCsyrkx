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

#include <cblas.h>
#include <cstdint>
#include <vector>

#include "cann_ops_blas.h"

inline aclblasStatus_t aclblasScnrm2_cpu(
    aclblasHandle_t handle, int64_t n, const aclblasComplex* x, int64_t incx, float* result)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (result == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0) {
        *result = 0.0f;
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    int64_t absInc = (incx > 0) ? incx : -incx;
    // Preserve float inputs exactly while avoiding long-vector FP32 reference accumulation error.
    // The norm is invariant under reversal, so gather with the absolute stride.
    std::vector<double> input(static_cast<size_t>(n) * 2);
    for (int64_t i = 0; i < n; ++i) {
        input[static_cast<size_t>(i) * 2] = x[i * absInc].real;
        input[static_cast<size_t>(i) * 2 + 1] = x[i * absInc].imag;
    }
    *result = static_cast<float>(cblas_dznrm2(n, input.data(), 1));
    return ACLBLAS_STATUS_SUCCESS;
}
