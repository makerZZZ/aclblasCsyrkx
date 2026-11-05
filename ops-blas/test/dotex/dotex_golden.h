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
#include <vector>

#include <cblas.h>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "dtype_compat.h"

// ── parameter validation ───────────────────────────────────────────────────────
inline aclblasStatus_t ValidateDotexCpuParams(
    int n, const float* x, const float* y, const float* result, aclDataType xType,
    aclDataType yType, aclDataType resultType, aclDataType executionType)
{
    if (n < 0 || result == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n > 0 && (x == nullptr || y == nullptr))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (xType != yType || resultType != xType || executionType != ACL_FLOAT)
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    if (xType != ACL_FLOAT && xType != ACL_FLOAT16 && xType != ACL_BF16)
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    return ACLBLAS_STATUS_SUCCESS;
}

// ── CPU reference (golden) ────────────────────────────────────────────────────
inline aclblasStatus_t aclblasDotEx_cpu(
    aclblasHandle_t handle, int n, const float* x, aclDataType xType, int incx, const float* y, aclDataType yType,
    int incy, float* result, aclDataType resultType, aclDataType executionType)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;

    if (n == 0) {
        *result = 0.0f;
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t vr = ValidateDotexCpuParams(n, x, y, result, xType, yType, resultType, executionType);
    if (vr != ACLBLAS_STATUS_SUCCESS)
        return vr;

    if (xType == ACL_FLOAT) {
        *result = cblas_sdot(n, x, incx, y, incy);
    } else {
        int absIncx = std::abs(incx), absIncy = std::abs(incy);
        float sum = 0.0;
        for (int64_t i = 0; i < n; i++) {
            int64_t xIdx = (incx > 0) ? (i * incx) : ((n - 1 - i) * absIncx);
            int64_t yIdx = (incy > 0) ? (i * incy) : ((n - 1 - i) * absIncy);
            sum += castToDtype(x[xIdx], static_cast<int32_t>(xType))
                 * castToDtype(y[yIdx], static_cast<int32_t>(yType));
        }
        *result = castToDtype(static_cast<float>(sum), static_cast<int32_t>(resultType));
    }
    return ACLBLAS_STATUS_SUCCESS;
}
