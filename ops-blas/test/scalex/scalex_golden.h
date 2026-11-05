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
#include <cstring>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "dtype_compat.h"
#include "scalex_param.h"

// aclblasScalex_cpu — reference implementation, same signature as aclblasScalex
// ─────────────────────────────────────────────────────────────────────────────
inline aclblasStatus_t aclblasScalex_cpu(
    aclblasHandle_t handle, int n, const void* alpha, aclDataType alphaType, void* x, aclDataType xType, int incx,
    aclDataType executionType)
{
    // Parameter validation — same order as Host-side spec
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n <= 0 || incx <= 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (alpha == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (x == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    // Type checks — only alpha=FP32 + x∈{FP16,BF16,FP32} + exec=FP32 supported
    if (alphaType != ACL_FLOAT)
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    if (xType != ACL_FLOAT && xType != ACL_FLOAT16 && xType != ACL_BF16)
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    if (executionType != ACL_FLOAT)
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    float alphaVal = *static_cast<const float*>(alpha);
    float* xFloat = static_cast<float*>(x);

    if (xType == ACL_FLOAT) {
        cblas_sscal(n, alphaVal, xFloat, incx);
    } else {
        for (int i = 0; i < n; i++) {
            int idx = i * incx;
            float val = xFloat[idx];
            float result = alphaVal * val;
            xFloat[idx] = castToDtype(result, static_cast<int32_t>(xType));
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}
