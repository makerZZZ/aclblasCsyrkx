/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AXPY_EX_GOLDEN_H
#define AXPY_EX_GOLDEN_H

#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"
#include "dtype_compat.h"

// aclblasAxpyEx_cpu — reference implementation, same signature as aclblasAxpyEx
inline aclblasStatus_t aclblasAxpyEx_cpu(
    aclblasHandle_t handle, int n, const void* alpha, aclDataType alphaType, const void* x, aclDataType xType, int incx,
    void* y, aclDataType yType, int incy, aclDataType executionType)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (alphaType != ACL_FLOAT || executionType != ACL_FLOAT)
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    if (xType != ACL_FLOAT16 && xType != ACL_BF16 && xType != ACL_FLOAT)
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    if (yType != xType)
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    if (alpha == nullptr || x == nullptr || y == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0 || incy == 0)
        return ACLBLAS_STATUS_INVALID_VALUE;

    float alphaVal = *static_cast<const float*>(alpha);
    float* yFloat = static_cast<float*>(y);
    const float* xFloat = static_cast<const float*>(x);

    if (xType == ACL_FLOAT) {
        // FP32 path: direct CBLAS saxpy (column-major, BLAS standard semantics)
        cblas_saxpy(n, alphaVal, xFloat, incx, yFloat, incy);
    } else {
        // FP16/BF16 path: per-element Cast→FP32 → alpha*x+y → Cast back to yType
        int32_t dtypeInt = static_cast<int32_t>(xType); // xType == yType (validated)
        for (int k = 0; k < n; k++) {
            int ix = (incx > 0) ? (k * incx) : ((n - 1 - k) * (-incx));
            int iy = (incy > 0) ? (k * incy) : ((n - 1 - k) * (-incy));
            // Simulate device dtype round-trip: input values are first quantised
            // to the target dtype, then the arithmetic is done in FP32, then the
            // result is quantised back.
            float xv = castToDtype(xFloat[ix], dtypeInt);
            float yv = castToDtype(yFloat[iy], dtypeInt);
            float result = alphaVal * xv + yv;
            yFloat[iy] = castToDtype(result, dtypeInt);
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#endif
