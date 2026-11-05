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

#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

// CPU golden for aclblasSrot — signature identical to the API (c / s passed by pointer,
// aligned with cublasSrot), returns aclblasStatus_t.
//
// Parameter validation is aligned with the NPU operator AND netlib srot.f (1.2 requirement):
//   - handle == nullptr            -> ACLBLAS_STATUS_HANDLE_IS_NULLPTR
//   - n <= 0                       -> ACLBLAS_STATUS_SUCCESS (no-op, x/y untouched)
//   - x == nullptr || y == nullptr -> ACLBLAS_STATUS_INVALID_VALUE (n > 0)
//   - c == nullptr || s == nullptr -> ACLBLAS_STATUS_INVALID_VALUE (n > 0)
//   - incx == 0 / incy == 0        -> NOT rejected (netlib reuses x[0]/y[0] N times)
//   - incx < 0 / incy < 0          -> NOT rejected (netlib walks from tail end)
//   - c == 1 && s == 0             -> NOT short-circuited (netlib runs the rotation)
//
// After validation, cblas_srot (column-major, OpenBLAS) is used as the reference. cblas_srot
// takes c / s by value, so the pointers are dereferenced (*c / *s) at the call site. Its srot
// implementation matches netlib srot.f boundary behavior including inc==0 / negative stride.
inline aclblasStatus_t aclblasSrot_cpu(
    aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (n <= 0)
        return ACLBLAS_STATUS_SUCCESS;
    if (x == nullptr || y == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (c == nullptr || s == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    cblas_srot(n, x, incx, y, incy, *c, *s);
    return ACLBLAS_STATUS_SUCCESS;
}
