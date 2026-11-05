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

#include <array>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline aclblasStatus_t aclblasSrotm_cpu(
    aclblasHandle_t handle,
    int64_t n, float* x, int64_t incx,
    float* y, int64_t incy, const float* sparam)
{
    if (handle == nullptr) return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (x == nullptr || y == nullptr || sparam == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0 || incy == 0) return ACLBLAS_STATUS_INVALID_VALUE;

    if (n <= 0 || sparam[0] == -2.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    cblas_srotm(static_cast<int>(n), x, static_cast<int>(incx),
                y, static_cast<int>(incy), sparam);
    return ACLBLAS_STATUS_SUCCESS;
}

