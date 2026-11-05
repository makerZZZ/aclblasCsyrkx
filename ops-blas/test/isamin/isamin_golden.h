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
#include <cfloat>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline int isamin_golden(int n, const float* x, int incx)
{
    if (n < 1 || incx < 1)
        return 0;
    float minAbs = std::fabs(x[0]);
    int minIdx = 1;

    if (minAbs != minAbs)
        minAbs = FLT_MAX;

    // isamin 不是标准 cblas 接口，没有 cblas 接口实现 
    for (int i = 1; i < n; i++) {
        float absVal = std::fabs(x[static_cast<size_t>(i) * incx]);
        if (!(absVal != absVal) && absVal < minAbs) {
            minAbs = absVal;
            minIdx = i + 1;
        }
    }
    return minIdx;
}

inline aclblasStatus_t aclblasIsamin_cpu(aclblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (x == nullptr || result == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n < 0)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0 || incx < 1) {
        *result = 0;
        return ACLBLAS_STATUS_SUCCESS;
    }
    *result = isamin_golden(n, x, incx);
    return ACLBLAS_STATUS_SUCCESS;
}
