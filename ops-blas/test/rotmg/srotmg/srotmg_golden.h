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

inline aclblasStatus_t aclblasSrotmg_cpu(
    aclblasHandle_t handle, float* d1, float* d2, float* x1, const float* y1, float* param)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (d1 == nullptr || d2 == nullptr || x1 == nullptr || y1 == nullptr || param == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    // Note: cblas_srotmg takes b2 by value, not by pointer
    cblas_srotmg(d1, d2, x1, *y1, param);
    return ACLBLAS_STATUS_SUCCESS;
}
