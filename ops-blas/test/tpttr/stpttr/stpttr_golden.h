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
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

extern "C" {
void stpttr_(const char* uplo, const int* n, const float* ap, float* a, const int* lda, int* info);
}

constexpr float kStpttrSentinel = -999.0f;

inline aclblasStatus_t aclblasStpttr_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* ap, float* a, int lda)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || lda < std::max(1, n))
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (uplo != ACLBLAS_LOWER && uplo != ACLBLAS_UPPER)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (ap == nullptr || a == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0)
        return ACLBLAS_STATUS_SUCCESS;

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < lda; i++) {
            a[j * lda + i] = kStpttrSentinel;
        }
    }

    char uplo_char = ToLapackUplo(uplo);
    int info = 0;
    stpttr_(&uplo_char, &n, ap, a, &lda, &info);
    if (info != 0)
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    return ACLBLAS_STATUS_SUCCESS;
}

