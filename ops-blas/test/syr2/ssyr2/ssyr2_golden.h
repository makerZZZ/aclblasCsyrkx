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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cblas_compat.h"

inline void aclblasSsyr2_cpu(
    aclblasFillMode_t uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* A, int lda)
{
    if (n <= 0)
        return;

    cblas_ssyr2(CblasColMajor, ToCblasUplo(uplo), n, alpha, x, incx, y, incy, A, lda);
}

