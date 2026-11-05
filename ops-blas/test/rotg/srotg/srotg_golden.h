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

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace {

constexpr float SROTG_GOLDEN_SAFMIN = 1.1754943508222875e-38f;
constexpr float SROTG_GOLDEN_SAFMAX = 1.7014118346046923e+38f;

inline float SrotgGoldenAbs(float v) { return v < 0.0f ? -v : v; }
inline float SrotgGoldenSign(float v) { return v < 0.0f ? -1.0f : 1.0f; }
inline float SrotgGoldenMax(float a, float b) { return a > b ? a : b; }
inline float SrotgGoldenMin(float a, float b) { return a < b ? a : b; }

// Reference implementation — mirrors the robust algorithm in srotg_host.cpp.
inline void SrotgGoldenCompute(float* a, float* b, float* c, float* s)
{
    const float aVal = *a;
    const float bVal = *b;
    const float absA = SrotgGoldenAbs(aVal);
    const float absB = SrotgGoldenAbs(bVal);

    if (absA == 0.0f && absB == 0.0f) {
        *a = 0.0f; *b = 0.0f; *c = 1.0f; *s = 0.0f;
        return;
    }

    const float scale = SrotgGoldenMin(SROTG_GOLDEN_SAFMAX,
                            SrotgGoldenMax(SROTG_GOLDEN_SAFMIN,
                                SrotgGoldenMax(absA, absB)));
    const float sa = aVal / scale;
    const float sb = bVal / scale;
    const float sigma = absA > absB ? SrotgGoldenSign(aVal) : SrotgGoldenSign(bVal);
    const float r = sigma * (scale * std::sqrt(sa * sa + sb * sb));

    if (r == 0.0f) {
        *a = r; *b = 0.0f; *c = 1.0f; *s = 0.0f;
        return;
    }

    const float cv = aVal / r;
    const float sv = bVal / r;
    const float z = absA > absB ? sv : (cv == 0.0f ? 1.0f : 1.0f / cv);

    *a = r; *b = z; *c = cv; *s = sv;
}

} // namespace

inline aclblasStatus_t aclblasSrotg_cpu(aclblasHandle_t handle, float* a, float* b, float* c, float* s)
{
    if (handle == nullptr)
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    if (a == nullptr || b == nullptr || c == nullptr || s == nullptr)
        return ACLBLAS_STATUS_INVALID_VALUE;

    SrotgGoldenCompute(a, b, c, s);
    return ACLBLAS_STATUS_SUCCESS;
}
