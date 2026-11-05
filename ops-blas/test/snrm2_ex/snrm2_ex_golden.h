/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SNRM2_EX_GOLDEN_H
#define SNRM2_EX_GOLDEN_H

#include <cmath>
#include <cstdint>

#include <securec.h>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "dtype_cast.h"

using blas_common::FloatToHalf;
using blas_common::HalfToFloat;

// CPU golden：缩放 L2 范数（与 NPU 实现的溢出安全算法一致）。
// scale = max(|x[i]|)；若 scale == 0 → result = 0；
// 否则 ssq = Σ (x[i]/scale)²；result = scale * sqrt(ssq)。
// 使用 double 累加以提高参考精度，避免大向量场景下 FP32 累加误差放大。
inline float snrm2_ex_cpu(aclDataType xtype, const void* x, int64_t n, int64_t incx)
{
    if (n <= 0)
        return 0.0f;

    uint64_t absInc = static_cast<uint64_t>(std::abs(incx));

    double scale = 0.0;
    double ssq = 0.0;

    for (int64_t i = 0; i < n; i++) {
        uint64_t idx = (incx >= 0) ? (static_cast<uint64_t>(i) * static_cast<uint64_t>(incx))
                                   : (static_cast<uint64_t>(n - 1 - i) * absInc);
        float xi;
        if (xtype == ACL_FLOAT16) {
            uint16_t h;
            if (memcpy_s(&h, sizeof(h), static_cast<const uint8_t*>(x) + idx * sizeof(uint16_t), sizeof(uint16_t)) != EOK) return 0.0f;
            xi = HalfToFloat(h);
        } else {
            if (memcpy_s(&xi, sizeof(xi), static_cast<const uint8_t*>(x) + idx * sizeof(float), sizeof(float)) != EOK) return 0.0f;
        }

        double axi = std::abs(static_cast<double>(xi));
        if (axi != 0.0) {
            if (scale < axi) {
                double ratio = scale / axi;
                ssq = 1.0 + ssq * ratio * ratio;
                scale = axi;
            } else {
                double ratio = axi / scale;
                ssq += ratio * ratio;
            }
        }
    }

    if (scale == 0.0)
        return 0.0f;
    return static_cast<float>(scale * std::sqrt(ssq));
}

#endif // SNRM2_EX_GOLDEN_H
