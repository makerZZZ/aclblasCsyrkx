/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "common/helper/kernel_constant.h"
#include "srotg_kernel.h"

using namespace AscendC;

constexpr uint32_t SROTG_SIMT_THREAD_NUM = SIMT_MIN_THREAD_NUM;

// SIMT 计算函数 — 仅 thread 0 执行标量 Givens 旋转参数生成
// r 用 hypotf 计算，与 host 侧 SrotgCpuCompute 完全一致，
// 避免 x86/Ascend 不同平台对朴素缩放式 sqrtf(aa*aa+bb*bb) 的 1 ULP 舍入差异。
__simt_vf__ __aicore__ LAUNCH_BOUND(SROTG_SIMT_THREAD_NUM)
inline void srotg_simt_compute(__gm__ float* aGm, __gm__ float* bGm,
                                __gm__ float* cGm, __gm__ float* sGm)
{
    if (threadIdx.x != 0) {
        return;
    }

    float a = *aGm;
    float b = *bGm;

    float absA = a >= 0.0f ? a : -a;
    float absB = b >= 0.0f ? b : -b;

    float roe = (absA > absB) ? a : b;
    float scale = absA + absB;

    float r;
    float z;
    float c;
    float s;

    if (scale == 0.0f) {
        c = 1.0f;
        s = 0.0f;
        r = 0.0f;
        z = 0.0f;
    } else {
        r = (roe >= 0.0f ? 1.0f : -1.0f) * hypotf(a, b);
        c = a / r;
        s = b / r;
        z = 1.0f;
        if (absA > absB) {
            z = s;
        } else if (c != 0.0f) {
            z = 1.0f / c;
        }
    }

    *aGm = r;
    *bGm = z;
    *cGm = c;
    *sGm = s;
}

extern "C" __global__ __aicore__ void srotg_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR s)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    asc_vf_call<srotg_simt_compute>(
        dim3{SROTG_SIMT_THREAD_NUM, 1, 1},
        reinterpret_cast<__gm__ float*>(a),
        reinterpret_cast<__gm__ float*>(b),
        reinterpret_cast<__gm__ float*>(c),
        reinterpret_cast<__gm__ float*>(s));
}

void srotg_kernel_do(uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* s, void* stream)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    srotg_kernel<<<1, nullptr, aclStream>>>(a, b, c, s);
}
