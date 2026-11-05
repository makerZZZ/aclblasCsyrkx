/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file snrm2_ex_tiling_data.h
 * \brief aclblasSnrm2Ex 算子的 Tiling 数据结构定义。
 */

#ifndef SNRM2_EX_TILING_DATA_H
#define SNRM2_EX_TILING_DATA_H

#include <cstdint>
#include "common/helper/kernel_constant.h"

// AIV 核数上限，用于 useCoreNum 裁剪保护。
constexpr uint32_t SNRM2_EX_MAX_CORE_NUM = 72;

// 单次搬入 UB 的最大元素数，由 UB 预算推导。
constexpr uint32_t SNRM2_EX_MAX_DATA_COUNT = 17920;

struct Snrm2ExTilingData {
    int64_t n;
    int64_t incx;
    uint32_t xtype; // 0 = FP32 (ACL_FLOAT)，1 = FP16 (ACL_FLOAT16)
    uint32_t useCoreNum;
    uint32_t maxDataCount;
    uint32_t batchPerCore; // 每个 core 分配的基础元素数
    uint32_t remain;       // 前 remain 个 core 各多处理一个元素
    uint32_t nthreads;     // 每 block 的 SIMT 线程数（incx != 1 路径）
};

#endif // SNRM2_EX_TILING_DATA_H
