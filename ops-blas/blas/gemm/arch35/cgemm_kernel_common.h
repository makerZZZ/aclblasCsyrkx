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

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

#define ASCENDC_CUBE_ONLY
#include "cgemm_kernel.h"

#define KERNEL_UTILS_LITE
#include "common/arch/hardware.h"
#include "common/helper/kernel_utils.h"

constexpr uint32_t GEMM_FP32_C0 = CGEMM_DATA_BLOCK_BYTES / sizeof(float);
constexpr uint16_t GEMM_ZERO_FLAG = 0;
constexpr uint16_t GEMM_FIRST_FLAG = 1;
