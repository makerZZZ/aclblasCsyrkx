/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_fp32_kernel_utils.h
 * \brief Shared constants and inline helpers for the FP32 matmul device kernel (arch35).
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "std/algorithm.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

constexpr static uint16_t TWO_ALIGN = 2;
constexpr static uint64_t DOUBLE_BUFFER_COUNT = 2;
constexpr static int64_t L1_SIZE = 512 * 1024;
constexpr static int64_t L0A_SIZE = 64 * 1024;
constexpr static int64_t L0B_SIZE = 64 * 1024;
constexpr static int64_t L0C_SIZE = 256 * 1024;
constexpr static uint16_t ZERO_FLAG = 0;
constexpr static uint16_t FIRST_FLAG = 1;
constexpr static int64_t B16_C0_SIZE = 16;
constexpr static int64_t B32_C0_SIZE = 8;

template <typename T>
__aicore__ inline constexpr bool IsFp4()
{
    return AscendC::IsSameType<T, fp4x2_e2m1_t>::value || AscendC::IsSameType<T, fp4x2_e1m2_t>::value;
}

// Single template avoids mixed-integer overload ambiguity in MX tile code.
template <typename T, typename U>
__aicore__ inline int64_t CeilDiv(T a, U b)
{
    int64_t ai = static_cast<int64_t>(a);
    int64_t bi = static_cast<int64_t>(b);
    if (bi == 0) {
        return ai;
    }
    return (ai + bi - 1) / bi;
}

__aicore__ inline int64_t CeilAlign(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

__aicore__ inline uint64_t CeilAlign(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline constexpr static int32_t BlasLtGetC0Size()
{
    if (sizeof(T) == sizeof(float)) {
        return B32_C0_SIZE;
    }
    return B16_C0_SIZE;
}

template <typename T>
__aicore__ inline constexpr static int32_t GetC0Size()
{
    return BlasLtGetC0Size<T>();
}
