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
 * \file aux_get_c0_size.h
 * \brief Local self-contained replacement for AscendC::AuxGetC0Size, decoupling
 *        blasLt MXFP8/MXFP4 kernels from the adv_api/matmul/matmul.h header.
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "utils/common_types.h"

namespace AscendC {

namespace AuxC0SizeDetail {
constexpr int32_t B4_C0SIZE = 64;
constexpr int32_t B8_C0SIZE = 32;
constexpr int32_t B16_C0SIZE = 16;
constexpr int32_t B32_C0SIZE = 8;

template <typename T, typename... Others>
struct IsTypeOneOf {
    static constexpr bool value = false;
};

template <typename T, typename First, typename... Others>
struct IsTypeOneOf<T, First, Others...> {
    static constexpr bool value = IsSameType<T, First>::value || IsTypeOneOf<T, Others...>::value;
};

template <typename T, typename... Others>
constexpr bool IsTypeOneOfV = IsTypeOneOf<T, Others...>::value;
} // namespace AuxC0SizeDetail

template <typename SrcT>
__aicore__ inline constexpr static int32_t AuxGetC0Size()
{
    if (sizeof(SrcT) == sizeof(float)) {
        return AuxC0SizeDetail::B32_C0SIZE;
    }
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    else if (AuxC0SizeDetail::IsTypeOneOfV<SrcT, uint8_t, int8_t, hifloat8_t, fp8_e4m3fn_t, fp8_e5m2_t,
                                           fp8_e8m0_t>) {
        return AuxC0SizeDetail::B8_C0SIZE;
    } else if (AuxC0SizeDetail::IsTypeOneOfV<SrcT, int4b_t, fp4x2_e1m2_t, fp4x2_e2m1_t>) {
        return AuxC0SizeDetail::B4_C0SIZE;
    }
#else
    else if (IsSameType<SrcT, int8_t>::value) {
        return AuxC0SizeDetail::B8_C0SIZE;
    } else if (IsSameType<SrcT, int4b_t>::value) {
        return AuxC0SizeDetail::B4_C0SIZE;
    }
#endif
    return AuxC0SizeDetail::B16_C0SIZE;
}

} // namespace AscendC
