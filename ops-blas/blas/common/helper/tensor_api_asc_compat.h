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
 * \file tensor_api_asc_compat.h
 * \brief Compatibility shims that let the ops-tensor tensor_api headers build
 *        with older CANN/ASC toolchains.
 *
 * CMake runs a small ASC compile probe per capability (cmake/asc_devkit_version.cmake)
 * and only force-includes this header ahead of the ASC translation units when the
 * probe actually reports that capability as missing; each section below is
 * additionally self-guarded, and nothing is injected for non-arch35 targets.
 *
 * Section 1 - arch-less ASC front-end pass:
 *   The ASC front-end compiles a translation unit twice: once with
 *   __NPU_ARCH__ defined for the target SoC, and once without it (an internal
 *   pass used for host-side/syntax work). CANN 9.1.0 declares the
 *   bf16/hif8/fp8/fp4 vector builtins and int4x2_t only inside headers gated on
 *   (__NPU_ARCH__ == 3510), so the arch-less pass never sees them. ops-tensor's
 *   c_api headers (c_api/asc_simd.h -> c_api/reg_compute, and
 *   impl/tensor_api/utils/constant_impl.h) use those types unconditionally,
 *   which fails with "unknown type name 'vector_bf16' / 'int4x2_t'". The aliases
 *   below reproduce the toolchain definitions with the same register width for
 *   the arch-less pass only; the arch pass keeps the real builtins. They are
 *   only ever used by never-instantiated reg_compute helpers, so the exact
 *   element type of the 8-bit/4-bit aliases does not affect generated code.
 *
 * Section 2 - older CANN debug-bus helpers:
 *   ops-tensor's print_tensor_debug_bus_impl.h calls
 *   __asc_aicore::get_debug_bus_local_addr_shift / get_debug_bus_loop_steplen,
 *   which newer CANN provides but CANN 9.1.0 does not. The two helpers are
 *   re-declared with the semantics of the newer toolchain (the values match
 *   ops-tensor's own npu_arch_3510 unit-test stub).
 */

#pragma once

// ---------------------------------------------------------------------------
// 1) Arch-less ASC pass: provide the arch-gated toolchain scalar/vector types.
// ---------------------------------------------------------------------------
#if defined(ASC_TENSOR_API_ARCH_TYPES_SHIM) && !defined(__NPU_ARCH__)

struct int4x2_t {
    char val;
};

using vector_bf16 = vector_f16;    // 128 x 16-bit
using vector_hif8 = vector_u8;     // 256 x 8-bit
using vector_f8e4m3 = vector_u8;   // 256 x 8-bit
using vector_f8e5m2 = vector_u8;   // 256 x 8-bit
using vector_f8e8m0 = vector_u8;   // 256 x 8-bit
using vector_s4x2 = vector_u8;     // 256 x 8-bit (2 x 4-bit per lane)
using vector_f4e2m1x2 = vector_u8; // 256 x 8-bit (2 x 4-bit per lane)
using vector_f4e1m2x2 = vector_u8; // 256 x 8-bit (2 x 4-bit per lane)

#endif // ASC_TENSOR_API_ARCH_TYPES_SHIM && !__NPU_ARCH__

// ---------------------------------------------------------------------------
// 2) Older CANN: two __asc_aicore debug-bus helpers used by ops-tensor.
//
// These are declared here without including kernel_operator.h: this header is
// force-included ahead of every ASC translation unit, including the arch35
// *_host.cpp files that are compiled as ASC but are really host code (where
// __gm__ must stay a no-op). Pulling in kernel_operator.h there would turn on
// device address-space semantics and break their pointer casts. The template
// parameter is deduced with `auto` and the hardware enum is reached through
// decltype, so no toolchain declaration is needed.
// ---------------------------------------------------------------------------
#if defined(ASC_TENSOR_API_DEBUG_BUS_SHIM)

namespace __asc_aicore {

// Byte step between consecutive debug-bus read chunks (BIAS uses 8B, others 32B).
template <auto HardwareType>
inline constexpr uint64_t get_debug_bus_loop_steplen()
{
    if constexpr (HardwareType == decltype(HardwareType)::BIAS) {
        return 8U;
    }
    return 32U;
}

// Shift applied to a local address before it is written to the debug-bus
// ADDR_LOW register: L1 is addressed in 32B units, everything else in bytes.
template <auto HardwareType>
inline constexpr uint32_t get_debug_bus_local_addr_shift()
{
    return HardwareType == decltype(HardwareType)::L1 ? 5U : 0U;
}

} // namespace __asc_aicore

#endif // ASC_TENSOR_API_DEBUG_BUS_SHIM
