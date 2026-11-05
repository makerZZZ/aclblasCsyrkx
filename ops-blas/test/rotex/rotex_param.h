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

#include <cstdint>
#include <string>
#include <unordered_map>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

// ── Common parameter validation (shared by golden.h and npu_wrapper.h) ──
inline aclblasStatus_t RotExValidateCommonParams(int n, const void* x, const void* y,
    const void* c, const void* s, int incx, int incy)
{
    if (n < 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return ACLBLAS_STATUS_SUCCESS;
    if (x == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (y == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (c == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (s == nullptr) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incx == 0) return ACLBLAS_STATUS_INVALID_VALUE;
    if (incy == 0) return ACLBLAS_STATUS_INVALID_VALUE;
    return ACLBLAS_STATUS_SUCCESS;
}

// ── aclDataType string -> int32_t parser ──
// Supported: ACL_FLOAT, ACL_FLOAT16, ACL_BF16, ACL_DOUBLE, ACL_COMPLEX64, ACL_COMPLEX128
inline int32_t parseAclDataType(const std::string& s)
{
    static const std::unordered_map<std::string, int32_t> t = {
        {"ACL_FLOAT",        static_cast<int32_t>(ACL_FLOAT)},
        {"FP32",             static_cast<int32_t>(ACL_FLOAT)},
        {"ACL_FLOAT16",      static_cast<int32_t>(ACL_FLOAT16)},
        {"FP16",             static_cast<int32_t>(ACL_FLOAT16)},
        {"ACL_BF16",         static_cast<int32_t>(ACL_BF16)},
        {"BF16",             static_cast<int32_t>(ACL_BF16)},
        {"ACL_DOUBLE",       static_cast<int32_t>(ACL_DOUBLE)},
        {"FP64",             static_cast<int32_t>(ACL_DOUBLE)},
        {"ACL_COMPLEX64",    static_cast<int32_t>(ACL_COMPLEX64)},
        {"C64",              static_cast<int32_t>(ACL_COMPLEX64)},
        {"ACL_COMPLEX128",   static_cast<int32_t>(ACL_COMPLEX128)},
        {"C128",             static_cast<int32_t>(ACL_COMPLEX128)},
        {"ACL_INT32",        static_cast<int32_t>(ACL_INT32)},
        {"INT32",            static_cast<int32_t>(ACL_INT32)},
    };
    return parseEnum<int32_t>(s, t, static_cast<int32_t>(ACL_FLOAT));
}

struct RotExParam : public BlasTestParamBase {
    int n = 0;
    int32_t xType = static_cast<int32_t>(ACL_FLOAT);
    int incx = 1;
    int32_t yType = static_cast<int32_t>(ACL_FLOAT);
    int incy = 1;
    int32_t csType = static_cast<int32_t>(ACL_FLOAT);
    int32_t executionType = static_cast<int32_t>(ACL_FLOAT);
    double cValue = 0.5;
    double sValue = 0.5;
    BlasFillMode xFill = BlasFillMode("RANDOM_NORM_1E6");
    BlasFillMode yFill = BlasFillMode("RANDOM_NORM_1E6");

    RotExParam(const csv_map& m) : BlasTestParamBase(m)
    {
        n             = parseInt(ReadMap(m, "n", "0"));
        xType         = parseAclDataType(ReadMap(m, "xType", "ACL_FLOAT"));
        incx          = parseInt(ReadMap(m, "incx", "1"));
        yType         = parseAclDataType(ReadMap(m, "yType", "ACL_FLOAT"));
        incy          = parseInt(ReadMap(m, "incy", "1"));
        csType        = parseAclDataType(ReadMap(m, "csType", "ACL_FLOAT"));
        executionType = parseAclDataType(ReadMap(m, "executionType", "ACL_FLOAT"));
        cValue        = parseDouble(ReadMap(m, "c_value", "0.5"));
        sValue        = parseDouble(ReadMap(m, "s_value", "0.5"));
        xFill         = BlasFillMode(ReadMap(m, "x_fill", "RANDOM_NORM_1E6"));
        yFill         = BlasFillMode(ReadMap(m, "y_fill", "RANDOM_NORM_1E6"));
    }
};
