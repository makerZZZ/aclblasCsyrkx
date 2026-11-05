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

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

// cs_ptr_mode — controls where the c / s scalar pointers live when passed to the operator.
// Aligned with the design (v2): the operator queries each pointer's location via
// aclrtPointerGetAttributes and treats it independently, so c / s need NOT be on the same side.
//   "host"   : c and s both as host stack scalars (v1 behavior, baseline regression).
//   "device" : c and s both allocated on device HBM (aclrtMalloc + H2D), passed by device ptr.
//   "mixed"  : c on host, s on device (verifies the independent per-pointer determination).
// Default is "host" so pre-existing CSV rows (which omit the column) keep v1 behavior.
// The actual parsing helper used at runtime lives in arch35/srot_npu_wrapper.h
// (SrotResolveCsPtrMode), which is called from aclblasSrot_npu.

// Parameter struct for aclblasSrot.
// Field order matches the aclblasSrot API signature:
//   aclblasSrot(handle, n, x, incx, y, incy, c, s)
// mere_threshold / mare_multiplier come from BlasTestParamBase (CSV columns),
// used in MERE_MARE precision mode (FP32 single benchmark: MERE <= 2^-13, MARE <= 10*2^-13).
struct SrotParam : public BlasTestParamBase {
    int n = 0;
    BlasFillMode x = parseFill("RANDOM_NORM_1");
    int incx = 1;
    BlasFillMode y = parseFill("RANDOM_NORM_1");
    int incy = 1;
    float c = 0.6f;
    float s = 0.8f;
    std::string csPtrMode = "host";

    SrotParam(const csv_map& m) : BlasTestParamBase(m)
    {
        n    = parseInt(ReadMap(m, "n", "0"));
        x    = parseFill(ReadMap(m, "x", "RANDOM_NORM_1"));
        incx = parseInt(ReadMap(m, "incx", "1"));
        y    = parseFill(ReadMap(m, "y", "RANDOM_NORM_1"));
        incy = parseInt(ReadMap(m, "incy", "1"));
        c    = parseFloat(ReadMap(m, "c", "0.6"));
        s    = parseFloat(ReadMap(m, "s", "0.8"));
        csPtrMode = ReadMap(m, "cs_ptr_mode", "host");
    }
};
