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

// NPU wrapper — c / s accepted by value, plus an optional csPtrMode that controls where the
// c / s pointers are materialized before being passed to the operator:
//   csPtrMode == "host"   : operator invoked with &c / &s (host stack scalars, v1 behavior).
//   csPtrMode == "device" : c and s each allocated on device (aclrtMalloc + H2D copy), operator
//                           invoked with the device pointers.
//   csPtrMode == "mixed"  : c stays on host (&c), s allocated on device.
// The operator auto-determines each pointer's location via aclrtPointerGetAttributes, so the
// wrapper only needs to construct the pointer at the desired side; the API signature
// (const float* c, const float* s) is unchanged.
//
// Fast paths (passed through to the operator without device allocation):
//   - handle == nullptr  : operator returns ACLBLAS_STATUS_HANDLE_IS_NULLPTR
//   - n <= 0             : operator returns ACLBLAS_STATUS_SUCCESS (no-op)
//   - x == nullptr or y == nullptr (n > 0): operator returns ACLBLAS_STATUS_INVALID_VALUE
//
// Normal path: malloc device buffers covering the stride access range
//   (element span = (n-1)*|inc|+1 for both x and y), H2D copy x/y, optionally H2D copy the
//   device-side c / s scalar(s), call the operator, synchronize, D2H copy back x/y
//   (in-place modification), then free.
//
// Every ACL call is checked; on any failure already-allocated device memory is freed via
// the freeAll lambda before returning a structured error code (no leak).

// Element span covered by stride access: (n-1)*|inc| + 1 elements (>= n when |inc|>=1).
static inline size_t SrotBufElems(int n, int inc)
{
    if (n <= 0)
        return 0;
    int absInc = (inc < 0) ? -inc : inc;
    if (absInc == 0)
        absInc = 1; // inc==0 reuses element 0; a single element span suffices.
    return static_cast<size_t>(n - 1) * static_cast<size_t>(absInc) + 1;
}

// Resolve csPtrMode ("host"/"device"/"mixed") into per-scalar on-device flags.
static inline void SrotResolveCsPtrMode(const std::string& csPtrMode, bool& cOnDevice, bool& sOnDevice)
{
    cOnDevice = (csPtrMode == "device");
    sOnDevice = (csPtrMode == "device") || (csPtrMode == "mixed");
}

// Allocate a device scalar (sizeof(float)) and H2D-copy the host value in. On success
// dScalar is set and scalarArg points to it; on failure dScalar stays nullptr.
// Returns ACLBLAS_STATUS_ALLOC_FAILED / ACLBLAS_STATUS_INTERNAL_ERROR on ACL failure.
static inline aclblasStatus_t SrotPrepareDeviceScalar(float hostVal, void*& dScalar, const float*& scalarArg)
{
    scalarArg = nullptr; // filled on success
    if (aclrtMalloc(&dScalar, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMemcpy(dScalar, sizeof(float), &hostVal, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(dScalar);
        dScalar = nullptr;
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    scalarArg = static_cast<const float*>(dScalar);
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasStatus_t aclblasSrot_npu(
    aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float c, const float s,
    const std::string& csPtrMode = "host")
{
    // Fast path: no device buffers needed (handle null / n<=0 / nullptr x or y). c / s are
    // passed as host pointers here regardless of csPtrMode, since the operator short-circuits
    // before touching them in these cases.
    if (handle == nullptr || n <= 0 || x == nullptr || y == nullptr) {
        return aclblasSrot(handle, n, x, incx, y, incy, &c, &s);
    }

    bool cOnDevice = false;
    bool sOnDevice = false;
    SrotResolveCsPtrMode(csPtrMode, cOnDevice, sOnDevice);

    const size_t xElems = SrotBufElems(n, incx);
    const size_t yElems = SrotBufElems(n, incy);
    const size_t xBytes = xElems * sizeof(float);
    const size_t yBytes = yElems * sizeof(float);

    void* dX = nullptr;
    void* dY = nullptr;
    void* dC = nullptr;
    void* dS = nullptr;

    auto freeAll = [&]() {
        if (dX)
            aclrtFree(dX);
        if (dY)
            aclrtFree(dY);
        if (dC)
            aclrtFree(dC);
        if (dS)
            aclrtFree(dS);
    };

    if (aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        freeAll();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        freeAll();
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    if (aclrtMemcpy(dX, xBytes, x, xBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        freeAll();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtMemcpy(dY, yBytes, y, yBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        freeAll();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    // Construct c / s pointers on the side indicated by csPtrMode. The operator auto-detects
    // each pointer's location, so the wrapper just materializes it where requested.
    const float* cArg = &c;
    const float* sArg = &s;
    if (cOnDevice) {
        aclblasStatus_t st = SrotPrepareDeviceScalar(c, dC, cArg);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            freeAll();
            return st;
        }
    }
    if (sOnDevice) {
        aclblasStatus_t st = SrotPrepareDeviceScalar(s, dS, sArg);
        if (st != ACLBLAS_STATUS_SUCCESS) {
            freeAll();
            return st;
        }
    }

    aclblasStatus_t ret = aclblasSrot(handle, n, static_cast<float*>(dX), incx, static_cast<float*>(dY), incy,
                                      cArg, sArg);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        freeAll();
        return ret;
    }

    if (aclrtSynchronizeDevice() != ACL_SUCCESS) {
        freeAll();
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    if (aclrtMemcpy(x, xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        freeAll();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (aclrtMemcpy(y, yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        freeAll();
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    freeAll();
    return ret;
}
