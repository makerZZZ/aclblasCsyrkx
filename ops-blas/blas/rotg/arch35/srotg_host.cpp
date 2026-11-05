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
 * \file srotg_host.cpp
 * \brief Construct Givens rotation parameters (BLAS Level 1 scalar operation).
 *        Arch35 (ascend950) host-side implementation.
 *
 *        rotg is a pure scalar computation: it reads two scalars (a, b) and writes back
 *        the rotation norm r (into a), a recovery parameter z (into b), together with the
 *        cosine c and sine s of the Givens rotation matrix.
 *
 *        The four pointers (a, b, c, s) must be all-host or all-device; mixed locations
 *        are rejected per the cuBLAS/BLAS standard.
 *
 *        Strategy:
 *        - All host pointers: direct CPU computation (no kernel, no memcpy).
 *        - All device pointers: launch a 1-block SIMT kernel so computation and results
 *          stay on device, ready for downstream ops like rot.
 */

#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "srotg_kernel.h"

// ==========================================================================
// Helper: check pointer location with aclrtPointerGetAttributes
// ==========================================================================
static aclblasStatus_t SrotgCheckPtrLocation(const void* ptr, bool* isDevice)
{
    aclrtPtrAttributes ptrAttr{};
    aclError aclRet = aclrtPointerGetAttributes(ptr, &ptrAttr);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSrotg", "aclrtPointerGetAttributes failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    *isDevice = (ptrAttr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Core srotg algorithm — reference BLAS CPU implementation
//
// Mirrors the device kernel so host and device paths produce identical results.
// Reference: http://www.netlib.org/blas/srotg.f
// ==========================================================================
static void SrotgCpuCompute(float* a, float* b, float* c, float* s)
{
    float sa = *a;
    float sb = *b;

    float absA = sa >= 0.0f ? sa : -sa;
    float absB = sb >= 0.0f ? sb : -sb;

    float roe = (absA > absB) ? sa : sb;
    float scale = absA + absB;

    float r;
    float z;
    float sc;
    float ss;

    if (scale == 0.0f) {
        sc = 1.0f;
        ss = 0.0f;
        r = 0.0f;
        z = 0.0f;
    } else {
        r = (roe >= 0.0f ? 1.0f : -1.0f) * hypotf(sa, sb);
        sc = sa / r;
        ss = sb / r;
        z = 1.0f;
        if (absA > absB) {
            z = ss;
        } else if (sc != 0.0f) {
            z = 1.0f / sc;
        }
    }

    *a = r;
    *b = z;
    *c = sc;
    *s = ss;
}

// ==========================================================================
// Parameter validation
// ==========================================================================
static aclblasStatus_t ValidateSrotgParams(aclblasHandle_t handle, float* a, float* b, float* c, float* s)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSrotg", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        OP_LOGE("aclblasSrotg", "input pointers contain a nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// ==========================================================================
// Public API: aclblasSrotg
//
// All four pointers (a, b, c, s) must reside on the same side:
//   - All host   → CPU computation
//   - All device → device kernel
//   - Mixed      → ACLBLAS_STATUS_INVALID_VALUE
// ==========================================================================
aclblasStatus_t aclblasSrotg(aclblasHandle_t handle, float* a, float* b, float* c, float* s)
{
    aclblasStatus_t status = ValidateSrotgParams(handle, a, b, c, s);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // 约束：四个指针必须同侧（全 host 或全 device）。以 a 的位置为基准，逐个
    // 比对 b/c/s，一旦发现不同侧立即报错返回（fail-fast），不再查询后续指针。
    bool aDev = false;
    status = SrotgCheckPtrLocation(a, &aDev);
    if (status != ACLBLAS_STATUS_SUCCESS) return status;

    const char* aSide = aDev ? "dev" : "host";
    auto expectSameSide = [&](void* ptr, const char* name) -> aclblasStatus_t {
        bool dev = false;
        aclblasStatus_t st = SrotgCheckPtrLocation(ptr, &dev);
        if (st != ACLBLAS_STATUS_SUCCESS) return st;
        if (dev != aDev) {
            OP_LOGE("aclblasSrotg",
                    "mixed host/device pointers not supported (a=%s, %s=%s)",
                    aSide, name, dev ? "dev" : "host");
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
        return ACLBLAS_STATUS_SUCCESS;
    };
    status = expectSameSide(b, "b");
    if (status != ACLBLAS_STATUS_SUCCESS) return status;
    status = expectSameSide(c, "c");
    if (status != ACLBLAS_STATUS_SUCCESS) return status;
    status = expectSameSide(s, "s");
    if (status != ACLBLAS_STATUS_SUCCESS) return status;

    // 四个指针同侧：全 host 走 CPU 计算，全 device 走 kernel
    if (!aDev) {
        SrotgCpuCompute(a, b, c, s);
        return ACLBLAS_STATUS_SUCCESS;
    }

    auto* h = handle;
    OP_LOGI("aclblasSrotg", "launching kernel: blocks=1");
    srotg_kernel_do(
        reinterpret_cast<uint8_t*>(a),
        reinterpret_cast<uint8_t*>(b),
        reinterpret_cast<uint8_t*>(c),
        reinterpret_cast<uint8_t*>(s),
        h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
