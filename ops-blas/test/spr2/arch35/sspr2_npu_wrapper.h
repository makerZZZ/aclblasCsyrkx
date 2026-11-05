/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SSPR2_NPU_H
#define SSPR2_NPU_H

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline void Sspr2PrepareVector(
    const float* host, int n, int inc, const float*& upload, int& useInc, size_t& bytes)
{
    if (host == nullptr) {
        upload = nullptr;
        bytes = 0;
        useInc = inc;
        return;
    }
    upload = host;
    useInc = inc;
    size_t absInc = static_cast<size_t>(inc == INT_MIN ? 0 : std::abs(inc));
    bytes = (static_cast<size_t>(n - 1) * absInc + 1) * sizeof(float);
}

inline aclError Sspr2CopyToDevice(void** devPtr, const void* hostPtr, size_t bytes)
{
    aclError ret = aclrtMalloc(devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
        return ret;
    return aclrtMemcpy(*devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
}

inline void Sspr2CleanupDeviceBuffers(void* dX, void* dY, void* dAP)
{
    if (dX)
        aclrtFree(dX);
    if (dY)
        aclrtFree(dY);
    if (dAP)
        aclrtFree(dAP);
}

inline aclblasStatus_t aclblasSspr2_npu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha,
    const float* x, int incx, const float* y, int incy, float* ap)
{
    if (handle == nullptr || n <= 0) {
        return aclblasSspr2(handle, uplo, n, alpha, x, incx, y, incy, ap);
    }

    const float *xUpload = nullptr, *yUpload = nullptr;
    int useIncx = incx, useIncy = incy;
    size_t xBytes = 0, yBytes = 0;
    Sspr2PrepareVector(x, n, incx, xUpload, useIncx, xBytes);
    Sspr2PrepareVector(y, n, incy, yUpload, useIncy, yBytes);

    size_t allocN = static_cast<size_t>(std::max(1, n));
    size_t apBytes = allocN * (allocN + 1) / 2 * sizeof(float);

    void *dX = nullptr, *dY = nullptr, *dAP = nullptr;
    if (xUpload != nullptr && Sspr2CopyToDevice(&dX, xUpload, xBytes) != ACL_SUCCESS) {
        Sspr2CleanupDeviceBuffers(dX, dY, dAP);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (yUpload != nullptr && Sspr2CopyToDevice(&dY, yUpload, yBytes) != ACL_SUCCESS) {
        Sspr2CleanupDeviceBuffers(dX, dY, dAP);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (ap != nullptr && Sspr2CopyToDevice(&dAP, ap, apBytes) != ACL_SUCCESS) {
        Sspr2CleanupDeviceBuffers(dX, dY, dAP);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclblasStatus_t ret = aclblasSspr2(
        handle, uplo, n, alpha,
        static_cast<const float*>(dX), useIncx,
        static_cast<const float*>(dY), useIncy,
        static_cast<float*>(dAP));

    aclError syncRet = aclrtSynchronizeDevice();
    if (syncRet != ACL_SUCCESS) {
        Sspr2CleanupDeviceBuffers(dX, dY, dAP);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    if (ret == ACLBLAS_STATUS_SUCCESS && ap != nullptr) {
        aclError copyRet = aclrtMemcpy(ap, apBytes, dAP, apBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (copyRet != ACL_SUCCESS) {
            Sspr2CleanupDeviceBuffers(dX, dY, dAP);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    Sspr2CleanupDeviceBuffers(dX, dY, dAP);
    return ret;
}

#endif // SSPR2_NPU_H
