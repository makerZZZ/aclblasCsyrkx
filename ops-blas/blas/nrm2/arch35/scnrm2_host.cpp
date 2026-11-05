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
 * \file scnrm2_host.cpp
 * \brief aclblasScnrm2 Host-side dispatch for ascend950 (arch35).
 */

#include <algorithm>
#include <cstdint>
#include <climits>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "scnrm2_tiling_data.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_norm.h"

namespace {

static aclblasStatus_t ValidateScnrm2Params(
    aclblasHandle_t handle, int n, int incx, const aclblasComplex* x, const float* result)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasScnrm2", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasScnrm2", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasScnrm2", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == INT32_MIN) {
        OP_LOGE("aclblasScnrm2", "incx must not be INT32_MIN");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n > 0 && x == nullptr) {
        OP_LOGE("aclblasScnrm2", "x must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t WriteZero(float* result)
{
    float zero = 0.0f;
    aclError aclRet = aclrtMemcpy(result, sizeof(float), &zero, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasScnrm2", "aclrtMemcpy zero result failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

struct AclnnNormDescriptors {
    aclTensor* input = nullptr;
    aclTensor* output = nullptr;
    aclScalar* ord = nullptr;
    aclIntArray* dims = nullptr;
};

static void DestroyAclnnNormDescriptors(AclnnNormDescriptors& d)
{
    if (d.input != nullptr) aclDestroyTensor(d.input);
    if (d.output != nullptr) aclDestroyTensor(d.output);
    if (d.ord != nullptr) aclDestroyScalar(d.ord);
    if (d.dims != nullptr) aclDestroyIntArray(d.dims);
}

static bool CreateAclnnNormDescriptors(
    int n, const aclblasComplex* x, float* result, AclnnNormDescriptors& d)
{
    int64_t inputShape[1] = {static_cast<int64_t>(n) * 2};
    int64_t inputStride[1] = {1};
    float ordValue = 2.0f;
    int64_t dimValue = 0;

    d.input = aclCreateTensor(inputShape, 1, ACL_FLOAT, inputStride, 0, ACL_FORMAT_ND,
        inputShape, 1, const_cast<aclblasComplex*>(x));
    d.output = aclCreateTensor(nullptr, 0, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
        nullptr, 0, result);
    d.ord = aclCreateScalar(&ordValue, ACL_FLOAT);
    d.dims = aclCreateIntArray(&dimValue, 1);
    if (d.input == nullptr || d.output == nullptr || d.ord == nullptr || d.dims == nullptr) {
        OP_LOGW("aclblasScnrm2", "failed to create aclnnNorm descriptors, falling back to custom kernel");
        DestroyAclnnNormDescriptors(d);
        return false;
    }
    return true;
}

static aclblasStatus_t TryLaunchAclnnNorm(
    _aclblas_handle* h, int n, const aclblasComplex* x, float* result)
{
    AclnnNormDescriptors d;
    if (!CreateAclnnNormDescriptors(n, x, result, d)) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclnnRet = aclnnNormGetWorkspaceSize(
        d.input, d.ord, d.dims, false, d.output, &workspaceSize, &executor);
    aclblasStatus_t status = ACLBLAS_STATUS_SUCCESS;
    if (aclnnRet != OK) {
        OP_LOGW("aclblasScnrm2",
            "aclnnNormGetWorkspaceSize failed, ret=%d, falling back to custom kernel", static_cast<int>(aclnnRet));
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    } else if (workspaceSize > GetEffectiveWorkspaceSize(h)) {
        OP_LOGW("aclblasScnrm2", "aclnnNorm workspace %zu > handle %zu, falling back to custom kernel",
            static_cast<size_t>(workspaceSize), GetEffectiveWorkspaceSize(h));
        status = ACLBLAS_STATUS_INTERNAL_ERROR;
    } else {
        void* workspace = workspaceSize == 0 ? nullptr : GetEffectiveWorkspace(h);
        aclnnRet = aclnnNorm(workspace, workspaceSize, executor, h->stream);
        if (aclnnRet != OK) {
            OP_LOGW("aclblasScnrm2", "aclnnNorm failed, ret=%d, falling back to custom kernel",
                static_cast<int>(aclnnRet));
            status = ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    DestroyAclnnNormDescriptors(d);

    return status;
}

static Scnrm2TilingData CalcScnrm2TilingData(int n, int incx, uint32_t aivCoreNum)
{
    Scnrm2TilingData tiling{};
    tiling.n = n;
    tiling.incx = incx;
    tiling.maxDataCount = SCNRM2_MAX_DATA_COUNT;

    bool useAiv = (incx == 1 && n >= 64);
    uint32_t workItems = useAiv ? static_cast<uint32_t>(n) * 2u : static_cast<uint32_t>(n);
    uint32_t useCoreNum = std::min(workItems, aivCoreNum);
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }
    if (useCoreNum > SCNRM2_MAX_CORE_NUM) {
        useCoreNum = SCNRM2_MAX_CORE_NUM;
    }

    tiling.useCoreNum = useCoreNum;
    tiling.batchPerCore = workItems / useCoreNum;
    tiling.remain = workItems % useCoreNum;
    if (!useAiv) {
        uint32_t maxItemsPerCore = tiling.batchPerCore + (tiling.remain > 0 ? 1 : 0);
        tiling.nthreads = std::min(CeilAlign<uint32_t>(maxItemsPerCore, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    }
    return tiling;
}

static aclblasStatus_t LaunchScnrm2Kernel(
    _aclblas_handle* h, const aclblasComplex* x, float* result, const Scnrm2TilingData& tiling)
{
    size_t workspaceBytes = static_cast<size_t>(tiling.useCoreNum) * 2 * sizeof(float);
    CHECK_RET(
        workspaceBytes <= GetEffectiveWorkspaceSize(h),
        OP_LOGE("aclblasScnrm2", "workspace %zu > handle %zu", workspaceBytes, GetEffectiveWorkspaceSize(h));
        return ACLBLAS_STATUS_ALLOC_FAILED);

    auto* workspace = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    OP_LOGD("aclblasScnrm2",
        "tiling: n=%d incx=%d useCoreNum=%u maxDataCount=%u batchPerCore=%u remain=%u nthreads=%u",
        tiling.n, tiling.incx, tiling.useCoreNum, tiling.maxDataCount,
        tiling.batchPerCore, tiling.remain, tiling.nthreads);

    // AIV and SIMT write scaled partials. After aclnnNorm, both kernels skip
    // ordinary results and only recompute numerically sensitive cases.
    scnrm2_kernel_do(reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
        reinterpret_cast<uint8_t*>(result), workspace, tiling, tiling.useCoreNum, h->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasScnrm2(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, float* result)
{
    OP_LOGD("aclblasScnrm2", "entry: n=%d, incx=%d", n, incx);

    aclblasStatus_t status = ValidateScnrm2Params(handle, n, incx, x, result);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    if (n <= 0) {
        return WriteZero(result);
    }

    bool checkNormResult = false;
    if (incx == 1 && n >= 2048) {
        aclblasStatus_t aclnnStatus = TryLaunchAclnnNorm(handle, n, x, result);
        if (aclnnStatus == ACLBLAS_STATUS_SUCCESS) {
            checkNormResult = true;
        }
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasScnrm2", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    Scnrm2TilingData tiling = CalcScnrm2TilingData(n, incx, aivCoreNum);
    tiling.checkNormResult = checkNormResult;
    return LaunchScnrm2Kernel(handle, x, result, tiling);
}
