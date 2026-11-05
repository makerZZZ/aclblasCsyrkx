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
 * \file csymv_host.cpp
 * \brief CSYMV Host implementation for ascend950 (DAV_3510)
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "csymv_tiling_data.h"

void csymv_kernel_do(
    uint8_t* alpha, uint8_t* a, uint8_t* x, uint8_t* beta, uint8_t* y, uint8_t* workSpace, uint32_t numBlocks,
    const CsymvTilingData& tiling, void* stream);

namespace {

struct ComplexScalarSource {
    aclblasComplex value{0.0f, 0.0f};
    bool isDevice = false;
};

struct CsymvArguments {
    aclblasHandle_t handle;
    aclblasFillMode_t uplo;
    int n;
    const aclblasComplex* alpha;
    const aclblasComplex* a;
    int lda;
    const aclblasComplex* x;
    int incx;
    const aclblasComplex* beta;
    aclblasComplex* y;
    int incy;
};

struct CsymvLaunchPlan {
    CsymvTilingData tiling{};
    aclrtStream stream = nullptr;
    uint8_t* workspace = nullptr;
    uint32_t numBlocks = 0U;
};

inline bool IsZero(const aclblasComplex& value) { return value.real == 0.0f && value.imag == 0.0f; }

inline bool IsOne(const aclblasComplex& value) { return value.real == 1.0f && value.imag == 0.0f; }

aclblasStatus_t PrepareScalarSource(const aclblasComplex* ptr, const char* name, ComplexScalarSource& source)
{
    aclrtPtrAttributes attributes{};
    aclError ret = aclrtPointerGetAttributes(ptr, &attributes);
    if (ret != ACL_SUCCESS) {
        OP_LOGE("aclblasCsymv", "aclrtPointerGetAttributes failed for %s, ret=%d", name, ret);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    source.isDevice = attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE;
    if (!source.isDevice) {
        source.value = *ptr;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ResolveDeviceScalars(
    const aclblasComplex* alpha, const aclblasComplex* beta, aclrtStream stream, ComplexScalarSource& alphaSource,
    ComplexScalarSource& betaSource)
{
    bool copied = false;
    if (alphaSource.isDevice) {
        aclError ret = aclrtMemcpyAsync(
            &alphaSource.value, sizeof(alphaSource.value), alpha, sizeof(*alpha), ACL_MEMCPY_DEVICE_TO_HOST, stream);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasCsymv", "copying alpha from device failed, ret=%d", ret);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        copied = true;
    }
    if (betaSource.isDevice) {
        aclError ret = aclrtMemcpyAsync(
            &betaSource.value, sizeof(betaSource.value), beta, sizeof(*beta), ACL_MEMCPY_DEVICE_TO_HOST, stream);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasCsymv", "copying beta from device failed, ret=%d", ret);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
        copied = true;
    }
    if (copied) {
        aclError ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            OP_LOGE("aclblasCsymv", "synchronizing scalar copies failed, ret=%d", ret);
            return ACLBLAS_STATUS_INTERNAL_ERROR;
        }
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateFixedParams(
    aclblasFillMode_t uplo, int n, int lda, int incx, int incy, const aclblasComplex* alpha, const aclblasComplex* beta)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasCsymv", "invalid uplo=%d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE("aclblasCsymv", "invalid lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0) {
        OP_LOGE("aclblasCsymv", "incx and incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        OP_LOGE("aclblasCsymv", "alpha and beta must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

CsymvTilingData MakeTilingData(
    uint32_t numBlocks, int n, int lda, aclblasFillMode_t uplo, const ComplexScalarSource& alpha,
    const ComplexScalarSource& beta, int incx, int incy)
{
    CsymvTilingData tiling{};
    tiling.nthreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(static_cast<uint32_t>(n), numBlocks), SIMT_MIN_THREAD_NUM),
        SIMT_MAX_THREAD_NUM);
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.alphaReal = alpha.value.real;
    tiling.alphaImag = alpha.value.imag;
    tiling.betaReal = beta.value.real;
    tiling.betaImag = beta.value.imag;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    tiling.alphaIsDevice = alpha.isDevice ? 1U : 0U;
    tiling.betaIsDevice = beta.isDevice ? 1U : 0U;
    tiling.useCoreNum = numBlocks;
    tiling.fastPath = 0U;
    return tiling;
}

aclblasStatus_t ValidateCall(const CsymvArguments& args, bool& complete)
{
    complete = false;
    if (args.handle == nullptr) {
        OP_LOGE("aclblasCsymv", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (args.n < 0) {
        OP_LOGE("aclblasCsymv", "invalid n=%d", args.n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (args.n == 0) {
        complete = true;
        return ACLBLAS_STATUS_SUCCESS;
    }
    return ValidateFixedParams(args.uplo, args.n, args.lda, args.incx, args.incy, args.alpha, args.beta);
}

aclblasStatus_t PrepareExecution(
    const CsymvArguments& args, ComplexScalarSource& alphaSource, ComplexScalarSource& betaSource, bool& complete)
{
    aclblasStatus_t status = PrepareScalarSource(args.alpha, "alpha", alphaSource);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = PrepareScalarSource(args.beta, "beta", betaSource);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    const bool hasMissingDataPointer = args.a == nullptr || args.x == nullptr || args.y == nullptr;
    if (hasMissingDataPointer) {
        status = ResolveDeviceScalars(args.alpha, args.beta, args.handle->stream, alphaSource, betaSource);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    const bool scalarValuesKnown = (!alphaSource.isDevice && !betaSource.isDevice) || hasMissingDataPointer;
    if (!scalarValuesKnown) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const bool alphaIsZero = IsZero(alphaSource.value);
    complete = alphaIsZero && IsOne(betaSource.value);
    if (!alphaIsZero && (args.a == nullptr || args.x == nullptr)) {
        OP_LOGE("aclblasCsymv", "A and x must not be nullptr when alpha is nonzero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!complete && args.y == nullptr) {
        OP_LOGE("aclblasCsymv", "y must not be nullptr when the result is written");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t BuildLaunchPlan(
    const CsymvArguments& args, const ComplexScalarSource& alphaSource, const ComplexScalarSource& betaSource,
    CsymvLaunchPlan& plan)
{
    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0U) {
        OP_LOGE("aclblasCsymv", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const bool fastPathCandidate = !alphaSource.isDevice && !betaSource.isDevice && args.incx == 1 && args.incy == 1 &&
                                   IsZero(betaSource.value) && !IsZero(alphaSource.value) && aivCoreNum >= 40U;
    constexpr size_t regTileSize = 64U;
    const size_t tileCount = CeilDiv<size_t>(static_cast<size_t>(args.n), regTileSize);
    const size_t regWorkspaceBytes = tileCount * tileCount * regTileSize * sizeof(aclblasComplex);
    const bool hasRegWorkspace = regWorkspaceBytes <= GetEffectiveWorkspaceSize(args.handle);
    // The reduce kernel holds one partial vector per tile and one output vector in UB.
    constexpr size_t regReduceOutputBytes = regTileSize * 2U * sizeof(float);
    static_assert(UB_SIZE > regReduceOutputBytes);
    const bool hasRegReduceUb = tileCount <= (UB_SIZE - regReduceOutputBytes) / regReduceOutputBytes;
    constexpr uint32_t regPathMinN = 384U;
    const bool useRegPath = fastPathCandidate && static_cast<uint32_t>(args.n) >= regPathMinN && hasRegWorkspace &&
                            hasRegReduceUb;
    if (useRegPath) {
        constexpr uint32_t rowTilesPerTask = 4U;
        const uint32_t tileCount32 = static_cast<uint32_t>(tileCount);
        const uint32_t rowBlockCount = CeilDiv<uint32_t>(tileCount32, rowTilesPerTask);
        uint32_t taskCount = 0U;
        for (uint32_t colTile = 0U; colTile < tileCount32; ++colTile) {
            const uint32_t diagonalBlock = colTile / rowTilesPerTask;
            taskCount += args.uplo == ACLBLAS_UPPER ? diagonalBlock + 1U : rowBlockCount - diagonalBlock;
        }
        plan.numBlocks = std::min(taskCount, aivCoreNum);
    } else if (fastPathCandidate) {
        constexpr uint32_t cooperativeWarpsPerBlock = SIMT_MIN_THREAD_NUM / 32U;
        plan.numBlocks = std::min(
            CeilDiv<uint32_t>(static_cast<uint32_t>(args.n), cooperativeWarpsPerBlock), aivCoreNum);
    } else {
        plan.numBlocks =
            std::min(CeilDiv<uint32_t>(static_cast<uint32_t>(args.n), SIMT_MIN_THREAD_NUM), aivCoreNum);
    }
    plan.tiling =
        MakeTilingData(plan.numBlocks, args.n, args.lda, args.uplo, alphaSource, betaSource, args.incx, args.incy);
    plan.tiling.fastPath = useRegPath ? 1U : (fastPathCandidate ? 2U : 0U);
    plan.stream = args.handle->stream;
    plan.workspace =
        plan.tiling.fastPath == 1U ? reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(args.handle)) : nullptr;
    return ACLBLAS_STATUS_SUCCESS;
}

void LaunchCsymv(
    const CsymvArguments& args, const ComplexScalarSource& alphaSource, const ComplexScalarSource& betaSource,
    const CsymvLaunchPlan& plan)
{
    OP_LOGD(
        "aclblasCsymv", "tiling: n=%u lda=%u uplo=%u nthreads=%u numBlocks=%u fastPath=%u", plan.tiling.n,
        plan.tiling.lda, plan.tiling.uplo, plan.tiling.nthreads, plan.numBlocks, plan.tiling.fastPath);
    csymv_kernel_do(
        alphaSource.isDevice ? reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(args.alpha)) : nullptr,
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(args.a)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(args.x)),
        betaSource.isDevice ? reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(args.beta)) : nullptr,
        reinterpret_cast<uint8_t*>(args.y), plan.workspace, plan.numBlocks, plan.tiling, plan.stream);
}

aclblasStatus_t ExecuteCsymv(const CsymvArguments& args)
{
    bool complete = false;
    aclblasStatus_t status = ValidateCall(args, complete);
    if (status != ACLBLAS_STATUS_SUCCESS || complete) {
        return status;
    }
    ComplexScalarSource alphaSource;
    ComplexScalarSource betaSource;
    status = PrepareExecution(args, alphaSource, betaSource, complete);
    if (status != ACLBLAS_STATUS_SUCCESS || complete) {
        return status;
    }
    CsymvLaunchPlan plan;
    status = BuildLaunchPlan(args, alphaSource, betaSource, plan);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    LaunchCsymv(args, alphaSource, betaSource, plan);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasCsymv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* A,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    const CsymvArguments args{handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy};
    return ExecuteCsymv(args);
}
