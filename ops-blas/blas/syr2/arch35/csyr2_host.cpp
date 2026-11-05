/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>

#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "csyr2_kernel.h"
#include "csyr2_tiling_data.h"
#include "log/log.h"

namespace {

constexpr const char* OP_NAME = "aclblasCsyr2";

bool IsZero(const aclblasComplex& value)
{
    return value.real == 0.0f && value.imag == 0.0f;
}

aclblasStatus_t ValidateStructuralParams(
    aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, int incx, int incy, int lda)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE(OP_NAME, "uplo must be UPPER(121) or LOWER(122), got %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (n < 0) {
        OP_LOGE(OP_NAME, "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE(OP_NAME, "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0) {
        OP_LOGE(OP_NAME, "incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        OP_LOGE(OP_NAME, "lda must be >= max(1, n), got lda=%d, n=%d", lda, n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr) {
        OP_LOGE(OP_NAME, "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t ValidateDataPointers(const aclblasComplex* x, const aclblasComplex* y, const aclblasComplex* A)
{
    if (x == nullptr) {
        OP_LOGE(OP_NAME, "x must not be nullptr for a non-no-op call");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (y == nullptr) {
        OP_LOGE(OP_NAME, "y must not be nullptr for a non-no-op call");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == nullptr) {
        OP_LOGE(OP_NAME, "A must not be nullptr for a non-no-op call");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

Csyr2TilingData CalculateCsyr2Tiling(
    uint32_t numBlocks, int n, int lda, aclblasFillMode_t uplo, const aclblasComplex& alpha, int incx, int incy)
{
    const uint32_t nValue = static_cast<uint32_t>(n);
    Csyr2TilingData tiling{};
    tiling.numThreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(nValue, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    tiling.rowsPerBlock = CeilDiv<uint32_t>(nValue, numBlocks);
    tiling.n = nValue;
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.alphaReal = alpha.real;
    tiling.alphaImag = alpha.imag;
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    return tiling;
}

} // namespace

extern "C" aclblasStatus_t aclblasCsyr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha,
    const aclblasComplex* x, int incx, const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    if (handle == nullptr) {
        OP_LOGE(OP_NAME, "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    const aclblasStatus_t structuralStatus = ValidateStructuralParams(uplo, n, alpha, incx, incy, lda);
    if (structuralStatus != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_NAME, "structural parameter validation failed, status=%d", static_cast<int>(structuralStatus));
        return structuralStatus;
    }

    // Structural validation intentionally precedes quick return.  A legal
    // no-op may pass null device buffers, but it must not hide an invalid
    // enum, dimension, stride, leading dimension, or alpha pointer.
    if (n == 0 || IsZero(*alpha)) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    const aclblasStatus_t pointerStatus = ValidateDataPointers(x, y, A);
    if (pointerStatus != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_NAME, "data pointer validation failed, status=%d", static_cast<int>(pointerStatus));
        return pointerStatus;
    }

    const aclblasStatus_t workspaceStatus =
        EnsureDefaultWorkspace(handle, static_cast<size_t>(n) * sizeof(aclblasComplex));
    if (workspaceStatus != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_NAME, "workspace preparation failed, status=%d", static_cast<int>(workspaceStatus));
        return workspaceStatus;
    }
    auto* workspace = static_cast<uint8_t*>(GetEffectiveWorkspace(handle));
    static_cast<void>(workspace);

    const uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE(OP_NAME, "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    const uint32_t nValue = static_cast<uint32_t>(n);
    const bool useAiv = incx == 1 && incy == 1 && n <= 4096;
    const uint32_t numBlocks = useAiv ? std::min(nValue, aivCoreNum) : std::max(
        std::min(CeilDiv<uint32_t>(nValue, SIMT_MIN_THREAD_NUM), aivCoreNum), static_cast<uint32_t>(1));
    const Csyr2TilingData tiling = CalculateCsyr2Tiling(numBlocks, n, lda, uplo, *alpha, incx, incy);

    OP_LOGD(
        OP_NAME,
        "tiling: n=%u lda=%u uplo=%u incx=%ld incy=%ld numThreads=%u rowsPerBlock=%u numBlocks=%u",
        tiling.n, tiling.lda, tiling.uplo, tiling.incx, tiling.incy, tiling.numThreads, tiling.rowsPerBlock, numBlocks);
    const auto launch = useAiv ? csyr2_aiv_kernel_do : csyr2_kernel_do;
    launch(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y)), reinterpret_cast<uint8_t*>(A), tiling, numBlocks,
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
