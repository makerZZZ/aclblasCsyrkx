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

#include <algorithm>
#include <cstdint>

#include "cann_ops_blas.h"

inline bool CsymvComplexIsZero(const aclblasComplex& value) { return value.real == 0.0f && value.imag == 0.0f; }

inline bool CsymvComplexIsOne(const aclblasComplex& value) { return value.real == 1.0f && value.imag == 0.0f; }

struct CsymvCpuArguments {
    int n;
    int lda;
    int incx;
    int incy;
    aclblasFillMode_t uplo;
    const aclblasComplex* alpha;
    const aclblasComplex* a;
    const aclblasComplex* x;
    const aclblasComplex* beta;
    aclblasComplex* y;
};

inline int64_t CsymvVectorIndex(int index, int n, int increment)
{
    const int64_t index64 = static_cast<int64_t>(index);
    const int64_t increment64 = static_cast<int64_t>(increment);
    return increment >= 0 ? index64 * increment64 : (static_cast<int64_t>(n) - 1 - index64) * (-increment64);
}

inline int64_t CsymvMatrixIndex(const CsymvCpuArguments& args, int row, int col)
{
    const int64_t row64 = static_cast<int64_t>(row);
    const int64_t col64 = static_cast<int64_t>(col);
    const int64_t direct = row64 + col64 * args.lda;
    const int64_t mirrored = col64 + row64 * args.lda;
    const bool useDirect = args.uplo == ACLBLAS_UPPER ? row <= col : row >= col;
    return useDirect ? direct : mirrored;
}

inline aclblasStatus_t ValidateCsymvCpuArguments(const CsymvCpuArguments& args)
{
    if (args.uplo != ACLBLAS_UPPER && args.uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (args.lda < std::max(1, args.n) || args.incx == 0 || args.incy == 0 || args.alpha == nullptr ||
        args.beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const bool alphaIsZero = CsymvComplexIsZero(*args.alpha);
    const bool noOp = alphaIsZero && CsymvComplexIsOne(*args.beta);
    if ((!alphaIsZero && (args.a == nullptr || args.x == nullptr)) || (!noOp && args.y == nullptr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline aclblasComplex CsymvCpuAccumulate(const CsymvCpuArguments& args, int row)
{
    aclblasComplex result{0.0f, 0.0f};
    if (CsymvComplexIsZero(*args.alpha)) {
        return result;
    }
    for (int col = 0; col < args.n; ++col) {
        const aclblasComplex aValue = args.a[CsymvMatrixIndex(args, row, col)];
        const aclblasComplex xValue = args.x[CsymvVectorIndex(col, args.n, args.incx)];
        result.real += aValue.real * xValue.real - aValue.imag * xValue.imag;
        result.imag += aValue.real * xValue.imag + aValue.imag * xValue.real;
    }
    return result;
}

inline void CsymvCpuStoreResult(const CsymvCpuArguments& args, int row, const aclblasComplex& accumulator)
{
    const int64_t yIndex = CsymvVectorIndex(row, args.n, args.incy);
    aclblasComplex oldY{0.0f, 0.0f};
    if (!CsymvComplexIsZero(*args.beta)) {
        oldY = args.y[yIndex];
    }
    const float productReal = args.alpha->real * accumulator.real - args.alpha->imag * accumulator.imag;
    const float productImag = args.alpha->real * accumulator.imag + args.alpha->imag * accumulator.real;
    if (CsymvComplexIsOne(*args.beta)) {
        args.y[yIndex].real = productReal + oldY.real;
        args.y[yIndex].imag = productImag + oldY.imag;
    } else {
        args.y[yIndex].real = productReal + args.beta->real * oldY.real - args.beta->imag * oldY.imag;
        args.y[yIndex].imag = productImag + args.beta->real * oldY.imag + args.beta->imag * oldY.real;
    }
}

inline aclblasStatus_t aclblasCsymv_cpu(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* A,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const CsymvCpuArguments args{n, lda, incx, incy, uplo, alpha, A, x, beta, y};
    const aclblasStatus_t status = ValidateCsymvCpuArguments(args);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (CsymvComplexIsZero(*alpha) && CsymvComplexIsOne(*beta)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    for (int row = 0; row < n; ++row) {
        CsymvCpuStoreResult(args, row, CsymvCpuAccumulate(args, row));
    }
    return ACLBLAS_STATUS_SUCCESS;
}
