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
#include <complex>
#include <vector>

#include "cann_ops_blas.h"

extern "C" {
void cgetrf_(const int* m, const int* n, std::complex<float>* a, const int* lda, int* ipiv, int* info);
void cgetri_(
    const int* n, std::complex<float>* a, const int* lda, const int* ipiv, std::complex<float>* work, const int* lwork,
    int* info);
}

inline std::complex<float> ToStdComplex(aclblasComplex value) { return {value.real, value.imag}; }

inline aclblasComplex ToAclblasComplex(std::complex<float> value) { return {value.real(), value.imag()}; }

inline aclblasStatus_t ValidateCgetriBatchedCpuParams(
    aclblasHandle_t handle, int n, const aclblasComplex* const Aarray[], int lda, aclblasComplex* const Carray[],
    int ldc, int* infoArray, int batchSize)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0 || batchSize < 0 || lda < std::max(1, n) || ldc < std::max(1, n)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || batchSize == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (Aarray == nullptr || Carray == nullptr || infoArray == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline int CgetriGetrfNoPivot(std::complex<float>* matrix, int n, int lda)
{
    for (int k = 0; k < n; k++) {
        std::complex<float> diagonal = matrix[k + k * lda];
        if (diagonal.real() == 0.0f && diagonal.imag() == 0.0f) {
            return k + 1;
        }
        for (int row = k + 1; row < n; row++) {
            matrix[row + k * lda] /= diagonal;
        }
        for (int col = k + 1; col < n; col++) {
            std::complex<float> upper = matrix[k + col * lda];
            for (int row = k + 1; row < n; row++) {
                matrix[row + col * lda] -= matrix[row + k * lda] * upper;
            }
        }
    }
    return 0;
}

inline int CgetriFactorSingle(std::vector<aclblasComplex>& matrix, int n, int lda, int* pivots, bool usePivot)
{
    std::vector<std::complex<float>> work(static_cast<size_t>(lda) * n);
    for (size_t index = 0; index < work.size(); index++) {
        work[index] = ToStdComplex(matrix[index]);
    }

    int info = 0;
    if (usePivot) {
        cgetrf_(&n, &n, work.data(), &lda, pivots, &info);
    } else {
        info = CgetriGetrfNoPivot(work.data(), n, lda);
    }

    for (size_t index = 0; index < work.size(); index++) {
        matrix[index] = ToAclblasComplex(work[index]);
    }
    return info;
}

inline int CgetriInvertSingle(
    const aclblasComplex* lu, int n, int lda, const int* pivots, aclblasComplex* output, int ldc)
{
    std::vector<std::complex<float>> inverse(static_cast<size_t>(ldc) * n, {0.0f, 0.0f});
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < n; row++) {
            inverse[row + col * ldc] = ToStdComplex(lu[row + col * lda]);
        }
    }

    std::vector<int> identityPivots(static_cast<size_t>(n));
    if (pivots == nullptr) {
        for (int index = 0; index < n; index++) {
            identityPivots[index] = index + 1;
        }
        pivots = identityPivots.data();
    }

    int lwork = -1;
    int info = 0;
    std::complex<float> workQuery{0.0f, 0.0f};
    cgetri_(&n, inverse.data(), &ldc, pivots, &workQuery, &lwork, &info);
    if (info != 0) {
        return info;
    }

    lwork = std::max(1, static_cast<int>(workQuery.real()));
    std::vector<std::complex<float>> work(static_cast<size_t>(lwork));
    cgetri_(&n, inverse.data(), &ldc, pivots, work.data(), &lwork, &info);
    if (info == 0) {
        for (size_t index = 0; index < inverse.size(); index++) {
            output[index] = ToAclblasComplex(inverse[index]);
        }
    }
    return info;
}

inline aclblasStatus_t aclblasCgetriBatchedCpu(
    aclblasHandle_t handle, int n, const aclblasComplex* const Aarray[], int lda, const int* PivotArray,
    aclblasComplex* const Carray[], int ldc, int* infoArray, int batchSize)
{
    aclblasStatus_t status = ValidateCgetriBatchedCpuParams(handle, n, Aarray, lda, Carray, ldc, infoArray, batchSize);
    if (status != ACLBLAS_STATUS_SUCCESS || n == 0 || batchSize == 0) {
        return status;
    }

    for (int batch = 0; batch < batchSize; batch++) {
        const int* pivots = PivotArray == nullptr ? nullptr : PivotArray + static_cast<size_t>(batch) * n;
        infoArray[batch] = CgetriInvertSingle(Aarray[batch], n, lda, pivots, Carray[batch], ldc);
    }
    return ACLBLAS_STATUS_SUCCESS;
}
