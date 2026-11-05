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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "blas_test.h"
#include "cgetri_batched_golden.h"
#include "cgetri_batched_npu_wrapper.h"
#include "cgetri_batched_param.h"
#include "csv_loader.h"

class CgetriBatchedTest : public BlasTest<CgetriBatchedParam> {};

namespace {

constexpr float CGETRI_RTOL = 0.0009765625f;
constexpr float CGETRI_ATOL = 0.0000152587890625f;
constexpr double CGETRI_REQUIRED_MATCHED_RATIO = 0.99;
constexpr double CGETRI_MAX_CONDITION_NUMBER = 1.0e6;
constexpr float CGETRI_BASE_MAX_ABS_ERROR = 0.01f;
constexpr aclblasComplex CGETRI_PADDING_SENTINEL{12345.0f, -12345.0f};

bool IsSameComplex(aclblasComplex lhs, aclblasComplex rhs) { return lhs.real == rhs.real && lhs.imag == rhs.imag; }

aclblasComplex ScaleComplex(aclblasComplex value, float scale) { return {value.real * scale, value.imag * scale}; }

aclblasComplex SampleComplex(std::mt19937& generator, bool useNormal)
{
    if (useNormal) {
        std::normal_distribution<float> distribution(0.0f, 1.0f);
        return {distribution(generator), distribution(generator)};
    }
    std::uniform_real_distribution<float> distribution(-5.0f, 5.0f);
    return {distribution(generator), distribution(generator)};
}

void FillGeneralMatrix(std::vector<aclblasComplex>& matrix, int n, int lda, std::mt19937& generator, bool useNormal)
{
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < n; row++) {
            matrix[row + col * lda] = SampleComplex(generator, useNormal);
        }
    }
}

void StrengthenDiagonal(std::vector<aclblasComplex>& matrix, int n, int lda)
{
    float boost = std::max(10.0f, 10.0f * static_cast<float>(n));
    for (int index = 0; index < n; index++) {
        matrix[index + index * lda].real += boost;
    }
}

void FillTriangularMatrix(
    std::vector<aclblasComplex>& matrix, int n, int lda, std::mt19937& generator, bool useNormal, bool lower)
{
    for (int col = 0; col < n; col++) {
        int firstRow = lower ? col : 0;
        int rowEnd = lower ? n : col + 1;
        for (int row = firstRow; row < rowEnd; row++) {
            matrix[row + col * lda] = SampleComplex(generator, useNormal);
        }
    }
    StrengthenDiagonal(matrix, n, lda);
}

void FillIllConditionedMatrix(std::vector<aclblasComplex>& matrix, int n, int lda)
{
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < n; row++) {
            matrix[row + col * lda] = {1.0f / static_cast<float>(row + col + 1), 0.0f};
        }
    }
    // 对角偏移把 n=16 Hilbert 矩阵的条件数从约 5.8e9 降至约 1.9e3，满足任务书要求。
    for (int index = 0; index < n; index++) {
        matrix[index + index * lda].real += 1.0e-3f;
    }
}

void FillCgetriMatrix(
    std::vector<aclblasComplex>& matrix, CgetriMatrixType type, int n, int lda, std::mt19937& generator, bool useNormal)
{
    switch (type) {
        case CgetriMatrixType::IDENTITY:
            for (int index = 0; index < n; index++)
                matrix[index + index * lda] = {1.0f, 0.0f};
            break;
        case CgetriMatrixType::LOWER_BIDIAGONAL:
            for (int col = 0; col < n; col++) {
                matrix[col + col * lda] = {1.0f, 0.0f};
                if (col + 1 < n)
                    matrix[col + 1 + col * lda] = {0.1f, 0.2f};
            }
            break;
        case CgetriMatrixType::DIAGONAL:
        case CgetriMatrixType::MIXED_DIAGONAL:
            for (int index = 0; index < n; index++) {
                aclblasComplex value = SampleComplex(generator, useNormal);
                value.real += value.real >= 0.0f ? 2.0f : -2.0f;
                matrix[index + index * lda] = value;
            }
            break;
        case CgetriMatrixType::UPPER_TRIANGULAR:
            FillTriangularMatrix(matrix, n, lda, generator, useNormal, false);
            break;
        case CgetriMatrixType::LOWER_TRIANGULAR:
            FillTriangularMatrix(matrix, n, lda, generator, useNormal, true);
            break;
        case CgetriMatrixType::ILL_CONDITIONED:
            FillIllConditionedMatrix(matrix, n, lda);
            break;
        default:
            FillGeneralMatrix(matrix, n, lda, generator, useNormal);
            StrengthenDiagonal(matrix, n, lda);
            break;
    }
}

void PermuteDenseMatrix(std::vector<aclblasComplex>& matrix, int n, int lda, int batch)
{
    if (n <= 1) {
        return;
    }
    auto unpermuted = matrix;
    int shift = batch % (n - 1) + 1;
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < n; row++) {
            int sourceRow = row + shift;
            if (sourceRow >= n) {
                sourceRow -= n;
            }
            matrix[row + col * lda] = unpermuted[sourceRow + col * lda];
        }
    }
}

int MixedDiagonalInfo(int n, int batch)
{
    // 在取模分支显式检查非零，便于静态检查验证除数。
    if (n >= 0 && n != 0) {
        return batch % 3 == 0 ? 0 : (batch % 3 == 1 ? (batch / 3) % n + 1 : n);
    }
    throw std::invalid_argument("Mixed diagonal matrices require n > 0");
}

void InjectCgetriSingularity(std::vector<aclblasComplex>& matrix, CgetriMatrixType type, int n, int lda, int batch)
{
    if (type == CgetriMatrixType::MIXED_DIAGONAL && batch % 3 != 0) {
        // 覆盖成功矩阵，以及首部、中间、末尾的零主元。
        int zero = MixedDiagonalInfo(n, batch) - 1;
        matrix[zero + zero * lda] = {0.0f, 0.0f};
    }
    if (type == CgetriMatrixType::SINGULAR_ZERO_COL || (type == CgetriMatrixType::MIXED && (batch & 1) != 0)) {
        for (int row = 0; row < n; row++)
            matrix[row] = {0.0f, 0.0f};
    } else if (type == CgetriMatrixType::SINGULAR_DEPENDENT_ROW && n > 1) {
        for (int col = 0; col < n; col++) {
            matrix[1 + col * lda] = ScaleComplex(matrix[col * lda], 2.0f);
        }
    }
}

std::vector<aclblasComplex> MakeCgetriMatrix(const CgetriBatchedParam& param, int n, int lda, int batch)
{
    if (n <= 0 || lda < n) {
        throw std::invalid_argument("Matrix generation requires n > 0 and lda >= n");
    }
    std::vector<aclblasComplex> matrix(static_cast<size_t>(lda) * n, {0.0f, 0.0f});
    std::mt19937 generator(param.randomSeed + static_cast<uint32_t>(batch) * 7919u);
    bool useNormal = ((param.randomSeed + static_cast<uint32_t>(batch)) & 1u) != 0;
    FillCgetriMatrix(matrix, param.matrixType, n, lda, generator, useNormal);
    if (param.matrixType == CgetriMatrixType::PERMUTED_DENSE) {
        PermuteDenseMatrix(matrix, n, lda, batch);
    }
    InjectCgetriSingularity(matrix, param.matrixType, n, lda, batch);
    return matrix;
}

bool IsBatchInvariantMatrix(CgetriMatrixType matrixType)
{
    return matrixType == CgetriMatrixType::IDENTITY || matrixType == CgetriMatrixType::LOWER_BIDIAGONAL ||
           matrixType == CgetriMatrixType::ILL_CONDITIONED;
}

struct CgetriTestData {
    std::vector<std::vector<aclblasComplex>> original;
    std::vector<std::vector<aclblasComplex>> lu;
    std::vector<const aclblasComplex*> luPointers;
    std::vector<std::vector<aclblasComplex>> actual;
    std::vector<aclblasComplex*> actualPointers;
    std::vector<int> pivots;
    std::vector<int> getrfInfo;
    std::vector<int> actualInfo;
};

void PrepareCgetriTestData(
    CgetriTestData& data, const CgetriBatchedParam& param, int n, int lda, int ldc, int batchSize, bool usePivot)
{
    data.original.resize(static_cast<size_t>(batchSize));
    data.lu.resize(static_cast<size_t>(batchSize));
    data.luPointers.resize(static_cast<size_t>(batchSize));
    data.actual.resize(static_cast<size_t>(batchSize));
    data.actualPointers.resize(static_cast<size_t>(batchSize));
    data.pivots.resize(usePivot ? static_cast<size_t>(n) * batchSize : 0);
    data.getrfInfo.resize(static_cast<size_t>(batchSize), -1);
    data.actualInfo.resize(static_cast<size_t>(batchSize), -1);

    for (int batch = 0; batch < batchSize; batch++) {
        int* pivot = usePivot ? data.pivots.data() + static_cast<size_t>(batch) * n : nullptr;
        bool reuseFirstBatch = batch > 0 && IsBatchInvariantMatrix(param.matrixType);
        if (reuseFirstBatch) {
            data.original[batch] = data.original[0];
            data.lu[batch] = data.lu[0];
            data.getrfInfo[batch] = data.getrfInfo[0];
            if (pivot != nullptr) {
                std::copy_n(data.pivots.data(), n, pivot);
            }
        } else {
            data.original[batch] = MakeCgetriMatrix(param, n, lda, batch);
            data.lu[batch] = data.original[batch];
            if (pivot != nullptr) {
                for (int index = 0; index < n; index++)
                    pivot[index] = index + 1;
            }
            data.getrfInfo[batch] = CgetriFactorSingle(data.lu[batch], n, lda, pivot, usePivot);
            if (param.matrixType == CgetriMatrixType::PERMUTED_DENSE && usePivot) {
                int swaps = 0;
                for (int index = 0; index < n; index++)
                    swaps += pivot[index] != index + 1;
                EXPECT_GT(swaps, 0) << param.caseName << " must exercise non-identity pivoting";
            }
            if (param.matrixType == CgetriMatrixType::MIXED_DIAGONAL) {
                int expectedInfo = MixedDiagonalInfo(n, batch);
                EXPECT_EQ(data.getrfInfo[batch], expectedInfo) << param.caseName << " batch=" << batch;
            }
        }
        data.luPointers[batch] = data.lu[batch].data();

        data.actual[batch].assign(static_cast<size_t>(ldc) * n, CGETRI_PADDING_SENTINEL);
        for (int col = 0; col < n; col++) {
            std::fill_n(data.actual[batch].data() + static_cast<size_t>(col) * ldc, n, aclblasComplex{0.0f, 0.0f});
        }
        data.actualPointers[batch] = data.actual[batch].data();
    }
}

struct ComponentStats {
    size_t matched = 0;
    size_t total = 0;
    float maxAbsError = 0.0f;
    float maxAllowedAbsError = CGETRI_BASE_MAX_ABS_ERROR;
    bool finite = true;
};

void UpdateComponentStats(float actual, float golden, ComponentStats& stats)
{
    stats.total++;
    if (!std::isfinite(actual) || !std::isfinite(golden)) {
        stats.finite = false;
        return;
    }
    float error = std::abs(actual - golden);
    stats.maxAbsError = std::max(stats.maxAbsError, error);
    float tolerance = CGETRI_ATOL + CGETRI_RTOL * std::abs(golden);
    if (error <= tolerance)
        stats.matched++;

    float next = std::nextafter(golden, std::numeric_limits<float>::infinity());
    float ulp = std::abs(next - golden);
    stats.maxAllowedAbsError = std::max(stats.maxAllowedAbsError, 32.0f * ulp);
}

void VerifyComponentStats(const ComponentStats& stats, const char* component, const std::string& caseName, int batch)
{
    double ratio = stats.total == 0 ? 0.0 : static_cast<double>(stats.matched) / stats.total;
    EXPECT_TRUE(stats.finite) << caseName << " batch=" << batch << " " << component << " contains Inf/NaN";
    EXPECT_GE(ratio, CGETRI_REQUIRED_MATCHED_RATIO)
        << caseName << " batch=" << batch << " " << component << " matched_ratio=" << ratio;
    EXPECT_LE(stats.maxAbsError, stats.maxAllowedAbsError)
        << caseName << " batch=" << batch << " " << component << " max_abs_error=" << stats.maxAbsError
        << " limit=" << stats.maxAllowedAbsError;
}

void MergeComponentStats(const ComponentStats& source, ComponentStats& target)
{
    target.matched += source.matched;
    target.total += source.total;
    target.maxAbsError = std::max(target.maxAbsError, source.maxAbsError);
    target.maxAllowedAbsError = std::max(target.maxAllowedAbsError, source.maxAllowedAbsError);
    target.finite = target.finite && source.finite;
}

double ComponentMatchedRatio(const ComponentStats& stats)
{
    return stats.total == 0 ? 0.0 : static_cast<double>(stats.matched) / stats.total;
}

double MatrixInfinityNorm(const std::vector<aclblasComplex>& matrix, int n, int ld)
{
    double norm = 0.0;
    for (int row = 0; row < n; row++) {
        double rowSum = 0.0;
        for (int col = 0; col < n; col++) {
            const aclblasComplex& value = matrix[row + col * ld];
            rowSum += std::hypot(static_cast<double>(value.real), static_cast<double>(value.imag));
        }
        norm = std::max(norm, rowSum);
    }
    return norm;
}

double VerifyConditionNumber(
    const std::vector<aclblasComplex>& original, int lda, const std::vector<aclblasComplex>& inverse, int ldc, int n,
    const std::string& caseName, int batch)
{
    double conditionNumber = MatrixInfinityNorm(original, n, lda) * MatrixInfinityNorm(inverse, n, ldc);
    EXPECT_TRUE(std::isfinite(conditionNumber)) << caseName << " batch=" << batch << " condition number is not finite";
    EXPECT_LE(conditionNumber, CGETRI_MAX_CONDITION_NUMBER)
        << caseName << " batch=" << batch << " condition_number_inf=" << conditionNumber;
    return conditionNumber;
}

void VerifyCgetriBatch(
    const CgetriTestData& data, const std::vector<aclblasComplex>& golden, int n, int ldc, const std::string& caseName,
    int batch, ComponentStats& realStats, ComponentStats& imagStats)
{
    bool checkTailColumns = caseName.rfind("TC_RG_TAIL_", 0) == 0;
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < n; row++) {
            size_t index = static_cast<size_t>(row) + static_cast<size_t>(col) * ldc;
            UpdateComponentStats(data.actual[batch][index].real, golden[index].real, realStats);
            UpdateComponentStats(data.actual[batch][index].imag, golden[index].imag, imagStats);
            if (checkTailColumns && col >= 64) {
                // 尾部 1..8 列逐元素检查，避免漏算被总体匹配比例掩盖。
                const auto& actual = data.actual[batch][index];
                const auto& expected = golden[index];
                EXPECT_LE(std::abs(actual.real - expected.real), CGETRI_ATOL + CGETRI_RTOL * std::abs(expected.real))
                    << caseName << " batch=" << batch << " row=" << row << " col=" << col;
                EXPECT_LE(std::abs(actual.imag - expected.imag), CGETRI_ATOL + CGETRI_RTOL * std::abs(expected.imag))
                    << caseName << " batch=" << batch << " row=" << row << " col=" << col;
            }
        }
        for (int row = n; row < ldc; row++) {
            size_t index = static_cast<size_t>(row) + static_cast<size_t>(col) * ldc;
            EXPECT_TRUE(IsSameComplex(data.actual[batch][index], CGETRI_PADDING_SENTINEL))
                << caseName << " batch=" << batch << " output padding modified at index=" << index;
        }
    }
    VerifyComponentStats(realStats, "real", caseName, batch);
    VerifyComponentStats(imagStats, "imag", caseName, batch);
}

void VerifyCgetriOutput(
    const CgetriTestData& data, const std::vector<std::vector<aclblasComplex>>& golden,
    const std::vector<int>& goldenInfo, int n, int lda, int ldc, int batchSize, const std::string& caseName)
{
    ComponentStats caseRealStats;
    ComponentStats caseImagStats;
    double maxConditionNumber = 0.0;
    int comparedBatches = 0;
    int singularBatches = 0;
    for (int batch = 0; batch < batchSize; batch++) {
        EXPECT_EQ(data.actualInfo[batch], goldenInfo[batch]) << caseName << " batch=" << batch << " info mismatch";
        if (goldenInfo[batch] != 0) {
            singularBatches++;
            continue;
        }

        double conditionNumber =
            VerifyConditionNumber(data.original[batch], lda, golden[batch], ldc, n, caseName, batch);
        maxConditionNumber = std::max(maxConditionNumber, conditionNumber);
        comparedBatches++;

        ComponentStats realStats;
        ComponentStats imagStats;
        VerifyCgetriBatch(data, golden[batch], n, ldc, caseName, batch, realStats, imagStats);
        MergeComponentStats(realStats, caseRealStats);
        MergeComponentStats(imagStats, caseImagStats);
    }

    std::cout << std::setprecision(10) << "ACCURACY_RESULT case=" << caseName << " compared_batches=" << comparedBatches
              << " singular_batches=" << singularBatches;
    if (comparedBatches > 0) {
        std::cout << " condition_inf_max=" << maxConditionNumber
                  << " real_ratio=" << ComponentMatchedRatio(caseRealStats)
                  << " real_max_abs=" << caseRealStats.maxAbsError
                  << " real_abs_limit=" << caseRealStats.maxAllowedAbsError
                  << " imag_ratio=" << ComponentMatchedRatio(caseImagStats)
                  << " imag_max_abs=" << caseImagStats.maxAbsError
                  << " imag_abs_limit=" << caseImagStats.maxAllowedAbsError
                  << " finite=" << (caseRealStats.finite && caseImagStats.finite ? 1 : 0);
    }
    std::cout << std::endl;
}

void RunErrorCase(const CgetriBatchedParam& param, aclblasHandle_t validHandle)
{
    aclblasComplex matrix{1.0f, 0.0f};
    const aclblasComplex* inputPointers[1] = {&matrix};
    aclblasComplex* outputPointers[1] = {&matrix};
    int info = 0;
    bool nullA = param.matrixType == CgetriMatrixType::NULLPTR_AARRAY;
    bool nullC = param.matrixType == CgetriMatrixType::NULLPTR_CARRAY;
    bool nullInfo = param.matrixType == CgetriMatrixType::NULLPTR_INFOARRAY;
    bool nullHandle = param.matrixType == CgetriMatrixType::NULLPTR_HANDLE;

    aclblasStatus_t status = aclblasCgetriBatched(
        nullHandle ? nullptr : validHandle, param.n, nullA ? nullptr : inputPointers, param.lda, nullptr,
        nullC ? nullptr : outputPointers, param.ldc, nullInfo ? nullptr : &info, param.batchSize);
    EXPECT_EQ(status, param.expectResult) << param.caseName;
}

void RunNoOpCase(const CgetriBatchedParam& param, aclblasHandle_t handle)
{
    aclblasStatus_t status = aclblasCgetriBatched(
        handle, param.n, nullptr, param.lda, nullptr, nullptr, param.ldc, nullptr, param.batchSize);
    EXPECT_EQ(status, param.expectResult) << param.caseName;
}

void RunAccuracyCase(const CgetriBatchedParam& param, aclblasHandle_t handle, aclrtStream stream)
{
    bool usePivot = param.pivotMode == CgetriPivotMode::PIVOT;
    CgetriTestData data;
    PrepareCgetriTestData(data, param, param.n, param.lda, param.ldc, param.batchSize, usePivot);

    std::vector<std::vector<aclblasComplex>> golden(static_cast<size_t>(param.batchSize));
    std::vector<aclblasComplex*> goldenPointers(static_cast<size_t>(param.batchSize));
    for (int batch = 0; batch < param.batchSize; batch++) {
        golden[batch].assign(static_cast<size_t>(param.ldc) * param.n, CGETRI_PADDING_SENTINEL);
        goldenPointers[batch] = golden[batch].data();
    }
    std::vector<int> goldenInfo(static_cast<size_t>(param.batchSize), -1);
    int goldenBatchSize = IsBatchInvariantMatrix(param.matrixType) && param.batchSize > 0 ? 1 : param.batchSize;
    aclblasStatus_t goldenStatus = aclblasCgetriBatchedCpu(
        handle, param.n, data.luPointers.data(), param.lda, usePivot ? data.pivots.data() : nullptr,
        goldenPointers.data(), param.ldc, goldenInfo.data(), goldenBatchSize);
    ASSERT_EQ(goldenStatus, ACLBLAS_STATUS_SUCCESS) << param.caseName;
    for (int batch = goldenBatchSize; batch < param.batchSize; batch++) {
        golden[batch] = golden[0];
        goldenInfo[batch] = goldenInfo[0];
    }

    aclblasStatus_t status = aclblasCgetriBatchedNpu(
        handle, param.n, data.luPointers.data(), param.lda, usePivot ? data.pivots.data() : nullptr,
        data.actualPointers.data(), param.ldc, data.actualInfo.data(), param.batchSize, stream);
    ASSERT_EQ(status, param.expectResult) << param.caseName;
    if (status == ACLBLAS_STATUS_SUCCESS) {
        VerifyCgetriOutput(data, golden, goldenInfo, param.n, param.lda, param.ldc, param.batchSize, param.caseName);
        if (param.matrixType == CgetriMatrixType::LOWER_BIDIAGONAL) {
            // For A = I + a*S, inv(A)'s first subdiagonal is exactly -a.
            // Check every such entry: a lost forward step can affect too few
            // elements to fail the aggregate 99% ratio for a larger matrix.
            for (int batch = 0; batch < param.batchSize; batch++) {
                for (int col = 0; col + 1 < param.n; col++) {
                    const auto& value = data.actual[batch][col + 1 + col * param.ldc];
                    EXPECT_NEAR(value.real, -0.1f, 1.0e-6f)
                        << param.caseName << " batch=" << batch << " column=" << col;
                    EXPECT_NEAR(value.imag, -0.2f, 1.0e-6f)
                        << param.caseName << " batch=" << batch << " column=" << col;
                }
            }
        }
    }
}

bool IsSpecialValueCase(CgetriMatrixType matrixType)
{
    return matrixType == CgetriMatrixType::INF_INPUT || matrixType == CgetriMatrixType::NAN_INPUT;
}

void VerifySpecialOutput(
    const CgetriBatchedParam& param, const std::vector<std::vector<aclblasComplex>>& output,
    const std::vector<int>& info)
{
    const int n = param.n;
    const int ldc = param.ldc;
    const int batchSize = param.batchSize;
    for (int batch = 0; batch < batchSize; batch++) {
        EXPECT_EQ(info[batch], 0) << param.caseName << " batch=" << batch;
        bool sawInf = false;
        bool sawNan = false;
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < n; row++) {
                const aclblasComplex value = output[batch][row + col * ldc];
                sawInf = sawInf || std::isinf(value.real) || std::isinf(value.imag);
                sawNan = sawNan || std::isnan(value.real) || std::isnan(value.imag);
            }
        }
        if (param.matrixType == CgetriMatrixType::INF_INPUT) {
            EXPECT_TRUE(sawInf || sawNan) << param.caseName << " batch=" << batch;
        } else {
            EXPECT_TRUE(sawNan) << param.caseName << " batch=" << batch;
        }
        std::cout << "SPECIAL_RESULT case=" << param.caseName << " batch=" << batch
                  << " kind=" << (param.matrixType == CgetriMatrixType::INF_INPUT ? "INF" : "NAN")
                  << " info=" << info[batch] << " saw_inf=" << sawInf << " saw_nan=" << sawNan << std::endl;
    }
}

void RunSpecialValueCase(const CgetriBatchedParam& param, aclblasHandle_t handle, aclrtStream stream)
{
    const int n = param.n;
    const int lda = param.lda;
    const int ldc = param.ldc;
    const int batchSize = param.batchSize;
    bool usePivot = param.pivotMode == CgetriPivotMode::PIVOT;
    float specialValue = param.matrixType == CgetriMatrixType::INF_INPUT ? std::numeric_limits<float>::infinity() :
                                                                           std::numeric_limits<float>::quiet_NaN();

    std::vector<std::vector<aclblasComplex>> lu(static_cast<size_t>(batchSize));
    std::vector<const aclblasComplex*> luPointers(static_cast<size_t>(batchSize));
    std::vector<std::vector<aclblasComplex>> output(static_cast<size_t>(batchSize));
    std::vector<aclblasComplex*> outputPointers(static_cast<size_t>(batchSize));
    std::vector<int> pivots(usePivot ? static_cast<size_t>(n) * batchSize : 0);
    std::vector<int> info(static_cast<size_t>(batchSize), -1);
    for (int batch = 0; batch < batchSize; batch++) {
        lu[batch].assign(static_cast<size_t>(lda) * n, {0.0f, 0.0f});
        output[batch].assign(static_cast<size_t>(ldc) * n, {0.0f, 0.0f});
        for (int index = 0; index < n; index++) {
            lu[batch][index + index * lda] = {1.0f, 0.0f};
            if (usePivot)
                pivots[static_cast<size_t>(batch) * n + index] = index + 1;
        }
        lu[batch][static_cast<size_t>(lda)] = {specialValue, 0.0f};
        luPointers[batch] = lu[batch].data();
        outputPointers[batch] = output[batch].data();
    }

    CgetriBatchedNpuWrapper buffers;
    ASSERT_EQ(
        buffers.Prepare(
            luPointers.data(), outputPointers.data(), usePivot ? pivots.data() : nullptr, n, lda, ldc, batchSize),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(buffers.Run(handle), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(buffers.Synchronize(stream), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(buffers.CopyResults(outputPointers.data(), info.data()), ACLBLAS_STATUS_SUCCESS);

    VerifySpecialOutput(param, output, info);
}

double PerformanceThresholdUs(const CgetriBatchedParam& param)
{
    const std::string source = __FILE__;
    static const auto baseline =
        GetCasesFromCsv<csv_map>(source.substr(0, source.find_last_of("/\\") + 1) + "gpu_baseline.csv");
    int index = std::stoi(param.caseName.substr(std::string("TC_PF_").size())) - 1001;
    if (baseline.size() != 200 || index < 0 || static_cast<size_t>(index) >= baseline.size()) {
        throw std::runtime_error("Missing performance baseline for " + param.caseName);
    }
    const auto& row = baseline[static_cast<size_t>(index)];
    double gpuMs = parseDouble(ReadMap(row, "gpu_ms"));
    if (parseInt(ReadMap(row, "n")) != param.n || parseInt(ReadMap(row, "batch_size")) != param.batchSize ||
        !std::isfinite(gpuMs) || gpuMs <= 0.0) {
        throw std::runtime_error("Invalid performance baseline for " + param.caseName);
    }
    return gpuMs * 1000.0 / 0.4;
}

void MeasureCgetriPerformance(
    const CgetriBatchedParam& param, CgetriBatchedNpuWrapper& buffers, aclblasHandle_t handle, aclrtStream stream,
    double threshold)
{
    const int warmup = 5;
    const int samples = 51;
    const int n = param.n;
    const int batchSize = param.batchSize;
    for (int iteration = 0; iteration < warmup; iteration++) {
        ASSERT_EQ(buffers.Run(handle), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(buffers.Synchronize(stream), ACLBLAS_STATUS_SUCCESS);

    auto begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < samples; iteration++) {
        ASSERT_EQ(buffers.Run(handle), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(buffers.Synchronize(stream), ACLBLAS_STATUS_SUCCESS);
    auto end = std::chrono::steady_clock::now();

    double totalUs = std::chrono::duration<double, std::micro>(end - begin).count();
    double averageUs = totalUs / samples;
    std::cout << std::setprecision(10) << "PERF_RESULT case=" << param.caseName << " n=" << n << " batch=" << batchSize
              << " warmup=" << warmup << " samples=" << samples << " avg_us=" << averageUs
              << " formal=1 threshold_us=" << threshold << " ratio=" << threshold * 0.4 / averageUs;
    std::cout << std::endl;
    EXPECT_LE(averageUs, threshold) << param.caseName;
}

void VerifyPerformanceOutput(
    const CgetriBatchedParam& param, const std::vector<std::vector<aclblasComplex>>& output,
    const std::vector<int>& info, const std::vector<aclblasComplex>& golden)
{
    const int n = param.n;
    const int ldc = param.ldc;
    const int batchSize = param.batchSize;
    for (int batch = 0; batch < batchSize; batch++) {
        EXPECT_EQ(info[batch], 0);
        ComponentStats realStats;
        ComponentStats imagStats;
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < n; row++) {
                size_t index = static_cast<size_t>(col) * ldc + row;
                UpdateComponentStats(output[batch][index].real, golden[index].real, realStats);
                UpdateComponentStats(output[batch][index].imag, golden[index].imag, imagStats);
            }
        }
        VerifyComponentStats(realStats, "real", param.caseName, batch);
        VerifyComponentStats(imagStats, "imag", param.caseName, batch);
    }
}

void RunPerformanceCase(
    const CgetriBatchedParam& param, aclblasHandle_t handle, aclrtStream stream, bool measurePerformance)
{
    double threshold = PerformanceThresholdUs(param);

    const int n = param.n;
    const int lda = param.lda;
    const int ldc = param.ldc;
    const int batchSize = param.batchSize;
    bool usePivot = param.pivotMode == CgetriPivotMode::PIVOT;
    // All matrices still execute on the NPU.  Factor one reproducible matrix
    // of the CSV-specified type and repeat its LU across the performance batch
    // to keep CPU reference preparation outside the measured interval.
    auto original = MakeCgetriMatrix(param, n, lda, 0);
    auto firstLu = original;
    std::vector<int> firstPivot(static_cast<size_t>(n));
    for (int index = 0; index < n; index++)
        firstPivot[index] = index + 1;
    ASSERT_EQ(CgetriFactorSingle(firstLu, n, lda, firstPivot.data(), usePivot), 0);

    std::vector<std::vector<aclblasComplex>> lu(static_cast<size_t>(batchSize));
    std::vector<const aclblasComplex*> luPointers(static_cast<size_t>(batchSize));
    std::vector<std::vector<aclblasComplex>> output(static_cast<size_t>(batchSize));
    std::vector<aclblasComplex*> outputPointers(static_cast<size_t>(batchSize));
    std::vector<int> pivots(static_cast<size_t>(n) * batchSize);
    std::vector<int> info(static_cast<size_t>(batchSize), -1);
    for (int batch = 0; batch < batchSize; batch++) {
        lu[batch] = firstLu;
        output[batch].assign(static_cast<size_t>(ldc) * n, {0.0f, 0.0f});
        std::copy(firstPivot.begin(), firstPivot.end(), pivots.begin() + static_cast<size_t>(batch) * n);
        luPointers[batch] = lu[batch].data();
        outputPointers[batch] = output[batch].data();
    }

    CgetriBatchedNpuWrapper buffers;
    ASSERT_EQ(
        buffers.Prepare(
            luPointers.data(), outputPointers.data(), usePivot ? pivots.data() : nullptr, n, lda, ldc, batchSize),
        ACLBLAS_STATUS_SUCCESS);
    if (measurePerformance) {
        ASSERT_NO_FATAL_FAILURE(MeasureCgetriPerformance(param, buffers, handle, stream, threshold));
    } else {
        ASSERT_EQ(buffers.Run(handle), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(buffers.Synchronize(stream), ACLBLAS_STATUS_SUCCESS);
        std::cout << "FUNCTIONAL_RESULT case=" << param.caseName << " n=" << n << " batch=" << batchSize
                  << " performance_measured=0" << std::endl;
    }

    ASSERT_EQ(buffers.CopyResults(outputPointers.data(), info.data()), ACLBLAS_STATUS_SUCCESS);
    std::vector<aclblasComplex> golden(static_cast<size_t>(ldc) * n);
    ASSERT_EQ(
        CgetriInvertSingle(firstLu.data(), n, lda, usePivot ? firstPivot.data() : nullptr, golden.data(), ldc), 0);
    VerifyConditionNumber(original, lda, golden, ldc, n, param.caseName, 0);
    VerifyPerformanceOutput(param, output, info, golden);
}

} // namespace

INSTANTIATE_TEST_SUITE_P(
    CgetriBatched, CgetriBatchedTest,
    ::testing::ValuesIn(GetCasesFromCsv<CgetriBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgetriBatchedParam>);

TEST_P(CgetriBatchedTest, CsvDriven)
{
    const CgetriBatchedParam& param = GetParam();
    if (param.IsPerformanceCase()) {
        const char* functional = std::getenv("CGETRI_TEST_FUNCTIONAL_ONLY");
        bool measurePerformance = functional == nullptr || std::string(functional) != "1";
        RecordProperty("performance_measured", measurePerformance ? "1" : "0");
        RunPerformanceCase(param, CgetriBatchedTest::handle_, CgetriBatchedTest::stream_, measurePerformance);
    } else if (param.expectResult != ACLBLAS_STATUS_SUCCESS) {
        RunErrorCase(param, CgetriBatchedTest::handle_);
    } else if (param.n == 0 || param.batchSize == 0) {
        RunNoOpCase(param, CgetriBatchedTest::handle_);
    } else if (IsSpecialValueCase(param.matrixType)) {
        RunSpecialValueCase(param, CgetriBatchedTest::handle_, CgetriBatchedTest::stream_);
    } else {
        RunAccuracyCase(param, CgetriBatchedTest::handle_, CgetriBatchedTest::stream_);
    }
}
