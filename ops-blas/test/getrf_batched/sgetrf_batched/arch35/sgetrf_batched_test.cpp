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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "sgetrf_batched_param.h"
#include "sgetrf_batched_golden.h"
#include "sgetrf_batched_npu_wrapper.h"

// ---------------------------------------------------------------------------
// Matrix generation helpers
// ---------------------------------------------------------------------------

static void GenerateRandomNonsingular(
    std::vector<float>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mat[j * lda + i] = dist(rng);
        }
        mat[j * lda + j] += static_cast<float>(n);
    }
}

static void GenerateSingularZeroCol(
    std::vector<float>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist)
{
    GenerateRandomNonsingular(mat, n, lda, rng, dist);
    for (int i = 0; i < n; i++) {
        mat[0 * lda + i] = 0.0f;
    }
}

static void GenerateIdentity(std::vector<float>& mat, int n, int lda)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mat[j * lda + i] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

static void GenerateDiagonallyDominant(
    std::vector<float>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mat[j * lda + i] = dist(rng);
        }
        mat[j * lda + j] = static_cast<float>(n);
    }
}

static void GenerateIllConditioned(std::vector<float>& mat, int n, int lda)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mat[j * lda + i] = 1.0f / static_cast<float>(i + j + 1);
        }
    }
}

static void GenerateSingularDependentRow(
    std::vector<float>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist)
{
    GenerateRandomNonsingular(mat, n, lda, rng, dist);
    int depRow = std::min(n - 1, 2);
    if (depRow > 0) {
        for (int j = 0; j < n; j++) {
            mat[j * lda + depRow] = 2.0f * mat[j * lda + 0];
        }
    }
}

static void GenerateMixed(
    std::vector<float>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist,
    int batchIdx)
{
    GenerateRandomNonsingular(mat, n, lda, rng, dist);
    if (batchIdx % 2 == 1) {
        for (int i = 0; i < n; i++) {
            mat[0 * lda + i] = 0.0f;
        }
    }
}

/**
 * Generate a single n×n matrix in column-major storage (lda stride).
 * Returns a vector of size lda * n.
 */
static std::vector<float> generateMatrix(
    int n, int lda, GetrfMatrixType type, uint32_t seed, int batchIdx = 0, int batchSize = 1)
{
    const size_t matSize = static_cast<size_t>(lda) * std::max(1, n);
    std::vector<float> mat(matSize, 0.0f);
    if (n <= 0)
        return mat;

    std::mt19937 rng(seed + batchIdx * 1000);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    switch (type) {
        case GetrfMatrixType::RANDOM_NONSINGULAR:
            GenerateRandomNonsingular(mat, n, lda, rng, dist);
            break;
        case GetrfMatrixType::SINGULAR_ZERO_COL:
            GenerateSingularZeroCol(mat, n, lda, rng, dist);
            break;
        case GetrfMatrixType::IDENTITY:
            GenerateIdentity(mat, n, lda);
            break;
        case GetrfMatrixType::DIAGONALLY_DOMINANT:
            GenerateDiagonallyDominant(mat, n, lda, rng, dist);
            break;
        case GetrfMatrixType::ILL_CONDITIONED:
            GenerateIllConditioned(mat, n, lda);
            break;
        case GetrfMatrixType::SINGULAR_DEPENDENT_ROW:
            GenerateSingularDependentRow(mat, n, lda, rng, dist);
            break;
        case GetrfMatrixType::MIXED:
            GenerateMixed(mat, n, lda, rng, dist, batchIdx);
            break;
    }
    return mat;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class SgetrfBatchedArch35Test : public BlasTest<SgetrfBatchedParam> {};

// Null handle test (TEST_F, not in CSV)
TEST_F(SgetrfBatchedArch35Test, NullHandle)
{
    aclblasStatus_t ret = aclblasSgetrfBatched_npu(nullptr, 32, nullptr, 32, nullptr, nullptr, 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    SgetrfBatched, SgetrfBatchedArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SgetrfBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SgetrfBatchedParam>);

static void TestErrorPath(const SgetrfBatchedParam& p, aclblasHandle_t handle)
{
    bool nullAarray = (p.matrixType == GetrfMatrixType::NULLPTR_AARRAY);
    bool nullInfo = (p.matrixType == GetrfMatrixType::NULLPTR_INFOARRAY);
    bool nullHandle = (p.matrixType == GetrfMatrixType::NULLPTR_HANDLE);
    bool usePivot = (p.pivotMode == GetrfPivotMode::PIVOT);

    int safeN = std::max(1, p.n);
    int safeBatch = std::max(1, p.batchSize);
    int safeLda = std::max(p.lda, safeN);

    std::vector<std::vector<float>> errMatrices;
    std::vector<float*> errPtrs;
    if (!nullAarray) {
        errMatrices.resize(safeBatch);
        errPtrs.resize(safeBatch);
        for (int b = 0; b < safeBatch; b++) {
            errMatrices[b].resize(static_cast<size_t>(safeLda) * safeN, 1.0f);
            errPtrs[b] = errMatrices[b].data();
        }
    }

    std::vector<int> errPivot(usePivot ? static_cast<size_t>(safeN) * safeBatch : 0, 0);
    std::vector<int> errInfo(safeBatch, 0);

    aclblasHandle_t h = nullHandle ? nullptr : handle;
    float** aPtr = nullAarray ? nullptr : errPtrs.data();
    int* pivPtr = usePivot ? errPivot.data() : nullptr;
    int* infoPtr = nullInfo ? nullptr : errInfo.data();

    aclblasStatus_t ret = aclblasSgetrfBatched_npu(h, p.n, aPtr, p.lda, pivPtr, infoPtr, p.batchSize);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

static void TestNoOpPath(const SgetrfBatchedParam& p, aclblasHandle_t handle)
{
    std::vector<float*> dummyA(std::max(0, p.batchSize), nullptr);
    std::vector<int> dummyInfo(std::max(0, p.batchSize), -1);
    float** aPtr = dummyA.empty() ? nullptr : dummyA.data();
    int* infoPtr = dummyInfo.empty() ? nullptr : dummyInfo.data();

    aclblasStatus_t ret = aclblasSgetrfBatched_npu(handle, p.n, aPtr, p.lda, nullptr, infoPtr, p.batchSize);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

static void VerifyInfoArray(
    const std::vector<int>& hostInfo, const std::vector<int>& goldenInfo, int batchSize, const std::string& caseName)
{
    for (int b = 0; b < batchSize; b++) {
        EXPECT_EQ(hostInfo[b], goldenInfo[b]) << "[" << caseName << "] batch=" << b
                                              << " info mismatch: NPU=" << hostInfo[b] << " golden=" << goldenInfo[b];
    }
}

static void VerifyPivotArray(
    const std::vector<int>& hostPivot, const std::vector<int>& goldenPivot, size_t pivotSize,
    const std::string& caseName)
{
    bool pivotMatch = true;
    for (size_t i = 0; i < pivotSize; i++) {
        if (hostPivot[i] != goldenPivot[i]) {
            pivotMatch = false;
            std::cout << "[" << caseName << "] pivot mismatch at index " << i << ": NPU=" << hostPivot[i]
                      << " golden=" << goldenPivot[i] << std::endl;
            break;
        }
    }
    EXPECT_TRUE(pivotMatch) << "[" << caseName << "] PivotArray mismatch";
}

static void VerifyLUMatrices(
    const std::vector<std::vector<float>>& hostMatrices, const std::vector<std::vector<float>>& goldenMatrices,
    const std::vector<int>& hostInfo, int n, int lda, int batchSize, const std::string& caseName)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, static_cast<const float*>(nullptr), static_cast<size_t>(0));

    for (int b = 0; b < batchSize; b++) {
        if (hostInfo[b] > 0) {
            continue;
        }

        std::vector<float> npuBlock(static_cast<size_t>(n) * n);
        std::vector<float> goldenBlock(static_cast<size_t>(n) * n);

        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                npuBlock[j * n + i] = hostMatrices[b][j * lda + i];
                goldenBlock[j * n + i] = goldenMatrices[b][j * lda + i];
            }
        }

        bool pass = Verifier::verifyVector(
            npuBlock.data(), goldenBlock.data(), static_cast<size_t>(n) * n, 1, cfg,
            caseName + "_batch" + std::to_string(b));
        EXPECT_TRUE(pass) << "[" << caseName << "] batch=" << b << " L/U matrix mismatch";
    }
}

struct GetrfTestData {
    std::vector<std::vector<float>> hostMatrices;
    std::vector<std::vector<float>> originalMatrices;
    std::vector<float*> hostPtrs;
    std::vector<int> hostPivot;
    std::vector<int> hostInfo;
    size_t pivotSize;
};

static void PrepareTestData(
    GetrfTestData& data, const SgetrfBatchedParam& p, int n, int lda, int batchSize, bool usePivot)
{
    data.hostMatrices.resize(batchSize);
    data.hostPtrs.resize(batchSize);
    data.originalMatrices.resize(batchSize);

    for (int b = 0; b < batchSize; b++) {
        data.hostMatrices[b] = generateMatrix(n, lda, p.matrixType, p.randomSeed, b, batchSize);
        data.hostPtrs[b] = data.hostMatrices[b].data();
        data.originalMatrices[b] = data.hostMatrices[b];
    }

    data.pivotSize = usePivot ? static_cast<size_t>(n) * batchSize : 0;
    data.hostPivot.resize(data.pivotSize, 0);
    data.hostInfo.resize(batchSize, -1);
}

static void RunNpuAndGetGolden(
    GetrfTestData& data, const SgetrfBatchedParam& p, int n, int lda, int batchSize, bool usePivot,
    std::vector<std::vector<float>>& goldenMatrices, std::vector<int>& goldenPivot, std::vector<int>& goldenInfo,
    aclblasHandle_t handle)
{
    int* pivotPtr = usePivot ? data.hostPivot.data() : nullptr;
    int* infoPtr = data.hostInfo.data();

    aclblasStatus_t ret = aclblasSgetrfBatched_npu(handle, n, data.hostPtrs.data(), lda, pivotPtr, infoPtr, batchSize);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));

    goldenMatrices.resize(batchSize);
    std::vector<float*> goldenPtrs(batchSize);
    for (int b = 0; b < batchSize; b++) {
        goldenMatrices[b] = data.originalMatrices[b];
        goldenPtrs[b] = goldenMatrices[b].data();
    }

    goldenPivot.resize(data.pivotSize, 0);
    goldenInfo.resize(batchSize, -1);
    int* goldenPivotPtr = usePivot ? goldenPivot.data() : nullptr;

    aclblasSgetrfBatched_cpu(handle, n, goldenPtrs.data(), lda, goldenPivotPtr, goldenInfo.data(), batchSize);
}

TEST_P(SgetrfBatchedArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const int n = p.n;
    const int lda = p.lda;
    const int batchSize = p.batchSize;
    const bool usePivot = (p.pivotMode == GetrfPivotMode::PIVOT);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        TestErrorPath(p, SgetrfBatchedArch35Test::handle_);
        return;
    }

    if (n <= 0 || batchSize <= 0) {
        TestNoOpPath(p, SgetrfBatchedArch35Test::handle_);
        return;
    }

    GetrfTestData data;
    PrepareTestData(data, p, n, lda, batchSize, usePivot);

    std::vector<std::vector<float>> goldenMatrices;
    std::vector<int> goldenPivot;
    std::vector<int> goldenInfo;
    RunNpuAndGetGolden(
        data, p, n, lda, batchSize, usePivot, goldenMatrices, goldenPivot, goldenInfo,
        SgetrfBatchedArch35Test::handle_);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    VerifyInfoArray(data.hostInfo, goldenInfo, batchSize, p.caseName);

    if (usePivot) {
        VerifyPivotArray(data.hostPivot, goldenPivot, data.pivotSize, p.caseName);
    }

    VerifyLUMatrices(data.hostMatrices, goldenMatrices, data.hostInfo, n, lda, batchSize, p.caseName);
}
