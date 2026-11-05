/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "sgetrs_batched_param.h"
#include "sgetrs_batched_golden.h"
#include "sgetrs_batched_npu_wrapper.h"

// ---------------------------------------------------------------------------
// Matrix generation helpers
// ---------------------------------------------------------------------------

static std::vector<float> GenerateBMatrix(int n, int nrhs, int ldb, uint32_t seed, int batchIdx = 0)
{
    const size_t matSize = static_cast<size_t>(ldb) * std::max(1, nrhs);
    std::vector<float> mat(matSize, 0.0f);
    if (n <= 0 || nrhs <= 0)
        return mat;

    std::mt19937 rng(seed + batchIdx * 1000 + 500);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            mat[j * ldb + i] = dist(rng);
        }
    }
    return mat;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class SgetrsBatchedArch35Test : public BlasTest<SgetrsBatchedParam> {};

// Null handle test (TEST_F, not in CSV)
TEST_F(SgetrsBatchedArch35Test, NullHandle)
{
    aclblasStatus_t ret = aclblasSgetrsBatched_npu(
        nullptr, ACLBLAS_OP_N, 32, 4, nullptr, 32, nullptr, nullptr, 32, nullptr, 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    SgetrsBatched, SgetrsBatchedArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SgetrsBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SgetrsBatchedParam>);

// ---------------------------------------------------------------------------
// Error path test
// ---------------------------------------------------------------------------

// Compute expected info value for error-path cases, matching host-side validation order.
// Returns 0 if no info is expected to be set (e.g. handle==nullptr).
static int GetExpectedErrorInfoValue(const SgetrsBatchedParam& p)
{
    if (p.matrixType == GetrsMatrixType::NULLPTR_HANDLE)
        return 0;
    if (p.trans != ACLBLAS_OP_N && p.trans != ACLBLAS_OP_T && p.trans != ACLBLAS_OP_C)
        return -2;
    if (p.n < 0)
        return -3;
    if (p.nrhs < 0 || p.nrhs > 256)
        return -4;
    if (p.n > 0 && p.nrhs > 0 && p.batchCount > 0 && p.matrixType == GetrsMatrixType::NULLPTR_AARRAY)
        return -5;
    if (p.lda < std::max(1, p.n))
        return -6;
    if (p.n > 0 && p.nrhs > 0 && p.batchCount > 0 && p.matrixType == GetrsMatrixType::NULLPTR_BARRAY)
        return -8;
    if (p.ldb < std::max(1, p.n))
        return -9;
    if (p.batchCount < 0)
        return -11;
    return 0;
}

static void TestErrorPath(const SgetrsBatchedParam& p, aclblasHandle_t handle)
{
    bool nullAarray = (p.matrixType == GetrsMatrixType::NULLPTR_AARRAY);
    bool nullBarray = (p.matrixType == GetrsMatrixType::NULLPTR_BARRAY);
    bool nullHandle = (p.matrixType == GetrsMatrixType::NULLPTR_HANDLE);

    int safeN = std::max(1, p.n);
    int safeNrhs = std::max(1, p.nrhs);
    int safeBatch = std::max(1, p.batchCount);
    int safeLda = std::max(p.lda, safeN);
    int safeLdb = std::max(p.ldb, safeN);

    // Prepare dummy A matrices
    auto [errAMatrices, errAPtrs] = MakeDummyConstMatrices(safeBatch, safeLda, safeN, 1.0f, nullAarray);

    // Prepare dummy B matrices
    std::vector<std::vector<float>> errBMatrices;
    std::vector<float*> errBPtrs;
    if (!nullBarray) {
        errBMatrices.resize(safeBatch);
        errBPtrs.resize(safeBatch);
        for (int b = 0; b < safeBatch; b++) {
            errBMatrices[b].resize(static_cast<size_t>(safeLdb) * safeNrhs, 0.0f);
            errBPtrs[b] = errBMatrices[b].data();
        }
    }

    std::vector<int> errIpiv(
        (p.pivotMode == GetrsPivotMode::PIVOT) ? static_cast<size_t>(safeN) * safeBatch : 0, 0);
    int errInfo = 0;

    aclblasHandle_t h = nullHandle ? nullptr : handle;
    const float** aPtr = nullAarray ? nullptr : errAPtrs.data();
    float** bPtr = nullBarray ? nullptr : errBPtrs.data();
    int* ipivPtr = (p.pivotMode == GetrsPivotMode::PIVOT) ? errIpiv.data() : nullptr;
    int* infoPtr = (p.infoMode == GetrsInfoMode::NULLPTR) ? nullptr : &errInfo;

    aclblasStatus_t ret = aclblasSgetrsBatched_npu(
        h, p.trans, p.n, p.nrhs, aPtr, p.lda, ipivPtr, bPtr, p.ldb, infoPtr, p.batchCount);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));

    if (infoPtr != nullptr) {
        int expectedInfo = GetExpectedErrorInfoValue(p);
        EXPECT_EQ(errInfo, expectedInfo)
            << "[" << p.caseName << "] info mismatch: expected " << expectedInfo << ", got " << errInfo;
    }
}

// ---------------------------------------------------------------------------
// No-op path test (n=0 or nrhs=0 or batchCount=0)
// ---------------------------------------------------------------------------
static void TestNoOpPath(const SgetrsBatchedParam& p, aclblasHandle_t handle)
{
    int safeBatch = std::max(0, p.batchCount);
    std::vector<const float*> dummyA(safeBatch, nullptr);
    std::vector<float*> dummyB(safeBatch, nullptr);
    int info = -1;

    const float** aPtr = dummyA.empty() ? nullptr : dummyA.data();
    float** bPtr = dummyB.empty() ? nullptr : dummyB.data();
    int* infoPtr = (p.infoMode == GetrsInfoMode::NULLPTR) ? nullptr : &info;

    aclblasStatus_t ret = aclblasSgetrsBatched_npu(
        handle, p.trans, p.n, p.nrhs, aPtr, p.lda, nullptr, bPtr, p.ldb, infoPtr, p.batchCount);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

// ---------------------------------------------------------------------------
// Test data structure
// ---------------------------------------------------------------------------
struct GetrsTestData {
    // Original matrices (before LU)
    std::vector<std::vector<float>> originalAMatrices;
    // LU-factored matrices (after getrf, used as input to getrs)
    std::vector<std::vector<float>> luMatrices;
    // Host pointer arrays for A (const)
    std::vector<const float*> luPtrs;
    // Original B matrices (right-hand side)
    std::vector<std::vector<float>> originalBMatrices;
    // NPU output B matrices (solution, overwritten by NPU)
    std::vector<std::vector<float>> npuBMatrices;
    std::vector<float*> npuBPtrs;
    // Pivot and info arrays
    std::vector<int> pivotArray;
    std::vector<int> rfInfoArray; // info from getrf
};

static void PrepareGetrsTestData(
    GetrsTestData& data, const SgetrsBatchedParam& p, int n, int nrhs, int lda, int ldb, int batchCount,
    bool usePivot)
{
    // Generate original A matrices and B matrices
    data.originalAMatrices.resize(batchCount);
    data.luMatrices.resize(batchCount);
    data.luPtrs.resize(batchCount);
    data.originalBMatrices.resize(batchCount);
    data.npuBMatrices.resize(batchCount);
    data.npuBPtrs.resize(batchCount);

    for (int b = 0; b < batchCount; b++) {
        data.originalAMatrices[b] = makeBlasLapackMatrix(n, lda, toBlasLapackMatrixType(p.matrixType), p.randomSeed, b);
        data.luMatrices[b] = data.originalAMatrices[b]; // copy for LU factorization
        data.luPtrs[b] = data.luMatrices[b].data();
        data.originalBMatrices[b] = GenerateBMatrix(n, nrhs, ldb, p.randomSeed, b);
        data.npuBMatrices[b] = data.originalBMatrices[b]; // copy for NPU input
        data.npuBPtrs[b] = data.npuBMatrices[b].data();
    }

    // Allocate pivot array
    data.pivotArray.resize(usePivot ? static_cast<size_t>(n) * batchCount : 0, 0);

    // Allocate info arrays
    data.rfInfoArray.resize(batchCount, -1);
}

/**
 * Perform CPU LU factorization on the batch of matrices.
 * Modifies data.luMatrices in-place and populates pivotArray and rfInfoArray.
 */
static void RunCpuGetrf(GetrsTestData& data, int n, int lda, int batchCount, bool usePivot)
{
    std::vector<float*> luPtrsNonConst(batchCount);
    for (int b = 0; b < batchCount; b++) {
        luPtrsNonConst[b] = data.luMatrices[b].data();
    }

    int* pivPtr = usePivot ? data.pivotArray.data() : nullptr;
    aclblasSgetrfBatched_cpu_for_getrs(
        n, luPtrsNonConst.data(), lda, pivPtr, data.rfInfoArray.data(), batchCount, usePivot);

    // Update const pointer array after LU
    for (int b = 0; b < batchCount; b++) {
        data.luPtrs[b] = data.luMatrices[b].data();
    }
}

/**
 * Run CPU golden getrs on the LU-factored matrices.
 * Returns golden solution matrices and golden info.
 */
static void RunCpuGetrs(
    const GetrsTestData& data, int n, int nrhs, int lda, int ldb, int batchCount, bool usePivot,
    aclblasOperation_t trans,
    std::vector<std::vector<float>>& goldenSolutions, std::vector<int>& goldenInfo, aclblasHandle_t handle)
{
    goldenSolutions.resize(batchCount);
    std::vector<const float*> luConstPtrs(batchCount);
    std::vector<float*> goldenPtrs(batchCount);

    for (int b = 0; b < batchCount; b++) {
        // Copy original B for golden computation (sgetrs overwrites B with solution)
        goldenSolutions[b] = data.originalBMatrices[b];
        luConstPtrs[b] = data.luMatrices[b].data();
        goldenPtrs[b] = goldenSolutions[b].data();
    }

    goldenInfo.resize(batchCount, -1);
    const int* pivPtr = usePivot ? data.pivotArray.data() : nullptr;

    // Build devIpiv for golden: if NO_PIVOT, pass nullptr
    aclblasSgetrsBatched_cpu(
        handle, trans, n, nrhs, luConstPtrs.data(), lda, pivPtr,
        goldenPtrs.data(), ldb, goldenInfo.data(), batchCount);
}

// ---------------------------------------------------------------------------
// Verification helpers
// ---------------------------------------------------------------------------
static void VerifySolutions(
    const std::vector<std::vector<float>>& npuSolutions,
    const std::vector<std::vector<float>>& goldenSolutions,
    const std::vector<int>& goldenInfo, int n, int nrhs, int ldb, int batchCount,
    const std::string& caseName)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, static_cast<const float*>(nullptr), static_cast<size_t>(0));

    for (int b = 0; b < batchCount; b++) {
        // Skip singular matrices (info > 0 means singular)
        if (goldenInfo[b] > 0) {
            continue;
        }

        // Extract n×nrhs block from ldb-strided storage
        std::vector<float> npuBlock(static_cast<size_t>(n) * nrhs);
        std::vector<float> goldenBlock(static_cast<size_t>(n) * nrhs);

        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < n; i++) {
                npuBlock[j * n + i] = npuSolutions[b][j * ldb + i];
                goldenBlock[j * n + i] = goldenSolutions[b][j * ldb + i];
            }
        }

        bool pass = Verifier::verifyVector(
            npuBlock.data(), goldenBlock.data(), static_cast<size_t>(n) * nrhs, 1, cfg,
            caseName + "_batch" + std::to_string(b));
        EXPECT_TRUE(pass) << "[" << caseName << "] batch=" << b << " solution matrix mismatch";
    }
}

// ---------------------------------------------------------------------------
// Normal path test helper
// ---------------------------------------------------------------------------
static void TestNormalPath(
    const SgetrsBatchedParam& p, int n, int nrhs, int lda, int ldb, int batchCount,
    bool usePivot, aclblasHandle_t handle)
{
    GetrsTestData data;
    PrepareGetrsTestData(data, p, n, nrhs, lda, ldb, batchCount, usePivot);
    RunCpuGetrf(data, n, lda, batchCount, usePivot);

    // Run golden getrs
    std::vector<std::vector<float>> goldenSolutions;
    std::vector<int> goldenInfo;
    RunCpuGetrs(data, n, nrhs, lda, ldb, batchCount, usePivot, p.trans,
                goldenSolutions, goldenInfo, handle);

    // Run NPU getrs
    int info = -1;
    int* infoPtr = (p.infoMode == GetrsInfoMode::NULLPTR) ? nullptr : &info;
    const int* ipivPtr = usePivot ? data.pivotArray.data() : nullptr;

    aclblasStatus_t ret = aclblasSgetrsBatched_npu(
        handle, p.trans, n, nrhs, data.luPtrs.data(), lda, ipivPtr,
        data.npuBPtrs.data(), ldb, infoPtr, batchCount);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return;

    if (p.infoMode == GetrsInfoMode::NORMAL && infoPtr != nullptr) {
        EXPECT_EQ(info, 0) << "[" << p.caseName << "] expected info==0, got " << info;
    }

    VerifySolutions(
        data.npuBMatrices, goldenSolutions, goldenInfo, n, nrhs, ldb, batchCount,
        p.caseName);
}

// ---------------------------------------------------------------------------
// Main parameterized test
// ---------------------------------------------------------------------------
TEST_P(SgetrsBatchedArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const int n = p.n;
    const int nrhs = p.nrhs;
    const int batchCount = p.batchCount;
    const bool usePivot = (p.pivotMode == GetrsPivotMode::PIVOT);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        TestErrorPath(p, SgetrsBatchedArch35Test::handle_);
    } else if (n <= 0 || nrhs <= 0 || batchCount <= 0) {
        TestNoOpPath(p, SgetrsBatchedArch35Test::handle_);
    } else {
        TestNormalPath(p, n, nrhs, p.lda, p.ldb, batchCount, usePivot, SgetrsBatchedArch35Test::handle_);
    }
}
