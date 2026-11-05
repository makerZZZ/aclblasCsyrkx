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
#include "fill.h"
#include "sgetri_batched_param.h"
#include "sgetri_batched_golden.h"
#include "sgetri_batched_npu_wrapper.h"

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class SgetriBatchedArch35Test : public BlasTest<SgetriBatchedParam> {};

// Null handle test (TEST_F, not in CSV)
TEST_F(SgetriBatchedArch35Test, NullHandle)
{
    aclblasStatus_t ret = aclblasSgetriBatched_npu(nullptr, 32, nullptr, 32, nullptr, nullptr, 32, nullptr, 4, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    SgetriBatched, SgetriBatchedArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SgetriBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SgetriBatchedParam>);

// ---------------------------------------------------------------------------
// Error path test
// ---------------------------------------------------------------------------
static void TestErrorPath(const SgetriBatchedParam& p, aclblasHandle_t handle, aclrtStream stream)
{
    bool nullAarray = (p.matrixType == GetriMatrixType::NULLPTR_AARRAY);
    bool nullCarray = (p.matrixType == GetriMatrixType::NULLPTR_CARRAY);
    bool nullInfo = (p.matrixType == GetriMatrixType::NULLPTR_INFOARRAY);
    bool nullHandle = (p.matrixType == GetriMatrixType::NULLPTR_HANDLE);
    bool usePivot = (p.pivotMode == GetriPivotMode::PIVOT);

    int safeN = std::max(1, p.n);
    int safeBatch = std::max(1, p.batchSize);
    int safeLda = std::max(p.lda, safeN);
    int safeLdc = std::max(p.ldc, safeN);

    // Prepare dummy matrices
    auto [errAMatrices, errAPtrs] = MakeDummyConstMatrices(safeBatch, safeLda, safeN, 1.0f, nullAarray);

    std::vector<std::vector<float>> errCMatrices;
    std::vector<float*> errCPtrs;
    if (!nullCarray) {
        errCMatrices.resize(safeBatch);
        errCPtrs.resize(safeBatch);
        for (int b = 0; b < safeBatch; b++) {
            errCMatrices[b].resize(static_cast<size_t>(safeLdc) * safeN, 0.0f);
            errCPtrs[b] = errCMatrices[b].data();
        }
    }

    std::vector<int> errPivot(usePivot ? static_cast<size_t>(safeN) * safeBatch : 0, 0);
    std::vector<int> errInfo(safeBatch, 0);

    aclblasHandle_t h = nullHandle ? nullptr : handle;
    const float** aPtr = nullAarray ? nullptr : errAPtrs.data();
    float** cPtr = nullCarray ? nullptr : errCPtrs.data();
    int* pivPtr = usePivot ? errPivot.data() : nullptr;
    int* infoPtr = nullInfo ? nullptr : errInfo.data();

    aclblasStatus_t ret = aclblasSgetriBatched_npu(h, p.n, aPtr, p.lda, pivPtr, cPtr, p.ldc, infoPtr, p.batchSize, stream);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

// ---------------------------------------------------------------------------
// No-op path test (n=0 or batchSize=0)
// ---------------------------------------------------------------------------
static void TestNoOpPath(const SgetriBatchedParam& p, aclblasHandle_t handle, aclrtStream stream)
{
    int safeBatch = std::max(0, p.batchSize);
    std::vector<const float*> dummyA(safeBatch, nullptr);
    std::vector<float*> dummyC(safeBatch, nullptr);
    std::vector<int> dummyInfo(safeBatch, -1);

    const float** aPtr = dummyA.empty() ? nullptr : dummyA.data();
    float** cPtr = dummyC.empty() ? nullptr : dummyC.data();
    int* infoPtr = dummyInfo.empty() ? nullptr : dummyInfo.data();

    aclblasStatus_t ret =
        aclblasSgetriBatched_npu(handle, p.n, aPtr, p.lda, nullptr, cPtr, p.ldc, infoPtr, p.batchSize, stream);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

// ---------------------------------------------------------------------------
// Verification helpers
// ---------------------------------------------------------------------------
static void VerifyInfoArray(
    const std::vector<int>& hostInfo, const std::vector<int>& goldenInfo, int batchSize, const std::string& caseName)
{
    for (int b = 0; b < batchSize; b++) {
        EXPECT_EQ(hostInfo[b], goldenInfo[b]) << "[" << caseName << "] batch=" << b
                                              << " info mismatch: NPU=" << hostInfo[b] << " golden=" << goldenInfo[b];
    }
}

static void VerifyInverseMatrices(
    const std::vector<std::vector<float>>& npuInverses, const std::vector<std::vector<float>>& goldenInverses,
    const std::vector<int>& goldenInfo, int n, int ldc, int batchSize, const std::string& caseName)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, static_cast<const float*>(nullptr), static_cast<size_t>(0));

    for (int b = 0; b < batchSize; b++) {
        // Skip singular matrices (info > 0 means singular)
        if (goldenInfo[b] > 0) {
            continue;
        }

        // Extract n×n block from ldc-strided storage
        std::vector<float> npuBlock(static_cast<size_t>(n) * n);
        std::vector<float> goldenBlock(static_cast<size_t>(n) * n);

        // Threshold near-zero elements to eliminate CPU/NPU floating-point discrepancies
        constexpr float kNearZeroThreshold = 1e-6f;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                float npuVal = npuInverses[b][j * ldc + i];
                float goldVal = goldenInverses[b][j * ldc + i];
                npuBlock[j * n + i] = (std::abs(npuVal) < kNearZeroThreshold) ? 0.0f : npuVal;
                goldenBlock[j * n + i] = (std::abs(goldVal) < kNearZeroThreshold) ? 0.0f : goldVal;
            }
        }

        bool pass = Verifier::verifyVector(
            npuBlock.data(), goldenBlock.data(), static_cast<size_t>(n) * n, 1, cfg,
            caseName + "_batch" + std::to_string(b));
        EXPECT_TRUE(pass) << "[" << caseName << "] batch=" << b << " inverse matrix mismatch";
    }
}

// ---------------------------------------------------------------------------
// Test data structure
// ---------------------------------------------------------------------------
struct GetriTestData {
    // Original matrices (before LU)
    std::vector<std::vector<float>> originalMatrices;
    // LU-factored matrices (after getrf, used as input to getri)
    std::vector<std::vector<float>> luMatrices;
    // Host pointer arrays
    std::vector<const float*> luPtrs;
    // Output inverse matrices
    std::vector<std::vector<float>> npuInverses;
    std::vector<float*> npuInversePtrs;
    // Pivot and info arrays
    std::vector<int> pivotArray;
    std::vector<int> rfInfoArray; // info from getrf
    std::vector<int> riInfoArray; // info from getri (NPU)
};

static void PrepareGetriTestData(
    GetriTestData& data, const SgetriBatchedParam& p, int n, int lda, int ldc, int batchSize, bool usePivot)
{
    // Generate original matrices
    data.originalMatrices.resize(batchSize);
    data.luMatrices.resize(batchSize);
    data.luPtrs.resize(batchSize);

    for (int b = 0; b < batchSize; b++) {
        data.originalMatrices[b] = makeBlasLapackMatrix(n, lda, toBlasLapackMatrixType(p.matrixType), p.randomSeed, b);
        data.luMatrices[b] = data.originalMatrices[b]; // copy for LU factorization
        data.luPtrs[b] = data.luMatrices[b].data();
    }

    // Allocate pivot array
    data.pivotArray.resize(usePivot ? static_cast<size_t>(n) * batchSize : 0, 0);

    // Allocate info arrays
    data.rfInfoArray.resize(batchSize, -1);
    data.riInfoArray.resize(batchSize, -1);

    // Allocate output inverse matrices
    data.npuInverses.resize(batchSize);
    data.npuInversePtrs.resize(batchSize);
    for (int b = 0; b < batchSize; b++) {
        data.npuInverses[b].resize(static_cast<size_t>(ldc) * n, 0.0f);
        data.npuInversePtrs[b] = data.npuInverses[b].data();
    }
}

/**
 * Perform CPU LU factorization on the batch of matrices.
 * Modifies data.luMatrices in-place and populates pivotArray and rfInfoArray.
 */
static void RunCpuGetrf(GetriTestData& data, int n, int lda, int batchSize, bool usePivot)
{
    // Create non-const pointer array for CPU getrf
    std::vector<float*> luPtrsNonConst(batchSize);
    for (int b = 0; b < batchSize; b++) {
        luPtrsNonConst[b] = data.luMatrices[b].data();
    }

    int* pivPtr = usePivot ? data.pivotArray.data() : nullptr;
    aclblasSgetrfBatched_cpu_for_getri(
        n, luPtrsNonConst.data(), lda, pivPtr, data.rfInfoArray.data(), batchSize, usePivot);

    // Update const pointer array after LU
    for (int b = 0; b < batchSize; b++) {
        data.luPtrs[b] = data.luMatrices[b].data();
    }
}

/**
 * Run CPU golden getri on the LU-factored matrices.
 * Returns golden inverse matrices and golden info.
 */
static void RunCpuGetri(
    const GetriTestData& data, int n, int lda, int ldc, int batchSize, bool usePivot,
    std::vector<std::vector<float>>& goldenInverses, std::vector<int>& goldenInfo, aclblasHandle_t handle)
{
    goldenInverses.resize(batchSize);
    std::vector<const float*> luConstPtrs(batchSize);
    std::vector<float*> goldenPtrs(batchSize);

    for (int b = 0; b < batchSize; b++) {
        goldenInverses[b].resize(static_cast<size_t>(ldc) * n, 0.0f);
        luConstPtrs[b] = data.luMatrices[b].data();
        goldenPtrs[b] = goldenInverses[b].data();
    }

    goldenInfo.resize(batchSize, -1);
    const int* pivPtr = usePivot ? data.pivotArray.data() : nullptr;

    aclblasSgetriBatched_cpu(
        handle, n, luConstPtrs.data(), lda, pivPtr, goldenPtrs.data(), ldc, goldenInfo.data(), batchSize);
}

// ---------------------------------------------------------------------------
// Normal path test helper
// ---------------------------------------------------------------------------
static void TestNormalPath(
    const SgetriBatchedParam& p, int n, int lda, int ldc, int batchSize, bool usePivot, aclblasHandle_t handle, aclrtStream stream)
{
    GetriTestData data;
    PrepareGetriTestData(data, p, n, lda, ldc, batchSize, usePivot);
    RunCpuGetrf(data, n, lda, batchSize, usePivot);

    std::vector<std::vector<float>> goldenInverses;
    std::vector<int> goldenInfo;
    RunCpuGetri(data, n, lda, ldc, batchSize, usePivot, goldenInverses, goldenInfo, handle);

    aclblasStatus_t ret = aclblasSgetriBatched_npu(
        handle, n, data.luPtrs.data(), lda, usePivot ? data.pivotArray.data() : nullptr, data.npuInversePtrs.data(),
        ldc, data.riInfoArray.data(), batchSize, stream);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return;

    VerifyInfoArray(data.riInfoArray, goldenInfo, batchSize, p.caseName);

    VerifyInverseMatrices(
        data.npuInverses, goldenInverses, goldenInfo, n, ldc, batchSize, p.caseName);
}

// ---------------------------------------------------------------------------
// Main parameterized test
// ---------------------------------------------------------------------------
TEST_P(SgetriBatchedArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const int n = p.n;
    const int batchSize = p.batchSize;
    const bool usePivot = (p.pivotMode == GetriPivotMode::PIVOT);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        TestErrorPath(p, SgetriBatchedArch35Test::handle_, SgetriBatchedArch35Test::stream_);
    } else if (n <= 0 || batchSize <= 0) {
        TestNoOpPath(p, SgetriBatchedArch35Test::handle_, SgetriBatchedArch35Test::stream_);
    } else {
        TestNormalPath(p, n, p.lda, p.ldc, batchSize, usePivot, SgetriBatchedArch35Test::handle_, SgetriBatchedArch35Test::stream_);
    }
}
