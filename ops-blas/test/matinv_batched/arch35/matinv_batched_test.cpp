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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "matinv_batched_param.h"
#include "matinv_batched_golden.h"
#include "matinv_batched_npu_wrapper.h"

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class MatinvBatchedArch35Test : public BlasTest<MatinvBatchedParam> {};

// Null handle test (TEST_F, not in CSV)
TEST_F(MatinvBatchedArch35Test, NullHandle) {
    aclblasStatus_t ret = aclblasSmatinvBatched_npu(nullptr, 16, nullptr, 16, nullptr, 16, nullptr, 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    MatinvBatched, MatinvBatchedArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<MatinvBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatinvBatchedParam>);

// ---------------------------------------------------------------------------
// Error path test
// ---------------------------------------------------------------------------
static void TestErrorPath(const MatinvBatchedParam& p, aclblasHandle_t handle)
{
    bool nullAll = (p.matrixType == MatinvMatrixType::NULLPTR_ALL);
    bool nullA = (p.matrixType == MatinvMatrixType::NULLPTR_A) || nullAll;
    bool nullAinv = (p.matrixType == MatinvMatrixType::NULLPTR_AINV) || nullAll;
    bool nullInfo = (p.matrixType == MatinvMatrixType::NULLPTR_INFO) || nullAll;

    int safeN = std::max(1, p.n);
    int safeBatch = std::max(1, p.batchSize);
    int safeLda = std::max(p.lda, safeN);
    int safeLdaInv = std::max(p.ldaInv, safeN);

    // Prepare dummy matrices (always need valid host data for the test)
    std::vector<std::vector<float>> inputMatrices;
    std::vector<const float*> inputPtrs;
    if (!nullA) {
        inputMatrices.resize(safeBatch);
        inputPtrs.resize(safeBatch);
        for (int b = 0; b < safeBatch; b++) {
            inputMatrices[b] = makeBlasLapackMatrix(
                safeN, safeLda, BlasLapackMatrixType::RANDOM_NONSINGULAR, p.randomSeed, b);
            inputPtrs[b] = inputMatrices[b].data();
        }
    }

    std::vector<std::vector<float>> outputMatrices;
    std::vector<float*> outputPtrs;
    if (!nullAinv) {
        outputMatrices.resize(safeBatch);
        outputPtrs.resize(safeBatch);
        for (int b = 0; b < safeBatch; b++) {
            outputMatrices[b].resize(static_cast<size_t>(safeLdaInv) * safeN, 0.0f);
            outputPtrs[b] = outputMatrices[b].data();
        }
    }

    std::vector<int> infoArray(safeBatch, 0);

    const float** aPtr = nullA ? nullptr : inputPtrs.data();
    float** ainvPtr = nullAinv ? nullptr : outputPtrs.data();
    int* infoPtr = nullInfo ? nullptr : infoArray.data();

    aclblasStatus_t ret = aclblasSmatinvBatched_npu(
        handle, p.n, aPtr, p.lda, ainvPtr, p.ldaInv, infoPtr, p.batchSize);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

// ---------------------------------------------------------------------------
// No-op path test (n=0 or batchSize=0)
// ---------------------------------------------------------------------------
static void TestNoOpPath(const MatinvBatchedParam& p, aclblasHandle_t handle)
{
    // For no-op cases, the wrapper should return SUCCESS without allocating device memory.
    // We pass nullptr arrays - the API short-circuits before accessing them.
    aclblasStatus_t ret = aclblasSmatinvBatched_npu(
        handle, p.n, nullptr, p.lda, nullptr, p.ldaInv, nullptr, p.batchSize);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

// ---------------------------------------------------------------------------
// Verification helpers
// ---------------------------------------------------------------------------
static void VerifyInfoArray(
    const std::vector<int>& hostInfo, const std::vector<int>& goldenInfo,
    int batchSize, const std::string& caseName)
{
    for (int b = 0; b < batchSize; b++) {
        EXPECT_EQ(hostInfo[b], goldenInfo[b])
            << "[" << caseName << "] batch=" << b
            << " info mismatch: NPU=" << hostInfo[b] << " golden=" << goldenInfo[b];
    }
}

static void VerifyInverseMatrices(
    const std::vector<std::vector<float>>& npuOutputs,
    const std::vector<std::vector<float>>& goldenOutputs,
    const std::vector<int>& goldenInfo,
    int n, int ldaInv, int batchSize,
    const std::string& caseName,
    double mereThreshold, double mareMultiplier)
{
    if (mereThreshold <= 0.0 && mareMultiplier <= 0.0)
        return;

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, static_cast<const float*>(nullptr), static_cast<size_t>(0));

    for (int b = 0; b < batchSize; b++) {
        // Skip singular matrices (info > 0 means singular, inverse is undefined)
        if (goldenInfo[b] > 0)
            continue;

        // Extract n×n block from lda_inv-strided storage
        std::vector<float> npuBlock(static_cast<size_t>(n) * n);
        std::vector<float> goldenBlock(static_cast<size_t>(n) * n);

        // Threshold near-zero elements to eliminate CPU/NPU floating-point discrepancies
        constexpr float kNearZeroThreshold = 1e-6f;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                float npuVal = npuOutputs[b][j * ldaInv + i];
                float goldVal = goldenOutputs[b][j * ldaInv + i];
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
// Normal path test helper
// ---------------------------------------------------------------------------
static void TestNormalPath(
    const MatinvBatchedParam& p, int n, int lda, int ldaInv, int batchSize, aclblasHandle_t handle)
{
    BlasLapackMatrixType lapackType = toBlasLapackType(p.matrixType);

    // 1. Generate input matrices
    std::vector<std::vector<float>> inputMatrices(batchSize);
    std::vector<const float*> inputPtrs(batchSize);
    std::vector<std::vector<float>> npuOutputs(batchSize);
    std::vector<float*> npuOutputPtrs(batchSize);

    for (int b = 0; b < batchSize; b++) {
        inputMatrices[b] = makeBlasLapackMatrix(n, lda, lapackType, p.randomSeed, b);
        inputPtrs[b] = inputMatrices[b].data();
        npuOutputs[b].resize(static_cast<size_t>(ldaInv) * n, 0.0f);
        npuOutputPtrs[b] = npuOutputs[b].data();
    }

    // 2. Generate golden data
    // Golden function: A[] (input, read-only) → Ainv[] (output, inverse written by golden)
    std::vector<std::vector<float>> goldenInputCopy(batchSize);
    std::vector<const float*> goldenInputCopyPtrs(batchSize);
    std::vector<std::vector<float>> goldenAinv(batchSize);
    std::vector<float*> goldenAinvPtrs(batchSize);

    for (int b = 0; b < batchSize; b++) {
        goldenInputCopy[b] = inputMatrices[b]; // copy of original input
        goldenInputCopyPtrs[b] = goldenInputCopy[b].data();
        goldenAinv[b].resize(static_cast<size_t>(ldaInv) * n, 0.0f);
        goldenAinvPtrs[b] = goldenAinv[b].data();
    }

    std::vector<int> goldenInfoClean(batchSize, -1);
    aclblasStatus_t goldenRet = aclblasSmatinvBatched_cpu(
        handle, n,
        goldenInputCopyPtrs.data(),
        lda,
        goldenAinvPtrs.data(),
        ldaInv,
        goldenInfoClean.data(),
        batchSize);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        FAIL() << "golden function failed with status " << static_cast<int>(goldenRet)
               << " (n=" << n << ", lda=" << lda << ", ldaInv=" << ldaInv << ", batchSize=" << batchSize << ")";
        return;
    }

    // 3. Run NPU
    std::vector<int> npuInfo(batchSize, -1);
    aclblasStatus_t ret = aclblasSmatinvBatched_npu(
        handle, n, inputPtrs.data(), lda, npuOutputPtrs.data(), ldaInv, npuInfo.data(), batchSize);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return;

    // 4. Verify info array
    VerifyInfoArray(npuInfo, goldenInfoClean, batchSize, p.caseName);

    VerifyInverseMatrices(
        npuOutputs, goldenAinv, goldenInfoClean,
        n, ldaInv, batchSize, p.caseName,
        p.mereThreshold, p.mareMultiplier);
}

// ---------------------------------------------------------------------------
// Main parameterized test
// ---------------------------------------------------------------------------
TEST_P(MatinvBatchedArch35Test, CsvDriven) {
    const auto& p = GetParam();
    const int n = p.n;
    const int batchSize = p.batchSize;

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        TestErrorPath(p, MatinvBatchedArch35Test::handle_);
    } else if (n <= 0 || batchSize <= 0) {
        TestNoOpPath(p, MatinvBatchedArch35Test::handle_);
    } else {
        TestNormalPath(p, n, p.lda, p.ldaInv, batchSize, MatinvBatchedArch35Test::handle_);
    }
}
