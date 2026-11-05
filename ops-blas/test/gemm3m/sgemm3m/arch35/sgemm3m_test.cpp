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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "sgemm3m_param.h"
#include "sgemm3m_golden.h"
#include "sgemm3m_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: prepare all test data (host vectors + pointers for merged A, B, C)
// A contains A1/A2/A3 merged along K; B contains B1/B2/B3 merged along K.
// Sub-matrices use different random seeds for independence (3M decomposition independence).
// ═══════════════════════════════════════════════════════════════════════════════

struct Gemm3mPreparedData {
    std::vector<float> aHost;
    std::vector<float> bHost;
    std::vector<float> cHost;
    const float* alphaPtr;
    const float* betaPtr;
    const float* aPtr;
    const float* bPtr;
    float* cPtr;
    float alphaVal;
    float betaVal;
};

// Pack 3 independent sub-matrices into a merged matrix (column-major).
// concatAlongRows=false: sub-matrices concatenated along columns (transA=N, transB=T)
//   merged has subRows rows and 3*subCols columns
// concatAlongRows=true: sub-matrices stacked along rows (transA=T, transB=N)
//   merged has 3*subRows rows and subCols columns
inline std::vector<float> PackMergedMatrix(
    const std::vector<float>& sub0, const std::vector<float>& sub1, const std::vector<float>& sub2,
    int subRows, int subCols, int ld, bool concatAlongRows)
{
    if (subRows <= 0 || subCols <= 0 || ld <= 0) return {};
    // Merged matrix physical dimensions (column-major)
    int mergedRows = concatAlongRows ? 3 * subRows : subRows;
    int mergedCols = concatAlongRows ? subCols : 3 * subCols;
    // Size must account for ld < mergedRows (same logic as makeBlasMatrix)
    const size_t storageSize = static_cast<size_t>(ld) * mergedCols;
    const size_t maxIndex = static_cast<size_t>(mergedRows - 1) + static_cast<size_t>(mergedCols - 1) * ld + 1;
    std::vector<float> merged(std::max(storageSize, maxIndex), 0.0f);

    const std::vector<float>* subs[3] = {&sub0, &sub1, &sub2};
    for (int s = 0; s < 3; s++) {
        const auto& sub = *subs[s];
        if (sub.empty()) continue;
        for (int col = 0; col < subCols; col++) {
            for (int row = 0; row < subRows; row++) {
                size_t subIdx = static_cast<size_t>(col) * ld + row;
                size_t mergedIdx = concatAlongRows
                    ? static_cast<size_t>(col) * ld + s * subRows + row
                    : static_cast<size_t>(s * subCols + col) * ld + row;
                merged[mergedIdx] = sub[subIdx];
            }
        }
    }
    return merged;
}

inline Gemm3mPreparedData Gemm3mPrepareTestData(const Gemm3mParam& p)
{
    Gemm3mPreparedData data;

    // Sub-matrix physical dimensions (each sub-matrix is M×K or K×M etc.)
    int subRowsA = (p.transA == ACLBLAS_OP_N) ? p.m : p.k;
    int subColsA = (p.transA == ACLBLAS_OP_N) ? p.k : p.m;
    int subRowsB = (p.transB == ACLBLAS_OP_N) ? p.k : p.n;
    int subColsB = (p.transB == ACLBLAS_OP_N) ? p.n : p.k;

    // Generate 3 independent sub-matrices with different seeds
    auto a1 = makeBlasMatrix(subRowsA, subColsA, p.lda, p.aFill, p.randomSeed + 0);
    auto a2 = makeBlasMatrix(subRowsA, subColsA, p.lda, p.aFill, p.randomSeed + 1);
    auto a3 = makeBlasMatrix(subRowsA, subColsA, p.lda, p.aFill, p.randomSeed + 2);
    auto b1 = makeBlasMatrix(subRowsB, subColsB, p.ldb, p.bFill, p.randomSeed + 3);
    auto b2 = makeBlasMatrix(subRowsB, subColsB, p.ldb, p.bFill, p.randomSeed + 4);
    auto b3 = makeBlasMatrix(subRowsB, subColsB, p.ldb, p.bFill, p.randomSeed + 5);
    data.cHost = makeBlasMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 6);

    // Pack sub-matrices into merged A and B
    // transA=N: A is M×(3K), sub-matrices concatenated along columns
    // transA=T: A is (3K)×M, sub-matrices stacked along rows
    bool aConcatAlongRows = (p.transA != ACLBLAS_OP_N);
    bool bConcatAlongRows = (p.transB == ACLBLAS_OP_N);
    data.aHost = PackMergedMatrix(a1, a2, a3, subRowsA, subColsA, p.lda, aConcatAlongRows);
    data.bHost = PackMergedMatrix(b1, b2, b3, subRowsB, subColsB, p.ldb, bConcatAlongRows);

    data.alphaVal = p.alpha;
    data.betaVal  = p.beta;
    data.alphaPtr = p.alphaNull ? nullptr : &data.alphaVal;
    data.betaPtr  = p.betaNull  ? nullptr : &data.betaVal;

    data.aPtr = (data.aHost.empty() || p.aNull) ? nullptr : data.aHost.data();
    data.bPtr = (data.bHost.empty() || p.bNull) ? nullptr : data.bHost.data();
    data.cPtr = (data.cHost.empty()  || p.cNull)  ? nullptr : data.cHost.data();

    return data;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test fixture
// ═══════════════════════════════════════════════════════════════════════════════

class Gemm3mArch35Test : public BlasTest<Gemm3mParam> {
};

// ── TEST_F: null handle ──
TEST_F(Gemm3mArch35Test, NullHandle)
{
    float alpha = 1.0f, beta = 0.0f;
    aclblasStatus_t ret = aclblasSgemm3m_npu(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8, &alpha,
        nullptr, 8, nullptr, 8,
        &beta, nullptr, 8);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    Gemm3m, Gemm3mArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<Gemm3mParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Gemm3mParam>);

// ── TEST_P: CSV-driven test (5-step flow) ──
//   1. Generate host data
//   2. Run NPU
//   3. Compare error code (early return for error cases)
//   4. Run CPU golden
//   5. Verify precision (Mixed Tolerance)
TEST_P(Gemm3mArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // Determine handle (nullptr for handle_null cases)
    aclblasHandle_t testHandle = Gemm3mArch35Test::handle_;
    if (p.description.find("handle_null") != std::string::npos) {
        testHandle = nullptr;
    }

    // Step 1: Generate host data
    auto data = Gemm3mPrepareTestData(p);

    // Step 1b: Save C_init copy BEFORE NPU execution (NPU overwrites cHost via cPtr)
    // CPU golden needs the original C_init, not the NPU result.
    std::vector<float> cInit(data.cHost);
    float* cInitPtr = cInit.empty() ? nullptr : cInit.data();

    // Step 2: Run NPU
    aclblasStatus_t ret = aclblasSgemm3m_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k, data.alphaPtr,
        data.aPtr, p.lda,
        data.bPtr, p.ldb,
        data.betaPtr, data.cPtr, p.ldc);

    // Step 3: Compare error code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) return;

    // Skip precision verification for M=0/N=0 (C not accessed) or C=nullptr
    if (p.m <= 0 || p.n <= 0 || data.cPtr == nullptr) return;

    // Step 3b: Skip precision verification for special-value cases (NaN/Inf/Extreme)
    // Per test design: these cases verify no-crash only, mixed tolerance not enforced.
    // NaN propagates, Inf saturates per IEEE 754; EXTREME may overflow to Inf/NaN.
    // The MixedToleranceStrategy handles NaN==NaN and Inf==Inf via shouldSkip, but
    // single-sided Inf or NaN-vs-finite mismatches would fail mixed tolerance thresholds.
    auto hasSpecialFill = [](const BlasFillMode& f) {
        if (f.pattern == BlasFillMode::P_EXTREME) return true;
        if (f.method == BlasFillMode::M_VALUE && (std::isnan(f.val1) || std::isinf(f.val1))) return true;
        return false;
    };
    if (hasSpecialFill(p.aFill) || hasSpecialFill(p.bFill)) {
        return;  // No-crash verification passed (expect_result == SUCCESS already checked)
    }

    // Step 4: Run CPU golden on the saved C_init copy (golden modifies C in-place)
    aclblasSgemm3m_cpu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k, data.alphaPtr,
        data.aPtr, p.lda,
        data.bPtr, p.ldb,
        data.betaPtr, cInitPtr, p.ldc);

    // Step 5: Verify precision (Mixed Tolerance, FP32)
    size_t cCount = static_cast<size_t>(p.ldc) * p.n;
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, cInitPtr, cCount);
    EXPECT_TRUE(Verifier::verifyVector(data.cPtr, cInitPtr, cCount, 1, cfg, p.caseName));
}
