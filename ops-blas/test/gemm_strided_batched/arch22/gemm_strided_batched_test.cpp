/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

#ifndef TEST_DEVICE_ID
#define TEST_DEVICE_ID 0
#endif
constexpr int DEFAULT_DEVICE = TEST_DEVICE_ID;
#undef TEST_DEVICE_ID
inline int GemmSbTestDevice()
{
    const char* value = std::getenv("ASCEND_DEVICE_ID");
    return value ? std::atoi(value) : DEFAULT_DEVICE;
}
#define TEST_DEVICE_ID GemmSbTestDevice()
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "gemm_strided_batched_param.h"
#include "gemm_strided_batched_golden.h"
#include "gemm_strided_batched_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Prepared host data spanning all batches (column-major, strides pre-applied).
// ═══════════════════════════════════════════════════════════════════════════════
struct GsbData {
    std::vector<float> aHost;   // A span over all batches
    std::vector<float> bHost;   // B span over all batches
    std::vector<float> cHost;   // C span (NPU output after run)
    std::vector<float> cGolden; // C span (CPU golden output)
};

inline GsbData PrepareData(const GemmStridedBatchedParam& p)
{
    GsbData d;
    const int physRowsA = gsbPhysRows(p.m, p.k, p.transA);
    const int physColsA = gsbPhysCols(p.m, p.k, p.transA);
    const int physRowsB = gsbPhysRows(p.k, p.n, p.transB);
    const int physColsB = gsbPhysCols(p.k, p.n, p.transB);

    BlasFillMode aFill = p.aNull ? parseFill("NULLPTR") : p.aFill;
    BlasFillMode bFill = p.bNull ? parseFill("NULLPTR") : p.bFill;
    BlasFillMode cFill = p.cNull ? parseFill("NULLPTR") : p.cFill;

    d.aHost = gsbMakeStridedBatched(physRowsA, physColsA, p.lda, p.effStrideA(), p.batchCount, aFill, p.randomSeed);
    d.bHost =
        gsbMakeStridedBatched(physRowsB, physColsB, p.ldb, p.effStrideB(), p.batchCount, bFill, p.randomSeed + 100);
    d.cHost = gsbMakeStridedBatched(p.m, p.n, p.ldc, p.effStrideC(), p.batchCount, cFill, p.randomSeed + 200);
    if (p.caseName.find("TC_RG_NORMAL_") == 0) {
        std::mt19937 rng(p.randomSeed);
        std::uniform_real_distribution<float> mean(-5.0f, 5.0f), sigma(0.1f, 2.0f);
        for (auto* values : {&d.aHost, &d.bHost, &d.cHost}) {
            std::normal_distribution<float> normal(mean(rng), sigma(rng));
            for (float& value : *values)
                value = normal(rng);
        }
    }
    d.cGolden = d.cHost; // identical starting C for the beta*C term
    return d;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fixture
// ═══════════════════════════════════════════════════════════════════════════════
class GemmStridedBatchedArch22Test : public BlasTest<GemmStridedBatchedParam> {
protected:
    static void SetUpTestSuite()
    {
        blas_test_detail::globalAcl().cleaned = false;
        BlasTest<GemmStridedBatchedParam>::SetUpTestSuite();
    }
};

// ── TEST_F: null handle (not driven by CSV verify path) ──
TEST_F(GemmStridedBatchedArch22Test, NullHandle)
{
    float alpha = 1.0f, beta = 0.0f;
    aclblasStatus_t ret = aclblasSgemmStridedBatched_npu(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, 4, &alpha, nullptr, 4, 16, nullptr, 4, 16, &beta, nullptr, 4, 16, 1,
        0, 0, 0);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

inline std::vector<GemmStridedBatchedParam> LoadArch22Cases()
{
    std::string path = ReplaceFileExtension2Csv(__FILE__);
    const char* fullTest = std::getenv("GSB_FULL_TEST");
    if (fullTest == nullptr || std::strcmp(fullTest, "1") != 0) {
        path.replace(path.rfind("_test.csv"), 9, "_smoke.csv");
        return GetCasesFromCsv<GemmStridedBatchedParam>(path);
    }
    auto cases = GetCasesFromCsv<GemmStridedBatchedParam>(path);
    path.replace(path.rfind("_test.csv"), 9, "_regression.csv");
    auto extra = GetCasesFromCsv<GemmStridedBatchedParam>(path);
    cases.insert(cases.end(), extra.begin(), extra.end());
    return cases;
}

TEST_F(GemmStridedBatchedArch22Test, NegativeStrides)
{
    float alpha = 1, beta = 0, value = 1;
    for (int which = 0; which < 3; ++which) {
        EXPECT_EQ(
            aclblasSgemmStridedBatched(
                handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 1, 1, 1, &alpha, &value, 1, which == 0 ? -1 : 1, &value, 1,
                which == 1 ? -1 : 1, &beta, &value, 1, which == 2 ? -1 : 1, 1),
            ACLBLAS_STATUS_INVALID_VALUE);
    }
}

// CSV error cases return before the golden runs, so the golden's own trans
// validation is checked directly here to match the operator's INVALID_ENUM contract.
TEST_F(GemmStridedBatchedArch22Test, GoldenInvalidTransReturnsEnum)
{
    float alpha = 1, beta = 0, value = 1;
    const auto badOp = static_cast<aclblasOperation_t>(99);
    EXPECT_EQ(static_cast<int>(ValidateGemmStridedBatchedParams(
                  handle_, badOp, ACLBLAS_OP_N, 4, 4, 4, &alpha, &value, 4, &value, 4, &beta, &value, 4, 1)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM));
    EXPECT_EQ(static_cast<int>(ValidateGemmStridedBatchedParams(
                  handle_, ACLBLAS_OP_N, badOp, 4, 4, 4, &alpha, &value, 4, &value, 4, &beta, &value, 4, 1)),
        static_cast<int>(ACLBLAS_STATUS_INVALID_ENUM));
}

INSTANTIATE_TEST_SUITE_P(
    GemmStridedBatched, GemmStridedBatchedArch22Test, ::testing::ValuesIn(LoadArch22Cases()),
    PrintCaseInfoString<GemmStridedBatchedParam>);

// ── Accuracy comparison helpers (thresholds mirror the ecosystem FP32 standard). ──
struct GsbCompareResult {
    size_t matched;
    size_t total;
    double maxError;
    bool absoluteOk;
};

inline void CompareOneElement(
    GsbCompareResult& r, int& nonfiniteReported, size_t idx, float actual, float expected)
{
    ++r.total;
    if (!std::isfinite(expected) || !std::isfinite(actual)) {
        const bool same = (std::isnan(expected) && std::isnan(actual)) || actual == expected;
        if (!same && nonfiniteReported++ < 4)
            std::cout << "[NONFINITE] index=" << idx << " actual=" << actual << " expected=" << expected << std::endl;
        r.matched += same;
        r.absoluteOk &= same;
        return;
    }
    const double error = std::abs(static_cast<double>(actual) - expected);
    r.maxError = std::max(r.maxError, error);
    r.matched += error <= std::ldexp(1.0, -16) + std::ldexp(1.0, -10) * std::abs(expected);
    const double ulp =
        static_cast<double>(std::nextafter(std::abs(expected), std::numeric_limits<float>::infinity())) -
        std::abs(expected);
    r.absoluteOk &= error <= 0.01 || error <= 32 * ulp;
}

// Judges only logical output elements and marks them; padding stays bitwise-checked by the caller.
inline GsbCompareResult CompareLogicalElements(
    const GemmStridedBatchedParam& p, const GsbData& d, std::vector<bool>& logical)
{
    GsbCompareResult r{0, 0, 0.0, true};
    int nonfiniteReported = 0;
    for (int batch = 0; batch < p.batchCount; ++batch) {
        for (int col = 0; col < p.n; ++col) {
            for (int row = 0; row < p.m; ++row) {
                const size_t idx = batch * p.effStrideC() + static_cast<size_t>(col) * p.ldc + row;
                logical[idx] = true;
                CompareOneElement(r, nonfiniteReported, idx, d.cHost[idx], d.cGolden[idx]);
            }
        }
    }
    return r;
}

// Returns the first non-logical index whose padding bytes changed, or logical.size() when intact.
inline size_t FirstPaddingMismatch(const GsbData& d, const std::vector<bool>& logical)
{
    for (size_t i = 0; i < logical.size(); ++i)
        if (!logical[i] && std::memcmp(&d.cHost[i], &d.cGolden[i], sizeof(float)) != 0)
            return i;
    return logical.size();
}

TEST_P(GemmStridedBatchedArch22Test, CsvDriven)
{
    const auto& p = GetParam();

    // handle==null cases run with a null handle.
    aclblasHandle_t testHandle = GemmStridedBatchedArch22Test::handle_;
    if (p.expectResult == ACLBLAS_STATUS_HANDLE_IS_NULLPTR) {
        testHandle = nullptr;
    }

    GsbData d = PrepareData(p);

    const float* alphaPtr = p.alphaNull ? nullptr : &p.alpha;
    const float* betaPtr = p.betaNull ? nullptr : &p.beta;
    const float* aPtr = d.aHost.empty() ? nullptr : d.aHost.data();
    const float* bPtr = d.bHost.empty() ? nullptr : d.bHost.data();
    float* cPtr = d.cHost.empty() ? nullptr : d.cHost.data();

    const size_t aBytes = d.aHost.size() * sizeof(float);
    const size_t bBytes = d.bHost.size() * sizeof(float);
    const size_t cBytes = d.cHost.size() * sizeof(float);

    // Step 2: run on NPU (device malloc/H2D/kernel/sync/D2H/free inside wrapper).
    aclblasStatus_t ret = aclblasSgemmStridedBatched_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtr, p.lda, p.effStrideA(), bPtr, p.ldb,
        p.effStrideB(), betaPtr, cPtr, p.ldc, p.effStrideC(), p.batchCount, aBytes, bBytes, cBytes, p.caseName.c_str());

    // Step 3: error / expected-return-code check.
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    // No-computation cases (m/n/batchCount==0): SUCCESS with C unchanged, no numeric compare.
    if (p.m == 0 || p.n == 0 || p.batchCount == 0 || cPtr == nullptr) {
        return;
    }

    // Step 4: CPU golden on the identical starting C copy.
    aclblasStatus_t goldenRet = aclblasSgemmStridedBatched_cpu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtr, p.lda, p.effStrideA(), bPtr, p.ldb,
        p.effStrideB(), betaPtr, d.cGolden.data(), p.ldc, p.effStrideC(), p.batchCount);
    ASSERT_EQ(static_cast<int>(goldenRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    std::vector<bool> logical(d.cHost.size(), false);
    const GsbCompareResult r = CompareLogicalElements(p, d, logical);
    const size_t padding = FirstPaddingMismatch(d, logical);
    ASSERT_EQ(padding, logical.size()) << "padding changed at " << padding;

    const double ratio = static_cast<double>(r.matched) / r.total;
    std::cout << "[ACCURACY] " << p.caseName << " matched_ratio=" << ratio << " max_abs_error=" << r.maxError
              << std::endl;
    EXPECT_GE(ratio, 0.99);
    EXPECT_TRUE(r.absoluteOk);
}
