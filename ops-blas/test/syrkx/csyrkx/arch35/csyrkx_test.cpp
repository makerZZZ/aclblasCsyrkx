/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// CSV-driven ST for aclblasCsyrkx (complex64, arch35 / Atlas 950PR).
//
// Verification regions (task doc §2.4/§2.5/§3.2):
//   1. uplo triangle vs cblas golden, real/imag separately:
//      |a-g| <= atol + rtol*|g|, matched_ratio >= 0.99, max_abs_error <= 1e-2 or 32*ULP;
//      non-finite elements follow the §3.2.4 equivalence rule (NaN/Inf handling)
//      and are excluded from the ratio/error statistics.
//   2. non-uplo triangle: bit-exact against the input-old C.
//   3. ldc padding rows (ldc > n): bit-exact against the host prefill.
//   4. return codes match expect_result.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "verify_uplo.h"
#include "csyrkx_param.h"
#include "csyrkx_golden.h"
#include "csyrkx_npu_wrapper.h"

class CsyrkxTest : public BlasTest<CsyrkxParam> {};

TEST_F(CsyrkxTest, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    aclblasComplex dummy{0.0f, 0.0f};
    aclblasStatus_t ret =
        aclblasCsyrkx(nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, &dummy, 4, &dummy, 4, &beta, &dummy, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

namespace {

bool IsNonFinite(float v) { return std::isnan(v) || std::isinf(v); }

// §3.2.4 equivalence: NaN/NaN, same-sign Inf/Inf and NaN/Inf are equivalent;
// a finite value against a non-finite one is not.
bool NonFiniteEquivalent(float actual, float golden)
{
    if (std::isnan(actual) && std::isnan(golden)) {
        return true;
    }
    if (std::isinf(actual) && std::isinf(golden)) {
        return actual == golden;
    }
    return (std::isnan(actual) && std::isinf(golden)) || (std::isinf(actual) && std::isnan(golden));
}

// Split a component pair list into finite (mixed-tolerance judged) and
// non-finite (§3.2.4 judged) buckets.
bool VerifyCsyrkxComponent(
    const std::string& caseName, const std::vector<float>& actual, const std::vector<float>& golden)
{
    std::vector<float> finiteActual;
    std::vector<float> finiteGolden;
    size_t nonFiniteCount = 0;
    size_t nonFinitePass = 0;
    for (size_t i = 0; i < actual.size(); i++) {
        if (IsNonFinite(actual[i]) || IsNonFinite(golden[i])) {
            nonFiniteCount++;
            nonFinitePass += NonFiniteEquivalent(actual[i], golden[i]) ? 1U : 0U;
        } else {
            finiteActual.push_back(actual[i]);
            finiteGolden.push_back(golden[i]);
        }
    }

    bool finitePass = true;
    if (!finiteActual.empty()) {
        VerifyConfig cfg;
        applyMixedTolerance(cfg, ACL_FLOAT, finiteGolden.data(), finiteGolden.size());
        finitePass = Verifier::verifyVector(
            finiteActual.data(), finiteGolden.data(), finiteActual.size(), 1, cfg, caseName);
    }
    const bool nonFiniteOk = (nonFinitePass == nonFiniteCount);
    if (nonFiniteCount > 0) {
        std::cout << "[" << caseName << "] non-finite elements: " << nonFinitePass << "/" << nonFiniteCount
                  << " equivalent" << std::endl;
    }
    return finitePass && nonFiniteOk;
}

// Bit-exact comparison for the non-uplo triangle and the ldc padding region.
bool VerifyExact(const std::string& caseName, const std::vector<float>& actual, const std::vector<float>& golden)
{
    if (actual.empty()) {
        return true;
    }
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    return Verifier::verifyVector(actual.data(), golden.data(), actual.size(), 1, cfg, caseName);
}

struct CsyrkxHostData {
    std::vector<aclblasComplex> aHost;
    std::vector<aclblasComplex> bHost;
    std::vector<aclblasComplex> cHost;
    std::vector<aclblasComplex> originalC;
    std::vector<aclblasComplex> cGolden;
    const aclblasComplex* aPtr = nullptr;
    const aclblasComplex* bPtr = nullptr;
    aclblasComplex* cPtr = nullptr;
    size_t cCount = 0;
};

// Host footprint guard (3 C copies + A + B); very large perf cases are run by
// the performance harness instead of the accuracy ST.
bool PrepareHostData(const CsyrkxParam& p, CsyrkxHostData& d)
{
    if (p.n <= 0 || p.ldc <= 0) {
        return true;
    }
    const int abCols = (p.k > 0) ? ((p.trans == ACLBLAS_OP_N) ? p.k : p.n) : 0;
    const size_t cBytes = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) * sizeof(aclblasComplex);
    size_t bytes = 3U * cBytes;
    if (abCols > 0 && p.lda > 0 && p.ldb > 0) {
        bytes += static_cast<size_t>(p.lda) * static_cast<size_t>(abCols) * sizeof(aclblasComplex);
        bytes += static_cast<size_t>(p.ldb) * static_cast<size_t>(abCols) * sizeof(aclblasComplex);
    }
    constexpr size_t kHostMemLimit = 1536ULL * 1024ULL * 1024ULL;
    if (bytes > kHostMemLimit) {
        std::cout << "[SKIP] host memory estimate (" << bytes / (1024 * 1024) << " MB) exceeds limit for n=" << p.n
                  << ", k=" << p.k << std::endl;
        return false;
    }

    const uint32_t seedA = p.randomSeed;
    const uint32_t seedB = p.randomSeed + 1U;
    const uint32_t seedC = p.randomSeed + 2U;
    try {
        d.aHost = p.nullA ? std::vector<aclblasComplex>{}
                          : makeBlasComplexMatrix(p.lda, abCols, p.lda, p.aFill, seedA);
        d.bHost = p.nullB ? std::vector<aclblasComplex>{}
                          : makeBlasComplexMatrix(p.ldb, abCols, p.ldb, p.bFill, seedB);
        d.cHost = p.nullC ? std::vector<aclblasComplex>{}
                          : makeBlasComplexMatrix(p.n, p.n, p.ldc, p.cFill, seedC);
    } catch (const std::bad_alloc&) {
        std::cout << "[SKIP] host allocation failed for n=" << p.n << ", k=" << p.k << std::endl;
        return false;
    }
    d.aPtr = d.aHost.empty() ? nullptr : d.aHost.data();
    d.bPtr = d.bHost.empty() ? nullptr : d.bHost.data();
    d.cPtr = d.cHost.empty() ? nullptr : d.cHost.data();
    d.cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);
    if (d.cPtr != nullptr) {
        d.originalC.assign(d.cHost.begin(), d.cHost.end());
        d.cGolden.assign(d.cHost.begin(), d.cHost.end());
    }
    return true;
}

// Region 2: non-uplo triangle must equal the input-old values.
bool VerifyNonUplo(const CsyrkxParam& p, const CsyrkxHostData& d)
{
    std::vector<float> actual;
    std::vector<float> golden;
    for (int j = 0; j < p.n; j++) {
        for (int i = 0; i < p.n; i++) {
            const bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo) {
                continue;
            }
            const size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(p.ldc);
            actual.push_back(d.cPtr[idx].real);
            golden.push_back(d.originalC[idx].real);
            actual.push_back(d.cPtr[idx].imag);
            golden.push_back(d.originalC[idx].imag);
        }
    }
    return VerifyExact(p.caseName + "_nonuplo", actual, golden);
}

// Region 3: ldc padding rows are never written.
bool VerifyLdcPadding(const CsyrkxParam& p, const CsyrkxHostData& d)
{
    if (p.ldc <= p.n) {
        return true;
    }
    std::vector<float> actual;
    std::vector<float> golden;
    for (int j = 0; j < p.n; j++) {
        for (int i = p.n; i < p.ldc; i++) {
            const size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(p.ldc);
            actual.push_back(d.cPtr[idx].real);
            golden.push_back(d.originalC[idx].real);
            actual.push_back(d.cPtr[idx].imag);
            golden.push_back(d.originalC[idx].imag);
        }
    }
    return VerifyExact(p.caseName + "_ldc_padding", actual, golden);
}

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Csyrkx, CsyrkxTest, ::testing::ValuesIn(GetCasesFromCsv<CsyrkxParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CsyrkxParam>);

TEST_P(CsyrkxTest, CsvDriven)
{
    const auto& p = GetParam();
    CsyrkxHostData d;
    if (!PrepareHostData(p, d)) {
        GTEST_SKIP() << "Skipped: host memory limit";
    }

    const aclblasComplex alpha{p.alphaReal, p.alphaImag};
    const aclblasComplex beta{p.betaReal, p.betaImag};
    const aclblasComplex* alphaPtr = p.nullAlpha ? nullptr : &alpha;
    const aclblasComplex* betaPtr = p.nullBeta ? nullptr : &beta;

    aclblasStatus_t ret = aclblasCsyrkx_npu(
        CsyrkxTest::handle_, p.uplo, p.trans, p.n, p.k, alphaPtr, d.aPtr, p.lda, d.bPtr, p.ldb, betaPtr, d.cPtr, p.ldc);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
        << "case=" << p.caseName << " expected=" << static_cast<int>(p.expectResult)
        << " got=" << static_cast<int>(ret);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0 || d.cPtr == nullptr) {
        return;
    }

    // Golden: alpha/beta are valid here (null-scalar cases expect INVALID_VALUE
    // and return above). Scalars are host values by construction.
    aclblasStatus_t goldenRet = aclblasCsyrkx_cpu(
        CsyrkxTest::handle_, p.uplo, p.trans, p.n, p.k, alphaPtr, d.aPtr, p.lda, d.bPtr, p.ldb, betaPtr,
        d.cGolden.data(), p.ldc, d.originalC.data());
    ASSERT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden failed for case=" << p.caseName;

    // Region 1: uplo triangle, real/imag judged separately.
    std::vector<float> npuUploRe, npuUploIm, goldUploRe, goldUploIm;
    std::vector<float> npuNonRe, npuNonIm, oldNonRe, oldNonIm;
    CollectUploNonUploElementsT<CsyrkxParam, float, aclblasComplex>(
        p, d.cPtr, d.cGolden.data(), npuUploRe, npuUploIm, goldUploRe, goldUploIm, npuNonRe, npuNonIm, oldNonRe,
        oldNonIm);
    EXPECT_TRUE(VerifyCsyrkxComponent(p.caseName + "_uplo_real", npuUploRe, goldUploRe));
    EXPECT_TRUE(VerifyCsyrkxComponent(p.caseName + "_uplo_imag", npuUploIm, goldUploIm));

    // Regions 2 and 3.
    EXPECT_TRUE(VerifyNonUplo(p, d));
    EXPECT_TRUE(VerifyLdcPadding(p, d));
}
