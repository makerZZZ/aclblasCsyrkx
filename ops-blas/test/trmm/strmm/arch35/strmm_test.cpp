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
#include <limits>
#include <string>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "strmm_param.h"
#include "strmm_golden.h"
#include "strmm_npu_wrapper.h"

// Column-major matrix buffer: m×n with leading dimension ld.
// makeBlasMatrix handles m<=0 / n<=0 / ld<=0 by returning an empty vector.
static inline std::vector<float> MakeMatrix(int rows, int cols, int ld,
    const BlasFillMode& fill, uint32_t seed)
{
    return makeBlasMatrix(rows, cols, ld, fill, seed);
}

// Triangular matrix: dimA×dimA column-major, only the uplo triangle is filled.
// The full matrix is generated with the original fill (preserving the CSV-specified
// val1/val2 value range), then the non-triangular part is manually zeroed by uplo.
static inline std::vector<float> MakeTriMatrix(int dimA, int lda,
    aclblasFillMode_t uplo, const BlasFillMode& fill, uint32_t seed)
{
    std::vector<float> data = makeBlasMatrix(dimA, dimA, lda, fill, seed);
    if (data.empty()) return data;
    for (int col = 0; col < dimA; ++col) {
        for (int row = 0; row < dimA; ++row) {
            bool inTriangle = (uplo == ACLBLAS_UPPER) ? (row <= col) : (row >= col);
            if (!inTriangle) {
                data[static_cast<size_t>(col) * lda + row] = 0.0f;
            }
        }
    }
    return data;
}

static inline void ApplySpecialValues(const std::string& desc,
    std::vector<float>& aHost, std::vector<float>& bHost)
{
    if (desc.find("nanA") != std::string::npos || desc.find("nan_A") != std::string::npos) {
        std::fill(aHost.begin(), aHost.end(), std::numeric_limits<float>::quiet_NaN());
    }
    if (desc.find("infA") != std::string::npos || desc.find("inf_A") != std::string::npos) {
        std::fill(aHost.begin(), aHost.end(), std::numeric_limits<float>::infinity());
    }
    if (desc.find("nanB") != std::string::npos || desc.find("nan_B") != std::string::npos) {
        std::fill(bHost.begin(), bHost.end(), std::numeric_limits<float>::quiet_NaN());
    }
}

class StrmmArch35Test : public BlasTest<StrmmParam> {};

TEST_F(StrmmArch35Test, NullHandle)
{
    float alpha = 1.0f;
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    std::vector<float> c(16, 0.0f);
    aclblasStatus_t ret = aclblasStrmm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 4, 4,
        &alpha, a.data(), 4, b.data(), 4, c.data(), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

static bool MallocDev(void** dPtr, size_t bytes)
{
    aclError ret = aclrtMalloc(dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(static_cast<int>(ret), ACL_SUCCESS) << "aclrtMalloc failed for dC";
    return ret == ACL_SUCCESS;
}

static std::vector<float> MakeUpperTriangular(int lda, int m)
{
    std::vector<float> a(static_cast<size_t>(lda) * m, 0.0f);
    for (int col = 0; col < m; ++col) {
        for (int row = 0; row <= col; ++row) {
            a[static_cast<size_t>(col) * lda + row] = 1.0f;
        }
    }
    return a;
}

static void VerifyAlphaZeroPadding(const std::vector<float>& cHost, int m, int n, int ldc, float sentinel)
{
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            EXPECT_FLOAT_EQ(cHost[static_cast<size_t>(j) * ldc + i], 0.0f)
                << "active row " << i << " col " << j << " should be zeroed";
        }
        for (int i = m; i < ldc; ++i) {
            EXPECT_FLOAT_EQ(cHost[static_cast<size_t>(j) * ldc + i], sentinel)
                << "padding row " << i << " col " << j << " should be preserved";
        }
    }
}

// alpha==0 fast path with ldc > m: HandleStrmmAlphaZero zeros the m active rows
// of each column via per-column memset, while padding rows (m..ldc-1) are
// preserved. Device C is pre-filled with a non-zero sentinel to verify that
// only the active rows are cleared and padding rows remain untouched.
TEST_F(StrmmArch35Test, AlphaZeroLdcPadding)
{
    const int m = 4;
    const int n = 4;
    const int lda = 4;
    const int ldb = 4;
    const int ldc = 8;  // ldc > m — padding rows exist (rows 4..7)
    const float alpha = 0.0f;
    const float sentinel = 42.0f;

    // Valid A (4x4 upper triangular) and B (4x4). When alpha==0 the host
    // fast path skips A/B entirely, but valid pointers exercise the full
    // parameter validation path before the alpha==0 early return.
    std::vector<float> aHost = MakeUpperTriangular(lda, m);
    std::vector<float> bHost(static_cast<size_t>(ldb) * n, 1.0f);

    // Pre-fill device C with a non-zero sentinel.
    const size_t cCount = static_cast<size_t>(ldc) * static_cast<size_t>(n);
    const size_t cBytes = cCount * sizeof(float);
    std::vector<float> cHost(cCount, sentinel);

    void* dC = nullptr;
    if (!MallocDev(&dC, cBytes)) return;
    aclError aclRet = aclrtMemcpy(dC, cBytes, cHost.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        EXPECT_EQ(static_cast<int>(aclRet), ACL_SUCCESS) << "aclrtMemcpy H2D failed for dC";
        FreeDev(dC);
        return;
    }

    aclblasStatus_t ret = aclblasStrmm(
        StrmmArch35Test::handle_,
        ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        m, n, &alpha, aHost.data(), lda, bHost.data(), ldb,
        static_cast<float*>(dC), ldc);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        ret = StrmmSyncAndCopyD2H(StrmmArch35Test::handle_, dC, cHost.data(), cBytes);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    }
    FreeDev(dC);

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        VerifyAlphaZeroPadding(cHost, m, n, ldc, sentinel);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Strmm, StrmmArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<StrmmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StrmmParam>);

// Builds the triangular A and rectangular B host buffers, applies special
// NaN/Inf values from the case description, and resolves the alpha/A/B pointers
// (nullptr when the corresponding null flag is set). Sets alphaOnDevice when the
// case description contains "dev_alpha" (device-pointer alpha path).
static void PrepareTestData(const StrmmParam& p, int aDim,
    std::vector<float>& aHost, std::vector<float>& bHost,
    const float*& alphaPtr, const float*& aPtr, const float*& bPtr,
    bool& alphaOnDevice)
{
    aHost = MakeTriMatrix(aDim, p.lda, p.uplo, p.aFill, p.randomSeed);
    bHost = MakeMatrix(p.m, p.n, p.ldb, p.bFill, p.randomSeed + 1);
    ApplySpecialValues(p.description, aHost, bHost);

    alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    aPtr = p.nullA ? nullptr : (aHost.empty() ? nullptr : aHost.data());
    bPtr = p.nullB ? nullptr : (bHost.empty() ? nullptr : bHost.data());

    alphaOnDevice = (!p.nullAlpha && p.description.find("dev_alpha") != std::string::npos);
}

// alpha==0 with nullA/nullB: the host alpha==0 fast path memsets C without
// reading A/B. A device dC is used so the host's aclrtMemset fast path works on
// device memory, then D2H back to the host C buffer. The result is
// bit-reproducible (C is all zeros), verified with EXACT equality.
static void HandleAlphaZeroNullAB(const StrmmParam& p, aclblasHandle_t handle,
    const float* alphaPtr, const float* aPtr, const float* bPtr, float* cPtr,
    bool alphaOnDevice)
{
    const size_t cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);
    const size_t cBytes = cCount * sizeof(float);
    void* dC = nullptr;
    if (!MallocDev(&dC, cBytes)) return;
    void* dAlpha = nullptr;
    const float* alphaArg = alphaPtr;
    if (alphaOnDevice && alphaPtr != nullptr) {
        aclError alphaRet = aclrtMalloc(&dAlpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (alphaRet != ACL_SUCCESS) {
            EXPECT_EQ(static_cast<int>(alphaRet), ACL_SUCCESS) << "aclrtMalloc failed for dAlpha";
            FreeDev(dC);
            return;
        }
        alphaRet = aclrtMemcpy(dAlpha, sizeof(float), alphaPtr, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (alphaRet != ACL_SUCCESS) {
            EXPECT_EQ(static_cast<int>(alphaRet), ACL_SUCCESS) << "aclrtMemcpy failed for dAlpha";
            FreeDev(dAlpha);
            FreeDev(dC);
            return;
        }
        alphaArg = static_cast<const float*>(dAlpha);
    }
    aclblasStatus_t ret = aclblasStrmm(
        handle,
        p.side, p.uplo, p.trans, p.diag, p.m, p.n,
        alphaArg, aPtr, p.lda, bPtr, p.ldb,
        static_cast<float*>(dC), p.ldc);
    if (ret == ACLBLAS_STATUS_SUCCESS && cPtr != nullptr) {
        ret = StrmmSyncAndCopyD2H(handle, dC, cPtr, cBytes);
    }
    FreeDev(dAlpha);
    FreeDev(dC);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret == ACLBLAS_STATUS_SUCCESS && cPtr != nullptr) {
        for (size_t i = 0; i < cCount; i++) {
            EXPECT_EQ(cPtr[i], 0.0f) << "C should be all zeros when alpha==0";
        }
    }
}

// Computes the golden C via aclblasStrmm_cpu and verifies the NPU result with
// mixed tolerance. alpha==0 uses EXACT equality (C = 0, bit-reproducible).
static void RunGoldenAndVerify(const StrmmParam& p, aclblasHandle_t handle,
    const float* alphaPtr, const float* aPtr, const float* bPtr, float* cPtr)
{
    std::vector<float> cGolden = std::vector<float>(static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n), 0.0f);
    float* goldPtr = cGolden.empty() ? nullptr : cGolden.data();

    aclblasStatus_t goldenRet = aclblasStrmm_cpu(
        handle,
        p.side, p.uplo, p.trans, p.diag, p.m, p.n,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb, goldPtr, p.ldc);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden computation failed";
        return;
    }

    const size_t cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);
    if (cCount == 0) return;

    VerifyConfig cfg;
    if (p.alpha == 0.0f) {
        cfg.mode = PrecisionMode::EXACT;
    } else {
        applyMixedTolerance(cfg, ACL_FLOAT, goldPtr, cCount);
    }
    EXPECT_TRUE(Verifier::verifyVector(cPtr, goldPtr, cCount, 1, cfg, p.caseName));
}

TEST_P(StrmmArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const int aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;

    std::vector<float> aHost, bHost;
    const float* alphaPtr = nullptr;
    const float* aPtr = nullptr;
    const float* bPtr = nullptr;
    bool alphaOnDevice = false;
    PrepareTestData(p, aDim, aHost, bHost, alphaPtr, aPtr, bPtr, alphaOnDevice);

    if (p.m <= 0 || p.n <= 0 || p.lda <= 0 || p.ldb <= 0 || p.ldc <= 0) {
        float dummy = 0.0f;
        aclblasStatus_t ret = aclblasStrmm_npu(
            StrmmArch35Test::handle_,
            p.side, p.uplo, p.trans, p.diag, p.m, p.n,
            alphaPtr, aPtr, p.lda, bPtr, p.ldb, &dummy, p.ldc);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    std::vector<float> cHost(static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n), 0.0f);
    float* cPtr = p.nullC ? nullptr : cHost.data();

    if (p.alpha == 0.0f && !p.nullAlpha && (p.nullA || p.nullB)) {
        HandleAlphaZeroNullAB(p, StrmmArch35Test::handle_, alphaPtr, aPtr, bPtr, cPtr, alphaOnDevice);
        return;
    }

    aclblasStatus_t ret = aclblasStrmm_npu(
        StrmmArch35Test::handle_,
        p.side, p.uplo, p.trans, p.diag, p.m, p.n,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb, cPtr, p.ldc,
        alphaOnDevice);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    RunGoldenAndVerify(p, StrmmArch35Test::handle_, alphaPtr, aPtr, bPtr, cPtr);
}
