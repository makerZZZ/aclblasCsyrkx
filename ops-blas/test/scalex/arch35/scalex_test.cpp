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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "scalex_param.h"
#include "scalex_golden.h"
#include "scalex_npu_wrapper.h"

// ── Test fixture ─────────────────────────────────────────────────────────────
class ScalexArch35Test : public BlasTest<ScalexParam> {};

// ── TEST_F: null handle (not in CSV) ─────────────────────────────────────────
TEST_F(ScalexArch35Test, NullHandle)
{
    float alphaVal = 2.5f;
    std::vector<float> xHost = makeBlasStrided(5, 1, "RANDOM", 42);
    aclblasStatus_t ret = aclblasScalex_npu(nullptr, 5, &alphaVal, ACL_FLOAT, xHost.data(), ACL_FLOAT, 1, ACL_FLOAT);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

TEST_F(ScalexArch35Test, NullHandleTakesPrecedenceOverQuickReturn)
{
    EXPECT_EQ(
        aclblasScalex(nullptr, 0, nullptr, ACL_FLOAT, nullptr, ACL_FLOAT, 1, ACL_FLOAT),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasScalex(nullptr, 5, nullptr, ACL_FLOAT, nullptr, ACL_FLOAT, 0, ACL_FLOAT),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(ScalexArch35Test, QuickReturnSkipsDataPointerValidation)
{
    EXPECT_EQ(
        aclblasScalex(ScalexArch35Test::handle_, 0, nullptr, ACL_FLOAT, nullptr, ACL_FLOAT, 1, ACL_FLOAT),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasScalex(ScalexArch35Test::handle_, 5, nullptr, ACL_FLOAT, nullptr, ACL_FLOAT, 0, ACL_FLOAT),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasScalex(ScalexArch35Test::handle_, 5, nullptr, ACL_FLOAT, nullptr, ACL_FLOAT, -1, ACL_FLOAT),
        ACLBLAS_STATUS_SUCCESS);
}

// ── TEST_F: alpha nullptr (not in CSV) ───────────────────────────────────────
TEST_F(ScalexArch35Test, AlphaNull)
{
    std::vector<float> xHost = makeBlasStrided(5, 1, "RANDOM", 42);
    aclblasStatus_t ret =
        aclblasScalex_npu(ScalexArch35Test::handle_, 5, nullptr, ACL_FLOAT, xHost.data(), ACL_FLOAT, 1, ACL_FLOAT);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

// ── CSV parameterised test suite ─────────────────────────────────────────────
INSTANTIATE_TEST_SUITE_P(
    Scalex, ScalexArch35Test, ::testing::ValuesIn(GetCasesFromCsv<ScalexParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ScalexParam>);

// ── TEST_P: 5-step CSV-driven flow ───────────────────────────────────────────
TEST_P(ScalexArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // Step 1: Generate host data
    std::vector<float> xHost;
    if (p.x.method != BlasFillMode::M_NULLPTR) {
        xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    }

    float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float alphaLocal = p.alphaVal;
    const void* alphaPtr = &alphaLocal;

    // Step 2: Save golden copy before NPU mutates xHost
    std::vector<float> golden = xHost;

    // Step 3: Execute NPU (wrapper handles nullptrs, packing, device memory)
    aclblasStatus_t ret = aclblasScalex_npu(
        ScalexArch35Test::handle_, p.n, alphaPtr, static_cast<aclDataType>(p.alphaType), xPtr,
        static_cast<aclDataType>(p.xType), p.incx, static_cast<aclDataType>(p.executionType), p.alphaOnDevice);

    // Step 3: Verify expected error code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    // Step 4: Compute golden (in-place on saved copy)
    aclblasStatus_t cpuRet = aclblasScalex_cpu(
        ScalexArch35Test::handle_, p.n, alphaPtr, static_cast<aclDataType>(p.alphaType), golden.data(),
        static_cast<aclDataType>(p.xType), p.incx, static_cast<aclDataType>(p.executionType));
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    // Step 5: Precision verification with mixed tolerance (threshold per dtype)
    VerifyConfig cfg;
    applyMixedTolerance(cfg, static_cast<aclDataType>(p.xType), golden.data(), static_cast<size_t>(std::max(0, p.n)));

    int absIncx = std::abs(p.incx);
    EXPECT_TRUE(
        Verifier::verifyVector(xPtr, golden.data(), static_cast<size_t>(std::max(0, p.n)), absIncx, cfg, p.caseName));
}
