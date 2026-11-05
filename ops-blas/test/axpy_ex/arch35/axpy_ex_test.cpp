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
#include "axpy_ex_param.h"
#include "axpy_ex_golden.h"
#include "axpy_ex_npu_wrapper.h"

// ── Test fixture ─────────────────────────────────────────────────────────────
class AxpyExArch35Test : public BlasTest<AxpyExParam> {};

// ── TEST_F: null handle (TF_01, not in CSV) ─────────────────────────────────
TEST_F(AxpyExArch35Test, NullHandle)
{
    float alphaVal = 2.5f;
    std::vector<float> xHost = makeBlasStrided(5, 1, "RANDOM_1_1", 42);
    std::vector<float> yHost = makeBlasStrided(5, 1, "RANDOM_1_1", 42);
    aclblasStatus_t ret = aclblasAxpyEx_npu(
        nullptr, 5, &alphaVal, ACL_FLOAT, xHost.data(), ACL_FLOAT, 1, yHost.data(), ACL_FLOAT, 1, ACL_FLOAT);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// ── CSV parameterised test suite ─────────────────────────────────────────────
INSTANTIATE_TEST_SUITE_P(
    AxpyEx, AxpyExArch35Test, ::testing::ValuesIn(GetCasesFromCsv<AxpyExParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AxpyExParam>);

// ── TEST_P: 5-step CSV-driven flow ───────────────────────────────────────────
TEST_P(AxpyExArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // Step 1: Generate host data (x and y independently)
    std::vector<float> xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed);

    const void* xPtr = xHost.empty() ? nullptr : xHost.data();
    void* yPtr = yHost.empty() ? nullptr : yHost.data();

    // R01: alphaIsNull=true → alphaPtr=nullptr (triggers op alpha==nullptr check)
    float alphaLocal = p.alphaVal();
    const void* alphaPtr = p.alphaIsNull ? nullptr : &alphaLocal;

    // Step 2: Save golden copies before NPU mutates yHost
    std::vector<float> xGolden = xHost;
    std::vector<float> yGolden = yHost;

    // Step 3: Execute NPU (wrapper handles nullptrs, dtype packing, device memory)
    aclblasStatus_t ret = aclblasAxpyEx_npu(
        AxpyExArch35Test::handle_, p.n, alphaPtr, static_cast<aclDataType>(p.alphaType), xPtr,
        static_cast<aclDataType>(p.xType), p.incx, yPtr, static_cast<aclDataType>(p.yType), p.incy,
        static_cast<aclDataType>(p.executionType), p.alphaOnDevice);

    // Step 3b: Verify expected error code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;
    if (p.n <= 0)
        return; // n=0 short-circuit, nothing to compare

    // Step 4: Compute golden (in-place on saved copy)
    aclblasStatus_t cpuRet = aclblasAxpyEx_cpu(
        AxpyExArch35Test::handle_, p.n, alphaPtr, static_cast<aclDataType>(p.alphaType), xGolden.data(),
        static_cast<aclDataType>(p.xType), p.incx, yGolden.data(), static_cast<aclDataType>(p.yType), p.incy,
        static_cast<aclDataType>(p.executionType));
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    // Step 5: Precision verification (MIXED_TOLERANCE, threshold per yType)
    VerifyConfig cfg;
    applyMixedTolerance(cfg, static_cast<aclDataType>(p.yType), yGolden.data(), static_cast<size_t>(std::max(0, p.n)));

    int absIncy = std::abs(p.incy);
    EXPECT_TRUE(
        Verifier::verifyVector(
            static_cast<const float*>(yPtr), yGolden.data(), static_cast<size_t>(std::max(0, p.n)), absIncy, cfg,
            p.caseName));
}
