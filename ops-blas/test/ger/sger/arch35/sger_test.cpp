/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "sger_param.h"
#include "sger_golden.h"
#include "sger_npu_wrapper.h"

class SgerArch35Test : public BlasTest<SgerParam> { };

TEST_F(SgerArch35Test, NullHandle) {
    aclblasStatus_t ret = aclblasSger_npu(nullptr, 4, 4, nullptr, nullptr, 1, nullptr, 1, nullptr, 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Sger, SgerArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SgerParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SgerParam>);

TEST_P(SgerArch35Test, CsvDriven) {
    const auto& p = GetParam();

    // Generate host data for all array parameters
    const int64_t absIncX = (p.incx >= 0) ? p.incx : -p.incx;
    const int64_t absIncY = (p.incy >= 0) ? p.incy : -p.incy;
    std::vector<float> xHost = makeBlasArray(p.m * absIncX, p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasArray(p.n * absIncY, p.y, p.randomSeed);
    std::vector<float> aHost = makeBlasArray(p.lda * p.n, p.A, p.randomSeed);

    // Set up alpha (scalar pointer)
    float alphaHost = p.alphaValue;
    const float* alphaPtr = (p.alpha.method == BlasFillMode::M_NULLPTR) ? nullptr : &alphaHost;

    // Map array pointers: empty vector means nullptr for error-path testing
    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float* yPtr = yHost.empty() ? nullptr : yHost.data();
    float* aPtr = aHost.empty() ? nullptr : aHost.data();

    // Step 1: Save original A for golden computation (before kernel modifies aHost)
    std::vector<float> golden = aHost;

    // Step 2: Execute on NPU
    aclblasStatus_t ret = aclblasSger_npu(
        SgerArch35Test::handle_,
        p.m, p.n, alphaPtr, xPtr, p.incx, yPtr, p.incy, aPtr, p.lda);

    // Step 3: Check return code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) return;

    // Step 4: Compute golden on CPU (golden = original_A + alpha * x * y^T)
    aclblasSger_cpu(
        SgerArch35Test::handle_,
        p.m, p.n, alphaPtr, xPtr, p.incx, yPtr, p.incy, golden.data(), p.lda);

    VerifyConfig cfg;
    if (p.alphaValue == 0.0f) {
        cfg.mode = PrecisionMode::EXACT;
    } else {
        applyMixedTolerance(cfg, ACL_FLOAT, golden.data(), aHost.size());
    }
    EXPECT_TRUE(Verifier::verifyVector(aPtr, golden.data(), aHost.size(), 1, cfg, p.caseName));
}
