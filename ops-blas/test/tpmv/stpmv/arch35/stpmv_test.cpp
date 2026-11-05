/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "stpmv_param.h"
#include "stpmv_golden.h"
#include "stpmv_npu_wrapper.h"

class StpmvArch35Test : public BlasTest<StpmvParam> {};

// Separate test for null handle (not in CSV)
TEST_F(StpmvArch35Test, NullHandle)
{
    aclblasStatus_t ret =
        aclblasStpmv_npu(nullptr, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, nullptr, nullptr, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Stpmv, StpmvArch35Test, ::testing::ValuesIn(GetCasesFromCsv<StpmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StpmvParam>);

TEST_P(StpmvArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // Step 1: Generate input data
    // For error-path tests, skip data allocation (use nullptr) to avoid huge memory
    // allocation. The API validates parameters before accessing data, so nullptr is safe.
    const bool isErrorPath = (p.expectResult != ACLBLAS_STATUS_SUCCESS);
    const int safeN = (!isErrorPath && p.n > 0) ? p.n : 0;
    const bool isUpper = (p.uplo == ACLBLAS_UPPER);
    std::vector<float> apHost = makeBlasTriangular(safeN, isUpper, p.ap, p.randomSeed);
    std::vector<float> xHost = makeBlasStrided(safeN, p.incx, p.x, p.randomSeed + 1);

    const float* apPtr = apHost.empty() ? nullptr : apHost.data();
    float* xPtr = xHost.empty() ? nullptr : xHost.data();

    // For null handle test, use nullptr instead of the fixture's handle
    aclblasHandle_t testHandle =
        (p.expectResult == ACLBLAS_STATUS_HANDLE_IS_NULLPTR) ? nullptr : StpmvArch35Test::handle_;

    // Step 2: Compute CPU golden BEFORE NPU call (NPU overwrites x in-place)
    std::vector<float> xGolden;
    if (p.expectResult == ACLBLAS_STATUS_SUCCESS && p.n > 0) {
        xGolden = xHost; // copy x for golden computation
        aclblasStpmv_cpu(StpmvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n, apHost.data(), xGolden.data(), p.incx);
    }

    // Step 3: Execute on NPU (in-place: xPtr is overwritten with result)
    aclblasStatus_t ret = aclblasStpmv_npu(testHandle, p.uplo, p.trans, p.diag, p.n, apPtr, xPtr, p.incx);

    // Step 4: Check return code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    // Step 5: Verify results
    // Both NPU and golden write results in-place with incx stride.
    // Extract logical elements from both buffers using BLAS stride rules:
    //   incx >= 0: logical element i at physical position i * |incx|
    //   incx <  0: logical element i at physical position (n-1-i) * |incx|
    VerifyConfig cfg;
    if (p.mereThreshold > 0.0) {
        cfg.mode = PrecisionMode::MERE_MARE;
        cfg.mereThreshold = p.mereThreshold;
        cfg.mareMultiplier = p.mareMultiplier;
    } else {
        cfg.mode = PrecisionMode::EXACT;
    }
    {
        const int absIncx = std::abs(p.incx);
        std::vector<float> xResult(static_cast<size_t>(p.n));
        std::vector<float> xGoldenExtracted(static_cast<size_t>(p.n));
        for (int i = 0; i < p.n; ++i) {
            int offset = (p.incx >= 0) ? (i * absIncx) : ((p.n - 1 - i) * absIncx);
            xResult[i] = xPtr[offset];
            xGoldenExtracted[i] = xGolden[offset];
        }
        EXPECT_TRUE(
            Verifier::verifyVector(
                xResult.data(), xGoldenExtracted.data(), static_cast<size_t>(p.n), 1, cfg, p.caseName));
    }
}
