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
#include "sspmv_param.h"
#include "sspmv_golden.h"
#include "sspmv_npu_wrapper.h"

class SspmvArch35Test : public BlasTest<SspmvParam> {};

TEST_F(SspmvArch35Test, NullHandle)
{
    float alpha = 1.0f, beta = 0.0f;
    std::vector<float> ap(55, 1.0f), x(10, 1.0f), y(10, 0.0f);
    aclblasStatus_t ret =
        aclblasSspmv_npu(nullptr, ACLBLAS_LOWER, 10, &alpha, ap.data(), x.data(), 1, &beta, y.data(), 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    Sspmv, SspmvArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SspmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SspmvParam>);

TEST_P(SspmvArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    auto apHost = makeBlasPacked(p.n, p.ap, p.randomSeed);
    auto xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed + 1);
    auto yHost = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed + 2);

    const float* alphaPtr = (p.alphaFill.method == BlasFillMode::M_NULLPTR) ? nullptr : &p.alpha;
    const float* betaPtr = (p.betaFill.method == BlasFillMode::M_NULLPTR) ? nullptr : &p.beta;

    std::vector<float> yNpu = yHost;
    aclblasStatus_t ret = aclblasSspmv_npu(
        SspmvArch35Test::handle_, p.uplo, p.n, alphaPtr, apHost.data(), xHost.data(), p.incx, betaPtr,
        yNpu.data(), p.incy);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    std::vector<float> yCpu = yHost;
    aclblasSspmv_cpu(
        SspmvArch35Test::handle_, p.uplo, p.n, alphaPtr, apHost.data(), xHost.data(), p.incx, betaPtr,
        yCpu.data(), p.incy);

    if (p.n == 0) {
        return;
    }

    const int absIncy = std::abs(p.incy);
    std::vector<float> npuLogical(p.n), cpuLogical(p.n);
    for (int i = 0; i < p.n; i++) {
        int idx = (p.incy > 0) ? (i * p.incy) : ((p.n - 1 - i) * absIncy);
        npuLogical[i] = yNpu[idx];
        cpuLogical[i] = yCpu[idx];
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, cpuLogical.data(), static_cast<size_t>(p.n));
    EXPECT_TRUE(
        Verifier::verifyVector(npuLogical.data(), cpuLogical.data(), static_cast<size_t>(p.n), 1, cfg, p.caseName));
}
