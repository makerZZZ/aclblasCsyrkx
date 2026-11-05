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
#include "sgbmv_param.h"
#include "sgbmv_golden.h"
#include "sgbmv_npu_wrapper.h"

class GbmvArch35Test : public BlasTest<GbmvParam> { };

INSTANTIATE_TEST_SUITE_P(
    Gbmv, GbmvArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<GbmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GbmvParam>);

TEST_P(GbmvArch35Test, CsvDriven) {
    const auto& p = GetParam();

    const bool isTransN = (p.trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? p.n : p.m;
    const int yCount = isTransN ? p.m : p.n;

    auto a = makeBlasBanded(p.m, p.n, p.lda, p.a, p.randomSeed);
    auto x = makeBlasStrided(xCount, p.incx, p.x, p.randomSeed);
    auto y = makeBlasStrided(yCount, p.incy, p.y, p.randomSeed);
    std::vector<float> result = y;
    std::vector<float> golden = y;

    aclblasStatus_t ret = aclblasSgbmv_npu(
        GbmvArch35Test::handle_, p.trans, p.m, p.n, p.kl, p.ku,
        &p.alpha, a.data(), p.lda,
        x.data(), p.incx,
        &p.beta, result.data(), p.incy);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasSgbmv_cpu(
        GbmvArch35Test::handle_, p.trans, p.m, p.n, p.kl, p.ku,
        &p.alpha, a.data(), p.lda,
        x.data(), p.incx,
        &p.beta, golden.data(), p.incy);

    const float* outPtr  = (p.incy < 0 && yCount > 0) ? result.data() + (yCount - 1) * (-p.incy) : result.data();
    const float* goldPtr = (p.incy < 0 && yCount > 0) ? golden.data() + (yCount - 1) * (-p.incy) : golden.data();

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldPtr, static_cast<size_t>(yCount));
    EXPECT_TRUE(Verifier::verifyVector(outPtr, goldPtr,
        static_cast<size_t>(yCount), p.incy, cfg, p.caseName));
}
