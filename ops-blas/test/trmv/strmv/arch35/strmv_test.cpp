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
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "strmv_param.h"
#include "strmv_golden.h"
#include "strmv_npu_wrapper.h"

class StrmvArch35Test : public BlasTest<StrmvParam> { };

INSTANTIATE_TEST_SUITE_P(
    Strmv, StrmvArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<StrmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StrmvParam>);

TEST_P(StrmvArch35Test, CsvDriven) {
    const auto& p = GetParam();

    bool isUpper = (p.uplo == ACLBLAS_UPPER);
    std::string matFill = isUpper ? "RANDOM_UPPER" : "RANDOM_LOWER";
    auto a = makeBlasMatrix(p.n, p.n, p.lda, matFill, p.randomSeed);
    auto x = makeBlasStrided(p.n, p.incx, "RANDOM", p.randomSeed + 1);
    std::vector<float> golden = x;

    aclblasStatus_t ret = aclblasStrmv_npu(
        StrmvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n,
        a.data(), p.lda, x.data(), p.incx);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasStrmv_cpu(
        StrmvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n,
        a.data(), p.lda, golden.data(), p.incx);

    const int absIncx = std::abs(p.incx);
    VerifyConfig cfg;
    const float* outPtr  = (p.incx < 0 && p.n > 0) ? x.data() + (p.n - 1) * absIncx : x.data();
    const float* goldPtr = (p.incx < 0 && p.n > 0) ? golden.data() + (p.n - 1) * absIncx : golden.data();
    applyMixedTolerance(cfg, ACL_FLOAT, goldPtr, static_cast<size_t>(p.n));

    EXPECT_TRUE(Verifier::verifyVector(outPtr, goldPtr,
        static_cast<size_t>(p.n), (p.incx < 0) ? -absIncx : absIncx, cfg, p.caseName));
}
