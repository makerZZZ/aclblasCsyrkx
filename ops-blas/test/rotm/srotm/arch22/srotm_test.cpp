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
#include "fill.h"
#include "srotm_param.h"
#include "srotm_golden.h"
#include "srotm_npu_wrapper.h"

class SrotmArch22Test : public BlasTest<SrotmParam> { };

INSTANTIATE_TEST_SUITE_P(
    Srotm, SrotmArch22Test,
    ::testing::ValuesIn(GetCasesFromCsv<SrotmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SrotmParam>);

TEST_P(SrotmArch22Test, CsvDriven) {
    const auto& p = GetParam();

    auto x = makeBlasStrided(p.n, p.incx, "RANDOM_NORM_2_2", p.randomSeed);
    auto y = makeBlasStrided(p.n, p.incy, "RANDOM_NORM_2_2", p.randomSeed);
    std::vector<float> resultX = x;
    std::vector<float> resultY = y;
    std::vector<float> goldenX = x;
    std::vector<float> goldenY = y;

    std::array<float, 5> sparam = p.sparam;

    aclblasStatus_t ret = aclblasSrotm_npu(
        SrotmArch22Test::handle_,
        p.n, resultX.data(), p.incx, resultY.data(), p.incy, sparam.data());

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasSrotm_cpu(
        SrotmArch22Test::handle_,
        p.n, goldenX.data(), p.incx, goldenY.data(), p.incy, sparam.data());

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = p.mereThreshold;
    cfg.mareMultiplier = p.mareMultiplier;

    EXPECT_TRUE(Verifier::verifyVector(resultX.data(), goldenX.data(),
        static_cast<size_t>(x.size()), 1, cfg, p.caseName));
    EXPECT_TRUE(Verifier::verifyVector(resultY.data(), goldenY.data(),
        static_cast<size_t>(y.size()), 1, cfg, p.caseName));
}
