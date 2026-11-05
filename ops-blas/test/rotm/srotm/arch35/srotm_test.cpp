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
#include "srotm_param.h"
#include "srotm_golden.h"
#include "srotm_npu_wrapper.h"

class SrotmArch35Test : public BlasTest<SrotmParam> { };

INSTANTIATE_TEST_SUITE_P(
    Srotm, SrotmArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SrotmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SrotmParam>);

TEST_P(SrotmArch35Test, CsvDriven) {
    const auto& p = GetParam();

    std::vector<float> xHost = makeBlasStrided(static_cast<int>(p.n), static_cast<int>(p.incx),
                                                "RANDOM_NORM_2_2", p.randomSeed);
    std::vector<float> yHost = makeBlasStrided(static_cast<int>(p.n), static_cast<int>(p.incy),
                                                "RANDOM_NORM_2_2", p.randomSeed + 1);

    float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float* yPtr = yHost.empty() ? nullptr : yHost.data();

    std::vector<float> xGolden = xHost;
    std::vector<float> yGolden = yHost;

    aclblasStatus_t ret = aclblasSrotm_npu(SrotmArch35Test::handle_,
        static_cast<int>(p.n), xPtr, static_cast<int>(p.incx),
        yPtr, static_cast<int>(p.incy), p.sparam.data());
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) return;

    aclblasSrotm_cpu(SrotmArch35Test::handle_,
        p.n, xGolden.data(), p.incx,
        yGolden.data(), p.incy, p.sparam.data());

    const size_t count = std::min(xHost.size(), xGolden.size());
    VerifyConfig xCfg;
    applyMixedTolerance(xCfg, ACL_FLOAT, xGolden.data(), static_cast<size_t>(p.n));
    EXPECT_TRUE(Verifier::verifyVector(xPtr, xGolden.data(), count, 1, xCfg, p.caseName + "_x"));

    const size_t yCount = std::min(yHost.size(), yGolden.size());
    VerifyConfig yCfg;
    applyMixedTolerance(yCfg, ACL_FLOAT, yGolden.data(), static_cast<size_t>(p.n));
    EXPECT_TRUE(Verifier::verifyVector(yPtr, yGolden.data(), yCount, 1, yCfg, p.caseName + "_y"));
}

TEST_F(SrotmArch35Test, NullHandle) {
    std::array<float, 5> sparam = {-1.0f, 0.5f, 0.8f, -0.3f, 0.6f};
    aclblasStatus_t ret = aclblasSrotm_npu(nullptr, 10, nullptr, 1, nullptr, 1, sparam.data());
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_NOT_INITIALIZED));
}
