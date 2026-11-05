/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "srotg_param.h"
#include "srotg_golden.h"
#include "srotg_npu_wrapper.h"

class SrotgArch22Test : public BlasTest<SrotgParam> { };

INSTANTIATE_TEST_SUITE_P(
    Srotg, SrotgArch22Test, ::testing::ValuesIn(GetCasesFromCsv<SrotgParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SrotgParam>);

TEST_P(SrotgArch22Test, CsvDriven)
{
    const auto &p = GetParam();

    std::vector<float> resultA(1, p.a);
    std::vector<float> resultB(1, p.b);
    std::vector<float> resultC(1, 0.0f);
    std::vector<float> resultS(1, 0.0f);
    std::vector<float> hostResultA = resultA;
    std::vector<float> hostResultB = resultB;
    std::vector<float> hostResultC = resultC;
    std::vector<float> hostResultS = resultS;

    std::vector<float> goldenA = resultA;
    std::vector<float> goldenB = resultB;
    std::vector<float> goldenC = resultC;
    std::vector<float> goldenS = resultS;

    aclblasStatus_t ret = aclblasSrotg_npu(
        SrotgArch22Test::handle_, resultA.data(), resultB.data(), resultC.data(), resultS.data());
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasStatus_t goldenRet = aclblasSrotg_cpu(
        SrotgArch22Test::handle_, goldenA.data(), goldenB.data(), goldenC.data(), goldenS.data());
    ASSERT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS);

    aclblasStatus_t hostRet = aclblasSrotg(
        SrotgArch22Test::handle_, hostResultA.data(), hostResultB.data(), hostResultC.data(), hostResultS.data());
    ASSERT_EQ(hostRet, ACLBLAS_STATUS_SUCCESS);

    VerifyConfig cfg;
    cfg.mode = p.verifyMode;
    cfg.mereThreshold = p.mereThreshold;
    cfg.mareMultiplier = p.mareMultiplier;
    EXPECT_TRUE(Verifier::verifyVector(
        resultA.data(), goldenA.data(), goldenA.size(), 1, cfg, p.caseName + "_r"));
    EXPECT_TRUE(Verifier::verifyVector(
        resultB.data(), goldenB.data(), goldenB.size(), 1, cfg, p.caseName + "_z"));
    EXPECT_TRUE(Verifier::verifyVector(
        resultC.data(), goldenC.data(), goldenC.size(), 1, cfg, p.caseName + "_c"));
    EXPECT_TRUE(Verifier::verifyVector(
        resultS.data(), goldenS.data(), goldenS.size(), 1, cfg, p.caseName + "_s"));
    EXPECT_TRUE(Verifier::verifyVector(
        hostResultA.data(), goldenA.data(), goldenA.size(), 1, cfg, p.caseName + "_host_r"));
    EXPECT_TRUE(Verifier::verifyVector(
        hostResultB.data(), goldenB.data(), goldenB.size(), 1, cfg, p.caseName + "_host_z"));
    EXPECT_TRUE(Verifier::verifyVector(
        hostResultC.data(), goldenC.data(), goldenC.size(), 1, cfg, p.caseName + "_host_c"));
    EXPECT_TRUE(Verifier::verifyVector(
        hostResultS.data(), goldenS.data(), goldenS.size(), 1, cfg, p.caseName + "_host_s"));
}
