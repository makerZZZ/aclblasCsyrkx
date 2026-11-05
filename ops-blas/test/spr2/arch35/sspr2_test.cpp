/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include <cmath>
#include <fstream>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "sspr2_param.h"
#include "sspr2_golden.h"
#include "sspr2_npu_wrapper.h"

class Sspr2Arch35Test : public BlasTest<Sspr2Param> { };

TEST(NullHandleTest, NullHandle)
{
    float alphaVal = 1.0f;
    aclblasStatus_t ret = aclblasSspr2(nullptr, ACLBLAS_UPPER, 4, &alphaVal, nullptr, 1, nullptr, 1, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

static std::vector<Sspr2Param> GetAllSspr2Cases()
{
    auto cases = GetCasesFromCsv<Sspr2Param>(ReplaceFileExtension2Csv(__FILE__));
    std::string dir;
    {
        std::string p(__FILE__);
        size_t s = p.rfind('/');
        if (s != std::string::npos) dir = p.substr(0, s + 1);
    }
    std::string pfPath = dir + "sspr2_profiling.csv";
    if (std::ifstream(pfPath).good()) {
        auto pf = GetCasesFromCsv<Sspr2Param>(pfPath);
        cases.insert(cases.end(), std::make_move_iterator(pf.begin()), std::make_move_iterator(pf.end()));
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(
    Sspr2, Sspr2Arch35Test,
    ::testing::ValuesIn(GetAllSspr2Cases()),
    PrintCaseInfoString<Sspr2Param>);

TEST_P(Sspr2Arch35Test, CsvDriven)
{
    const auto& p = GetParam();

    int absIncx = (p.incx == INT_MIN) ? 0 : std::abs(p.incx);
    int absIncy = (p.incy == INT_MIN) ? 0 : std::abs(p.incy);
    size_t xLen = (p.n > 0 && absIncx > 0) ? static_cast<size_t>((p.n - 1) * absIncx + 1) : 0;
    size_t yLen = (p.n > 0 && absIncy > 0) ? static_cast<size_t>((p.n - 1) * absIncy + 1) : 0;
    size_t apLen = (p.n > 0) ? static_cast<size_t>(p.n) * (p.n + 1) / 2 : 0;

    std::vector<float> xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed + 1);

    bool useUpper = (p.uplo == ACLBLAS_UPPER);
    std::vector<float> apHost;
    if (p.ap.pattern == BlasFillMode::P_UPPER || p.ap.pattern == BlasFillMode::P_LOWER) {
        apHost = makeBlasPacked(p.n, p.ap, p.randomSeed);
    } else {
        apHost = makeBlasTriangular(p.n, useUpper, p.ap, p.randomSeed);
    }

    std::vector<float> apOrig = apHost;

    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    const float* yPtr = yHost.empty() ? nullptr : yHost.data();
    float* apPtr = apHost.empty() ? nullptr : apHost.data();
    const float* alphaPtr = p.alphaNull ? nullptr : &p.alpha;

    aclblasStatus_t ret = aclblasSspr2_npu(
        Sspr2Arch35Test::handle_, p.uplo, p.n, alphaPtr, xPtr, p.incx, yPtr, p.incy, apPtr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) return;
    if (p.n == 0) return;

    std::vector<float> golden = apOrig;
    aclblasSspr2_cpu(
        Sspr2Arch35Test::handle_, p.uplo, p.n, &p.alpha,
        xHost.data(), p.incx, yHost.data(), p.incy, golden.data());

    VerifyConfig cfg;
    if (p.alpha == 0.0f && !p.alphaNull) {
        cfg.mode = PrecisionMode::EXACT;
        EXPECT_TRUE(Verifier::verifyVector(apPtr, golden.data(), apLen, 1, cfg, p.caseName));
    } else {
        applyMixedTolerance(cfg, ACL_FLOAT, golden.data(), apLen);
        EXPECT_TRUE(Verifier::verifyVector(apPtr, golden.data(), apLen, 1, cfg, p.caseName));
    }
}
