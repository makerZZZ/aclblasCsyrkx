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
#include <cstring>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "rotex_param.h"
#include "rotex_golden.h"
#include "rotex_npu_wrapper.h"

// ── Helpers ───────────────────────────────────────────────────────────────────
static void RunVerifyInline(const RotExParam& p, const float* xOut, const float* yOut,
                            const float* gx, const float* gy,
                            aclDataType xT, aclDataType yT, int absX, int absY) {
    VerifyConfig cfgX;
    applyMixedTolerance(cfgX, xT, gx, static_cast<size_t>(std::max(0, p.n)));
    VerifyConfig cfgY;
    applyMixedTolerance(cfgY, yT, gy, static_cast<size_t>(std::max(0, p.n)));
    int xs = (p.incx != 0) ? absX : 1, ys = (p.incy != 0) ? absY : 1;
    EXPECT_TRUE(Verifier::verifyVector(xOut, gx, static_cast<size_t>(std::max(0, p.n)), xs, cfgX, p.caseName + "_x"));
    EXPECT_TRUE(Verifier::verifyVector(yOut, gy, static_cast<size_t>(std::max(0, p.n)), ys, cfgY, p.caseName + "_y"));
}

// ── Test fixture ─────────────────────────────────────────────────────────────
class RotExArch35Test : public BlasTest<RotExParam> { };

// ── TEST_F: null handle (not in CSV) ─────────────────────────────────────────
TEST_F(RotExArch35Test, NullHandle) {
    float xData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float yData[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float cVal = 0.5f;
    float sVal = 0.5f;
    aclblasStatus_t ret = aclblasRotEx_npu(
        nullptr, 5,
        xData, ACL_FLOAT, 1,
        yData, ACL_FLOAT, 1,
        &cVal, &sVal, ACL_FLOAT, ACL_FLOAT);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// ── CSV parameterised test suite ─────────────────────────────────────────────
INSTANTIATE_TEST_SUITE_P(
    RotEx, RotExArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<RotExParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<RotExParam>);

// ── TEST_P: 5-step CSV-driven flow ───────────────────────────────────────────
TEST_P(RotExArch35Test, CsvDriven) {
    const auto& p = GetParam();
    auto xType  = static_cast<aclDataType>(p.xType);
    auto yType  = static_cast<aclDataType>(p.yType);
    auto csType = static_cast<aclDataType>(p.csType);
    auto execType = static_cast<aclDataType>(p.executionType);
    int xEF = rotExElemFloats(p.xType), yEF = rotExElemFloats(p.yType);
    int absIncX = std::abs(p.incx), absIncY = std::abs(p.incy);

    // Step 1: host data
    std::vector<float> xHost, yHost;
    if (p.xFill.method != BlasFillMode::M_NULLPTR && p.n > 0) {
        xHost = (xEF == 1) ? makeBlasStrided(p.n, p.incx, p.xFill, p.randomSeed)
                : makeBlasArray(static_cast<int64_t>((p.n - 1) * absIncX * xEF + xEF), p.xFill, p.randomSeed);
    }
    if (p.yFill.method != BlasFillMode::M_NULLPTR && p.n > 0) {
        yHost = (yEF == 1) ? makeBlasStrided(p.n, p.incy, p.yFill, p.randomSeed)
                : makeBlasArray(static_cast<int64_t>((p.n - 1) * absIncY * yEF + yEF), p.yFill, p.randomSeed);
    }
    float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float* yPtr = yHost.empty() ? nullptr : yHost.data();

    // c/s scalars
    float cF = static_cast<float>(p.cValue), sF = static_cast<float>(p.sValue);
    uint16_t cBf16 = rotExFloatToBf16(cF), sBf16 = rotExFloatToBf16(sF);
    uint16_t cFp16 = rotExFloatToFp16(cF), sFp16 = rotExFloatToFp16(sF);
    const void* cPtr = (p.cValue <= -99999.0) ? nullptr :
        (csType == ACL_BF16) ? static_cast<const void*>(&cBf16) :
        (csType == ACL_FLOAT16) ? static_cast<const void*>(&cFp16) : static_cast<const void*>(&cF);
    const void* sPtr = (p.sValue <= -99999.0) ? nullptr :
        (csType == ACL_BF16) ? static_cast<const void*>(&sBf16) :
        (csType == ACL_FLOAT16) ? static_cast<const void*>(&sFp16) : static_cast<const void*>(&sF);

    // Step 2: golden copy
    std::vector<float> goldenX = xHost, goldenY = yHost;

    // Step 3: NPU
    aclblasStatus_t ret = aclblasRotEx_npu(RotExArch35Test::handle_, p.n, xPtr, xType, p.incx,
                                           yPtr, yType, p.incy, cPtr, sPtr, csType, execType);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) return;

    // Step 4: golden
    float* gxPtr = goldenX.empty() ? nullptr : goldenX.data();
    float* gyPtr = goldenY.empty() ? nullptr : goldenY.data();
    aclblasStatus_t cpuRet = aclblasRotEx_cpu(RotExArch35Test::handle_, p.n, gxPtr, xType, p.incx,
                                              gyPtr, yType, p.incy, cPtr, sPtr, csType, execType);
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    (void)cpuRet;

    // Step 5: verify
    RunVerifyInline(p, xPtr, yPtr, goldenX.data(), goldenY.data(), xType, yType, absIncX, absIncY);
}
