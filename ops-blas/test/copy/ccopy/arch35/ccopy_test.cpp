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
#include <cstdlib>
#include <cstring>
#include <vector>

#include <securec.h>

#include "blas_test.h"
#include "ccopy_golden.h"
#include "ccopy_npu_wrapper.h"
#include "ccopy_param.h"
#include "csv_loader.h"
#include "fill.h"

class CcopyTest : public BlasTest<CcopyParam> {};

TEST_F(CcopyTest, ZeroLengthIgnoresOtherArguments)
{
    aclblasStatus_t ret = aclblasCcopy_npu(nullptr, 0, nullptr, 0, nullptr, 0);
    EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CcopyTest, NullHandle)
{
    aclblasStatus_t ret = aclblasCcopy_npu(nullptr, 5, nullptr, 1, nullptr, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Ccopy, CcopyTest, ::testing::ValuesIn(GetCasesFromCsv<CcopyParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CcopyParam>);

TEST_P(CcopyTest, CsvDriven)
{
    const auto& p = GetParam();
    ASSERT_GE(p.xAlignOffset, 0);
    ASSERT_GE(p.yAlignOffset, 0);
    int64_t xSpan = p.n > 0 ? static_cast<int64_t>(std::abs(p.incx)) * (p.n - 1) + 1 : 1;
    int64_t ySpan = p.n > 0 ? static_cast<int64_t>(std::abs(p.incy)) * (p.n - 1) + 1 : 1;
    int64_t xAlloc = xSpan + std::abs(p.xAlignOffset);
    int64_t yAlloc = ySpan + std::abs(p.yAlignOffset);

    std::vector<aclblasComplex> xHost =
        makeBlasComplexMatrix(static_cast<int>(xAlloc), 1, static_cast<int>(xAlloc), p.x, p.randomSeed);
    std::vector<aclblasComplex> yHost =
        makeBlasComplexMatrix(static_cast<int>(yAlloc), 1, static_cast<int>(yAlloc), p.y, p.randomSeed + 1U);

    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data() + p.xAlignOffset;
    aclblasComplex* yPtr = yHost.empty() ? nullptr : yHost.data() + p.yAlignOffset;
    aclblasStatus_t ret = aclblasCcopy_npu(
        CcopyTest::handle_, p.n, xPtr, p.incx, yPtr, p.incy, static_cast<size_t>(p.xAlignOffset),
        static_cast<size_t>(p.yAlignOffset));

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n <= 0) {
        return;
    }

    std::vector<aclblasComplex> goldenX =
        makeBlasComplexMatrix(static_cast<int>(xAlloc), 1, static_cast<int>(xAlloc), p.x, p.randomSeed);
    std::vector<aclblasComplex> goldenY =
        makeBlasComplexMatrix(static_cast<int>(yAlloc), 1, static_cast<int>(yAlloc), p.y, p.randomSeed + 1U);
    aclblasStatus_t goldenRet = aclblasCcopy_cpu(
        CcopyTest::handle_, p.n, goldenX.data() + p.xAlignOffset, p.incx, goldenY.data() + p.yAlignOffset, p.incy);
    ASSERT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS);

    ASSERT_NE(yPtr, nullptr);
    EXPECT_EQ(std::memcmp(yHost.data(), goldenY.data(), static_cast<size_t>(yAlloc) * sizeof(aclblasComplex)), 0)
        << "[" << p.caseName << "] output, alignment prefix, or stride holes are not bit-exact";
}

TEST_F(CcopyTest, PreservesRawBitsAndStrideHoles)
{
    constexpr int n = 3;
    constexpr int incx = -2;
    constexpr int incy = 3;
    std::vector<aclblasComplex> x(5);
    std::vector<aclblasComplex> y(7);
    std::vector<aclblasComplex> golden(7);

    const uint32_t xBits[10] = {0x7fc12345U, 0xff800000U, 0x3f800000U, 0x80000000U, 0x00000001U,
                                0x7f800000U, 0xbf800000U, 0x7fa54321U, 0x00000000U, 0x40490fdbU};
    const uint32_t sentinelBits[2] = {0xdeadbeefU, 0x12345678U};
    ASSERT_EQ(memcpy_s(x.data(), x.size() * sizeof(aclblasComplex), xBits, sizeof(xBits)), EOK);
    for (auto& value : y) {
        ASSERT_EQ(memcpy_s(&value, sizeof(value), sentinelBits, sizeof(sentinelBits)), EOK);
    }
    golden = y;
    ASSERT_EQ(aclblasCcopy_cpu(CcopyTest::handle_, n, x.data(), incx, golden.data(), incy), ACLBLAS_STATUS_SUCCESS);

    ASSERT_EQ(aclblasCcopy_npu(CcopyTest::handle_, n, x.data(), incx, y.data(), incy), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(std::memcmp(y.data(), golden.data(), y.size() * sizeof(aclblasComplex)), 0);
}
