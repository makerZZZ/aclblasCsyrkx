/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdlib>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "isamin_param.h"
#include "isamin_golden.h"
#include "isamin_npu_wrapper.h"

class IsaminArch35Test : public BlasTest<IsaminParam> {};

TEST_F(IsaminArch35Test, NullHandle)
{
    int result = 0;
    aclblasStatus_t ret = aclblasIsamin_npu(nullptr, 5, nullptr, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}

INSTANTIATE_TEST_SUITE_P(
    Isamin, IsaminArch35Test, ::testing::ValuesIn(GetCasesFromCsv<IsaminParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<IsaminParam>);

TEST_P(IsaminArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    std::cout << "\n=== RUN  [" << p.caseName << "] n=" << p.n << " incx=" << p.incx
              << " x_method=" << static_cast<int>(p.x.method) << " desc=" << p.description << " ===" << std::endl;

    const int absIncx = std::abs(p.incx);
    const int64_t xLen =
        (p.n > 0 && absIncx > 0) ? static_cast<int64_t>(1) + static_cast<int64_t>(p.n - 1) * absIncx : 0;
    std::vector<float> xHost = makeBlasArray(xLen, p.x, p.randomSeed);

    const float* xPtr = xHost.empty() ? nullptr : xHost.data();

    int result = 0;
    aclblasStatus_t ret = aclblasIsamin_npu(IsaminArch35Test::handle_, p.n, xPtr, p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
        << "[" << p.caseName << "] unexpected return code: got " << static_cast<int>(ret) << " expect "
        << static_cast<int>(p.expectResult);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    int golden = 0;
    aclblasIsamin_cpu(IsaminArch35Test::handle_, p.n, xHost.data(), p.incx, &golden);

    EXPECT_EQ(result, golden) << "[" << p.caseName << "] index mismatch: NPU=" << result << " golden=" << golden;
}
