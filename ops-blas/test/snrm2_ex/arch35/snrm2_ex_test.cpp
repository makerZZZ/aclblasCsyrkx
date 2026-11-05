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
#include "dtype_utils.h"
#include "snrm2_ex_param.h"
#include "snrm2_ex_golden.h"
#include "snrm2_ex_npu_wrapper.h"

class Snrm2ExArch35Test : public BlasTest<Snrm2ExParam> {};

TEST_F(Snrm2ExArch35Test, NullHandle)
{
    float dummy = 0.0f;
    aclblasStatus_t ret = aclblasSnrm2Ex_npu(nullptr, ACL_FLOAT, &dummy, 4, 1, &dummy);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}

TEST_F(Snrm2ExArch35Test, NullResult)
{
    float dummy = 0.0f;
    aclblasStatus_t ret = aclblasSnrm2Ex_npu(Snrm2ExArch35Test::handle_, ACL_FLOAT, &dummy, 4, 1, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(
    Snrm2Ex, Snrm2ExArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<Snrm2ExParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Snrm2ExParam>);

TEST_P(Snrm2ExArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    std::cout << "\n=== RUN  [" << p.caseName << "] xtype=" << static_cast<int>(p.xtype)
              << " n=" << p.n << " incx=" << p.incx
              << " x_method=" << static_cast<int>(p.x.method) << " desc=" << p.description
              << " ===" << std::endl;

    // 步骤 1：构造输入数据
    std::vector<float> xFloat = makeBlasStrided(
        static_cast<int>(p.n), static_cast<int>(p.incx), p.x, p.randomSeed);

    std::vector<uint8_t> xBytes;
    const void* xPtr = nullptr;
    if (!xFloat.empty()) {
        xBytes = quantizeToBytes(xFloat, p.xtype);
        xPtr = xBytes.data();
    }

    // 步骤 2：调用 NPU wrapper
    float result = 0.0f;
    aclblasStatus_t ret =
        aclblasSnrm2Ex_npu(Snrm2ExArch35Test::handle_, p.xtype, xPtr, p.n, p.incx, &result);

    // 步骤 3：校验返回值
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
        << "[" << p.caseName << "] unexpected return code: got " << static_cast<int>(ret)
        << " expect " << static_cast<int>(p.expectResult);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    // n=0：结果必须为 0.0f
    if (p.n == 0) {
        EXPECT_FLOAT_EQ(result, 0.0f) << "[" << p.caseName << "] n=0 but result != 0";
        return;
    }

    // 步骤 4：计算 golden
    float golden = snrm2_ex_cpu(p.xtype, xPtr, p.n, p.incx);

    // 步骤 5：精度校验
    VerifyConfig cfg;
    aclDataType scalarDtype = (p.xtype == ACL_FLOAT16) ? ACL_FLOAT16 : ACL_FLOAT;
    applyMixedTolerance(cfg, scalarDtype, golden);
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, p.caseName));
}
