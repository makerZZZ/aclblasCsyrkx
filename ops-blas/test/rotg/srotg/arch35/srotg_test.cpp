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

#include "fill.h"
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "srotg_param.h"
#include "srotg_golden.h"
#include "srotg_npu_wrapper.h"

class SrotgArch35Test : public BlasTest<SrotgParam> {};

static void VerifySrotgResult(float result, float golden, const std::string& caseName)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, golden);
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, caseName));
}

// ── 参数校验测试（单独 TEST_F，不依赖 CSV）──

TEST_F(SrotgArch35Test, NullHandle)
{
    aclblasStatus_t ret = aclblasSrotg_npu(nullptr, nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SrotgArch35Test, NullA)
{
    float b = 2.0f;
    float c = 0.0f;
    float s = 0.0f;
    aclblasStatus_t ret = aclblasSrotg_npu(SrotgArch35Test::handle_, nullptr, &b, &c, &s);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(SrotgArch35Test, NullB)
{
    float a = 1.0f;
    float c = 0.0f;
    float s = 0.0f;
    aclblasStatus_t ret = aclblasSrotg_npu(SrotgArch35Test::handle_, &a, nullptr, &c, &s);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(SrotgArch35Test, NullC)
{
    float a = 1.0f;
    float b = 2.0f;
    float s = 0.0f;
    aclblasStatus_t ret = aclblasSrotg_npu(SrotgArch35Test::handle_, &a, &b, nullptr, &s);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(SrotgArch35Test, NullS)
{
    float a = 1.0f;
    float b = 2.0f;
    float c = 0.0f;
    aclblasStatus_t ret = aclblasSrotg_npu(SrotgArch35Test::handle_, &a, &b, &c, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// 混合 Host/Device 指针必须被拒绝（a 为 device，b/c/s 为 host）
TEST_F(SrotgArch35Test, MixedHostDevicePointers)
{
    float bHost = 2.0f;
    float cHost = 0.0f;
    float sHost = 0.0f;

    void* aDev = nullptr;
    ASSERT_EQ(aclrtMalloc(&aDev, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);

    aclblasStatus_t ret = aclblasSrotg(
        SrotgArch35Test::handle_, static_cast<float*>(aDev), &bHost, &cHost, &sHost);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);

    if (aDev) aclrtFree(aDev);
}

// ── CSV 驱动功能测试 ──

INSTANTIATE_TEST_SUITE_P(
    Srotg, SrotgArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SrotgParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SrotgParam>);

// CSV-driven: all-device path (via NPU wrapper)
TEST_P(SrotgArch35Test, CsvDrivenDevice)
{
    const auto& p = GetParam();

    // Step 1: 准备 host 数据
    float aHost = p.a;
    float bHost = p.b;
    float cHost = 0.0f;
    float sHost = 0.0f;

    // Step 2: NPU 执行（wrapper 内部走 device 指针路径）
    aclblasStatus_t ret = aclblasSrotg_npu(
        SrotgArch35Test::handle_, &aHost, &bHost, &cHost, &sHost);

    // Step 3: 检查返回码
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    // Step 4: 计算 CPU golden
    float goldenA = p.a;
    float goldenB = p.b;
    float goldenC = 0.0f;
    float goldenS = 0.0f;
    aclblasSrotg_cpu(SrotgArch35Test::handle_, &goldenA, &goldenB, &goldenC, &goldenS);

    // Step 5: 逐一验证 4 个 FP32 输出标量
    VerifySrotgResult(aHost, goldenA, p.caseName + "_a");
    VerifySrotgResult(bHost, goldenB, p.caseName + "_b");
    VerifySrotgResult(cHost, goldenC, p.caseName + "_c");
    VerifySrotgResult(sHost, goldenS, p.caseName + "_s");
}

// CSV-driven: all-host path (aclrtMallocHost pinned memory, direct call)
TEST_P(SrotgArch35Test, CsvDrivenHost)
{
    const auto& p = GetParam();

    float *a = nullptr;
    float *b = nullptr;
    float *c = nullptr;
    float *s = nullptr;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&a), sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&b), sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&c), sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&s), sizeof(float)), ACL_SUCCESS);

    *a = p.a;
    *b = p.b;

    // Golden
    float goldenA = p.a;
    float goldenB = p.b;
    float goldenC = 0.0f;
    float goldenS = 0.0f;
    EXPECT_EQ(aclblasSrotg_cpu(SrotgArch35Test::handle_, &goldenA, &goldenB, &goldenC, &goldenS),
              ACLBLAS_STATUS_SUCCESS);

    aclblasStatus_t ret = aclblasSrotg(SrotgArch35Test::handle_, a, b, c, s);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        VerifySrotgResult(*a, goldenA, p.caseName + "_a");
        VerifySrotgResult(*b, goldenB, p.caseName + "_b");
        VerifySrotgResult(*c, goldenC, p.caseName + "_c");
        VerifySrotgResult(*s, goldenS, p.caseName + "_s");
    }

    if (a) aclrtFreeHost(a);
    if (b) aclrtFreeHost(b);
    if (c) aclrtFreeHost(c);
    if (s) aclrtFreeHost(s);
}
