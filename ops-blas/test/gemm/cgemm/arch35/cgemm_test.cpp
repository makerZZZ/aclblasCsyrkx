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
#include <cstdint>
#include <vector>

#include "blas_test.h"
#include "cgemm_golden.h"
#include "cgemm_npu_wrapper.h"
#include "cgemm_param.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"

namespace {

std::vector<aclblasComplex> MakeCgemmMatrix(
    int rows, int columns, int leadingDimension, const BlasFillMode& fill, uint32_t seed)
{
    auto real = makeBlasMatrix(rows, columns, leadingDimension, fill, seed);
    auto imag = makeBlasMatrix(rows, columns, leadingDimension, fill, seed + 101U);
    if (real.empty() || imag.empty()) {
        return {};
    }
    std::vector<aclblasComplex> result(real.size());
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = aclblasComplex{real[index], imag[index]};
    }
    return result;
}

} // namespace

class CgemmTest : public BlasTest<CgemmParam> {};

TEST_F(CgemmTest, NullHandle)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{0.0f, 0.0f};
    const aclblasStatus_t status = aclblasCgemm_npu(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8, &alpha, nullptr, 8, nullptr, 8, &beta, nullptr, 8);
    EXPECT_EQ(status, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Cgemm, CgemmTest, ::testing::ValuesIn(GetCasesFromCsv<CgemmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgemmParam>);

TEST_P(CgemmTest, CsvDriven)
{
    const auto& parameter = GetParam();
    const int physicalRowsA = parameter.transA == ACLBLAS_OP_N ? parameter.m : parameter.k;
    const int physicalColumnsA = parameter.transA == ACLBLAS_OP_N ? parameter.k : parameter.m;
    const int physicalRowsB = parameter.transB == ACLBLAS_OP_N ? parameter.k : parameter.n;
    const int physicalColumnsB = parameter.transB == ACLBLAS_OP_N ? parameter.n : parameter.k;

    auto a = MakeCgemmMatrix(physicalRowsA, physicalColumnsA, parameter.lda, parameter.aFill, parameter.randomSeed);
    auto b =
        MakeCgemmMatrix(physicalRowsB, physicalColumnsB, parameter.ldb, parameter.bFill, parameter.randomSeed + 1U);
    auto c = MakeCgemmMatrix(parameter.m, parameter.n, parameter.ldc, parameter.cFill, parameter.randomSeed + 2U);
    std::vector<aclblasComplex> golden = c;

    const aclblasComplex alpha{parameter.alphaReal, parameter.alphaImag};
    const aclblasComplex beta{parameter.betaReal, parameter.betaImag};
    const aclblasComplex* alphaPointer = parameter.alphaNull ? nullptr : &alpha;
    const aclblasComplex* betaPointer = parameter.betaNull ? nullptr : &beta;
    const aclblasComplex* aPointer = parameter.aNull || a.empty() ? nullptr : a.data();
    const aclblasComplex* bPointer = parameter.bNull || b.empty() ? nullptr : b.data();
    aclblasComplex* cPointer = parameter.cNull || c.empty() ? nullptr : c.data();
    aclblasHandle_t handle = CgemmTest::handle_;
    if (parameter.description.find("handle_null") != std::string::npos) {
        handle = nullptr;
    }

    const aclblasStatus_t status = aclblasCgemm_npu(
        handle, parameter.transA, parameter.transB, parameter.m, parameter.n, parameter.k, alphaPointer, aPointer,
        parameter.lda, bPointer, parameter.ldb, betaPointer, cPointer, parameter.ldc);
    EXPECT_EQ(status, parameter.expectResult);
    if (parameter.expectResult != ACLBLAS_STATUS_SUCCESS || parameter.m <= 0 || parameter.n <= 0 ||
        cPointer == nullptr) {
        return;
    }

    const aclblasStatus_t goldenStatus = aclblasCgemmCpu(
        CgemmTest::handle_, parameter.transA, parameter.transB, parameter.m, parameter.n, parameter.k, &alpha, a.data(),
        parameter.lda, b.data(), parameter.ldb, &beta, golden.data(), parameter.ldc);
    ASSERT_EQ(goldenStatus, ACLBLAS_STATUS_SUCCESS);

    const size_t complexCount = static_cast<size_t>(parameter.ldc) * parameter.n;
    const size_t componentCount = complexCount * 2;
    VerifyConfig configuration;
    if (parameter.alphaReal == 0.0f && parameter.alphaImag == 0.0f) {
        configuration.mode = PrecisionMode::EXACT;
    } else {
        applyMixedTolerance(configuration, ACL_FLOAT, reinterpret_cast<const float*>(golden.data()), componentCount);
    }
    EXPECT_TRUE(Verifier::verifyVector(
        reinterpret_cast<const float*>(c.data()), reinterpret_cast<const float*>(golden.data()), componentCount, 1,
        configuration, parameter.caseName));
}
