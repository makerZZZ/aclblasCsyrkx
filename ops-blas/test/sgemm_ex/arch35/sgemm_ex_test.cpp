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
#include "sgemm_ex_param.h"
#include "sgemm_ex_golden.h"
#include "sgemm_ex_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Test fixture
// ═══════════════════════════════════════════════════════════════════════════════

class SgemmExArch35Test : public BlasTest<SgemmExParam> {
};

// ── TEST_F: null handle (L0_15, not CSV-driven) ──
TEST_F(SgemmExArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    aclblasStatus_t ret = aclblasSgemmEx_npu(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8,
        &alpha, nullptr, 8, nullptr, 8,
        &beta, nullptr, 8, ACLBLAS_GEMM_DEFAULT);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    SgemmEx, SgemmExArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SgemmExParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SgemmExParam>);

// ═══════════════════════════════════════════════════════════════════════════════
// CSV-driven parameterised test (5-step flow)
//   1. Generate host data  2. Run NPU  3. Check return code
//   4. Run CPU golden       5. Verify precision
// ═══════════════════════════════════════════════════════════════════════════════

TEST_P(SgemmExArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // ── Step 1: Generate host data ──
    // Physical dimensions (column-major):
    //   transA=N → A is M×K (physRows=M, physCols=K)
    //   transA=T → A is K×M (physRows=K, physCols=M)
    int physRowsA = (p.transA == ACLBLAS_OP_N) ? p.m : p.k;
    int physColsA = (p.transA == ACLBLAS_OP_N) ? p.k : p.m;
    int physRowsB = (p.transB == ACLBLAS_OP_N) ? p.k : p.n;
    int physColsB = (p.transB == ACLBLAS_OP_N) ? p.n : p.k;

    std::vector<float> aHost = makeBlasMatrix(physRowsA, physColsA, p.lda, p.aFill, p.randomSeed);
    std::vector<float> bHost = makeBlasMatrix(physRowsB, physColsB, p.ldb, p.bFill, p.randomSeed + 1);
    std::vector<float> cHost = makeBlasMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2);

    const float* aPtr = (aHost.empty() || p.aNull) ? nullptr : aHost.data();
    const float* bPtr = (bHost.empty() || p.bNull) ? nullptr : bHost.data();
    float* cPtr = (cHost.empty() || p.cNull) ? nullptr : cHost.data();
    const float* alphaPtr = p.alphaNull ? nullptr : &p.alpha;
    const float* betaPtr = p.betaNull ? nullptr : &p.beta;

    // Make a copy of C for golden computation (NPU will overwrite cHost in-place)
    std::vector<float> cGolden;
    if (cPtr != nullptr) {
        cGolden = cHost; // deep copy
    }

    // ── Step 2: Run NPU ──
    aclblasStatus_t ret = aclblasSgemmEx_npu(
        SgemmExArch35Test::handle_, p.transA, p.transB, p.m, p.n, p.k,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc, p.algo);

    // ── Step 3: Check return code ──
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    // Skip verification for cases with no output (m==0, n==0, C==null)
    if (p.m <= 0 || p.n <= 0 || cPtr == nullptr) {
        return;
    }

    // ── Step 4: Run CPU golden ──
    float* cGoldenPtr = cGolden.empty() ? nullptr : cGolden.data();
    aclblasSgemmEx_cpu(
        SgemmExArch35Test::handle_, p.transA, p.transB, p.m, p.n, p.k,
        alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cGoldenPtr, p.ldc, p.algo);

    // ── Step 5: Verify precision (MERE/MARE, FP32: threshold=2^-13, multiplier=10) ──
    size_t cCount = static_cast<size_t>(p.ldc) * p.n;
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = 1.0 / 8192.0; // 2^-13 ≈ 0.000122
    cfg.mareMultiplier = 10.0;
    EXPECT_TRUE(Verifier::verifyVector(cPtr, cGolden.data(), cCount, 1, cfg, p.caseName));
}
