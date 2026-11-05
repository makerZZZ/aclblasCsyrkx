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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "gemm_strided_batched_param.h"
#include "gemm_strided_batched_golden.h"
#include "gemm_strided_batched_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Prepared host data spanning all batches (column-major, strides pre-applied).
// ═══════════════════════════════════════════════════════════════════════════════
struct GsbData {
    std::vector<float> aHost;   // A span over all batches
    std::vector<float> bHost;   // B span over all batches
    std::vector<float> cHost;   // C span (NPU output after run)
    std::vector<float> cGolden; // C span (CPU golden output)
};

inline GsbData PrepareData(const GemmStridedBatchedParam& p)
{
    GsbData d;
    const int physRowsA = gsbPhysRows(p.m, p.k, p.transA);
    const int physColsA = gsbPhysCols(p.m, p.k, p.transA);
    const int physRowsB = gsbPhysRows(p.k, p.n, p.transB);
    const int physColsB = gsbPhysCols(p.k, p.n, p.transB);

    BlasFillMode aFill = p.aNull ? parseFill("NULLPTR") : p.aFill;
    BlasFillMode bFill = p.bNull ? parseFill("NULLPTR") : p.bFill;
    BlasFillMode cFill = p.cNull ? parseFill("NULLPTR") : p.cFill;

    d.aHost = gsbMakeStridedBatched(
        physRowsA, physColsA, p.lda, p.effStrideA(), p.batchCount, aFill, p.randomSeed);
    d.bHost = gsbMakeStridedBatched(
        physRowsB, physColsB, p.ldb, p.effStrideB(), p.batchCount, bFill, p.randomSeed + 100);
    d.cHost = gsbMakeStridedBatched(
        p.m, p.n, p.ldc, p.effStrideC(), p.batchCount, cFill, p.randomSeed + 200);
    d.cGolden = d.cHost; // identical starting C for the beta*C term
    return d;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fixture
// ═══════════════════════════════════════════════════════════════════════════════
class GemmStridedBatchedArch35Test : public BlasTest<GemmStridedBatchedParam> {
};

// ── TEST_F: null handle (not driven by CSV verify path) ──
TEST_F(GemmStridedBatchedArch35Test, NullHandle)
{
    float alpha = 1.0f, beta = 0.0f;
    aclblasStatus_t ret = aclblasSgemmStridedBatched_npu(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, 4, &alpha, nullptr, 4, 16, nullptr, 4, 16, &beta, nullptr, 4, 16, 1,
        0, 0, 0);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    GemmStridedBatched, GemmStridedBatchedArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<GemmStridedBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GemmStridedBatchedParam>);

TEST_P(GemmStridedBatchedArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // handle==null cases run with a null handle.
    aclblasHandle_t testHandle = GemmStridedBatchedArch35Test::handle_;
    if (p.expectResult == ACLBLAS_STATUS_HANDLE_IS_NULLPTR) {
        testHandle = nullptr;
    }

    GsbData d = PrepareData(p);

    const float* alphaPtr = p.alphaNull ? nullptr : &p.alpha;
    const float* betaPtr = p.betaNull ? nullptr : &p.beta;
    const float* aPtr = d.aHost.empty() ? nullptr : d.aHost.data();
    const float* bPtr = d.bHost.empty() ? nullptr : d.bHost.data();
    float* cPtr = d.cHost.empty() ? nullptr : d.cHost.data();

    const size_t aBytes = d.aHost.size() * sizeof(float);
    const size_t bBytes = d.bHost.size() * sizeof(float);
    const size_t cBytes = d.cHost.size() * sizeof(float);

    // Step 2: run on NPU (device malloc/H2D/kernel/sync/D2H/free inside wrapper).
    aclblasStatus_t ret = aclblasSgemmStridedBatched_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtr, p.lda, p.effStrideA(), bPtr, p.ldb,
        p.effStrideB(), betaPtr, cPtr, p.ldc, p.effStrideC(), p.batchCount, aBytes, bBytes, cBytes);

    // Step 3: error / expected-return-code check.
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    // No-computation cases (m/n/batchCount==0): SUCCESS with C unchanged, no numeric compare.
    if (p.m == 0 || p.n == 0 || p.batchCount == 0 || cPtr == nullptr) {
        return;
    }

    // Step 4: CPU golden on the identical starting C copy.
    aclblasStatus_t goldenRet = aclblasSgemmStridedBatched_cpu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, d.aHost.empty() ? nullptr : d.aHost.data(), p.lda,
        p.effStrideA(), d.bHost.empty() ? nullptr : d.bHost.data(), p.ldb, p.effStrideB(), betaPtr, d.cGolden.data(),
        p.ldc, p.effStrideC(), p.batchCount);
    ASSERT_EQ(static_cast<int>(goldenRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    // Step 5: precision compare (MERE/MARE, aligned with test plan 1.3.B §4).
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, d.cGolden.data(), d.cHost.size());

    EXPECT_TRUE(Verifier::verifyVector(
        d.cHost.data(), d.cGolden.data(), d.cHost.size(), 1, cfg, p.caseName));
}
