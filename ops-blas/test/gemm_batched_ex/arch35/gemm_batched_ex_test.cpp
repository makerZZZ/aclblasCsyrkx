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
#include <cstring>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "gemm_batched_ex_param.h"
#include "gemm_batched_ex_golden.h"
#include "gemm_batched_ex_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: prepare all batched test data
// ═══════════════════════════════════════════════════════════════════════════════

struct BatchedTestData {
    std::vector<std::vector<uint8_t>> aBatchBytes;
    std::vector<std::vector<uint8_t>> bBatchBytes;
    std::vector<std::vector<uint8_t>> cBatchBytes;
    std::vector<std::vector<float>> aGoldenBatch;
    std::vector<std::vector<float>> bGoldenBatch;
    std::vector<std::vector<float>> cGoldenBatch;
};

inline BatchedTestData GenerateBatchedTestData(const GemmBatchedExParam& p)
{
    BatchedTestData data;
    int safeBatch = std::max(0, p.batchCount);
    int physRowsA = physRows(p.m, p.k, p.transA);
    int physColsA = physCols(p.m, p.k, p.transA);
    int physRowsB = physRows(p.k, p.n, p.transB);
    int physColsB = physCols(p.k, p.n, p.transB);

    data.aBatchBytes.resize(safeBatch);
    data.bBatchBytes.resize(safeBatch);
    data.cBatchBytes.resize(safeBatch);
    data.aGoldenBatch.resize(safeBatch);
    data.bGoldenBatch.resize(safeBatch);
    data.cGoldenBatch.resize(safeBatch);

    for (int b = 0; b < safeBatch; b++) {
        uint32_t batchSeed = p.randomSeed + static_cast<uint32_t>(b * 3);

        auto aFloat = makeBlasMatrix(physRowsA, physColsA, p.lda, p.aFill, batchSeed);
        auto bFloat = makeBlasMatrix(physRowsB, physColsB, p.ldb, p.bFill, batchSeed + 1);
        auto cFloat = makeBlasMatrix(p.m, p.n, p.ldc, p.cFill, batchSeed + 2);

        data.aBatchBytes[b] = QuantizeMatrix(aFloat, p.aFill, p.Atype, p.aarrayNull);
        data.bBatchBytes[b] = QuantizeMatrix(bFloat, p.bFill, p.Btype, p.barrayNull);
        data.cBatchBytes[b] = QuantizeMatrix(cFloat, p.cFill, p.Ctype, p.carrayNull);

        data.aGoldenBatch[b] = PrepareGoldenData(aFloat, p.Atype, p.aarrayNull);
        data.bGoldenBatch[b] = PrepareGoldenData(bFloat, p.Btype, p.barrayNull);
        data.cGoldenBatch[b] = PrepareGoldenData(cFloat, p.Ctype, p.carrayNull);
    }
    return data;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: build pointer arrays for batched GEMM
// ═══════════════════════════════════════════════════════════════════════════════

struct BatchedPtrArrays {
    std::vector<const void*> aPtrArray;
    std::vector<const void*> bPtrArray;
    std::vector<void*> cPtrArray;
};

inline BatchedPtrArrays BuildBatchPtrArrays(
    BatchedTestData& data, const GemmBatchedExParam& p, int safeBatch)
{
    BatchedPtrArrays ptrs;
    ptrs.aPtrArray.resize(safeBatch, nullptr);
    ptrs.bPtrArray.resize(safeBatch, nullptr);
    ptrs.cPtrArray.resize(safeBatch, nullptr);

    for (int b = 0; b < safeBatch; b++) {
        if (!data.aBatchBytes[b].empty() && !p.aarrayNull)
            ptrs.aPtrArray[b] = data.aBatchBytes[b].data();
        if (!data.bBatchBytes[b].empty() && !p.barrayNull)
            ptrs.bPtrArray[b] = data.bBatchBytes[b].data();
        if (!data.cBatchBytes[b].empty() && !p.carrayNull)
            ptrs.cPtrArray[b] = data.cBatchBytes[b].data();
    }
    return ptrs;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helper: run NPU, CPU golden, and verify precision per batch
// ═══════════════════════════════════════════════════════════════════════════════

inline void RunAndVerifyBatched(
    const GemmBatchedExParam& p, aclblasHandle_t testHandle,
    BatchedTestData& data, BatchedPtrArrays& ptrs, int safeBatch,
    const void* alphaPtr, const void* betaPtr,
    const void* const* aPtrArrPtr, const void* const* bPtrArrPtr, void* const* cPtrArrPtr)
{
    aclblasStatus_t ret = aclblasGemmBatchedEx_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k,
        alphaPtr, aPtrArrPtr, p.Atype, p.lda,
        bPtrArrPtr, p.Btype, p.ldb,
        betaPtr, cPtrArrPtr, p.Ctype, p.ldc,
        p.batchCount, p.computeType, p.algo);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (p.m <= 0 || p.n <= 0 || safeBatch == 0 || cPtrArrPtr == nullptr) return;

    VerifyConfig cfg;
    // Precision standard: docs/zh/ops_precision_standard/experimental_standard.md
    // (mixed tolerance: |actual - golden| <= atol + rtol * |golden|, matched_ratio >= 0.99,
    //  max_abs_error <= max(fixed_limit, 32 * ULP)).
    // FP8 input quantization noise dominates precision: use the FP8 input's thresholds.
    auto isFp8Input = (p.Atype == ACL_FLOAT8_E4M3FN || p.Atype == ACL_FLOAT8_E5M2 ||
                       p.Btype == ACL_FLOAT8_E4M3FN || p.Btype == ACL_FLOAT8_E5M2);
    auto applyStdThreshold = [&cfg](aclDataType dtype) {
        auto d = getMixedToleranceDefaults(dtype);
        applyMixedToleranceInternal(cfg, d);
        cfg.mixedMaxAbsErrorLimit = d.maxAbsErrorLimitFixed;
    };
    if (isFp8Input) {
        // Use the looser of the two FP8 input types (E5M2 rtol 0.5 > E4M3 rtol 0.25).
        auto aTol = getMixedToleranceDefaults(p.Atype);
        auto bTol = getMixedToleranceDefaults(p.Btype);
        applyStdThreshold((bTol.rtol > aTol.rtol) ? p.Btype : p.Atype);
    } else {
        applyStdThreshold(p.Ctype);
    }

    int safeLdc = std::max(1, p.ldc);
    size_t cCount = static_cast<size_t>(safeLdc) * p.n;

    for (int b = 0; b < safeBatch; b++) {
        if (ptrs.cPtrArray[b] == nullptr) continue;

        std::vector<float> cNpuFloat = dequantizeFromBytes(data.cBatchBytes[b], p.Ctype, cCount);

        const float* aGoldenPtr = data.aGoldenBatch[b].empty() ? nullptr : data.aGoldenBatch[b].data();
        const float* bGoldenPtr = data.bGoldenBatch[b].empty() ? nullptr : data.bGoldenBatch[b].data();
        float* cGoldenPtr = data.cGoldenBatch[b].empty() ? nullptr : data.cGoldenBatch[b].data();

        const void* goldenAPtr = aGoldenPtr;
        const void* goldenBPtr = bGoldenPtr;
        void* goldenCPtr = cGoldenPtr;
        const void* const goldenAArray[] = { goldenAPtr };
        const void* const goldenBArray[] = { goldenBPtr };
        void* const goldenCArray[] = { goldenCPtr };

        aclblasGemmBatchedEx_cpu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k,
            alphaPtr, goldenAArray, ACL_FLOAT, p.lda,
            goldenBArray, ACL_FLOAT, p.ldb,
            betaPtr, goldenCArray, p.Ctype, p.ldc, 1, p.computeType);

        std::string batchCaseName = p.caseName + "_batch" + std::to_string(b);
        EXPECT_TRUE(Verifier::verifyVector(
            cNpuFloat.data(), cGoldenPtr, cCount, 1, cfg, batchCaseName));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test fixture
// ═══════════════════════════════════════════════════════════════════════════════

class GemmBatchedExArch35Test : public BlasTest<GemmBatchedExParam> { };

// ── TEST_F: null handle ──
TEST_F(GemmBatchedExArch35Test, NullHandle)
{
    float alpha = 1.0f, beta = 0.0f;
    aclblasStatus_t ret = aclblasGemmBatchedEx_npu(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N,
        8, 8, 8,
        &alpha, nullptr, ACL_FLOAT16, 8, nullptr, ACL_FLOAT16, 8,
        &beta, nullptr, ACL_FLOAT16, 8, 4,
        ACLBLAS_COMPUTE_32F);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    GemmBatchedEx, GemmBatchedExArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<GemmBatchedExParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GemmBatchedExParam>);

TEST_P(GemmBatchedExArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    aclblasHandle_t testHandle = GemmBatchedExArch35Test::handle_;
    if (p.description.find("handle_null") != std::string::npos) {
        testHandle = nullptr;
    }

    int safeBatch = std::max(0, p.batchCount);
    auto data = GenerateBatchedTestData(p);
    auto ptrs = BuildBatchPtrArrays(data, p, safeBatch);

    // Scalar pointers
    const void* alphaPtr = (p.alphaNull) ? nullptr : &p.alpha;
    const void* betaPtr  = (p.betaNull)  ? nullptr : &p.beta;
    const void* const* aPtrArrPtr = (p.aarrayNull || safeBatch == 0) ? nullptr : ptrs.aPtrArray.data();
    const void* const* bPtrArrPtr = (p.barrayNull || safeBatch == 0) ? nullptr : ptrs.bPtrArray.data();
    void* const* cPtrArrPtr = (p.carrayNull || safeBatch == 0) ? nullptr : ptrs.cPtrArray.data();

    // Early return for error cases
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t ret = aclblasGemmBatchedEx_npu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k,
            alphaPtr, aPtrArrPtr, p.Atype, p.lda,
            bPtrArrPtr, p.Btype, p.ldb,
            betaPtr, cPtrArrPtr, p.Ctype, p.ldc,
            p.batchCount, p.computeType, p.algo);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    RunAndVerifyBatched(p, testHandle, data, ptrs, safeBatch,
                        alphaPtr, betaPtr, aPtrArrPtr, bPtrArrPtr, cPtrArrPtr);
}
