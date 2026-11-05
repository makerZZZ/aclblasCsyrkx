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
#include "fill.h"
#include "sgemm_batched_param.h"
#include "sgemm_batched_golden.h"
#include "sgemm_batched_npu_wrapper.h"

struct SgemmBatchedTestData {
    std::vector<std::vector<float>> aBatch;
    std::vector<std::vector<float>> bBatch;
    std::vector<std::vector<float>> cBatch;
    std::vector<std::vector<float>> aGolden;
    std::vector<std::vector<float>> bGolden;
    std::vector<std::vector<float>> cGolden;
};

inline SgemmBatchedTestData GenerateSgemmBatchedTestData(const SgemmBatchedParam& p)
{
    SgemmBatchedTestData data;
    int safeBatch = std::max(0, p.batchCount);
    int physRowsA = sgemmBatchedPhysRows(p.m, p.k, p.transA);
    int physColsA = sgemmBatchedPhysCols(p.m, p.k, p.transA);
    int physRowsB = sgemmBatchedPhysRows(p.k, p.n, p.transB);
    int physColsB = sgemmBatchedPhysCols(p.k, p.n, p.transB);

    data.aBatch.resize(safeBatch);
    data.bBatch.resize(safeBatch);
    data.cBatch.resize(safeBatch);
    data.aGolden.resize(safeBatch);
    data.bGolden.resize(safeBatch);
    data.cGolden.resize(safeBatch);

    for (int b = 0; b < safeBatch; b++) {
        uint32_t seed = p.randomSeed + static_cast<uint32_t>(b * 3);
        data.aBatch[b] = makeBlasArray(
            static_cast<int64_t>(std::max(1, p.lda)) * std::max(1, physColsA), p.aFill, seed);
        data.bBatch[b] = makeBlasArray(
            static_cast<int64_t>(std::max(1, p.ldb)) * std::max(1, physColsB), p.bFill, seed + 1);
        data.cBatch[b] = makeBlasArray(
            static_cast<int64_t>(std::max(1, p.ldc)) * std::max(1, p.n), p.cFill, seed + 2);
        data.aGolden[b] = data.aBatch[b];
        data.bGolden[b] = data.bBatch[b];
        data.cGolden[b] = data.cBatch[b];
    }
    return data;
}

struct SgemmBatchedPtrArrays {
    std::vector<const float*> aPtrs;
    std::vector<const float*> bPtrs;
    std::vector<float*> cPtrs;
};

class SgemmBatchedArch35Test : public BlasTest<SgemmBatchedParam> { };

TEST_F(SgemmBatchedArch35Test, NullHandle)
{
    float alpha = 1.0f, beta = 0.0f;
    aclblasStatus_t ret = aclblasSgemmBatched(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8,
        &alpha, nullptr, 8, nullptr, 8,
        &beta, nullptr, 8, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    SgemmBatched, SgemmBatchedArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SgemmBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SgemmBatchedParam>);

inline void VerifySgemmBatchedPrecision(
    const SgemmBatchedParam& p, SgemmBatchedTestData& data,
    SgemmBatchedPtrArrays& ptrs, int safeBatch, size_t cCount,
    aclblasHandle_t testHandle)
{
    for (int b = 0; b < safeBatch; b++) {
        if (ptrs.cPtrs[b] == nullptr) continue;

        const float* const goldenA[] = { data.aGolden[b].data() };
        const float* const goldenB[] = { data.bGolden[b].data() };
        float* const goldenC[] = { data.cGolden[b].data() };

        aclblasSgemmBatched_cpu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k,
            &p.alpha, goldenA, p.lda, goldenB, p.ldb,
            &p.beta, goldenC, p.ldc, 1);

        std::string caseName = p.caseName + "_batch" + std::to_string(b);

        const float* out = data.cBatch[b].data();
        const float* gold = data.cGolden[b].data();
        VerifyConfig cfg;
        applyMixedTolerance(cfg, ACL_FLOAT, gold, cCount);

        EXPECT_TRUE(Verifier::verifyVector(out, gold, cCount, 1, cfg, caseName));
    }
}

TEST_P(SgemmBatchedArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    aclblasHandle_t testHandle = SgemmBatchedArch35Test::handle_;
    if (p.description.find("handle_null") != std::string::npos) {
        testHandle = nullptr;
    }

    int safeBatch = std::max(0, p.batchCount);
    auto data = GenerateSgemmBatchedTestData(p);
    auto ptrs = BuildGemmBatchedPtrsTpl<SgemmBatchedTestData, SgemmBatchedParam, SgemmBatchedPtrArrays>(data, p, safeBatch);

    const float* alphaPtr = p.alphaNull ? nullptr : &p.alpha;
    const float* betaPtr = p.betaNull ? nullptr : &p.beta;
    const float* const* aPtrArr = (p.aarrayNull || safeBatch == 0) ? nullptr : ptrs.aPtrs.data();
    const float* const* bPtrArr = (p.barrayNull || safeBatch == 0) ? nullptr : ptrs.bPtrs.data();
    float* const* cPtrArr = (p.carrayNull || safeBatch == 0) ? nullptr : ptrs.cPtrs.data();

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t ret = aclblasSgemmBatched_npu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k,
            alphaPtr, aPtrArr, p.lda, bPtrArr, p.ldb,
            betaPtr, cPtrArr, p.ldc, p.batchCount);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    aclblasStatus_t ret = aclblasSgemmBatched_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k,
        alphaPtr, aPtrArr, p.lda, bPtrArr, p.ldb,
        betaPtr, cPtrArr, p.ldc, p.batchCount);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (p.m <= 0 || p.n <= 0 || safeBatch == 0 || cPtrArr == nullptr) return;

    size_t cCount = static_cast<size_t>(std::max(1, p.ldc)) * p.n;
    VerifySgemmBatchedPrecision(p, data, ptrs, safeBatch, cCount, testHandle);
}
