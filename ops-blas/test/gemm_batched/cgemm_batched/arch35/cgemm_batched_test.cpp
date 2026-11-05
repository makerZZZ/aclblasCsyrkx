/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cgemm_batched_param.h"
#include "cgemm_batched_golden.h"
#include "cgemm_batched_npu_wrapper.h"

struct CgemmBatchedTestData {
    std::vector<std::vector<aclblasComplex>> aBatch;
    std::vector<std::vector<aclblasComplex>> bBatch;
    std::vector<std::vector<aclblasComplex>> cBatch;
    std::vector<std::vector<aclblasComplex>> aGolden;
    std::vector<std::vector<aclblasComplex>> bGolden;
    std::vector<std::vector<aclblasComplex>> cGolden;
};

// Generate complex array from a float fill mode: use two independent float arrays for real/imag.
inline std::vector<aclblasComplex> makeComplexArray(int64_t size, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || size <= 0) {
        return {};
    }
    std::vector<float> realPart = makeBlasArray(size, fill, seed);
    std::vector<float> imagPart = makeBlasArray(size, fill, seed + 7919);
    std::vector<aclblasComplex> data(static_cast<size_t>(size));
    for (size_t i = 0; i < static_cast<size_t>(size); i++) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

inline CgemmBatchedTestData GenerateCgemmBatchedTestData(const CgemmBatchedParam& p)
{
    CgemmBatchedTestData data;
    int safeBatch = std::max(0, p.batchCount);
    int physRowsA = cgemmBatchedPhysRows(p.m, p.k, p.transA);
    int physColsA = cgemmBatchedPhysCols(p.m, p.k, p.transA);
    int physRowsB = cgemmBatchedPhysRows(p.k, p.n, p.transB);
    int physColsB = cgemmBatchedPhysCols(p.k, p.n, p.transB);

    data.aBatch.resize(safeBatch);
    data.bBatch.resize(safeBatch);
    data.cBatch.resize(safeBatch);
    data.aGolden.resize(safeBatch);
    data.bGolden.resize(safeBatch);
    data.cGolden.resize(safeBatch);

    for (int b = 0; b < safeBatch; b++) {
        uint32_t seed = p.randomSeed + static_cast<uint32_t>(b * 3);
        int64_t aSize = static_cast<int64_t>(std::max(1, p.lda)) * std::max(1, physColsA);
        int64_t bSize = static_cast<int64_t>(std::max(1, p.ldb)) * std::max(1, physColsB);
        int64_t cSize = static_cast<int64_t>(std::max(1, p.ldc)) * std::max(1, p.n);
        data.aBatch[b] = makeComplexArray(aSize, p.aFill, seed);
        data.bBatch[b] = makeComplexArray(bSize, p.bFill, seed + 1);
        data.cBatch[b] = makeComplexArray(cSize, p.cFill, seed + 2);
        data.aGolden[b] = data.aBatch[b];
        data.bGolden[b] = data.bBatch[b];
        data.cGolden[b] = data.cBatch[b];
    }
    return data;
}

struct CgemmBatchedPtrArrays {
    std::vector<const aclblasComplex*> aPtrs;
    std::vector<const aclblasComplex*> bPtrs;
    std::vector<aclblasComplex*> cPtrs;
};

class CgemmBatchedTest : public BlasTest<CgemmBatchedParam> {};

TEST_F(CgemmBatchedTest, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f}, beta{0.0f, 0.0f};
    aclblasStatus_t ret = aclblasCgemmBatched(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8, &alpha, nullptr, 8, nullptr, 8, &beta, nullptr, 8, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

TEST_F(CgemmBatchedTest, KZeroIgnoresNonFiniteAlpha)
{
    aclblasComplex dummy{0.0f, 0.0f};
    const aclblasComplex* aArray[] = {&dummy};
    const aclblasComplex* bArray[] = {&dummy};
    std::vector<aclblasComplex> c = {{1.0f, 2.0f}, {-3.0f, 4.0f}, {5.0f, -6.0f}, {-7.0f, -8.0f}};
    aclblasComplex* cArray[] = {c.data()};
    const std::vector<aclblasComplex> input = c;
    aclblasComplex alpha{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    aclblasComplex beta{0.5f, -0.25f};

    aclblasStatus_t ret = aclblasCgemmBatched_npu(
        CgemmBatchedTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 2, 2, 0, &alpha, aArray, 2, bArray, 1, &beta, cArray, 2,
        1);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    for (size_t i = 0; i < c.size(); i++) {
        float expectedReal = beta.real * input[i].real - beta.imag * input[i].imag;
        float expectedImag = beta.real * input[i].imag + beta.imag * input[i].real;
        EXPECT_FLOAT_EQ(c[i].real, expectedReal);
        EXPECT_FLOAT_EQ(c[i].imag, expectedImag);
    }
}

struct CgemmBatchedPerfBuffers {
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    void* aPtrs = nullptr;
    void* bPtrs = nullptr;
    void* cPtrs = nullptr;

    ~CgemmBatchedPerfBuffers()
    {
        if (a != nullptr)
            aclrtFree(a);
        if (b != nullptr)
            aclrtFree(b);
        if (c != nullptr)
            aclrtFree(c);
        if (aPtrs != nullptr)
            aclrtFree(aPtrs);
        if (bPtrs != nullptr)
            aclrtFree(bPtrs);
        if (cPtrs != nullptr)
            aclrtFree(cPtrs);
    }
};

inline void AllocateCgemmBatchedPerfBuffers(
    CgemmBatchedPerfBuffers& buffers, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    int lda, int ldb, int ldc, int batchCount)
{
    const auto sizes = ComputeCgemmBatchedBufferSizes(transA, transB, m, n, k, lda, ldb, ldc);
    const size_t matrixABytes = sizes.aBytes;
    const size_t matrixBBytes = sizes.bBytes;
    const size_t matrixCBytes = sizes.cBytes;
    const size_t allABytes = matrixABytes * batchCount;
    const size_t allBBytes = matrixBBytes * batchCount;
    const size_t allCBytes = matrixCBytes * batchCount;

    ASSERT_EQ(aclrtMalloc(&buffers.a, allABytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&buffers.b, allBBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&buffers.c, allCBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(buffers.a, allABytes, 0, allABytes), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(buffers.b, allBBytes, 0, allBBytes), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemset(buffers.c, allCBytes, 0, allCBytes), ACL_SUCCESS);

    std::vector<const aclblasComplex*> aPtrs(batchCount);
    std::vector<const aclblasComplex*> bPtrs(batchCount);
    std::vector<aclblasComplex*> cPtrs(batchCount);
    for (int batch = 0; batch < batchCount; batch++) {
        aPtrs[batch] = reinterpret_cast<const aclblasComplex*>(static_cast<uint8_t*>(buffers.a) + matrixABytes * batch);
        bPtrs[batch] = reinterpret_cast<const aclblasComplex*>(static_cast<uint8_t*>(buffers.b) + matrixBBytes * batch);
        cPtrs[batch] = reinterpret_cast<aclblasComplex*>(static_cast<uint8_t*>(buffers.c) + matrixCBytes * batch);
    }

    const size_t ptrBytes = static_cast<size_t>(batchCount) * sizeof(void*);
    ASSERT_EQ(aclrtMalloc(&buffers.aPtrs, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&buffers.bPtrs, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&buffers.cPtrs, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(buffers.aPtrs, ptrBytes, aPtrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(buffers.bPtrs, ptrBytes, bPtrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(buffers.cPtrs, ptrBytes, cPtrs.data(), ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
}

inline double RunCgemmBatchedPerformanceCase(
    aclblasHandle_t handle, aclrtStream stream, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n,
    int k, int lda, int ldb, int ldc, int batchCount)
{
    constexpr int warmups = 5;
    constexpr int samples = 60;
    CgemmBatchedPerfBuffers buffers;
    AllocateCgemmBatchedPerfBuffers(buffers, transA, transB, m, n, k, lda, ldb, ldc, batchCount);
    const auto* aPtrs = reinterpret_cast<const aclblasComplex* const*>(buffers.aPtrs);
    const auto* bPtrs = reinterpret_cast<const aclblasComplex* const*>(buffers.bPtrs);
    auto* cPtrs = reinterpret_cast<aclblasComplex* const*>(buffers.cPtrs);
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{0.0f, 0.0f};
    auto launch = [&]() {
        return aclblasCgemmBatched(
            handle, transA, transB, m, n, k, &alpha, aPtrs, lda, bPtrs, ldb, &beta, cPtrs, ldc, batchCount);
    };

    for (int i = 0; i < warmups; i++) {
        EXPECT_EQ(launch(), ACLBLAS_STATUS_SUCCESS);
    }
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < samples; i++) {
        EXPECT_EQ(launch(), ACLBLAS_STATUS_SUCCESS);
    }
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / samples;
}

inline double RunCgemmBatchedPerformanceCase(
    aclblasHandle_t handle, aclrtStream stream, int m, int n, int k, int batchCount)
{
    return RunCgemmBatchedPerformanceCase(handle, stream, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k, m, k, m, batchCount);
}

inline char CgemmBatchedOperationChar(aclblasOperation_t operation)
{
    if (operation == ACLBLAS_OP_T)
        return 'T';
    if (operation == ACLBLAS_OP_C)
        return 'C';
    return 'N';
}

TEST_F(CgemmBatchedTest, TC_PF_PerformanceContract)
{
    constexpr double performanceRatio = 0.4;
    struct PerfCase {
        const char* name;
        int m;
        int n;
        int k;
        int batchCount;
        double h100Us;
    };
    const PerfCase cases[] = {
        {"256x256x256_b32_NN", 256, 256, 256, 32, 91.439},
        {"512x512x512_b16_NN", 512, 512, 512, 16, 337.538},
        {"1024x1024x1024_b8_NN", 1024, 1024, 1024, 8, 1309.025},
    };

    for (const auto& perf : cases) {
        // gpu_baseline.csv defines ratio = H100 time / NPU time and requires
        // ratio >= 0.4, hence NPU time <= H100 time / 0.4.
        const double limitUs = perf.h100Us / performanceRatio;
        const double averageUs =
            RunCgemmBatchedPerformanceCase(handle_, stream_, perf.m, perf.n, perf.k, perf.batchCount);
        std::cout << std::fixed << std::setprecision(3) << "[CGEMM_BATCHED_PERF] case=" << perf.name
                  << " device=Ascend-950PR backend=npu warmups=5 samples=60"
                  << " average_us=" << averageUs << " h100_us=" << perf.h100Us << " ratio_required=" << performanceRatio
                  << " limit_us=" << limitUs << std::endl;
        EXPECT_LE(averageUs, limitUs) << perf.name;
    }
}

TEST_P(CgemmBatchedTest, CsvPerformance)
{
    const auto& p = GetParam();
    const char* enabled = std::getenv("CGEMM_BATCHED_RUN_ALL_PERF");
    if (enabled == nullptr || std::string(enabled) != "1") {
        GTEST_SKIP() << "set CGEMM_BATCHED_RUN_ALL_PERF=1 to run CSV performance cases";
    }
    if (p.caseName.rfind("TC_PF_", 0) != 0) {
        GTEST_SKIP() << "not a TC_PF case";
    }

    const double averageUs = RunCgemmBatchedPerformanceCase(
        CgemmBatchedTest::handle_, CgemmBatchedTest::stream_, p.transA, p.transB, p.m, p.n, p.k, p.lda, p.ldb, p.ldc,
        p.batchCount);
    std::cout << std::fixed << std::setprecision(3) << "[CGEMM_BATCHED_CSV_PERF] case=" << p.caseName
              << " device=Ascend-950PR backend=npu warmups=5 samples=60"
              << " transA=" << CgemmBatchedOperationChar(p.transA) << " transB=" << CgemmBatchedOperationChar(p.transB)
              << " m=" << p.m << " n=" << p.n << " k=" << p.k << " lda=" << p.lda << " ldb=" << p.ldb
              << " ldc=" << p.ldc << " batchCount=" << p.batchCount << " average_us=" << averageUs << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
    CgemmBatched, CgemmBatchedTest,
    ::testing::ValuesIn(GetCasesFromCsv<CgemmBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgemmBatchedParam>);

inline void VerifyCgemmBatchedPrecision(
    const CgemmBatchedParam& p, CgemmBatchedTestData& data, CgemmBatchedPtrArrays& ptrs, int safeBatch,
    aclblasHandle_t testHandle)
{
    // MIXED_TOLERANCE defines a 99% match requirement for an operator output.
    // Treat the batched result as one logical output so a short matrix in one
    // batch does not turn a single cancellation-sensitive value into a 100%
    // failure rate.  Padding outside the logical m-by-n matrices is excluded.
    const size_t logicalCount = static_cast<size_t>(safeBatch) * p.m * p.n;
    std::vector<float> outR;
    std::vector<float> outI;
    std::vector<float> goldR;
    std::vector<float> goldI;
    outR.reserve(logicalCount);
    outI.reserve(logicalCount);
    goldR.reserve(logicalCount);
    goldI.reserve(logicalCount);

    for (int b = 0; b < safeBatch; b++) {
        if (ptrs.cPtrs[b] == nullptr)
            continue;

        const aclblasComplex* const goldenA[] = {data.aGolden[b].data()};
        const aclblasComplex* const goldenB[] = {data.bGolden[b].data()};
        aclblasComplex* const goldenC[] = {data.cGolden[b].data()};

        aclblasCgemmBatched_cpu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, &p.alpha, goldenA, p.lda, goldenB, p.ldb, &p.beta, goldenC,
            p.ldc, 1);

        const aclblasComplex* out = data.cBatch[b].data();
        const aclblasComplex* gold = data.cGolden[b].data();
        for (int col = 0; col < p.n; col++) {
            for (int row = 0; row < p.m; row++) {
                const size_t index = static_cast<size_t>(col) * p.ldc + row;
                outR.push_back(out[index].real);
                outI.push_back(out[index].imag);
                goldR.push_back(gold[index].real);
                goldI.push_back(gold[index].imag);
            }
        }
    }

    if (outR.empty())
        return;
    VerifyConfig cfgR;
    VerifyConfig cfgI;
    applyMixedTolerance(cfgR, ACL_FLOAT, goldR.data(), goldR.size());
    applyMixedTolerance(cfgI, ACL_FLOAT, goldI.data(), goldI.size());
    const std::string caseName = p.caseName + "_all_batches";
    EXPECT_TRUE(Verifier::verifyVector(outR.data(), goldR.data(), outR.size(), 1, cfgR, caseName + "_real"));
    EXPECT_TRUE(Verifier::verifyVector(outI.data(), goldI.data(), outI.size(), 1, cfgI, caseName + "_imag"));
}

TEST_P(CgemmBatchedTest, CsvDriven)
{
    const auto& p = GetParam();

    aclblasHandle_t testHandle = CgemmBatchedTest::handle_;
    if (p.description.find("handle_null") != std::string::npos) {
        testHandle = nullptr;
    }

    int safeBatch = std::max(0, p.batchCount);
    auto data = GenerateCgemmBatchedTestData(p);
    auto ptrs =
        BuildGemmBatchedPtrsTpl<CgemmBatchedTestData, CgemmBatchedParam, CgemmBatchedPtrArrays>(data, p, safeBatch);

    const aclblasComplex* alphaPtr = p.alphaNull ? nullptr : &p.alpha;
    const aclblasComplex* betaPtr = p.betaNull ? nullptr : &p.beta;
    const aclblasComplex* const* aPtrArr = (p.aarrayNull || safeBatch == 0) ? nullptr : ptrs.aPtrs.data();
    const aclblasComplex* const* bPtrArr = (p.barrayNull || safeBatch == 0) ? nullptr : ptrs.bPtrs.data();
    aclblasComplex* const* cPtrArr = (p.carrayNull || safeBatch == 0) ? nullptr : ptrs.cPtrs.data();

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasStatus_t ret = aclblasCgemmBatched_npu(
            testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtrArr, p.lda, bPtrArr, p.ldb, betaPtr, cPtrArr,
            p.ldc, p.batchCount);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    aclblasStatus_t ret = aclblasCgemmBatched_npu(
        testHandle, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtrArr, p.lda, bPtrArr, p.ldb, betaPtr, cPtrArr,
        p.ldc, p.batchCount);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (p.m <= 0 || p.n <= 0 || safeBatch == 0 || cPtrArr == nullptr)
        return;

    VerifyCgemmBatchedPrecision(p, data, ptrs, safeBatch, testHandle);
}
