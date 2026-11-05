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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"
#include "dtype_cast.h"
#include "gemm_strided_batched_ex_golden.h"
#include "gemm_strided_batched_ex_npu_wrapper.h"

struct StridedTestData {
    std::vector<uint8_t> aBytes;
    std::vector<uint8_t> bBytes;
    std::vector<uint8_t> cBytes;
    std::vector<float> aGolden;
    std::vector<float> bGolden;
    std::vector<float> cGolden;
    int64_t aAllocStride = 1;
    int64_t bAllocStride = 1;
    int64_t cAllocStride = 1;
    int64_t aBaseOffset = 0;
    int64_t bBaseOffset = 0;
    int64_t cBaseOffset = 0;
};

inline int64_t MatrixFootprint(int ld, int cols) { return static_cast<int64_t>(std::max(1, ld)) * std::max(1, cols); }

inline size_t StridedAllocationElements(int batchCount, int64_t allocStride, int64_t footprint)
{
    if (batchCount <= 0)
        return static_cast<size_t>(footprint);
    return static_cast<size_t>((batchCount - 1) * allocStride + footprint);
}

inline void CopyQuantizedBatch(
    std::vector<uint8_t>& destination, int64_t elementOffset, aclDataType dtype, const std::vector<uint8_t>& source)
{
    size_t byteOffset = static_cast<size_t>(elementOffset) * aclDataTypeSize(dtype);
    if (!source.empty()) {
        std::copy(source.begin(), source.end(), destination.begin() + byteOffset);
    }
}

inline void CopyGoldenBatch(std::vector<float>& destination, int64_t elementOffset, const std::vector<float>& source)
{
    if (!source.empty()) {
        std::copy(source.begin(), source.end(), destination.begin() + elementOffset);
    }
}

inline StridedTestData GenerateStridedTestData(const GemmStridedBatchedExParam& p)
{
    StridedTestData data;
    const int aCols = physCols(p.m, p.k, p.transA);
    const int bCols = physCols(p.k, p.n, p.transB);
    const int64_t aFootprint = MatrixFootprint(p.lda, aCols);
    const int64_t bFootprint = MatrixFootprint(p.ldb, bCols);
    const int64_t cFootprint = MatrixFootprint(p.ldc, p.n);
    data.aAllocStride = std::max(aFootprint, std::abs(p.strideA));
    data.bAllocStride = std::max(bFootprint, std::abs(p.strideB));
    data.cAllocStride = std::max(cFootprint, std::abs(p.strideC));
    const int safeBatch = std::max(0, p.batchCount);
    data.aBaseOffset = p.strideA < 0 && safeBatch > 0 ? (safeBatch - 1) * data.aAllocStride : 0;
    data.bBaseOffset = p.strideB < 0 && safeBatch > 0 ? (safeBatch - 1) * data.bAllocStride : 0;
    data.cBaseOffset = p.strideC < 0 && safeBatch > 0 ? (safeBatch - 1) * data.cAllocStride : 0;

    data.aBytes.resize(
        StridedAllocationElements(safeBatch, data.aAllocStride, aFootprint) * aclDataTypeSize(p.Atype), 0x5a);
    data.bBytes.resize(
        StridedAllocationElements(safeBatch, data.bAllocStride, bFootprint) * aclDataTypeSize(p.Btype), 0x5a);
    data.cBytes.resize(
        StridedAllocationElements(safeBatch, data.cAllocStride, cFootprint) * aclDataTypeSize(p.Ctype), 0x5a);
    data.aGolden.assign(StridedAllocationElements(safeBatch, data.aAllocStride, aFootprint), 0.0f);
    data.bGolden.assign(StridedAllocationElements(safeBatch, data.bAllocStride, bFootprint), 0.0f);
    data.cGolden.assign(StridedAllocationElements(safeBatch, data.cAllocStride, cFootprint), 0.0f);

    for (int batch = 0; batch < safeBatch; ++batch) {
        uint32_t seed = p.randomSeed + static_cast<uint32_t>(batch * 3);
        auto aFloat = makeBlasMatrix(physRows(p.m, p.k, p.transA), aCols, p.lda, p.aFill, seed);
        auto bFloat = makeBlasMatrix(physRows(p.k, p.n, p.transB), bCols, p.ldb, p.bFill, seed + 1);
        auto cFloat = makeBlasMatrix(p.m, p.n, p.ldc, p.cFill, seed + 2);
        CopyQuantizedBatch(
            data.aBytes, data.aBaseOffset + batch * p.strideA, p.Atype,
            QuantizeMatrix(aFloat, p.aFill, p.Atype, false));
        CopyQuantizedBatch(
            data.bBytes, data.bBaseOffset + batch * p.strideB, p.Btype,
            QuantizeMatrix(bFloat, p.bFill, p.Btype, false));
        CopyQuantizedBatch(
            data.cBytes, data.cBaseOffset + batch * p.strideC, p.Ctype,
            QuantizeMatrix(cFloat, p.cFill, p.Ctype, false));
        CopyGoldenBatch(data.aGolden, data.aBaseOffset + batch * p.strideA, PrepareGoldenData(aFloat, p.Atype, false));
        CopyGoldenBatch(data.bGolden, data.bBaseOffset + batch * p.strideB, PrepareGoldenData(bFloat, p.Btype, false));
        CopyGoldenBatch(data.cGolden, data.cBaseOffset + batch * p.strideC, PrepareGoldenData(cFloat, p.Ctype, false));
    }
    return data;
}

class GemmStridedBatchedExTest : public BlasTest<GemmStridedBatchedExParam> {};

inline bool ComplexIsZero(aclblasComplex value) { return value.real == 0.0f && value.imag == 0.0f; }

struct ComplexStridedCase {
    const char* name;
    aclblasOperation_t transA;
    aclblasOperation_t transB;
    aclblasComputeType_t computeType;
    aclblasGemmAlgo_t algo;
    aclblasComplex alpha;
    aclblasComplex beta;
    bool negativeStride;
    bool nullInputs;
    bool nanC;
};

const ComplexStridedCase COMPLEX_STRIDED_CASES[] = {
    {"compute32_nn",
     ACLBLAS_OP_N,
     ACLBLAS_OP_N,
     ACLBLAS_COMPUTE_32F,
     ACLBLAS_GEMM_ALGO0,
     {0.75f, -0.25f},
     {-0.5f, 0.125f},
     false,
     false,
     false},
    {"compute32_tn",
     ACLBLAS_OP_T,
     ACLBLAS_OP_N,
     ACLBLAS_COMPUTE_32F,
     ACLBLAS_GEMM_DEFAULT,
     {-0.25f, 0.5f},
     {0.5f, -0.125f},
     false,
     false,
     false},
    {"pedantic_cn",
     ACLBLAS_OP_C,
     ACLBLAS_OP_N,
     ACLBLAS_COMPUTE_32F_PEDANTIC,
     ACLBLAS_GEMM_DEFAULT,
     {0.75f, -0.25f},
     {-0.5f, 0.125f},
     false,
     false,
     false},
    {"fast16f_tt",
     ACLBLAS_OP_T,
     ACLBLAS_OP_T,
     ACLBLAS_COMPUTE_32F_FAST_16F,
     ACLBLAS_GEMM_DEFAULT,
     {-0.5f, 0.25f},
     {0.25f, -0.125f},
     false,
     false,
     false},
    {"fast16bf_nt",
     ACLBLAS_OP_N,
     ACLBLAS_OP_T,
     ACLBLAS_COMPUTE_32F_FAST_16BF,
     ACLBLAS_GEMM_DEFAULT,
     {1.0f, 0.5f},
     {-0.25f, 0.25f},
     false,
     false,
     false},
    {"fast_tf32_cc_negative_stride",
     ACLBLAS_OP_C,
     ACLBLAS_OP_C,
     ACLBLAS_COMPUTE_32F_FAST_TF32,
     ACLBLAS_GEMM_DEFAULT,
     {0.5f, -0.75f},
     {0.125f, 0.25f},
     true,
     false,
     false},
    {"alpha_zero_null_inputs",
     ACLBLAS_OP_C,
     ACLBLAS_OP_T,
     ACLBLAS_COMPUTE_32F,
     ACLBLAS_GEMM_DEFAULT,
     {0.0f, 0.0f},
     {-0.5f, 0.25f},
     false,
     true,
     false},
    {"alpha_zero_beta_one_noop",
     ACLBLAS_OP_N,
     ACLBLAS_OP_C,
     ACLBLAS_COMPUTE_32F,
     ACLBLAS_GEMM_DEFAULT,
     {0.0f, 0.0f},
     {1.0f, 0.0f},
     false,
     true,
     false},
    {"beta_zero_does_not_read_c",
     ACLBLAS_OP_T,
     ACLBLAS_OP_C,
     ACLBLAS_COMPUTE_32F_PEDANTIC,
     ACLBLAS_GEMM_DEFAULT,
     {0.75f, 0.25f},
     {0.0f, 0.0f},
     false,
     false,
     true},
};

struct ComplexStridedLayout {
    static constexpr int M = 3;
    static constexpr int N = 2;
    static constexpr int K = 4;
    static constexpr int BATCH_COUNT = 3;
    static constexpr int LDC = M + 2;

    int lda;
    int ldb;
    int64_t aFootprint;
    int64_t bFootprint;
    int64_t cFootprint;
    int64_t aAllocStride;
    int64_t bAllocStride;
    int64_t cAllocStride;
    int64_t strideA;
    int64_t strideB;
    int64_t strideC;
    int64_t aBase;
    int64_t bBase;
    int64_t cBase;

    explicit ComplexStridedLayout(const ComplexStridedCase& testCase)
        : lda(testCase.transA == ACLBLAS_OP_N ? M + 2 : K + 2),
          ldb(testCase.transB == ACLBLAS_OP_N ? K + 2 : N + 2),
          aFootprint(static_cast<int64_t>(lda) * physCols(M, K, testCase.transA)),
          bFootprint(static_cast<int64_t>(ldb) * physCols(K, N, testCase.transB)),
          cFootprint(static_cast<int64_t>(LDC) * N),
          aAllocStride(aFootprint + 3),
          bAllocStride(bFootprint + 5),
          cAllocStride(cFootprint + 7),
          strideA(testCase.negativeStride ? -aAllocStride : aAllocStride),
          strideB(testCase.negativeStride ? -bAllocStride : bAllocStride),
          strideC(testCase.negativeStride ? -cAllocStride : cAllocStride),
          aBase(testCase.negativeStride ? (BATCH_COUNT - 1) * aAllocStride : 0),
          bBase(testCase.negativeStride ? (BATCH_COUNT - 1) * bAllocStride : 0),
          cBase(testCase.negativeStride ? (BATCH_COUNT - 1) * cAllocStride : 0)
    {}

    size_t ASize() const { return static_cast<size_t>((BATCH_COUNT - 1) * aAllocStride + aFootprint); }
    size_t BSize() const { return static_cast<size_t>((BATCH_COUNT - 1) * bAllocStride + bFootprint); }
    size_t CSize() const { return static_cast<size_t>((BATCH_COUNT - 1) * cAllocStride + cFootprint); }

    int64_t COffset(int batch, int col, int row) const
    {
        return cBase + static_cast<int64_t>(batch) * strideC + static_cast<int64_t>(col) * LDC + row;
    }
};

inline void ComplexStridedGolden(
    const ComplexStridedCase& testCase, const ComplexStridedLayout& layout, const std::vector<aclblasComplex>& a,
    const std::vector<aclblasComplex>& b, std::vector<aclblasComplex>& c)
{
    for (int batch = 0; batch < ComplexStridedLayout::BATCH_COUNT; ++batch) {
        aclblasComplex* cBatch = c.data() + layout.cBase + batch * layout.strideC;
        if (ComplexIsZero(testCase.alpha)) {
            // Preserve the API's null A/B semantics without passing null matrix pointers to CBLAS.
            for (int col = 0; col < ComplexStridedLayout::N; ++col) {
                for (int row = 0; row < ComplexStridedLayout::M; ++row) {
                    aclblasComplex& value = cBatch[static_cast<int64_t>(col) * ComplexStridedLayout::LDC + row];
                    const float real = value.real;
                    const float imag = value.imag;
                    value.real = testCase.beta.real * real - testCase.beta.imag * imag;
                    value.imag = testCase.beta.real * imag + testCase.beta.imag * real;
                }
            }
            continue;
        }
        const aclblasComplex* aBatch = a.data() + layout.aBase + batch * layout.strideA;
        const aclblasComplex* bBatch = b.data() + layout.bBase + batch * layout.strideB;
        cblas_cgemm(
            CblasColMajor, ToCblasOp(testCase.transA), ToCblasOp(testCase.transB), ComplexStridedLayout::M,
            ComplexStridedLayout::N, ComplexStridedLayout::K, &testCase.alpha, aBatch, layout.lda, bBatch, layout.ldb,
            &testCase.beta, cBatch, ComplexStridedLayout::LDC);
    }
}

inline void InitializeComplexInputs(std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& b)
{
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = {static_cast<float>(i % 7) * 0.25f, static_cast<float>(static_cast<int>(i % 5) - 2) * 0.125f};
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = {static_cast<float>(i % 3) * -0.5f, static_cast<float>(i % 4) * 0.25f};
    }
}

inline void InitializeComplexOutput(std::vector<aclblasComplex>& c)
{
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = {static_cast<float>(i % 4) * 0.125f, static_cast<float>(i % 6) * -0.0625f};
    }
}

inline void SetLogicalComplexOutput(
    std::vector<aclblasComplex>& c, const ComplexStridedLayout& layout, aclblasComplex value)
{
    for (int batch = 0; batch < ComplexStridedLayout::BATCH_COUNT; ++batch) {
        for (int col = 0; col < ComplexStridedLayout::N; ++col) {
            for (int row = 0; row < ComplexStridedLayout::M; ++row) {
                c[layout.COffset(batch, col, row)] = value;
            }
        }
    }
}

inline void VerifyComplexOutput(
    const ComplexStridedCase& testCase, const ComplexStridedLayout& layout, const std::vector<aclblasComplex>& before,
    const std::vector<aclblasComplex>& golden, const std::vector<aclblasComplex>& actual)
{
    std::vector<bool> writable(actual.size(), false);
    for (int batch = 0; batch < ComplexStridedLayout::BATCH_COUNT; ++batch) {
        for (int col = 0; col < ComplexStridedLayout::N; ++col) {
            for (int row = 0; row < ComplexStridedLayout::M; ++row) {
                const int64_t offset = layout.COffset(batch, col, row);
                writable[offset] = true;
                EXPECT_NEAR(actual[offset].real, golden[offset].real, 1.0e-5f);
                EXPECT_NEAR(actual[offset].imag, golden[offset].imag, 1.0e-5f);
            }
        }
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!writable[i]) {
            EXPECT_EQ(actual[i].real, before[i].real) << testCase.name << " modified real guard element " << i;
            EXPECT_EQ(actual[i].imag, before[i].imag) << testCase.name << " modified imag guard element " << i;
        }
    }
}

inline void RunComplexStridedCase(aclblasHandle_t handle, const ComplexStridedCase& testCase)
{
    SCOPED_TRACE(testCase.name);
    const ComplexStridedLayout layout(testCase);
    std::vector<aclblasComplex> a;
    std::vector<aclblasComplex> b;
    if (!testCase.nullInputs) {
        a.resize(layout.ASize());
        b.resize(layout.BSize());
        InitializeComplexInputs(a, b);
    }
    std::vector<aclblasComplex> c(layout.CSize());
    InitializeComplexOutput(c);
    if (testCase.nanC) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        SetLogicalComplexOutput(c, layout, {nan, nan});
    }
    const auto before = c;
    auto golden = c;
    ComplexStridedGolden(testCase, layout, a, b, golden);
    ASSERT_EQ(
        aclblasGemmStridedBatchedEx_npu(
            handle, testCase.transA, testCase.transB, ComplexStridedLayout::M, ComplexStridedLayout::N,
            ComplexStridedLayout::K, &testCase.alpha, a.empty() ? nullptr : a.data(), a.size() * sizeof(aclblasComplex),
            ACL_COMPLEX64, layout.lda, layout.strideA, b.empty() ? nullptr : b.data(),
            b.size() * sizeof(aclblasComplex), ACL_COMPLEX64, layout.ldb, layout.strideB, &testCase.beta, c.data(),
            c.size() * sizeof(aclblasComplex), ACL_COMPLEX64, ComplexStridedLayout::LDC, layout.strideC,
            ComplexStridedLayout::BATCH_COUNT, testCase.computeType, testCase.algo, layout.aBase, layout.bBase,
            layout.cBase),
        ACLBLAS_STATUS_SUCCESS);
    VerifyComplexOutput(testCase, layout, before, golden, c);
}

struct ScalarStorage {
    float alphaFloat;
    float betaFloat;
    uint16_t alphaHalf;
    uint16_t betaHalf;
    int32_t alphaInt;
    int32_t betaInt;

    explicit ScalarStorage(const GemmStridedBatchedExParam& p)
        : alphaFloat(p.alpha),
          betaFloat(p.beta),
          alphaHalf(blas_common::FloatToHalf(p.alpha)),
          betaHalf(blas_common::FloatToHalf(p.beta)),
          alphaInt(static_cast<int32_t>(p.alpha)),
          betaInt(static_cast<int32_t>(p.beta))
    {}

    const void* Alpha(const GemmStridedBatchedExParam& p) const
    {
        if (p.alphaNull)
            return nullptr;
        if (p.computeType == ACLBLAS_COMPUTE_16F || p.computeType == ACLBLAS_COMPUTE_16F_PEDANTIC)
            return &alphaHalf;
        if (p.computeType == ACLBLAS_COMPUTE_32I || p.computeType == ACLBLAS_COMPUTE_32I_PEDANTIC)
            return &alphaInt;
        return &alphaFloat;
    }

    const void* Beta(const GemmStridedBatchedExParam& p) const
    {
        if (p.betaNull)
            return nullptr;
        if (p.computeType == ACLBLAS_COMPUTE_16F || p.computeType == ACLBLAS_COMPUTE_16F_PEDANTIC)
            return &betaHalf;
        if (p.computeType == ACLBLAS_COMPUTE_32I || p.computeType == ACLBLAS_COMPUTE_32I_PEDANTIC)
            return &betaInt;
        return &betaFloat;
    }
};

inline void RunInvalidCase(
    aclblasHandle_t handle, const GemmStridedBatchedExParam& p, const void* alpha, const void* beta)
{
    alignas(16) uint8_t dummy[16] = {};
    const void* a = p.aarrayNull ? nullptr : dummy;
    const void* b = p.barrayNull ? nullptr : dummy;
    void* c = p.carrayNull ? nullptr : dummy;
    aclblasStatus_t status = aclblasGemmStridedBatchedEx(
        handle, p.transA, p.transB, p.m, p.n, p.k, alpha, a, p.Atype, p.lda, p.strideA, b, p.Btype, p.ldb, p.strideB,
        beta, c, p.Ctype, p.ldc, p.strideC, p.batchCount, p.computeType, p.algo);
    EXPECT_EQ(status, p.expectResult);
}

inline void RunEmptyCase(
    aclblasHandle_t handle, const GemmStridedBatchedExParam& p, const void* alpha, const void* beta)
{
    aclblasStatus_t status = aclblasGemmStridedBatchedEx(
        handle, p.transA, p.transB, p.m, p.n, p.k, alpha, nullptr, p.Atype, p.lda, p.strideA, nullptr, p.Btype, p.ldb,
        p.strideB, beta, nullptr, p.Ctype, p.ldc, p.strideC, p.batchCount, p.computeType, p.algo);
    EXPECT_EQ(status, ACLBLAS_STATUS_SUCCESS);
}

inline void VerifySuccessfulCase(
    const GemmStridedBatchedExParam& p, const StridedTestData& data, const std::vector<uint8_t>& cBefore)
{
    std::vector<float> cNpu = dequantizeFromBytes(data.cBytes, p.Ctype, data.cBytes.size() / aclDataTypeSize(p.Ctype));
    const size_t cFootprint = static_cast<size_t>(p.ldc) * p.n;
    const size_t cElementBytes = aclDataTypeSize(p.Ctype);
    std::vector<bool> writable(data.cBytes.size(), false);
    for (int batch = 0; batch < p.batchCount; ++batch) {
        const size_t offset = static_cast<size_t>(data.cBaseOffset + batch * p.strideC);
        VerifyConfig config;
        if (p.Ctype == ACL_INT32) {
            config.mode = PrecisionMode::INTEGER;
        } else {
            applyMixedTolerance(config, p.Ctype, data.cGolden.data() + offset, cFootprint);
        }
        EXPECT_TRUE(Verifier::verifyVector(
            cNpu.data() + offset, data.cGolden.data() + offset, cFootprint, 1, config,
            p.caseName + "_batch" + std::to_string(batch)));
        if (batch + 1 < p.batchCount && p.strideC > static_cast<int64_t>(cFootprint)) {
            const size_t gapBegin = (offset + cFootprint) * cElementBytes;
            const size_t gapEnd = static_cast<size_t>((batch + 1) * p.strideC) * cElementBytes;
            EXPECT_TRUE(
                std::equal(data.cBytes.begin() + gapBegin, data.cBytes.begin() + gapEnd, cBefore.begin() + gapBegin))
                << p.caseName << " modified the strideC gap after batch " << batch;
        }
        for (int col = 0; col < p.n; ++col) {
            for (int row = 0; row < p.m; ++row) {
                const size_t element = offset + static_cast<size_t>(col) * p.ldc + row;
                for (size_t byte = 0; byte < cElementBytes; ++byte) {
                    writable[element * cElementBytes + byte] = true;
                }
            }
        }
    }
    for (size_t byte = 0; byte < data.cBytes.size(); ++byte) {
        if (!writable[byte]) {
            ASSERT_EQ(data.cBytes[byte], cBefore[byte]) << p.caseName << " modified guard/padding byte " << byte;
        }
    }
}

inline void RunSuccessfulCase(
    aclblasHandle_t handle, const GemmStridedBatchedExParam& p, const void* alpha, const void* beta)
{
    StridedTestData data = GenerateStridedTestData(p);
    if (p.batchCount > 1) {
        ASSERT_EQ(data.aAllocStride, std::abs(p.strideA));
        ASSERT_EQ(data.bAllocStride, std::abs(p.strideB));
        ASSERT_EQ(data.cAllocStride, std::abs(p.strideC));
    }
    const std::vector<uint8_t> cBefore = data.cBytes;
    aclblasStatus_t status = aclblasGemmStridedBatchedEx_npu(
        handle, p.transA, p.transB, p.m, p.n, p.k, alpha, data.aBytes.data(), data.aBytes.size(), p.Atype, p.lda,
        p.strideA, data.bBytes.data(), data.bBytes.size(), p.Btype, p.ldb, p.strideB, beta, data.cBytes.data(),
        data.cBytes.size(), p.Ctype, p.ldc, p.strideC, p.batchCount, p.computeType, p.algo, data.aBaseOffset,
        data.bBaseOffset, data.cBaseOffset);
    ASSERT_EQ(status, ACLBLAS_STATUS_SUCCESS);
    GemmStridedBatchedExGolden(
        p, data.aGolden.data() + data.aBaseOffset, data.bGolden.data() + data.bBaseOffset,
        data.cGolden.data() + data.cBaseOffset);
    VerifySuccessfulCase(p, data, cBefore);
}

TEST_F(GemmStridedBatchedExTest, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    aclblasStatus_t status = aclblasGemmStridedBatchedEx(
        nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8, &alpha, nullptr, ACL_FLOAT16, 8, 64, nullptr, ACL_FLOAT16, 8, 64,
        &beta, nullptr, ACL_FLOAT16, 8, 64, 2, ACLBLAS_COMPUTE_32F, ACLBLAS_GEMM_DEFAULT);
    EXPECT_EQ(status, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(GemmStridedBatchedExTest, PerformanceSmoke)
{
    constexpr int m = 256;
    constexpr int n = 256;
    constexpr int k = 256;
    constexpr int batchCount = 4;
    constexpr int warmup = 3;
    constexpr int iterations = 10;
    constexpr int64_t strideA = static_cast<int64_t>(m) * k;
    constexpr int64_t strideB = static_cast<int64_t>(k) * n;
    constexpr int64_t strideC = static_cast<int64_t>(m) * n;
    const size_t aBytes = strideA * batchCount * sizeof(uint16_t);
    const size_t bBytes = strideB * batchCount * sizeof(uint16_t);
    const size_t cBytes = strideC * batchCount * sizeof(uint16_t);
    std::vector<uint16_t> a(aBytes / sizeof(uint16_t), 0x3c00); // FP16 1.0
    std::vector<uint16_t> b(bBytes / sizeof(uint16_t), 0x3c00);
    std::vector<uint16_t> c(cBytes / sizeof(uint16_t), 0);
    StridedGemmDeviceBuffers device;
    ASSERT_EQ(AllocateAndCopyStrided(&device.a, a.data(), aBytes), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(AllocateAndCopyStrided(&device.b, b.data(), bBytes), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(AllocateAndCopyStrided(&device.c, c.data(), cBytes), ACLBLAS_STATUS_SUCCESS);

    float alpha = 1.0f;
    float beta = 0.0f;
    auto launch = [&]() {
        return aclblasGemmStridedBatchedEx(
            handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k, &alpha, device.a, ACL_FLOAT16, m, strideA, device.b,
            ACL_FLOAT16, k, strideB, &beta, device.c, ACL_FLOAT16, m, strideC, batchCount, ACLBLAS_COMPUTE_32F,
            ACLBLAS_GEMM_DEFAULT);
    };
    for (int i = 0; i < warmup; ++i)
        ASSERT_EQ(launch(), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
        ASSERT_EQ(launch(), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    auto end = std::chrono::steady_clock::now();
    device.Cleanup();

    double elapsedUs = std::chrono::duration<double, std::micro>(end - begin).count() / iterations;
    double tflops = (2.0 * m * n * k * batchCount) / (elapsedUs * 1.0e6);
    std::cout << "[PERF][ascend950] aclblasGemmStridedBatchedEx m=" << m << " n=" << n << " k=" << k
              << " batch=" << batchCount << " average_us=" << elapsedUs << " TFLOPS=" << tflops << std::endl;
}

TEST_F(GemmStridedBatchedExTest, UserWorkspaceTooSmall)
{
    constexpr int m = 17;
    constexpr int n = 19;
    constexpr int k = 64;
    constexpr int batch = 2;
    constexpr int64_t strideA = 1200;
    constexpr int64_t strideB = 1300;
    constexpr int64_t strideC = 480;
    std::vector<uint16_t> a(strideA * batch, 0x3c00);
    std::vector<uint16_t> b(strideB * batch, 0x3c00);
    std::vector<uint16_t> c(strideC * batch, 0);
    StridedGemmDeviceBuffers device;
    ASSERT_EQ(AllocateAndCopyStrided(&device.a, a.data(), a.size() * sizeof(uint16_t)), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(AllocateAndCopyStrided(&device.b, b.data(), b.size() * sizeof(uint16_t)), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(AllocateAndCopyStrided(&device.c, c.data(), c.size() * sizeof(uint16_t)), ACLBLAS_STATUS_SUCCESS);
    void* tinyWorkspace = nullptr;
    ASSERT_EQ(aclrtMalloc(&tinyWorkspace, 64, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclblasSetWorkspace(handle_, tinyWorkspace, 64), ACLBLAS_STATUS_SUCCESS);

    float alpha = 0.5f;
    float beta = 0.25f;
    EXPECT_EQ(
        aclblasGemmStridedBatchedEx(
            handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k, &alpha, device.a, ACL_FLOAT16, m, strideA, device.b,
            ACL_FLOAT16, k, strideB, &beta, device.c, ACL_FLOAT16, m, strideC, batch, ACLBLAS_COMPUTE_32F,
            ACLBLAS_GEMM_ALGO6),
        ACLBLAS_STATUS_ALLOC_FAILED);

    // Switching back to the shared stream resets the handle to its library-owned workspace
    // before the temporary user workspace is released.
    ASSERT_EQ(aclblasSetStream(handle_, stream_), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtFree(tinyWorkspace), ACL_SUCCESS);
}

TEST_F(GemmStridedBatchedExTest, ComplexFp32SupportedComputeAndTransposeModes)
{
    for (const auto& testCase : COMPLEX_STRIDED_CASES) {
        RunComplexStridedCase(handle_, testCase);
    }
}

TEST_F(GemmStridedBatchedExTest, ComplexFp32RejectsUnsupportedCombinations)
{
    alignas(16) aclblasComplex complexStorage[4] = {};
    alignas(16) float floatStorage[8] = {};
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{0.0f, 0.0f};
    auto invoke = [&](aclDataType aType, const void* a, aclDataType bType, const void* b, aclDataType cType, void* c,
                      aclblasComputeType_t computeType, aclblasGemmAlgo_t algo) {
        return aclblasGemmStridedBatchedEx(
            handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 1, 1, 1, &alpha, a, aType, 1, 1, b, bType, 1, 1, &beta, c, cType, 1, 1,
            1, computeType, algo);
    };

    EXPECT_EQ(
        invoke(
            ACL_COMPLEX64, complexStorage, ACL_FLOAT, floatStorage, ACL_COMPLEX64, complexStorage, ACLBLAS_COMPUTE_32F,
            ACLBLAS_GEMM_DEFAULT),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(
        invoke(
            ACL_COMPLEX64, complexStorage, ACL_COMPLEX64, complexStorage, ACL_FLOAT, floatStorage, ACLBLAS_COMPUTE_32F,
            ACLBLAS_GEMM_DEFAULT),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(
        invoke(
            ACL_COMPLEX64, complexStorage, ACL_COMPLEX64, complexStorage, ACL_COMPLEX64, complexStorage,
            ACLBLAS_COMPUTE_16F, ACLBLAS_GEMM_DEFAULT),
        ACLBLAS_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(
        invoke(
            ACL_COMPLEX64, complexStorage, ACL_COMPLEX64, complexStorage, ACL_COMPLEX64, complexStorage,
            ACLBLAS_COMPUTE_32F, ACLBLAS_GEMM_ALGO1),
        ACLBLAS_STATUS_NOT_SUPPORTED);
}

INSTANTIATE_TEST_SUITE_P(
    GemmStridedBatchedEx, GemmStridedBatchedExTest,
    ::testing::ValuesIn(GetCasesFromCsv<GemmStridedBatchedExParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GemmStridedBatchedExParam>);

TEST_P(GemmStridedBatchedExTest, CsvDriven)
{
    const auto& p = GetParam();
    const ScalarStorage scalars(p);
    const void* alpha = scalars.Alpha(p);
    const void* beta = scalars.Beta(p);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        RunInvalidCase(handle_, p, alpha, beta);
        return;
    }
    if (p.batchCount == 0 || p.m == 0 || p.n == 0) {
        RunEmptyCase(handle_, p, alpha, beta);
        return;
    }
    RunSuccessfulCase(handle_, p, alpha, beta);
}
