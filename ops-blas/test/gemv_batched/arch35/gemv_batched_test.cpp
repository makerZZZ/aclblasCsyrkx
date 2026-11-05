/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "gemv_batched_param.h"
#include "gemv_batched_golden.h"
#include "gemv_batched_npu_wrapper.h"

// ============================================================
// FP16 conversion helpers (for dtype 0=HSH, 2=HSS)
// ============================================================
static uint16_t FloatToHalf(float val)
{
    uint32_t f; memcpy(&f, &val, sizeof(f));
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x3FF;
    if (exp <= 0)
        return static_cast<uint16_t>(sign);
    if (exp >= 31)
        return static_cast<uint16_t>(sign | 0x7C00);
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

static float HalfToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = (sign << 31) | (mant << 13);
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result; memcpy(&result, &f, sizeof(result));
    return result;
}

// ============================================================
// bf16 conversion helpers (for dtype 3=TST, 4=TSS)
// ============================================================
static uint16_t FloatToBfloat(float val)
{
    uint32_t f; memcpy(&f, &val, sizeof(f));
    return static_cast<uint16_t>(f >> 16);
}

static float BfloatToFloat(uint16_t b)
{
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float r; memcpy(&r, &f, sizeof(r));
    return r;
}

// ============================================================
// Quantize helpers
// ============================================================
static void QuantizeRoundTripWithHalf(const std::vector<float>& src, std::vector<float>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++)
        dst[i] = HalfToFloat(FloatToHalf(src[i]));
}

static void QuantizeRoundTripWithBf16(const std::vector<float>& src, std::vector<float>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++)
        dst[i] = BfloatToFloat(FloatToBfloat(src[i]));
}

static void QuantizeToHalf(const std::vector<float>& src, std::vector<uint16_t>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++)
        dst[i] = FloatToHalf(src[i]);
}

static void QuantizeToBf16(const std::vector<float>& src, std::vector<uint16_t>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++)
        dst[i] = FloatToBfloat(src[i]);
}

static void ConvertHalfToFloat(const std::vector<uint16_t>& src, std::vector<float>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++)
        dst[i] = HalfToFloat(src[i]);
}

static void ConvertBf16ToFloat(const std::vector<uint16_t>& src, std::vector<float>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++)
        dst[i] = BfloatToFloat(src[i]);
}

// ============================================================
// Test fixture
// ============================================================
class GemvBatchedArch35Test : public BlasTest<GemvBatchedParam> {};

TEST_F(GemvBatchedArch35Test, NullHandle)
{
    float alpha = 1.0f, beta = 0.0f;
    std::vector<float> af(64, 1.0f), xf(8, 1.0f), yf(8, 0.0f);
    aclblasStatus_t ret =
        aclblasGemvBatchedS_npu(nullptr, ACLBLAS_OP_N, 8, 8, &alpha, af.data(), 8, xf.data(), 1, &beta, yf.data(), 1, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    GemvBatched, GemvBatchedArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<GemvBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GemvBatchedParam>);

TEST_P(GemvBatchedArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // For error cases, use safe dimensions for data generation
    // (m<0 or n<0 would overflow makeBlasArray's size_t parameter)
    int safeM = std::max(1, std::abs(p.m));
    int safeN = std::max(1, std::abs(p.n));
    int safeBatchCount = std::max(1, p.batchCount);
    int safeLda = std::max(safeM, p.lda);

    const bool isTransN = (p.trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? safeN : safeM;
    const int yCount = isTransN ? safeM : safeN;
    const size_t xStride = static_cast<size_t>((xCount - 1) * std::abs(p.incx) + 1);
    const size_t yStride = static_cast<size_t>((yCount - 1) * std::abs(p.incy) + 1);

    // Generate float data using safe dimensions
    auto aFloat = makeBlasArray(
        static_cast<int64_t>(safeBatchCount) * safeLda * safeN, p.a,
        p.randomSeed);

    std::vector<float> xFloat(safeBatchCount * xStride);
    std::vector<float> yFloat(safeBatchCount * yStride);
    for (int b = 0; b < safeBatchCount; b++) {
        if (p.x.method != BlasFillMode::M_NULLPTR) {
            auto xBatch = makeBlasStrided(xCount, p.incx, p.x, p.randomSeed + 1 + b);
            for (size_t i = 0; i < xStride; i++)
                xFloat[b * xStride + i] = xBatch[i];
        }
        if (p.y.method != BlasFillMode::M_NULLPTR) {
            auto yBatch = makeBlasStrided(yCount, p.incy, p.y, p.randomSeed + 2 + b);
            for (size_t i = 0; i < yStride; i++)
                yFloat[b * yStride + i] = yBatch[i];
        }
    }

    const float* alphaPtr = (p.alphaFill.method == BlasFillMode::M_NULLPTR) ? nullptr : &p.alpha;
    const float* betaPtr = (p.betaFill.method == BlasFillMode::M_NULLPTR) ? nullptr : &p.beta;
    const float* aPtr = (p.a.method == BlasFillMode::M_NULLPTR) ? nullptr : aFloat.data();
    const float* xPtr = (p.x.method == BlasFillMode::M_NULLPTR) ? nullptr : xFloat.data();
    float* yErrPtr = (p.y.method == BlasFillMode::M_NULLPTR) ? nullptr : yFloat.data();

    // Early-return / error cases — always tested via dtype 1 (S: float in/out)
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        std::vector<float> yNpu(yFloat);
        aclblasStatus_t ret = aclblasGemvBatchedS_npu(
            GemvBatchedArch35Test::handle_, p.trans, p.m, p.n, alphaPtr,
            aPtr, p.lda, xPtr, p.incx, betaPtr, yErrPtr, p.incy, p.batchCount);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    // === Normal case: dispatch by dtype ===
    aclblasStatus_t ret = ACLBLAS_STATUS_SUCCESS;
    std::vector<float> npuFloat(p.batchCount * yStride);  // logical float NPU output
    std::vector<float> goldenFloat(p.batchCount * yStride); // logical float golden output
    std::vector<float> aGolden;  // quantized A for golden
    std::vector<float> xGolden;  // quantized x for golden

    switch (p.dtype) {
    case 1: { // S: float in, float out
        aGolden = aFloat;
        xGolden = xFloat;
        npuFloat = yFloat;
        ret = aclblasGemvBatchedS_npu(
            GemvBatchedArch35Test::handle_, p.trans, p.m, p.n, alphaPtr,
            aFloat.data(), p.lda, xFloat.data(), p.incx, betaPtr,
            npuFloat.data(), p.incy, p.batchCount);
        goldenFloat = yFloat;
        break;
    }
    case 0: { // HSH: FP16 in, FP16 out
        std::vector<uint16_t> aHalf, xHalf, yHalf;
        QuantizeToHalf(aFloat, aHalf);
        QuantizeToHalf(xFloat, xHalf);
        QuantizeToHalf(yFloat, yHalf);
        QuantizeRoundTripWithHalf(aFloat, aGolden);
        QuantizeRoundTripWithHalf(xFloat, xGolden);
        std::vector<float> yGoldenTmp;
        QuantizeRoundTripWithHalf(yFloat, yGoldenTmp);
        std::vector<uint16_t> yNpu = yHalf;
        ret = aclblasGemvBatchedHSH_npu(
            GemvBatchedArch35Test::handle_, p.trans, p.m, p.n, alphaPtr,
            aHalf.data(), p.lda, xHalf.data(), p.incx, betaPtr,
            yNpu.data(), p.incy, p.batchCount);
        ConvertHalfToFloat(yNpu, npuFloat);
        goldenFloat = yGoldenTmp;
        break;
    }
    case 2: { // HSS: FP16 in, float out
        std::vector<uint16_t> aHalf, xHalf;
        QuantizeToHalf(aFloat, aHalf);
        QuantizeToHalf(xFloat, xHalf);
        QuantizeRoundTripWithHalf(aFloat, aGolden);
        QuantizeRoundTripWithHalf(xFloat, xGolden);
        goldenFloat = yFloat;
        npuFloat = yFloat;
        ret = aclblasGemvBatchedHSS_npu(
            GemvBatchedArch35Test::handle_, p.trans, p.m, p.n, alphaPtr,
            aHalf.data(), p.lda, xHalf.data(), p.incx, betaPtr,
            npuFloat.data(), p.incy, p.batchCount);
        break;
    }
    case 3: { // TST: bf16 in, bf16 out
        std::vector<uint16_t> aBf16, xBf16, yBf16;
        QuantizeToBf16(aFloat, aBf16);
        QuantizeToBf16(xFloat, xBf16);
        QuantizeToBf16(yFloat, yBf16);
        QuantizeRoundTripWithBf16(aFloat, aGolden);
        QuantizeRoundTripWithBf16(xFloat, xGolden);
        std::vector<float> yGoldenTmp;
        QuantizeRoundTripWithBf16(yFloat, yGoldenTmp);
        std::vector<uint16_t> yNpu = yBf16;
        ret = aclblasGemvBatchedTST_npu(
            GemvBatchedArch35Test::handle_, p.trans, p.m, p.n, alphaPtr,
            aBf16.data(), p.lda, xBf16.data(), p.incx, betaPtr,
            yNpu.data(), p.incy, p.batchCount);
        ConvertBf16ToFloat(yNpu, npuFloat);
        goldenFloat = yGoldenTmp;
        break;
    }
    case 4: { // TSS: bf16 in, float out
        std::vector<uint16_t> aBf16, xBf16;
        QuantizeToBf16(aFloat, aBf16);
        QuantizeToBf16(xFloat, xBf16);
        QuantizeRoundTripWithBf16(aFloat, aGolden);
        QuantizeRoundTripWithBf16(xFloat, xGolden);
        goldenFloat = yFloat;
        npuFloat = yFloat;
        ret = aclblasGemvBatchedTSS_npu(
            GemvBatchedArch35Test::handle_, p.trans, p.m, p.n, alphaPtr,
            aBf16.data(), p.lda, xBf16.data(), p.incx, betaPtr,
            npuFloat.data(), p.incy, p.batchCount);
        break;
    }
    default:
        ASSERT_TRUE(false) << "Unknown dtype: " << p.dtype;
        return;
    }

    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    // Run CPU golden
    aclblasGemvBatched_cpu(
        GemvBatchedArch35Test::handle_, p.trans, p.m, p.n, alphaPtr, aGolden.data(), p.lda,
        xGolden.data(), p.incx, betaPtr, goldenFloat.data(), p.incy, p.batchCount);

    if (yCount == 0 || p.batchCount == 0 || p.mereThreshold <= 0.0) {
        return;
    }

    // De-stride for verification
    const int absIncy = std::abs(p.incy);
    std::vector<float> npuLogical(p.batchCount * yCount);
    std::vector<float> cpuLogical(p.batchCount * yCount);
    for (int b = 0; b < p.batchCount; b++) {
        for (int i = 0; i < yCount; i++) {
            int yIdx = (p.incy > 0) ? (i * p.incy) : ((yCount - 1 - i) * absIncy);
            npuLogical[b * yCount + i] = npuFloat[b * yStride + yIdx];
            cpuLogical[b * yCount + i] = goldenFloat[b * yStride + yIdx];
        }
    }

    aclDataType outDtype = ACL_FLOAT;
    if (p.dtype == 0 || p.dtype == 2) outDtype = ACL_FLOAT16;
    else if (p.dtype == 3 || p.dtype == 4) outDtype = ACL_BF16;

    VerifyConfig cfg;
    applyMixedTolerance(cfg, outDtype, cpuLogical.data(), cpuLogical.size());
    EXPECT_TRUE(Verifier::verifyVector(
        npuLogical.data(), cpuLogical.data(), static_cast<size_t>(p.batchCount) * yCount, 1, cfg, p.caseName));
}
