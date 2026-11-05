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
#include <cstdint>
#include <cstring>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "gemv_strided_batched_param.h"
#include "gemv_strided_batched_golden.h"
#include "gemv_strided_batched_npu_wrapper.h"

// ============================================================
// FP16 conversion helpers (for dtype 0=HSH, 2=HSS)
// ============================================================
static uint16_t FloatToHalf(float val)
{
    uint32_t f; memcpy_s(&f, sizeof(f), &val, sizeof(f));
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x3FF;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        uint32_t m = (mant | 0x400u);
        uint32_t shift = static_cast<uint32_t>(1 - exp);
        uint32_t half = 1u << (shift - 1);
        uint32_t rounded = (m + half) >> shift;
        return static_cast<uint16_t>(sign | rounded);
    }
    if (exp >= 31) {
        uint32_t fp32Exp = (f >> 23) & 0xFF;
        if (fp32Exp == 0xFF && ((f & 0x7FFFFF) != 0)) {
            return static_cast<uint16_t>(sign | 0x7C00 | 0x200);
        }
        return static_cast<uint16_t>(sign | 0x7C00);
    }
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

static float HalfToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            float val = std::ldexp(static_cast<float>(mant), -24);
            uint32_t bits; memcpy_s(&bits, sizeof(bits), &val, sizeof(bits));
            f = (sign << 31) | (bits & 0x7FFFFFFFu);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result; memcpy_s(&result, sizeof(result), &f, sizeof(result));
    return result;
}

// ============================================================
// bf16 conversion helpers (for dtype 3=TST, 4=TSS)
// ============================================================
static uint16_t FloatToBfloat(float val)
{
    uint32_t f; memcpy_s(&f, sizeof(f), &val, sizeof(f));
    uint32_t lsb = (f >> 16) & 1;
    uint32_t roundingBias = 0x7FFFu + lsb;
    return static_cast<uint16_t>((f + roundingBias) >> 16);
}

static float BfloatToFloat(uint16_t b)
{
    uint32_t f = static_cast<uint32_t>(b) << 16;
    float r; memcpy_s(&r, sizeof(r), &f, sizeof(r));
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
// Dtype dispatch helpers
// ============================================================
namespace {
struct DtypeRunResult {
    std::vector<float> npuFloat;
    std::vector<float> goldenFloat;
    std::vector<float> aGolden;
    std::vector<float> xGolden;
    aclblasStatus_t ret = ACLBLAS_STATUS_SUCCESS;
};
}  // namespace

static DtypeRunResult RunDtypeS(aclblasHandle_t handle, const GemvStridedBatchedParam& p,
    const std::vector<float>& aFloat, const std::vector<float>& xFloat,
    const std::vector<float>& yFloat)
{
    DtypeRunResult r;
    r.aGolden = aFloat; r.xGolden = xFloat;
    r.npuFloat = yFloat; r.goldenFloat = yFloat;
    r.ret = aclblasSgemvStridedBatched_npu(
        handle, p.trans, p.m, p.n, &p.alpha,
        aFloat.data(), p.lda, p.strideA, xFloat.data(), p.incx, p.stridex,
        &p.beta, r.npuFloat.data(), p.incy, p.stridey, p.batchCount);
    return r;
}

static DtypeRunResult RunDtypeHSH(aclblasHandle_t handle, const GemvStridedBatchedParam& p,
    const std::vector<float>& aFloat, const std::vector<float>& xFloat,
    const std::vector<float>& yFloat)
{
    DtypeRunResult r;
    std::vector<uint16_t> aHalf, xHalf, yHalf;
    QuantizeToHalf(aFloat, aHalf); QuantizeToHalf(xFloat, xHalf); QuantizeToHalf(yFloat, yHalf);
    QuantizeRoundTripWithHalf(aFloat, r.aGolden); QuantizeRoundTripWithHalf(xFloat, r.xGolden);
    QuantizeRoundTripWithHalf(yFloat, r.goldenFloat);
    std::vector<uint16_t> yNpu = yHalf;
    r.ret = aclblasHSHgemvStridedBatched_npu(
        handle, p.trans, p.m, p.n, &p.alpha,
        aHalf.data(), p.lda, p.strideA, xHalf.data(), p.incx, p.stridex,
        &p.beta, yNpu.data(), p.incy, p.stridey, p.batchCount);
    ConvertHalfToFloat(yNpu, r.npuFloat);
    return r;
}

static DtypeRunResult RunDtypeHSS(aclblasHandle_t handle, const GemvStridedBatchedParam& p,
    const std::vector<float>& aFloat, const std::vector<float>& xFloat,
    const std::vector<float>& yFloat)
{
    DtypeRunResult r;
    std::vector<uint16_t> aHalf, xHalf;
    QuantizeToHalf(aFloat, aHalf); QuantizeToHalf(xFloat, xHalf);
    QuantizeRoundTripWithHalf(aFloat, r.aGolden); QuantizeRoundTripWithHalf(xFloat, r.xGolden);
    r.goldenFloat = yFloat; r.npuFloat = yFloat;
    r.ret = aclblasHSSgemvStridedBatched_npu(
        handle, p.trans, p.m, p.n, &p.alpha,
        aHalf.data(), p.lda, p.strideA, xHalf.data(), p.incx, p.stridex,
        &p.beta, r.npuFloat.data(), p.incy, p.stridey, p.batchCount);
    return r;
}

static DtypeRunResult RunDtypeTST(aclblasHandle_t handle, const GemvStridedBatchedParam& p,
    const std::vector<float>& aFloat, const std::vector<float>& xFloat,
    const std::vector<float>& yFloat)
{
    DtypeRunResult r;
    std::vector<uint16_t> aBf16, xBf16, yBf16;
    QuantizeToBf16(aFloat, aBf16); QuantizeToBf16(xFloat, xBf16); QuantizeToBf16(yFloat, yBf16);
    QuantizeRoundTripWithBf16(aFloat, r.aGolden); QuantizeRoundTripWithBf16(xFloat, r.xGolden);
    QuantizeRoundTripWithBf16(yFloat, r.goldenFloat);
    std::vector<uint16_t> yNpu = yBf16;
    r.ret = aclblasTSTgemvStridedBatched_npu(
        handle, p.trans, p.m, p.n, &p.alpha,
        aBf16.data(), p.lda, p.strideA, xBf16.data(), p.incx, p.stridex,
        &p.beta, yNpu.data(), p.incy, p.stridey, p.batchCount);
    ConvertBf16ToFloat(yNpu, r.npuFloat);
    return r;
}

static DtypeRunResult RunDtypeTSS(aclblasHandle_t handle, const GemvStridedBatchedParam& p,
    const std::vector<float>& aFloat, const std::vector<float>& xFloat,
    const std::vector<float>& yFloat)
{
    DtypeRunResult r;
    std::vector<uint16_t> aBf16, xBf16;
    QuantizeToBf16(aFloat, aBf16); QuantizeToBf16(xFloat, xBf16);
    QuantizeRoundTripWithBf16(aFloat, r.aGolden); QuantizeRoundTripWithBf16(xFloat, r.xGolden);
    r.goldenFloat = yFloat; r.npuFloat = yFloat;
    r.ret = aclblasTSSgemvStridedBatched_npu(
        handle, p.trans, p.m, p.n, &p.alpha,
        aBf16.data(), p.lda, p.strideA, xBf16.data(), p.incx, p.stridex,
        &p.beta, r.npuFloat.data(), p.incy, p.stridey, p.batchCount);
    return r;
}

static DtypeRunResult DispatchDtype(aclblasHandle_t handle, const GemvStridedBatchedParam& p,
    const std::vector<float>& aFloat, const std::vector<float>& xFloat,
    const std::vector<float>& yFloat)
{
    switch (p.dtype) {
    case 0: return RunDtypeHSH(handle, p, aFloat, xFloat, yFloat);
    case 1: return RunDtypeS(handle, p, aFloat, xFloat, yFloat);
    case 2: return RunDtypeHSS(handle, p, aFloat, xFloat, yFloat);
    case 3: return RunDtypeTST(handle, p, aFloat, xFloat, yFloat);
    case 4: return RunDtypeTSS(handle, p, aFloat, xFloat, yFloat);
    default: ADD_FAILURE() << "Unknown dtype: " << p.dtype; return {};
    }
}

static void FillBatchedVec(std::vector<float>& out, int batchCount, int vecCount, int inc,
    BlasFillMode fill, std::int64_t vecStride, int seed, int seedOff)
{
    if (fill.method == BlasFillMode::M_NULLPTR) return;
    for (int b = 0; b < batchCount; b++) {
        auto batch = makeBlasStrided(vecCount, inc, fill, seed + seedOff + b);
        size_t off = static_cast<size_t>(b) * vecStride;
        size_t len = std::min(out.size() - off, batch.size());
        for (size_t i = 0; i < len; i++) out[off + i] = batch[i];
    }
}

// ============================================================
// Test fixture
// ============================================================
class GemvStridedBatchedTest : public BlasTest<GemvStridedBatchedParam> {};

TEST_F(GemvStridedBatchedTest, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    std::vector<float> a(64, 1.0f);
    std::vector<float> x(8, 1.0f);
    std::vector<float> y(8, 0.0f);
    aclblasStatus_t ret = aclblasSgemvStridedBatched(
        nullptr, ACLBLAS_OP_N, 8, 8, &alpha, a.data(), 8, 64, x.data(), 1, 8, &beta, y.data(), 1, 8, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

namespace {
struct DeviceBuffersGuard {
    void *dA = nullptr, *dX = nullptr, *dY = nullptr;
    ~DeviceBuffersGuard() { CleanupDeviceBuffers(dA, dX, dY); }
};
}  // namespace

TEST_F(GemvStridedBatchedTest, NullAlpha)
{
    std::vector<float> a(16, 1.0f);
    std::vector<float> x(4, 1.0f);
    std::vector<float> y(4, 0.0f);
    DeviceBuffersGuard bufs;
    ASSERT_EQ(CopyToDevice(&bufs.dA, a.data(), a.size() * sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(CopyToDevice(&bufs.dX, x.data(), x.size() * sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(CopyToDevice(&bufs.dY, y.data(), y.size() * sizeof(float)), ACL_SUCCESS);
    aclblasStatus_t ret = aclblasSgemvStridedBatched(
        GemvStridedBatchedTest::handle_, ACLBLAS_OP_N, 4, 4, nullptr,
        static_cast<const float*>(bufs.dA), 4, 16, static_cast<const float*>(bufs.dX), 1, 4,
        nullptr, static_cast<float*>(bufs.dY), 1, 4, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

TEST_F(GemvStridedBatchedTest, NullBeta)
{
    float alpha = 1.0f;
    std::vector<float> a(16, 1.0f);
    std::vector<float> x(4, 1.0f);
    std::vector<float> y(4, 0.0f);
    DeviceBuffersGuard bufs;
    ASSERT_EQ(CopyToDevice(&bufs.dA, a.data(), a.size() * sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(CopyToDevice(&bufs.dX, x.data(), x.size() * sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(CopyToDevice(&bufs.dY, y.data(), y.size() * sizeof(float)), ACL_SUCCESS);
    aclblasStatus_t ret = aclblasSgemvStridedBatched(
        GemvStridedBatchedTest::handle_, ACLBLAS_OP_N, 4, 4, &alpha,
        static_cast<const float*>(bufs.dA), 4, 16, static_cast<const float*>(bufs.dX), 1, 4,
        nullptr, static_cast<float*>(bufs.dY), 1, 4, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

INSTANTIATE_TEST_SUITE_P(
    GemvStridedBatched, GemvStridedBatchedTest,
    ::testing::ValuesIn(GetCasesFromCsv<GemvStridedBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GemvStridedBatchedParam>);

// ============================================================
// Verification: de-stride y and compare using MIXED_TOLERANCE
// ============================================================
static void VerifyGemvStridedBatchedOutput(
    const GemvStridedBatchedParam& p, int yCount,
    const std::vector<float>& npuFloat, const std::vector<float>& goldenFloat)
{
    if (yCount == 0 || p.batchCount == 0)
        return;

    const int absIncy = std::abs(p.incy);
    const size_t yStride = static_cast<size_t>(p.stridey);
    std::vector<float> npuLogical(static_cast<size_t>(p.batchCount) * yCount);
    std::vector<float> cpuLogical(static_cast<size_t>(p.batchCount) * yCount);
    for (int b = 0; b < p.batchCount; b++) {
        for (int i = 0; i < yCount; i++) {
            int yIdx = (p.incy > 0) ? (i * p.incy) : ((yCount - 1 - i) * absIncy);
            npuLogical[static_cast<size_t>(b) * yCount + i] = npuFloat[static_cast<size_t>(b) * yStride + yIdx];
            cpuLogical[static_cast<size_t>(b) * yCount + i] = goldenFloat[static_cast<size_t>(b) * yStride + yIdx];
        }
    }

    aclDataType dtype = ACL_FLOAT;
    switch (p.dtype) {
    case 0: dtype = ACL_FLOAT16; break;
    case 3: dtype = ACL_BF16;  break;
    default: dtype = ACL_FLOAT; break;
    }
    VerifyConfig cfg;
    applyMixedTolerance(cfg, dtype, cpuLogical.data(), cpuLogical.size());
    EXPECT_TRUE(Verifier::verifyVector(
        npuLogical.data(), cpuLogical.data(), static_cast<size_t>(p.batchCount) * yCount, 1, cfg, p.caseName));
}

// ============================================================
// Safe params + test data generation
// ============================================================
struct SafeGemvParams {
    int m, n, batchCount, lda, incx, incy, xCount, yCount;
    int64_t strideA, stridex, stridey;
};

static SafeGemvParams MakeSafeParams(const GemvStridedBatchedParam& p)
{
    SafeGemvParams s;
    s.m = std::max(1, std::abs(p.m));
    s.n = std::max(1, std::abs(p.n));
    s.batchCount = std::max(1, p.batchCount);
    s.lda = std::max(s.m, p.lda);
    s.incx = (p.incx == 0 || p.incx == INT32_MIN) ? 1 : p.incx;
    s.incy = (p.incy == 0 || p.incy == INT32_MIN) ? 1 : p.incy;
    const bool isTransN = (p.trans == ACLBLAS_OP_N);
    s.xCount = isTransN ? s.n : s.m;
    s.yCount = isTransN ? s.m : s.n;
    s.strideA = std::max(static_cast<int64_t>(s.lda) * s.n, p.strideA);
    s.stridex = std::max<int64_t>(1, p.stridex);
    s.stridey = std::max<int64_t>(1, p.stridey);
    return s;
}

static void GenGemvStridedData(
    const GemvStridedBatchedParam& p, const SafeGemvParams& s,
    std::vector<float>& aFloat, std::vector<float>& xFloat, std::vector<float>& yFloat)
{
    aFloat = makeBlasArray(
        static_cast<int64_t>(s.batchCount) * s.strideA, p.a, p.randomSeed);
    xFloat.assign(static_cast<size_t>(s.batchCount) * s.stridex, 0.0f);
    yFloat.assign(static_cast<size_t>(s.batchCount) * s.stridey, 0.0f);
    FillBatchedVec(xFloat, s.batchCount, s.xCount, s.incx, p.x, s.stridex, p.randomSeed, 1);
    FillBatchedVec(yFloat, s.batchCount, s.yCount, s.incy, p.y, s.stridey, p.randomSeed, 2);
}

TEST_P(GemvStridedBatchedTest, CsvDriven)
{
    const auto& p = GetParam();
    auto s = MakeSafeParams(p);
    std::vector<float> aFloat, xFloat, yFloat;
    GenGemvStridedData(p, s, aFloat, xFloat, yFloat);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        const float* aPtr = (p.a.method == BlasFillMode::M_NULLPTR) ? nullptr : aFloat.data();
        const float* xPtr = (p.x.method == BlasFillMode::M_NULLPTR) ? nullptr : xFloat.data();
        std::vector<float> yNpu(yFloat);
        float* yPtr = (p.y.method == BlasFillMode::M_NULLPTR) ? nullptr : yNpu.data();
        aclblasStatus_t ret = aclblasSgemvStridedBatched_npu(
            handle_, p.trans, p.m, p.n, &p.alpha, aPtr, p.lda, p.strideA,
            xPtr, p.incx, p.stridex, &p.beta, yPtr, p.incy, p.stridey, p.batchCount);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    auto r = DispatchDtype(handle_, p, aFloat, xFloat, yFloat);
    ASSERT_EQ(r.ret, ACLBLAS_STATUS_SUCCESS);
    aclblasGemvStridedBatched_cpu(
        p.trans, p.m, p.n, p.alpha, r.aGolden.data(), p.lda, p.strideA,
        r.xGolden.data(), p.incx, p.stridex, p.beta,
        r.goldenFloat.data(), p.incy, p.stridey, p.batchCount);
    VerifyGemvStridedBatchedOutput(p, s.yCount, r.npuFloat, r.goldenFloat);
}
