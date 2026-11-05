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
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cgemv_param.h"
#include "cgemv_golden.h"
#include "cgemv_npu_wrapper.h"

// Generate a column-major complex matrix: m rows, n cols, column stride lda (complex units).
// Returns interleaved float vector of size lda * n * 2. Only the leading min(m, lda) rows
// of each column are filled (guards negative lda cases that never reach the kernel).
static std::vector<float> cgemvMakeComplexMatrixCM(
    int m, int n, int lda, const cgemv_test::FillMode& fill, uint32_t seed)
{
    if (fill.base.method == BlasFillMode::M_NULLPTR || m <= 0 || n <= 0 || lda <= 0) {
        return {};
    }
    const size_t storageSize = static_cast<size_t>(lda) * n * 2;
    std::vector<float> data(storageSize, 0.0f);

    std::mt19937 rngReal(seed ? seed : 42);
    std::mt19937 rngImag((seed ? seed : 42) + 2000);
    auto genReal = cgemv_test::createGenerator(fill, rngReal);
    auto genImag = cgemv_test::createGenerator(fill, rngImag);

    const int fillRows = std::min(m, lda);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < fillRows; i++) {
            size_t idx = (static_cast<size_t>(j) * lda + i) * 2;
            size_t flatIdx = static_cast<size_t>(j) * fillRows + i;
            data[idx] = genReal->at(flatIdx);
            data[idx + 1] = genImag->at(flatIdx);
        }
    }
    return data;
}

// Generate a strided complex vector with `count` logical elements and stride inc (complex units).
// Storage holds (count-1)*|inc|+1 complex elements as interleaved floats; negative inc places
// logical element i at storage position (count-1-i)*|inc| (Netlib reverse-traversal layout).
static std::vector<float> cgemvMakeComplexStrided(int count, int inc, const cgemv_test::FillMode& fill, uint32_t seed)
{
    if (fill.base.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    const uint64_t absInc = std::max<uint64_t>(1U, cgemvAbsStride(inc));
    const size_t storageCplx = (count > 0) ? (static_cast<size_t>(count - 1) * absInc + 1) : 1;
    std::vector<float> data(storageCplx * 2, 0.0f);
    if (count <= 0) {
        return data;
    }

    std::mt19937 rngReal(seed ? seed : 42);
    std::mt19937 rngImag((seed ? seed : 42) + 2000);
    auto genReal = cgemv_test::createGenerator(fill, rngReal);
    auto genImag = cgemv_test::createGenerator(fill, rngImag);

    for (int i = 0; i < count; i++) {
        uint64_t logicalOffset =
            (inc > 0) ? static_cast<uint64_t>(i) * absInc : static_cast<uint64_t>(count - 1 - i) * absInc;
        size_t idx = static_cast<size_t>(logicalOffset) * 2;
        data[idx] = genReal->at(i);
        data[idx + 1] = genImag->at(i);
    }
    return data;
}

// -- Test fixture ------------------------------------------------------------
class CgemvArch35Test : public BlasTest<CgemvParam> {
#if !defined(INSTANTIATE_TEST_SUITE_P)
public:
    // GoogleTest 1.8 invokes the legacy fixture lifecycle names.
    static void SetUpTestCase() { BlasTest<CgemvParam>::SetUpTestSuite(); }

    static void TearDownTestCase() { BlasTest<CgemvParam>::TearDownTestSuite(); }
#endif
};

struct CgemvLocalContext {
    aclrtStream stream = nullptr;
    aclblasHandle_t handle = nullptr;

    ~CgemvLocalContext()
    {
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
    }
};

// E01: handle = nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
TEST_F(CgemvArch35Test, NullHandle)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {0.0f, 0.0f};
    std::vector<float> a(8 * 8 * 2, 1.0f), x(8 * 2, 1.0f), y(8 * 2, 0.0f);
    aclblasStatus_t ret = aclblasCgemv(
        nullptr, ACLBLAS_OP_N, 8, 8, &alpha, reinterpret_cast<const aclblasComplex*>(a.data()), 8,
        reinterpret_cast<const aclblasComplex*>(x.data()), 1, &beta, reinterpret_cast<aclblasComplex*>(y.data()), 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// alpha=0, beta=1 is a true no-op: A, x, and y may all be null because the
// host returns before constructing a device launch.  The CSV path cannot
// compare a successful non-empty case whose y buffer is intentionally null.
TEST_F(CgemvArch35Test, QuickReturnNullDataPointers)
{
    aclblasComplex alpha = {0.0f, 0.0f}, beta = {1.0f, 0.0f};
    EXPECT_EQ(
        aclblasCgemv_npu(
            CgemvArch35Test::handle_, ACLBLAS_OP_C, 16, 17, &alpha, nullptr, 16, nullptr, -3, &beta, nullptr, -2),
        ACLBLAS_STATUS_SUCCESS);

    // Empty matrices return after validation as well; data pointers are not
    // dereferenced even when alpha/beta would otherwise require computation.
    alpha = {2.0f, -1.0f};
    beta = {0.5f, 0.25f};
    EXPECT_EQ(
        aclblasCgemv_npu(
            CgemvArch35Test::handle_, ACLBLAS_OP_N, 0, 17, &alpha, nullptr, 1, nullptr, 2, &beta, nullptr, -3),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasCgemv_npu(
            CgemvArch35Test::handle_, ACLBLAS_OP_T, 16, 0, &alpha, nullptr, 16, nullptr, -2, &beta, nullptr, 3),
        ACLBLAS_STATUS_SUCCESS);
}

// The public contract excludes only zero strides. INT32_MIN is valid too;
// length-one vectors exercise it without requiring an impractically sparse allocation.
TEST_F(CgemvArch35Test, ExtremeNegativeStrideLengthOne)
{
    constexpr int kMinStride = std::numeric_limits<int>::min();
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {0.0f, 0.0f};
    const aclblasComplex a = {2.0f, 3.0f};
    const aclblasComplex x = {4.0f, 5.0f};

    for (aclblasOperation_t trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
        aclblasComplex y = {0.0f, 0.0f};
        ASSERT_EQ(
            aclblasCgemv_npu(
                CgemvArch35Test::handle_, trans, 1, 1, &alpha, &a, 1, &x, kMinStride, &beta, &y, kMinStride),
            ACLBLAS_STATUS_SUCCESS);
        if (trans == ACLBLAS_OP_C) {
            EXPECT_FLOAT_EQ(y.real, 23.0f);
            EXPECT_FLOAT_EQ(y.imag, -2.0f);
        } else {
            EXPECT_FLOAT_EQ(y.real, -7.0f);
            EXPECT_FLOAT_EQ(y.imag, 22.0f);
        }
    }
}

// A/x accept the full finite FP32 domain. Keep the direct T/C dot-product
// accumulation order from splitting cancellation terms into independent
// chains that can overflow before their finite sum is formed.
struct CgemvCancellationChecks {
    static constexpr int kM = 65;
    static constexpr int kN = 1;
    static constexpr int kLda = 65;
    static constexpr float kFiniteTreeH = 0x1p+25f;
    static constexpr float kProductH = 0x1p+100f;
    static constexpr int kWideM = 129;
    static constexpr int kWideN = 4097;
    static constexpr int kWideGuard = 8;
    static constexpr aclblasComplex kGuard = {12345.25f, -54321.5f};
    aclblasHandle_t handle;
    aclrtStream stream;
    std::vector<aclblasComplex> a;
    std::vector<aclblasComplex> x;
    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.0f, 0.0f};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    aclblasComplex yNpu = {nan, nan};
    aclblasComplex yCpu = {nan, nan};
    aclblasComplex aOne = {0.0f, 0.0f};
    aclblasComplex xOne = {0.0f, 0.0f};

    CgemvCancellationChecks(aclblasHandle_t inputHandle, aclrtStream inputStream)
        : handle(inputHandle),
          stream(inputStream),
          a(kLda * kN, aclblasComplex{0.0f, 0.0f}),
          x(kM, aclblasComplex{0.0f, 0.0f})
    {}

    void CheckOrderedAccumulation()
    {
        constexpr float kH = 0x1.8p+126f;
        constexpr float kExpectedImag = 0x1.8p+127f;
        constexpr int kRows[] = {0, 32, 64};
        constexpr float kSigns[] = {1.0f, -1.0f, 1.0f};
        for (int k = 0; k < 3; ++k) {
            a[kRows[k]] = {kH, kSigns[k] * kH};
            x[kRows[k]] = {1.0f, kSigns[k]};
        }
        ASSERT_EQ(
            aclblasCgemv_npu(handle, ACLBLAS_OP_T, kM, kN, &alpha, a.data(), kLda, x.data(), 1, &beta, &yNpu, 1),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            aclblasCgemv_scalar(handle, ACLBLAS_OP_T, kM, kN, &alpha, a.data(), kLda, x.data(), 1, &beta, &yCpu, 1),
            ACLBLAS_STATUS_SUCCESS);

        ASSERT_TRUE(std::isfinite(yCpu.real));
        ASSERT_TRUE(std::isfinite(yCpu.imag));
        EXPECT_FLOAT_EQ(yCpu.real, 0.0f);
        EXPECT_FLOAT_EQ(yCpu.imag, kExpectedImag);
        EXPECT_TRUE(std::isfinite(yNpu.real));
        EXPECT_TRUE(std::isfinite(yNpu.imag));
        EXPECT_FLOAT_EQ(yNpu.real, yCpu.real);
        EXPECT_FLOAT_EQ(yNpu.imag, yCpu.imag);
    }

    void CheckTreeCancellation(float expectedReal)
    {
        for (aclblasOperation_t trans : {ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            yNpu = {nan, nan};
            yCpu = {nan, nan};
            ASSERT_EQ(
                aclblasCgemv_npu(handle, trans, kM, kN, &alpha, a.data(), kLda, x.data(), 1, &beta, &yNpu, 1),
                ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(
                aclblasCgemv_scalar(handle, trans, kM, kN, &alpha, a.data(), kLda, x.data(), 1, &beta, &yCpu, 1),
                ACLBLAS_STATUS_SUCCESS);

            ASSERT_TRUE(std::isfinite(yCpu.real));
            ASSERT_TRUE(std::isfinite(yCpu.imag));
            EXPECT_FLOAT_EQ(yCpu.real, expectedReal);
            EXPECT_FLOAT_EQ(yCpu.imag, 0.0f);
            EXPECT_TRUE(std::isfinite(yNpu.real));
            EXPECT_TRUE(std::isfinite(yNpu.imag));
            EXPECT_FLOAT_EQ(yNpu.real, yCpu.real);
            EXPECT_FLOAT_EQ(yNpu.imag, yCpu.imag);
        }
    }

    void CheckLaneCancellation()
    {
        // A parallel tree reduction can combine same-sign terms before their
        // cancellation partners, changing a finite row-order result into a
        // non-finite value. Exercise both T and C with identical real inputs.
        constexpr float kLaneH = 0x1.8p+127f;
        std::fill(a.begin(), a.end(), aclblasComplex{0.0f, 0.0f});
        std::fill(x.begin(), x.end(), aclblasComplex{0.0f, 0.0f});
        constexpr float kLaneSigns[] = {-1.0f, 1.0f, -1.0f, 1.0f};
        for (int row = 0; row < 4; ++row) {
            a[row] = {kLaneSigns[row] * kLaneH, 0.0f};
            x[row] = {1.0f, 0.0f};
        }
        ASSERT_NO_FATAL_FAILURE(CheckTreeCancellation(0.0f));
    }

    void CheckFiniteTreeCancellation()
    {
        // Non-finite checks alone do not protect arithmetic order. This entirely
        // finite sequence is 1 in row order, while a cross-lane tree first rounds
        // (-2^25 + 1) back to -2^25 and then produces 0 after adding +2^25.
        // The absolute error of 1 is far outside the FLOAT32 acceptance limit.
        std::fill(a.begin(), a.end(), aclblasComplex{0.0f, 0.0f});
        std::fill(x.begin(), x.end(), aclblasComplex{0.0f, 0.0f});
        a[0] = {-kFiniteTreeH, 0.0f};
        a[1] = {kFiniteTreeH, 0.0f};
        a[2] = {1.0f, 0.0f};
        x[0] = {1.0f, 0.0f};
        x[1] = {1.0f, 0.0f};
        x[2] = {1.0f, 0.0f};
        ASSERT_NO_FATAL_FAILURE(CheckTreeCancellation(1.0f));
    }

    void CheckFastCancellation()
    {
        // The same finite witness must remain ordered at a shape which used to
        // enter the T/C warp-tree path.  Exercise every output, including the
        // first and final columns, with NaN-prefilled y so a partial write cannot
        // hide behind the value check.
        constexpr int kFastN = 64;
        std::vector<aclblasComplex> aFast(kM * kFastN, aclblasComplex{0.0f, 0.0f});
        std::vector<aclblasComplex> xFast(kM, aclblasComplex{0.0f, 0.0f});
        xFast[0] = {1.0f, 0.0f};
        xFast[1] = {1.0f, 0.0f};
        xFast[2] = {1.0f, 0.0f};
        for (int col = 0; col < kFastN; ++col) {
            aFast[static_cast<size_t>(col) * kM] = {-kFiniteTreeH, 0.0f};
            aFast[static_cast<size_t>(col) * kM + 1] = {kFiniteTreeH, 0.0f};
            aFast[static_cast<size_t>(col) * kM + 2] = {1.0f, 0.0f};
        }
        for (aclblasOperation_t trans : {ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            std::vector<aclblasComplex> yNpuFast(kFastN, aclblasComplex{nan, nan});
            std::vector<aclblasComplex> yCpuFast(kFastN, aclblasComplex{nan, nan});
            ASSERT_EQ(
                aclblasCgemv_npu(
                    handle, trans, kM, kFastN, &alpha, aFast.data(), kM, xFast.data(), 1, &beta, yNpuFast.data(), 1),
                ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(
                aclblasCgemv_scalar(
                    handle, trans, kM, kFastN, &alpha, aFast.data(), kM, xFast.data(), 1, &beta, yCpuFast.data(), 1),
                ACLBLAS_STATUS_SUCCESS);
            for (int col = 0; col < kFastN; ++col) {
                ASSERT_TRUE(std::isfinite(yCpuFast[col].real)) << "col=" << col;
                ASSERT_TRUE(std::isfinite(yCpuFast[col].imag)) << "col=" << col;
                EXPECT_FLOAT_EQ(yCpuFast[col].real, 1.0f) << "col=" << col;
                EXPECT_FLOAT_EQ(yCpuFast[col].imag, 0.0f) << "col=" << col;
                EXPECT_TRUE(std::isfinite(yNpuFast[col].real)) << "col=" << col;
                EXPECT_TRUE(std::isfinite(yNpuFast[col].imag)) << "col=" << col;
                EXPECT_FLOAT_EQ(yNpuFast[col].real, yCpuFast[col].real) << "col=" << col;
                EXPECT_FLOAT_EQ(yNpuFast[col].imag, yCpuFast[col].imag) << "col=" << col;
            }
        }
    }

    static void FillWideInputs(std::vector<aclblasComplex>& aWide, std::vector<aclblasComplex>& xWide)
    {
        for (int row = 0; row < kWideM; ++row) {
            xWide[row] = {static_cast<float>((row % 11) - 5) * 0x1p-5f, static_cast<float>((row % 7) - 3) * 0x1p-6f};
            for (int col = 0; col < kWideN; ++col) {
                aWide[static_cast<size_t>(col) * kWideM + row] = {
                    static_cast<float>(((row * 3 + col * 5) % 17) - 8) * 0x1p-7f,
                    static_cast<float>(((row * 7 + col * 2) % 19) - 9) * 0x1p-8f};
            }
        }
    }

    static void CheckWideOutput(
        aclblasOperation_t trans, const std::vector<aclblasComplex>& yNpuWide,
        const std::vector<aclblasComplex>& yCpuWide)
    {
        std::vector<float> npuRe(kWideN), npuIm(kWideN), cpuRe(kWideN), cpuIm(kWideN);
        for (int col = 0; col < kWideN; ++col) {
            npuRe[col] = yNpuWide[col].real;
            npuIm[col] = yNpuWide[col].imag;
            cpuRe[col] = yCpuWide[col].real;
            cpuIm[col] = yCpuWide[col].imag;
        }
        VerifyConfig cfgRe;
        applyMixedTolerance(cfgRe, ACL_FLOAT, cpuRe.data(), cpuRe.size());
        const char* realLabel = (trans == ACLBLAS_OP_T) ? "wide_fixed_mte_t_real" : "wide_fixed_mte_c_real";
        EXPECT_TRUE(Verifier::verifyVector(npuRe.data(), cpuRe.data(), npuRe.size(), 1, cfgRe, realLabel));
        VerifyConfig cfgIm;
        applyMixedTolerance(cfgIm, ACL_FLOAT, cpuIm.data(), cpuIm.size());
        const char* imagLabel = (trans == ACLBLAS_OP_T) ? "wide_fixed_mte_t_imag" : "wide_fixed_mte_c_imag";
        EXPECT_TRUE(Verifier::verifyVector(npuIm.data(), cpuIm.data(), npuIm.size(), 1, cfgIm, imagLabel));
        for (int g = 0; g < kWideGuard; ++g) {
            EXPECT_FLOAT_EQ(yNpuWide[kWideN + g].real, kGuard.real) << "guard=" << g;
            EXPECT_FLOAT_EQ(yNpuWide[kWideN + g].imag, kGuard.imag) << "guard=" << g;
        }
    }

    void CheckWideCancellation()
    {
        // On the 56-AIV acceptance target, thirteen supplied PF cases have
        // 65--74 columns per slab.  A 64-AIV development target does not reach
        // that residual loop with the supplied manifest, so synthesize one shape
        // that maps to chunkLen=65 on 64 AIVs and chunkLen=74 on 56 AIVs.  This
        // checks both T/C results under the official mixed tolerance, including
        // the second pass through warp 0 and a short final slab.  Guard elements
        // after y make an off-by-one residual store visible.
        std::vector<aclblasComplex> aWide(static_cast<size_t>(kWideM) * kWideN, aclblasComplex{0.0f, 0.0f});
        std::vector<aclblasComplex> xWide(kWideM, aclblasComplex{0.0f, 0.0f});
        FillWideInputs(aWide, xWide);
        DeviceBuffer dAWide(aWide.size() * sizeof(aclblasComplex));
        DeviceBuffer dXWide(xWide.size() * sizeof(aclblasComplex));
        dAWide.copyFromHost(aWide.data(), aWide.size() * sizeof(aclblasComplex));
        dXWide.copyFromHost(xWide.data(), xWide.size() * sizeof(aclblasComplex));
        for (aclblasOperation_t trans : {ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            std::vector<aclblasComplex> yNpuWide(kWideN + kWideGuard, kGuard);
            std::fill(yNpuWide.begin(), yNpuWide.begin() + kWideN, aclblasComplex{nan, nan});
            DeviceBuffer dYWide(yNpuWide.size() * sizeof(aclblasComplex));
            dYWide.copyFromHost(yNpuWide.data(), yNpuWide.size() * sizeof(aclblasComplex));
            ASSERT_EQ(
                aclblasCgemv(
                    handle, trans, kWideM, kWideN, &alpha, static_cast<const aclblasComplex*>(dAWide.ptr()), kWideM,
                    static_cast<const aclblasComplex*>(dXWide.ptr()), 1, &beta,
                    static_cast<aclblasComplex*>(dYWide.ptr()), 1),
                ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
            dYWide.copyToHost(yNpuWide.data(), yNpuWide.size() * sizeof(aclblasComplex));

            std::vector<aclblasComplex> yCpuWide(kWideN, aclblasComplex{0.0f, 0.0f});
            ASSERT_EQ(
                aclblasCgemv_cpu(
                    handle, trans, kWideM, kWideN, &alpha, aWide.data(), kWideM, xWide.data(), 1, &beta,
                    yCpuWide.data(), 1),
                ACLBLAS_STATUS_SUCCESS);
            CheckWideOutput(trans, yNpuWide, yCpuWide);
        }
    }

    static void ExpectProductBoundaryClass(aclblasOperation_t trans, const aclblasComplex& value)
    {
        // Keep the four real products and the following Sub/Add independently
        // rounded.  Contracting the complex product changes the required FP32
        // non-finite classes: T is (NaN,+Inf), while C is (+Inf,NaN).
        if (trans == ACLBLAS_OP_C) {
            EXPECT_TRUE(std::isinf(value.real));
            EXPECT_FALSE(std::signbit(value.real));
            EXPECT_TRUE(std::isnan(value.imag));
        } else {
            EXPECT_TRUE(std::isnan(value.real));
            EXPECT_TRUE(std::isinf(value.imag));
            EXPECT_FALSE(std::signbit(value.imag));
        }
    }

    void CheckProductRoutes()
    {
        // Cover N/T/C with both unit and non-unit input strides using the same
        // component-wise oracle.
        constexpr int kUbDim = 32;
        std::vector<aclblasComplex> aUb(kUbDim * kUbDim, aclblasComplex{0.0f, 0.0f});
        aUb[0] = {kProductH, kProductH};
        for (int routeIncx : {1, 2}) {
            std::vector<aclblasComplex> xRoute(
                static_cast<size_t>(kUbDim - 1) * routeIncx + 1, aclblasComplex{0.0f, 0.0f});
            xRoute[0] = {kProductH, kProductH};
            for (aclblasOperation_t trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                std::vector<aclblasComplex> yNpuRoute(kUbDim, aclblasComplex{nan, nan});
                std::vector<aclblasComplex> yCpuRoute(kUbDim, aclblasComplex{nan, nan});
                ASSERT_EQ(
                    aclblasCgemv_npu(
                        handle, trans, kUbDim, kUbDim, &alpha, aUb.data(), kUbDim, xRoute.data(), routeIncx, &beta,
                        yNpuRoute.data(), 1),
                    ACLBLAS_STATUS_SUCCESS);
                ASSERT_EQ(
                    aclblasCgemv_scalar(
                        handle, trans, kUbDim, kUbDim, &alpha, aUb.data(), kUbDim, xRoute.data(), routeIncx, &beta,
                        yCpuRoute.data(), 1),
                    ACLBLAS_STATUS_SUCCESS);
                ExpectProductBoundaryClass(trans, yCpuRoute[0]);
                ExpectProductBoundaryClass(trans, yNpuRoute[0]);
            }
        }
    }

    void CheckAlphaProductBoundary()
    {
        // Isolate the same four-product boundary in alpha*acc and beta*y.  Both
        // are part of the public FLOAT32 domain and must not be hidden behind a
        // dot-product-only repair.
        aOne = {kProductH, kProductH};
        xOne = {1.0f, 0.0f};
        alpha = {kProductH, kProductH};
        beta = {0.0f, 0.0f};
        yNpu = {nan, nan};
        yCpu = {nan, nan};
        ASSERT_EQ(
            aclblasCgemv_npu(handle, ACLBLAS_OP_N, 1, 1, &alpha, &aOne, 1, &xOne, 1, &beta, &yNpu, 1),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            aclblasCgemv_scalar(handle, ACLBLAS_OP_N, 1, 1, &alpha, &aOne, 1, &xOne, 1, &beta, &yCpu, 1),
            ACLBLAS_STATUS_SUCCESS);
        ExpectProductBoundaryClass(ACLBLAS_OP_N, yCpu);
        ExpectProductBoundaryClass(ACLBLAS_OP_N, yNpu);
    }

    void CheckBetaProductBoundary()
    {
        aOne = {0.0f, 0.0f};
        xOne = {0.0f, 0.0f};
        alpha = {1.0f, 0.0f};
        beta = {kProductH, kProductH};
        yNpu = {kProductH, kProductH};
        yCpu = yNpu;
        ASSERT_EQ(
            aclblasCgemv_npu(handle, ACLBLAS_OP_N, 1, 1, &alpha, &aOne, 1, &xOne, 1, &beta, &yNpu, 1),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            aclblasCgemv_scalar(handle, ACLBLAS_OP_N, 1, 1, &alpha, &aOne, 1, &xOne, 1, &beta, &yCpu, 1),
            ACLBLAS_STATUS_SUCCESS);
        ExpectProductBoundaryClass(ACLBLAS_OP_N, yCpu);
        ExpectProductBoundaryClass(ACLBLAS_OP_N, yNpu);
    }

    void CheckScaleProductBoundary()
    {
        // alpha=0 selects CgemvScale and must not read A/x.  Hard-check the same
        // separated-product class without asking the host CBLAS implementation to
        // accept null matrix/vector pointers.
        alpha = {0.0f, 0.0f};
        yNpu = {kProductH, kProductH};
        ASSERT_EQ(
            aclblasCgemv_npu(handle, ACLBLAS_OP_N, 1, 1, &alpha, nullptr, 1, nullptr, 1, &beta, &yNpu, 1),
            ACLBLAS_STATUS_SUCCESS);
        ExpectProductBoundaryClass(ACLBLAS_OP_N, yNpu);
    }

    void CheckNonfiniteShortcuts()
    {
        // Preserve the exact-one and exact-zero shortcuts around non-finite y.
        // beta=1 must not evaluate 0*Inf; beta=0 in scale-only mode must not read y.
        aOne = {0.0f, 0.0f};
        xOne = {0.0f, 0.0f};
        alpha = {1.0f, 0.0f};
        beta = {1.0f, 0.0f};
        yNpu = {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
        yCpu = yNpu;
        ASSERT_EQ(
            aclblasCgemv_npu(handle, ACLBLAS_OP_N, 1, 1, &alpha, &aOne, 1, &xOne, 1, &beta, &yNpu, 1),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            aclblasCgemv_scalar(handle, ACLBLAS_OP_N, 1, 1, &alpha, &aOne, 1, &xOne, 1, &beta, &yCpu, 1),
            ACLBLAS_STATUS_SUCCESS);
        EXPECT_TRUE(std::isinf(yCpu.real));
        EXPECT_FALSE(std::signbit(yCpu.real));
        EXPECT_TRUE(std::isinf(yCpu.imag));
        EXPECT_TRUE(std::signbit(yCpu.imag));
        EXPECT_FLOAT_EQ(yNpu.real, yCpu.real);
        EXPECT_FLOAT_EQ(yNpu.imag, yCpu.imag);

        alpha = {0.0f, 0.0f};
        beta = {0.0f, 0.0f};
        yNpu = {std::numeric_limits<float>::infinity(), nan};
        ASSERT_EQ(
            aclblasCgemv_npu(handle, ACLBLAS_OP_N, 1, 1, &alpha, nullptr, 1, nullptr, 1, &beta, &yNpu, 1),
            ACLBLAS_STATUS_SUCCESS);
        EXPECT_FLOAT_EQ(yNpu.real, 0.0f);
        EXPECT_FALSE(std::signbit(yNpu.real));
        EXPECT_FLOAT_EQ(yNpu.imag, 0.0f);
        EXPECT_FALSE(std::signbit(yNpu.imag));
    }
};

constexpr aclblasComplex CgemvCancellationChecks::kGuard;

TEST_F(CgemvArch35Test, FiniteCancellationOrderedAccumulation)
{
    CgemvCancellationChecks checks(CgemvArch35Test::handle_, CgemvArch35Test::stream_);
    ASSERT_NO_FATAL_FAILURE(checks.CheckOrderedAccumulation());
    ASSERT_NO_FATAL_FAILURE(checks.CheckLaneCancellation());
    ASSERT_NO_FATAL_FAILURE(checks.CheckFiniteTreeCancellation());
    ASSERT_NO_FATAL_FAILURE(checks.CheckFastCancellation());
    ASSERT_NO_FATAL_FAILURE(checks.CheckWideCancellation());
    ASSERT_NO_FATAL_FAILURE(checks.CheckProductRoutes());
    ASSERT_NO_FATAL_FAILURE(checks.CheckAlphaProductBoundary());
    ASSERT_NO_FATAL_FAILURE(checks.CheckBetaProductBoundary());
    ASSERT_NO_FATAL_FAILURE(checks.CheckScaleProductBoundary());
    ASSERT_NO_FATAL_FAILURE(checks.CheckNonfiniteShortcuts());
}

struct CgemvNSlabBoundaryShape {
    int m;
    int n;
    int lda;
};

static constexpr size_t CGEMV_BOUNDARY_GUARD_COUNT = 8U;
static constexpr aclblasComplex CGEMV_BOUNDARY_GUARD = {12345.25F, -54321.5F};

static void CgemvFillNSlabBoundary(
    const CgemvNSlabBoundaryShape& shape, std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& x)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::fill(a.begin() + CGEMV_BOUNDARY_GUARD_COUNT, a.end() - CGEMV_BOUNDARY_GUARD_COUNT, aclblasComplex{nan, nan});
    for (int col = 0; col < shape.n; ++col) {
        x[CGEMV_BOUNDARY_GUARD_COUNT + col] = {
            static_cast<float>(col % 9 - 4) * 0x1p-2F, static_cast<float>(col % 7 - 3) * 0x1p-2F};
        for (int row = 0; row < shape.m; ++row) {
            size_t index = CGEMV_BOUNDARY_GUARD_COUNT + static_cast<size_t>(col) * shape.lda + row;
            a[index] = {
                static_cast<float>((row + col) % 17 - 8) * 0x1p-3F,
                static_cast<float>((row + 2 * col) % 13 - 6) * 0x1p-4F};
        }
    }
}

static void CgemvCheckBoundaryReadOnly(DeviceBuffer& buffer, const std::vector<aclblasComplex>& original)
{
    std::vector<aclblasComplex> readBack(original.size());
    buffer.copyToHost(readBack.data(), readBack.size() * sizeof(aclblasComplex));
    EXPECT_EQ(std::memcmp(readBack.data(), original.data(), original.size() * sizeof(aclblasComplex)), 0);
}

static void CgemvCheckNSlabBoundaryOutput(
    const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& expected)
{
    for (size_t row = 0; row < expected.size(); ++row) {
        EXPECT_FLOAT_EQ(actual[CGEMV_BOUNDARY_GUARD_COUNT + row].real, expected[row].real) << "row=" << row;
        EXPECT_FLOAT_EQ(actual[CGEMV_BOUNDARY_GUARD_COUNT + row].imag, expected[row].imag) << "row=" << row;
    }
    for (size_t guard = 0; guard < CGEMV_BOUNDARY_GUARD_COUNT; ++guard) {
        for (size_t offset : {guard, CGEMV_BOUNDARY_GUARD_COUNT + expected.size() + guard}) {
            EXPECT_FLOAT_EQ(actual[offset].real, CGEMV_BOUNDARY_GUARD.real) << "guard=" << offset;
            EXPECT_FLOAT_EQ(actual[offset].imag, CGEMV_BOUNDARY_GUARD.imag) << "guard=" << offset;
        }
    }
}

static void CgemvCheckNSlabBoundary(const CgemvNSlabBoundaryShape& shape, aclblasHandle_t handle, aclrtStream stream)
{
    SCOPED_TRACE(::testing::Message() << "N MTE m=" << shape.m << " n=" << shape.n << " lda=" << shape.lda);
    constexpr size_t guards = 2U * CGEMV_BOUNDARY_GUARD_COUNT;
    std::vector<aclblasComplex> a(static_cast<size_t>(shape.lda) * shape.n + guards, CGEMV_BOUNDARY_GUARD);
    std::vector<aclblasComplex> x(shape.n + guards, CGEMV_BOUNDARY_GUARD);
    std::vector<aclblasComplex> y(shape.m + guards, CGEMV_BOUNDARY_GUARD);
    CgemvFillNSlabBoundary(shape, a, x);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::fill(y.begin() + CGEMV_BOUNDARY_GUARD_COUNT, y.end() - CGEMV_BOUNDARY_GUARD_COUNT, aclblasComplex{nan, nan});
    std::vector<aclblasComplex> expected(shape.m, aclblasComplex{nan, nan});
    DeviceBuffer dA(a.size() * sizeof(aclblasComplex));
    DeviceBuffer dX(x.size() * sizeof(aclblasComplex));
    DeviceBuffer dY(y.size() * sizeof(aclblasComplex));
    dA.copyFromHost(a.data(), a.size() * sizeof(aclblasComplex));
    dX.copyFromHost(x.data(), x.size() * sizeof(aclblasComplex));
    dY.copyFromHost(y.data(), y.size() * sizeof(aclblasComplex));
    const aclblasComplex alpha = {1.0F, 0.0F};
    const aclblasComplex beta = {0.0F, 0.0F};
    ASSERT_EQ(
        aclblasCgemv(
            handle, ACLBLAS_OP_N, shape.m, shape.n, &alpha,
            static_cast<const aclblasComplex*>(dA.ptr()) + CGEMV_BOUNDARY_GUARD_COUNT, shape.lda,
            static_cast<const aclblasComplex*>(dX.ptr()) + CGEMV_BOUNDARY_GUARD_COUNT, 1, &beta,
            static_cast<aclblasComplex*>(dY.ptr()) + CGEMV_BOUNDARY_GUARD_COUNT, 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    dY.copyToHost(y.data(), y.size() * sizeof(aclblasComplex));
    ASSERT_EQ(
        aclblasCgemv_cpu(
            handle, ACLBLAS_OP_N, shape.m, shape.n, &alpha, a.data() + CGEMV_BOUNDARY_GUARD_COUNT, shape.lda,
            x.data() + CGEMV_BOUNDARY_GUARD_COUNT, 1, &beta, expected.data(), 1),
        ACLBLAS_STATUS_SUCCESS);
    CgemvCheckNSlabBoundaryOutput(y, expected);
    CgemvCheckBoundaryReadOnly(dA, a);
    CgemvCheckBoundaryReadOnly(dX, x);
}

struct CgemvSegmentedBoundaryChecks {
    static constexpr int kM = 2049;
    static constexpr int kN = 1025;
    static constexpr int kLda = kM;
    static constexpr int kGuardCount = 8;
    static constexpr int kRows[] = {0, 1024, 1025, 2048};
    static constexpr aclblasComplex kXValues[] = {
        {0x1p+0f, 0x1p-1f},
        {-0x1p-1f, 0x1p+0f},
        {0x1p-2f, -0x1p+0f},
        {-0x1p+0f, -0x1p-2f},
    };
    static constexpr aclblasComplex kGuard = {12345.25f, -54321.5f};
    static_assert(kM * 2U * sizeof(float) == 16392U, "unexpected x payload size");
    static_assert((kM * 2U * sizeof(float)) % 32U != 0U, "x payload must exercise MTE padding");
    std::vector<aclblasComplex> a =
        std::vector<aclblasComplex>(static_cast<size_t>(kLda) * kN, aclblasComplex{0.0f, 0.0f});
    std::vector<aclblasComplex> x = std::vector<aclblasComplex>(kM, aclblasComplex{0.0f, 0.0f});

    void FillInputs()
    {
        for (size_t slot = 0; slot < sizeof(kRows) / sizeof(kRows[0]); ++slot) {
            x[kRows[slot]] = kXValues[slot];
        }

        auto makeAValue = [](int col, int slot) {
            int realNumerator = ((col * (2 * slot + 1) + 3 * slot) % 17) - 8;
            int imagNumerator = ((col * (3 * slot + 2) + 5 * slot) % 13) - 6;
            return aclblasComplex{
                static_cast<float>(realNumerator) * 0x1p-3f, static_cast<float>(imagNumerator) * 0x1p-4f};
        };
        for (int col = 0; col < kN; ++col) {
            for (size_t slot = 0; slot < sizeof(kRows) / sizeof(kRows[0]); ++slot) {
                a[static_cast<size_t>(col) * kLda + kRows[slot]] = makeAValue(col, static_cast<int>(slot));
            }
        }
    }

    aclblasComplex ExpectedAt(aclblasOperation_t trans, int col) const
    {
        // Every nonzero input is a small dyadic number.  Thus all four complex
        // products and both segment sums are exactly representable in binary32,
        // independent of the otherwise different reduction grouping.
        float expectedR = 0.0f;
        float expectedI = 0.0f;
        for (size_t slot = 0; slot < sizeof(kRows) / sizeof(kRows[0]); ++slot) {
            const aclblasComplex av = a[static_cast<size_t>(col) * kLda + kRows[slot]];
            const aclblasComplex xv = x[kRows[slot]];
            const float ai = (trans == ACLBLAS_OP_C) ? -av.imag : av.imag;
            const float termR = av.real * xv.real - ai * xv.imag;
            const float termI = av.real * xv.imag + ai * xv.real;
            expectedR += termR;
            expectedI += termI;
        }
        return aclblasComplex{expectedR, expectedI};
    }

    void Check(aclblasHandle_t handle, aclrtStream stream)
    {
        DeviceBuffer dA(a.size() * sizeof(aclblasComplex));
        DeviceBuffer dX(x.size() * sizeof(aclblasComplex));
        dA.copyFromHost(a.data(), a.size() * sizeof(aclblasComplex));
        dX.copyFromHost(x.data(), x.size() * sizeof(aclblasComplex));

        const aclblasComplex alpha = {1.0f, 0.0f};
        const aclblasComplex beta = {0.0f, 0.0f};
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (aclblasOperation_t trans : {ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            std::vector<aclblasComplex> y(kN + kGuardCount, kGuard);
            std::fill(y.begin(), y.begin() + kN, aclblasComplex{nan, nan});
            DeviceBuffer dY(y.size() * sizeof(aclblasComplex));
            dY.copyFromHost(y.data(), y.size() * sizeof(aclblasComplex));

            ASSERT_EQ(
                aclblasCgemv(
                    handle, trans, kM, kN, &alpha, static_cast<const aclblasComplex*>(dA.ptr()), kLda,
                    static_cast<const aclblasComplex*>(dX.ptr()), 1, &beta, static_cast<aclblasComplex*>(dY.ptr()), 1),
                ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
            dY.copyToHost(y.data(), y.size() * sizeof(aclblasComplex));

            for (int col = 0; col < kN; ++col) {
                const aclblasComplex expected = ExpectedAt(trans, col);
                EXPECT_FLOAT_EQ(y[col].real, expected.real) << "trans=" << static_cast<int>(trans) << " col=" << col;
                EXPECT_FLOAT_EQ(y[col].imag, expected.imag) << "trans=" << static_cast<int>(trans) << " col=" << col;
            }
            for (int guard = 0; guard < kGuardCount; ++guard) {
                EXPECT_FLOAT_EQ(y[kN + guard].real, kGuard.real)
                    << "trans=" << static_cast<int>(trans) << " guard=" << guard;
                EXPECT_FLOAT_EQ(y[kN + guard].imag, kGuard.imag)
                    << "trans=" << static_cast<int>(trans) << " guard=" << guard;
            }
        }
    }
};

constexpr int CgemvSegmentedBoundaryChecks::kRows[];
constexpr aclblasComplex CgemvSegmentedBoundaryChecks::kXValues[];
constexpr aclblasComplex CgemvSegmentedBoundaryChecks::kGuard;

// Path-local coverage for the transport-only T/C segmented-direct MTE variant.
// On 64 AIVs this shape has 61 chunks of at most 17 columns; on 56 AIVs it has
// 54 chunks of at most 19 columns.  Both geometries select segShift=1, while
// the odd m makes the two row segments 1025 and 1024 elements and leaves the
// 16392-byte x payload unaligned to the 32-byte MTE transfer boundary.
TEST_F(CgemvArch35Test, SegmentedDirectMtePathLocalBoundaries)
{
    // Exact dyadic products cover row tails, one/eight MTE tiles, the final
    // x broadcast, and the bounded-slab fallback without rounding ambiguity.
    for (const auto& shape :
         {CgemvNSlabBoundaryShape{513, 7168, 517}, {1025, 7119, 1027}, {769, 896, 773}, {513, 7169, 517}}) {
        ASSERT_NO_FATAL_FAILURE(CgemvCheckNSlabBoundary(shape, CgemvArch35Test::handle_, CgemvArch35Test::stream_));
    }
    // Aligned tall slabs cover the row threshold, padding and row tails,
    // the two-tile column limit, and its fallback on 56- and 64-AIV devices.
    for (const auto& shape :
         {CgemvNSlabBoundaryShape{2048, 1024, 2048}, {2049, 1025, 2052}, {2048, 1792, 2048}, {2050, 2049, 2052}}) {
        ASSERT_NO_FATAL_FAILURE(CgemvCheckNSlabBoundary(shape, CgemvArch35Test::handle_, CgemvArch35Test::stream_));
    }
    CgemvSegmentedBoundaryChecks checks;
    checks.FillInputs();
    ASSERT_NO_FATAL_FAILURE(checks.Check(CgemvArch35Test::handle_, CgemvArch35Test::stream_));
}

// Parallel reductions must not turn a finite Netlib-order result into Inf when
// a same-sign pair begins immediately after a candidate partition boundary.
static void CgemvFillPartitionBoundary(std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& x)
{
    constexpr int kLda = 512;
    constexpr float kH = 0x1.8p+127f;
    struct BoundaryCase {
        int row;
        int boundary;
        float sign;
    };
    constexpr BoundaryCase kOverflowCases[] = {
        {0, 8, 1.0f},  {1, 8, -1.0f},  {2, 10, 1.0f},   {3, 10, -1.0f},
        {4, 73, 1.0f}, {5, 73, -1.0f}, {10, 256, 1.0f}, {11, 256, -1.0f},
    };
    for (const auto& test : kOverflowCases) {
        const int cols[] = {test.boundary - 1, test.boundary, test.boundary + 1};
        const float signs[] = {-test.sign, test.sign, test.sign};
        for (int k = 0; k < 3; ++k) {
            a[static_cast<size_t>(cols[k]) * kLda + test.row] = {signs[k] * kH, 0.0f};
            x[cols[k]] = {1.0f, 0.0f};
        }
    }

    // Non-finite-only repair is insufficient.  In serial order 1+B-B is
    // exactly zero, but placing 1 in the preceding chunk and B,-B in the next
    // produces indistinguishable finite partials and the wrong value one.
    constexpr float kFiniteB = 0x1p+80f;
    constexpr BoundaryCase kFiniteCases[] = {
        {6, 8, 1.0f}, {7, 8, -1.0f}, {8, 10, 1.0f}, {9, 10, -1.0f}, {12, 256, 1.0f}, {13, 256, -1.0f},
    };
    for (const auto& test : kFiniteCases) {
        const int cols[] = {test.boundary - 1, test.boundary, test.boundary + 1};
        const float values[] = {test.sign, test.sign * kFiniteB, -test.sign * kFiniteB};
        for (int k = 0; k < 3; ++k) {
            a[static_cast<size_t>(cols[k]) * kLda + test.row] = {values[k], 0.0f};
            x[cols[k]] = {1.0f, 0.0f};
        }
    }
}

TEST_F(CgemvArch35Test, FiniteCancellationPartitionBoundaries)
{
    constexpr int kM = 512;
    constexpr int kN = 512;
    constexpr int kLda = 512;
    std::vector<aclblasComplex> a(kLda * kN, aclblasComplex{0.0f, 0.0f});
    std::vector<aclblasComplex> x(kN, aclblasComplex{0.0f, 0.0f});

    CgemvFillPartitionBoundary(a, x);
    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.0f, 0.0f};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<aclblasComplex> yNpu(kM, aclblasComplex{nan, nan});
    std::vector<aclblasComplex> yCpu(kM, aclblasComplex{nan, nan});
    ASSERT_EQ(
        aclblasCgemv_npu(
            CgemvArch35Test::handle_, ACLBLAS_OP_N, kM, kN, &alpha, a.data(), kLda, x.data(), 1, &beta, yNpu.data(), 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCgemv_scalar(
            CgemvArch35Test::handle_, ACLBLAS_OP_N, kM, kN, &alpha, a.data(), kLda, x.data(), 1, &beta, yCpu.data(), 1),
        ACLBLAS_STATUS_SUCCESS);

    for (int row = 0; row < kM; ++row) {
        ASSERT_TRUE(std::isfinite(yCpu[row].real)) << "CPU row=" << row;
        ASSERT_TRUE(std::isfinite(yCpu[row].imag)) << "CPU row=" << row;
        EXPECT_TRUE(std::isfinite(yNpu[row].real)) << "NPU row=" << row;
        EXPECT_TRUE(std::isfinite(yNpu[row].imag)) << "NPU row=" << row;
        EXPECT_FLOAT_EQ(yNpu[row].real, yCpu[row].real) << "row=" << row;
        EXPECT_FLOAT_EQ(yNpu[row].imag, yCpu[row].imag) << "row=" << row;
    }
}

static void CgemvVerifyStridedOutput(
    const std::vector<float>& yNpu, const std::vector<float>& yCpu, int count, int incy, const std::string& name)
{
    // De-stride output into logical order and split real/imag for separate verification
    const uint64_t absIncy = cgemvAbsStride(incy);
    std::vector<float> npuRe(count), npuIm(count), cpuRe(count), cpuIm(count);
    for (int i = 0; i < count; i++) {
        uint64_t logicalOffset =
            (incy > 0) ? static_cast<uint64_t>(i) * absIncy : static_cast<uint64_t>(count - 1 - i) * absIncy;
        size_t idx = static_cast<size_t>(logicalOffset) * 2;
        npuRe[i] = yNpu[idx];
        npuIm[i] = yNpu[idx + 1];
        cpuRe[i] = yCpu[idx];
        cpuIm[i] = yCpu[idx + 1];
    }

    // COMPLEX64 precision: real/imag parts judged separately under the FLOAT32
    // mixed-tolerance standard (rtol=2^-10, atol=2^-16, ratio>=0.99, max(1e-2, 32ULP))
    VerifyConfig cfgRe;
    applyMixedTolerance(cfgRe, ACL_FLOAT, cpuRe.data(), static_cast<size_t>(count));
    EXPECT_TRUE(
        Verifier::verifyVector(npuRe.data(), cpuRe.data(), static_cast<size_t>(count), 1, cfgRe, name + "_real"));

    VerifyConfig cfgIm;
    applyMixedTolerance(cfgIm, ACL_FLOAT, cpuIm.data(), static_cast<size_t>(count));
    EXPECT_TRUE(
        Verifier::verifyVector(npuIm.data(), cpuIm.data(), static_cast<size_t>(count), 1, cfgIm, name + "_imag"));
}

struct CgemvWorkspaceShape {
    int m, n, lda, incx, incy;
    float betaRe, betaIm;
    const char* name;
};

static void CgemvCheckSmallWorkspaceShape(
    const CgemvWorkspaceShape& sh, aclblasHandle_t handle, const cgemv_test::FillMode& fill)
{
    auto aHost = cgemvMakeComplexMatrixCM(sh.m, sh.n, sh.lda, fill, 20280001U);
    auto xHost = cgemvMakeComplexStrided(sh.n, sh.incx, fill, 20280002U);
    auto yHost = cgemvMakeComplexStrided(sh.m, sh.incy, fill, 20280003U);
    aclblasComplex alpha = {1.5f, -0.5f};
    aclblasComplex beta = {sh.betaRe, sh.betaIm};
    const auto* aPtr = reinterpret_cast<const aclblasComplex*>(aHost.data());
    const auto* xPtr = reinterpret_cast<const aclblasComplex*>(xHost.data());

    std::vector<float> yNpu = yHost;
    ASSERT_EQ(
        aclblasCgemv_npu(
            handle, ACLBLAS_OP_N, sh.m, sh.n, &alpha, aPtr, sh.lda, xPtr, sh.incx, &beta,
            reinterpret_cast<aclblasComplex*>(yNpu.data()), sh.incy),
        ACLBLAS_STATUS_SUCCESS)
        << sh.name;
    std::vector<float> yCpu = yHost;
    ASSERT_EQ(
        aclblasCgemv_cpu(
            handle, ACLBLAS_OP_N, sh.m, sh.n, &alpha, aPtr, sh.lda, xPtr, sh.incx, &beta,
            reinterpret_cast<aclblasComplex*>(yCpu.data()), sh.incy),
        ACLBLAS_STATUS_SUCCESS)
        << sh.name;

    CgemvVerifyStridedOutput(yNpu, yCpu, sh.m, sh.incy, sh.name);
}

// A small user workspace must not affect the trans=N implementation. Inputs
// that fit in UB and inputs that remain in GM are both covered.
TEST_F(CgemvArch35Test, SmallWorkspaceCompatibilityN)
{
    CgemvLocalContext context;
    ASSERT_EQ(aclrtCreateStream(&context.stream), ACL_SUCCESS);
    ASSERT_EQ(aclblasCreate(&context.handle), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasSetStream(context.handle, context.stream), ACLBLAS_STATUS_SUCCESS);
    constexpr size_t kTinyWs = 256;
    DeviceBuffer tinyWs(kTinyWs);
    ASSERT_EQ(aclblasSetWorkspace(context.handle, tinyWs.ptr(), kTinyWs), ACLBLAS_STATUS_SUCCESS);

    const CgemvWorkspaceShape shapes[] = {
        {4096, 64, 4096, 1, 1, 0.0f, 0.0f, "small_ws_nub_4096_64"},       // CgemvNUb
        {33, 300, 40, 1, 1, 1.0f, 0.0f, "small_ws_nub_33_300_ldpad"},     // CgemvNUb, lda padding
        {1000, 9000, 1000, 1, -2, 0.5f, 0.25f, "small_ws_ngm_1000_9000"}, // CgemvNGm (x exceeds UB)
    };
    const auto fill = cgemv_test::parseFill("RANDOM_NORM_5_5");
    for (const auto& sh : shapes) {
        ASSERT_NO_FATAL_FAILURE(CgemvCheckSmallWorkspaceShape(sh, context.handle, fill));
    }
}

// The ordered N512 path must remain available when the slab workspace is too small.
TEST_F(CgemvArch35Test, N512SmallWorkspace)
{
    constexpr int kDim = 512;
    constexpr size_t kWorkspaceBytes = 256;
    DeviceBuffer workspace(kWorkspaceBytes);
    const std::vector<uint8_t> guard(kWorkspaceBytes, 0x5A);
    workspace.copyFromHost(guard.data(), guard.size());
    CgemvLocalContext context;
    ASSERT_EQ(aclrtCreateStream(&context.stream), ACL_SUCCESS);
    ASSERT_EQ(aclblasCreate(&context.handle), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasSetStream(context.handle, context.stream), ACLBLAS_STATUS_SUCCESS);

    const auto fill = cgemv_test::parseFill("RANDOM_NORM_5_5");
    const auto a = cgemvMakeComplexMatrixCM(kDim, kDim, kDim, fill, 20260917U);
    const auto x = cgemvMakeComplexStrided(kDim, 1, fill, 20260918U);
    const auto initialY = cgemvMakeComplexStrided(kDim, 1, fill, 20260919U);
    const aclblasComplex alpha = {1.0f, 0.0f}, beta = {0.0f, 0.0f};
    const auto* aPtr = reinterpret_cast<const aclblasComplex*>(a.data());
    const auto* xPtr = reinterpret_cast<const aclblasComplex*>(x.data());
    auto expected = initialY;
    ASSERT_EQ(
        aclblasCgemv_cpu(
            context.handle, ACLBLAS_OP_N, kDim, kDim, &alpha, aPtr, kDim, xPtr, 1, &beta,
            reinterpret_cast<aclblasComplex*>(expected.data()), 1),
        ACLBLAS_STATUS_SUCCESS);

    const size_t workspaceSizes[] = {1, kWorkspaceBytes};
    for (size_t bytes : workspaceSizes) {
        SCOPED_TRACE(bytes);
        ASSERT_EQ(aclblasSetWorkspace(context.handle, workspace.ptr(), bytes), ACLBLAS_STATUS_SUCCESS);
        auto actual = initialY;
        ASSERT_EQ(
            aclblasCgemv_npu(
                context.handle, ACLBLAS_OP_N, kDim, kDim, &alpha, aPtr, kDim, xPtr, 1, &beta,
                reinterpret_cast<aclblasComplex*>(actual.data()), 1),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_NO_FATAL_FAILURE(CgemvVerifyStridedOutput(actual, expected, kDim, 1, "n512_small_workspace"));
        std::vector<uint8_t> after(kWorkspaceBytes);
        workspace.copyToHost(after.data(), after.size());
        EXPECT_EQ(after, guard);
    }
}

#if defined(INSTANTIATE_TEST_SUITE_P)
INSTANTIATE_TEST_SUITE_P(
    Cgemv, CgemvArch35Test, ::testing::ValuesIn(GetCasesFromCsv<CgemvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgemvParam>);
#else
// GoogleTest 1.8 uses the legacy spelling for parameterized-test registration.
INSTANTIATE_TEST_CASE_P(
    Cgemv, CgemvArch35Test, ::testing::ValuesIn(GetCasesFromCsv<CgemvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgemvParam>);
#endif

static void CgemvPrintPerformance(const CgemvParam& p, const CgemvPerfMeasurement& perf, int warmup, int iters)
{
    const char transChar = (p.trans == ACLBLAS_OP_N) ? 'N' : ((p.trans == ACLBLAS_OP_T) ? 'T' : 'C');
    std::printf(
        "CGEMV_PERF_RESULT case=%s device=%d trans=%c m=%d n=%d incx=%d incy=%d warmup=%d iters=%d "
        "avg_us=%.6f\n",
        p.caseName.c_str(), TEST_DEVICE_ID, transChar, p.m, p.n, p.incx, p.incy, warmup, iters, perf.avgUs);
    std::fflush(stdout);
}

TEST_P(CgemvArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    const bool isTransN = (p.trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? p.n : p.m;
    const int yCount = isTransN ? p.m : p.n;

    // Generate input data with distinct seeds for A, x, y (interleaved complex floats)
    auto aHost = cgemvMakeComplexMatrixCM(p.m, p.n, p.lda, p.a, p.randomSeed);
    auto xHost = cgemvMakeComplexStrided(xCount, p.incx, p.x, p.randomSeed + 1);
    auto yHost = cgemvMakeComplexStrided(yCount, p.incy, p.y, p.randomSeed + 2);

    aclblasComplex alpha = {p.alphaRe, p.alphaIm};
    aclblasComplex beta = {p.betaRe, p.betaIm};
    const aclblasComplex* alphaPtr = p.alphaNull ? nullptr : &alpha;
    const aclblasComplex* betaPtr = p.betaNull ? nullptr : &beta;
    const aclblasComplex* aPtr = aHost.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(aHost.data());
    const aclblasComplex* xPtr = xHost.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(xHost.data());

    // NPU path: copy y, run kernel, result lands in yNpu
    std::vector<float> yNpu = yHost;
    aclblasComplex* yNpuPtr = yNpu.empty() ? nullptr : reinterpret_cast<aclblasComplex*>(yNpu.data());
    aclblasStatus_t ret = aclblasCgemv_npu(
        CgemvArch35Test::handle_, p.trans, p.m, p.n, alphaPtr, aPtr, p.lda, xPtr, p.incx, betaPtr, yNpuPtr, p.incy);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    // CPU golden path: same inputs, cblas_cgemv (Netlib) reference
    std::vector<float> yCpu = yHost;
    aclblasComplex* yCpuPtr = yCpu.empty() ? nullptr : reinterpret_cast<aclblasComplex*>(yCpu.data());
    aclblasStatus_t cpuRet = aclblasCgemv_cpu(
        CgemvArch35Test::handle_, p.trans, p.m, p.n, alphaPtr, aPtr, p.lda, xPtr, p.incx, betaPtr, yCpuPtr, p.incy);
    ASSERT_EQ(cpuRet, ACLBLAS_STATUS_SUCCESS);

    if (yCount == 0) {
        return;
    }

    CgemvVerifyStridedOutput(yNpu, yCpu, yCount, p.incy, p.caseName);

    const char* perfEnabled = std::getenv("CGEMV_ENABLE_PERF");
    const bool measurePerformance = perfEnabled != nullptr && std::string(perfEnabled) == "1";
    if (p.caseName.rfind("TC_PF_", 0) == 0 && measurePerformance) {
        constexpr int kPerfWarmup = 30;
        constexpr int kPerfIters = 100;
        const CgemvPerfMeasurement perf = aclblasCgemv_npu_benchmark(
            CgemvArch35Test::handle_, CgemvArch35Test::stream_, p.trans, p.m, p.n, alphaPtr, aPtr, p.lda, xPtr, p.incx,
            betaPtr, reinterpret_cast<const aclblasComplex*>(yHost.data()), p.incy, kPerfWarmup, kPerfIters);
        ASSERT_EQ(perf.status, ACLBLAS_STATUS_SUCCESS);

        CgemvPrintPerformance(p, perf, kPerfWarmup, kPerfIters);
    }
}
