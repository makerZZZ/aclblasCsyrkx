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
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "csyr2_golden.h"

namespace {
using csyr2_test::AccuracyConfig;
using csyr2_test::Accumulate;
using csyr2_test::Csyr2Golden;
using csyr2_test::LogicalOffset;
using csyr2_test::Merge;
using csyr2_test::UlpForFloat;

aclblasComplex C(float real, float imag)
{
    return {real, imag};
}

TEST(Csyr2Golden, HandCheckUpperAndLowerAndDiagonalImaginary)
{
    const aclblasComplex alpha = C(1.0f, 1.0f);
    const std::vector<aclblasComplex> x = {C(1.0f, 2.0f), C(-1.0f, 1.0f)};
    const std::vector<aclblasComplex> y = {C(2.0f, -1.0f), C(3.0f, 2.0f)};
    const std::vector<aclblasComplex> initial = {C(1.0f, 1.0f), C(2.0f, 3.0f), C(2.0f, 3.0f), C(-1.0f, 4.0f)};

    std::vector<aclblasComplex> upper = initial;
    Csyr2Golden(ACLBLAS_UPPER, 2, alpha, x.data(), 1, y.data(), 1, upper.data(), 2);
    EXPECT_FLOAT_EQ(upper[0].real, 3.0f);
    EXPECT_FLOAT_EQ(upper[0].imag, 15.0f);
    EXPECT_TRUE(csyr2_test::BitwiseEqual(upper[1], initial[1]));
    EXPECT_FLOAT_EQ(upper[2].real, -11.0f);
    EXPECT_FLOAT_EQ(upper[2].imag, 12.0f);
    EXPECT_FLOAT_EQ(upper[3].real, -13.0f);
    EXPECT_FLOAT_EQ(upper[3].imag, -4.0f);

    std::vector<aclblasComplex> lower = initial;
    Csyr2Golden(ACLBLAS_LOWER, 2, alpha, x.data(), 1, y.data(), 1, lower.data(), 2);
    EXPECT_FLOAT_EQ(lower[0].real, 3.0f);
    EXPECT_FLOAT_EQ(lower[0].imag, 15.0f);
    EXPECT_FLOAT_EQ(lower[1].real, -11.0f);
    EXPECT_FLOAT_EQ(lower[1].imag, 12.0f);
    EXPECT_TRUE(csyr2_test::BitwiseEqual(lower[2], initial[2]));
    EXPECT_FLOAT_EQ(lower[3].real, -13.0f);
    EXPECT_FLOAT_EQ(lower[3].imag, -4.0f);
}

TEST(Csyr2Golden, NegativeAndPositiveStridesAreEquivalent)
{
    constexpr int n = 4;
    const aclblasComplex alpha = C(0.5f, -0.25f);
    // The two physical layouts encode the same logical vector: positive inc
    // walks offsets 0,2,4,6; negative inc walks 6,4,2,0 from the same base.
    const std::vector<aclblasComplex> xPositive = {
        C(1, 2), C(99, 99), C(3, 4), C(98, 98), C(5, 6), C(97, 97), C(7, 8)};
    const std::vector<aclblasComplex> xNegative = {
        C(7, 8), C(99, 99), C(5, 6), C(98, 98), C(3, 4), C(97, 97), C(1, 2)};
    const std::vector<aclblasComplex> yPositive = {
        C(2, 1), C(96, 96), C(4, 3), C(95, 95), C(6, 5), C(94, 94), C(8, 7)};
    const std::vector<aclblasComplex> yNegative = {
        C(8, 7), C(96, 96), C(6, 5), C(95, 95), C(4, 3), C(94, 94), C(2, 1)};
    const auto xLogical = csyr2_test::MakeLogicalVector(xPositive, n, 2);
    const auto yLogical = csyr2_test::MakeLogicalVector(yPositive, n, 2);

    std::vector<aclblasComplex> aPositive(16, C(1, -2));
    std::vector<aclblasComplex> aNegative(16, C(1, -2));
    Csyr2Golden(ACLBLAS_UPPER, n, alpha, xPositive.data(), 2, yPositive.data(), 2, aPositive.data(), 4);
    Csyr2Golden(ACLBLAS_UPPER, n, alpha, xNegative.data(), -2, yNegative.data(), -2, aNegative.data(), 4);

    for (int col = 0; col < n; ++col) {
        for (int row = 0; row <= col; ++row) {
            const int64_t idx = static_cast<int64_t>(col) * 4 + row;
            EXPECT_FLOAT_EQ(aPositive[idx].real, aNegative[idx].real);
            EXPECT_FLOAT_EQ(aPositive[idx].imag, aNegative[idx].imag);
        }
    }
    EXPECT_EQ(xLogical.size(), static_cast<size_t>(n));
    EXPECT_EQ(yLogical.size(), static_cast<size_t>(n));
}

TEST(Csyr2Golden, NoOpLeavesEveryBitUnchangedIncludingNaN)
{
    const aclblasComplex qnan = C(std::numeric_limits<float>::quiet_NaN(), -0.0f);
    const std::vector<aclblasComplex> x = {qnan};
    const std::vector<aclblasComplex> y = {qnan};
    std::vector<aclblasComplex> a = {C(-0.0f, std::numeric_limits<float>::quiet_NaN())};
    const auto before = a;

    Csyr2Golden(ACLBLAS_UPPER, 1, C(0.0f, 0.0f), x.data(), 1, y.data(), 1, a.data(), 1);
    EXPECT_TRUE(csyr2_test::BitwiseEqual(a[0], before[0]));

    Csyr2Golden(ACLBLAS_UPPER, 0, C(1.0f, 2.0f), nullptr, 1, nullptr, 1, a.data(), 1);
    EXPECT_TRUE(csyr2_test::BitwiseEqual(a[0], before[0]));
}

TEST(Csyr2Golden, PaddingAndUntouchedTriangleAreNotModified)
{
    constexpr int n = 3;
    constexpr int lda = 5;
    const aclblasComplex alpha = C(1.0f, 0.0f);
    const std::vector<aclblasComplex> x = {C(1, 0), C(2, 0), C(3, 0)};
    const std::vector<aclblasComplex> y = {C(4, 0), C(5, 0), C(6, 0)};
    const aclblasComplex sentinel = C(-123.0f, 456.0f);
    std::vector<aclblasComplex> a(static_cast<size_t>(lda * n), sentinel);
    const auto before = a;

    Csyr2Golden(ACLBLAS_UPPER, n, alpha, x.data(), 1, y.data(), 1, a.data(), lda);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            const int64_t idx = static_cast<int64_t>(col) * lda + row;
            if (row > col) {
                EXPECT_TRUE(csyr2_test::BitwiseEqual(a[idx], before[idx]));
            }
        }
        for (int row = n; row < lda; ++row) {
            const int64_t idx = static_cast<int64_t>(col) * lda + row;
            EXPECT_TRUE(csyr2_test::BitwiseEqual(a[idx], before[idx]));
        }
    }
}

TEST(Csyr2Accuracy, MixedToleranceAndCapAreIndependent)
{
    AccuracyConfig cfg;
    csyr2_test::ComplexAccuracy stats;
    const float golden = 1.0f;
    const float matched = 1.0f + std::ldexp(1.0f, -15);
    Accumulate(stats, C(matched, matched), C(golden, golden), 0, cfg);
    EXPECT_EQ(stats.real.matched, 1u);
    EXPECT_EQ(stats.real.capFailCount, 0u);

    csyr2_test::ComplexAccuracy capFail;
    Accumulate(capFail, C(1.02f, 1.0f), C(1.0f, 1.0f), 3, cfg);
    EXPECT_EQ(capFail.real.capFailCount, 1u);
    EXPECT_FALSE(csyr2_test::ComponentPass(capFail.real, cfg));
}

TEST(Csyr2Accuracy, SpecialValuesNeverConsumeTheOnePercentBudget)
{
    AccuracyConfig cfg;
    csyr2_test::ComplexAccuracy stats;
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    Accumulate(stats, C(qnan, std::numeric_limits<float>::infinity()),
               C(qnan, -std::numeric_limits<float>::infinity()), 0, cfg);
    for (size_t i = 1; i < 100; ++i) {
        Accumulate(stats, C(1.0f, 1.0f), C(1.0f, 1.0f), i, cfg);
    }
    EXPECT_EQ(stats.real.specialMismatch, 0u);
    EXPECT_EQ(stats.imag.specialMismatch, 1u);
    EXPECT_FALSE(csyr2_test::ComponentPass(stats.imag, cfg));
}

TEST(Csyr2Accuracy, NaNAndSignedInfinityClassification)
{
    AccuracyConfig cfg;
    csyr2_test::ComplexAccuracy stats;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Accumulate(stats, C(nan, std::numeric_limits<float>::infinity()),
               C(nan, std::numeric_limits<float>::infinity()), 0, cfg);
    EXPECT_EQ(stats.real.matched, 1u);
    EXPECT_EQ(stats.imag.matched, 1u);

    csyr2_test::ComplexAccuracy mismatch;
    Accumulate(mismatch, C(std::numeric_limits<float>::infinity(), 0),
               C(-std::numeric_limits<float>::infinity(), 0), 0, cfg);
    EXPECT_EQ(mismatch.real.specialMismatch, 1u);
}

TEST(Csyr2Accuracy, BoundaryRatio99Passes98Fails)
{
    AccuracyConfig cfg;
    csyr2_test::ComplexAccuracy ninetyNine;
    for (size_t i = 0; i < 100; ++i) {
        const float actual = (i == 0) ? 1.001f : 1.0f;
        Accumulate(ninetyNine, C(actual, 1.0f), C(1.0f, 1.0f), i, cfg);
    }
    EXPECT_GE(ninetyNine.real.matchedRatio(), 0.99);
    EXPECT_TRUE(csyr2_test::ComponentPass(ninetyNine.real, cfg));

    csyr2_test::ComplexAccuracy ninetyEight;
    for (size_t i = 0; i < 100; ++i) {
        const float actual = (i < 2) ? 1.001f : 1.0f;
        Accumulate(ninetyEight, C(actual, 1.0f), C(1.0f, 1.0f), i, cfg);
    }
    EXPECT_LT(ninetyEight.real.matchedRatio(), 0.99);
    EXPECT_FALSE(csyr2_test::ComponentPass(ninetyEight.real, cfg));
}

TEST(Csyr2Accuracy, ULPHandlesZeroSubnormalAndPowerOfTwo)
{
    EXPECT_DOUBLE_EQ(UlpForFloat(0.0), std::ldexp(1.0, -149));
    EXPECT_DOUBLE_EQ(UlpForFloat(std::ldexp(1.0, -126)), std::ldexp(1.0, -149));
    EXPECT_DOUBLE_EQ(UlpForFloat(1.0), std::ldexp(1.0, -23));
    EXPECT_DOUBLE_EQ(UlpForFloat(2.0), std::ldexp(1.0, -22));
}

TEST(Csyr2Accuracy, StreamMergeEqualsSinglePass)
{
    AccuracyConfig cfg;
    csyr2_test::ComplexAccuracy whole;
    csyr2_test::ComplexAccuracy first;
    csyr2_test::ComplexAccuracy second;
    for (size_t i = 0; i < 64; ++i) {
        const float actual = 1.0f + static_cast<float>(i % 3) * std::ldexp(1.0f, -15);
        Accumulate(whole, C(actual, actual), C(1.0f, 1.0f), i, cfg);
        if (i < 31) {
            Accumulate(first, C(actual, actual), C(1.0f, 1.0f), i, cfg);
        } else {
            Accumulate(second, C(actual, actual), C(1.0f, 1.0f), i, cfg);
        }
    }
    Merge(first, second);
    EXPECT_EQ(first.real.total, whole.real.total);
    EXPECT_EQ(first.real.matched, whole.real.matched);
    EXPECT_DOUBLE_EQ(first.real.mereSum, whole.real.mereSum);
    EXPECT_DOUBLE_EQ(first.real.mare, whole.real.mare);
    EXPECT_EQ(first.real.worstIndex, whole.real.worstIndex);
}

TEST(Csyr2Golden, LogicalOffsetUses64BitForLargeStrideArithmetic)
{
    EXPECT_EQ(LogicalOffset(0, 3, -3), 6);
    EXPECT_EQ(LogicalOffset(2, 3, -3), 0);
    EXPECT_EQ(LogicalOffset(1, 3, 2000000000), 2000000000LL);
}

TEST(Csyr2Golden, InfinityPropagationMatchesObservedCublas)
{
    const float inf = std::numeric_limits<float>::infinity();
    const aclblasComplex alpha{0.5f, -0.25f};
    const std::vector<aclblasComplex> x{{inf, 1.0f}, {1.0f, 2.0f}};
    const std::vector<aclblasComplex> y{{2.0f, 1.0f}, {3.0f, 4.0f}};
    std::vector<aclblasComplex> upper(4, {0.5f, 0.75f});
    auto lower = upper;
    Csyr2Golden(ACLBLAS_UPPER, 2, alpha, x.data(), 1, y.data(), 1, upper.data(), 2);
    Csyr2Golden(ACLBLAS_LOWER, 2, alpha, x.data(), 1, y.data(), 1, lower.data(), 2);
    EXPECT_TRUE(std::isnan(upper[2].imag));
    EXPECT_EQ(inf, lower[1].imag);
}

} // namespace
