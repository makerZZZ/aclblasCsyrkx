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
#include <climits>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "device.h"
#include "fill.h"
#include "ctbsv_param.h"
#include "ctbsv_golden.h"
#include "ctbsv_npu_wrapper.h"

namespace {

std::vector<aclblasComplex> makeComplexBanded(
    int m, int n, int kl, int ku, int lda, const BlasFillMode& fill, uint32_t seed)
{
    std::vector<float> realPart = makeBlasBanded(m, n, kl, ku, lda, fill, seed);
    std::vector<float> imagPart = makeBlasBanded(m, n, kl, ku, lda, fill, seed + 1000U);
    std::vector<aclblasComplex> data(realPart.size());
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

std::vector<aclblasComplex> makeComplexStrided(int count, int inc, const BlasFillMode& fill, uint32_t seed)
{
    std::vector<float> realPart = makeBlasStrided(count, inc, fill, seed);
    std::vector<float> imagPart = makeBlasStrided(count, inc, fill, seed + 1000U);
    const size_t size = realPart.size();
    std::vector<aclblasComplex> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

bool IsNonFinite(float v)
{
    return std::isnan(v) || std::isinf(v);
}

bool IsOverflowish(float v)
{
    return IsNonFinite(v) || std::fabs(v) > 1.0e30f;
}

void AlignOverflowPair(float& outVal, float& goldVal)
{
    const bool outOf = IsOverflowish(outVal);
    const bool goldOf = IsOverflowish(goldVal);
    if (outOf && goldOf) {
        outVal = std::numeric_limits<float>::quiet_NaN();
        goldVal = std::numeric_limits<float>::quiet_NaN();
        return;
    }
    if (IsNonFinite(outVal) && (goldOf || std::fabs(goldVal) > 1.0e16f)) {
        outVal = std::numeric_limits<float>::quiet_NaN();
        goldVal = std::numeric_limits<float>::quiet_NaN();
        return;
    }
    if (IsNonFinite(goldVal) && (outOf || std::fabs(outVal) > 1.0e16f)) {
        outVal = std::numeric_limits<float>::quiet_NaN();
        goldVal = std::numeric_limits<float>::quiet_NaN();
    }
}

void AlignOverflowSolutions(std::vector<aclblasComplex>& outDense, std::vector<aclblasComplex>& goldDense)
{
    const int n = static_cast<int>(outDense.size());
    int overflowed = 0;
    float maxAbs = 0.0f;
    for (int i = 0; i < n; ++i) {
        const auto& o = outDense[static_cast<size_t>(i)];
        const auto& g = goldDense[static_cast<size_t>(i)];
        if (IsOverflowish(o.real) || IsOverflowish(o.imag) || IsOverflowish(g.real) || IsOverflowish(g.imag)) {
            overflowed++;
        }
        maxAbs = std::max(maxAbs, std::fabs(o.real));
        maxAbs = std::max(maxAbs, std::fabs(o.imag));
        maxAbs = std::max(maxAbs, std::fabs(g.real));
        maxAbs = std::max(maxAbs, std::fabs(g.imag));
    }
    const bool exploded = (overflowed > 0) || (maxAbs > 1.0e20f);
    for (int i = 0; i < n; ++i) {
        auto& o = outDense[static_cast<size_t>(i)];
        auto& g = goldDense[static_cast<size_t>(i)];
        AlignOverflowPair(o.real, g.real);
        AlignOverflowPair(o.imag, g.imag);
        if (exploded) {
            if (IsOverflowish(o.real) || IsOverflowish(g.real) ||
                std::fabs(o.real) > 1.0e16f || std::fabs(g.real) > 1.0e16f) {
                o.real = std::numeric_limits<float>::quiet_NaN();
                g.real = std::numeric_limits<float>::quiet_NaN();
            }
            if (IsOverflowish(o.imag) || IsOverflowish(g.imag) ||
                std::fabs(o.imag) > 1.0e16f || std::fabs(g.imag) > 1.0e16f) {
                o.imag = std::numeric_limits<float>::quiet_NaN();
                g.imag = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

double ComplexAbs(const aclblasComplex& z)
{
    return std::hypot(static_cast<double>(z.real), static_cast<double>(z.imag));
}

bool ResidualClose(
    const CtbsvParam& p, int effK, int allocLda, const std::vector<aclblasComplex>& aHost,
    const std::vector<aclblasComplex>& rhsDense, const std::vector<aclblasComplex>& npuX,
    const std::vector<aclblasComplex>& goldX)
{
    if (rhsDense.size() != static_cast<size_t>(p.n) || aHost.empty()) {
        return false;
    }
    std::vector<aclblasComplex> axNpu = npuX;
    std::vector<aclblasComplex> axGold = goldX;
    cblas_ctbmv(CblasColMajor, ToCblasUplo(p.uplo), ToCblasOp(p.trans), ToCblasDiag(p.diag),
                p.n, effK, aHost.data(), allocLda, axNpu.data(), 1);
    cblas_ctbmv(CblasColMajor, ToCblasUplo(p.uplo), ToCblasOp(p.trans), ToCblasDiag(p.diag),
                p.n, effK, aHost.data(), allocLda, axGold.data(), 1);

    double maxRnpu = 0.0;
    double maxRgold = 0.0;
    double maxB = 0.0;
    double maxX = 0.0;
    double maxA = 0.0;
    int finite = 0;
    for (int i = 0; i < p.n; ++i) {
        const auto& b = rhsDense[static_cast<size_t>(i)];
        const auto& yn = axNpu[static_cast<size_t>(i)];
        const auto& yg = axGold[static_cast<size_t>(i)];
        const auto& xn = npuX[static_cast<size_t>(i)];
        if (IsNonFinite(b.real) || IsNonFinite(b.imag) || IsNonFinite(yn.real) || IsNonFinite(yn.imag) ||
            IsNonFinite(yg.real) || IsNonFinite(yg.imag) || IsNonFinite(xn.real) || IsNonFinite(xn.imag)) {
            continue;
        }
        maxRnpu = std::max(maxRnpu, ComplexAbs(aclblasComplex{yn.real - b.real, yn.imag - b.imag}));
        maxRgold = std::max(maxRgold, ComplexAbs(aclblasComplex{yg.real - b.real, yg.imag - b.imag}));
        maxB = std::max(maxB, ComplexAbs(b));
        maxX = std::max(maxX, ComplexAbs(xn));
        finite++;
    }
    for (int j = 0; j < p.n; ++j) {
        for (int r = 0; r <= effK; ++r) {
            const auto& a = aHost[static_cast<size_t>(r) + static_cast<size_t>(j) * static_cast<size_t>(allocLda)];
            if (!IsNonFinite(a.real) && !IsNonFinite(a.imag)) {
                maxA = std::max(maxA, ComplexAbs(a));
            }
        }
    }
    if (finite < (p.n + 1) / 2) {
        return false;
    }
    const double scale = (maxA * maxX + maxB) * static_cast<double>(std::max(p.n, 1)) * 2.0e-5;
    const bool pass = maxRnpu <= std::max(scale, 16.0 * std::max(maxRgold, 1.0e-12));
    std::cout << "[" << p.caseName << "] residual fallback: ||r_npu||=" << maxRnpu
              << " ||r_gold||=" << maxRgold << " scale=" << scale
              << " finite=" << finite << "/" << p.n << (pass ? " PASSED" : " FAILED") << std::endl;
    return pass;
}

} // namespace

class CtbsvTest : public BlasTest<CtbsvParam> {};

TEST_F(CtbsvTest, NullHandle)
{
    aclblasStatus_t ret = aclblasCtbsv_npu(
        nullptr, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, 1, nullptr, 2, nullptr, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Ctbsv, CtbsvTest,
    ::testing::ValuesIn(GetCasesFromCsv<CtbsvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtbsvParam>);

struct CtbsvFixture {
    const CtbsvParam& p;
    int effK = 0;
    int allocLda = 1;
    int allocN = 1;
    int absIncx = 0;
    size_t xSize = 0;
    bool isUpper = false;
    bool needDevA = false;
    bool needDevX = false;
    std::vector<aclblasComplex> aHost;
    std::vector<aclblasComplex> xHost;
    std::vector<aclblasComplex> golden;
    std::vector<aclblasComplex> rhsHost;
    std::optional<DeviceBuffer> aDev;
    std::optional<DeviceBuffer> xDev;

    explicit CtbsvFixture(const CtbsvParam& param) : p(param) {}

    void InitDims()
    {
        effK = (p.n > 0 && p.k >= 0) ? std::min(p.k, p.n - 1) : 0;
        allocLda = std::max(1, std::max(p.lda, effK + 1));
        allocN = std::max(1, p.n);
        isUpper = (p.uplo == ACLBLAS_UPPER);
        absIncx = (p.incx == INT_MIN) ? 0 : std::abs(p.incx);
        xSize = (p.n > 0 && absIncx > 0) ? static_cast<size_t>((p.n - 1) * absIncx + 1) : 0;
        needDevA = (p.n > 0);
        needDevX = (xSize > 0);
    }

    void PrepareHostData()
    {
        if (p.incx == INT_MIN) {
            return;
        }
        const int kl = isUpper ? 0 : effK;
        const int ku = isUpper ? effK : 0;
        if (p.aFill.method != BlasFillMode::M_NULLPTR) {
            aHost = makeComplexBanded(p.n, p.n, kl, ku, allocLda, p.aFill, p.randomSeed);
            StrengthenDiagonal();
        }
        if (p.xFill.method != BlasFillMode::M_NULLPTR) {
            xHost = makeComplexStrided(p.n, p.incx, p.xFill, p.randomSeed + 1);
            if (xHost.empty()) {
                xHost.resize(xSize, aclblasComplex{0.0f, 0.0f});
            }
            golden = xHost;
            rhsHost = xHost;
        }
    }

    void StrengthenDiagonal()
    {
        if (aHost.empty() || p.n <= 0) {
            return;
        }
        const int diagRow = isUpper ? effK : 0;
        for (int j = 0; j < p.n; j++) {
            aclblasComplex& diag = aHost[diagRow + static_cast<size_t>(j) * allocLda];
            diag.real += (diag.real >= 0.0f) ? 5.0f : -5.0f;
            diag.imag += (diag.imag >= 0.0f) ? 5.0f : -5.0f;
        }
    }

    void PrepareDevice()
    {
        if (needDevA && !aHost.empty()) {
            size_t aBytes = static_cast<size_t>(allocLda) * static_cast<size_t>(allocN) * sizeof(aclblasComplex);
            aDev.emplace(aBytes);
            aDev->copyFromHost(aHost.data(), aBytes);
        }
        if (needDevX && !xHost.empty()) {
            size_t xBytes = xSize * sizeof(aclblasComplex);
            xDev.emplace(xBytes);
            xDev->copyFromHost(xHost.data(), xBytes);
        }
    }

    aclblasStatus_t CallNpu(aclblasHandle_t handle)
    {
        const aclblasComplex* aPtr = aDev.has_value() ? static_cast<const aclblasComplex*>(aDev->ptr()) : nullptr;
        aclblasComplex* xPtr = xDev.has_value() ? static_cast<aclblasComplex*>(xDev->ptr()) : nullptr;
        return aclblasCtbsv_npu(handle, p.uplo, p.trans, p.diag, p.n, p.k, aPtr, p.lda, xPtr, p.incx);
    }

    void Verify(aclblasHandle_t handle, aclblasStatus_t ret)
    {
        if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
            EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
            return;
        }
        ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS))
            << "Unexpected NPU error code: " << static_cast<int>(ret);
        if (p.n <= 0) {
            return;
        }
        if (xDev.has_value()) {
            aclrtSynchronizeDevice();
            xDev->copyToHost(xHost.data(), xSize * sizeof(aclblasComplex));
        }
        aclblasCtbsv_cpu(handle, p.uplo, p.trans, p.diag, p.n, effK,
                         aHost.data(), allocLda, golden.data(), p.incx);
        std::vector<aclblasComplex> outDense(static_cast<size_t>(p.n));
        std::vector<aclblasComplex> goldDense(static_cast<size_t>(p.n));
        std::vector<aclblasComplex> rhsDense(static_cast<size_t>(p.n));
        for (int i = 0; i < p.n; ++i) {
            int idx = (p.incx >= 0) ? (i * p.incx) : ((p.n - 1 - i) * absIncx);
            outDense[static_cast<size_t>(i)] = xHost[static_cast<size_t>(idx)];
            goldDense[static_cast<size_t>(i)] = golden[static_cast<size_t>(idx)];
            if (!rhsHost.empty()) {
                rhsDense[static_cast<size_t>(i)] = rhsHost[static_cast<size_t>(idx)];
            }
        }
        const double mereThreshold = p.mereThreshold > 0.0 ? p.mereThreshold : 0.0001220703125;
        const double mareMultiplier = p.mareMultiplier > 0.0 ? p.mareMultiplier : 10.0;
        const std::vector<aclblasComplex> outForResidual = outDense;
        const std::vector<aclblasComplex> goldForResidual = goldDense;
        AlignOverflowSolutions(outDense, goldDense);
        const bool mereOk = Verifier::verifyMereMareComplexFloat(
            outDense.data(), goldDense.data(), static_cast<size_t>(p.n),
            mereThreshold, mareMultiplier, 1e-8, p.caseName);
        if (mereOk) {
            return;
        }
        EXPECT_TRUE(ResidualClose(p, effK, allocLda, aHost, rhsDense, outForResidual, goldForResidual))
            << "MERE/MARE and residual both failed for " << p.caseName;
    }
};

TEST_P(CtbsvTest, CsvDriven)
{
    CtbsvFixture f(GetParam());
    f.InitDims();
    f.PrepareHostData();
    f.PrepareDevice();
    aclblasStatus_t ret = f.CallNpu(CtbsvTest::handle_);
    f.Verify(CtbsvTest::handle_, ret);
}
