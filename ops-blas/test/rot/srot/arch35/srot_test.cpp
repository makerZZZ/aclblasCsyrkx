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
#include <algorithm>
#include <string>
#include <vector>

#include "fill.h"
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "srot_param.h"
#include "srot_golden.h"
#include "srot_npu_wrapper.h"

class SrotArch35Test : public BlasTest<SrotParam> {};

// Null handle test — separate TEST_F (not in CSV).
TEST_F(SrotArch35Test, NullHandle)
{
    float x[5] = {1, 2, 3, 4, 5};
    float y[5] = {6, 7, 8, 9, 10};
    aclblasStatus_t ret = aclblasSrot_npu(nullptr, 5, x, 1, y, 1, 0.6f, 0.8f);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SrotArch35Test, NullHandleTakesPrecedenceOverQuickReturn)
{
    EXPECT_EQ(aclblasSrot(nullptr, 0, nullptr, 1, nullptr, 1, nullptr, nullptr), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SrotArch35Test, QuickReturnSkipsDataPointerValidation)
{
    EXPECT_EQ(
        aclblasSrot(SrotArch35Test::handle_, 0, nullptr, 1, nullptr, 1, nullptr, nullptr), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasSrot(SrotArch35Test::handle_, -1, nullptr, 1, nullptr, 1, nullptr, nullptr), ACLBLAS_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    Srot, SrotArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SrotParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SrotParam>);

// Helper: build a host buffer matching the stride access pattern.
// Continuous (|inc|==1) uses makeBlasArray (size = n); strided uses makeBlasStrided
// (size = (n-1)*|inc|+1, elements laid out at the stride positions).
//
// inc==0 special case: netlib srot.f / cblas_srot repeatedly access element 0 (serial
// accumulation, only x[0]/y[0] is touched). The NPU kernel matches this netlib
// semantics (empirically verified: only element 0 is rotated in-place N times).
// However the NPU wrapper's element span (SrotBufElems) coerces absInc 0 -> 1 before
// sizing the device buffer, yielding a span of n elements. We therefore size the host
// buffer to n (via makeBlasArray) whenever inc==0 so the H2D copy does not read past
// the host buffer end, while the golden (cblas_srot with inc==0) only mutates element 0.
static inline std::vector<float> MakeSrotBuffer(int n, int inc, const BlasFillMode& fill, uint32_t seed)
{
    int absInc = std::abs(inc);
    if (absInc <= 1) {
        // |inc|==1 (contiguous) AND inc==0 (device span coerces to n elements).
        return makeBlasArray(static_cast<int64_t>(n), fill, seed);
    }
    return makeBlasStrided(n, inc, fill, seed);
}

TEST_P(SrotArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // Step 1: generate host data (x/y are both input and output — in-place).
    std::vector<float> xHost = MakeSrotBuffer(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> yHost = MakeSrotBuffer(p.n, p.incy, p.y, p.randomSeed);

    float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float* yPtr = yHost.empty() ? nullptr : yHost.data();

    // Step 2: execute on NPU. csPtrMode ("host"/"device"/"mixed") tells the wrapper where to
    // materialize the c / s scalar pointers; the operator auto-detects each pointer's location.
    aclblasStatus_t ret =
        aclblasSrot_npu(SrotArch35Test::handle_, p.n, xPtr, p.incx, yPtr, p.incy, p.c, p.s, p.csPtrMode);

    // Step 3: check return code against expected.
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    // n <= 0 is a no-op (x/y untouched) — nothing to verify numerically.
    if (p.n <= 0)
        return;

    // Step 4: compute CPU golden on a fresh copy of the original data.
    std::vector<float> goldenX = MakeSrotBuffer(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> goldenY = MakeSrotBuffer(p.n, p.incy, p.y, p.randomSeed);
    aclblasSrot_cpu(SrotArch35Test::handle_, p.n, goldenX.data(), p.incx, goldenY.data(), p.incy, &p.c, &p.s);
    // Step 5: verify both x and y. Precision mode is chosen by data characteristics:
    //   - NaN inputs (TC_L1_20): rotation propagates NaN; verify each element is NaN
    //     directly (std::isnan), bypassing Verifier (NaN rel-err is undefined).
    //   - INF inputs with identity rotation c==1&&s==0 (TC_L1_19): result must equal
    //     the input bit-for-bit; verify EXACT.
    //   - Otherwise: FP32 mixed tolerance.
    int absIncX = std::max(1, std::abs(p.incx));
    int absIncY = std::max(1, std::abs(p.incy));

    auto containsNaN = [](const float* buf, int n, int inc) -> bool {
        int step = std::max(1, std::abs(inc));
        for (int i = 0; i < n; i++) {
            if (std::isnan(buf[static_cast<int64_t>(i) * step]))
                return true;
        }
        return false;
    };

    if (containsNaN(xPtr, p.n, p.incx) || containsNaN(yPtr, p.n, p.incy)) {
        // NaN propagation: every output element must be NaN (c*x+s*y with any NaN -> NaN).
        bool allNanX = true;
        bool allNanY = true;
        for (int i = 0; i < p.n; i++) {
            if (!std::isnan(xPtr[static_cast<int64_t>(i) * absIncX]))
                allNanX = false;
            if (!std::isnan(yPtr[static_cast<int64_t>(i) * absIncY]))
                allNanY = false;
        }
        std::cout << "[" << p.caseName << "] NaN-propagation check: x_allNan=" << allNanX << " y_allNan=" << allNanY
                  << std::endl;
        EXPECT_TRUE(allNanX && allNanY);
        return;
    }

    bool identityRotation = (p.c == 1.0f && p.s == 0.0f);
    auto containsInf = [](const float* buf, int n, int inc) -> bool {
        int step = std::max(1, std::abs(inc));
        for (int i = 0; i < n; i++) {
            if (std::isinf(buf[static_cast<int64_t>(i) * step]))
                return true;
        }
        return false;
    };

    if (identityRotation && (containsInf(xPtr, p.n, p.incx) || containsInf(yPtr, p.n, p.incy))) {
        // Identity rotation (c==1, s==0) must not short-circuit: x' = 1*x + 0*y = x,
        // y' = 1*y - 0*x = y. Verify bit-exact equality against the golden (which is
        // the original input, since cblas_srot with c=1 s=0 also returns input unchanged).
        VerifyConfig cfg;
        cfg.mode = PrecisionMode::EXACT;
        EXPECT_TRUE(Verifier::verifyVector(
            xPtr, goldenX.data(), static_cast<size_t>(p.n), static_cast<int64_t>(absIncX), cfg, p.caseName + "_x"));
        EXPECT_TRUE(Verifier::verifyVector(
            yPtr, goldenY.data(), static_cast<size_t>(p.n), static_cast<int64_t>(absIncY), cfg, p.caseName + "_y"));
        return;
    }

    VerifyConfig xCfg;
    applyMixedTolerance(xCfg, ACL_FLOAT, goldenX.data(), static_cast<size_t>(p.n));
    VerifyConfig yCfg;
    applyMixedTolerance(yCfg, ACL_FLOAT, goldenY.data(), static_cast<size_t>(p.n));

    EXPECT_TRUE(Verifier::verifyVector(
        xPtr, goldenX.data(), static_cast<size_t>(p.n), static_cast<int64_t>(absIncX), xCfg, p.caseName + "_x"));
    EXPECT_TRUE(Verifier::verifyVector(
        yPtr, goldenY.data(), static_cast<size_t>(p.n), static_cast<int64_t>(absIncY), yCfg, p.caseName + "_y"));
}
