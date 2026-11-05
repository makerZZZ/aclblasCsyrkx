/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "device.h"
#include "trsv_param.h"
#include "trsv_golden.h"
#include "strsv_npu_wrapper.h"

class TrsvArch35Test : public BlasTest<TrsvParam> {};

TEST_F(TrsvArch35Test, NullHandle)
{
    // aclblasStrsv host checks handle first: null handle returns
    // ACLBLAS_STATUS_HANDLE_IS_NULLPTR.
    aclblasStatus_t ret =
        aclblasStrsv_npu(nullptr, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, nullptr, 5, nullptr, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Trsv, TrsvArch35Test, ::testing::ValuesIn(GetCasesFromCsv<TrsvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<TrsvParam>);

TEST_P(TrsvArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    const int64_t allocN = std::max(int64_t(1), std::abs(p.n));
    const int64_t allocLda = std::max(allocN, p.lda);

    // ── Generate A matrix ──
    auto aHost = makeBlasArray(allocLda * allocN, p.A, p.randomSeed);
    if (!aHost.empty()) {
        float boost = std::max(5.0f, static_cast<float>(allocN));
        for (int64_t i = 0; i < allocN; i++) {
            float& diag = aHost[i + i * allocLda];
            diag += (diag >= 0.0f ? boost : -boost);
        }
    }

    // ── Generate x vector (RHS) with stride ──
    const int ni = static_cast<int>(p.n);
    auto xHost = makeBlasStrided(std::abs(ni), static_cast<int>(p.incx), p.x, p.randomSeed);

    // ── Save golden copy before NPU modifies xHost in-place ──
    std::vector<float> golden = xHost;

    // ── Allocate device buffers (skip for nullptr data or n<=0) ──
    const bool needDevA = !aHost.empty() && p.n > 0;
    const bool needDevX = !xHost.empty() && p.n > 0;

    size_t aBytes =
        needDevA ? (static_cast<size_t>(allocLda) * static_cast<size_t>(allocN) * sizeof(float)) : sizeof(float);
    DeviceBuffer aDev(aBytes);
    if (needDevA) {
        aDev.copyFromHost(aHost.data(), aBytes);
    }

    size_t xBytes =
        needDevX ? ((static_cast<size_t>(std::abs(p.incx)) * static_cast<size_t>(std::max(int64_t(1), p.n) - 1) + 1) *
                    sizeof(float)) :
                   sizeof(float);
    DeviceBuffer xDev(xBytes);
    if (needDevX) {
        xDev.copyFromHost(xHost.data(), xBytes);
    }

    // ── Determine pointers: pass nullptr for null data or n<=0 ──
    const float* aDevPtr = needDevA ? static_cast<const float*>(aDev.ptr()) : nullptr;
    float* xDevPtr = needDevX ? static_cast<float*>(xDev.ptr()) : nullptr;

    // ── Execute NPU ──
    aclblasStatus_t ret =
        aclblasStrsv_npu(TrsvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n, aDevPtr, p.lda, xDevPtr, p.incx);

    // ── Verify ──
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS))
        << "Unexpected NPU error code: " << static_cast<int>(ret);

    // Copy x result back from device to host
    if (needDevX) {
        xDev.copyToHost(xHost.data(), xBytes);
    }

    aclblasStrsv_cpu(TrsvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n, aHost.data(), p.lda, golden.data(), p.incx);

    const int absIncx = std::abs(static_cast<int>(p.incx));
    const int absNi = std::abs(ni);
    const float* outPtr = (p.incx < 0 && absNi > 0) ? xHost.data() + (absNi - 1) * absIncx : xHost.data();
    const float* goldPtr = (p.incx < 0 && absNi > 0) ? golden.data() + (absNi - 1) * absIncx : golden.data();

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldPtr, static_cast<size_t>(absNi));
    EXPECT_TRUE(Verifier::verifyVector(outPtr, goldPtr, static_cast<size_t>(absNi), p.incx, cfg, p.caseName));
}
