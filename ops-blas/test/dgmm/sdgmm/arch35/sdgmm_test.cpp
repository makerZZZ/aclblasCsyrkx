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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "sdgmm_param.h"
#include "sdgmm_golden.h"
#include "sdgmm_npu_wrapper.h"

// ── Test fixture ─────────────────────────────────────────────────────────────
class SdgmmArch35Test : public BlasTest<SdgmmParam> { };

// ── TEST_F: null handle (not in CSV) ─────────────────────────────────────────
TEST_F(SdgmmArch35Test, NullHandle) {
    aclblasStatus_t ret = aclblasSdgmm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, 4, 4,
        nullptr, 4, nullptr, 1, nullptr, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// ── Helper: shared dX/dAC malloc + memcpy for in-place tests ──────────────────
static bool SdgmmInPlaceSetup(void** dX, void** dAC,
    const std::vector<float>& xHost, const std::vector<float>& aHost,
    size_t xBytes, size_t acBytes, size_t aCopyBytes)
{
    aclError ret = aclrtMalloc(dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(ret, ACL_SUCCESS) << "aclrtMalloc dX failed";
    if (ret != ACL_SUCCESS) return false;

    ret = aclrtMalloc(dAC, acBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(ret, ACL_SUCCESS) << "aclrtMalloc dAC failed";
    if (ret != ACL_SUCCESS) { aclrtFree(*dX); return false; }

    ret = aclrtMemcpy(*dX, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(ret, ACL_SUCCESS) << "aclrtMemcpy H2D failed for dX";
    if (ret != ACL_SUCCESS) { aclrtFree(*dX); aclrtFree(*dAC); return false; }

    ret = aclrtMemcpy(*dAC, acBytes, aHost.data(), aCopyBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(ret, ACL_SUCCESS) << "aclrtMemcpy H2D failed for dAC";
    if (ret != ACL_SUCCESS) { aclrtFree(*dX); aclrtFree(*dAC); return false; }

    return true;
}

// ── TEST_F: in-place A==C, lda != ldc → INVALID_VALUE ─────────────────────────
// The wrapper allocates separate device buffers for A and C, so the host-side
// A==C overlap check (sdgmm_host.cpp:63-66) can never trigger through it.
// This test bypasses the wrapper and calls aclblasSdgmm directly, passing the
// same device pointer for A and C to verify the overlap rejection.
TEST_F(SdgmmArch35Test, InPlaceLdaNeLdc) {
    constexpr int m = 4, n = 4, lda = 4, ldc = 5, incx = 1;
    const int maxLd = std::max(lda, ldc);
    const size_t acElems = static_cast<size_t>(maxLd) * static_cast<size_t>(n);
    const size_t acBytes = acElems * sizeof(float);
    const size_t xBytes = static_cast<size_t>(m) * sizeof(float);
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(float);

    std::vector<float> xHost = makeBlasStrided(m, incx, "RANDOM_NORM_5_5", 42);
    std::vector<float> aHost = makeBlasMatrix(m, n, lda, "RANDOM_NORM_5_5", 42);

    void* dX = nullptr;
    void* dAC = nullptr;
    // Only lda*n elements of A are valid; the rest of dAC is padding for ldc.
    if (!SdgmmInPlaceSetup(&dX, &dAC, xHost, aHost, xBytes, acBytes, aBytes)) return;

    // A and C point to the same device buffer; lda != ldc must be rejected.
    aclblasStatus_t dgmmRet = aclblasSdgmm(
        SdgmmArch35Test::handle_, ACLBLAS_SIDE_LEFT, m, n,
        static_cast<const float*>(dAC), lda,
        static_cast<const float*>(dX), incx,
        static_cast<float*>(dAC), ldc);
    EXPECT_EQ(static_cast<int>(dgmmRet), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));

    aclrtFree(dX);
    aclrtFree(dAC);
}

// ── TEST_F: in-place A==C, lda == ldc → SUCCESS ───────────────────────────────
// Verifies that in-place execution (A==C, lda==ldc) succeeds and produces
// correct results. Bypasses the wrapper to pass the same device pointer.
TEST_F(SdgmmArch35Test, InPlaceLdaEqLdc) {
    constexpr int m = 4, n = 4, lda = 4, ldc = 4, incx = 1;
    const size_t acElems = static_cast<size_t>(lda) * static_cast<size_t>(n);
    const size_t acBytes = acElems * sizeof(float);
    const size_t xBytes = static_cast<size_t>(m) * sizeof(float);

    // aHost is preserved for golden computation (NPU overwrites device copy only).
    std::vector<float> xHost = makeBlasStrided(m, incx, "RANDOM_NORM_5_5", 42);
    std::vector<float> aHost = makeBlasMatrix(m, n, lda, "RANDOM_NORM_5_5", 42);

    void* dX = nullptr;
    void* dAC = nullptr;
    if (!SdgmmInPlaceSetup(&dX, &dAC, xHost, aHost, xBytes, acBytes, acBytes)) return;

    // A and C point to the same device buffer; lda == ldc allows in-place execution.
    aclblasStatus_t dgmmRet = aclblasSdgmm(
        SdgmmArch35Test::handle_, ACLBLAS_SIDE_LEFT, m, n,
        static_cast<const float*>(dAC), lda,
        static_cast<const float*>(dX), incx,
        static_cast<float*>(dAC), ldc);
    EXPECT_EQ(static_cast<int>(dgmmRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (dgmmRet != ACLBLAS_STATUS_SUCCESS) {
        aclrtFree(dX);
        aclrtFree(dAC);
        return;
    }

    // Sync and copy in-place result back to host.
    EXPECT_EQ(aclrtSynchronizeStream(SdgmmArch35Test::stream_), ACL_SUCCESS);
    std::vector<float> cHost(acElems, kBlasSentinel);
    EXPECT_EQ(aclrtMemcpy(cHost.data(), acBytes, dAC, acBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

    // Golden computed from original aHost (not modified by NPU).
    std::vector<float> goldenC(acElems, kBlasSentinel);
    aclblasStatus_t cpuRet = aclblasSdgmm_cpu(
        SdgmmArch35Test::handle_, ACLBLAS_SIDE_LEFT, m, n,
        aHost.data(), lda, xHost.data(), incx, goldenC.data(), ldc);
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    // Precision verification — mixed tolerance (ops-precision-standard).
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenC.data(), goldenC.size());
    EXPECT_TRUE(Verifier::verifyVector(
        cHost.data(), goldenC.data(), cHost.size(), 1, cfg, "InPlaceLdaEqLdc"));

    aclrtFree(dX);
    aclrtFree(dAC);
}

// ── CSV parameterised test suite ─────────────────────────────────────────────
INSTANTIATE_TEST_SUITE_P(
    Sdgmm, SdgmmArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<SdgmmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SdgmmParam>);

// ── TEST_P: 5-step CSV-driven flow ───────────────────────────────────────────
TEST_P(SdgmmArch35Test, CsvDriven) {
    const auto& p = GetParam();

    // Step 1: Generate host data
    // x length: mode=L -> m, mode=R -> n
    const int xLen = (p.mode == ACLBLAS_SIDE_LEFT) ? p.m : p.n;
    std::vector<float> xHost;
    if (p.nullx == 0 && p.m > 0 && p.n > 0) {
        // MIXED_RANDOM_INF / MIXED_RANDOM_NAN fills (first half RANDOM, second half
        // special value) need the dedicated mixed generator; everything else uses
        // the standard strided generator.
        const std::string& xRaw = p.xFillRaw;
        if (xRaw.rfind("MIXED", 0) == 0) {
            xHost = makeBlasMixed(xLen, p.incx, xRaw, p.randomSeed);
        } else {
            xHost = makeBlasStrided(xLen, p.incx, p.xFill, p.randomSeed);
        }
    }

    std::vector<float> aHost;
    if (p.nullA == 0 && p.m > 0 && p.n > 0) {
        aHost = makeBlasMatrix(p.m, p.n, p.lda, p.aFill, p.randomSeed);
    }

    // C is the output buffer; initialise with sentinel so unmodified padding
    // (if any) matches the golden's untouched region.
    std::vector<float> cHost;
    if (p.nullC == 0 && p.m > 0 && p.n > 0) {
        cHost.assign(static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n), kBlasSentinel);
    }

    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    const float* aPtr = aHost.empty() ? nullptr : aHost.data();
    float*       cPtr = cHost.empty() ? nullptr : cHost.data();

    // Step 2: Execute on NPU (wrapper handles nullptr passthrough, device memory)
    aclblasStatus_t ret = aclblasSdgmm_npu(
        SdgmmArch35Test::handle_, p.mode, p.m, p.n,
        aPtr, p.lda, xPtr, p.incx, cPtr, p.ldc);

    // Step 3: Verify expected return code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) return;

    // m==0 or n==0: operator returns SUCCESS without computing; no output to verify
    if (p.m == 0 || p.n == 0) return;

    // Step 4: Compute golden on CPU
    // xHost and aHost are not modified by the NPU wrapper (only cHost is written
    // via D2H), so they remain valid for golden computation.
    std::vector<float> goldenC(cHost.size(), kBlasSentinel);
    aclblasStatus_t cpuRet = aclblasSdgmm_cpu(
        SdgmmArch35Test::handle_, p.mode, p.m, p.n,
        aPtr, p.lda, xPtr, p.incx, goldenC.data(), p.ldc);
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    // Step 5: Precision verification — mixed tolerance (ops-precision-standard).
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenC.data(), cHost.size());

    // Compare entire C storage (ldc * n elements, stride 1).
    // Padding rows (if lda/ldc > m) are sentinel in both cHost and goldenC.
    EXPECT_TRUE(Verifier::verifyVector(
        cPtr, goldenC.data(), cHost.size(), 1, cfg, p.caseName));
}
