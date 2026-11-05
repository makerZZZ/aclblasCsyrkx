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
#include <optional>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "device.h"
#include "fill.h"
#include "stbsv_param.h"
#include "stbsv_golden.h"
#include "stbsv_npu_wrapper.h"

class StbsvArch35Test : public BlasTest<StbsvParam> {};

INSTANTIATE_TEST_SUITE_P(
    Stbsv, StbsvArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<StbsvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StbsvParam>);

struct StbsvFixture {
    const StbsvParam& p;
    int effK = 0;
    int allocLda = 1;
    int allocN = 1;
    int absIncx = 0;
    size_t xSize = 0;
    bool isUpper = false;
    bool needDevA = false;
    bool needDevX = false;
    std::vector<float> aHost;
    std::vector<float> xHost;
    std::vector<float> golden;
    std::optional<DeviceBuffer> aDev;
    std::optional<DeviceBuffer> xDev;

    explicit StbsvFixture(const StbsvParam& param) : p(param) {}

    void InitDims()
    {
        effK = (p.n > 0) ? std::min(p.k, p.n - 1) : 0;
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
        const int kl = isUpper ? 0 : effK;
        const int ku = isUpper ? effK : 0;
        aHost = makeBlasBanded(p.n, p.n, kl, ku, allocLda, "RANDOM_2", p.randomSeed);
        StrengthenDiagonal();
        xHost = makeBlasStrided(p.n, p.incx, "RANDOM_2", p.randomSeed + 1);
        if (xHost.empty()) {
            xHost.resize(xSize, 0.0f);
        }
        golden = xHost;
    }

    void StrengthenDiagonal()
    {
        if (aHost.empty() || p.n <= 0) {
            return;
        }
        const int diagRow = isUpper ? effK : 0;
        for (int j = 0; j < p.n; j++) {
            float& diag = aHost[diagRow + static_cast<size_t>(j) * allocLda];
            diag += (diag >= 0.0f) ? 5.0f : -5.0f;
        }
    }

    void PrepareDevice()
    {
        if (needDevA) {
            size_t aBytes = static_cast<size_t>(allocLda) * static_cast<size_t>(allocN) * sizeof(float);
            aDev.emplace(aBytes);
            aDev->copyFromHost(aHost.data(), aBytes);
        }
        if (needDevX) {
            size_t xBytes = xSize * sizeof(float);
            xDev.emplace(xBytes);
            xDev->copyFromHost(xHost.data(), xBytes);
        }
    }

    aclblasStatus_t CallNpu(aclblasHandle_t handle)
    {
        const float* aPtr = aDev.has_value() ? static_cast<const float*>(aDev->ptr()) : nullptr;
        float* xPtr = xDev.has_value() ? static_cast<float*>(xDev->ptr()) : nullptr;
        return aclblasStbsv_npu(handle, p.uplo, p.trans, p.diag, p.n, p.k, aPtr, p.lda, xPtr, p.incx);
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
            xDev->copyToHost(xHost.data(), xSize * sizeof(float));
        }
        aclblasStbsv_cpu(handle, p.uplo, p.trans, p.diag, p.n, effK,
                         aHost.data(), allocLda, golden.data(), p.incx);
        const float* outPtr = (p.incx < 0) ? xHost.data() + (p.n - 1) * absIncx : xHost.data();
        const float* goldPtr = (p.incx < 0) ? golden.data() + (p.n - 1) * absIncx : golden.data();
        VerifyConfig cfg;
        applyMixedTolerance(cfg, ACL_FLOAT, goldPtr, static_cast<size_t>(p.n));
        EXPECT_TRUE(Verifier::verifyVector(outPtr, goldPtr,
            static_cast<size_t>(p.n), (p.incx < 0) ? -absIncx : absIncx, cfg, p.caseName));
    }
};

TEST_P(StbsvArch35Test, CsvDriven)
{
    StbsvFixture f(GetParam());
    f.InitDims();
    f.PrepareHostData();
    f.PrepareDevice();
    aclblasStatus_t ret = f.CallNpu(StbsvArch35Test::handle_);
    f.Verify(StbsvArch35Test::handle_, ret);
}
