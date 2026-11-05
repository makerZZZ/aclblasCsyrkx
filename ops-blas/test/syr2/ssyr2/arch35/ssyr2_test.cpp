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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "ssyr2_param.h"
#include "ssyr2_golden.h"
#include "ssyr2_npu_wrapper.h"

class Ssyr2Arch35Test : public BlasTest<Ssyr2Param> {};

TEST_F(Ssyr2Arch35Test, NullHandle)
{
    float alpha = 1.0f;
    float dummy = 0.0f;
    aclblasStatus_t ret = aclblasSsyr2(nullptr, ACLBLAS_UPPER, 4, &alpha, &dummy, 1, &dummy, 1, &dummy, 4);
    EXPECT_NE(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
}

INSTANTIATE_TEST_SUITE_P(
    Ssyr2, Ssyr2Arch35Test, ::testing::ValuesIn(GetCasesFromCsv<Ssyr2Param>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Ssyr2Param>);

TEST_P(Ssyr2Arch35Test, CsvDriven)
{
    const auto& p = GetParam();

    size_t aSize = (p.n > 0) ? static_cast<size_t>(p.lda) * p.n : 0;

    std::vector<float> xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed + 1);
    std::vector<float> aHost =
        makeBlasArray(static_cast<int64_t>(aSize), p.a, p.randomSeed);
    std::vector<float> aOrig =
        makeBlasArray(static_cast<int64_t>(aSize), p.a, p.randomSeed);

    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    const float* yPtr = yHost.empty() ? nullptr : yHost.data();
    float* aPtr = aHost.empty() ? nullptr : aHost.data();

    aclblasStatus_t ret =
        aclblasSsyr2_npu(Ssyr2Arch35Test::handle_, p.uplo, p.n, p.alpha, xPtr, p.incx, yPtr, p.incy, aPtr, p.lda);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;
    if (p.n == 0)
        return;

    std::vector<float> golden = aOrig;
    aclblasSsyr2_cpu(p.uplo, p.n, p.alpha, xHost.data(), p.incx, yHost.data(), p.incy, golden.data(), p.lda);

    std::vector<float> triOut, triGold;
    for (int j = 0; j < p.n; j++) {
        int iStart = (p.uplo == ACLBLAS_UPPER) ? 0 : j;
        int iEnd = (p.uplo == ACLBLAS_UPPER) ? j : (p.n - 1);
        for (int i = iStart; i <= iEnd; i++) {
            int64_t idx = static_cast<int64_t>(j) * p.lda + i;
            triOut.push_back(aHost[idx]);
            triGold.push_back(golden[idx]);
        }
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, triGold.data(), triGold.size());
    EXPECT_TRUE(Verifier::verifyVector(triOut.data(), triGold.data(), triOut.size(), 1, cfg, p.caseName));

    uint32_t unchErr = 0;
    for (int j = 0; j < p.n; j++) {
        int iStart = (p.uplo == ACLBLAS_UPPER) ? j + 1 : 0;
        int iEnd = (p.uplo == ACLBLAS_UPPER) ? p.n - 1 : j - 1;
        for (int i = iStart; i <= iEnd; i++) {
            int64_t idx = static_cast<int64_t>(j) * p.lda + i;
            if (aHost[idx] != aOrig[idx])
                unchErr++;
        }
    }
    EXPECT_EQ(unchErr, 0u) << "[" << p.caseName << "] Unchanged region: " << unchErr << " errors";
}
