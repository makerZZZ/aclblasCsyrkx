/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <string>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#ifdef SSYMM_ARCH35
#include "fill.h"
// ===== arch35: int + BlasFillMode =====
struct SsymmParam : public BlasTestParamBase {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    int m = 0;
    int n = 0;
    float alpha = 1.0f;
    bool nullAlpha = false;
    int lda = 0;
    int ldb = 0;
    float beta = 0.0f;
    bool nullBeta = false;
    int ldc = 0;
    BlasFillMode aFill = parseFill("RANDOM_NORM_5_5");
    BlasFillMode bFill = parseFill("RANDOM_NORM_5_5");
    BlasFillMode cFill = parseFill("RANDOM_NORM_5_5");

    SsymmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        side  = parseSideMode(ReadMap(map, "side", "LEFT"));
        uplo  = parseFillMode(ReadMap(map, "uplo", "LOWER"));
        m     = parseInt(ReadMap(map, "m", "0"));
        n     = parseInt(ReadMap(map, "n", "0"));

        std::string alphaStr = ReadMap(map, "alpha", "1.0");
        nullAlpha = (alphaStr == "null" || alphaStr == "nullptr");
        alpha = nullAlpha ? 0.0f : parseFloat(alphaStr, 1.0f);

        lda = parseInt(ReadMap(map, "lda", "0"));
        ldb = parseInt(ReadMap(map, "ldb", "0"));

        std::string betaStr = ReadMap(map, "beta", "0.0");
        nullBeta = (betaStr == "null" || betaStr == "nullptr");
        beta = nullBeta ? 0.0f : parseFloat(betaStr, 0.0f);

        ldc = parseInt(ReadMap(map, "ldc", "0"));
        aFill = parseFill(ReadMap(map, "a_fill", "RANDOM_NORM_5_5"));
        bFill = parseFill(ReadMap(map, "b_fill", "RANDOM_NORM_5_5"));
        cFill = parseFill(ReadMap(map, "c_fill", "RANDOM_NORM_5_5"));
    }
};
#else
// ===== arch22: int64_t (original) =====
struct SsymmParam : public BlasTestParamBase {
    aclblasSideMode_t side = ACLBLAS_SIDE_LEFT;
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    int64_t m = 0;
    int64_t n = 0;
    float   alpha = 1.0f;
    bool    nullAlpha = false;
    int64_t lda = 0;
    int64_t ldb = 0;
    float   beta = 0.0f;
    bool    nullBeta = false;
    int64_t ldc = 0;

    SsymmParam(const csv_map& map) : BlasTestParamBase(map)
    {
        side  = parseSideMode(ReadMap(map, "side", "LEFT"));
        uplo  = parseFillMode(ReadMap(map, "uplo", "LOWER"));
        m     = parseInt64(ReadMap(map, "m", "0"));
        n     = parseInt64(ReadMap(map, "n", "0"));

        std::string alphaStr = ReadMap(map, "alpha", "1.0");
        nullAlpha = (alphaStr == "null" || alphaStr == "nullptr");
        alpha = nullAlpha ? 0.0f : parseFloat(alphaStr, 1.0f);

        lda = parseInt64(ReadMap(map, "lda", "0"));
        ldb = parseInt64(ReadMap(map, "ldb", "0"));

        std::string betaStr = ReadMap(map, "beta", "0.0");
        nullBeta = (betaStr == "null" || betaStr == "nullptr");
        beta = nullBeta ? 0.0f : parseFloat(betaStr, 0.0f);

        ldc = parseInt64(ReadMap(map, "ldc", "0"));
    }
};
#endif
