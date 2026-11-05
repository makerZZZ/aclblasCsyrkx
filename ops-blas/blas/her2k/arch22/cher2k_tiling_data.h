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

#include <cstddef>
#include <cstdint>

struct Cher2kTilingData {
    uint32_t n;
    uint32_t k;
    uint32_t nAligned;
    uint32_t kAligned;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint32_t trans;
    uint32_t uplo;
    float alphaReal;
    float alphaImag;
    float beta;
    uint32_t aivCoreNum;
    uint32_t aicCoreNum;
    uint32_t computeProduct;
    uint32_t useThreeM;
    uint32_t smallPath;
    uint32_t outputTransposePath;
    // Chemm-style tile-owned 3M Cube producer.
    uint32_t sg3MmadPath;
    // Low-rank OP_N path: Cube emits Hermitian real and transposed-imaginary
    // planes directly, so the AIV consumer never materializes or mirrors Q.
    uint32_t directHermitianPath;
    // Direct-H output contract. OP_N always uses it when directHermitianPath
    // is selected; OP_C uses one or two 64-wide K panels for selected buckets.
    uint32_t directOutputPath;
    // Non-zero when alpha/beta are device pointers.  The host then has no
    // resolved value and every kernel re-reads the scalars from GM, so the
    // alphaReal/alphaImag/beta fields above are unused placeholders instead of
    // the values this call will apply.
    uint32_t deviceScalars;
};

static inline size_t CalcCher2kWorkspaceBytes(uint32_t nAligned, uint32_t kAligned, bool useThreeM)
{
    const uint64_t nk = static_cast<uint64_t>(nAligned) * kAligned;
    const uint64_t nn = static_cast<uint64_t>(nAligned) * nAligned;
    const uint64_t totalFloats = 4ULL * nk + 4ULL * nn + (useThreeM ? nk : 0ULL);
    const uint64_t bytes = totalFloats * sizeof(float);
    return static_cast<size_t>((bytes + 31ULL) & ~31ULL);
}

static inline size_t CalcCher2kSgemm3BatchWorkspaceBytes(uint32_t nAligned, uint32_t kAligned)
{
    const uint64_t nk = static_cast<uint64_t>(nAligned) * kAligned;
    const uint64_t nn = static_cast<uint64_t>(nAligned) * nAligned;
    // [Ar, Ai, Ar+Ai], [Br, Bi, Br+Bi], [P1, P2, P3].
    const uint64_t floats = 6ULL * nk + 3ULL * nn;
    return static_cast<size_t>((floats * sizeof(float) + 31ULL) & ~31ULL);
}
