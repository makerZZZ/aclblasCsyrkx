/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file csymm_tiling_data.h
 * \brief CSYMM-specific tiling. Phases 0 and 1 reuse the shared structures in
 *        common/helper/complex_blas3_tiling_data.h.
 */

#ifndef CSYMM_TILING_DATA_H
#define CSYMM_TILING_DATA_H

#include <cstdint>
#include "common/helper/complex_blas3_tiling_data.h"

// Phase 2: assemble the full m x n complex C from four real temps.
//
//   Pr = t1 - t2      Pi = t3 + t4
//   C  = alpha*P + beta*C_old            (alpha and beta both complex)
//
// The temps hold C^T in row-major, i.e. temp[j * tempLdc + i] is the (i, j) element
// of the logical result. C is column-major, so element (i, j) sits at
// (j * ldc + i) * 2 floats. Walking a column of C therefore reads a contiguous run
// of each temp and writes a contiguous run of C -- unlike the rank-K operators,
// whose combine stage reads its temps transposed.
//
// There is no triangle and no diagonal special case: CSYMM's output is a general
// matrix. The symmetric structure of A has already been consumed by Phase 0b.
struct CsymmCombineTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t ldc;           // leading dimension of C, in complex elements
    uint32_t tempLdc;       // row stride of the four temps, in floats
    uint32_t skipAlphaTerm; // alpha == 0: skip the alpha term entirely
    uint32_t isBetaZero;    // beta == 0: do not read C_old
    // Non-zero when Phase 1 used the paired-K layout: t1 and t3 already hold the
    // finished real and imaginary sums, so t2 / t4 are unused and no combining
    // subtraction is left to do here.
    uint32_t pairedTemps;
    // Non-zero when Phase 0 scaled A down: the sums are multiplied by
    // `unscaleFactor` (the exact reciprocal power of two) before alpha is applied.
    uint32_t unscaleEnabled;
    float unscaleFactor;
    float alphaRe;
    float alphaIm;
    float betaRe;
    float betaIm;
};

#endif // CSYMM_TILING_DATA_H
