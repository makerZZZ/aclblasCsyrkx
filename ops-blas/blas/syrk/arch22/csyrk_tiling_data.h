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
 * \file csyrk_tiling_data.h
 * \brief CSYRK-specific tiling. Phases 0 and 1 reuse the shared structures in
 *        common/helper/complex_blas3_tiling_data.h; only the combine stage
 *        differs between operators.
 */

#ifndef CSYRK_TILING_DATA_H
#define CSYRK_TILING_DATA_H

#include <cstdint>
#include "common/helper/complex_blas3_tiling_data.h"

// Phase 2: apply complex alpha/beta to the two GEMM results.
//
// Phase 1 produces Pr and Pi directly, via K-concatenated operands:
//   Pr = P*Q^T = Ar*Ar^T - Ai*Ai^T      (symmetric)
//   Pi = P*R^T = Ar*Ai^T + Ai*Ar^T      (symmetric)
// so this stage no longer combines four temps -- it only scales:
//   C = alpha * (Pr + i*Pi) + beta * C_old        (alpha, beta complex)
//
// Two properties worth stating because they differ from CHERK:
//   1. alpha and beta are complex, so the scaling expands to four products per
//      part;
//   2. C is *symmetric* rather than Hermitian, so the diagonal imaginary part is
//      a genuine result value and must NOT be forced to zero.
//
// Both Pr and Pi are symmetric, so the transposed read that Phase 2 performs on
// the temp buffers yields the same value -- no pointer swap is needed anywhere.
struct CsyrkCombineTilingData {
    uint32_t n;
    uint32_t ldc;           // leading dimension of C, in complex elements
    uint32_t tempLdc;       // row stride of the two temps, in floats
    uint32_t uploMode;      // CBLAS3_UPLO_UPPER / CBLAS3_UPLO_LOWER
    uint32_t skipAlphaTerm; // alpha == 0 or k == 0: skip the alpha term entirely
    uint32_t isBetaZero;    // beta == 0: do not read C_old
    float alphaRe;
    float alphaIm;
    float betaRe;
    float betaIm;
};

#endif // CSYRK_TILING_DATA_H
