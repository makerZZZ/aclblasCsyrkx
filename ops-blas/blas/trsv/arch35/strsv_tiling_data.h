/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file strsv_tiling_data.h
 * \brief Tiling data for single-precision triangular solver (unpacked LDA format).
 */

#pragma once

#include <cstdint>

struct StrsvTilingData {
    uint32_t n;
    uint32_t uplo;       // ACLBLAS_UPPER(121) or ACLBLAS_LOWER(122)
    uint32_t trans;      // ACLBLAS_OP_N(111), ACLBLAS_OP_T(112), or ACLBLAS_OP_C(113)
    uint32_t diag;       // ACLBLAS_NON_UNIT(131) or ACLBLAS_UNIT(132)
    int32_t incx;        // x vector increment (may be negative)
    int32_t lda;         // A matrix leading dimension
    uint32_t numThreads; // SIMT thread count (0 = scalar path, >0 = SIMT path)
};
