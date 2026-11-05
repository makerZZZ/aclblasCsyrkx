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

#include <cassert>

#include <cblas.h>

#include "cann_ops_blas.h"

inline CBLAS_TRANSPOSE ToCblasOp(aclblasOperation_t op)
{
    switch (op) {
        case ACLBLAS_OP_N:
            return CblasNoTrans;
        case ACLBLAS_OP_T:
            return CblasTrans;
        case ACLBLAS_OP_C:
            return CblasConjTrans;
        default:
            assert(false && "Invalid aclblasOperation_t");
            return CblasNoTrans;
    }
}

inline CBLAS_UPLO ToCblasUplo(aclblasFillMode_t uplo)
{
    switch (uplo) {
        case ACLBLAS_UPPER:
            return CblasUpper;
        case ACLBLAS_LOWER:
            return CblasLower;
        default:
            assert(false && "Invalid aclblasFillMode_t");
            return CblasUpper;
    }
}

inline char ToLapackUplo(aclblasFillMode_t uplo)
{
    switch (uplo) {
        case ACLBLAS_UPPER:
            return 'U';
        case ACLBLAS_LOWER:
            return 'L';
        default:
            assert(false && "Invalid aclblasFillMode_t");
            return 'U';
    }
}

inline CBLAS_DIAG ToCblasDiag(aclblasDiagType_t diag)
{
    switch (diag) {
        case ACLBLAS_UNIT:
            return CblasUnit;
        case ACLBLAS_NON_UNIT:
            return CblasNonUnit;
        default:
            assert(false && "Invalid aclblasDiagType_t");
            return CblasNonUnit;
    }
}

