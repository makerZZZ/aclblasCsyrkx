/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STBSV_NPU_WRAPPER_H
#define STBSV_NPU_WRAPPER_H

#include "acl/acl.h"
#include "cann_ops_blas.h"

inline aclblasStatus_t aclblasStbsv_npu(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int n,
    int k,
    const float* A,
    int lda,
    float* x,
    int incx)
{
    return aclblasStbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx);
}

#endif // STBSV_NPU_WRAPPER_H
