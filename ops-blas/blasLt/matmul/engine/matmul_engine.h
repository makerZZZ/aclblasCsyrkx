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
 * \file matmul_engine.h
 * \brief Matmul engine entry: dtype routing (FP32 / MXFP8 / MXFP4) plus the alpha/beta epilogue,
 *        driven by a plain MatmulProblem produced by the API layer.
 */

#pragma once

#include "aclblaslt_matmul_problem.h"

#include "cann_ops_blasLt.h"

// Run the two-stage matmul pipeline (MMAD kernel + optional epilogue) for the given problem.
aclblasStatus_t MatmulLaunch(const MatmulProblem& problem);
