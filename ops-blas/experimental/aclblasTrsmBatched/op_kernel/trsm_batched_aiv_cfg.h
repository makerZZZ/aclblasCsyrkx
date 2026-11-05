/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trsm_batched_aiv_cfg.h
 * \brief 各 AIV 子类共享的派生配置（在 TrsmMixAiv::Init 中一次性计算，其余类通过指针只读/更新）。
 *        solveColStart/solveColEnd/coopAiv 会在协作列拆分时更新，故以指针共享而非按值拷贝。
 */

#pragma once

#include "trsm_batched_kernel_common.h"

using BufVecCalc = AscendC::TBuf<AscendC::TPosition::VECCALC>;

struct TrsmAivCfg {
    const __gm__ TrsmBatchedTilingData* tiling;
    int32_t kDim, nCols, nb, nColsAligned, nbAligned, colTile, ts;
    int32_t kDimOrig, nColsOrig;
    int32_t solveColStart, solveColEnd;
    int32_t splitId;
    int32_t splitNcols;
    bool forward, needTA, right, padOn, is12Mode, coopAiv;
};
