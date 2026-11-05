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
 * \file ctrsm_batched_kernel_aiv_cfg.h
 * \brief 各 AIV 子类共享的派生配置与 UB buffer 指针。由编排类 CtrsmMixAivImpl::Init 一次性填充，
 *        子类通过指针只读/更新（colStart/End、buildRowStart/End 会在分列/分行时更新）。
 *        buffer 由编排类统一分配，此处仅保存指针，保留原有别名布局以满足 192KB 预算。
 */

#pragma once

#include "ctrsm_batched_kernel_common.h"

using BufVecCalc = AscendC::TBuf<AscendC::TPosition::VECCALC>;

struct CtrsmAivCfg {
    static constexpr int32_t TS = 64; // TiledTranspose 的分块大小

    const __gm__ CtrsmBatchedTilingData* td;   // tiling 参数指针
    int32_t kDim, nCols, nb, nColsAligned, nbAligned;
    int32_t colTile, gatherChunk, kDimOrig, nColsOrig, srcBufFloats, aEffStride;
    int32_t colStart, colEnd;         // 当前 AIV 处理的列范围（dual 模式分列）
    int32_t buildRowStart, buildRowEnd; // BuildPaddedB 处理的行范围
    int32_t localNCols, localNColsAligned, nColsOffset; // 多核拆分时的本组 nCols 参数
    bool needTA, conjA, padOn, useOrigA;

    // UB buffer 指针（实体由编排类持有）
    BufVecCalc *bufPanelA_real, *bufPanelA_imag;
    BufVecCalc *bufPanelB_real, *bufPanelB_imag;
    BufVecCalc *bufNeg_real, *bufNeg_imag, *bufNeg_imag_neg;
    BufVecCalc *bufRankK, *bufRow;
    // 专用 gather 偏移表 buffer（Init 时构建一次，永不复用，供 LoadPanelA 复用）
    BufVecCalc *bufGatherEven, *bufGatherOdd;
};
