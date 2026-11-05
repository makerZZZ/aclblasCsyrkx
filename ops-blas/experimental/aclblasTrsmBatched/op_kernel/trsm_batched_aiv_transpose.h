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
 * \file trsm_batched_aiv_transpose.h
 * \brief GM->GM 分块转置引擎。使用共享 UB buffer(bufPanelB 作源、bufNeg 作目标)和分块尺寸 ts。
 *        无求解语义，供规范化(A^T/B^T)与结果回写复用。
 */

#pragma once

#include "trsm_batched_aiv_cfg.h"

class TrsmTranspose {
public:
    __aicore__ inline TrsmTranspose() {}

    // 绑定共享配置与转置用的两个 UB buffer（与 panel-solve 复用同一片 UB）
    __aicore__ inline void Bind(const TrsmAivCfg* cfg, BufVecCalc* bufSrc, BufVecCalc* bufDst)
    {
        cfg_ = cfg;
        bufSrc_ = bufSrc;
        bufDst_ = bufDst;
    }

    // Tiled transpose: dst[c][r] = src[r][c]. Uses TransDataTo5HD<float> for the small-channel
    // transpose per tile (16 rows at a time), following ascendc-api-best-practices.
    __aicore__ inline void Run(__gm__ float* src, int32_t rows, int32_t cols,
                               int32_t srcLd, __gm__ float* dst, int32_t dstLd)
    {
        int32_t ts = cfg_->ts;
        AscendC::LocalTensor<float> ubSrc = bufSrc_->Get<float>();
        AscendC::LocalTensor<float> ubDst = bufDst_->Get<float>();
        AscendC::GlobalTensor<float> gmS, gmD;
        gmS.SetGlobalBuffer(src, (uint32_t)((int64_t)rows * srcLd));
        gmD.SetGlobalBuffer(dst, (uint32_t)((int64_t)cols * dstLd));
        AscendC::DataCopyPadExtParams<float> padP{false, 0, 0, 0};
        for (int32_t rb = 0; rb < rows; rb += ts) {
            int32_t tr = (ts < rows - rb) ? ts : (rows - rb);
            for (int32_t cb = 0; cb < cols; cb += ts) {
                int32_t tc = (ts < cols - cb) ? ts : (cols - cb);
                int32_t tcA = CEIL_ALIGN(tc, FLOAT_ALIGN);
                AscendC::DataCopyExtParams ldP((uint16_t)tr, (uint32_t)(tc * sizeof(float)),
                    (int64_t)((srcLd - tc) * sizeof(float)), (int64_t)((tcA - tc) / FLOAT_ALIGN), 0);
                AscendC::DataCopyPad(ubSrc, gmS[(int64_t)rb * srcLd + cb], ldP, padP);
                AscendC::PipeBarrier<PIPE_ALL>();
                // TransposeTileBlock / TransposeTile16 为无状态子过程，位于共享头 trsm_mix_common.h
                TransposeTileBlock(ubSrc, ubDst, gmD, tcA, tr, tc, dstLd, cb, rb);
            }
        }
    }

private:
    const TrsmAivCfg* cfg_;
    BufVecCalc* bufSrc_;
    BufVecCalc* bufDst_;
};
