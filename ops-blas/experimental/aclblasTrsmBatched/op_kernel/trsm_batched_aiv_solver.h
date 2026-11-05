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
 * \file trsm_batched_aiv_solver.h
 * \brief Panel 三角求解器：alpha 缩放、panel TRSV（外积/内积两条路径）、结果与 Xneg 回写。
 *        通过 Bind 绑定共享配置与所需 UB buffer；不拥有 buffer，仅引用编排类分配的那一份。
 */

#pragma once

#include "trsm_batched_aiv_cfg.h"

class TrsmPanelSolver {
public:
    __aicore__ inline TrsmPanelSolver() {}

    __aicore__ inline void Bind(const TrsmAivCfg* cfg,
        BufVecCalc* bufPanelA, BufVecCalc* bufAt, BufVecCalc* bufColIdx,
        BufVecCalc* bufPanelB, BufVecCalc* bufNeg, BufVecCalc* bufRankK,
        BufVecCalc* bufBcA, BufVecCalc* bufBcB, BufVecCalc* bufBcTmp)
    {
        cfg_ = cfg;
        bufPanelA_ = bufPanelA; bufAt_ = bufAt; bufColIdx_ = bufColIdx;
        bufPanelB_ = bufPanelB; bufNeg_ = bufNeg; bufRankK_ = bufRankK;
        bufBcA_ = bufBcA; bufBcB_ = bufBcB; bufBcTmp_ = bufBcTmp;
    }

    // Apply alpha scaling to effective B matrix over the owned column range
    __aicore__ inline void AlphaScale(__gm__ float* effB, int32_t effLdb)
    {
        float alpha = cfg_->tiling->alphaReal;
        int32_t asStart = cfg_->coopAiv ? cfg_->solveColStart : 0;
        int32_t asEnd = cfg_->coopAiv ? cfg_->solveColEnd : cfg_->nCols;
        AscendC::LocalTensor<float> ub = bufRankK_->Get<float>();
        AscendC::GlobalTensor<float> gmBT;
        gmBT.SetGlobalBuffer(effB, (uint32_t)(cfg_->kDim * effLdb));
        for (int32_t row = 0; row < cfg_->kDim; row++) {
            AlphaScaleRow(ub, gmBT, row, effLdb, alpha, asStart, asEnd);
        }
    }

    // Scale one row of B over the column range [csStart, csEnd)
    __aicore__ inline void AlphaScaleRow(AscendC::LocalTensor<float>& ub,
                                          AscendC::GlobalTensor<float>& gmBT,
                                          int32_t row, int32_t effLdb, float alpha,
                                          int32_t csStart, int32_t csEnd)
    {
        for (int32_t cs = csStart; cs < csEnd; cs += cfg_->colTile) {
            int32_t ct = (cfg_->colTile < csEnd - cs) ? cfg_->colTile : (csEnd - cs);
            int32_t ctAligned = CEIL_ALIGN(ct, FLOAT_ALIGN);
            AscendC::DataCopyPad(ub, gmBT[row * effLdb + cs],
                {1, (uint16_t)(ct * (int32_t)sizeof(float)), 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Muls(ub, ub, alpha, ctAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(gmBT[row * effLdb + cs], ub,
                {1, (uint16_t)(ct * (int32_t)sizeof(float)), 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // Pre-load panel A diagonal block into UB (look-ahead: call during AIC GEMM wait)
    __aicore__ inline void LoadPanelA(__gm__ float* effA, int32_t effLda,
                                       int32_t panelStart, int32_t actualNb)
    {
        int32_t nbAligned = cfg_->nbAligned;
        AscendC::LocalTensor<float> ubA = bufPanelA_->Get<float>();
        AscendC::GlobalTensor<float> gmAT;
        gmAT.SetGlobalBuffer(effA, (uint32_t)(cfg_->kDim * effLda));
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams aIn((uint16_t)actualNb, (uint32_t)(actualNb * sizeof(float)),
            (int64_t)((effLda - actualNb) * sizeof(float)),
            (int64_t)((nbAligned - actualNb) / FLOAT_ALIGN), 0);
        AscendC::DataCopyPad(ubA, gmAT[panelStart * effLda + panelStart], aIn, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Drive panel triangular solve: load A panel, prepare column indices, tile over columns
    __aicore__ inline void PanelTrsv(__gm__ float* effA, int32_t effLda,
                                      __gm__ float* effB, int32_t effLdb, __gm__ float* gmXneg,
                                      int32_t panelStart, int32_t actualNb, bool swapA,
                                      bool aPreloaded = false)
    {
        int32_t nbAligned = cfg_->nbAligned;
        int32_t kDim = cfg_->kDim;
        AscendC::LocalTensor<float> ubA = bufPanelA_->Get<float>();

        AscendC::GlobalTensor<float> gmAT, gmBT;
        gmAT.SetGlobalBuffer(effA, (uint32_t)(kDim * effLda));
        gmBT.SetGlobalBuffer(effB, (uint32_t)(kDim * effLdb));

        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        if (!aPreloaded) {
            AscendC::DataCopyExtParams aIn((uint16_t)actualNb, (uint32_t)(actualNb * sizeof(float)),
                (int64_t)((effLda - actualNb) * sizeof(float)),
                (int64_t)((nbAligned - actualNb) / FLOAT_ALIGN), 0);
            AscendC::DataCopyPad(ubA, gmAT[panelStart * effLda + panelStart], aIn, padParams);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        bool useOuter = (cfg_->nColsAligned <= 64);
        AscendC::LocalTensor<float> ubAt = bufAt_->Get<float>();
        if (useOuter && !swapA) {
            AscendC::LocalTensor<int32_t> offI = bufColIdx_->Get<int32_t>();
            AscendC::CreateVecIndex(offI, 0, (uint32_t)actualNb);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(offI, offI, nbAligned * (int32_t)sizeof(float), actualNb);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::LocalTensor<uint32_t> off = bufColIdx_->Get<uint32_t>();
            for (int32_t c = 0; c < actualNb; c++) {
                AscendC::Gather(ubAt[c * nbAligned], ubA, off,
                                (uint32_t)(c * (int32_t)sizeof(float)), (uint32_t)actualNb);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::LocalTensor<float>& ubAtRef = (useOuter && swapA) ? ubA : ubAt;

        AscendC::GlobalTensor<float> gmXnegT;
        gmXnegT.SetGlobalBuffer(gmXneg, (uint32_t)(cfg_->nb * cfg_->splitNcols));

        int32_t loopStart = cfg_->coopAiv ? cfg_->solveColStart : 0;
        int32_t loopEnd = cfg_->coopAiv ? cfg_->solveColEnd : cfg_->nCols;
        for (int32_t cs = loopStart; cs < loopEnd; cs += cfg_->colTile) {
            int32_t ct = (cfg_->colTile < loopEnd - cs) ? cfg_->colTile : (loopEnd - cs);
            int32_t ctAligned = CEIL_ALIGN(ct, FLOAT_ALIGN);
            PanelTrsvTile(ubA, ubAtRef, useOuter, swapA, gmBT, gmXnegT, effLdb,
                          panelStart, actualNb, cs, ct, ctAligned);
        }
    }

    // Process one column tile of PanelTrsv: load B tile, solve, write back + Xneg
    __aicore__ inline void PanelTrsvTile(
        AscendC::LocalTensor<float>& ubA, AscendC::LocalTensor<float>& ubAt,
        bool useOuter, bool swapA, AscendC::GlobalTensor<float>& gmBT,
        AscendC::GlobalTensor<float>& gmXnegT, int32_t effLdb,
        int32_t panelStart, int32_t actualNb, int32_t cs, int32_t ct, int32_t ctAligned)
    {
        AscendC::LocalTensor<float> ubB = bufPanelB_->Get<float>();
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams bIn((uint16_t)actualNb, (uint32_t)(ct * sizeof(float)),
            (int64_t)((effLdb - ct) * sizeof(float)),
            (int64_t)((ctAligned - ct) / FLOAT_ALIGN), 0);
        AscendC::DataCopyPad(ubB, gmBT[panelStart * effLdb + cs], bIn, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();

        int32_t start = cfg_->forward ? 0 : actualNb - 1;
        int32_t end = cfg_->forward ? actualNb : -1;
        int32_t step = cfg_->forward ? 1 : -1;

        if (useOuter) {
            SolveOuter(ubA, ubAt, ubB, actualNb, ctAligned, start, end, step);
        } else {
            SolveInner(ubA, ubB, swapA, actualNb, ctAligned, start, end, step);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        WriteBackSolved(ubB, gmBT, gmXnegT, effLdb, panelStart, actualNb, cs, ct, ctAligned);
    }

    // Outer-product triangular solve using Broadcast+Mul (small nCols path)
    __aicore__ inline void SolveOuter(
        AscendC::LocalTensor<float>& ubA, AscendC::LocalTensor<float>& ubAt,
        AscendC::LocalTensor<float>& ubB, int32_t actualNb, int32_t ctAligned,
        int32_t start, int32_t end, int32_t step)
    {
        int32_t nbAligned = cfg_->nbAligned;
        AscendC::LocalTensor<float> bcA = bufBcA_->Get<float>();
        AscendC::LocalTensor<float> bcB = bufBcB_->Get<float>();
        AscendC::LocalTensor<float> tmpb = bufBcTmp_->Get<float>();
        uint32_t dstShape[2] = {(uint32_t)actualNb, (uint32_t)ctAligned};
        uint32_t srcCol[2] = {(uint32_t)actualNb, 1};
        uint32_t srcRow[2] = {1, (uint32_t)ctAligned};
        for (int32_t i = start; i != end; i += step) {
            if (cfg_->tiling->diag == DIAG_NONUNIT) {
                float recip = 1.0f / ubA.GetValue(i * nbAligned + i);
                AscendC::Muls(ubB[i * ctAligned], ubB[i * ctAligned], recip, ctAligned);
                AscendC::PipeBarrier<PIPE_V>();
            }
            int32_t es = cfg_->forward ? i + 1 : 0;
            int32_t ee = cfg_->forward ? actualNb : i;
            int32_t cnt = ee - es;
            if (cnt > 0) {
                AscendC::Broadcast<float, 2, 1>(bcA, ubAt[i * nbAligned], dstShape, srcCol);
                AscendC::Broadcast<float, 2, 0>(bcB, ubB[i * ctAligned], dstShape, srcRow);
                AscendC::Mul(tmpb, bcA, bcB, actualNb * ctAligned);
                AscendC::Sub(ubB[es * ctAligned], ubB[es * ctAligned],
                             tmpb[es * ctAligned], cnt * ctAligned);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Inner-loop triangular solve using Axpy (large nCols path)
    __aicore__ inline void SolveInner(
        AscendC::LocalTensor<float>& ubA, AscendC::LocalTensor<float>& ubB,
        bool swapA, int32_t actualNb, int32_t ctAligned,
        int32_t start, int32_t end, int32_t step)
    {
        int32_t nbAligned = cfg_->nbAligned;
        for (int32_t i = start; i != end; i += step) {
            if (cfg_->tiling->diag == DIAG_NONUNIT) {
                float recip = 1.0f / ubA.GetValue(i * nbAligned + i);
                AscendC::Muls(ubB[i * ctAligned], ubB[i * ctAligned], recip, ctAligned);
                AscendC::PipeBarrier<PIPE_V>();
            }
            int32_t es = cfg_->forward ? i + 1 : 0;
            int32_t ee = cfg_->forward ? actualNb : i;
            for (int32_t k = es; k < ee; k++) {
                float aVal = swapA ? ubA.GetValue(i * nbAligned + k)
                                   : ubA.GetValue(k * nbAligned + i);
                AscendC::Axpy(ubB[k * ctAligned], ubB[i * ctAligned], -aVal, ctAligned);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // Write solved B tile back to GM and write negated Xneg for AIC GEMM
    __aicore__ inline void WriteBackSolved(
        AscendC::LocalTensor<float>& ubB, AscendC::GlobalTensor<float>& gmBT,
        AscendC::GlobalTensor<float>& gmXnegT, int32_t effLdb,
        int32_t panelStart, int32_t actualNb, int32_t cs, int32_t ct, int32_t ctAligned)
    {
        WriteBackSolvedB(ubB, gmBT, effLdb, panelStart, actualNb, cs, ct, ctAligned);

        AscendC::LocalTensor<float> ubNeg = bufNeg_->Get<float>();
        AscendC::Muls(ubNeg, ubB, -1.0f, actualNb * ctAligned);
        AscendC::PipeBarrier<PIPE_V>();
        WriteBackSolvedXneg(ubNeg, gmXnegT, actualNb, cs, ct, ctAligned);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Write solved tile rows to gmBT (aligned or row-by-row fallback)
    __aicore__ inline void WriteBackSolvedB(
        AscendC::LocalTensor<float>& ubB, AscendC::GlobalTensor<float>& gmBT,
        int32_t effLdb, int32_t panelStart, int32_t actualNb,
        int32_t cs, int32_t ct, int32_t ctAligned)
    {
        if (ct % FLOAT_ALIGN == 0) {
            AscendC::DataCopyExtParams bOut((uint16_t)actualNb, (uint32_t)(ct * sizeof(float)),
                (int64_t)((ctAligned - ct) / FLOAT_ALIGN),
                (int64_t)((effLdb - ct) * sizeof(float)), 0);
            AscendC::DataCopyPad(gmBT[panelStart * effLdb + cs], ubB, bOut);
            return;
        }
        AscendC::DataCopyExtParams bOutRow(1, (uint32_t)(ct * sizeof(float)), 0, 0, 0);
        for (int32_t r = 0; r < actualNb; r++) {
            AscendC::DataCopyPad(gmBT[(panelStart + r) * effLdb + cs], ubB[r * ctAligned], bOutRow);
        }
    }

    // Write negated Xneg tile to gmXnegT (aligned or row-by-row fallback)
    __aicore__ inline void WriteBackSolvedXneg(
        AscendC::LocalTensor<float>& ubNeg, AscendC::GlobalTensor<float>& gmXnegT,
        int32_t actualNb, int32_t cs, int32_t ct, int32_t ctAligned)
    {
        int32_t xnegStride = cfg_->splitNcols;
        int32_t pieceStart = cfg_->solveColStart;
        if (cfg_->coopAiv) {
            int32_t splitFactor = cfg_->tiling->splitFactor;
            int32_t pieceWidth = CEIL_ALIGN(cfg_->nCols / splitFactor, FLOAT_ALIGN);
            pieceStart = cfg_->splitId * pieceWidth;
        }
        int32_t localCs = cs - pieceStart;
        if (ct % FLOAT_ALIGN == 0) {
            AscendC::DataCopyExtParams negOut((uint16_t)actualNb, (uint32_t)(ct * sizeof(float)),
                (int64_t)((ctAligned - ct) / FLOAT_ALIGN),
                (int64_t)((xnegStride - ct) * sizeof(float)), 0);
            AscendC::DataCopyPad(gmXnegT[localCs], ubNeg, negOut);
            return;
        }
        for (int32_t r = 0; r < actualNb; r++) {
            AscendC::DataCopyPad(gmXnegT[r * xnegStride + localCs], ubNeg[r * ctAligned],
                {1, (uint16_t)(ct * sizeof(float)), 0, 0});
        }
    }

private:
    const TrsmAivCfg* cfg_;
    BufVecCalc* bufPanelA_; BufVecCalc* bufAt_; BufVecCalc* bufColIdx_;
    BufVecCalc* bufPanelB_; BufVecCalc* bufNeg_; BufVecCalc* bufRankK_;
    BufVecCalc* bufBcA_; BufVecCalc* bufBcB_; BufVecCalc* bufBcTmp_;
};
