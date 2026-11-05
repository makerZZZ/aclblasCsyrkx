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
 * \file ctrsm_batched_kernel_aiv_solver.h
 * \brief Panel 复数三角求解器（按求解方向模板化）：加载对角块A、列分块前代/回代、写回结果与 Xneg。
 *          FORWARD=true  下三角前代（i: 0..actualNb-1，更新 i+1..）
 *          FORWARD=false 上三角回代（i: actualNb-1..0，更新 0..i-1）
 */

#pragma once

#include "ctrsm_batched_kernel_aiv_cfg.h"
#include "ctrsm_batched_kernel_aiv_convert.h"

template <bool FORWARD>
class CtrsmPanelSolver {
public:
    __aicore__ inline CtrsmPanelSolver() {}

    __aicore__ inline void Bind(CtrsmAivCfg* cfg, CtrsmConvert* cvt) { cfg_ = cfg; cvt_ = cvt; }

    // 从AoS源加载Panel对角块A到UB，解交织为实部ubAR和虚部ubAI
    // 优化：actualNb 行合并为一次多行 DMA + 一次同步（替代逐行 DMA），摊薄小矩阵搬运开销
    __aicore__ inline void LoadPanelA(__gm__ float* effA, int32_t effAStride,
                                       AscendC::LocalTensor<float>& ubAR,
                                       AscendC::LocalTensor<float>& ubAI,
                                       int32_t panelStart, int32_t actualNb)
    {
        int32_t nbAligned = cfg_->nbAligned, kDim = cfg_->kDim, gatherChunk = cfg_->gatherChunk;
        AscendC::Duplicate(ubAR, 0.0f, nbAligned * nbAligned);
        AscendC::Duplicate(ubAI, 0.0f, nbAligned * nbAligned);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<float> ubSrc = cfg_->bufPanelB_real->Get<float>();
        AscendC::GlobalTensor<float> gA;
        gA.SetGlobalBuffer(effA, (uint32_t)((int64_t)kDim * effAStride));
        int32_t aosColFloats = 2 * actualNb;
        int32_t rowUB = CEIL_ALIGN(aosColFloats, FLOAT_ALIGN);   // UB 每行对齐步长
        AscendC::LocalTensor<int32_t> offEven = cfg_->bufGatherEven->Get<int32_t>();
        AscendC::LocalTensor<int32_t> offOdd = cfg_->bufGatherOdd->Get<int32_t>();
        int32_t actualNbA = CEIL_ALIGN(actualNb, FLOAT_ALIGN);
        constexpr int32_t PANEL_EVT = 1;
        // 一次多行 DMA：actualNb 行，每行 aosColFloats floats，源行距 effAStride
        int64_t srcOff = (int64_t)panelStart * effAStride + 2 * panelStart;
        AscendC::DataCopyExtParams rp((uint16_t)actualNb, (uint32_t)(aosColFloats * sizeof(float)),
            (uint32_t)((effAStride - aosColFloats) * sizeof(float)),
            (uint32_t)(((rowUB - aosColFloats) * sizeof(float)) / 32), 0);
        AscendC::DataCopyPad(ubSrc, gA[srcOff], rp, {false, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(PANEL_EVT);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(PANEL_EVT);
        for (int32_t r = 0; r < actualNb; r++) {
            AscendC::LocalTensor<float> rowSrc = ubSrc[r * rowUB];
            for (int32_t c = 0; c < actualNbA; c += gatherChunk) {
                int32_t len = (gatherChunk < actualNbA - c) ? gatherChunk : (actualNbA - c);
                AscendC::Gather(ubAR[r * nbAligned + c], rowSrc[c * 2],
                    offEven.ReinterpretCast<uint32_t>(), 0, len);
            }
            for (int32_t c = 0; c < actualNbA; c += gatherChunk) {
                int32_t len = (gatherChunk < actualNbA - c) ? gatherChunk : (actualNbA - c);
                AscendC::Gather(ubAI[r * nbAligned + c], rowSrc[c * 2],
                    offOdd.ReinterpretCast<uint32_t>(), 0, len);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        if (cfg_->conjA && cfg_->useOrigA) {
            AscendC::Muls(ubAI, ubAI, -1.0f, nbAligned * nbAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // 对一个Panel执行复数三角求解：按列分块求解→写回。ubAR/ubAI 须已由 LoadPanelA 填充
    __aicore__ inline void PanelTrsv(__gm__ float* effA, int32_t effAStride,
                                      __gm__ float* effBR, __gm__ float* effBI, int32_t effLdb,
                                      __gm__ float* gmXR, __gm__ float* gmXI,
                                      int32_t panelStart, int32_t actualNb, bool needXneg)
    {
        int32_t kDim = cfg_->kDim, nb = cfg_->nb, nColsAligned = cfg_->nColsAligned, colTile = cfg_->colTile;
        AscendC::LocalTensor<float> ubAR = cfg_->bufPanelA_real->Get<float>();
        AscendC::LocalTensor<float> ubAI = cfg_->bufPanelA_imag->Get<float>();
        AscendC::GlobalTensor<float> gBR, gBI, gXR, gXI;
        gBR.SetGlobalBuffer(effBR, (uint32_t)((int64_t)kDim * effLdb));
        gBI.SetGlobalBuffer(effBI, (uint32_t)((int64_t)kDim * effLdb));
        gXR.SetGlobalBuffer(gmXR, (uint32_t)((int64_t)2 * nb * nColsAligned));
        gXI.SetGlobalBuffer(gmXI, (uint32_t)((int64_t)2 * nb * nColsAligned));
        for (int32_t cs = cfg_->colStart; cs < cfg_->colEnd; cs += colTile) {
            int32_t ct = (cs + colTile <= cfg_->colEnd) ? colTile : (cfg_->colEnd - cs);
            int32_t ctA = CEIL_ALIGN(ct, FLOAT_ALIGN);
            SolveAndWriteTile(ubAR, ubAI, gBR, gBI, gXR, gXI,
                              effLdb, panelStart, actualNb, cs, ct, ctA, needXneg);
        }
    }

    // 求解一个列分块：加载B→求解→UB内准备所有写出数据→统一MTE3搬出
    // Counter 模式提升到 solve 循环外部，整个求解阶段仅一对 SetMaskCount/SetMaskNorm
    __aicore__ inline void SolveAndWriteTile(
        AscendC::LocalTensor<float>& ubAR, AscendC::LocalTensor<float>& ubAI,
        AscendC::GlobalTensor<float>& gBR, AscendC::GlobalTensor<float>& gBI,
        AscendC::GlobalTensor<float>& gXR, AscendC::GlobalTensor<float>& gXI,
        int32_t effLdb, int32_t panelStart, int32_t actualNb,
        int32_t cs, int32_t ct, int32_t ctA, bool needXneg)
    {
        AscendC::LocalTensor<float> ubB = cfg_->bufPanelB_real->Get<float>();
        LoadBTile(ubB, gBR, gBI, effLdb, panelStart, actualNb, cs, ct, ctA);
        AscendC::PipeBarrier<PIPE_ALL>();
        SolveTileInner(ubAR, ubAI, ubB, actualNb, ctA);
        WriteBackTile(ubB, gBR, gBI, gXR, gXI, effLdb, panelStart, actualNb, cs, ct, ctA, needXneg);
    }

    // 求解循环：前代/回代核心
    __aicore__ inline void SolveTileInner(
        AscendC::LocalTensor<float>& ubAR, AscendC::LocalTensor<float>& ubAI,
        AscendC::LocalTensor<float>& ubB, int32_t actualNb, int32_t ctA)
    {
        int32_t start = FORWARD ? 0 : (actualNb - 1);
        int32_t end = FORWARD ? actualNb : -1;
        int32_t step = FORWARD ? 1 : -1;
        int32_t w2 = 2 * ctA;
        uint8_t repCtA = (uint8_t)((ctA + 63) / 64);
        uint8_t rep2CtA = (uint8_t)((w2 + 63) / 64);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(w2);
        for (int32_t i = start; i != end; i += step) {
            if (cfg_->td->diag == DIAG_NONUNIT) SolveDiagInter(ubAR, ubAI, ubB, i, ctA, repCtA, rep2CtA);
            UpdateTrailingRowsInter(ubAR, ubAI, ubB, i, actualNb, ctA, rep2CtA);
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }

    // 写回：解交织ubB→ubBR/ubBI，准备Xneg，统一MTE3搬出
    __aicore__ inline void WriteBackTile(
        AscendC::LocalTensor<float>& ubB,
        AscendC::GlobalTensor<float>& gBR, AscendC::GlobalTensor<float>& gBI,
        AscendC::GlobalTensor<float>& gXR, AscendC::GlobalTensor<float>& gXI,
        int32_t effLdb, int32_t panelStart, int32_t actualNb,
        int32_t cs, int32_t ct, int32_t ctA, bool needXneg)
    {
        // 解交织 ubB [R|I|R|I|...] → 连续 ubBR / ubBI，用 strided DataCopy 替代 for 循环
        AscendC::LocalTensor<float> ubBR = cfg_->bufPanelB_imag->Get<float>();
        AscendC::LocalTensor<float> ubBI = cfg_->bufNeg_imag->Get<float>();
        if (((ctA * (int32_t)sizeof(float)) % 32 == 0)) {
            uint16_t blockLen = (uint16_t)(ctA * sizeof(float) / 32);
            uint16_t srcStride = blockLen;  // 源行跨 2*ctA floats，burst=ctA floats → gap=ctA → stride=ctA/8 blocks
            uint16_t dstStride = 0;         // 目标连续无 gap
            AscendC::DataCopyParams deintP{(uint8_t)actualNb, blockLen, srcStride, dstStride};
            AscendC::DataCopy(ubBR, ubB, deintP);
            AscendC::DataCopy(ubBI, ubB[ctA], deintP);
        } else {
            int64_t srcOff = 0;
            int32_t dstOff = 0;
            for (int32_t r = 0; r < actualNb; r++) {
                AscendC::DataCopy(ubBR[dstOff], ubB[srcOff], ctA);
                AscendC::DataCopy(ubBI[dstOff], ubB[srcOff + ctA], ctA);
                srcOff += 2 * ctA;
                dstOff += ctA;
            }
        }
        if (needXneg) {
            AscendC::LocalTensor<float> negR = cfg_->bufNeg_real->Get<float>();
            AscendC::LocalTensor<float> negIN = cfg_->bufNeg_imag_neg->Get<float>();
            int32_t negLen = actualNb * ctA;
            uint8_t negRep = (uint8_t)((negLen + 63) / 64);
            AscendC::SetMaskCount();
            AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(negLen);
            AscendC::Muls<float, false>(negR, ubBR, -1.0f, AscendC::MASK_PLACEHOLDER, negRep, {1, 1, 8, 8});
            AscendC::Muls<float, false>(negIN, ubBI, -1.0f, AscendC::MASK_PLACEHOLDER, negRep, {1, 1, 8, 8});
            AscendC::SetMaskNorm();
            AscendC::ResetMask();
        }
        // 所有 Vec 计算完成，一次性通知 MTE3 开始写出
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        StoreBTile(ubBR, ubBI, gBR, gBI, effLdb, panelStart, actualNb, cs, ct, ctA);
        if (needXneg) {
            WriteBackXnegDma(ubBI, gXR, gXI, actualNb, cs, ct, ctA);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 搬入 B 列分块到 ubB 交错布局
    __aicore__ inline void LoadBTile(
        AscendC::LocalTensor<float>& ubB,
        AscendC::GlobalTensor<float>& gBR, AscendC::GlobalTensor<float>& gBI,
        int32_t effLdb, int32_t panelStart, int32_t actualNb,
        int32_t cs, int32_t ct, int32_t ctA)
    {
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        int32_t dstGapFloats = 2 * ctA - ct;
        if (((dstGapFloats * (int32_t)sizeof(float)) % 32) == 0) {
            AscendC::DataCopyExtParams bIn((uint16_t)actualNb, (uint32_t)(ct * sizeof(float)),
                (uint32_t)((effLdb - ct) * sizeof(float)),
                (uint32_t)((dstGapFloats * sizeof(float)) / 32), 0);
            AscendC::DataCopyPad(ubB, gBR[(int64_t)panelStart * effLdb + cs], bIn, pad);
            AscendC::DataCopyPad(ubB[ctA], gBI[(int64_t)panelStart * effLdb + cs], bIn, pad);
        } else {
            AscendC::DataCopyExtParams rowP(1, (uint32_t)(ct * sizeof(float)), 0, 0, 0);
            int64_t ubOff = 0;
            int64_t gmOff = (int64_t)panelStart * effLdb + cs;
            for (int32_t r = 0; r < actualNb; r++) {
                AscendC::DataCopyPad(ubB[ubOff], gBR[gmOff], rowP, pad);
                AscendC::DataCopyPad(ubB[ubOff + ctA], gBI[gmOff], rowP, pad);
                ubOff += 2 * ctA;
                gmOff += effLdb;
            }
        }
    }

    // 从连续布局 ubBR/ubBI strided DMA 写出 B 到 GM
    __aicore__ inline void StoreBTile(
        AscendC::LocalTensor<float>& ubBR, AscendC::LocalTensor<float>& ubBI,
        AscendC::GlobalTensor<float>& gBR, AscendC::GlobalTensor<float>& gBI,
        int32_t effLdb, int32_t panelStart, int32_t actualNb,
        int32_t cs, int32_t ct, int32_t ctA)
    {
        int64_t gmBOff = (int64_t)panelStart * effLdb + cs;
        uint32_t ubGap32B = (uint32_t)(((ctA - ct) * sizeof(float)) / 32);
        uint32_t gmGapBytes = (uint32_t)((effLdb - ct) * sizeof(float));
        if (((ctA - ct) * (int32_t)sizeof(float)) % 32 == 0) {
            AscendC::DataCopyExtParams bOut((uint16_t)actualNb, (uint32_t)(ct * sizeof(float)),
                ubGap32B, gmGapBytes, 0);
            AscendC::DataCopyPad(gBR[gmBOff], ubBR, bOut);
            AscendC::DataCopyPad(gBI[gmBOff], ubBI, bOut);
        } else {
            AscendC::DataCopyExtParams bOut(1, (uint32_t)(ct * sizeof(float)), 0, 0, 0);
            int32_t ubOff = 0;
            for (int32_t r = 0; r < actualNb; r++) {
                AscendC::DataCopyPad(gBR[gmBOff], ubBR[ubOff], bOut);
                AscendC::DataCopyPad(gBI[gmBOff], ubBI[ubOff], bOut);
                ubOff += ctA;
                gmBOff += effLdb;
            }
        }
    }

    // Xneg 纯 MTE3 写出（negR/negIN 已在 UB 中准备好）
    __aicore__ inline void WriteBackXnegDma(
        AscendC::LocalTensor<float>& ubBI,
        AscendC::GlobalTensor<float>& gXR, AscendC::GlobalTensor<float>& gXI,
        int32_t actualNb, int32_t cs, int32_t ct, int32_t ctA)
    {
        int32_t nColsAligned = cfg_->nColsAligned;
        AscendC::LocalTensor<float> negR = cfg_->bufNeg_real->Get<float>();
        AscendC::LocalTensor<float> negIN = cfg_->bufNeg_imag_neg->Get<float>();
        uint32_t xUbGap32B = (uint32_t)(((ctA - ct) * sizeof(float)) / 32);
        uint32_t xGmGapBytes = (uint32_t)((2 * nColsAligned - ct) * sizeof(float));
        if (((ctA - ct) * (int32_t)sizeof(float)) % 32 == 0) {
            AscendC::DataCopyExtParams xOut((uint16_t)actualNb, (uint32_t)(ct * sizeof(float)),
                xUbGap32B, xGmGapBytes, 0);
            AscendC::DataCopyPad(gXR[cs], negR, xOut);
            AscendC::DataCopyPad(gXR[nColsAligned + cs], ubBI, xOut);
            AscendC::DataCopyPad(gXI[cs], negIN, xOut);
            AscendC::DataCopyPad(gXI[nColsAligned + cs], negR, xOut);
        } else {
            AscendC::DataCopyExtParams xOut(1, (uint32_t)(ct * sizeof(float)), 0, 0, 0);
            int64_t xOff = cs;
            int32_t ubOff = 0;
            for (int32_t r = 0; r < actualNb; r++) {
                AscendC::DataCopyPad(gXR[xOff], negR[ubOff], xOut);
                AscendC::DataCopyPad(gXR[xOff + nColsAligned], ubBI[ubOff], xOut);
                AscendC::DataCopyPad(gXI[xOff], negIN[ubOff], xOut);
                AscendC::DataCopyPad(gXI[xOff + nColsAligned], negR[ubOff], xOut);
                xOff += 2 * nColsAligned;
                ubOff += ctA;
            }
        }
    }

    // 交错对角除法：X[i]=B[i]/A[i,i]=B[i]·conj(A)/|A|²
    // Counter 模式已由调用者设置，仅需切换 mask 长度
    __aicore__ inline void SolveDiagInter(
        AscendC::LocalTensor<float>& ubAR, AscendC::LocalTensor<float>& ubAI,
        AscendC::LocalTensor<float>& ubB, int32_t i, int32_t ctA,
        uint8_t repCtA, uint8_t rep2CtA)
    {
        int32_t nbAligned = cfg_->nbAligned;
        int64_t o = (int64_t)i * 2 * ctA;
        AscendC::LocalTensor<float> row = ubB[o];
        AscendC::LocalTensor<float> bR = ubB[o];
        AscendC::LocalTensor<float> bI = ubB[o + ctA];
        float aRe = ubAR.GetValue(i * nbAligned + i);
        float aIm = ubAI.GetValue(i * nbAligned + i);
        float denom = aRe * aRe + aIm * aIm;
        float invRe = aRe / denom;
        float invIm = -aIm / denom;
        AscendC::LocalTensor<float> src24 = cfg_->bufNeg_imag_neg->Get<float>();
        AscendC::DataCopy(src24, bI, ctA);
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(ctA);
        AscendC::Muls<float, false>(src24[ctA], bR, -1.0f, AscendC::MASK_PLACEHOLDER, repCtA, {1, 1, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2 * ctA);
        AscendC::Muls<float, false>(row, row, invRe, AscendC::MASK_PLACEHOLDER, rep2CtA, {1, 1, 8, 8});
        AscendC::Axpy<float, float, false>(row, src24, -invIm, AscendC::MASK_PLACEHOLDER, rep2CtA, {1, 1, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 交错尾行更新：B[k] -= A[k,i]*X[i]
    // Counter 模式已由调用者设置（w2 长度），直接使用 isSetMask=false
    __aicore__ inline void UpdateTrailingRowsInter(
        AscendC::LocalTensor<float>& ubAR, AscendC::LocalTensor<float>& ubAI,
        AscendC::LocalTensor<float>& ubB, int32_t i, int32_t actualNb, int32_t ctA,
        uint8_t repTime)
    {
        int32_t nbAligned = cfg_->nbAligned;
        int32_t es = FORWARD ? i + 1 : 0;
        int32_t ee = FORWARD ? actualNb : i;
        if (es >= ee) return;
        int64_t oi = (int64_t)i * 2 * ctA;
        AscendC::LocalTensor<float> rowI = ubB[oi];
        AscendC::LocalTensor<float> src24 = cfg_->bufNeg_real->Get<float>();
        AscendC::DataCopy(src24, ubB[oi + ctA], ctA);
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(ctA);
        AscendC::Muls<float, false>(src24[ctA], ubB[oi], -1.0f, AscendC::MASK_PLACEHOLDER,
            (uint8_t)((ctA + 63) / 64), {1, 1, 8, 8});
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(2 * ctA);
        int32_t w2 = 2 * ctA;
        int32_t k = es;
        int32_t idx = es * nbAligned + i;
        int64_t rowOff = (int64_t)es * 2 * ctA;
        UpdateTrailingUnrolled(ubAR, ubAI, ubB, rowI, src24, k, idx, rowOff, ee, nbAligned, w2, repTime);
        UpdateTrailingRemainder(ubAR, ubAI, ubB, rowI, src24, k, idx, rowOff, ee, nbAligned, w2, repTime);
    }

    // 尾行更新4x展开部分
    __aicore__ inline void UpdateTrailingUnrolled(
        AscendC::LocalTensor<float>& ubAR, AscendC::LocalTensor<float>& ubAI,
        AscendC::LocalTensor<float>& ubB, AscendC::LocalTensor<float>& rowI,
        AscendC::LocalTensor<float>& src24,
        int32_t& k, int32_t& idx, int64_t& rowOff, int32_t ee,
        int32_t nbAligned, int32_t w2, uint8_t repTime)
    {
        for (; k + 3 < ee; k += 4) {
            float sRe0 = ubAR.GetValue(idx);
            float sIm0 = ubAI.GetValue(idx);
            float sRe1 = ubAR.GetValue(idx + nbAligned);
            float sIm1 = ubAI.GetValue(idx + nbAligned);
            float sRe2 = ubAR.GetValue(idx + 2 * nbAligned);
            float sIm2 = ubAI.GetValue(idx + 2 * nbAligned);
            float sRe3 = ubAR.GetValue(idx + 3 * nbAligned);
            float sIm3 = ubAI.GetValue(idx + 3 * nbAligned);
            AscendC::LocalTensor<float> rowK0 = ubB[rowOff];
            AscendC::LocalTensor<float> rowK1 = ubB[rowOff + w2];
            AscendC::LocalTensor<float> rowK2 = ubB[rowOff + 2 * w2];
            AscendC::LocalTensor<float> rowK3 = ubB[rowOff + 3 * w2];
            AscendC::Axpy<float, float, false>(rowK0, rowI,  -sRe0, AscendC::MASK_PLACEHOLDER, repTime, {1, 1, 8, 8});
            AscendC::Axpy<float, float, false>(rowK0, src24,  sIm0, AscendC::MASK_PLACEHOLDER, repTime, {1, 1, 8, 8});
            AscendC::Axpy<float, float, false>(rowK1, rowI,  -sRe1, AscendC::MASK_PLACEHOLDER, repTime, {1, 1, 8, 8});
            AscendC::Axpy<float, float, false>(rowK1, src24,  sIm1, AscendC::MASK_PLACEHOLDER, repTime, {1, 1, 8, 8});
            AscendC::Axpy<float, float, false>(rowK2, rowI,  -sRe2, AscendC::MASK_PLACEHOLDER, repTime, {1, 1, 8, 8});
            AscendC::Axpy<float, float, false>(rowK2, src24,  sIm2, AscendC::MASK_PLACEHOLDER, repTime, {1, 1, 8, 8});
            AscendC::Axpy<float, float, false>(rowK3, rowI,  -sRe3, AscendC::MASK_PLACEHOLDER, repTime, {1, 1, 8, 8});
            AscendC::Axpy<float, float, false>(rowK3, src24,  sIm3, AscendC::MASK_PLACEHOLDER, repTime, {1, 1, 8, 8});
            idx += 4 * nbAligned;
            rowOff += 4 * w2;
        }
    }

    // 尾行更新余数部分
    __aicore__ inline void UpdateTrailingRemainder(
        AscendC::LocalTensor<float>& ubAR, AscendC::LocalTensor<float>& ubAI,
        AscendC::LocalTensor<float>& ubB, AscendC::LocalTensor<float>& rowI,
        AscendC::LocalTensor<float>& src24,
        int32_t& k, int32_t& idx, int64_t& rowOff, int32_t ee,
        int32_t nbAligned, int32_t w2, uint8_t repTime)
    {
        for (; k < ee; k++) {
            float sRe = ubAR.GetValue(idx);
            float sIm = ubAI.GetValue(idx);
            AscendC::LocalTensor<float> rowK = ubB[rowOff];
            AscendC::Axpy<float, float, false>(rowK, rowI,  -sRe, AscendC::MASK_PLACEHOLDER, repTime, {1, 1, 8, 8});
            AscendC::Axpy<float, float, false>(rowK, src24,  sIm, AscendC::MASK_PLACEHOLDER, repTime, {1, 1, 8, 8});
            idx += nbAligned;
            rowOff += w2;
        }
    }

private:
    CtrsmAivCfg* cfg_;
    CtrsmConvert* cvt_;
};
