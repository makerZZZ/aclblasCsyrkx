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
 * \file ctrsm_batched_kernel_aiv_canon_b.h
 * \brief B 矩阵规范化与回写器（按左右乘模板化）：AoS->SoA 构建、alpha 缩放，
 *        以及求解结果回写（Left 逐面板行回写 / Right 转置后整体交织回写）。
 */

#pragma once

#include "ctrsm_batched_kernel_aiv_cfg.h"
#include "ctrsm_batched_kernel_aiv_convert.h"

template <bool RIGHT>
class CtrsmCanonB {
public:
    __aicore__ inline CtrsmCanonB() {}

    __aicore__ inline void Bind(CtrsmAivCfg* cfg, CtrsmConvert* cvt) { cfg_ = cfg; cvt_ = cvt; }

    // 构建补零后的B矩阵SoA工作区：清零→解交织→alpha缩放（按buildRowStart/End分行）
    __aicore__ inline void BuildPaddedB(__gm__ float* gmB,
                                         __gm__ float* gmBc_real, __gm__ float* gmBc_imag,
                                         __gm__ float* gmTemp)
    {
        int32_t nColsAligned = cfg_->nColsAligned;
        if (cfg_->padOn) {
            cvt_->ZeroGmRows(gmBc_real + (int64_t)cfg_->buildRowStart * nColsAligned,
                       cfg_->buildRowEnd - cfg_->buildRowStart, nColsAligned, nColsAligned);
            cvt_->ZeroGmRows(gmBc_imag + (int64_t)cfg_->buildRowStart * nColsAligned,
                       cfg_->buildRowEnd - cfg_->buildRowStart, nColsAligned, nColsAligned);
        }
        if (!RIGHT) {
            DeinterleaveBLeft(gmB, gmBc_real, gmBc_imag);
        } else {
            DeinterleaveBRight(gmB, gmBc_real, gmBc_imag, gmTemp);
        }
        float aRe = cfg_->td->alphaReal;
        float aIm = cfg_->td->alphaImag;
        if (aRe != 1.0f || aIm != 0.0f) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AlphaScale(gmBc_real, gmBc_imag, nColsAligned, aRe, aIm);
        }
    }

    // 将SoA结果写回原始AoS格式的B矩阵（仅Right模式调用）
    // rowBeg/rowEnd 限定输出行范围（dual-AIV 两核各写一半）
    __aicore__ inline void WriteBackPaddedB(__gm__ float* gmBcR, __gm__ float* gmBcI,
                                             __gm__ float* gmB, __gm__ float* gmTemp,
                                             int32_t rowBeg, int32_t rowEnd)
    {
        InterleaveBRight(gmBcR, gmBcI, gmB, gmTemp, rowBeg, rowEnd);
    }

    // Right dual 前端阶段1：清零本核 gmBc 行段(buildRow) + 解交织 [tBeg,tEnd) 行到 temp
    __aicore__ inline void BuildBRightPhase1(__gm__ float* gmB,
        __gm__ float* gmBc_real, __gm__ float* gmBc_imag, __gm__ float* gmTemp,
        int32_t tBeg, int32_t tEnd)
    {
        int32_t nColsAligned = cfg_->nColsAligned;
        if (cfg_->padOn) {
            cvt_->ZeroGmRows(gmBc_real + (int64_t)cfg_->buildRowStart * nColsAligned,
                       cfg_->buildRowEnd - cfg_->buildRowStart, nColsAligned, nColsAligned);
            cvt_->ZeroGmRows(gmBc_imag + (int64_t)cfg_->buildRowStart * nColsAligned,
                       cfg_->buildRowEnd - cfg_->buildRowStart, nColsAligned, nColsAligned);
        }
        DeinterleaveBRightToTemp(gmB, gmTemp, tBeg, tEnd);
    }

    // Right dual 前端阶段2：转置 temp→gmBc 本核输出行段 [cbBeg,cbEnd) + alpha 缩放本核行段
    __aicore__ inline void BuildBRightPhase2(__gm__ float* gmBc_real, __gm__ float* gmBc_imag,
        __gm__ float* gmTemp, int32_t cbBeg, int32_t cbEnd)
    {
        TransposeBTempToBc(gmBc_real, gmBc_imag, gmTemp, cbBeg, cbEnd);
        float aRe = cfg_->td->alphaReal;
        float aIm = cfg_->td->alphaImag;
        if (aRe != 1.0f || aIm != 0.0f) {
            AlphaScale(gmBc_real, gmBc_imag, cfg_->nColsAligned, aRe, aIm);
        }
    }

    // 仅 alpha 缩放（buildRowStart/End 已在外部设置）
    __aicore__ inline void BuildBRightPhase2Alpha(__gm__ float* gmBc_real, __gm__ float* gmBc_imag,
        int32_t effLdb, float aRe, float aIm)
    {
        AlphaScale(gmBc_real, gmBc_imag, effLdb, aRe, aIm);
    }

    // 将已求解的nb行从SoA数据交织回写到用户B矩阵（Left模式逐面板回写）
    // 单列分块时直接从 UB 读取（省 MTE2），多列分块退回 GM 搬入
    __aicore__ inline void WriteBackPanelRows(__gm__ float* gmBcR, __gm__ float* gmBcI,
                                              __gm__ float* gmB,
                                              int32_t panelStart, int32_t actualNb)
    {
        int32_t nColsAligned = cfg_->nColsAligned;
        int32_t mOrig = cfg_->td->m, nOrig = cfg_->td->n;
        int32_t endRow = panelStart + actualNb;
        if (endRow > mOrig) endRow = mOrig;
        int32_t localNCols = cfg_->localNCols;
        int32_t clipEnd = (localNCols < nOrig) ? localNCols : nOrig;
        int32_t wbColStart = (cfg_->colStart < clipEnd) ? cfg_->colStart : clipEnd;
        int32_t wbColEnd = (cfg_->colEnd < clipEnd) ? cfg_->colEnd : clipEnd;
        int32_t wbCount = wbColEnd - wbColStart;
        if (wbCount <= 0) return;
        int32_t gmBColStart = wbColStart + cfg_->nColsOffset;
        if (cfg_->colTile >= nColsAligned) {
            WriteBackPanelRowsFromUB(gmB, panelStart, endRow, gmBColStart, wbCount);
        } else {
            WriteBackPanelRowsFromGM(gmBcR, gmBcI, gmB, panelStart, endRow, wbColStart, wbCount, gmBColStart);
        }
    }

    // 单列分块：直接从 UB 已有的 ubBR/ubBI 读取，批量交织后 1 条 strided DMA 写出
    __aicore__ inline void WriteBackPanelRowsFromUB(__gm__ float* gmB,
        int32_t panelStart, int32_t endRow, int32_t wbColStart, int32_t wbCount)
    {
        int32_t mOrig = cfg_->td->m;
        int32_t wbCountA = CEIL_ALIGN(wbCount, FLOAT_ALIGN);
        AscendC::LocalTensor<float> ubBR = cfg_->bufPanelB_imag->Get<float>();
        AscendC::LocalTensor<float> ubBI = cfg_->bufNeg_imag->Get<float>();
        AscendC::LocalTensor<float> ubR = cfg_->bufPanelB_real->Get<float>();
        AscendC::LocalTensor<float> ub = cfg_->bufRow->Get<float>();
        int32_t rowCount = endRow - panelStart;
        AscendC::DataCopy(ubR, ubBI, rowCount * wbCountA);
        AscendC::PipeBarrier<PIPE_V>();
        cvt_->BuildInterleaveOffsets(wbCount);
        AscendC::LocalTensor<int32_t> offInter = cfg_->bufNeg_imag->Get<int32_t>();
        // 批量交织：16 行全部 InterleaveRow 到 bufNeg_real（作为连续大 buffer 使用）
        // bufNeg_real 大小 = nb*colTile = 1024f，需要 rowCount*wbCount*2 = 16*128 = 2048f
        // 不够！改用 bufPanelB_real 的后半段（4096f 总量，前 rowCount*wbCountA 已被 ubR 占）
        // ubR 占 rowCount*wbCountA = 16*64 = 1024f，bufPanelB_real 剩余 4096-1024 = 3072f >= 2048f ✓
        int32_t outRowFloats = wbCount * 2;
        int32_t outRowAligned = CEIL_ALIGN(outRowFloats, FLOAT_ALIGN);
        AscendC::LocalTensor<float> ubOutAll = cfg_->bufPanelB_real->Get<float>();
        int32_t outBase = rowCount * wbCountA;  // ubR 占用的尾部之后
        AscendC::LocalTensor<float> rowReal = cfg_->bufNeg_imag_neg->Get<float>();
        for (int32_t i = 0; i < rowCount; i++) {
            int32_t rowUbOff = i * wbCountA;
            AscendC::DataCopy(rowReal, ubBR[rowUbOff], wbCountA);
            AscendC::DataCopy(ub, ubR[rowUbOff], wbCountA);
            AscendC::LocalTensor<float> ubOutRow = ubOutAll[outBase + i * outRowAligned];
            cvt_->InterleaveRow(rowReal, ub, ubOutRow, offInter, wbCount);
        }
        // 统一 strided DMA 写出
        AscendC::GlobalTensor<float> gB;
        gB.SetGlobalBuffer(gmB, (uint32_t)((int64_t)mOrig * cfg_->td->ldb * 2));
        int64_t gbOff = (int64_t)panelStart * cfg_->td->ldb * 2 + wbColStart * 2;
        uint32_t ubOutGap32B = (uint32_t)(((outRowAligned - outRowFloats) * sizeof(float)) / 32);
        uint32_t gmGapBytes = (uint32_t)((cfg_->td->ldb * 2 - outRowFloats) * sizeof(float));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        if (((outRowAligned - outRowFloats) * (int32_t)sizeof(float)) % 32 == 0) {
            AscendC::DataCopyExtParams wbOut((uint16_t)rowCount,
                (uint32_t)(outRowFloats * sizeof(float)), ubOutGap32B, gmGapBytes, 0);
            AscendC::DataCopyPad(gB[gbOff], ubOutAll[outBase], wbOut);
        } else {
            AscendC::DataCopyExtParams wbOut(1, (uint32_t)(outRowFloats * sizeof(float)), 0, 0, 0);
            for (int32_t i = 0; i < rowCount; i++) {
                AscendC::DataCopyPad(gB[gbOff], ubOutAll[outBase + i * outRowAligned], wbOut);
                gbOff += cfg_->td->ldb * 2;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 多列分块：从 GM (gmBc) 搬入 SoA 数据再交织写出
    // 当 wbCount 超出 UB 容量时按列分 chunk 处理
    __aicore__ inline void WriteBackPanelRowsFromGM(__gm__ float* gmBcR, __gm__ float* gmBcI,
        __gm__ float* gmB, int32_t panelStart, int32_t endRow,
        int32_t wbColStart, int32_t wbCount, int32_t gmBColStart)
    {
        int32_t kDim = cfg_->kDim, nColsAligned = cfg_->nColsAligned;
        int32_t mOrig = cfg_->td->m;
        AscendC::LocalTensor<float> ub = cfg_->bufRow->Get<float>();
        AscendC::LocalTensor<float> ubR = cfg_->bufPanelB_real->Get<float>();
        AscendC::LocalTensor<float> ubOut = cfg_->bufNeg_real->Get<float>();
        AscendC::GlobalTensor<float> gBcR, gBcI, gB;
        gBcR.SetGlobalBuffer(gmBcR, (uint32_t)((int64_t)kDim * nColsAligned));
        gBcI.SetGlobalBuffer(gmBcI, (uint32_t)((int64_t)kDim * nColsAligned));
        gB.SetGlobalBuffer(gmB, (uint32_t)((int64_t)mOrig * cfg_->td->ldb * 2));
        constexpr int32_t EVT_ID = 0;
        int32_t ldbStride = cfg_->td->ldb * 2;

        // InterleaveRow 需要: ubR 容量 >= 2*countA, ubOut 容量 >= 2*countA
        // countA = CEIL_ALIGN(chunkCols, FLOAT_ALIGN)
        // ubR = bufPanelB_real (srcBufFloats), ubOut = bufNeg_real (nb*colTile)
        int32_t maxColsR = cfg_->srcBufFloats / 2;
        int32_t maxColsOut = cfg_->nb * cfg_->colTile / 2;
        int32_t maxChunkCols = (maxColsR < maxColsOut) ? maxColsR : maxColsOut;
        maxChunkCols = (maxChunkCols / FLOAT_ALIGN) * FLOAT_ALIGN;
        if (maxChunkCols < FLOAT_ALIGN) maxChunkCols = FLOAT_ALIGN;

        for (int32_t cs = 0; cs < wbCount; cs += maxChunkCols) {
            int32_t chunkCols = (maxChunkCols < wbCount - cs) ? maxChunkCols : (wbCount - cs);
            cvt_->BuildInterleaveOffsets(chunkCols);
            AscendC::LocalTensor<int32_t> offInter = cfg_->bufNeg_imag->Get<int32_t>();
            int64_t gbOff = (int64_t)panelStart * ldbStride + (gmBColStart + cs) * 2;
            int64_t bcOff = (int64_t)panelStart * nColsAligned + wbColStart + cs;
            for (int32_t i = panelStart; i < endRow; i++) {
                AscendC::DataCopyPad(ubR, gBcR[bcOff],
                    {1, (uint32_t)(chunkCols * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
                AscendC::DataCopyPad(ub, gBcI[bcOff],
                    {1, (uint32_t)(chunkCols * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVT_ID);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVT_ID);
                if (i > panelStart) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVT_ID);
                }
                cvt_->InterleaveRow(ubR, ub, ubOut, offInter, chunkCols);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVT_ID);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVT_ID);
                AscendC::DataCopyPad(gB[gbOff], ubOut,
                    {1, (uint32_t)(chunkCols * 2 * sizeof(float)), 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVT_ID);
                gbOff += ldbStride;
                bcOff += nColsAligned;
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVT_ID);
        }
    }

private:
    // 将SoA工作空间逐行清零已提取到 CtrsmConvert::ZeroGmRows

    // Deinterleave helper methods split to separate file
    #include "ctrsm_batched_kernel_aiv_canon_b_deinterleave.h"

    // 阶段2：将temp转置到gmBc工作区，仅产出[cbBeg,cbEnd)输出行（=temp列，dual两核各转一段）
    __aicore__ inline void TransposeBTempToBc(__gm__ float* gmBcR, __gm__ float* gmBcI,
                                              __gm__ float* gmTemp, int32_t cbBeg, int32_t cbEnd)
    {
        int32_t kDim = cfg_->kDim, kDimOrig = cfg_->kDimOrig;
        int32_t nColsAligned = cfg_->nColsAligned, nColsOrig = cfg_->nColsOrig;
        if (cbEnd > kDimOrig) cbEnd = kDimOrig;
        if (cbBeg < 0) cbBeg = 0;
        if (cbBeg >= cbEnd) return;
        __gm__ float* tempR = gmTemp;
        __gm__ float* tempI = gmTemp + (int64_t)nColsOrig * kDim;
        cvt_->TiledTransposePair(tempR, tempI, nColsOrig, kDimOrig, kDim, gmBcR, gmBcI, nColsAligned,
                                 cbBeg, cbEnd);
    }

    // 对B矩阵乘以复数alpha系数（按buildRow范围分行处理）
    __aicore__ inline void AlphaScale(__gm__ float* gmBcR, __gm__ float* gmBcI,
                                       int32_t effLdb, float aRe, float aIm)
    {
        int32_t kDim = cfg_->kDim, nColsAligned = cfg_->nColsAligned, colTile = cfg_->colTile;
        AscendC::LocalTensor<float> ubR = cfg_->bufPanelB_real->Get<float>();
        AscendC::LocalTensor<float> ubI = cfg_->bufPanelB_imag->Get<float>();
        AscendC::LocalTensor<float> ubTmp = cfg_->bufRankK->Get<float>();
        AscendC::GlobalTensor<float> gR, gI;
        gR.SetGlobalBuffer(gmBcR, (uint32_t)((int64_t)kDim * effLdb));
        gI.SetGlobalBuffer(gmBcI, (uint32_t)((int64_t)kDim * effLdb));
        int32_t ct = (colTile < nColsAligned) ? colTile : nColsAligned;
        for (int32_t i = cfg_->buildRowStart; i < cfg_->buildRowEnd; i++) {
            for (int32_t cs = 0; cs < nColsAligned; cs += ct) {
                AlphaScaleChunk(gR, gI, ubR, ubI, ubTmp, effLdb, i, cs, ct, aRe, aIm);
            }
        }
    }

    // 对一个分块执行复数alpha缩放：newR = aRe*R - aIm*I, newI = aRe*I + aIm*R
    __aicore__ inline void AlphaScaleChunk(
        AscendC::GlobalTensor<float>& gR, AscendC::GlobalTensor<float>& gI,
        AscendC::LocalTensor<float>& ubR, AscendC::LocalTensor<float>& ubI,
        AscendC::LocalTensor<float>& ubTmp,
        int32_t effLdb, int32_t row, int32_t cs, int32_t ct, float aRe, float aIm)
    {
        int32_t nColsAligned = cfg_->nColsAligned;
        int32_t cw = (cs + ct <= nColsAligned) ? ct : (nColsAligned - cs);
        int32_t cwA = CEIL_ALIGN(cw, FLOAT_ALIGN);
        AscendC::DataCopyExtParams rp(1, (uint32_t)(cw * sizeof(float)), 0, 0, 0);
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(ubR, gR[(int64_t)row * effLdb + cs], rp, pad);
        AscendC::DataCopyPad(ubI, gI[(int64_t)row * effLdb + cs], rp, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Muls(ubTmp, ubR, aRe, cwA);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Axpy(ubTmp, ubI, -aIm, cwA);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(ubI, ubI, aRe, cwA);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Axpy(ubI, ubR, aIm, cwA);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(ubR, ubTmp, cwA);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyPad(gR[(int64_t)row * effLdb + cs], ubR, rp);
        AscendC::DataCopyPad(gI[(int64_t)row * effLdb + cs], ubI, rp);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 右侧模式回写：单 pass SoA→AoS 转置直接写回 B（替代 TiledTransposePair + 逐行 Interleave）
    __aicore__ inline void InterleaveBRight(__gm__ float* gmBcR, __gm__ float* gmBcI,
                                             __gm__ float* gmB, __gm__ float* gmTemp,
                                             int32_t rowBeg, int32_t rowEnd)
    {
        int32_t kDimOrig = cfg_->kDimOrig;
        int32_t nColsAligned = cfg_->nColsAligned, nColsOrig = cfg_->nColsOrig;
        int32_t mOrig = cfg_->td->m;
        if (rowBeg < 0) rowBeg = 0;
        if (rowEnd > mOrig) rowEnd = mOrig;
        if (rowBeg >= rowEnd) return;
        int32_t ldb2 = cfg_->td->ldb * 2;
        cvt_->TransposeSoADirectToAoS(gmBcR, gmBcI, kDimOrig, nColsOrig, nColsAligned,
            gmB, ldb2, rowBeg, rowEnd);
    }

    CtrsmAivCfg* cfg_;
    CtrsmConvert* cvt_;
};
