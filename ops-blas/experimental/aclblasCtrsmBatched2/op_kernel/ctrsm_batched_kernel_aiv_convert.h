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
 * \file ctrsm_batched_kernel_aiv_convert.h
 * \brief 复数格式转换工具（与计算路径无关）：AoS<->SoA 解交织/交织、Gather 偏移表、分块转置。
 *        供规范化器、求解器、回写共用。仅引用 cfg 中的 UB buffer 指针，不拥有 buffer。
 */

#pragma once

#include "ctrsm_batched_kernel_aiv_cfg.h"

class CtrsmConvert {
public:
    __aicore__ inline CtrsmConvert() {}

    // 计算转置 tile 列数（根据可用 buffer 大小动态确定）
    __aicore__ inline int32_t CalcTileCols()
    {
        int32_t tileCols = (cfg_->srcBufFloats / 48 / FLOAT_ALIGN) * FLOAT_ALIGN;
        int32_t tcImag = (cfg_->nb * cfg_->colTile / 16 / FLOAT_ALIGN) * FLOAT_ALIGN;
        if (tcImag < tileCols) tileCols = tcImag;
        if (tileCols > 512) tileCols = 512;
        if (tileCols < FLOAT_ALIGN) tileCols = FLOAT_ALIGN;
        return tileCols;
    }

    __aicore__ inline void Bind(CtrsmAivCfg* cfg) { cfg_ = cfg; }

    // 将GM工作空间逐行清零（stride为一行的float数，zeroCols为每行实际清零列数）。
    // A/B 规范化共用：用 bufRow 复制一段零，按行 DMA 写回。
    // 当 zeroCols 超过 bufRow 容量时分段清零。
    __aicore__ inline void ZeroGmRows(__gm__ float* ws, int32_t rows, int32_t stride, int32_t zeroCols)
    {
        AscendC::LocalTensor<float> ub = cfg_->bufRow->Get<float>();
        AscendC::GlobalTensor<float> gW;
        gW.SetGlobalBuffer(ws, (uint32_t)((int64_t)rows * stride));
        int32_t kDimAligned = CEIL_ALIGN(cfg_->kDim, FLOAT_ALIGN);
        int32_t rowMax = kDimAligned;
        if (cfg_->nColsAligned > rowMax) rowMax = cfg_->nColsAligned;
        if (rowMax < FLOAT_ALIGN) rowMax = FLOAT_ALIGN;
        int32_t chunkSize = (zeroCols <= rowMax) ? zeroCols : rowMax;
        AscendC::Duplicate(ub, 0.0f, chunkSize);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < rows; i++) {
            int64_t rowBase = (int64_t)i * stride;
            for (int32_t cs = 0; cs < zeroCols; cs += chunkSize) {
                int32_t cw = (chunkSize < zeroCols - cs) ? chunkSize : (zeroCols - cs);
                AscendC::DataCopyExtParams wRow(1, (uint32_t)(cw * sizeof(float)), 0, 0, 0);
                AscendC::DataCopyPad(gW[rowBase + cs], ub, wRow);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 构建Gather所需的偶数/奇数字节偏移表（写入专用 buffer，Init 时构建一次，永久复用）
    __aicore__ inline void BuildDeinterleaveOffsets()
    {
        AscendC::LocalTensor<int32_t> offEven = cfg_->bufGatherEven->Get<int32_t>();
        AscendC::LocalTensor<int32_t> offOdd = cfg_->bufGatherOdd->Get<int32_t>();
        for (int32_t i = 0; i < FLOAT_ALIGN; i++) {
            offEven.SetValue(i, i * 2 * (int32_t)sizeof(float));
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t c = FLOAT_ALIGN; c < cfg_->gatherChunk; c += FLOAT_ALIGN) {
            AscendC::Adds(offEven[c], offEven, c * 2 * (int32_t)sizeof(float), FLOAT_ALIGN);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Adds(offOdd, offEven, (int32_t)sizeof(float), cfg_->gatherChunk);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 将一行AoS数据拆分为实部和虚部（vreducev2 硬件指令版本）
    __aicore__ inline void DeinterleaveRow(AscendC::LocalTensor<float>& src,
                                            AscendC::LocalTensor<float>& dstReal,
                                            int32_t count, float signIm)
    {
        AscendC::LocalTensor<float> dstImag = cfg_->bufPanelB_imag->Get<float>();
        int32_t countA = CEIL_ALIGN(count, FLOAT_ALIGN);
        int32_t numRepeats = (count * 2 + 63) / 64;
        vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(dstReal.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t*>(src.GetPhyAddr()),
                  nullptr, numRepeats, 1, 1, 8, 8);
        vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(dstImag.GetPhyAddr()),
                  reinterpret_cast<__ubuf__ uint32_t*>(src.GetPhyAddr()),
                  nullptr, numRepeats, 1, 2, 8, 8);
        AscendC::PipeBarrier<PIPE_V>();
        if (signIm != 1.0f) {
            AscendC::Muls(dstImag, dstImag, signIm, countA);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 构建InterleaveRow所需的交织字节偏移表（实部虚部交替排列）
    __aicore__ inline void BuildInterleaveOffsets(int32_t count)
    {
        AscendC::LocalTensor<int32_t> offInter = cfg_->bufNeg_imag->Get<int32_t>();
        int32_t countA = CEIL_ALIGN(count, FLOAT_ALIGN);
        for (int32_t i = 0; i < 4; i++) {
            offInter.SetValue(2 * i, i * (int32_t)sizeof(float));
            offInter.SetValue(2 * i + 1, countA * (int32_t)sizeof(float) + i * (int32_t)sizeof(float));
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t g = FLOAT_ALIGN; g < cfg_->gatherChunk; g += FLOAT_ALIGN) {
            AscendC::Adds(offInter[g], offInter, (g / 2) * (int32_t)sizeof(float), FLOAT_ALIGN);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    // 将实部虚部交织合并为一行AoS数据（Gather向量化版本）
    __aicore__ inline void InterleaveRow(AscendC::LocalTensor<float>& real,
                                          AscendC::LocalTensor<float>& imag,
                                          AscendC::LocalTensor<float>& out,
                                          AscendC::LocalTensor<int32_t>& offInter,
                                          int32_t count)
    {
        int32_t countA = CEIL_ALIGN(count, FLOAT_ALIGN);
        AscendC::DataCopy(real[countA], imag, countA);
        AscendC::PipeBarrier<PIPE_V>();
        int32_t halfChunk = cfg_->gatherChunk / 2;
        for (int32_t c = 0; c < countA; c += halfChunk) {
            int32_t remain = countA - c;
            int32_t len = (halfChunk < remain) ? halfChunk : remain;
            int32_t outLen = 2 * len;
            AscendC::Gather(out[c * 2], real[c], offInter.ReinterpretCast<uint32_t>(), 0, outLen);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // 分块转置：dst[c][r] = src[r][c]，使用TransDataTo5HD硬件加速
    // colBeg/colLimit 限定产出的目标行范围（=src 列范围），默认全量；用于 dual-AIV 行切分
    __aicore__ inline void TiledTranspose(__gm__ float* src, int32_t rows, int32_t cols,
                                          int32_t srcLd, __gm__ float* dst, int32_t dstLd,
                                          int32_t colBeg = 0, int32_t colLimit = -1)
    {
        if (colLimit < 0 || colLimit > cols) colLimit = cols;
        if (colBeg < 0) colBeg = 0;
        AscendC::LocalTensor<float> ubSrc = cfg_->bufPanelB_real->Get<float>();
        AscendC::LocalTensor<float> ubDst = cfg_->bufNeg_imag_neg->Get<float>();
        AscendC::GlobalTensor<float> gmS, gmD;
        gmS.SetGlobalBuffer(src, (uint32_t)((int64_t)rows * srcLd));
        gmD.SetGlobalBuffer(dst, (uint32_t)((int64_t)cols * dstLd));
        AscendC::DataCopyPadExtParams<float> padP{false, 0, 0, 0};
        for (int32_t rb = 0; rb < rows; rb += CtrsmAivCfg::TS) {
            int32_t tr = (CtrsmAivCfg::TS < rows - rb) ? CtrsmAivCfg::TS : (rows - rb);
            for (int32_t cb = colBeg; cb < colLimit; cb += CtrsmAivCfg::TS) {
                int32_t tc = (CtrsmAivCfg::TS < colLimit - cb) ? CtrsmAivCfg::TS : (colLimit - cb);
                int32_t tcA = CEIL_ALIGN(tc, FLOAT_ALIGN);
                AscendC::DataCopyExtParams ldP((uint16_t)tr, (uint32_t)(tc * sizeof(float)),
                    (int64_t)((srcLd - tc) * sizeof(float)), (int64_t)((tcA - tc) / FLOAT_ALIGN), 0);
                AscendC::DataCopyPad(ubSrc, gmS[(int64_t)rb * srcLd + cb], ldP, padP);
                AscendC::PipeBarrier<PIPE_ALL>();
                TransposeTileBlock(ubSrc, ubDst, gmD, tcA, tr, tc, dstLd, cb, rb);
            }
        }
    }

    // 对实部和虚部两个平面同时执行TiledTranspose（常见成对调用模式）
    __aicore__ inline void TiledTransposePair(__gm__ float* srcR, __gm__ float* srcI,
                                              int32_t rows, int32_t cols, int32_t srcLd,
                                              __gm__ float* dstR, __gm__ float* dstI, int32_t dstLd,
                                              int32_t colBeg = 0, int32_t colLimit = -1)
    {
        TiledTranspose(srcR, rows, cols, srcLd, dstR, dstLd, colBeg, colLimit);
        TiledTranspose(srcI, rows, cols, srcLd, dstI, dstLd, colBeg, colLimit);
    }

    // 对 AoS workspace 做 padding：行尾清零 + 补零行（清零+对角写1）
    // 在转置完成后（barrier 之后）调用，各核处理自己负责的行范围
    __aicore__ inline void PadAoSWorkspace(__gm__ float* gmAc,
        int32_t kDimOrig, int32_t kDim, int32_t wsStride,
        int32_t rowBeg, int32_t rowEnd)
    {
        AscendC::LocalTensor<float> ub = cfg_->bufRow->Get<float>();
        AscendC::GlobalTensor<float> gAc;
        gAc.SetGlobalBuffer(gmAc, (uint32_t)((int64_t)kDim * wsStride));
        // 有效行的尾部 padding 清零
        int32_t tailFloats = wsStride - 2 * kDimOrig;
        if (tailFloats > 0) {
            int32_t tailAligned = CEIL_ALIGN(tailFloats, FLOAT_ALIGN);
            AscendC::Duplicate(ub, 0.0f, tailAligned);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t i = rowBeg; i < rowEnd; i++) {
                AscendC::DataCopyPad(gAc[(int64_t)i * wsStride + 2 * kDimOrig], ub,
                    {1, (uint32_t)(tailFloats * sizeof(float)), 0, 0, 0});
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        // 补零行：清零整行 + 对角写 1.0（由 rowEnd==kDimOrig 的核负责）
        if (rowEnd == kDimOrig) {
            int32_t rowMax = CEIL_ALIGN(kDim, FLOAT_ALIGN);
            if (cfg_->nColsAligned > rowMax) rowMax = cfg_->nColsAligned;
            for (int32_t i = kDimOrig; i < kDim; i++) {
                AscendC::Duplicate(ub, 0.0f, rowMax);
                AscendC::PipeBarrier<PIPE_V>();
                for (int32_t cs = 0; cs < wsStride; cs += rowMax) {
                    int32_t cw = (rowMax < wsStride - cs) ? rowMax : (wsStride - cs);
                    AscendC::DataCopyPad(gAc[(int64_t)i * wsStride + cs], ub,
                        {1, (uint32_t)(cw * sizeof(float)), 0, 0, 0});
                }
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            PadDiagonalIdentity(gAc, ub, kDimOrig, kDim, wsStride);
        }
    }

    // TransposeADirectToAoS 的单 tile 处理：加载/解交织/行交替/转置写出
    __aicore__ inline void TransposeADirectToAoSTile(
        AscendC::GlobalTensor<float>& gA, AscendC::GlobalTensor<float>& gOut,
        AscendC::LocalTensor<float>& bufLoad, AscendC::LocalTensor<float>& bufImag,
        AscendC::LocalTensor<float>& bufDst,
        int32_t rb, int32_t tr, int32_t trValid, int32_t cb, int32_t tc,
        int32_t kDimOrig, int32_t lda, int32_t wsStride, float signIm,
        bool padOn, int32_t kDimPad)
    {
        int32_t tcA = CEIL_ALIGN(tc, FLOAT_ALIGN);
        int32_t aosRowFloats = tc * 2;
        int32_t aosRowAligned = CEIL_ALIGN(aosRowFloats, FLOAT_ALIGN);

        // Step 1: MTE2 加载 16 行 AoS 到 bufLoad
        if (padOn) {
            int32_t zeroTotal = CEIL_ALIGN(tr * aosRowAligned, FLOAT_ALIGN);
            AscendC::Duplicate(bufLoad, 0.0f, zeroTotal);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (trValid > 0 && cb < kDimOrig) {
            int32_t tcLoad = (tc < kDimOrig - cb) ? tc : (kDimOrig - cb);
            int32_t aosLoad = tcLoad * 2;
            int32_t aosLoadAligned = CEIL_ALIGN(aosLoad, FLOAT_ALIGN);
            int32_t dstGap = (aosRowAligned - aosLoadAligned) / FLOAT_ALIGN;
            AscendC::DataCopyExtParams ldP((uint16_t)trValid,
                (uint32_t)(aosLoad * sizeof(float)),
                (uint32_t)((lda * 2 - aosLoad) * sizeof(float)),
                (uint32_t)dstGap, 0);
            AscendC::DataCopyPad(bufLoad, gA[(int64_t)rb * lda * 2 + cb * 2], ldP,
                {false, 0, 0, 0});
        } else if (!padOn) {
            int32_t zeroTotal = CEIL_ALIGN(tr * aosRowAligned, FLOAT_ALIGN);
            AscendC::Duplicate(bufLoad, 0.0f, zeroTotal);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // Step 2: VEC 解交织
        int32_t realBase = tr * aosRowAligned;
        DeinterleaveRowsVreducev2(bufLoad, bufImag, tr, tc, tcA, aosRowAligned, realBase);
        if (signIm != 1.0f) {
            int32_t total = CEIL_ALIGN(tr * tcA, FLOAT_ALIGN);
            AscendC::Muls(bufImag, bufImag, signIm, total);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Step 3: 行交替排列
        for (int32_t r = tr - 1; r >= 0; r--) {
            AscendC::DataCopy(bufLoad[(2 * r + 1) * tcA], bufImag[r * tcA], tcA);
            AscendC::DataCopy(bufLoad[2 * r * tcA], bufLoad[realBase + r * tcA], tcA);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // Step 4: TransposeTile16 → bufDst → MTE3 写 GM
        int32_t totalRows = 2 * tr;
        TransposeTile16AndWriteGm(bufLoad, bufDst, gOut, tcA, totalRows, tc,
            wsStride, (int64_t)cb, (int64_t)rb * 2);
    }

    // 单 pass A 转置：从 GM 加载 AoS → UB 解交织+行交替 → 转置 → 直接写 AoS 到 gmAcOut
    // 3 块 buffer 轮转：bufLoad(加载AoS) → bufWork(行交替) → bufDst(转置输出)
    // colBeg/colLimit 控制输出列范围（用于 dual-AIV 各做一半）
    __aicore__ inline void TransposeADirectToAoS(
        __gm__ float* gmA, int32_t kDimOrig, int32_t lda,
        __gm__ float* gmAcOut, int32_t wsStride, float signIm,
        int32_t colBeg = 0, int32_t colLimit = -1,
        bool padOn = false, int32_t kDimPad = -1)
    {
        if (colLimit < 0 || colLimit > kDimPad) colLimit = kDimPad;
        if (colBeg < 0) colBeg = 0;
        if (kDimPad < 0) kDimPad = kDimOrig;
        constexpr int32_t CHUNK = 16;
        int32_t tileCols = CalcTileCols();
        AscendC::LocalTensor<float> bufLoad = cfg_->bufPanelB_real->Get<float>();
        AscendC::LocalTensor<float> bufImag = cfg_->bufPanelB_imag->Get<float>();
        AscendC::LocalTensor<float> bufDst = cfg_->bufNeg_imag_neg->Get<float>();
        AscendC::GlobalTensor<float> gA, gOut;
        gA.SetGlobalBuffer(gmA, (uint32_t)((int64_t)kDimOrig * lda * 2));
        gOut.SetGlobalBuffer(gmAcOut, (uint32_t)((int64_t)kDimPad * wsStride));

        for (int32_t rb = 0; rb < kDimPad; rb += CHUNK) {
            int32_t tr = (CHUNK < kDimPad - rb) ? CHUNK : (kDimPad - rb);
            int32_t trValid = (rb + tr <= kDimOrig) ? tr : ((rb < kDimOrig) ? (kDimOrig - rb) : 0);
            int32_t tileCount = (colLimit - colBeg + tileCols - 1) / tileCols;

            for (int32_t tileIdx = 0; tileIdx < tileCount; tileIdx++) {
                int32_t cb = colBeg + tileIdx * tileCols;
                int32_t tc = (tileCols < colLimit - cb) ? tileCols : (colLimit - cb);
                TransposeADirectToAoSTile(gA, gOut, bufLoad, bufImag, bufDst,
                    rb, tr, trValid, cb, tc, kDimOrig, lda, wsStride, signIm, padOn, kDimPad);
            }
        }
    }

    // TransposeAoSDirectToSoA 的单 tile 处理
    __aicore__ inline void TransposeAoSDirectToSoATile(
        AscendC::GlobalTensor<float>& gB,
        AscendC::GlobalTensor<float>& gDstR, AscendC::GlobalTensor<float>& gDstI,
        AscendC::LocalTensor<float>& bufLoad, AscendC::LocalTensor<float>& bufImag,
        AscendC::LocalTensor<float>& bufDst,
        int32_t rb, int32_t tr, int32_t cb, int32_t tc, int32_t cols, int32_t ldb,
        int32_t dstStride)
    {
        int32_t tcA = CEIL_ALIGN(tc, FLOAT_ALIGN);
        int32_t aosRowFloats = tc * 2;
        int32_t aosRowAligned = CEIL_ALIGN(aosRowFloats, FLOAT_ALIGN);
        int32_t zeroTotal = CEIL_ALIGN(tr * aosRowAligned, FLOAT_ALIGN);
        AscendC::Duplicate(bufLoad, 0.0f, zeroTotal);
        AscendC::PipeBarrier<PIPE_V>();
        if (cb < cols) {
            int32_t tcLoad = (tc < cols - cb) ? tc : (cols - cb);
            int32_t aosLoad = tcLoad * 2;
            int32_t aosLoadAligned = CEIL_ALIGN(aosLoad, FLOAT_ALIGN);
            int32_t dstGap = (aosRowAligned - aosLoadAligned) / FLOAT_ALIGN;
            AscendC::DataCopyExtParams ldP((uint16_t)tr,
                (uint32_t)(aosLoad * sizeof(float)),
                (uint32_t)((ldb * 2 - aosLoad) * sizeof(float)), (uint32_t)dstGap, 0);
            AscendC::DataCopyPad(bufLoad, gB[(int64_t)rb * ldb * 2 + cb * 2], ldP,
                {false, 0, 0, 0});
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        int32_t realBase = tr * aosRowAligned;
        DeinterleaveRowsVreducev2(bufLoad, bufImag, tr, tc, tcA, aosRowAligned, realBase);
        for (int32_t r = 0; r < tr; r++) {
            AscendC::DataCopy(bufLoad[r * tcA], bufLoad[realBase + r * tcA], tcA);
        }
        AscendC::PipeBarrier<PIPE_V>();
        TransposeTile16AndWriteGmPair(bufLoad, bufImag, bufDst, gDstR, gDstI,
            tcA, tr, tc, dstStride, (int64_t)cb, (int64_t)rb);
    }

    // 单 pass B 构建：从 GM 加载 AoS → 解交织 → 分别转置实部/虚部写出到 SoA GM
    __aicore__ inline void TransposeAoSDirectToSoA(
        __gm__ float* gmB, int32_t rows, int32_t cols, int32_t ldb,
        __gm__ float* gmDstR, __gm__ float* gmDstI, int32_t dstStride,
        int32_t colBeg = 0, int32_t colLimit = -1)
    {
        if (colLimit < 0 || colLimit > cols) colLimit = cols;
        if (colBeg < 0) colBeg = 0;
        constexpr int32_t CHUNK = 16;
        int32_t tileCols = CalcTileCols();
        AscendC::LocalTensor<float> bufLoad = cfg_->bufPanelB_real->Get<float>();
        AscendC::LocalTensor<float> bufImag = cfg_->bufPanelB_imag->Get<float>();
        AscendC::LocalTensor<float> bufDst = cfg_->bufNeg_imag_neg->Get<float>();
        AscendC::GlobalTensor<float> gB, gDstR, gDstI;
        gB.SetGlobalBuffer(gmB, (uint32_t)((int64_t)rows * ldb * 2));
        gDstR.SetGlobalBuffer(gmDstR, (uint32_t)((int64_t)colLimit * dstStride));
        gDstI.SetGlobalBuffer(gmDstI, (uint32_t)((int64_t)colLimit * dstStride));
        for (int32_t rb = 0; rb < rows; rb += CHUNK) {
            int32_t tr = (CHUNK < rows - rb) ? CHUNK : (rows - rb);
            int32_t tileCount = (colLimit - colBeg + tileCols - 1) / tileCols;
            for (int32_t tileIdx = 0; tileIdx < tileCount; tileIdx++) {
                int32_t cb = colBeg + tileIdx * tileCols;
                int32_t tc = (tileCols < colLimit - cb) ? tileCols : (colLimit - cb);
                TransposeAoSDirectToSoATile(gB, gDstR, gDstI, bufLoad, bufImag, bufDst,
                    rb, tr, cb, tc, cols, ldb, dstStride);
            }
        }
    }

    // TransposeSoADirectToAoS 的单 tile 处理
    __aicore__ inline void TransposeSoADirectToAoSTile(
        AscendC::GlobalTensor<float>& gSrcR, AscendC::GlobalTensor<float>& gSrcI,
        AscendC::GlobalTensor<float>& gOut,
        AscendC::LocalTensor<float>& bufLoad, AscendC::LocalTensor<float>& bufImag,
        AscendC::LocalTensor<float>& bufDst,
        int32_t rb, int32_t tr, int32_t cb, int32_t tc,
        int32_t srcStride, int32_t dstStride)
    {
        int32_t tcA = CEIL_ALIGN(tc, FLOAT_ALIGN);
        AscendC::DataCopyExtParams ldP((uint16_t)tr,
            (uint32_t)(tc * sizeof(float)),
            (uint32_t)((srcStride - tc) * sizeof(float)), 0, 0);
        AscendC::DataCopyPadExtParams<float> padPL{true, 0,
            static_cast<uint8_t>(tcA - tc), 0.0f};
        AscendC::DataCopyPad(bufLoad, gSrcR[(int64_t)rb * srcStride + cb], ldP, padPL);
        AscendC::DataCopyPad(bufImag, gSrcI[(int64_t)rb * srcStride + cb], ldP, padPL);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t r = tr - 1; r >= 0; r--) {
            AscendC::DataCopy(bufLoad[(2 * r + 1) * tcA], bufImag[r * tcA], tcA);
            AscendC::DataCopy(bufLoad[2 * r * tcA], bufLoad[r * tcA], tcA);
        }
        AscendC::PipeBarrier<PIPE_V>();
        int32_t totalRows = 2 * tr;
        TransposeTile16AndWriteGm(bufLoad, bufDst, gOut, tcA, totalRows, tc,
            dstStride, (int64_t)cb, (int64_t)rb * 2);
    }

    // 单 pass SoA->AoS 转置回写：从 SoA GM 加载 -> 行交替 -> 转置 -> 写 AoS GM
    __aicore__ inline void TransposeSoADirectToAoS(
        __gm__ float* gmSrcR, __gm__ float* gmSrcI,
        int32_t rows, int32_t cols, int32_t srcStride,
        __gm__ float* gmDstAoS, int32_t dstStride,
        int32_t colBeg = 0, int32_t colLimit = -1)
    {
        if (colLimit < 0 || colLimit > cols) colLimit = cols;
        if (colBeg < 0) colBeg = 0;
        constexpr int32_t CHUNK = 16;
        int32_t tileCols = CalcTileCols();
        AscendC::LocalTensor<float> bufLoad = cfg_->bufPanelB_real->Get<float>();
        AscendC::LocalTensor<float> bufImag = cfg_->bufPanelB_imag->Get<float>();
        AscendC::LocalTensor<float> bufDst = cfg_->bufNeg_imag_neg->Get<float>();
        AscendC::GlobalTensor<float> gSrcR, gSrcI, gOut;
        gSrcR.SetGlobalBuffer(gmSrcR, (uint32_t)((int64_t)rows * srcStride));
        gSrcI.SetGlobalBuffer(gmSrcI, (uint32_t)((int64_t)rows * srcStride));
        gOut.SetGlobalBuffer(gmDstAoS, (uint32_t)((int64_t)cols * dstStride));
        for (int32_t rb = 0; rb < rows; rb += CHUNK) {
            int32_t tr = (CHUNK < rows - rb) ? CHUNK : (rows - rb);
            int32_t tileCount = (colLimit - colBeg + tileCols - 1) / tileCols;
            for (int32_t tileIdx = 0; tileIdx < tileCount; tileIdx++) {
                int32_t cb = colBeg + tileIdx * tileCols;
                int32_t tc = (tileCols < colLimit - cb) ? tileCols : (colLimit - cb);
                TransposeSoADirectToAoSTile(gSrcR, gSrcI, gOut, bufLoad, bufImag, bufDst,
                    rb, tr, cb, tc, srcStride, dstStride);
            }
        }
    }


    // 共用：对多行 AoS 数据执行 vreducev2 解交织，将实部写入 bufReal[realBase+r*tcA]，虚部写入 bufImag[r*tcA]
    __aicore__ inline void DeinterleaveRowsVreducev2(
        AscendC::LocalTensor<float>& bufSrc,
        AscendC::LocalTensor<float>& bufImag,
        int32_t tr, int32_t tc, int32_t tcA, int32_t aosRowAligned, int32_t realBase)
    {
        for (int32_t r = 0; r < tr; r++) {
            int32_t numRepeats = (tc * 2 + 63) / 64;
            vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(
                bufSrc.GetPhyAddr() + (realBase + r * tcA) * sizeof(float)),
                reinterpret_cast<__ubuf__ uint32_t*>(
                bufSrc.GetPhyAddr() + r * aosRowAligned * sizeof(float)),
                nullptr, numRepeats, 1, 1, 8, 8);
            vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(
                bufImag.GetPhyAddr() + r * tcA * sizeof(float)),
                reinterpret_cast<__ubuf__ uint32_t*>(
                bufSrc.GetPhyAddr() + r * aosRowAligned * sizeof(float)),
                nullptr, numRepeats, 1, 2, 8, 8);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // 共用：TransposeTile16 + DMA 写出到 GM（用于 AoS 转置输出）
    __aicore__ inline void TransposeTile16AndWriteGm(
        AscendC::LocalTensor<float>& bufSrc,
        AscendC::LocalTensor<float>& bufDst,
        AscendC::GlobalTensor<float>& gOut,
        int32_t tcA, int32_t totalRows, int32_t tc,
        int32_t outStride, int64_t outColBase, int64_t outRowBase)
    {
        for (int32_t ro = 0; ro < totalRows; ro += 16) {
            int32_t chunkRows = (16 < totalRows - ro) ? 16 : (totalRows - ro);
            TransposeTile16(bufSrc, bufDst, tcA, totalRows, ro, chunkRows);
            AscendC::PipeBarrier<PIPE_ALL>();
            int32_t srcStride32B = 2 - (int32_t)((chunkRows + 7) / 8);
            AscendC::DataCopyExtParams stP((uint16_t)tc,
                (uint32_t)(chunkRows * sizeof(float)),
                (int64_t)srcStride32B,
                (int64_t)((outStride - chunkRows) * sizeof(float)), 0);
            AscendC::DataCopyPad(gOut[outColBase * outStride + outRowBase + ro],
                bufDst, stP);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // 共用：TransposeTile16 + DMA 写出（SoA 双平面：实部+虚部各转置写出）
    __aicore__ inline void TransposeTile16AndWriteGmPair(
        AscendC::LocalTensor<float>& bufReal,
        AscendC::LocalTensor<float>& bufImag,
        AscendC::LocalTensor<float>& bufDst,
        AscendC::GlobalTensor<float>& gDstR, AscendC::GlobalTensor<float>& gDstI,
        int32_t tcA, int32_t totalRows, int32_t tc,
        int32_t dstStride, int64_t colBase, int64_t rowBase)
    {
        for (int32_t ro = 0; ro < totalRows; ro += 16) {
            int32_t chunkRows = (16 < totalRows - ro) ? 16 : (totalRows - ro);
            int32_t srcStride32B = 2 - (int32_t)((chunkRows + 7) / 8);
            AscendC::DataCopyExtParams stP((uint16_t)tc,
                (uint32_t)(chunkRows * sizeof(float)),
                (int64_t)srcStride32B,
                (int64_t)((dstStride - chunkRows) * sizeof(float)), 0);
            TransposeTile16(bufReal, bufDst, tcA, totalRows, ro, chunkRows);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(gDstR[colBase * dstStride + rowBase + ro], bufDst, stP);
            AscendC::PipeBarrier<PIPE_ALL>();
            TransposeTile16(bufImag, bufDst, tcA, totalRows, ro, chunkRows);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(gDstI[colBase * dstStride + rowBase + ro], bufDst, stP);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // 共用：对补零行写入单位对角线（real=1.0）
    __aicore__ inline void PadDiagonalIdentity(
        AscendC::GlobalTensor<float>& gAc, AscendC::LocalTensor<float>& ub,
        int32_t kDimOrig, int32_t kDim, int32_t wsStride)
    {
        for (int32_t i = kDimOrig; i < kDim; i++) {
            int32_t diagPos = 2 * i;
            int32_t alignedStart = (diagPos / FLOAT_ALIGN) * FLOAT_ALIGN;
            AscendC::DataCopyPad(ub, gAc[(int64_t)i * wsStride + alignedStart],
                {1, (uint32_t)(FLOAT_ALIGN * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            ub.SetValue(diagPos - alignedStart, 1.0f);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(gAc[(int64_t)i * wsStride + alignedStart], ub,
                {1, (uint32_t)(FLOAT_ALIGN * sizeof(float)), 0, 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    CtrsmAivCfg* cfg_;
};
