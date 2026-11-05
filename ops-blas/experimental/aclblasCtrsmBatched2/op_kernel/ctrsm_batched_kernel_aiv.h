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
 * \file ctrsm_batched_kernel_aiv.h
 * \brief 复数三角求解 AIV 编排类，按计算路径模板化。原单一大类已按职责拆成协作的多个类：
 *          - CtrsmAivCfg        共享派生配置 + UB buffer 指针
 *          - CtrsmConvert       复数 AoS<->SoA 转换/分块转置工具（路径无关）
 *          - CtrsmCanonA        A 规范化（补零/转置/共轭）
 *          - CtrsmCanonB<RIGHT> B 规范化与回写（左/右乘）
 *          - CtrsmPanelSolver<FORWARD> Panel 三角求解（前代/回代）
 *          - CtrsmMixAivImpl<FORWARD,RIGHT> 编排类：持有 pipe/UB buffer/cfg，装配并驱动子类
 *        UB buffer 由编排类统一分配（保留原有别名布局以满足 192KB 预算），子类仅持指针引用。
 *
 * 四条计算路径由模板参数在编译期固定（消除运行时分支），对应文件末尾四个类型别名；
 * 内核入口按 side/uplo/transa 分派到对应路径类型。
 */

#pragma once

#include "ctrsm_batched_kernel_common.h"
#include "ctrsm_batched_kernel_aiv_cfg.h"
#include "ctrsm_batched_kernel_aiv_convert.h"
#include "ctrsm_batched_kernel_aiv_canon_a.h"
#include "ctrsm_batched_kernel_aiv_canon_b.h"
#include "ctrsm_batched_kernel_aiv_solver.h"

template <bool FORWARD, bool RIGHT>
class CtrsmMixAivImpl {
public:
    __aicore__ inline CtrsmMixAivImpl() {}

    // 初始化：解析 tiling、设置全局内存指针、分配 UB 缓冲区、装配子对象
    __aicore__ inline void Init(GM_ADDR aArrayGm, GM_ADDR bArrayGm, GM_ADDR gemmWsGm,
                                const __gm__ CtrsmBatchedTilingData* t, AscendC::TPipe* pipeIn)
    {
        cfg.td = t;
        pipe = pipeIn;
        blockIdx = AscendC::GetBlockIdx();
        cfg.kDim = RIGHT ? t->n : t->m;
        cfg.nCols = RIGHT ? t->m : t->n;
        cfg.nb = t->nb;
        cfg.kDimOrig = cfg.kDim;
        cfg.nColsOrig = cfg.nCols;
        int32_t kDimPad = CEIL_ALIGN(cfg.kDim, FLOAT_ALIGN);
        int32_t nColsPad = CEIL_ALIGN(cfg.nCols, FLOAT_ALIGN);
        cfg.padOn = (cfg.kDim != kDimPad) || (cfg.nCols != nColsPad);
        if (cfg.padOn) { cfg.kDim = kDimPad; cfg.nCols = nColsPad; }
        cfg.nColsAligned = CEIL_ALIGN(cfg.nCols, FLOAT_ALIGN);
        cfg.nbAligned = CEIL_ALIGN(cfg.nb, FLOAT_ALIGN);
        cfg.needTA = NeedTransA(t);
        cfg.conjA = IsConjTransA(t);
        gmAArray.SetGlobalBuffer((__gm__ uint64_t*)aArrayGm, t->batchCount);
        gmBArray.SetGlobalBuffer((__gm__ uint64_t*)bArrayGm, t->batchCount);
        gemmWsBase = gemmWsGm;
        cfg.useOrigA = (t->useOrigA == 1);
        cfg.aEffStride = t->aEffStride;
        cfg.localNCols = cfg.nColsOrig;
        cfg.localNColsAligned = cfg.nColsAligned;
        cfg.nColsOffset = 0;
        cfg.colStart = 0;
        cfg.colEnd = cfg.nColsAligned;
        InitBuffers();
        WireSubObjects();
        convert.BuildDeinterleaveOffsets();   // gather 偏移表构建一次，后续 LoadPanelA 复用
    }

    // 初始化UB缓冲区分配：根据矩阵维度计算各缓冲区大小并分配
    __aicore__ inline void InitBuffers()
    {
        int32_t kDimAligned = CEIL_ALIGN(cfg.kDim, FLOAT_ALIGN);
        int32_t rowMax = kDimAligned;
        if (cfg.nColsAligned > rowMax) rowMax = cfg.nColsAligned;
        if (rowMax < FLOAT_ALIGN) rowMax = FLOAT_ALIGN;
        rowMax = CEIL_ALIGN(rowMax, 32);
        pipe->InitBuffer(bufRow, rowMax * sizeof(float));
        pipe->InitBuffer(bufPanelA_real, cfg.nbAligned * cfg.nbAligned * sizeof(float));
        pipe->InitBuffer(bufPanelA_imag, cfg.nbAligned * cfg.nbAligned * sizeof(float));

        int32_t fixedFloats = 2 * cfg.nbAligned * cfg.nbAligned + cfg.nbAligned + rowMax;
        int32_t perColFloats = 5 * cfg.nb + 1;
        constexpr int32_t UB_FLOATS = 192 * 1024 / (int32_t)sizeof(float);
        int32_t colTileMax = (UB_FLOATS - fixedFloats) / perColFloats;
        colTileMax = (colTileMax / FLOAT_ALIGN) * FLOAT_ALIGN;
        cfg.colTile = (cfg.nColsAligned <= colTileMax) ? cfg.nColsAligned : colTileMax;
        if (cfg.colTile < FLOAT_ALIGN) cfg.colTile = FLOAT_ALIGN;
        int32_t baseBuf = cfg.nb * cfg.colTile;
        int32_t interBuf = 2 * baseBuf;   // 交错 B 存储 [实|虚] 需 nb*2*colTile
        int32_t panelBRealSize = (CtrsmAivCfg::TS * CtrsmAivCfg::TS > interBuf) ? CtrsmAivCfg::TS * CtrsmAivCfg::TS : interBuf;
        int32_t negINSize = (CtrsmAivCfg::TS * 16 > baseBuf) ? CtrsmAivCfg::TS * 16 : baseBuf;
        int32_t totalUsed = fixedFloats + panelBRealSize + 4 * baseBuf + negINSize + cfg.colTile;
        while (totalUsed > UB_FLOATS && cfg.colTile > FLOAT_ALIGN) {
            cfg.colTile -= FLOAT_ALIGN;
            baseBuf = cfg.nb * cfg.colTile;
            interBuf = 2 * baseBuf;
            panelBRealSize = (CtrsmAivCfg::TS * CtrsmAivCfg::TS > interBuf) ? CtrsmAivCfg::TS * CtrsmAivCfg::TS : interBuf;
            negINSize = (CtrsmAivCfg::TS * 16 > baseBuf) ? CtrsmAivCfg::TS * 16 : baseBuf;
            totalUsed = fixedFloats + panelBRealSize + 4 * baseBuf + negINSize + cfg.colTile;
        }
        baseBuf = cfg.nb * cfg.colTile;
        interBuf = 2 * baseBuf;
        panelBRealSize = (CtrsmAivCfg::TS * CtrsmAivCfg::TS > interBuf) ? CtrsmAivCfg::TS * CtrsmAivCfg::TS : interBuf;
        negINSize = (CtrsmAivCfg::TS * 16 > baseBuf) ? CtrsmAivCfg::TS * 16 : baseBuf;
        pipe->InitBuffer(bufPanelB_real, panelBRealSize * sizeof(float));
        cfg.srcBufFloats = panelBRealSize;
        pipe->InitBuffer(bufPanelB_imag, baseBuf * sizeof(float));
        pipe->InitBuffer(bufRankK, cfg.colTile * sizeof(float));
        pipe->InitBuffer(bufNeg_real, baseBuf * sizeof(float));
        pipe->InitBuffer(bufNeg_imag, baseBuf * sizeof(float));
        pipe->InitBuffer(bufNeg_imag_neg, negINSize * sizeof(float));
        int32_t gc = cfg.nb * cfg.colTile;
        cfg.gatherChunk = (gc < 256) ? gc : 256;
        cfg.gatherChunk = (cfg.gatherChunk / FLOAT_ALIGN) * FLOAT_ALIGN;
        if (cfg.gatherChunk < FLOAT_ALIGN) cfg.gatherChunk = FLOAT_ALIGN;
        // 专用 gather 偏移表 buffer（gatherChunk 个 int，固定内容）
        pipe->InitBuffer(bufGatherEven, cfg.gatherChunk * sizeof(int32_t));
        pipe->InitBuffer(bufGatherOdd, cfg.gatherChunk * sizeof(int32_t));
    }

    // 向量核主入口：根据 numSplits/dualAivMode 选择多核拆分/分列/原始独立模式
    __aicore__ inline void Process12()
    {
        if (cfg.td->numSplits > 1) {
            int32_t aivBlock = (int32_t)blockIdx / 2;
            int32_t half = (int32_t)blockIdx % 2;
            int32_t batch = aivBlock / cfg.td->numSplits;
            int32_t splitIdx = aivBlock % cfg.td->numSplits;
            bool isLast = (splitIdx == cfg.td->numSplits - 1);
            cfg.localNCols = isLast ? cfg.td->lastNCols : cfg.td->splitNCols;
            cfg.localNColsAligned = isLast ? cfg.td->lastNColsAligned : cfg.td->splitNColsAligned;
            cfg.nColsOffset = splitIdx * cfg.td->splitNCols;
            int32_t localNColsAligned = cfg.localNColsAligned;
            cfg.nColsAligned = localNColsAligned;
            cfg.nCols = localNColsAligned;
            int32_t colMid = CEIL_ALIGN(localNColsAligned / 2, FLOAT_ALIGN);
            if (colMid > localNColsAligned) colMid = localNColsAligned;
            if (half == 0) { cfg.colStart = 0; cfg.colEnd = colMid; }
            else { cfg.colStart = colMid; cfg.colEnd = localNColsAligned; }
            ProcessOneBatchSplit(batch, half);
        } else if (cfg.td->dualAivMode) {
            int32_t batch = (int32_t)blockIdx / 2;
            int32_t half = (int32_t)blockIdx % 2;
            int32_t colMid = CEIL_ALIGN(cfg.nColsAligned / 2, FLOAT_ALIGN);
            if (colMid > cfg.nColsAligned) colMid = cfg.nColsAligned;
            if (half == 0) { cfg.colStart = 0; cfg.colEnd = colMid; }
            else { cfg.colStart = colMid; cfg.colEnd = cfg.nColsAligned; }
            ProcessOneBatchPadded(batch);
        } else {
            int32_t batch = (int32_t)blockIdx;
            cfg.colStart = 0;
            cfg.colEnd = cfg.nColsAligned;
            if (batch < cfg.td->batchCount) {
                cfg.buildRowStart = 0;
                cfg.buildRowEnd = cfg.kDim;
                ProcessOneBatchOriginal(batch);
            } else {
                DummyPanelSync();
            }
        }
    }

private:
    __aicore__ inline void WireSubObjects()
    {
        cfg.bufPanelA_real = &bufPanelA_real; cfg.bufPanelA_imag = &bufPanelA_imag;
        cfg.bufPanelB_real = &bufPanelB_real; cfg.bufPanelB_imag = &bufPanelB_imag;
        cfg.bufNeg_real = &bufNeg_real; cfg.bufNeg_imag = &bufNeg_imag; cfg.bufNeg_imag_neg = &bufNeg_imag_neg;
        cfg.bufRankK = &bufRankK; cfg.bufRow = &bufRow;
        cfg.bufGatherEven = &bufGatherEven; cfg.bufGatherOdd = &bufGatherOdd;
        convert.Bind(&cfg);
        canonA.Bind(&cfg, &convert);
        canonB.Bind(&cfg, &convert);
        solver.Bind(&cfg, &convert);
    }

    // 判断当前Panel是否有Trail区域需要GEMM更新
    __aicore__ inline bool PanelHasTrailAiv(int32_t idx, int32_t numPanels)
    {
        int32_t p = FORWARD ? idx : (numPanels - 1 - idx);
        int32_t panelStart = p * cfg.nb;
        int32_t actualNb = (cfg.nb < cfg.kDim - panelStart) ? cfg.nb : (cfg.kDim - panelStart);
        int32_t trailRows = FORWARD ? (cfg.kDim - (panelStart + actualNb)) : panelStart;
        return trailRows > 0;
    }

    // 空batch时执行虚拟同步，配合Cube核保持跨核flag对齐
    __aicore__ inline void DummyPanelSync()
    {
        int32_t numPanels = (cfg.kDim + cfg.nb - 1) / cfg.nb;
        for (int32_t idx = 0; idx < numPanels; idx++) {
            if (!PanelHasTrailAiv(idx, numPanels)) continue;
            AscendC::PipeBarrier<PIPE_MTE3>();
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_TRSV);
            AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(FLAG_GEMM);
        }
    }

    // 准备A矩阵工作区（转置/padding），返回有效A指针和stride
    __aicore__ inline void PrepareAMatrix(int32_t batch, __gm__ float* coreWs,
        __gm__ float* gmBc_real, __gm__ float* gmBc_imag, __gm__ float* gmGemm,
        __gm__ float*& effA, int32_t& effAStride)
    {
        __gm__ float* gmA = (__gm__ float*)gmAArray.GetValue(batch);
        __gm__ float* gmAc = coreWs;
        if (cfg.useOrigA) {
            effA = gmA;
            effAStride = cfg.aEffStride;
        } else if (cfg.needTA) {
            int32_t wsStride = 2 * cfg.kDim;
            float signIm = cfg.conjA ? -1.0f : 1.0f;
            convert.TransposeADirectToAoS(gmA, cfg.kDimOrig, cfg.td->lda,
                gmAc, wsStride, signIm, 0, cfg.kDim, cfg.padOn, cfg.kDim);
            effA = gmAc;
            effAStride = wsStride;
        } else {
            canonA.BuildPaddedA(gmA, gmAc, gmBc_real, gmBc_imag, gmGemm);
            effA = gmAc;
            effAStride = 2 * cfg.kDim;
        }
    }

    // 准备B矩阵、执行求解循环、回写结果
    __aicore__ inline void PrepareBAndSolve(int32_t batch, __gm__ float* effA, int32_t effAStride,
        __gm__ float* gmBc_real, __gm__ float* gmBc_imag, __gm__ float* gmGemm)
    {
        __gm__ float* gmB = (__gm__ float*)gmBArray.GetValue(batch);
        if (RIGHT) {
            int32_t rTotal = (cfg.td->m < cfg.nColsOrig) ? cfg.td->m : cfg.nColsOrig;
            int32_t deintCols = (cfg.td->n < cfg.kDimOrig) ? cfg.td->n : cfg.kDimOrig;
            if (cfg.padOn) {
                convert.ZeroGmRows(gmBc_real, cfg.kDim, cfg.nColsAligned, cfg.nColsAligned);
                convert.ZeroGmRows(gmBc_imag, cfg.kDim, cfg.nColsAligned, cfg.nColsAligned);
            }
            convert.TransposeAoSDirectToSoA(gmB, rTotal, deintCols, cfg.td->ldb,
                gmBc_real, gmBc_imag, cfg.nColsAligned, 0, cfg.kDimOrig);
            float aRe = cfg.td->alphaReal;
            float aIm = cfg.td->alphaImag;
            if (aRe != 1.0f || aIm != 0.0f) {
                cfg.buildRowStart = 0;
                cfg.buildRowEnd = cfg.kDim;
                canonB.BuildBRightPhase2Alpha(gmBc_real, gmBc_imag, cfg.nColsAligned, aRe, aIm);
            }
        } else {
            cfg.buildRowStart = 0;
            cfg.buildRowEnd = cfg.kDim;
            canonB.BuildPaddedB(gmB, gmBc_real, gmBc_imag, gmGemm);
        }
        RunPanelLoop(effA, effAStride, gmBc_real, gmBc_imag, gmGemm, gmB);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (RIGHT) {
            canonB.WriteBackPaddedB(gmBc_real, gmBc_imag, gmB, gmGemm, 0, cfg.td->m);
        }
    }

    // 原始单AIV处理路径（无分列，workspace按blockIdx索引）
    __aicore__ inline void ProcessOneBatchOriginal(int32_t batch)
    {
        __gm__ float* coreWs = (__gm__ float*)gemmWsBase
            + (int64_t)blockIdx * (cfg.td->workspaceOffset / sizeof(float));
        int64_t aPlaneSize = (int64_t)cfg.kDim * cfg.kDim;
        __gm__ float* gmAc = coreWs;
        int64_t bPlaneSize = (int64_t)cfg.kDim * cfg.nColsAligned;
        __gm__ float* gmBc_real = coreWs + 2 * aPlaneSize;
        __gm__ float* gmBc_imag = gmBc_real + bPlaneSize;
        __gm__ float* gmGemm = gmBc_imag + bPlaneSize;
        __gm__ float* effA;
        int32_t effAStride;
        PrepareAMatrix(batch, coreWs, gmBc_real, gmBc_imag, gmGemm, effA, effAStride);
        PrepareBAndSolve(batch, effA, effAStride, gmBc_real, gmBc_imag, gmGemm);
    }
    // 处理单个batch（dual分列路径）：构建A/B工作区、执行Panel循环、回写结果
    __aicore__ inline void ProcessOneBatchPadded(int32_t batch)
    {
        __gm__ float* gmA = (__gm__ float*)gmAArray.GetValue(batch);
        __gm__ float* gmB = (__gm__ float*)gmBArray.GetValue(batch);
        __gm__ float* coreWs = (__gm__ float*)gemmWsBase
            + (int64_t)batch * (cfg.td->workspaceOffset / sizeof(float));
        int64_t aPlaneSize = (int64_t)cfg.kDim * cfg.kDim;
        __gm__ float* gmAc = coreWs;
        int64_t bPlaneSize = (int64_t)cfg.kDim * cfg.nColsAligned;
        __gm__ float* gmBc_real = coreWs + 2 * aPlaneSize;
        __gm__ float* gmBc_imag = gmBc_real + bPlaneSize;
        __gm__ float* gmGemm = gmBc_imag + bPlaneSize;
        int32_t half = (int32_t)blockIdx % 2;
        __gm__ float* effA;
        int32_t effAStride;

        PrepareAWithSync(gmA, gmAc, gmBc_real, gmBc_imag, gmGemm, half, effA, effAStride);
        PrepareBWithSync(gmB, gmBc_real, gmBc_imag, gmGemm, half);

        RunPanelLoop(effA, effAStride, gmBc_real, gmBc_imag, gmGemm, gmB);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (RIGHT) {
            DualAivBarrier();
            // dual-AIV 写回按输出行切分：两核各写一半，消除整矩阵重复转置/回写
            int32_t mOrig = cfg.td->m;
            int32_t rowMid = ((mOrig / 2 + CtrsmAivCfg::TS - 1) / CtrsmAivCfg::TS) * CtrsmAivCfg::TS;
            if (rowMid > mOrig) rowMid = mOrig;
            int32_t wbBeg = (half == 0) ? 0 : rowMid;
            int32_t wbEnd = (half == 0) ? rowMid : mOrig;
            canonB.WriteBackPaddedB(gmBc_real, gmBc_imag, gmB, gmGemm, wbBeg, wbEnd);
        }
    }

    // 多核拆分路径：每个 block 的 2 AIV 协同处理一个 split（复用 dual-AIV 同步协议）
    __aicore__ inline void ProcessOneBatchSplit(int32_t batch, int32_t half)
    {
        __gm__ float* gmA = (__gm__ float*)gmAArray.GetValue(batch);
        __gm__ float* gmBOrig = (__gm__ float*)gmBArray.GetValue(batch);
        __gm__ float* gmB;
        if (RIGHT) {
            gmB = gmBOrig + (int64_t)cfg.nColsOffset * cfg.td->ldb * 2;
        } else {
            gmB = gmBOrig;
        }
        int32_t savedNColsOrig = cfg.nColsOrig;
        cfg.nColsOrig = cfg.localNCols;
        int32_t aivBlock = (int32_t)blockIdx / 2;
        int32_t splitIdx = aivBlock % cfg.td->numSplits;
        __gm__ float* batchWs = (__gm__ float*)gemmWsBase
            + (int64_t)batch * (cfg.td->workspaceOffset / sizeof(float));
        int64_t aPlaneSize = (int64_t)cfg.kDim * cfg.kDim;
        __gm__ float* gmAc = batchWs;
        int32_t wsNCols = (cfg.td->splitNColsAligned > cfg.td->lastNColsAligned)
                          ? cfg.td->splitNColsAligned : cfg.td->lastNColsAligned;
        int64_t bPlaneSize = (int64_t)cfg.kDim * wsNCols;
        int64_t splitBGemmFloats = cfg.td->splitBGemmSize / sizeof(float);
        __gm__ float* splitWs = batchWs + 2 * aPlaneSize + (int64_t)splitIdx * splitBGemmFloats;
        __gm__ float* gmBc_real = splitWs;
        __gm__ float* gmBc_imag = gmBc_real + bPlaneSize;
        __gm__ float* gmGemm = gmBc_imag + bPlaneSize;
        __gm__ float* effA;
        int32_t effAStride;

        PrepareAWithSync(gmA, gmAc, gmBc_real, gmBc_imag, gmGemm, half, effA, effAStride);
        PrepareBSplit(gmB, gmBc_real, gmBc_imag, gmGemm, half);

        RunPanelLoop(effA, effAStride, gmBc_real, gmBc_imag, gmGemm, gmB);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (RIGHT) {
            DualAivBarrier();
            int32_t localM = cfg.localNCols;
            int32_t rowMid = ((localM / 2 + CtrsmAivCfg::TS - 1) / CtrsmAivCfg::TS) * CtrsmAivCfg::TS;
            if (rowMid > localM) rowMid = localM;
            int32_t wbBeg = (half == 0) ? 0 : rowMid;
            int32_t wbEnd = (half == 0) ? rowMid : localM;
            canonB.WriteBackPaddedB(gmBc_real, gmBc_imag, gmB, gmGemm, wbBeg, wbEnd);
        }
        cfg.nColsOrig = savedNColsOrig;
    }

    // 共用：Left 模式下按 half 分行构建 B 工作区
    __aicore__ inline void PrepareBLeftHalf(__gm__ float* gmB,
        __gm__ float* gmBc_real, __gm__ float* gmBc_imag, __gm__ float* gmGemm,
        int32_t half)
    {
        int32_t totalRows = cfg.kDim;
        int32_t rowMid = CEIL_ALIGN(totalRows / 2, FLOAT_ALIGN);
        if (rowMid > totalRows) rowMid = totalRows;
        cfg.buildRowStart = (half == 0) ? 0 : rowMid;
        cfg.buildRowEnd = (half == 0) ? rowMid : totalRows;
        canonB.BuildPaddedB(gmB, gmBc_real, gmBc_imag, gmGemm);
        DualAivBarrier();
    }

    // 共用：Right 模式下按 half 分列解交织 B 并可选 alpha 缩放
    __aicore__ inline void PrepareBRightHalf(__gm__ float* gmB,
        __gm__ float* gmBc_real, __gm__ float* gmBc_imag,
        int32_t half, int32_t rTotal)
    {
        int32_t nColsAligned = cfg.nColsAligned;
        int32_t kDimOrig = cfg.kDimOrig;
        int32_t kDim = cfg.kDim;
        int32_t cbMid = CEIL_ALIGN(kDim / 2, FLOAT_ALIGN);
        if (cbMid > kDim) cbMid = kDim;
        int32_t cbBeg = (half == 0) ? 0 : cbMid;
        int32_t cbEnd = (half == 0) ? cbMid : kDim;
        cfg.buildRowStart = cbBeg;
        cfg.buildRowEnd = (cbEnd < kDimOrig) ? cbEnd : kDimOrig;
        int32_t nOrig = cfg.td->n;
        int32_t deintCols = (nOrig < kDimOrig) ? nOrig : kDimOrig;
        convert.TransposeAoSDirectToSoA(gmB, rTotal, deintCols, cfg.td->ldb,
            gmBc_real, gmBc_imag, nColsAligned, cbBeg, cbEnd);
        float aRe = cfg.td->alphaReal;
        float aIm = cfg.td->alphaImag;
        if (aRe != 1.0f || aIm != 0.0f) {
            canonB.BuildBRightPhase2Alpha(gmBc_real, gmBc_imag, nColsAligned, aRe, aIm);
        }
        DualAivBarrier();
    }

    // B 工作区准备（split 模式）：按 half 分行清零/解交织，复用 dual-AIV 协议
    __aicore__ inline void PrepareBSplit(__gm__ float* gmB,
        __gm__ float* gmBc_real, __gm__ float* gmBc_imag, __gm__ float* gmGemm,
        int32_t half)
    {
        if (!RIGHT) {
            PrepareBLeftHalf(gmB, gmBc_real, gmBc_imag, gmGemm, half);
            return;
        }
        int32_t nColsOrig = cfg.nColsOrig;
        int32_t kDimOrig = cfg.kDimOrig;
        int32_t rTotal = (nColsOrig < kDimOrig) ? nColsOrig : kDimOrig;
        PrepareBRightHalf(gmB, gmBc_real, gmBc_imag, half, rTotal);
    }

    // A工作区准备：needTA 时单 pass 转置（两核并行各做一半输出列），否则单核直接复制
    __aicore__ inline void PrepareAWithSync(__gm__ float* gmA, __gm__ float* gmAc,
        __gm__ float* gmBc_real, __gm__ float* gmBc_imag, __gm__ float* gmGemm,
        int32_t half, __gm__ float*& effA, int32_t& effAStride)
    {
        if (!cfg.useOrigA) {
            if (cfg.needTA) {
                int32_t kDimOrig = cfg.kDimOrig;
                int32_t kDim = cfg.kDim;
                int32_t wsStride = 2 * kDim;
                float signIm = cfg.conjA ? -1.0f : 1.0f;
                int32_t colMid = CEIL_ALIGN(kDim / 2, FLOAT_ALIGN);
                if (colMid > kDim) colMid = kDim;
                int32_t cBeg = (half == 0) ? 0 : colMid;
                int32_t cEnd = (half == 0) ? colMid : kDim;
                convert.TransposeADirectToAoS(gmA, kDimOrig, cfg.td->lda,
                    gmAc, wsStride, signIm, cBeg, cEnd, cfg.padOn, kDim);
                DualAivBarrier();
            } else {
                if (half == 0) {
                    canonA.BuildPaddedA(gmA, gmAc, gmBc_real, gmBc_imag, gmGemm);
                }
                DualAivBarrier();
            }
            effA = gmAc;
            effAStride = 2 * cfg.kDim;
        } else {
            effA = gmA;
            effAStride = cfg.aEffStride;
        }
    }

    // 双 AIV 屏障：AI Core 内两 AIV 各完成本核 workspace 构建后经 mode1 核间同步互等
    // （替代原 GM flag 忙等轮询，消除轮询循环与 PIPE_ALL；PIPE_MTE3 保证 GM 写对对方可见）
    __aicore__ inline void DualAivBarrier()
    {
        AscendC::CrossCoreSetFlag<1, PIPE_MTE3>(FLAG_DUAL_AIV);
        AscendC::CrossCoreWaitFlag<1, PIPE_MTE3>(FLAG_DUAL_AIV);
    }

    // B工作区准备：设置分行范围、构建补零B，并按 left/right 与 half 完成跨AIV同步
    __aicore__ inline void PrepareBWithSync(__gm__ float* gmB,
        __gm__ float* gmBc_real, __gm__ float* gmBc_imag, __gm__ float* gmGemm,
        int32_t half)
    {
        if (!RIGHT) {
            PrepareBLeftHalf(gmB, gmBc_real, gmBc_imag, gmGemm, half);
            return;
        }
        // Right dual：单 pass 构建（两核各做一半输出行，含 pad）
        int32_t mOrig = cfg.td->m;
        int32_t nColsOrig = cfg.nColsOrig;
        int32_t rTotal = (mOrig < nColsOrig) ? mOrig : nColsOrig;
        PrepareBRightHalf(gmB, gmBc_real, gmBc_imag, half, rTotal);
    }

    // 逐Panel执行三角求解，并与AIC核同步完成Trail区域GEMM更新
    // 优化：在等待AIC GEMM期间预加载下一Panel的A对角块（与GEMM并行）
    __aicore__ inline void RunPanelLoop(__gm__ float* effA, int32_t effAStride,
                                         __gm__ float* gmBcR, __gm__ float* gmBcI,
                                         __gm__ float* gmGemm, __gm__ float* gmB)
    {
        int32_t nb = cfg.nb, kDim = cfg.kDim, nColsAligned = cfg.nColsAligned;
        int32_t numPanels = (kDim + nb - 1) / nb;
        AscendC::LocalTensor<float> ubAR = bufPanelA_real.Get<float>();
        AscendC::LocalTensor<float> ubAI = bufPanelA_imag.Get<float>();
        for (int32_t idx = 0; idx < numPanels; idx++) {
            int32_t p = FORWARD ? idx : (numPanels - 1 - idx);
            int32_t panelStart = p * nb;
            int32_t actualNb = (nb < kDim - panelStart) ? nb : (kDim - panelStart);
            int32_t slot = XnegSlot(idx, numPanels, FORWARD);
            int64_t xnegPlaneSize = (int64_t)2 * LIM_GROUP * 2 * nb * nColsAligned;
            __gm__ float* xR = gmGemm + (int64_t)slot * 2 * nb * nColsAligned;
            __gm__ float* xI = gmGemm + xnegPlaneSize + (int64_t)slot * 2 * nb * nColsAligned;
            if (idx == 0) {
                solver.LoadPanelA(effA, effAStride, ubAR, ubAI, panelStart, actualNb);
            }
            solver.PanelTrsv(effA, effAStride, gmBcR, gmBcI, nColsAligned,
                             xR, xI, panelStart, actualNb, PanelHasTrailAiv(idx, numPanels));
            if (PanelHasTrailAiv(idx, numPanels)) {
                AscendC::PipeBarrier<PIPE_MTE3>();
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_TRSV);
                if (!RIGHT) {
                    canonB.WriteBackPanelRows(gmBcR, gmBcI, gmB, panelStart, actualNb);
                }
                int32_t nextIdx = idx + 1;
                if (nextIdx < numPanels) {
                    int32_t np = FORWARD ? nextIdx : (numPanels - 1 - nextIdx);
                    int32_t nextStart = np * nb;
                    int32_t nextNb = (nb < kDim - nextStart) ? nb : (kDim - nextStart);
                    solver.LoadPanelA(effA, effAStride, ubAR, ubAI, nextStart, nextNb);
                }
                AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(FLAG_GEMM);
            } else if (!RIGHT) {
                canonB.WriteBackPanelRows(gmBcR, gmBcI, gmB, panelStart, actualNb);
            }
        }
    }

    // ---- owned resources ----
    AscendC::TPipe* pipe;
    CtrsmAivCfg cfg;
    CtrsmConvert convert;
    CtrsmCanonA canonA;
    CtrsmCanonB<RIGHT> canonB;
    CtrsmPanelSolver<FORWARD> solver;
    AscendC::GlobalTensor<uint64_t> gmAArray, gmBArray;
    __gm__ uint8_t* gemmWsBase;
    uint32_t blockIdx;
    BufVecCalc bufPanelA_real, bufPanelA_imag, bufPanelB_real, bufPanelB_imag;
    BufVecCalc bufNeg_real, bufNeg_imag, bufNeg_imag_neg, bufRankK, bufRow;
    BufVecCalc bufGatherEven, bufGatherOdd;
};

// 四条计算路径的具体类型别名
using CtrsmLowerLeft  = CtrsmMixAivImpl<true,  false>; // 下三角前代 + 左乘
using CtrsmLowerRight = CtrsmMixAivImpl<true,  true>;  // 下三角前代 + 右乘
using CtrsmUpperLeft  = CtrsmMixAivImpl<false, false>; // 上三角回代 + 左乘
using CtrsmUpperRight = CtrsmMixAivImpl<false, true>;  // 上三角回代 + 右乘
