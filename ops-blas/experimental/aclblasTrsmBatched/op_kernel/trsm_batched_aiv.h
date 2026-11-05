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
 * \file trsm_batched_aiv.h
 * \brief AIV 批量三角求解的编排类。原单一大类已按职责拆成协作的多个类：
 *          - TrsmAivCfg         共享派生配置
 *          - TrsmTranspose      分块转置引擎
 *          - TrsmCanonicalizer  A/B 规范化与回写
 *          - TrsmPanelSolver    panel 三角求解与 Xneg 写回
 *          - TrsmMixAiv         编排类：持有 pipe/UB buffer/cfg，装配并驱动上述子类
 *        UB buffer 由本编排类统一分配（保留原有别名布局以满足 192KB 预算），
 *        子类只持指针引用，不各自分配。
 */

#pragma once

#include "trsm_batched_kernel_common.h"
#include "trsm_batched_aiv_cfg.h"
#include "trsm_batched_aiv_transpose.h"
#include "trsm_batched_aiv_solver.h"
#include "trsm_batched_aiv_canon.h"

// ============ AIV: canonicalize (A^T / B^T) + panel TRSV + alpha + rank-K ============
class TrsmMixAiv {
public:
    __aicore__ inline TrsmMixAiv() {}

    // Initialize AIV with tiling params, array pointers, buffer setup, and batch range
    __aicore__ inline void Init(GM_ADDR aArrayGm, GM_ADDR bArrayGm, GM_ADDR gemmWsGm,
                                const __gm__ TrsmBatchedTilingData* t, AscendC::TPipe* pipeIn,
                                bool is12 = false)
    {
        cfg.tiling = t;
        pipe = pipeIn;
        blockIdx = AscendC::GetBlockIdx();
        cfg.is12Mode = is12;
        cfg.right = (t->side == SIDE_RIGHT);
        cfg.kDim = cfg.right ? t->n : t->m;
        cfg.nCols = cfg.right ? t->m : t->n;
        cfg.nb = t->nb;
        cfg.kDimOrig = cfg.kDim;
        cfg.nColsOrig = cfg.nCols;
        int32_t kDimPad = CEIL_ALIGN(cfg.kDim, FLOAT_ALIGN);
        int32_t nColsPad = CEIL_ALIGN(cfg.nCols, FLOAT_ALIGN);
        cfg.padOn = (cfg.kDim != kDimPad) || (cfg.nCols != nColsPad);
        if (cfg.padOn) { cfg.kDim = kDimPad; cfg.nCols = nColsPad; }
        cfg.nColsAligned = CEIL_ALIGN(cfg.nCols, FLOAT_ALIGN);
        cfg.nbAligned = CEIL_ALIGN(cfg.nb, FLOAT_ALIGN);
        cfg.solveColStart = 0;
        cfg.solveColEnd = cfg.nCols;
        cfg.coopAiv = false;
        cfg.splitId = 0;
        cfg.splitNcols = cfg.nColsAligned;

        cfg.needTA = NeedTransA(t);
        int32_t effUplo = cfg.needTA ? (t->uplo == UPLO_UPPER ? UPLO_LOWER : UPLO_UPPER) : t->uplo;
        cfg.forward = (effUplo == UPLO_LOWER);

        gmAArray.SetGlobalBuffer((__gm__ uint64_t*)aArrayGm, t->batchCount);
        gmBArray.SetGlobalBuffer((__gm__ uint64_t*)bArrayGm, t->batchCount);
        gemmWsBase = gemmWsGm;

        int32_t perCore = t->batchCount / t->totalCores;
        int32_t remainder = t->batchCount % t->totalCores;
        batchStart = blockIdx * perCore + ((int32_t)blockIdx < remainder ? blockIdx : remainder);
        batchEnd = batchStart + perCore + ((int32_t)blockIdx < remainder ? 1 : 0);

        InitBuffers();
        WireSubObjects();
    }

    // Allocate UB buffers for panel solve, transpose, and padding
    __aicore__ inline void InitBuffers()
    {
        int32_t kDimAligned = CEIL_ALIGN(cfg.kDim, FLOAT_ALIGN);
        pipe->InitBuffer(bufPanelA, cfg.nbAligned * cfg.nbAligned * sizeof(float));
        pipe->InitBuffer(bufAt, cfg.nb * cfg.nbAligned * sizeof(float));
        pipe->InitBuffer(bufColIdx, cfg.nbAligned * sizeof(int32_t));

        bool useOuter = (cfg.nColsAligned <= 64);
        int32_t fixedFloats = cfg.nbAligned * cfg.nbAligned + cfg.nb * cfg.nbAligned + cfg.nbAligned;
        int32_t perColFloats = 2 * cfg.nb + 1;
        if (useOuter) perColFloats += 3 * cfg.nb;
        if (cfg.padOn) {
            int32_t padBuf = kDimAligned > cfg.nColsAligned ? kDimAligned : cfg.nColsAligned;
            fixedFloats += padBuf;
        }
        constexpr int32_t UB_FLOATS = 192 * 1024 / (int32_t)sizeof(float);
        int32_t colTileMax = (UB_FLOATS - fixedFloats) / perColFloats;
        colTileMax = (colTileMax / FLOAT_ALIGN) * FLOAT_ALIGN;
        cfg.colTile = (cfg.nColsAligned <= colTileMax) ? cfg.nColsAligned : colTileMax;
        if (cfg.colTile < FLOAT_ALIGN) cfg.colTile = FLOAT_ALIGN;

        if (useOuter) {
            pipe->InitBuffer(bufBcA, cfg.nb * cfg.colTile * sizeof(float));
            pipe->InitBuffer(bufBcB, cfg.nb * cfg.colTile * sizeof(float));
            pipe->InitBuffer(bufBcTmp, cfg.nb * cfg.colTile * sizeof(float));
        }
        InitTransposeBuffers(useOuter, kDimAligned);
        if (cfg.padOn) {
            int32_t rowMax = kDimAligned > cfg.colTile ? kDimAligned : cfg.colTile;
            if (cfg.nColsAligned > rowMax) rowMax = cfg.nColsAligned;
            pipe->InitBuffer(bufRow, rowMax * sizeof(float));
        }
    }

    // Size and allocate transpose tile buffers (panelB, rankK, neg)
    __aicore__ inline void InitTransposeBuffers(bool useOuter, int32_t kDimAligned)
    {
        int32_t baseBuf = cfg.nb * cfg.colTile;
        int32_t fixedOther = cfg.nbAligned * cfg.nbAligned + cfg.nb * cfg.nbAligned + cfg.nbAligned + cfg.colTile;
        if (useOuter) fixedOther += 3 * cfg.nb * cfg.colTile;
        if (cfg.padOn) {
            int32_t rowMax = kDimAligned > cfg.colTile ? kDimAligned : cfg.colTile;
            if (cfg.nColsAligned > rowMax) rowMax = cfg.nColsAligned;
            fixedOther += rowMax;
        }
        constexpr int32_t UB_FLOATS_TOTAL = 192 * 1024 / (int32_t)sizeof(float);
        int32_t availForTranspose = UB_FLOATS_TOTAL - fixedOther;
        int32_t tsCap = FLOAT_ALIGN;
        while (tsCap + FLOAT_ALIGN <= 128) {
            int32_t nextTs = tsCap + FLOAT_ALIGN;
            if (nextTs * nextTs + nextTs * 16 > availForTranspose) break;
            tsCap = nextTs;
        }
        cfg.ts = (tsCap < FLOAT_ALIGN) ? FLOAT_ALIGN : tsCap;
        int32_t panelBSize = (cfg.ts * cfg.ts > baseBuf) ? cfg.ts * cfg.ts : baseBuf;
        int32_t negSize = (cfg.ts * 16 > baseBuf) ? cfg.ts * 16 : baseBuf;
        pipe->InitBuffer(bufPanelB, panelBSize * sizeof(float));
        pipe->InitBuffer(bufRankK, cfg.colTile * sizeof(float));
        pipe->InitBuffer(bufNeg, negSize * sizeof(float));
    }

    // Process batch range in 1_1 mode (loop over batchStart..batchEnd)
    __aicore__ inline void Process()
    {
        for (int32_t batch = batchStart; batch < batchEnd; batch++) {
            ProcessOneBatch(batch);
        }
    }

    // MIX_1_2: this AIV's global id (block_idx*2 + subBlockId == blockIdx here) is its batch and slot.
    // 1_1 vs 1_2 is only batch->core packing; padding is orthogonal, so 1_2 dispatches to the
    // padded path for non-8/needTA just like 1_1 (small extra latency on those non-perf cases).
    __aicore__ inline void Process12()
    {
        int32_t splitFactor = cfg.tiling->splitFactor;
        if (cfg.tiling->coopMode == 1 && splitFactor > 1) {
            int32_t aivIdx = (int32_t)blockIdx;
            int32_t piece = aivIdx / 2;
            int32_t batch = piece / splitFactor;
            int32_t splitId = piece % splitFactor;
            int32_t subBlock = aivIdx % 2;
            SetSplitColumnRange(splitId, splitFactor, subBlock);
            if (cfg.padOn) ProcessOneBatchPadded(batch);
            else ProcessOneBatchCore(batch);
        } else if (cfg.tiling->coopMode == 1) {
            int32_t batch = (int32_t)blockIdx / 2;
            SetCoopColumnRange((int32_t)blockIdx % 2);
            if (cfg.padOn) ProcessOneBatchPadded(batch);
            else ProcessOneBatchCore(batch);
        } else {
            Process12NonCoop();
        }
    }

    __aicore__ inline void Process12NonCoop()
    {
        int32_t numPanels = (cfg.kDim + cfg.nb - 1) / cfg.nb;
        if (numPanels <= 1) {
            Process12SinglePanel();
        } else {
            Process12MultiPanel();
        }
    }

    __aicore__ inline void Process12SinglePanel()
    {
        int32_t totalBlocks = (int32_t)AscendC::GetBlockNum();
        int32_t batchCount = cfg.tiling->batchCount;
        int32_t stride = totalBlocks * 2;
        for (int32_t batch = (int32_t)blockIdx; batch < batchCount; batch += stride) {
            if (cfg.padOn) ProcessOneBatchPadded(batch);
            else ProcessOneBatchCore(batch);
        }
    }

    __aicore__ inline void Process12MultiPanel()
    {
        bool oddLast = (cfg.tiling->batchCount % 2 == 1);
        int32_t lastBatch = cfg.tiling->batchCount - 1;
        if (oddLast && (int32_t)blockIdx >= lastBatch) {
            int32_t subBlock = ((int32_t)blockIdx == lastBatch) ? 0 : 1;
            SetCoopColumnRange(subBlock);
            if (cfg.padOn) ProcessOneBatchPadded(lastBatch);
            else ProcessOneBatchCore(lastBatch);
        } else if ((int32_t)blockIdx < cfg.tiling->batchCount) {
            if (cfg.padOn) ProcessOneBatchPadded((int32_t)blockIdx);
            else ProcessOneBatchCore((int32_t)blockIdx);
        }
    }

private:
    // Wire sub-objects to shared cfg and the UB buffers owned by this orchestrator.
    __aicore__ inline void WireSubObjects()
    {
        transpose.Bind(&cfg, &bufPanelB, &bufNeg);
        canon.Bind(&cfg, &transpose, &bufRow, &bufPanelB, &bufNeg);
        solver.Bind(&cfg, &bufPanelA, &bufAt, &bufColIdx,
                    &bufPanelB, &bufNeg, &bufRankK, &bufBcA, &bufBcB, &bufBcTmp);
    }

    // Set cooperative column range for column-split mode
    __aicore__ inline void SetCoopColumnRange(int32_t subBlock)
    {
        cfg.coopAiv = true;
        int32_t colSplit = CEIL_ALIGN(cfg.nCols / 2, FLOAT_ALIGN);
        if (colSplit <= 0) colSplit = FLOAT_ALIGN;
        if (colSplit >= cfg.nCols) colSplit = cfg.nCols;
        cfg.solveColStart = (subBlock == 0) ? 0 : colSplit;
        cfg.solveColEnd = (subBlock == 0) ? colSplit : cfg.nCols;
    }

    // Set column range for multi-split mode (splitFactor > 1, 2 AIVs per piece)
    __aicore__ inline void SetSplitColumnRange(int32_t splitId, int32_t splitFactor, int32_t subBlock)
    {
        cfg.coopAiv = true;
        cfg.splitId = splitId;
        if (splitFactor < 1) splitFactor = 1;
        int32_t pieceWidth = CEIL_ALIGN(cfg.nCols / splitFactor, FLOAT_ALIGN);
        if (pieceWidth < FLOAT_ALIGN) pieceWidth = FLOAT_ALIGN;
        int32_t pieceStart = splitId * pieceWidth;
        int32_t pieceEnd = pieceStart + pieceWidth;
        if (pieceEnd > cfg.nCols) pieceEnd = cfg.nCols;
        if (pieceStart >= cfg.nCols) { pieceStart = cfg.nCols; pieceEnd = cfg.nCols; }
        cfg.splitNcols = pieceEnd - pieceStart;
        int32_t halfWidth = CEIL_ALIGN((pieceEnd - pieceStart) / 2, FLOAT_ALIGN);
        if (halfWidth < FLOAT_ALIGN) halfWidth = FLOAT_ALIGN;
        cfg.solveColStart = (subBlock == 0) ? pieceStart : pieceStart + halfWidth;
        cfg.solveColEnd = (subBlock == 0) ? pieceStart + halfWidth : pieceEnd;
        if (cfg.solveColStart > cfg.nCols) cfg.solveColStart = cfg.nCols;
        if (cfg.solveColEnd > cfg.nCols) cfg.solveColEnd = cfg.nCols;
    }

    __aicore__ inline void ProcessOneBatch(int32_t batch)
    {
        if (cfg.padOn) ProcessOneBatchPadded(batch);
        else ProcessOneBatchCore(batch);
    }

    // Compute workspace pointers for a given batch
    __aicore__ inline void SetupWorkspacePointers(int32_t batch,
        __gm__ float*& gmAc, __gm__ float*& gmBc, __gm__ float*& gmGemm)
    {
        __gm__ float* coreWs = (__gm__ float*)gemmWsBase
            + (int64_t)batch * (cfg.tiling->workspaceOffset / sizeof(float));
        int32_t splitFactor = cfg.tiling->splitFactor;
        gmAc = coreWs;
        gmBc = coreWs + (int64_t)splitFactor * cfg.kDim * cfg.kDim;
        __gm__ float* gmGemmBase = gmBc + (int64_t)cfg.kDim * cfg.nColsAligned;
        int32_t maxMN = cfg.kDim > cfg.nColsAligned ? cfg.kDim : cfg.nColsAligned;
        int32_t maxMNAligned = CEIL_ALIGN(maxMN, FLOAT_ALIGN);
        gmGemm = gmGemmBase + (int64_t)cfg.splitId * 2 * maxMN * cfg.tiling->splitNcolsMax;
    }

    // Transpose A into gmAc for the non-padded (core) path
    __aicore__ inline void TransposeACore(__gm__ float* gmA, __gm__ float* gmAc)
    {
        if (cfg.coopAiv) {
            int32_t kDimOrig = cfg.kDim;
            int32_t nb = cfg.nb;
            int32_t halfRows = CEIL_ALIGN(kDimOrig / 2, nb);
            if (halfRows > kDimOrig) halfRows = kDimOrig;
            bool isSubBlock0 = (cfg.solveColStart == 0);
            int32_t myRowStart = isSubBlock0 ? 0 : halfRows;
            int32_t myRowEnd = isSubBlock0 ? halfRows : kDimOrig;
            int32_t rowCount = myRowEnd - myRowStart;
            if (rowCount > 0) {
                transpose.Run(gmA + (int64_t)myRowStart * cfg.tiling->lda,
                              rowCount, cfg.kDim, cfg.tiling->lda,
                              gmAc + myRowStart, cfg.kDim);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::CrossCoreSetFlag<1, PIPE_MTE3>(10);
            AscendC::CrossCoreWaitFlag(10);
        } else {
            transpose.Run(gmA, cfg.kDim, cfg.kDim, cfg.tiling->lda, gmAc, cfg.kDim);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // Non-padded path: canonicalize A/B, panel TRSV loop, sync with AIC
    __aicore__ inline void ProcessOneBatchCore(int32_t batch)
    {
        __gm__ float* gmA = (__gm__ float*)gmAArray.GetValue(batch);
        __gm__ float* gmB = (__gm__ float*)gmBArray.GetValue(batch);
        __gm__ float* gmAc;
        __gm__ float* gmBc;
        __gm__ float* gmGemm;
        SetupWorkspacePointers(batch, gmAc, gmBc, gmGemm);

        if (cfg.needTA) {
            TransposeACore(gmA, gmAc);
        }

        bool aivTransB = cfg.right && (cfg.is12Mode || cfg.kDim > 512 || cfg.nCols > 512);
        __gm__ float* effB;
        int32_t effLdb;
        canon.PrepareEffB(gmB, gmBc, aivTransB, effB, effLdb);

        if (cfg.tiling->alphaReal != 1.0f) {
            solver.AlphaScale(effB, effLdb);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        PanelLoopCore(gmA, gmB, gmAc, gmBc, gmGemm, effB, effLdb, aivTransB);

        if (cfg.right && !aivTransB) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::PipeBarrier<PIPE_MTE3>();
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_TRSV);
            AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(FLAG_GEMM);
        }
    }

    // Panel loop body extracted from ProcessOneBatchCore
    __aicore__ inline void PanelLoopCore(__gm__ float* gmA, __gm__ float* gmB,
                                          __gm__ float* gmAc, __gm__ float* gmBc,
                                          __gm__ float* gmGemm,
                                          __gm__ float* effB, int32_t effLdb,
                                          bool aivTransB)
    {
        int32_t numPanels = (cfg.kDim + cfg.nb - 1) / cfg.nb;
        bool aPreloaded = false;
        for (int32_t idx = 0; idx < numPanels; idx++) {
            int32_t p = cfg.forward ? idx : (numPanels - 1 - idx);
            int32_t panelStart = p * cfg.nb;
            int32_t actualNb = (cfg.nb < cfg.kDim - panelStart) ? cfg.nb : (cfg.kDim - panelStart);
            int32_t trailRows = cfg.forward ? cfg.kDim - (panelStart + actualNb) : panelStart;

            int32_t slot = XnegSlot(idx, numPanels, cfg.forward);
            __gm__ float* gmXneg = gmGemm + (int64_t)slot * cfg.nb * cfg.splitNcols;
            solver.PanelTrsv(gmA, cfg.tiling->lda, effB, effLdb, gmXneg,
                             panelStart, actualNb, cfg.needTA, aPreloaded);
            AscendC::PipeBarrier<PIPE_ALL>();

            if (trailRows > 0) {
                AscendC::PipeBarrier<PIPE_MTE3>();
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_TRSV);
                if (aivTransB) canon.PostTransposePanel(gmBc, gmB, panelStart, actualNb);
                int32_t nextIdx = idx + 1;
                if (nextIdx < numPanels) {
                    int32_t nextP = cfg.forward ? nextIdx : (numPanels - 1 - nextIdx);
                    int32_t nextPanelStart = nextP * cfg.nb;
                    int32_t nextActualNb = (cfg.nb < cfg.kDim - nextPanelStart)
                                           ? cfg.nb : (cfg.kDim - nextPanelStart);
                    solver.LoadPanelA(gmA, cfg.tiling->lda, nextPanelStart, nextActualNb);
                    aPreloaded = true;
                }
                AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(FLAG_GEMM);
            } else {
                if (aivTransB) {
                    canon.PostTransposePanel(gmBc, gmB, panelStart, actualNb);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                aPreloaded = false;
            }
        }
    }

    // Post-panel sync: signal TRSV done, optionally transpose, wait for GEMM
    __aicore__ inline void SyncAfterPanel(bool aivTransB, int32_t trailRows,
                                           __gm__ float* gmBc, __gm__ float* gmB,
                                           int32_t panelStart, int32_t actualNb)
    {
        if (trailRows > 0) {
            AscendC::PipeBarrier<PIPE_MTE3>();
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_TRSV);
            if (aivTransB) canon.PostTransposePanel(gmBc, gmB, panelStart, actualNb);
            AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(FLAG_GEMM);
        } else if (aivTransB) {
            canon.PostTransposePanel(gmBc, gmB, panelStart, actualNb);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // Generalized (padded) path: block-level canonical prepare, scales to any size.
    __aicore__ inline void ProcessOneBatchPadded(int32_t batch)
    {
        __gm__ float* gmA = (__gm__ float*)gmAArray.GetValue(batch);
        __gm__ float* gmB = (__gm__ float*)gmBArray.GetValue(batch);
        __gm__ float* gmAc;
        __gm__ float* gmBc;
        __gm__ float* gmGemm;
        SetupWorkspacePointers(batch, gmAc, gmBc, gmGemm);

        canon.BuildPaddedA(gmA, gmAc);
        if (cfg.coopAiv) {
            AscendC::CrossCoreSetFlag<1, PIPE_MTE3>(10);
            AscendC::CrossCoreWaitFlag(10);
        }
        canon.BuildPaddedB(gmB, gmBc);
        AscendC::PipeBarrier<PIPE_ALL>();

        int32_t numPanels = (cfg.kDim + cfg.nb - 1) / cfg.nb;
        bool aPreloaded = false;
        for (int32_t idx = 0; idx < numPanels; idx++) {
            int32_t p = cfg.forward ? idx : (numPanels - 1 - idx);
            int32_t panelStart = p * cfg.nb;
            int32_t actualNb = (cfg.nb < cfg.kDim - panelStart) ? cfg.nb : (cfg.kDim - panelStart);
            int32_t trailRows = cfg.forward ? (cfg.kDim - (panelStart + actualNb)) : panelStart;
            int32_t slot = XnegSlot(idx, numPanels, cfg.forward);
            __gm__ float* gmXneg = gmGemm + (int64_t)slot * cfg.nb * cfg.splitNcols;
            solver.PanelTrsv(gmAc, cfg.kDim, gmBc, cfg.nColsAligned, gmXneg,
                             panelStart, actualNb, false, aPreloaded);
            AscendC::PipeBarrier<PIPE_ALL>();
            if (trailRows > 0) {
                AscendC::PipeBarrier<PIPE_MTE3>();
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_TRSV);
                int32_t nextIdx = idx + 1;
                if (nextIdx < numPanels) {
                    int32_t nextP = cfg.forward ? nextIdx : (numPanels - 1 - nextIdx);
                    int32_t nextPanelStart = nextP * cfg.nb;
                    int32_t nextActualNb = (cfg.nb < cfg.kDim - nextPanelStart)
                                           ? cfg.nb : (cfg.kDim - nextPanelStart);
                    solver.LoadPanelA(gmAc, cfg.kDim, nextPanelStart, nextActualNb);
                    aPreloaded = true;
                }
                AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(FLAG_GEMM);
                canon.WriteBackPaddedBRows(gmBc, gmB, panelStart, actualNb);
            } else {
                canon.WriteBackPaddedBRows(gmBc, gmB, panelStart, actualNb);
                aPreloaded = false;
            }
        }
        if (cfg.right) {
            AscendC::PipeBarrier<PIPE_ALL>();
            canon.WriteBackPaddedB(gmBc, gmB);
        }
    }

    // ---- owned resources ----
    AscendC::TPipe* pipe;
    TrsmAivCfg cfg;
    TrsmTranspose transpose;
    TrsmCanonicalizer canon;
    TrsmPanelSolver solver;
    AscendC::GlobalTensor<uint64_t> gmAArray, gmBArray;
    __gm__ uint8_t* gemmWsBase;
    uint32_t blockIdx;
    int32_t batchStart, batchEnd;
    BufVecCalc bufPanelA, bufPanelB, bufRankK, bufNeg, bufBcA, bufBcB, bufBcTmp, bufAt, bufColIdx, bufRow;
};
