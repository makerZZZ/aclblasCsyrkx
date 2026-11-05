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
 * \file trsm_batched_aic.h
 * \brief AIC class for batched triangular solve: GEMM rank-K updates via Matmul library,
 *        with grouped panel scheduling and 1:2 cooperative dispatch.
 */

#pragma once

#include "trsm_batched_kernel_common.h"

// ============ AIC: GEMM via Matmul library (always transa=N on effA) ============
typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> MmA;
typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> MmB;
typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> MmC;

class TrsmMixAic {
public:
    Matmul<MmA, MmB, MmC> mm;
    TCubeTiling cubeTiling;

    __aicore__ inline TrsmMixAic() {}

    // Initialize AIC with tiling params, array pointers, workspace, and batch range.
    __aicore__ inline void Init(GM_ADDR aArrayGm, GM_ADDR bArrayGm, GM_ADDR gemmWsGm,
                                const __gm__ TrsmBatchedTilingData* t, const TCubeTiling& ct,
                                GM_ADDR idGm)
    {
        tiling = t;
        cubeTiling = ct;
        idBase = idGm;
        blockIdx = AscendC::GetBlockIdx();
        right = (tiling->side == SIDE_RIGHT);
        kDim = right ? tiling->n : tiling->m;
        nCols = right ? tiling->m : tiling->n;
        nb = tiling->nb;
        // Generalization: padded space when kDim/nCols not multiples of FLOAT_ALIGN. AIV builds
        // canonical padded A/B in workspace; AIC only does rank-K on padded (Cube-safe) dims.
        int32_t kDimPad = CEIL_ALIGN(kDim, FLOAT_ALIGN);
        int32_t nColsPad = CEIL_ALIGN(nCols, FLOAT_ALIGN);
        padOn = (kDim != kDimPad) || (nCols != nColsPad);
        if (padOn) { kDim = kDimPad; nCols = nColsPad; }
        nColsAligned = CEIL_ALIGN(nCols, FLOAT_ALIGN);
        needTA = NeedTransA(tiling);
        {
            int32_t effUplo = needTA ? (tiling->uplo == UPLO_UPPER ? UPLO_LOWER : UPLO_UPPER)
                                     : tiling->uplo;
            forward = (effUplo == UPLO_LOWER);
        }

        gmAArray.SetGlobalBuffer((__gm__ uint64_t*)aArrayGm, tiling->batchCount);
        gmBArray.SetGlobalBuffer((__gm__ uint64_t*)bArrayGm, tiling->batchCount);
        gemmWsBase = gemmWsGm;

        int32_t perCore = tiling->batchCount / tiling->totalCores;
        int32_t remainder = tiling->batchCount % tiling->totalCores;
        batchStart = blockIdx * perCore + ((int32_t)blockIdx < remainder ? blockIdx : remainder);
        batchEnd = batchStart + perCore + ((int32_t)blockIdx < remainder ? 1 : 0);

        splitColStart = 0;
        splitNcols = nCols;
        splitId = 0;
    }

    // Loop over assigned batch range, processing one batch at a time (1_1 mode).
    __aicore__ inline void Process()
    {
        for (int32_t batch = batchStart; batch < batchEnd; batch++) {
            ProcessOneBatch(batch);
        }
    }

    // MIX_1_2: group g (=blockIdx) serves batches 2g and 2g+1. Mode-2 is an AND-barrier
    // (one Wait/Set syncs BOTH AIVs together), so each sync point does both batches' cube work.
    __aicore__ inline void Process12()
    {
        int32_t g = (int32_t)blockIdx;
        int32_t b0, b1;
        bool hasB1;
        int32_t splitFactor = tiling->splitFactor;
        if (tiling->coopMode == 1 && splitFactor > 1) {
            int32_t batch = g / splitFactor;
            int32_t sid = g % splitFactor;
            SetSplitRange(sid, splitFactor);
            b0 = batch;
            b1 = -1;
            hasB1 = false;
        } else if (tiling->coopMode == 1) {
            b0 = g;
            b1 = -1;
            hasB1 = false;
        } else {
            b0 = 2 * g;
            b1 = 2 * g + 1;
            hasB1 = (b1 < tiling->batchCount);
        }
        int32_t numPanels = (kDim + nb - 1) / nb;

        if (numPanels <= 2) {
            Process12Simple(b0, b1, numPanels, hasB1);
        } else {
            Process12Grouped(b0, b1, numPanels, hasB1);
        }
    }

    // Simple panel loop for ≤2 panels: rank-K for b0 and optionally b1.
    __aicore__ inline void Process12Simple(int32_t b0, int32_t b1, int32_t numPanels, bool hasB1)
    {
        for (int32_t idx = 0; idx < numPanels; idx++) {
            if (!PanelHasTrail(idx, numPanels)) continue;
            AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(FLAG_TRSV);
            RankKBody(b0, b0, idx, numPanels);
            if (hasB1) RankKBody(b1, b1, idx, numPanels);
            AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_GEMM);
        }
    }

    // Grouped panel scheduling with direct/pre-update/big-group GEMM for b0 and optionally b1.
    __aicore__ inline void Process12Grouped(int32_t b0, int32_t b1, int32_t numPanels, bool hasB1)
    {
        int32_t numGroups = (numPanels + LIM_GROUP - 1) / LIM_GROUP;
        for (int32_t grp = 0; grp < numGroups; grp++) {
            int32_t idxStart, idxEnd, groupSize, groupEndRow, groupBoundaryRow;
            GroupBounds(grp, numPanels, idxStart, idxEnd, groupSize, groupEndRow, groupBoundaryRow);

            for (int32_t idx = idxStart; idx <= idxEnd; idx++) {
                if (!PanelHasTrail(idx, numPanels)) continue;
                AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(FLAG_TRSV);
                bool hasMore0 = DirectRankK(b0, b0, idx, numPanels);
                bool hasMore1 = hasB1 ? DirectRankK(b1, b1, idx, numPanels) : false;
                AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_GEMM);
                if (hasMore0) PreUpdateRankKWithinGroup(b0, b0, idx, numPanels,
                                                        groupEndRow, groupBoundaryRow);
                if (hasMore1) PreUpdateRankKWithinGroup(b1, b1, idx, numPanels,
                                                        groupEndRow, groupBoundaryRow);
            }

            int32_t bigM = forward ? (kDim - groupEndRow) : groupBoundaryRow;
            if (bigM > 0) {
                BigGroupGemm(b0, b0, idxStart, groupSize, numPanels);
                if (hasB1) BigGroupGemm(b1, b1, idxStart, groupSize, numPanels);
            }
        }
    }

private:
    __aicore__ inline void SetSplitRange(int32_t sid, int32_t splitFactor)
    {
        splitId = sid;
        if (splitFactor < 1) splitFactor = 1;
        int32_t pieceWidth = CEIL_ALIGN(nCols / splitFactor, FLOAT_ALIGN);
        if (pieceWidth < FLOAT_ALIGN) pieceWidth = FLOAT_ALIGN;
        splitColStart = sid * pieceWidth;
        splitNcols = pieceWidth;
        if (splitColStart + splitNcols > nCols) splitNcols = nCols - splitColStart;
        if (splitColStart >= nCols) { splitColStart = 0; splitNcols = 0; }
    }

    // Check if the panel at the given index has trailing rows needing a rank-K update.
    __aicore__ inline bool PanelHasTrail(int32_t idx, int32_t numPanels)
    {
        int32_t p = forward ? idx : (numPanels - 1 - idx);
        int32_t panelStart = p * nb;
        int32_t actualNb = (nb < kDim - panelStart) ? nb : (kDim - panelStart);
        int32_t trailRows = forward ? (kDim - (panelStart + actualNb)) : panelStart;
        return trailRows > 0;
    }

    // Compute per-(batch,slot) GM pointers. batch = data index, slot = workspace slot.
    __aicore__ inline void Ptrs(int32_t batch, int32_t slot, __gm__ float*& effA, int32_t& effLda,
                                __gm__ float*& effB, int32_t& effLdb, __gm__ float*& gmGemm,
                                __gm__ float*& gmBorig, __gm__ float*& gmBc)
    {
        __gm__ float* gmA = (__gm__ float*)gmAArray.GetValue(batch);
        gmBorig = (__gm__ float*)gmBArray.GetValue(batch);
        __gm__ float* coreWs = (__gm__ float*)gemmWsBase
            + (int64_t)slot * (tiling->workspaceOffset / sizeof(float));
        int32_t splitFactor = tiling->splitFactor;
        __gm__ float* gmAc = coreWs;
        gmBc = coreWs + (int64_t)splitFactor * kDim * kDim;
        __gm__ float* gmGemmBase = gmBc + (int64_t)kDim * nColsAligned;
        int32_t maxMN = kDim > nColsAligned ? kDim : nColsAligned;
        int32_t maxMNAligned = CEIL_ALIGN(maxMN, FLOAT_ALIGN);
        gmGemm = gmGemmBase + (int64_t)splitId * 2 * maxMN * tiling->splitNcolsMax;
        effA = (needTA || padOn) ? gmAc : gmA;
        effLda = (needTA || padOn) ? kDim : tiling->lda;
        effB = (right || padOn) ? gmBc : gmBorig;
        effLdb = (right || padOn) ? nColsAligned : tiling->ldb;
    }

    // Transpose B into workspace via Cube GEMM (RIGHT non-padded path).
    __aicore__ inline void TransInBody(int32_t batch, int32_t slot)
    {
        __gm__ float *effA, *effB, *gmGemm, *gmBorig, *gmBc; int32_t effLda, effLdb;
        Ptrs(batch, slot, effA, effLda, effB, effLdb, gmGemm, gmBorig, gmBc);
        int32_t maxMN = kDim > nCols ? kDim : nCols;
        AscendC::GlobalTensor<float> idT, gmBorigT, gmBcT;
        idT.SetGlobalBuffer((__gm__ float*)idBase, (uint32_t)(maxMN * maxMN));
        gmBorigT.SetGlobalBuffer(gmBorig, (uint32_t)(tiling->m * tiling->ldb));
        gmBcT.SetGlobalBuffer(gmBc, (uint32_t)(kDim * nColsAligned));
        mm.SetOrgShape(kDim, nCols, maxMN, tiling->ldb, nColsAligned);
        mm.SetSingleShape(kDim, nCols, kDim);
        mm.SetTensorA(idT[0], false);
        mm.SetTensorB(gmBorigT[0], true);
        mm.IterateAll(gmBcT);
    }

    // Full trailing rank-K GEMM update for one panel.
    __aicore__ inline void RankKBody(int32_t batch, int32_t slot, int32_t idx, int32_t numPanels)
    {
        int32_t p = forward ? idx : (numPanels - 1 - idx);
        int32_t panelStart = p * nb;
        int32_t actualNb = (nb < kDim - panelStart) ? nb : (kDim - panelStart);
        int32_t trailStart, trailRows;
        if (forward) { trailStart = panelStart + actualNb; trailRows = kDim - trailStart; }
        else { trailStart = 0; trailRows = panelStart; }
        if (trailRows <= 0) return;
        __gm__ float *effA, *effB, *gmGemm, *gmBorig, *gmBc; int32_t effLda, effLdb;
        Ptrs(batch, slot, effA, effLda, effB, effLdb, gmGemm, gmBorig, gmBc);
        int32_t xSlot = XnegSlot(idx, numPanels, forward);
        __gm__ float* gmXneg = gmGemm + (int64_t)xSlot * nb * splitNcols;
        AscendC::GlobalTensor<float> gmAT, gmBT, gmXnegT;
        gmAT.SetGlobalBuffer(effA, (uint32_t)((int64_t)kDim * effLda));
        gmBT.SetGlobalBuffer(effB, (uint32_t)((int64_t)kDim * effLdb));
        gmXnegT.SetGlobalBuffer(gmXneg, (uint32_t)(nb * splitNcols));
        int32_t aOffset = trailStart * effLda + panelStart;
        mm.SetOrgShape(trailRows, splitNcols, effLda, splitNcols, effLdb);
        mm.SetSingleShape(trailRows, splitNcols, actualNb);
        mm.SetTensorA(gmAT[aOffset], false);
        mm.SetTensorB(gmXnegT[0], false);
        mm.IterateAll(gmBT[trailStart * effLdb + splitColStart], 1);
    }

    // Direct update: only the next panel's rows (small GEMM, on critical path).
    // Returns true if there are remaining rows for pre-update.
    __aicore__ inline bool DirectRankK(int32_t batch, int32_t slot, int32_t idx, int32_t numPanels)
    {
        int32_t p = forward ? idx : (numPanels - 1 - idx);
        int32_t panelStart = p * nb;
        int32_t actualNb = (nb < kDim - panelStart) ? nb : (kDim - panelStart);
        int32_t trailStart, trailRows;
        if (forward) { trailStart = panelStart + actualNb; trailRows = kDim - trailStart; }
        else { trailStart = 0; trailRows = panelStart; }
        if (trailRows <= 0) return false;

        int32_t directRows = (nb < trailRows) ? nb : trailRows;
        int32_t directStart = forward ? trailStart : (trailStart + trailRows - directRows);

        __gm__ float *effA, *effB, *gmGemm, *gmBorig, *gmBc; int32_t effLda, effLdb;
        Ptrs(batch, slot, effA, effLda, effB, effLdb, gmGemm, gmBorig, gmBc);
        int32_t xSlot = XnegSlot(idx, numPanels, forward);
        __gm__ float* gmXneg = gmGemm + (int64_t)xSlot * nb * splitNcols;
        AscendC::GlobalTensor<float> gmAT, gmBT, gmXnegT;
        gmAT.SetGlobalBuffer(effA, (uint32_t)((int64_t)kDim * effLda));
        gmBT.SetGlobalBuffer(effB, (uint32_t)((int64_t)kDim * effLdb));
        gmXnegT.SetGlobalBuffer(gmXneg, (uint32_t)(nb * splitNcols));
        int32_t aOffset = directStart * effLda + panelStart;
        mm.SetOrgShape(directRows, splitNcols, effLda, splitNcols, effLdb);
        mm.SetSingleShape(directRows, splitNcols, actualNb);
        mm.SetTensorA(gmAT[aOffset], false);
        mm.SetTensorB(gmXnegT[0], false);
        mm.IterateAll(gmBT[directStart * effLdb + splitColStart], 1);
        return (trailRows > directRows);
    }

    // Pre-update within group: only update rows up to groupEndRow (forward) or down to
    // groupBoundaryRow (backward), avoiding double-counting with BigGroupGemm.
    __aicore__ inline void PreUpdateRankKWithinGroup(int32_t batch, int32_t slot, int32_t idx,
                                                     int32_t numPanels, int32_t groupEndRow,
                                                     int32_t groupBoundaryRow)
    {
        int32_t p = forward ? idx : (numPanels - 1 - idx);
        int32_t panelStart = p * nb;
        int32_t actualNb = (nb < kDim - panelStart) ? nb : (kDim - panelStart);
        int32_t trailStart, trailRows;
        if (forward) { trailStart = panelStart + actualNb; trailRows = kDim - trailStart; }
        else { trailStart = 0; trailRows = panelStart; }

        int32_t directRows = (nb < trailRows) ? nb : trailRows;
        int32_t preRows = trailRows - directRows;
        if (preRows <= 0) return;

        int32_t preStart;
        if (forward) {
            preStart = trailStart + directRows;
            preRows = groupEndRow - preStart;
        } else {
            preStart = groupBoundaryRow;
            int32_t directStart = trailStart + trailRows - directRows;
            preRows = directStart - groupBoundaryRow;
        }
        if (preRows <= 0) return;

        __gm__ float *effA, *effB, *gmGemm, *gmBorig, *gmBc; int32_t effLda, effLdb;
        Ptrs(batch, slot, effA, effLda, effB, effLdb, gmGemm, gmBorig, gmBc);
        int32_t xSlot = XnegSlot(idx, numPanels, forward);
        __gm__ float* gmXneg = gmGemm + (int64_t)xSlot * nb * splitNcols;
        AscendC::GlobalTensor<float> gmAT, gmBT, gmXnegT;
        gmAT.SetGlobalBuffer(effA, (uint32_t)((int64_t)kDim * effLda));
        gmBT.SetGlobalBuffer(effB, (uint32_t)((int64_t)kDim * effLdb));
        gmXnegT.SetGlobalBuffer(gmXneg, (uint32_t)(nb * splitNcols));
        int32_t aOffset = preStart * effLda + panelStart;
        mm.SetOrgShape(preRows, splitNcols, effLda, splitNcols, effLdb);
        mm.SetSingleShape(preRows, splitNcols, actualNb);
        mm.SetTensorA(gmAT[aOffset], false);
        mm.SetTensorB(gmXnegT[0], false);
        mm.IterateAll(gmBT[preStart * effLdb + splitColStart], 1);
    }

    // Big group GEMM at group boundary: K = groupSize*nb, updates all rows outside the group.
    __aicore__ inline void BigGroupGemm(int32_t batch, int32_t slot, int32_t idxStart,
                                         int32_t groupSize, int32_t numPanels)
    {
        int32_t bigM, bigStart, colStart;
        if (forward) {
            int32_t pLast = idxStart + groupSize - 1;
            int32_t groupEndRow = ((pLast + 2) * nb < kDim) ? (pLast + 2) * nb : kDim;
            bigM = kDim - groupEndRow;
            bigStart = groupEndRow;
            colStart = idxStart * nb;
        } else {
            int32_t pLast = numPanels - 1 - (idxStart + groupSize - 1);
            int32_t groupBoundaryRow = ((pLast - 1) * nb > 0) ? (pLast - 1) * nb : 0;
            bigM = groupBoundaryRow;
            bigStart = 0;
            colStart = pLast * nb;
        }
        if (bigM <= 0) return;

        __gm__ float *effA, *effB, *gmGemm, *gmBorig, *gmBc; int32_t effLda, effLdb;
        Ptrs(batch, slot, effA, effLda, effB, effLdb, gmGemm, gmBorig, gmBc);

        int32_t g = idxStart / LIM_GROUP;
        int32_t groupBase = (g % 2) * LIM_GROUP;
        __gm__ float* gmXnegGroup = gmGemm + (int64_t)groupBase * nb * splitNcols;

        int32_t totalK = groupSize * nb;
        AscendC::GlobalTensor<float> gmAT, gmBT, gmXnegT;
        gmAT.SetGlobalBuffer(effA, (uint32_t)((int64_t)kDim * effLda));
        gmBT.SetGlobalBuffer(effB, (uint32_t)((int64_t)kDim * effLdb));
        gmXnegT.SetGlobalBuffer(gmXnegGroup, (uint32_t)(totalK * splitNcols));
        int32_t aOffset = bigStart * effLda + colStart;
        mm.SetOrgShape(bigM, splitNcols, effLda, splitNcols, effLdb);
        mm.SetSingleShape(bigM, splitNcols, totalK);
        mm.SetTensorA(gmAT[aOffset], false);
        mm.SetTensorB(gmXnegT[0], false);
        mm.IterateAll(gmBT[bigStart * effLdb + splitColStart], 1);
    }

    // Transpose workspace B back to original B (RIGHT non-padded path).
    __aicore__ inline void TransOutBody(int32_t batch, int32_t slot)
    {
        __gm__ float *effA, *effB, *gmGemm, *gmBorig, *gmBc; int32_t effLda, effLdb;
        Ptrs(batch, slot, effA, effLda, effB, effLdb, gmGemm, gmBorig, gmBc);
        int32_t maxMN = kDim > nCols ? kDim : nCols;
        AscendC::GlobalTensor<float> idT, gmBorigT, gmBcT;
        idT.SetGlobalBuffer((__gm__ float*)idBase, (uint32_t)(maxMN * maxMN));
        gmBorigT.SetGlobalBuffer(gmBorig, (uint32_t)(tiling->m * tiling->ldb));
        gmBcT.SetGlobalBuffer(gmBc, (uint32_t)(kDim * nColsAligned));
        mm.SetOrgShape(nCols, kDim, maxMN, nColsAligned, tiling->ldb);
        mm.SetSingleShape(nCols, kDim, nCols);
        mm.SetTensorA(idT[0], false);
        mm.SetTensorB(gmBcT[0], true);
        mm.IterateAll(gmBorigT);
    }

    // Process one batch in 1_1 mode with optional TransIn/TransOut for RIGHT side.
    __aicore__ inline void ProcessOneBatch(int32_t batch)
    {
        bool aivTransB = right && (kDim > 512 || nCols > 512);
        if (right && !aivTransB && !padOn) {
            AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(FLAG_TRSV);
            TransInBody(batch, blockIdx);
            AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_GEMM);
        }

        int32_t numPanels = (kDim + nb - 1) / nb;
        if (numPanels <= 2) {
            ProcessOneBatchSimple(batch, numPanels);
        } else {
            ProcessOneBatchGrouped(batch, numPanels);
        }

        if (right && !aivTransB && !padOn) {
            AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(FLAG_TRSV);
            TransOutBody(batch, blockIdx);
            AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_GEMM);
        }
    }

    // Simple panel loop for 1_1 mode (≤2 panels).
    __aicore__ inline void ProcessOneBatchSimple(int32_t batch, int32_t numPanels)
    {
        for (int32_t idx = 0; idx < numPanels; idx++) {
            if (!PanelHasTrail(idx, numPanels)) continue;
            AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(FLAG_TRSV);
            RankKBody(batch, blockIdx, idx, numPanels);
            AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_GEMM);
        }
    }

    // Grouped panel scheduling for 1_1 mode with direct/pre-update/big-group GEMM.
    __aicore__ inline void ProcessOneBatchGrouped(int32_t batch, int32_t numPanels)
    {
        int32_t numGroups = (numPanels + LIM_GROUP - 1) / LIM_GROUP;
        for (int32_t g = 0; g < numGroups; g++) {
            int32_t idxStart, idxEnd, groupSize, groupEndRow, groupBoundaryRow;
            GroupBounds(g, numPanels, idxStart, idxEnd, groupSize, groupEndRow, groupBoundaryRow);

            for (int32_t idx = idxStart; idx <= idxEnd; idx++) {
                if (!PanelHasTrail(idx, numPanels)) continue;
                AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(FLAG_TRSV);
                bool hasMore = DirectRankK(batch, blockIdx, idx, numPanels);
                AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_GEMM);
                if (hasMore) PreUpdateRankKWithinGroup(batch, blockIdx, idx, numPanels,
                                                       groupEndRow, groupBoundaryRow);
            }

            int32_t bigM = forward ? (kDim - groupEndRow) : groupBoundaryRow;
            if (bigM > 0) BigGroupGemm(batch, blockIdx, idxStart, groupSize, numPanels);
        }
    }

    // Compute group start/end/size/boundary for grouped panel scheduling.
    __aicore__ inline void GroupBounds(int32_t grp, int32_t numPanels,
                                        int32_t& idxStart, int32_t& idxEnd, int32_t& groupSize,
                                        int32_t& groupEndRow, int32_t& groupBoundaryRow)
    {
        ::GroupBounds(grp, numPanels, nb, kDim, forward,
                      idxStart, idxEnd, groupSize, groupEndRow, groupBoundaryRow);
    }

    const __gm__ TrsmBatchedTilingData* tiling;
    AscendC::GlobalTensor<uint64_t> gmAArray, gmBArray;
    __gm__ uint8_t* gemmWsBase;
    __gm__ uint8_t* idBase;
    uint32_t blockIdx;
    int32_t kDim, nCols, nb, nColsAligned;
    int32_t batchStart, batchEnd;
    bool forward, needTA, right, padOn;
    int32_t splitColStart, splitNcols, splitId;
};
