/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "ctrsm_batched_kernel_common.h"

typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> MmA;
typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> MmB;
typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> MmC;

class CtrsmMixAic {
public:
    Matmul<MmA, MmB, MmC> mm;
    TCubeTiling cubeTiling;

    __aicore__ inline CtrsmMixAic() {}

    __aicore__ inline void Init(GM_ADDR aArrayGm, GM_ADDR bArrayGm, GM_ADDR gemmWsGm,
                                const __gm__ CtrsmBatchedTilingData* t, const TCubeTiling& ct)
    {
        td = t;
        cubeTiling = ct;
        blockIdx = AscendC::GetBlockIdx();
        right = (td->side == SIDE_RIGHT);
        kDim = right ? td->n : td->m;
        nCols = right ? td->m : td->n;
        nb = td->nb;
        int32_t kDimPad = CEIL_ALIGN(kDim, FLOAT_ALIGN);
        int32_t nColsPad = CEIL_ALIGN(nCols, FLOAT_ALIGN);
        if (kDim != kDimPad || nCols != nColsPad) { kDim = kDimPad; nCols = nColsPad; }
        nColsAligned = CEIL_ALIGN(nCols, FLOAT_ALIGN);
        numSplits = td->numSplits;
        if (numSplits > 1) {
            int32_t splitIdx = (int32_t)blockIdx % numSplits;
            bool isLast = (splitIdx == numSplits - 1);
            int32_t localNColsAligned = isLast ? td->lastNColsAligned : td->splitNColsAligned;
            nCols = localNColsAligned;
            nColsAligned = localNColsAligned;
        }
        bool needTA = NeedTransA(td);
        int32_t effUplo = needTA ? (td->uplo == UPLO_UPPER ? UPLO_LOWER : UPLO_UPPER)
                                 : td->uplo;
        forward = (effUplo == UPLO_LOWER);
        gmAArray.SetGlobalBuffer((__gm__ uint64_t*)aArrayGm, td->batchCount);
        gmBArray.SetGlobalBuffer((__gm__ uint64_t*)bArrayGm, td->batchCount);
        gemmWsBase = gemmWsGm;
        useOrigA = (td->useOrigA == 1);
        aEffStride = td->aEffStride;
    }

    __aicore__ inline void Process12()
    {
        int32_t numPanels = (kDim + nb - 1) / nb;
        if (numSplits > 1) {
            int32_t slot = (int32_t)blockIdx;
            if (numPanels <= 2) {
                Process12Simple(slot, -1, numPanels);
            } else {
                Process12Grouped(slot, -1, numPanels);
            }
        } else if (td->dualAivMode) {
            int32_t batch = (int32_t)blockIdx;
            if (numPanels <= 2) {
                Process12Simple(batch, -1, numPanels);
            } else {
                Process12Grouped(batch, -1, numPanels);
            }
        } else {
            int32_t g = (int32_t)blockIdx;
            int32_t b0 = 2 * g, b1 = 2 * g + 1;
            if (numPanels <= 2) {
                Process12Simple(b0, b1, numPanels);
            } else {
                Process12Grouped(b0, b1, numPanels);
            }
        }
    }

private:
    __aicore__ inline void Ptrs(int32_t slot, __gm__ float*& eA, int32_t& eAStride,
                                __gm__ float*& eBR, __gm__ float*& eBI,
                                int32_t& eLdb, __gm__ float*& gmGemm)
    {
        int32_t batch = (numSplits > 1) ? (slot / numSplits) : slot;
        int32_t splitIdx = (numSplits > 1) ? (slot % numSplits) : 0;
        __gm__ float* batchWs = (__gm__ float*)gemmWsBase
            + (int64_t)batch * (td->workspaceOffset / sizeof(float));
        int64_t aPlane = (int64_t)kDim * kDim;
        if (useOrigA) {
            eA = (__gm__ float*)gmAArray.GetValue(batch);
            eAStride = aEffStride;
        } else {
            eA = batchWs;
            eAStride = 2 * kDim;
        }
        int32_t wsNCols = (numSplits > 1) ?
            ((td->splitNColsAligned > td->lastNColsAligned) ? td->splitNColsAligned : td->lastNColsAligned)
            : nColsAligned;
        int64_t bPlane = (int64_t)kDim * wsNCols;
        int64_t splitBGemmFloats = td->splitBGemmSize / sizeof(float);
        __gm__ float* splitWs = batchWs + 2 * aPlane + (int64_t)splitIdx * splitBGemmFloats;
        eBR = splitWs;
        eBI = eBR + bPlane;
        eLdb = nColsAligned;
        gmGemm = eBI + bPlane;
    }

    __aicore__ inline void SetupXnegPtrs(int32_t slot, int32_t idx, int32_t numPanels,
                                          __gm__ float*& xR, __gm__ float*& xI,
                                          __gm__ float* gmGemm)
    {
        int32_t xSlot = XnegSlot(idx, numPanels, forward);
        int64_t xnegPlane = (int64_t)2 * LIM_GROUP * 2 * nb * nColsAligned;
        xR = gmGemm + (int64_t)xSlot * 2 * nb * nColsAligned;
        xI = gmGemm + xnegPlane + (int64_t)xSlot * 2 * nb * nColsAligned;
    }

    __aicore__ inline void DoComplexGemm(
        AscendC::GlobalTensor<float>& gA,
        AscendC::GlobalTensor<float>& gBR, AscendC::GlobalTensor<float>& gBI,
        AscendC::GlobalTensor<float>& gXReal, AscendC::GlobalTensor<float>& gXImag,
        int32_t eAStride, int32_t eLdb,
        int32_t rowStart, int32_t rowCount, int32_t colStart, int32_t actualK)
    {
        int32_t aOff = rowStart * eAStride + colStart * 2;
        mm.SetOrgShape(rowCount, nCols, eAStride, nColsAligned, eLdb);
        mm.SetSingleShape(rowCount, nCols, 2 * actualK);
        mm.SetTensorA(gA[aOff], false);
        mm.SetTensorB(gXReal[0], false);
        mm.IterateAll(gBR[rowStart * eLdb], 1);
        mm.SetOrgShape(rowCount, nCols, eAStride, nColsAligned, eLdb);
        mm.SetSingleShape(rowCount, nCols, 2 * actualK);
        mm.SetTensorA(gA[aOff], false);
        mm.SetTensorB(gXImag[0], false);
        mm.IterateAll(gBI[rowStart * eLdb], 1);
    }

    // 组装本 slot 的 A/B/Xneg 全局张量并执行一次复数 GEMM 更新。
    // DirectRankK / PreUpdateRankKWithinGroup / FullTrailUpdateWithinGroup 共用此流程，
    // 仅更新的行区间 [rowStart, rowStart+rowCount) 不同。
    __aicore__ inline void SetupAndGemm(int32_t slot, int32_t idx, int32_t numPanels,
                                        int32_t rowStart, int32_t rowCount,
                                        int32_t panelStart, int32_t actualNb)
    {
        __gm__ float *eA, *eBR, *eBI, *gmGemm, *xR, *xI;
        int32_t eAStride, eLdb;
        Ptrs(slot, eA, eAStride, eBR, eBI, eLdb, gmGemm);
        SetupXnegPtrs(slot, idx, numPanels, xR, xI, gmGemm);
        AscendC::GlobalTensor<float> gA, gBR, gBI, gXReal, gXImag;
        gA.SetGlobalBuffer(eA, (uint32_t)((int64_t)kDim * eAStride));
        gBR.SetGlobalBuffer(eBR, (uint32_t)((int64_t)kDim * eLdb));
        gBI.SetGlobalBuffer(eBI, (uint32_t)((int64_t)kDim * eLdb));
        gXReal.SetGlobalBuffer(xR, (uint32_t)((int64_t)2 * nb * nColsAligned));
        gXImag.SetGlobalBuffer(xI, (uint32_t)((int64_t)2 * nb * nColsAligned));
        DoComplexGemm(gA, gBR, gBI, gXReal, gXImag,
                      eAStride, eLdb, rowStart, rowCount, panelStart, actualNb);
    }

    __aicore__ inline void ComputeTrail(int32_t idx, int32_t numPanels,
                                         int32_t& panelStart, int32_t& actualNb,
                                         int32_t& trailStart, int32_t& trailRows)
    {
        int32_t p = forward ? idx : (numPanels - 1 - idx);
        panelStart = p * nb;
        actualNb = (nb < kDim - panelStart) ? nb : (kDim - panelStart);
        if (forward) { trailStart = panelStart + actualNb; trailRows = kDim - trailStart; }
        else { trailStart = 0; trailRows = panelStart; }
    }

    __aicore__ inline bool DirectRankK(int32_t slot, int32_t idx, int32_t numPanels)
    {
        int32_t panelStart, actualNb, trailStart, trailRows;
        ComputeTrail(idx, numPanels, panelStart, actualNb, trailStart, trailRows);
        if (trailRows <= 0) return false;
        int32_t directRows = (nb < trailRows) ? nb : trailRows;
        int32_t directStart = forward ? trailStart : (trailStart + trailRows - directRows);
        SetupAndGemm(slot, idx, numPanels, directStart, directRows, panelStart, actualNb);
        return (trailRows > directRows);
    }

    __aicore__ inline void PreUpdateRankKWithinGroup(int32_t slot, int32_t idx, int32_t numPanels,
                                                     int32_t groupEndRow, int32_t groupBoundaryRow)
    {
        int32_t panelStart, actualNb, trailStart, trailRows;
        ComputeTrail(idx, numPanels, panelStart, actualNb, trailStart, trailRows);
        int32_t directRows = (nb < trailRows) ? nb : trailRows;
        int32_t preStart, preRows;
        if (forward) { preStart = trailStart + directRows; preRows = groupEndRow - preStart; }
        else {
            int32_t directStart = trailStart + trailRows - directRows;
            preStart = groupBoundaryRow;
            preRows = directStart - groupBoundaryRow;
        }
        if (preRows <= 0) return;
        SetupAndGemm(slot, idx, numPanels, preStart, preRows, panelStart, actualNb);
    }

    __aicore__ inline void BigGroupGemm(int32_t slot, int32_t idxStart, int32_t groupSize,
                                         int32_t numPanels)
    {
        int32_t bigM, bigStart, colStart;
        ComputeBigGemmBounds(idxStart, groupSize, numPanels, bigM, bigStart, colStart);
        if (bigM <= 0) return;
        __gm__ float *eA, *eBR, *eBI, *gmGemm;
        int32_t eAStride, eLdb;
        Ptrs(slot, eA, eAStride, eBR, eBI, eLdb, gmGemm);
        int32_t g = idxStart / LIM_GROUP;
        int32_t groupBase = (g % 2) * LIM_GROUP;
        int64_t xnegPlane = (int64_t)2 * LIM_GROUP * 2 * nb * nColsAligned;
        __gm__ float* xR = gmGemm + (int64_t)groupBase * 2 * nb * nColsAligned;
        __gm__ float* xI = gmGemm + xnegPlane + (int64_t)groupBase * 2 * nb * nColsAligned;
        int32_t totalK = groupSize * nb;
        int32_t maxK = kDim - colStart;
        if (totalK > maxK) totalK = maxK;
        AscendC::GlobalTensor<float> gA, gBR, gBI, gXReal, gXImag;
        gA.SetGlobalBuffer(eA, (uint32_t)((int64_t)kDim * eAStride));
        gBR.SetGlobalBuffer(eBR, (uint32_t)((int64_t)kDim * eLdb));
        gBI.SetGlobalBuffer(eBI, (uint32_t)((int64_t)kDim * eLdb));
        gXReal.SetGlobalBuffer(xR, (uint32_t)((int64_t)2 * totalK * nColsAligned));
        gXImag.SetGlobalBuffer(xI, (uint32_t)((int64_t)2 * totalK * nColsAligned));
        DoComplexGemm(gA, gBR, gBI, gXReal, gXImag,
                      eAStride, eLdb, bigStart, bigM, colStart, totalK);
    }

    // 用组内 panel 子区间 [panelOffset, panelOffset+accPanels) 的 Xneg 更新远端行块
    // [bigStart, bigStart+bigM)。K=accPanels*nb。用于把 BigGroupGemm 按 K 拆分、提前与 AIV 并行
    __aicore__ inline void FarGemm(int32_t slot, int32_t idxStart, int32_t panelOffset,
                                    int32_t accPanels, int32_t bigStart, int32_t bigM)
    {
        if (bigM <= 0 || accPanels <= 0) return;
        __gm__ float *eA, *eBR, *eBI, *gmGemm;
        int32_t eAStride, eLdb;
        Ptrs(slot, eA, eAStride, eBR, eBI, eLdb, gmGemm);
        int32_t g = idxStart / LIM_GROUP;
        int32_t groupBase = (g % 2) * LIM_GROUP;
        int64_t xnegPlane = (int64_t)2 * LIM_GROUP * 2 * nb * nColsAligned;
        int32_t baseSlot = groupBase + panelOffset;
        __gm__ float* xR = gmGemm + (int64_t)baseSlot * 2 * nb * nColsAligned;
        __gm__ float* xI = gmGemm + xnegPlane + (int64_t)baseSlot * 2 * nb * nColsAligned;
        int32_t totalK = accPanels * nb;
        int32_t colStart = (idxStart + panelOffset) * nb;
        AscendC::GlobalTensor<float> gA, gBR, gBI, gXReal, gXImag;
        gA.SetGlobalBuffer(eA, (uint32_t)((int64_t)kDim * eAStride));
        gBR.SetGlobalBuffer(eBR, (uint32_t)((int64_t)kDim * eLdb));
        gBI.SetGlobalBuffer(eBI, (uint32_t)((int64_t)kDim * eLdb));
        gXReal.SetGlobalBuffer(xR, (uint32_t)((int64_t)2 * totalK * nColsAligned));
        gXImag.SetGlobalBuffer(xI, (uint32_t)((int64_t)2 * totalK * nColsAligned));
        DoComplexGemm(gA, gBR, gBI, gXReal, gXImag,
                      eAStride, eLdb, bigStart, bigM, colStart, totalK);
    }

    __aicore__ inline void ComputeBigGemmBounds(int32_t idxStart, int32_t groupSize,
                                                 int32_t numPanels,
                                                 int32_t& bigM, int32_t& bigStart, int32_t& colStart)
    {
        if (forward) {
            int32_t pLast = idxStart + groupSize - 1;
            int32_t groupEndRow = (((pLast + 2) * nb < kDim) ? (pLast + 2) * nb : kDim);
            bigM = kDim - groupEndRow;
            bigStart = groupEndRow;
            colStart = idxStart * nb;
        } else {
            int32_t pLast = numPanels - 1 - (idxStart + groupSize - 1);
            int32_t groupBoundaryRow = (((pLast - 1) * nb > 0) ? (pLast - 1) * nb : 0);
            bigM = groupBoundaryRow;
            bigStart = 0;
            colStart = pLast * nb;
        }
    }

    __aicore__ inline bool PanelHasTrail(int32_t idx, int32_t numPanels)
    {
        int32_t panelStart, actualNb, trailStart, trailRows;
        ComputeTrail(idx, numPanels, panelStart, actualNb, trailStart, trailRows);
        return trailRows > 0;
    }

    __aicore__ inline void Process12Simple(int32_t b0, int32_t b1, int32_t numPanels)
    {
        bool useMerged = (td->dualAivMode == 0);
        for (int32_t idx = 0; idx < numPanels; idx++) {
            if (!PanelHasTrail(idx, numPanels)) continue;
            AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(FLAG_TRSV);
            if (useMerged) {
                FullTrailUpdateWithinGroup(b0, idx, numPanels, kDim, 0);
                if (b1 >= 0 && b1 < td->batchCount)
                    FullTrailUpdateWithinGroup(b1, idx, numPanels, kDim, 0);
                AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_GEMM);
            } else {
                bool hasMore0 = DirectRankK(b0, idx, numPanels);
                bool hasMore1 = (b1 >= 0 && b1 < td->batchCount) ? DirectRankK(b1, idx, numPanels) : false;
                AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_GEMM);
                if (hasMore0) PreUpdateRankKWithinGroup(b0, idx, numPanels, kDim, 0);
                if (hasMore1) PreUpdateRankKWithinGroup(b1, idx, numPanels, kDim, 0);
            }
        }
    }

    __aicore__ inline void GroupBounds(int32_t grp, int32_t numPanels,
                                        int32_t& idxStart, int32_t& idxEnd, int32_t& groupSize,
                                        int32_t& groupEndRow, int32_t& groupBoundaryRow)
    {
        ::GroupBounds(grp, numPanels, nb, kDim, forward,
                      idxStart, idxEnd, groupSize, groupEndRow, groupBoundaryRow);
    }

    __aicore__ inline void Process12Grouped(int32_t b0, int32_t b1, int32_t numPanels)
    {
        int32_t numGroups = (numPanels + LIM_GROUP - 1) / LIM_GROUP;
        for (int32_t grp = 0; grp < numGroups; grp++) {
            int32_t idxStart, idxEnd, groupSize, groupEndRow, groupBoundaryRow;
            GroupBounds(grp, numPanels, idxStart, idxEnd, groupSize, groupEndRow, groupBoundaryRow);
            ProcessOneGroup(b0, b1, numPanels, idxStart, idxEnd, groupSize,
                           groupEndRow, groupBoundaryRow);
        }
    }

    // 单个 panel 的 trailing 更新（merged 全量 / direct+preupdate 两条路径）
    __aicore__ inline void UpdateOnePanel(int32_t b0, int32_t b1, int32_t idx, int32_t numPanels,
                                           int32_t groupEndRow, int32_t groupBoundaryRow, bool useMerged)
    {
        bool hasB1 = (b1 >= 0 && b1 < td->batchCount);
        if (useMerged) {
            FullTrailUpdateWithinGroup(b0, idx, numPanels, groupEndRow, groupBoundaryRow);
            if (hasB1) FullTrailUpdateWithinGroup(b1, idx, numPanels, groupEndRow, groupBoundaryRow);
            AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_GEMM);
            return;
        }
        bool hasMore0 = DirectRankK(b0, idx, numPanels);
        bool hasMore1 = hasB1 ? DirectRankK(b1, idx, numPanels) : false;
        AscendC::PipeBarrier<PIPE_FIX>();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(FLAG_GEMM);
        if (hasMore0) PreUpdateRankKWithinGroup(b0, idx, numPanels, groupEndRow, groupBoundaryRow);
        if (hasMore1) PreUpdateRankKWithinGroup(b1, idx, numPanels, groupEndRow, groupBoundaryRow);
    }

    // 组末远端更新收尾：doSplit 补齐剩余段，否则做整块 BigGroupGemm
    __aicore__ inline void FinishGroupFar(int32_t b0, int32_t b1, int32_t idxStart, int32_t groupSize,
                                           int32_t numPanels, int32_t groupEndRow, int32_t groupBoundaryRow,
                                           bool doSplit, int32_t doneP, int32_t bigStartF, int32_t bigMF)
    {
        if (doSplit) {
            int32_t rem = groupSize - doneP;
            if (rem > 0) FarGemm(b0, idxStart, doneP, rem, bigStartF, bigMF);
            return;
        }
        int32_t bigM = forward ? (kDim - groupEndRow) : groupBoundaryRow;
        if (bigM <= 0) return;
        BigGroupGemm(b0, idxStart, groupSize, numPanels);
        if (b1 >= 0 && b1 < td->batchCount) BigGroupGemm(b1, idxStart, groupSize, numPanels);
    }

    __aicore__ inline void ProcessOneGroup(int32_t b0, int32_t b1, int32_t numPanels,
                                            int32_t idxStart, int32_t idxEnd, int32_t groupSize,
                                            int32_t groupEndRow, int32_t groupBoundaryRow)
    {
        bool useMerged = (td->dualAivMode == 0);
        constexpr int32_t FAR_STEP = 2;
        // forward: 把远端 BigGroupGemm 按 K 分成多段（每 FAR_STEP 个 panel 一段），
        // 每段解完后立即用该段 X（K=FAR_STEP*nb）更新远端大块，与 AIV 解后续 panel 并行。
        // 各段 K 区间不相交 → 累加不重复；远端行 AIV 不碰 → 无 race。
        int32_t bigStartF = forward ? groupEndRow : 0;
        int32_t bigMF = forward ? (kDim - groupEndRow) : 0;
        bool doSplit = false;
        int32_t doneP = 0;   // 已用 FarGemm 处理的 panel 数

        for (int32_t idx = idxStart; idx <= idxEnd; idx++) {
            if (!PanelHasTrail(idx, numPanels)) continue;
            AscendC::CrossCoreWaitFlag<2, PIPE_FIX>(FLAG_TRSV);
            UpdateOnePanel(b0, b1, idx, numPanels, groupEndRow, groupBoundaryRow, useMerged);
            // 每积累满 FAR_STEP 个 panel，立即对远端大块做一段更新（与 AIV 并行）
            if (doSplit && (idx - idxStart + 1 - doneP) >= FAR_STEP) {
                FarGemm(b0, idxStart, doneP, FAR_STEP, bigStartF, bigMF);
                doneP += FAR_STEP;
            }
        }
        FinishGroupFar(b0, b1, idxStart, groupSize, numPanels, groupEndRow, groupBoundaryRow,
                       doSplit, doneP, bigStartF, bigMF);
    }

    __aicore__ inline void FullTrailUpdateWithinGroup(int32_t slot, int32_t idx, int32_t numPanels,
                                                      int32_t groupEndRow, int32_t groupBoundaryRow)
    {
        int32_t panelStart, actualNb, trailStart, trailRows;
        ComputeTrail(idx, numPanels, panelStart, actualNb, trailStart, trailRows);
        if (trailRows <= 0) return;
        int32_t updateStart, updateRows;
        if (forward) {
            updateStart = trailStart;
            updateRows = groupEndRow - trailStart;
        } else {
            updateStart = groupBoundaryRow;
            updateRows = trailStart + trailRows - groupBoundaryRow;
        }
        if (updateRows <= 0) return;
        SetupAndGemm(slot, idx, numPanels, updateStart, updateRows, panelStart, actualNb);
    }

    const __gm__ CtrsmBatchedTilingData* td;
    AscendC::GlobalTensor<uint64_t> gmAArray, gmBArray;
    __gm__ uint8_t* gemmWsBase;
    uint32_t blockIdx;
    int32_t kDim, nCols, nb, nColsAligned;
    int32_t aEffStride;
    int32_t numSplits;
    bool forward, right, useOrigA;
};
