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
 * \file trsm_batched_aiv_canon.h
 * \brief 规范化器：把用户 A/B 转成求解所需的 canonical 形式（转置/补零/alpha），
 *        以及求解后把 X' 回写为用户 B。含非协作 fast 路径与协作列拆分 coop 路径。
 *        转置委托给 TrsmTranspose；仅使用 bufRow 这一片 UB。
 */

#pragma once

#include "trsm_batched_aiv_cfg.h"
#include "trsm_batched_aiv_transpose.h"

class TrsmCanonicalizer {
public:
    __aicore__ inline TrsmCanonicalizer() {}

    __aicore__ inline void Bind(const TrsmAivCfg* cfg, TrsmTranspose* tr,
                                 BufVecCalc* bufRow, BufVecCalc* bufBatch = nullptr,
                                 BufVecCalc* bufBatch2 = nullptr)
    {
        cfg_ = cfg;
        tr_ = tr;
        bufRow_ = bufRow;
        bufBatch_ = bufBatch;
        bufBatch2_ = bufBatch2;
    }

    // Bc rows [ps..ps+nbP) -> B columns [ps..ps+nbP): post-transpose solved rows.
    __aicore__ inline void PostTransposePanel(__gm__ float* gmBc, __gm__ float* gmB,
                                               int32_t ps, int32_t nbP)
    {
        if (!cfg_->coopAiv) {
            tr_->Run(gmBc + (int64_t)ps * cfg_->nColsAligned,
                     nbP, cfg_->nColsOrig, cfg_->nColsAligned,
                     gmB + ps, cfg_->tiling->ldb);
            return;
        }
        int32_t origStart = (cfg_->solveColStart < cfg_->nColsOrig) ? cfg_->solveColStart : cfg_->nColsOrig;
        int32_t origEnd = (cfg_->solveColEnd < cfg_->nColsOrig) ? cfg_->solveColEnd : cfg_->nColsOrig;
        int32_t width = origEnd - origStart;
        if (width <= 0) return;
        tr_->Run(gmBc + (int64_t)ps * cfg_->nColsAligned + origStart,
                 nbP, width, cfg_->nColsAligned,
                 gmB + (int64_t)origStart * cfg_->tiling->ldb + ps, cfg_->tiling->ldb);
    }

    // Prepare effective B pointer: transpose B if needed, or sync with AIC for right-side non-transpose
    __aicore__ inline void PrepareEffB(__gm__ float* gmB, __gm__ float* gmBc,
                                        bool aivTransB, __gm__ float*& effB, int32_t& effLdb)
    {
        if (cfg_->right) {
            if (aivTransB) {
                PrepareEffBCoopTranspose(gmB, gmBc);
                AscendC::PipeBarrier<PIPE_ALL>();
            } else {
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(FLAG_TRSV);
                AscendC::CrossCoreWaitFlag<2, PIPE_MTE2>(FLAG_GEMM);
            }
            effB = gmBc; effLdb = cfg_->nColsAligned;
        } else {
            effB = gmB; effLdb = cfg_->tiling->ldb;
        }
    }

    // Perform cooperative or full transpose of B into gmBc for right-side solve
    __aicore__ inline void PrepareEffBCoopTranspose(__gm__ float* gmB, __gm__ float* gmBc)
    {
        if (cfg_->coopAiv) {
            int32_t origStart = (cfg_->solveColStart < cfg_->nColsOrig) ? cfg_->solveColStart : cfg_->nColsOrig;
            int32_t origEnd = (cfg_->solveColEnd < cfg_->nColsOrig) ? cfg_->solveColEnd : cfg_->nColsOrig;
            int32_t origWidth = origEnd - origStart;
            if (origWidth > 0) {
                tr_->Run(gmB + (int64_t)origStart * cfg_->tiling->ldb,
                         origWidth, cfg_->tiling->n, cfg_->tiling->ldb,
                         gmBc + origStart, cfg_->nColsAligned);
            }
        } else {
            tr_->Run(gmB, cfg_->tiling->m, cfg_->tiling->n, cfg_->tiling->ldb, gmBc, cfg_->nColsAligned);
        }
    }

    // canonical A' (needTA ? A^T : A), kDimOrig x kDimOrig, into gmAc (kDim x kDim, ld=kDim) with
    // identity-diagonal padding on rows/cols [kDimOrig, kDim).
    // When coopAiv=true, two AIVs split the work: subBlock=0 handles first half rows, subBlock=1 second half.
    __aicore__ inline void BuildPaddedA(__gm__ float* gmA, __gm__ float* gmAc)
    {
        int32_t kDim = cfg_->kDim, kDimOrig = cfg_->kDimOrig;
        AscendC::GlobalTensor<float> gmAcT;
        gmAcT.SetGlobalBuffer(gmAc, (uint32_t)(kDim * kDim));

        int32_t myRowStart = 0, myRowEnd = kDimOrig;
        if (cfg_->coopAiv) {
            int32_t nb = cfg_->nb;
            int32_t halfRows = CEIL_ALIGN(kDimOrig / 2, nb);
            if (halfRows > kDimOrig) halfRows = kDimOrig;
            bool isSubBlock0 = (cfg_->solveColStart == 0);
            myRowStart = isSubBlock0 ? 0 : halfRows;
            myRowEnd = isSubBlock0 ? halfRows : kDimOrig;
        }

        if (cfg_->needTA) {
            BuildPaddedATranspose(gmA, gmAc, myRowStart, myRowEnd);
        } else {
            BuildPaddedACopy(gmA, gmAcT, myRowStart, myRowEnd);
        }

        BuildPaddedAIdentityRows(gmAcT);
    }

    // Transpose path for BuildPaddedA: transpose rows [myRowStart, myRowEnd) of A into gmAc
    __aicore__ inline void BuildPaddedATranspose(__gm__ float* gmA, __gm__ float* gmAc,
                                                  int32_t myRowStart, int32_t myRowEnd)
    {
        int32_t kDim = cfg_->kDim, kDimOrig = cfg_->kDimOrig;
        int32_t rowCount = myRowEnd - myRowStart;
        if (rowCount > 0) {
            tr_->Run(gmA + (int64_t)myRowStart * cfg_->tiling->lda,
                     rowCount, kDimOrig, cfg_->tiling->lda,
                     gmAc + myRowStart, kDim);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Non-transpose batch copy path for BuildPaddedA: copy rows [myRowStart, myRowEnd) with padding
    __aicore__ inline void BuildPaddedACopy(__gm__ float* gmA, AscendC::GlobalTensor<float>& gmAcT,
                                             int32_t myRowStart, int32_t myRowEnd)
    {
        int32_t kDimOrig = cfg_->kDimOrig;
        AscendC::GlobalTensor<float> gmAT;
        gmAT.SetGlobalBuffer(gmA, (uint32_t)(kDimOrig * cfg_->tiling->lda));
        if (bufBatch_ != nullptr) {
            BuildPaddedACopyBatch(gmAT, gmAcT, myRowStart, myRowEnd);
        } else {
            BuildPaddedACopyRow(gmAT, gmAcT, myRowStart, myRowEnd);
        }
    }

    // Ping-pong batch copy path for BuildPaddedACopy
    __aicore__ inline void BuildPaddedACopyBatch(AscendC::GlobalTensor<float>& gmAT,
                                                  AscendC::GlobalTensor<float>& gmAcT,
                                                  int32_t myRowStart, int32_t myRowEnd)
    {
        int32_t kDim = cfg_->kDim, kDimOrig = cfg_->kDimOrig;
        int32_t lda = cfg_->tiling->lda;
        AscendC::LocalTensor<float> ubPing = bufBatch_->Get<float>();
        AscendC::LocalTensor<float> ubPong = (bufBatch2_ != nullptr)
            ? bufBatch2_->Get<float>() : ubPing;
        int32_t halfFloats = ubPing.GetSize() / sizeof(float);
        int32_t rowsPerBatch = halfFloats / kDim;
        if (rowsPerBatch > 4095) rowsPerBatch = 4095;
        if (rowsPerBatch > (myRowEnd - myRowStart)) rowsPerBatch = (myRowEnd - myRowStart);
        if (rowsPerBatch < 1) rowsPerBatch = 1;
        int32_t rPad = kDim - kDimOrig;
        AscendC::DataCopyPadExtParams<float> padIn{true, 0, (uint8_t)rPad, 0.0f};
        bool usePing = true;
        int32_t firstBatch = (rowsPerBatch < (myRowEnd - myRowStart)) ? rowsPerBatch : (myRowEnd - myRowStart);
        AscendC::DataCopyExtParams rFirst((uint16_t)firstBatch,
            (uint32_t)(kDimOrig * sizeof(float)),
            (int64_t)((lda - kDimOrig) * sizeof(float)), 0, 0);
        AscendC::DataCopyPad(ubPing, gmAT[myRowStart * lda], rFirst, padIn);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t i = myRowStart; i < myRowEnd; i += rowsPerBatch) {
            int32_t batch = (rowsPerBatch < myRowEnd - i) ? rowsPerBatch : (myRowEnd - i);
            AscendC::DataCopyExtParams wParams((uint16_t)batch,
                (uint32_t)(kDim * sizeof(float)), 0, 0, 0);
            int32_t nextI = i + rowsPerBatch;
            AscendC::LocalTensor<float>& curBuf = usePing ? ubPing : ubPong;
            if (nextI < myRowEnd) {
                AscendC::LocalTensor<float>& nextBuf = usePing ? ubPong : ubPing;
                int32_t nextBatch = (rowsPerBatch < myRowEnd - nextI) ? rowsPerBatch : (myRowEnd - nextI);
                AscendC::DataCopyExtParams rNext((uint16_t)nextBatch,
                    (uint32_t)(kDimOrig * sizeof(float)),
                    (int64_t)((lda - kDimOrig) * sizeof(float)), 0, 0);
                AscendC::DataCopyPad(gmAcT[(int64_t)i * kDim], curBuf, wParams);
                AscendC::DataCopyPad(nextBuf, gmAT[nextI * lda], rNext, padIn);
                AscendC::PipeBarrier<PIPE_ALL>();
                usePing = !usePing;
            } else {
                AscendC::DataCopyPad(gmAcT[(int64_t)i * kDim], curBuf, wParams);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    // Row-by-row fallback path for BuildPaddedACopy
    __aicore__ inline void BuildPaddedACopyRow(AscendC::GlobalTensor<float>& gmAT,
                                                AscendC::GlobalTensor<float>& gmAcT,
                                                int32_t myRowStart, int32_t myRowEnd)
    {
        int32_t kDim = cfg_->kDim, kDimOrig = cfg_->kDimOrig;
        int32_t lda = cfg_->tiling->lda;
        AscendC::LocalTensor<float> ub = bufRow_->Get<float>();
        AscendC::Duplicate(ub, 0.0f, kDim);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyExtParams rRow(1, (uint32_t)(kDimOrig * sizeof(float)), 0, 0, 0);
        AscendC::DataCopyExtParams wRow(1, (uint32_t)(kDim * sizeof(float)), 0, 0, 0);
        for (int32_t i = myRowStart; i < myRowEnd; i++) {
            AscendC::DataCopyPad(ub, gmAT[i * lda], rRow, pad);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(gmAcT[(int64_t)i * kDim], ub, wRow);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // Pad rows [kDimOrig, kDim) with identity diagonal
    __aicore__ inline void BuildPaddedAIdentityRows(AscendC::GlobalTensor<float>& gmAcT)
    {
        int32_t kDim = cfg_->kDim, kDimOrig = cfg_->kDimOrig;
        AscendC::LocalTensor<float> ub = bufRow_->Get<float>();
        AscendC::DataCopyExtParams wRow(1, (uint32_t)(kDim * sizeof(float)), 0, 0, 0);
        for (int32_t i = kDimOrig; i < kDim; i++) {
            AscendC::Duplicate(ub, 0.0f, kDim);
            AscendC::PipeBarrier<PIPE_ALL>();
            ub.SetValue(i, 1.0f);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(gmAcT[(int64_t)i * kDim], ub, wRow);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // canonical B' (right ? B^T : B), into gmBc (kDim x nColsAligned), alpha-scaled, zero-padded.
    __aicore__ inline void BuildPaddedB(__gm__ float* gmB, __gm__ float* gmBc)
    {
        if (!cfg_->coopAiv) {
            BuildPaddedBFast(gmB, gmBc);
            return;
        }
        BuildPaddedBCoopCopy(gmB, gmBc);
        BuildPaddedBCoopScale(gmBc);
    }

    // Non-cooperative fast path: copy B to padded workspace (workspace pre-zeroed by host)
    __aicore__ inline void BuildPaddedBFast(__gm__ float* gmB, __gm__ float* gmBc)
    {
        int32_t kDim = cfg_->kDim, kDimOrig = cfg_->kDimOrig;
        int32_t nColsAligned = cfg_->nColsAligned, nColsOrig = cfg_->nColsOrig;
        float alpha = cfg_->tiling->alphaReal;
        AscendC::GlobalTensor<float> gmBT, gmBcT;
        gmBT.SetGlobalBuffer(gmB, (uint32_t)(cfg_->tiling->m * cfg_->tiling->ldb));
        gmBcT.SetGlobalBuffer(gmBc, (uint32_t)(kDim * nColsAligned));

        if (cfg_->right) {
            BuildPaddedBFastTranspose(gmB, gmBc);
        } else if (bufBatch_ != nullptr) {
            BuildPaddedBFastBatchCopy(gmBT, gmBcT);
        } else {
            BuildPaddedBFastRowCopy(gmBT, gmBcT);
        }
        if (alpha != 1.0f) {
            BuildPaddedBFastAlphaScale(gmBcT);
        }
    }

    // Right-side transpose path for BuildPaddedBFast
    __aicore__ inline void BuildPaddedBFastTranspose(__gm__ float* gmB, __gm__ float* gmBc)
    {
        int32_t kDimOrig = cfg_->kDimOrig, nColsOrig = cfg_->nColsOrig;
        int32_t nColsAligned = cfg_->nColsAligned;
        tr_->Run(gmB, nColsOrig, kDimOrig, cfg_->tiling->ldb, gmBc, nColsAligned);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    // Batch copy (ping-pong) path for BuildPaddedBFast
    __aicore__ inline void BuildPaddedBFastBatchCopy(AscendC::GlobalTensor<float>& gmBT,
                                                      AscendC::GlobalTensor<float>& gmBcT)
    {
        int32_t kDimOrig = cfg_->kDimOrig;
        int32_t nColsAligned = cfg_->nColsAligned, nColsOrig = cfg_->nColsOrig;
        int32_t ldb = cfg_->tiling->ldb;
        AscendC::LocalTensor<float> ubPing = bufBatch_->Get<float>();
        AscendC::LocalTensor<float> ubPong = (bufBatch2_ != nullptr)
            ? bufBatch2_->Get<float>() : ubPing;
        int32_t halfFloats = ubPing.GetSize() / sizeof(float);
        int32_t rowsPerBatch = halfFloats / nColsAligned;
        if (rowsPerBatch > 4095) rowsPerBatch = 4095;
        if (rowsPerBatch > kDimOrig) rowsPerBatch = kDimOrig;
        if (rowsPerBatch < 1) rowsPerBatch = 1;
        int32_t rPad = nColsAligned - nColsOrig;
        AscendC::DataCopyPadExtParams<float> padIn{true, 0, (uint8_t)rPad, 0.0f};
        bool usePing = true;
        int32_t firstBatch = (rowsPerBatch < kDimOrig) ? rowsPerBatch : kDimOrig;
        AscendC::DataCopyExtParams rFirst((uint16_t)firstBatch,
            (uint32_t)(nColsOrig * sizeof(float)),
            (int64_t)((ldb - nColsOrig) * sizeof(float)), 0, 0);
        AscendC::DataCopyPad(ubPing, gmBT[0], rFirst, padIn);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < kDimOrig; i += rowsPerBatch) {
            int32_t batch = (rowsPerBatch < kDimOrig - i) ? rowsPerBatch : (kDimOrig - i);
            AscendC::DataCopyExtParams wParams((uint16_t)batch,
                (uint32_t)(nColsAligned * sizeof(float)), 0, 0, 0);
            int32_t nextI = i + rowsPerBatch;
            AscendC::LocalTensor<float>& curBuf = usePing ? ubPing : ubPong;
            if (nextI < kDimOrig) {
                AscendC::LocalTensor<float>& nextBuf = usePing ? ubPong : ubPing;
                int32_t nextBatch = (rowsPerBatch < kDimOrig - nextI) ? rowsPerBatch : (kDimOrig - nextI);
                AscendC::DataCopyExtParams rNext((uint16_t)nextBatch,
                    (uint32_t)(nColsOrig * sizeof(float)),
                    (int64_t)((ldb - nColsOrig) * sizeof(float)), 0, 0);
                AscendC::DataCopyPad(gmBcT[(int64_t)i * nColsAligned], curBuf, wParams);
                AscendC::DataCopyPad(nextBuf, gmBT[nextI * ldb], rNext, padIn);
                AscendC::PipeBarrier<PIPE_ALL>();
                usePing = !usePing;
            } else {
                AscendC::DataCopyPad(gmBcT[(int64_t)i * nColsAligned], curBuf, wParams);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    // Row-by-row fallback copy for BuildPaddedBFast (no batch buffer)
    __aicore__ inline void BuildPaddedBFastRowCopy(AscendC::GlobalTensor<float>& gmBT,
                                                    AscendC::GlobalTensor<float>& gmBcT)
    {
        int32_t kDimOrig = cfg_->kDimOrig;
        int32_t nColsAligned = cfg_->nColsAligned, nColsOrig = cfg_->nColsOrig;
        AscendC::LocalTensor<float> ub = bufRow_->Get<float>();
        AscendC::Duplicate(ub, 0.0f, nColsAligned);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyExtParams rRow(1, (uint32_t)(nColsOrig * sizeof(float)), 0, 0, 0);
        AscendC::DataCopyExtParams wRowFull(1, (uint32_t)(nColsAligned * sizeof(float)), 0, 0, 0);
        for (int32_t i = 0; i < kDimOrig; i++) {
            AscendC::DataCopyPad(ub, gmBT[i * cfg_->tiling->ldb], rRow, pad);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(gmBcT[(int64_t)i * nColsAligned], ub, wRowFull);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // Alpha scaling loop for BuildPaddedBFast
    __aicore__ inline void BuildPaddedBFastAlphaScale(AscendC::GlobalTensor<float>& gmBcT)
    {
        int32_t kDimOrig = cfg_->kDimOrig;
        int32_t nColsAligned = cfg_->nColsAligned;
        float alpha = cfg_->tiling->alphaReal;
        AscendC::LocalTensor<float> ub = bufRow_->Get<float>();
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyExtParams rRow(1, (uint32_t)(nColsAligned * sizeof(float)), 0, 0, 0);
        AscendC::DataCopyExtParams wRowFull(1, (uint32_t)(nColsAligned * sizeof(float)), 0, 0, 0);
        for (int32_t i = 0; i < kDimOrig; i++) {
            AscendC::DataCopyPad(ub, gmBcT[(int64_t)i * nColsAligned], rRow, pad);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Muls(ub, ub, alpha, nColsAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(gmBcT[(int64_t)i * nColsAligned], ub, wRowFull);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // Cooperative path: zero owned columns of gmBc, then copy/transpose B data into them
    __aicore__ inline void BuildPaddedBCoopCopy(__gm__ float* gmB, __gm__ float* gmBc)
    {
        int32_t kDim = cfg_->kDim, kDimOrig = cfg_->kDimOrig, nColsAligned = cfg_->nColsAligned;
        AscendC::LocalTensor<float> ub = bufRow_->Get<float>();
        AscendC::GlobalTensor<float> gmBT, gmBcT;
        gmBT.SetGlobalBuffer(gmB, (uint32_t)(cfg_->tiling->m * cfg_->tiling->ldb));
        gmBcT.SetGlobalBuffer(gmBc, (uint32_t)(kDim * nColsAligned));

        int32_t origStart = (cfg_->solveColStart < cfg_->nColsOrig) ? cfg_->solveColStart : cfg_->nColsOrig;
        int32_t origEnd = (cfg_->solveColEnd < cfg_->nColsOrig) ? cfg_->solveColEnd : cfg_->nColsOrig;
        int32_t origWidth = origEnd - origStart;
        int32_t myWidth = cfg_->solveColEnd - cfg_->solveColStart;

        int32_t zeroWidth = CEIL_ALIGN(myWidth, FLOAT_ALIGN);
        AscendC::Duplicate(ub, 0.0f, zeroWidth);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < kDim; i++) {
            AscendC::DataCopyExtParams zRow(1, (uint32_t)(myWidth * sizeof(float)), 0, 0, 0);
            AscendC::DataCopyPad(gmBcT[(int64_t)i * nColsAligned + cfg_->solveColStart], ub, zRow);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        if (origWidth <= 0) return;
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        if (cfg_->right) {
            tr_->Run(gmB + (int64_t)origStart * cfg_->tiling->ldb, origWidth, kDimOrig,
                     cfg_->tiling->ldb, gmBc + origStart, nColsAligned);
            AscendC::PipeBarrier<PIPE_ALL>();
        } else {
            for (int32_t i = 0; i < kDimOrig; i++) {
                AscendC::DataCopyExtParams rRow(1, (uint32_t)(origWidth * sizeof(float)), 0, 0, 0);
                AscendC::DataCopyPad(ub, gmBT[i * cfg_->tiling->ldb + origStart], rRow, pad);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams wRow(1, (uint32_t)(origWidth * sizeof(float)), 0, 0, 0);
                AscendC::DataCopyPad(gmBcT[(int64_t)i * nColsAligned + origStart], ub, wRow);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    // Cooperative path: apply alpha scaling to the owned columns of gmBc
    __aicore__ inline void BuildPaddedBCoopScale(__gm__ float* gmBc)
    {
        int32_t kDim = cfg_->kDim, kDimOrig = cfg_->kDimOrig, nColsAligned = cfg_->nColsAligned;
        float alpha = cfg_->tiling->alphaReal;
        int32_t origStart = (cfg_->solveColStart < cfg_->nColsOrig) ? cfg_->solveColStart : cfg_->nColsOrig;
        int32_t origEnd = (cfg_->solveColEnd < cfg_->nColsOrig) ? cfg_->solveColEnd : cfg_->nColsOrig;
        int32_t origWidth = origEnd - origStart;
        if (alpha == 1.0f || origWidth <= 0) return;

        AscendC::LocalTensor<float> ub = bufRow_->Get<float>();
        AscendC::GlobalTensor<float> gmBcT;
        gmBcT.SetGlobalBuffer(gmBc, (uint32_t)(kDim * nColsAligned));
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        for (int32_t i = 0; i < kDimOrig; i++) {
            for (int32_t cs = cfg_->solveColStart; cs < cfg_->solveColEnd; cs += cfg_->colTile) {
                int32_t ct = (cfg_->colTile < cfg_->solveColEnd - cs) ? cfg_->colTile : (cfg_->solveColEnd - cs);
                int32_t ctAligned = CEIL_ALIGN(ct, FLOAT_ALIGN);
                AscendC::DataCopyExtParams rRow(1, (uint32_t)(ct * sizeof(float)), 0, 0, 0);
                AscendC::DataCopyPad(ub, gmBcT[(int64_t)i * nColsAligned + cs], rRow, pad);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Muls(ub, ub, alpha, ctAligned);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams wRow(1, (uint32_t)(ct * sizeof(float)), 0, 0, 0);
                AscendC::DataCopyPad(gmBcT[(int64_t)i * nColsAligned + cs], ub, wRow);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    // Solved X' (in gmBc) -> original B. right: X = X'^T. Dispatches to fast or coop path.
    __aicore__ inline void WriteBackPaddedB(__gm__ float* gmBc, __gm__ float* gmB)
    {
        if (!cfg_->coopAiv) {
            WriteBackPaddedBFast(gmBc, gmB);
            return;
        }
        WriteBackPaddedBCoop(gmBc, gmB);
    }

    // Non-cooperative writeback: transpose or row-copy from gmBc to gmB
    __aicore__ inline void WriteBackPaddedBFast(__gm__ float* gmBc, __gm__ float* gmB)
    {
        int32_t kDimOrig = cfg_->kDimOrig, nColsAligned = cfg_->nColsAligned, nColsOrig = cfg_->nColsOrig;
        if (cfg_->right) {
            tr_->Run(gmBc, kDimOrig, nColsOrig, nColsAligned, gmB, cfg_->tiling->ldb);
            return;
        }
        AscendC::LocalTensor<float> ub = bufRow_->Get<float>();
        AscendC::GlobalTensor<float> gmBcT, gmBT;
        gmBcT.SetGlobalBuffer(gmBc, (uint32_t)(cfg_->kDim * nColsAligned));
        gmBT.SetGlobalBuffer(gmB, (uint32_t)(cfg_->tiling->m * cfg_->tiling->ldb));
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyExtParams rRow(1, (uint32_t)(nColsOrig * sizeof(float)), 0, 0, 0);
        for (int32_t i = 0; i < kDimOrig; i++) {
            AscendC::DataCopyPad(ub, gmBcT[(int64_t)i * nColsAligned], rRow, pad);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams wRow(1, (uint32_t)(nColsOrig * sizeof(float)), 0, 0, 0);
            AscendC::DataCopyPad(gmBT[i * cfg_->tiling->ldb], ub, wRow);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // Write back a range of rows [rowStart, rowStart+rowCount) from gmBc to gmB
    __aicore__ inline void WriteBackPaddedBRows(__gm__ float* gmBc, __gm__ float* gmB,
                                                 int32_t rowStart, int32_t rowCount)
    {
        if (cfg_->right) return;
        int32_t nColsAligned = cfg_->nColsAligned, nColsOrig = cfg_->nColsOrig;
        int32_t rowEnd = rowStart + rowCount;
        if (rowEnd > cfg_->kDimOrig) rowEnd = cfg_->kDimOrig;
        if (rowStart >= rowEnd) return;

        int32_t colStart = 0, colWidth = nColsOrig;
        if (cfg_->coopAiv) {
            colStart = (cfg_->solveColStart < nColsOrig) ? cfg_->solveColStart : nColsOrig;
            int32_t colEnd = (cfg_->solveColEnd < nColsOrig) ? cfg_->solveColEnd : nColsOrig;
            colWidth = colEnd - colStart;
            if (colWidth <= 0) return;
        }

        AscendC::LocalTensor<float> ub = bufRow_->Get<float>();
        AscendC::GlobalTensor<float> gmBcT, gmBT;
        gmBcT.SetGlobalBuffer(gmBc, (uint32_t)(cfg_->kDim * nColsAligned));
        gmBT.SetGlobalBuffer(gmB, (uint32_t)(cfg_->tiling->m * cfg_->tiling->ldb));
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        for (int32_t i = rowStart; i < rowEnd; i++) {
            AscendC::DataCopyExtParams rRow(1, (uint32_t)(colWidth * sizeof(float)), 0, 0, 0);
            AscendC::DataCopyPad(ub, gmBcT[(int64_t)i * nColsAligned + colStart], rRow, pad);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams wRow(1, (uint32_t)(colWidth * sizeof(float)), 0, 0, 0);
            AscendC::DataCopyPad(gmBT[i * cfg_->tiling->ldb + colStart], ub, wRow);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // Cooperative writeback: transpose or row-copy only the owned column range
    __aicore__ inline void WriteBackPaddedBCoop(__gm__ float* gmBc, __gm__ float* gmB)
    {
        int32_t kDimOrig = cfg_->kDimOrig, nColsAligned = cfg_->nColsAligned;
        int32_t origStart = (cfg_->solveColStart < cfg_->nColsOrig) ? cfg_->solveColStart : cfg_->nColsOrig;
        int32_t origEnd = (cfg_->solveColEnd < cfg_->nColsOrig) ? cfg_->solveColEnd : cfg_->nColsOrig;
        int32_t width = origEnd - origStart;
        if (width <= 0) return;

        if (cfg_->right) {
            tr_->Run(gmBc + origStart, kDimOrig, width, nColsAligned,
                     gmB + (int64_t)origStart * cfg_->tiling->ldb, cfg_->tiling->ldb);
            return;
        }
        AscendC::LocalTensor<float> ub = bufRow_->Get<float>();
        AscendC::GlobalTensor<float> gmBcT, gmBT;
        gmBcT.SetGlobalBuffer(gmBc, (uint32_t)(cfg_->kDim * nColsAligned));
        gmBT.SetGlobalBuffer(gmB, (uint32_t)(cfg_->tiling->m * cfg_->tiling->ldb));
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        for (int32_t i = 0; i < kDimOrig; i++) {
            AscendC::DataCopyExtParams rRow(1, (uint32_t)(width * sizeof(float)), 0, 0, 0);
            AscendC::DataCopyPad(ub, gmBcT[(int64_t)i * nColsAligned + origStart], rRow, pad);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams wRow(1, (uint32_t)(width * sizeof(float)), 0, 0, 0);
            AscendC::DataCopyPad(gmBT[i * cfg_->tiling->ldb + origStart], ub, wRow);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    const TrsmAivCfg* cfg_;
    TrsmTranspose* tr_;
    BufVecCalc* bufRow_;
    BufVecCalc* bufBatch_;
    BufVecCalc* bufBatch2_;
};
