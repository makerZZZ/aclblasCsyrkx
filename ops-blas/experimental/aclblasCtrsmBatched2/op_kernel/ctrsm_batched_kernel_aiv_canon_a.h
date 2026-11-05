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
 * \file ctrsm_batched_kernel_aiv_canon_a.h
 * \brief A 矩阵规范化器：把用户 A 转成补零对齐的 AoS 工作区（清零→直拷/转置(+共轭)→补单位对角）。
 *        转置分快速路径与分块路径，均委托 CtrsmConvert 做解交织/交织/分块转置。
 *        与计算路径(forward/right)无关，依赖 needTA/conjA 运行时标志。
 */

#pragma once

#include "ctrsm_batched_kernel_aiv_cfg.h"
#include "ctrsm_batched_kernel_aiv_convert.h"

class CtrsmCanonA {
public:
    __aicore__ inline CtrsmCanonA() {}

    __aicore__ inline void Bind(CtrsmAivCfg* cfg, CtrsmConvert* cvt) { cfg_ = cfg; cvt_ = cvt; }

    // 构建补零后的A矩阵AoS工作区：清零→直接复制（仅 NoTrans 调用）
    __aicore__ inline void BuildPaddedA(__gm__ float* gmA, __gm__ float* gmAc,
                                         __gm__ float* gmTmp1, __gm__ float* gmTmp2,
                                         __gm__ float* gmExtra)
    {
        int32_t wsStride = 2 * cfg_->kDim;
        if (cfg_->padOn) {
            cvt_->ZeroGmRows(gmAc, cfg_->kDim, wsStride, wsStride);
        }
        float signIm = cfg_->conjA ? -1.0f : 1.0f;
        CopyADirectAoS(gmA, gmAc, wsStride, signIm);
        if (cfg_->padOn) {
            PadIdentityRowsAoS(gmAc, wsStride);
        }
    }

    // 将AoS工作空间逐行清零已提取到 CtrsmConvert::ZeroGmRows

    // 直接复制A矩阵AoS到workspace（不转置），可选共轭翻转虚部
    __aicore__ inline void CopyADirectAoS(__gm__ float* gmA, __gm__ float* gmAc,
                                           int32_t wsStride, float signIm)
    {
        int32_t kDimOrig = cfg_->kDimOrig, kDim = cfg_->kDim;
        AscendC::GlobalTensor<float> gA, gAc;
        gA.SetGlobalBuffer(gmA, (uint32_t)((int64_t)kDimOrig * cfg_->td->lda * 2));
        gAc.SetGlobalBuffer(gmAc, (uint32_t)((int64_t)kDim * wsStride));
        int32_t srcRowFloats = kDimOrig * 2;
        int32_t srcRowAligned = CEIL_ALIGN(srcRowFloats, FLOAT_ALIGN);  // = slice UB 步长
        int32_t batchRows = cfg_->srcBufFloats / srcRowAligned;
        if (batchRows > kDimOrig) batchRows = kDimOrig;
        if (batchRows < 1) batchRows = 1;
        uint32_t srcStrideBytes = (uint32_t)((cfg_->td->lda * 2 - srcRowFloats) * (int32_t)sizeof(float));
        if (signIm == 1.0f) {
            CopyDirectPlain(gA, gAc, wsStride, srcRowFloats, srcRowAligned, batchRows, srcStrideBytes);
        } else {
            CopyDirectConj(gA, gAc, wsStride, srcRowFloats, srcRowAligned, batchRows, srcStrideBytes);
        }
    }

    // 纯拷贝路径：批量载入 → 批量写回
    __aicore__ inline void CopyDirectPlain(AscendC::GlobalTensor<float>& gA,
        AscendC::GlobalTensor<float>& gAc, int32_t wsStride, int32_t srcRowFloats,
        int32_t srcRowAligned, int32_t batchRows, uint32_t srcStrideBytes)
    {
        int32_t kDimOrig = cfg_->kDimOrig;
        AscendC::LocalTensor<float> ubTmp = cfg_->bufPanelB_real->Get<float>();
        uint32_t ubGap = (uint32_t)(((srcRowAligned - srcRowFloats) * (int32_t)sizeof(float)) / 32);
        uint32_t gmGap = (uint32_t)((wsStride - srcRowFloats) * (int32_t)sizeof(float));
        for (int32_t bStart = 0; bStart < kDimOrig; bStart += batchRows) {
            int32_t bCount = (batchRows < kDimOrig - bStart) ? batchRows : (kDimOrig - bStart);
            BatchLoadRows(gA, ubTmp, bStart, bCount, srcRowFloats, srcStrideBytes);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams wbP((uint16_t)bCount,
                (uint32_t)(srcRowFloats * sizeof(float)), ubGap, gmGap, 0);
            AscendC::DataCopyPad(gAc[(int64_t)bStart * wsStride], ubTmp, wbP);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // 共轭路径：构建 conjMask → 批量载入 → 各行独立 Mul → 各行独立写回
    __aicore__ inline void CopyDirectConj(AscendC::GlobalTensor<float>& gA,
        AscendC::GlobalTensor<float>& gAc, int32_t wsStride, int32_t srcRowFloats,
        int32_t srcRowAligned, int32_t batchRows, uint32_t srcStrideBytes)
    {
        int32_t kDimOrig = cfg_->kDimOrig;
        AscendC::LocalTensor<float> ubSrc = cfg_->bufPanelB_real->Get<float>();
        AscendC::LocalTensor<float> conjMask = cfg_->bufNeg_imag_neg->Get<float>();
        for (int32_t j = 0; j < FLOAT_ALIGN / 2; j++) {
            conjMask.SetValue(2 * j, 1.0f);
            conjMask.SetValue(2 * j + 1, -1.0f);
        }
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t c = FLOAT_ALIGN; c < srcRowAligned; c += FLOAT_ALIGN) {
            AscendC::DataCopy(conjMask[c], conjMask, FLOAT_ALIGN);
            AscendC::PipeBarrier<PIPE_V>();
        }
        for (int32_t bStart = 0; bStart < kDimOrig; bStart += batchRows) {
            int32_t bCount = (batchRows < kDimOrig - bStart) ? batchRows : (kDimOrig - bStart);
            BatchLoadRows(gA, ubSrc, bStart, bCount, srcRowFloats, srcStrideBytes);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int32_t j = 0; j < bCount; j++) {
                AscendC::Mul(ubSrc[j * srcRowAligned], ubSrc[j * srcRowAligned], conjMask, srcRowAligned);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            uint32_t ubGap = (uint32_t)(((srcRowAligned - srcRowFloats) * (int32_t)sizeof(float)) / 32);
            uint32_t gmGap = (uint32_t)((wsStride - srcRowFloats) * (int32_t)sizeof(float));
            AscendC::DataCopyExtParams wbP((uint16_t)bCount,
                (uint32_t)(srcRowFloats * sizeof(float)), ubGap, gmGap, 0);
            AscendC::DataCopyPad(gAc[(int64_t)bStart * wsStride], ubSrc, wbP);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // 多行批量载入 A 的 AoS 行到 UB（dstStride=0 自动 32B 对齐打包，slice 步长 = srcRowAligned）
    __aicore__ inline void BatchLoadRows(AscendC::GlobalTensor<float>& gA,
        AscendC::LocalTensor<float>& ubDst, int32_t bStart, int32_t bCount,
        int32_t srcRowFloats, uint32_t srcStrideBytes)
    {
        if (bCount > 1) {
            AscendC::DataCopyExtParams rBatch((uint16_t)bCount,
                (uint32_t)(srcRowFloats * sizeof(float)), srcStrideBytes, 0, 0);
            AscendC::DataCopyPad(ubDst, gA[(int64_t)bStart * cfg_->td->lda * 2], rBatch, {false, 0, 0, 0});
        } else {
            AscendC::DataCopyExtParams rRow(1, (uint32_t)(srcRowFloats * sizeof(float)), 0, 0, 0);
            AscendC::DataCopyPad(ubDst, gA[(int64_t)bStart * cfg_->td->lda * 2], rRow, {false, 0, 0, 0});
        }
    }

    // 对补零行填充单位对角线（real=1.0, imag=0.0）
    __aicore__ inline void PadIdentityRowsAoS(__gm__ float* gmAc, int32_t wsStride)
    {
        AscendC::LocalTensor<float> ub = cfg_->bufRow->Get<float>();
        AscendC::GlobalTensor<float> gAc;
        gAc.SetGlobalBuffer(gmAc, (uint32_t)((int64_t)cfg_->kDim * wsStride));
        cvt_->PadDiagonalIdentity(gAc, ub, cfg_->kDimOrig, cfg_->kDim, wsStride);
    }

private:
    CtrsmAivCfg* cfg_;
    CtrsmConvert* cvt_;
};
