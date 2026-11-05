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
 * \file trsm_mix_common.h
 * \brief 由 StrsmBatched / CtrsmBatched 两个 MIX 内核共享、且与具体 tiling 数据类型无关的
 *        内核内常量与工具函数。仅收纳逐字节等价、无成员状态依赖的部分：
 *        - 跨核同步 flag 常量与分组常量
 *        - CopyCubeTiling（GM -> TCubeTiling 拷贝）
 *        - XnegSlot（Xneg 双缓冲槽位索引）
 *        - GroupBounds（分组调度的组边界计算；nb/kDim/forward 由 AIC 类成员传入）
 *        - TransposeTile16 / TransposeTileBlock（分块转置的无状态子过程，AIV 侧共用）
 *        算子各自差异的部分（如 NeedTransA / IsConjTransA）仍保留在各自 common.h。
 */

#pragma once

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace {
constexpr int32_t FLAG_TRSV = 6;   // AIV -> AIC : panel TRSV done
constexpr int32_t FLAG_GEMM = 7;   // AIC -> AIV : GEMM done
constexpr int32_t FLAG_DUAL_AIV = 8;  // AIV <-> AIV (mode1) : dual-AIV workspace barrier
constexpr int32_t LIM_GROUP = 16;  // panels per group for big-GEMM accumulation strategy

__aicore__ inline void CopyCubeTiling(TCubeTiling* dst, GM_ADDR src)
{
    uint32_t* p = reinterpret_cast<uint32_t*>(dst);
    auto s = reinterpret_cast<__gm__ uint32_t*>(src);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, p++) {
        *p = *(s + i);
    }
}

// Xneg slot index: double-buffered groups of LIM_GROUP slots.
// Forward: ascending slot within group (matches A column order).
// Backward: reversed slot within group so Xneg memory order = ascending physical panel order.
__aicore__ inline int32_t XnegSlot(int32_t idx, int32_t numPanels, bool forward)
{
    int32_t g = idx / LIM_GROUP;
    int32_t groupBase = (g % 2) * LIM_GROUP;
    int32_t idxStart = g * LIM_GROUP;
    int32_t idxEnd = ((g + 1) * LIM_GROUP - 1 < numPanels - 1)
                     ? (g + 1) * LIM_GROUP - 1 : numPanels - 1;
    if (forward) {
        return groupBase + (idx - idxStart);
    } else {
        return groupBase + (idxEnd - idx);
    }
}

// Compute group start/end/size/boundary for grouped panel scheduling.
// nb/kDim/forward are AIC members passed in so this stays free of the op-specific class.
__aicore__ inline void GroupBounds(int32_t grp, int32_t numPanels, int32_t nb, int32_t kDim,
                                   bool forward, int32_t& idxStart, int32_t& idxEnd,
                                   int32_t& groupSize, int32_t& groupEndRow, int32_t& groupBoundaryRow)
{
    idxStart = grp * LIM_GROUP;
    idxEnd = ((grp + 1) * LIM_GROUP - 1 < numPanels - 1)
             ? (grp + 1) * LIM_GROUP - 1 : numPanels - 1;
    groupSize = idxEnd - idxStart + 1;
    if (forward) {
        groupEndRow = (((idxEnd + 2) * nb < kDim) ? (idxEnd + 2) * nb : kDim);
        groupBoundaryRow = 0;
    } else {
        int32_t pLast = numPanels - 1 - idxEnd;
        groupBoundaryRow = ((pLast - 1) * nb > 0) ? (pLast - 1) * nb : 0;
        groupEndRow = kDim;
    }
}

// TransDataTo5HD<float> for one chunk of <=16 source rows.
// Input: ubSrc in [tr, tcA] layout (row-major, stride=tcA).
// Transposes rows [ro..ro+chunkRows-1] -> ubDst in [tcA, 16] layout (stride=16).
__aicore__ inline void TransposeTile16(
    AscendC::LocalTensor<float>& ubSrc, AscendC::LocalTensor<float>& ubDst,
    int32_t tcA, int32_t tr, int32_t ro, int32_t chunkRows)
{
    AscendC::LocalTensor<float> srcList[16];
    AscendC::LocalTensor<float> dstList[16];
    for (int32_t i = 0; i < 16; i++) {
        int32_t row = (ro + i < tr) ? (ro + i) : ro;
        srcList[i] = ubSrc[row * tcA];
    }
    for (int32_t i = 0; i < 8; i++) {
        dstList[i * 2] = ubDst[i * 16];
        dstList[i * 2 + 1] = ubDst[i * 16 + 8];
    }
    uint32_t repeats = (uint32_t)(tcA / 8);
    uint16_t dstRS = (repeats == 1) ? 0 : 16;
    uint16_t srcRS = (repeats == 1) ? 0 : 1;
    AscendC::TransDataTo5HDParams params(false, false, static_cast<uint8_t>(repeats), dstRS, srcRS);
    AscendC::TransDataTo5HD<float>(dstList, srcList, params);
}

// Transpose one tile block in chunks of 16 rows, writing transposed results to GM.
__aicore__ inline void TransposeTileBlock(
    AscendC::LocalTensor<float>& ubSrc, AscendC::LocalTensor<float>& ubDst,
    AscendC::GlobalTensor<float>& gmD, int32_t tcA, int32_t tr, int32_t tc,
    int32_t dstLd, int32_t cb, int32_t rb)
{
    for (int32_t ro = 0; ro < tr; ro += 16) {
        int32_t chunkRows = (16 < tr - ro) ? 16 : (tr - ro);
        TransposeTile16(ubSrc, ubDst, tcA, tr, ro, chunkRows);
        AscendC::PipeBarrier<PIPE_ALL>();

        int32_t srcStride32B = 2 - (int32_t)((chunkRows + 7) / 8);
        AscendC::DataCopyExtParams stP((uint16_t)tc, (uint32_t)(chunkRows * sizeof(float)),
            (int64_t)srcStride32B, (int64_t)((dstLd - chunkRows) * sizeof(float)), 0);
        AscendC::DataCopyPad(gmD[(int64_t)cb * dstLd + rb + ro], ubDst, stP);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
}  // namespace
