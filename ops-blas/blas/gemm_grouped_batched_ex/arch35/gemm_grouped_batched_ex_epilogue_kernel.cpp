/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "gemm_grouped_batched_ex_tiling_data.h"

using namespace AscendC;

namespace {

__aicore__ inline void CopyEpilogueGroup(
    GroupedGemmGroupData *dst, const __gm__ GroupedGemmGroupData *src)
{
    static_assert(sizeof(GroupedGemmGroupData) % sizeof(uint64_t) == 0,
        "GroupedGemmGroupData must be copied in 64-bit chunks");
    uint64_t *dst64 = reinterpret_cast<uint64_t *>(dst);
    auto src64 = reinterpret_cast<const __gm__ uint64_t *>(src);
    for (uint32_t i = 0; i < sizeof(GroupedGemmGroupData) / sizeof(uint64_t); ++i) {
        dst64[i] = src64[i];
    }
}

__aicore__ inline bool FindEpilogueGroup(__gm__ uint8_t *tilingGm, uint32_t taskId,
    GroupedGemmGroupData &group, uint32_t &localTask)
{
    const auto *header = reinterpret_cast<const __gm__ GroupedGemmTilingHeader *>(tilingGm);
    const auto *groups = reinterpret_cast<const __gm__ GroupedGemmGroupData *>(
        tilingGm + sizeof(GroupedGemmTilingHeader));
    uint32_t low = 0;
    uint32_t high = header->groupCount;
    while (low < high) {
        uint32_t mid = low + (high - low) / 2;
        if (groups[mid].epilogueTaskStart <= taskId) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (low == 0) { return false; }
    uint32_t index = low - 1;
    uint32_t start = groups[index].epilogueTaskStart;
    uint32_t count = groups[index].epilogueTaskCount;
    if (taskId - start < count) {
        CopyEpilogueGroup(&group, &groups[index]);
        localTask = taskId - start;
        return true;
    }
    return false;
}

struct EpilogueTaskInfo {
    uint32_t localProblem;
    uint32_t column;
    uint32_t rowStart;
    uint32_t count;
    uint32_t alignedCount;
    uint32_t problemIdx;
    uint64_t cOffset;
};

__aicore__ inline bool DecodeEpilogueTask(const GroupedGemmTilingHeader *header,
    const GroupedGemmGroupData &group, uint32_t localTask, EpilogueTaskInfo &info)
{
    if (header->epilogueTile == 0) { return false; }
    uint32_t tilesPerColumn = (static_cast<uint32_t>(group.originalM) + header->epilogueTile - 1) /
                              header->epilogueTile;
    uint32_t tasksPerProblem = static_cast<uint32_t>(group.originalN) * tilesPerColumn;
    if (tasksPerProblem == 0 || tilesPerColumn == 0) { return false; }
    info.localProblem = localTask / tasksPerProblem;
    uint32_t matrixTask = localTask - info.localProblem * tasksPerProblem;
    info.column = matrixTask / tilesPerColumn;
    info.rowStart = (matrixTask - info.column * tilesPerColumn) * header->epilogueTile;
    info.count = static_cast<uint32_t>(group.originalM) - info.rowStart;
    if (info.count > header->epilogueTile) { info.count = header->epilogueTile; }
    info.alignedCount = (info.count + 15u) & ~15u;
    info.problemIdx = group.batchStart + info.localProblem;
    info.cOffset = static_cast<uint64_t>(info.column) * group.originalLdc + info.rowStart;
    return true;
}

__aicore__ inline void LoadGemmResult(__gm__ uint8_t *workspace,
    const GroupedGemmGroupData &group, const EpilogueTaskInfo &info, LocalTensor<float> resultLocal)
{
    if (group.hasGemm == 0) {
        Duplicate(resultLocal, 0.0f, info.alignedCount);
        PipeBarrier<PIPE_V>();
        return;
    }
    GlobalTensor<float> workspaceGM;
    workspaceGM.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    uint64_t tempOffset = group.workspaceOffset +
        static_cast<uint64_t>(info.localProblem) * group.originalM * group.originalN +
        static_cast<uint64_t>(info.column) * group.originalM + info.rowStart;
    DataCopyPad(resultLocal, workspaceGM[tempOffset],
        DataCopyExtParams{1, static_cast<uint32_t>(info.count * sizeof(float)), 0, 0, 0},
        DataCopyPadExtParams<float>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    if (group.alpha != 1.0f) {
        Muls(resultLocal, resultLocal, group.alpha, info.count);
        PipeBarrier<PIPE_V>();
    }
}

template <typename C_TYPE>
__aicore__ inline void AccumulateOutput(GlobalTensor<C_TYPE> &cGM,
    const GroupedGemmGroupData &group, const EpilogueTaskInfo &info,
    LocalTensor<float> resultLocal, LocalTensor<float> cFloatLocal, LocalTensor<C_TYPE> cLocal)
{
    if (group.beta == 0.0f) { return; }
    DataCopyPad(cLocal, cGM[info.cOffset],
        DataCopyExtParams{1, static_cast<uint32_t>(info.count * sizeof(C_TYPE)), 0, 0, 0},
        DataCopyPadExtParams<C_TYPE>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(cFloatLocal, cLocal, RoundMode::CAST_NONE, info.alignedCount);
    PipeBarrier<PIPE_V>();
    if (group.beta != 1.0f) {
        Muls(cFloatLocal, cFloatLocal, group.beta, info.count);
        PipeBarrier<PIPE_V>();
    }
    Add(resultLocal, resultLocal, cFloatLocal, info.count);
    PipeBarrier<PIPE_V>();
}

template <typename C_TYPE>
__aicore__ inline void RunEpilogueTask(__gm__ uint8_t *carray, __gm__ uint8_t *workspace,
    const GroupedGemmTilingHeader *header, const GroupedGemmGroupData &group, uint32_t localTask,
    LocalTensor<float> resultLocal, LocalTensor<float> cFloatLocal,
    LocalTensor<C_TYPE> cLocal, LocalTensor<C_TYPE> outLocal)
{
    EpilogueTaskInfo info{};
    if (!DecodeEpilogueTask(header, group, localTask, info)) { return; }
    __gm__ uint64_t *cPtrArray = reinterpret_cast<__gm__ uint64_t *>(carray);
    GlobalTensor<C_TYPE> cGM;
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ C_TYPE *>(cPtrArray[info.problemIdx]));
    LoadGemmResult(workspace, group, info, resultLocal);
    AccumulateOutput(cGM, group, info, resultLocal, cFloatLocal, cLocal);
    constexpr RoundMode roundMode = IsSameType<C_TYPE, bfloat16_t>::value ?
        RoundMode::CAST_RINT : RoundMode::CAST_ROUND;
    Cast(outLocal, resultLocal, roundMode, info.alignedCount);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(cGM[info.cOffset], outLocal,
        DataCopyExtParams{1, static_cast<uint32_t>(info.count * sizeof(C_TYPE)), 0, 0, 0});
}

template <typename C_TYPE>
__aicore__ inline void RunGroupedEpilogue(
    __gm__ uint8_t *carray, __gm__ uint8_t *workspace, __gm__ uint8_t *tilingGm)
{
    TPipe pipe;
    TBuf<QuePosition::VECCALC> resultBuf;
    TBuf<QuePosition::VECCALC> cFloatBuf;
    TBuf<QuePosition::VECCALC> cBuf;
    TBuf<QuePosition::VECCALC> outBuf;
    pipe.InitBuffer(resultBuf, 256 * sizeof(float));
    pipe.InitBuffer(cFloatBuf, 256 * sizeof(float));
    pipe.InitBuffer(cBuf, 256 * sizeof(C_TYPE));
    pipe.InitBuffer(outBuf, 256 * sizeof(C_TYPE));
    LocalTensor<float> resultLocal = resultBuf.Get<float>();
    LocalTensor<float> cFloatLocal = cFloatBuf.Get<float>();
    LocalTensor<C_TYPE> cLocal = cBuf.Get<C_TYPE>();
    LocalTensor<C_TYPE> outLocal = outBuf.Get<C_TYPE>();

    const auto *headerGm = reinterpret_cast<const __gm__ GroupedGemmTilingHeader *>(tilingGm);
    GroupedGemmTilingHeader header{};
    header.groupCount = headerGm->groupCount;
    header.totalEpilogueTasks = headerGm->totalEpilogueTasks;
    header.epilogueTile = headerGm->epilogueTile;
    uint32_t blockIdx = GetBlockIdx();
    uint32_t blockCount = GetBlockNum();
    for (uint32_t taskId = blockIdx; taskId < header.totalEpilogueTasks; taskId += blockCount) {
        GroupedGemmGroupData group{};
        uint32_t localTask = 0;
        if (FindEpilogueGroup(tilingGm, taskId, group, localTask)) {
            RunEpilogueTask<C_TYPE>(carray, workspace, &header, group, localTask,
                resultLocal, cFloatLocal, cLocal, outLocal);
        }
    }
}

} // namespace

__global__ __aicore__ void gemm_grouped_batched_ex_epilogue_fp16(
    __gm__ uint8_t *carray, __gm__ uint8_t *workspace, __gm__ uint8_t *tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunGroupedEpilogue<half>(carray, workspace, tilingGm);
}

__global__ __aicore__ void gemm_grouped_batched_ex_epilogue_bf16(
    __gm__ uint8_t *carray, __gm__ uint8_t *workspace, __gm__ uint8_t *tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RunGroupedEpilogue<bfloat16_t>(carray, workspace, tilingGm);
}

void gemm_grouped_batched_ex_epilogue_kernel_do(uint32_t numBlocks, void *stream,
    uint8_t *carray, uint8_t *workspace, uint8_t *tilingGm, int dtypeCase)
{
    if (dtypeCase == GROUPED_GEMM_BF16) {
        gemm_grouped_batched_ex_epilogue_bf16<<<numBlocks, nullptr, stream>>>(carray, workspace, tilingGm);
    } else {
        gemm_grouped_batched_ex_epilogue_fp16<<<numBlocks, nullptr, stream>>>(carray, workspace, tilingGm);
    }
}
