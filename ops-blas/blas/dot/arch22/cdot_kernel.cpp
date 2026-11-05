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
#include "common/helper/kernel_utils.h"
#include "common/iterator/iterator.h"
#include "common/compute/simd.h"
#ifdef __CCE_KT_TEST__
#undef __aicore__
#define __aicore__
#else
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#endif

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 8 - len % 8;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    gm_to_ub_align<ArchType::ASCEND_V220, float>(
        dst, src,
        0,  // sid
        nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_ub2gm(
    AscendC::GlobalTensor<float> dst,
    AscendC::LocalTensor<float> src,
    uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    ub_to_gm_align<ArchType::ASCEND_V220, float>(
        dst, src,
        0,  // sid
        nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void cdot_single_iteration_aiv(
    AscendC::GlobalTensor<float> gm_x, AscendC::GlobalTensor<float> gm_y, AscendC::GlobalTensor<float> gm_result,
    AscendC::LocalTensor<float> ub_x, AscendC::LocalTensor<float> ub_y, AscendC::LocalTensor<float> ub_x_real,
    AscendC::LocalTensor<float> ub_x_imag, AscendC::LocalTensor<float> ub_y_real, AscendC::LocalTensor<float> ub_y_imag,
    AscendC::LocalTensor<float> ub_acc_real, AscendC::LocalTensor<float> ub_acc_imag, uint32_t len, uint32_t event_id,
    uint32_t conj)
{
    uint32_t repeatTime = (len + 63) / 64;         // 3
    uint32_t computeRepeat = (len / 2 + 63) / 64;  // 2

    uint32_t realOffset = 0;
    uint32_t imagOffset = computeRepeat * 64;

    AscendC::Duplicate<float, false>(ub_x, (float)0.0, 64, computeRepeat * 2, 1, 8);
    AscendC::Duplicate<float, false>(ub_y, (float)0.0, 64, computeRepeat * 2, 1, 8);

    SET_FLAG(V, MTE2, event_id);
    WAIT_FLAG(V, MTE2, event_id);

    copy_vec_gm2ub(ub_x, gm_x, len);
    copy_vec_gm2ub(ub_y, gm_y, len);

    SET_FLAG(MTE2, V, event_id);
    WAIT_FLAG(MTE2, V, event_id);

    AscendC::Duplicate<float, false>(ub_x_real, (float)0.0, 64, computeRepeat, 1, 8);
    AscendC::Duplicate<float, false>(ub_x_imag, (float)0.0, 64, computeRepeat, 1, 8);
    AscendC::Duplicate<float, false>(ub_y_real, (float)0.0, 64, computeRepeat, 1, 8);
    AscendC::Duplicate<float, false>(ub_y_imag, (float)0.0, 64, computeRepeat, 1, 8);

    PIPE_BARRIER(V);

    uint32_t mask = 0;
    uint64_t rsvdCnt = 0;
    AscendC::GatherMask<float>(ub_x_real, ub_x, 1, false, mask,
                               {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);

    AscendC::GatherMask<float>(ub_x_imag, ub_x, 2, false, mask,
                               {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);

    AscendC::GatherMask<float>(ub_y_real, ub_y, 1, false, mask, 
                               {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);

    AscendC::GatherMask<float>(ub_y_imag, ub_y, 2, false, mask, 
                               {1, static_cast<uint16_t>(repeatTime), 8, 8}, rsvdCnt);

    PIPE_BARRIER(V);

    if (conj) {
        muls_v<ArchType::ASCEND_V220, float>(ub_x_imag, ub_x_imag, -1.0f, computeRepeat, 1, 1, 8, 8);
        PIPE_BARRIER(V);
    }

    uint64_t mul_mask = 64;
    AscendC::Mul(ub_x, ub_x_real, ub_y_real, mul_mask, computeRepeat, {1, 1, 1, 8, 8, 8});  // x.R * y.R
    AscendC::Mul(ub_y, ub_x_real, ub_y_imag, mul_mask, computeRepeat, {1, 1, 1, 8, 8, 8});  // x.R * y.I
    PIPE_BARRIER(V);
    mul_v<ArchType::ASCEND_V220, float>(ub_x_real, ub_x_imag, ub_y_imag, computeRepeat, 1, 1, 1, 8, 8, 8);  // x.I * y.I
    mul_v<ArchType::ASCEND_V220, float>(ub_y_real, ub_x_imag, ub_y_real, computeRepeat, 1, 1, 1, 8, 8, 8);  // x.I * y.R

    PIPE_BARRIER(V);
    // x.R * y.R - x.I * y.I
    sub_v<ArchType::ASCEND_V220, float>(ub_x_real, ub_x, ub_x_real, computeRepeat, 1, 1, 1, 8, 8, 8);
    // x.I * y.R + x.R * y.I
    add_v<ArchType::ASCEND_V220, float>(ub_y_real, ub_y_real, ub_y, computeRepeat, 1, 1, 1, 8, 8, 8);

    AscendC::Duplicate<float, false>(ub_x_imag, (float)0.0, 64, 1, 1, 8);
    AscendC::Duplicate<float, false>(ub_y_imag, (float)0.0, 64, 1, 1, 8);

    PIPE_BARRIER(V);
    cadd_v<ArchType::ASCEND_V220, float>(ub_x_imag, ub_x_real, computeRepeat, 1, 1, 8);
    cadd_v<ArchType::ASCEND_V220, float>(ub_y_imag, ub_y_real, computeRepeat, 1, 1, 8);

    if (computeRepeat > 0) {
        PIPE_BARRIER(V);
        cadd_v<ArchType::ASCEND_V220, float>(ub_x_imag, ub_x_imag, 1, 1, 1, 8);
        cadd_v<ArchType::ASCEND_V220, float>(ub_y_imag, ub_y_imag, 1, 1, 1, 8);
    }

    PIPE_BARRIER(V);

    add_v<ArchType::ASCEND_V220, float>(ub_acc_real, ub_acc_real, ub_x_imag, 1, 1, 1, 1, 8, 8, 8);
    add_v<ArchType::ASCEND_V220, float>(ub_acc_imag, ub_acc_imag, ub_y_imag, 1, 1, 1, 1, 8, 8, 8);
}

__aicore__ __inline__ __attribute__((always_inline)) void cdot_process_aiv(
    AscendC::GlobalTensor<float> gm_x, AscendC::GlobalTensor<float> gm_y, AscendC::GlobalTensor<float> gm_result,
    AscendC::GlobalTensor<float> gm_workpsace, uint32_t offset, uint32_t compute_num, uint32_t conj,
    uint32_t vec_core_num)
{
    auto core_idx = AscendC::GetBlockIdx();

    uint32_t maxDataCount = 23 * 1024 / 4;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;

    AscendC::LocalTensor<float> ub_x_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> ub_x_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(23 * 1024);
    AscendC::LocalTensor<float> ub_y_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(23 * 2 * 1024);
    AscendC::LocalTensor<float> ub_y_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(23 * 3 * 1024);
    AscendC::LocalTensor<float> ub_x_real_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(23 * 4 * 1024);
    AscendC::LocalTensor<float> ub_x_real_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>((92 + 11.5 * 1) * 1024);
    AscendC::LocalTensor<float> ub_x_imag_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>((92 + 11.5 * 2) * 1024);
    AscendC::LocalTensor<float> ub_x_imag_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>((92 + 11.5 * 3) * 1024);
    AscendC::LocalTensor<float> ub_y_real_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>((92 + 11.5 * 4) * 1024);
    AscendC::LocalTensor<float> ub_y_real_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>((92 + 11.5 * 5) * 1024);
    AscendC::LocalTensor<float> ub_y_imag_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>((92 + 11.5 * 6) * 1024);
    AscendC::LocalTensor<float> ub_y_imag_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>((92 + 11.5 * 7) * 1024);        // 172.5kb
    AscendC::LocalTensor<float> ub_acc_real_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>((92 + 11.5 * 8) * 1024);      //184kb
    AscendC::LocalTensor<float> ub_acc_imag_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>((92 + 11.5 * 8 + 1) * 1024);  //184kb
    AscendC::LocalTensor<float> ub_acc_real_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>((92 + 11.5 * 8 + 2) * 1024);  //185kb
    AscendC::LocalTensor<float> ub_acc_imag_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>((92 + 11.5 * 8 + 3) * 1024);  //185kb

    AscendC::Duplicate<float, false>(ub_acc_real_ping, (float)0.0, 64, 1, 1, 8);
    AscendC::Duplicate<float, false>(ub_acc_imag_ping, (float)0.0, 64, 1, 1, 8);
    AscendC::Duplicate<float, false>(ub_acc_real_pong, (float)0.0, 64, 1, 1, 8);
    AscendC::Duplicate<float, false>(ub_acc_imag_pong, (float)0.0, 64, 1, 1, 8);

    PIPE_BARRIER(V);

    uint32_t repeatTimes = compute_num / maxDataCount;
    uint32_t remainNum = compute_num % maxDataCount;

    uint32_t ping_flag = 1;

    uint32_t curr_offset = offset;

    if (repeatTimes > 0) {
        SET_FLAG(V, MTE2, EVENT_ID0);
        SET_FLAG(V, MTE2, EVENT_ID1);

        for (uint32_t i = 0; i < repeatTimes; i++) {
            AscendC::LocalTensor<float> ub_x = ping_flag ? ub_x_ping : ub_x_pong;
            AscendC::LocalTensor<float> ub_y = ping_flag ? ub_y_ping : ub_y_pong;
            AscendC::LocalTensor<float> ub_x_real = ping_flag ? ub_x_real_ping : ub_x_real_pong;
            AscendC::LocalTensor<float> ub_x_imag = ping_flag ? ub_x_imag_ping : ub_x_imag_pong;
            AscendC::LocalTensor<float> ub_y_real = ping_flag ? ub_y_real_ping : ub_y_real_pong;
            AscendC::LocalTensor<float> ub_y_imag = ping_flag ? ub_y_imag_ping : ub_y_imag_pong;
            AscendC::LocalTensor<float> ub_acc_real = ping_flag ? ub_acc_real_ping : ub_acc_real_pong;
            AscendC::LocalTensor<float> ub_acc_imag = ping_flag ? ub_acc_imag_ping : ub_acc_imag_pong;

            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            WAIT_FLAG(V, MTE2, event_id);

            cdot_single_iteration_aiv(gm_x[curr_offset], gm_y [curr_offset], gm_result, ub_x, ub_y, ub_x_real,
                                      ub_x_imag, ub_y_real, ub_y_imag, ub_acc_real, ub_acc_imag, maxDataCount, event_id,
                                      conj);

            curr_offset += maxDataCount;

            SET_FLAG(V, MTE2, event_id);
            ping_flag = 1 - ping_flag;
        }
        WAIT_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID1);
    }

    if (remainNum > 0) {
        AscendC::LocalTensor<float> ub_x = ping_flag ? ub_x_ping : ub_x_pong;
        AscendC::LocalTensor<float> ub_y = ping_flag ? ub_y_ping : ub_y_pong;
        AscendC::LocalTensor<float> ub_x_real = ping_flag ? ub_x_real_ping : ub_x_real_pong;
        AscendC::LocalTensor<float> ub_x_imag = ping_flag ? ub_x_imag_ping : ub_x_imag_pong;
        AscendC::LocalTensor<float> ub_y_real = ping_flag ? ub_y_real_ping : ub_y_real_pong;
        AscendC::LocalTensor<float> ub_y_imag = ping_flag ? ub_y_imag_ping : ub_y_imag_pong;
        AscendC::LocalTensor<float> ub_acc_real = ping_flag ? ub_acc_real_ping : ub_acc_real_pong;
        AscendC::LocalTensor<float> ub_acc_imag = ping_flag ? ub_acc_imag_ping : ub_acc_imag_pong;

        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

        cdot_single_iteration_aiv(gm_x[curr_offset], gm_y[curr_offset], gm_result, ub_x, ub_y, ub_x_real, ub_x_imag,
                                  ub_y_real, ub_y_imag, ub_acc_real, ub_acc_imag, remainNum, event_id, conj);
    }
    PIPE_BARRIER(ALL);

    if (repeatTimes > 0) {
        add_v<ArchType::ASCEND_V220, float>(ub_acc_real_ping, ub_acc_real_ping, ub_acc_real_pong, 1, 1, 1, 1, 8, 8, 8);
        add_v<ArchType::ASCEND_V220, float>(ub_acc_imag_ping, ub_acc_imag_ping, ub_acc_imag_pong, 1, 1, 1, 1, 8, 8, 8);
        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);
    }

    uint32_t imag_gm_offset = 40;
    copy_vec_ub2gm(gm_workpsace[core_idx], ub_acc_real_ping, 1);
    copy_vec_ub2gm(gm_workpsace[imag_gm_offset + core_idx], ub_acc_imag_ping, 1);
    PIPE_BARRIER(ALL);

    FftsCrossCoreSync<PIPE_MTE3, 0>(0);
    WaitFlagDev(0);

    if (core_idx == 0) {
        // sum all results
        AscendC::LocalTensor<float> ub_sum_real = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
        AscendC::LocalTensor<float> ub_sum_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>(64 * 4);
        AscendC::LocalTensor<float> ub_result_real = buf.GetBuffer<BufferType::ASCEND_UB, float>(64 * 4 * 2);
        AscendC::LocalTensor<float> ub_result_imag = buf.GetBuffer<BufferType::ASCEND_UB, float>(64 * 4 * 3);
        AscendC::Duplicate<float, false>(ub_sum_real, (float)0.0, 64, 4, 1, 8);

        SET_FLAG(V, MTE2, EVENT_ID0);
        WAIT_FLAG(V, MTE2, EVENT_ID0);

        copy_vec_gm2ub(ub_result_real, gm_workpsace, vec_core_num);
        copy_vec_gm2ub(ub_result_imag, gm_workpsace[imag_gm_offset], vec_core_num);

        SET_FLAG(MTE2, V, EVENT_ID0);
        WAIT_FLAG(MTE2, V, EVENT_ID0);

        cadd_v<ArchType::ASCEND_V220, float>(ub_sum_real, ub_result_real, 1, 1, 1, 8);
        cadd_v<ArchType::ASCEND_V220, float>(ub_sum_imag, ub_result_imag, 1, 1, 1, 8);

        SET_FLAG(V, MTE3, EVENT_ID0);
        WAIT_FLAG(V, MTE3, EVENT_ID0);

        copy_vec_ub2gm(gm_result, ub_sum_real, 1);
        copy_vec_ub2gm(gm_result[1], ub_sum_imag, 1);
    }

    PIPE_BARRIER(ALL);
}

__global__ __aicore__ __vector__ void cdot(GM_ADDR x, GM_ADDR y, GM_ADDR result,
                                GM_ADDR workSpace, GM_ADDR tilingGm)
{
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    SetAtomicnone();

    auto core_idx = AscendC::GetBlockIdx();
    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);

    uint32_t n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf));  // num of float elements
    uint32_t coreNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 4));
    uint32_t isConj = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 8));
    uint32_t offset = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 12 + 4 * core_idx));
    uint32_t cal_num = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 12 + 40 * 4 + 4 * core_idx));

    AscendC::GlobalTensor<float> x_tensor;
    AscendC::GlobalTensor<float> y_tensor;
    AscendC::GlobalTensor<float> result_tensor;
    AscendC::GlobalTensor<float> workspace_tensor;
    x_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x));
    y_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y));
    result_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(result));
    workspace_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workSpace));

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> ub_clear = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::Duplicate<float, false>(ub_clear, (float)0.0, 64, 2, 1, 8);

    SET_FLAG(V, MTE3, EVENT_ID0);
    WAIT_FLAG(V, MTE3, EVENT_ID0);

    // set workspace to 0
    copy_vec_ub2gm(workspace_tensor, ub_clear, coreNum * 2);

    FftsCrossCoreSync<PIPE_MTE3, 0>(0);
    WaitFlagDev(0);

    cdot_process_aiv(x_tensor, y_tensor, result_tensor, workspace_tensor, offset, cal_num, isConj, coreNum);
}

void cdot_kernel_do(GM_ADDR x, GM_ADDR y, GM_ADDR result, GM_ADDR workSpace, GM_ADDR tilingGm,
                    uint32_t numBlocks, void *stream)
{
    cdot<<<numBlocks, nullptr, stream>>>(x, y, result, workSpace, tilingGm);
}
