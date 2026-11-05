/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "common/memory/mem.h"
#include <type_traits>

#if __DAV_C220_VEC__
__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    gm_to_ub_align<ArchType::ASCEND_V220, float>(
        dst, src,
        0,
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
        0,
        nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void vec_single_core_compute_lower(
    AscendC::LocalTensor<float> ub_x,
    AscendC::LocalTensor<float> ub_y,
    AscendC::LocalTensor<float> ub_out,
    AscendC::GlobalTensor<float> gm_A,
    uint32_t ub_offset, uint32_t gm_offset, uint32_t len, uint32_t event_id)
{
    uint32_t repeat_num = (len + 63) / 64;
    float temp_x = ub_x.GetValue(ub_offset);

    muls_v<ArchType::ASCEND_V220, float>(
        ub_out, ub_y, temp_x,
        repeat_num,
        1,
        1,
        8,
        8
    );

    SET_FLAG(V, MTE3, event_id);
    WAIT_FLAG(V, MTE3, event_id);

    copy_vec_ub2gm(gm_A[gm_offset], ub_out, len);
}

__aicore__ __inline__ __attribute__((always_inline)) void ssyr_lower_aiv(
    uint32_t vec_idx,
    AscendC::GlobalTensor<float> x,
    AscendC::GlobalTensor<float> y,
    AscendC::GlobalTensor<float> A,
    float alpha, uint32_t n, uint32_t core_num)
{
    if (core_num == 0) {
        core_num = 1;
    }

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> ub_x = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> ub_y = buf.GetBuffer<BufferType::ASCEND_UB, float>(47.5 * 1024);
    AscendC::LocalTensor<float> ub_out_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(95 * 1024);
    AscendC::LocalTensor<float> ub_out_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(142.5 * 1024);

    uint32_t max_data_count = 12160;

    uint32_t row_block_num;
    uint32_t row_remain_num;

    uint32_t ping_flag = 1;

    row_block_num = n / max_data_count;
    row_remain_num = n % max_data_count;

    if (row_remain_num > 0)
        row_block_num++;

    uint32_t curr_compute_row_num = max_data_count;

    for (uint32_t row_block_idx = 0; row_block_idx < row_block_num; row_block_idx++) {
        if (row_block_idx == row_block_num - 1 && row_remain_num > 0)
            curr_compute_row_num = row_remain_num;

        uint32_t core_compute_row_num = curr_compute_row_num / core_num;
        uint32_t remain_row_num = curr_compute_row_num % core_num;

        uint32_t use_core_num = core_num;

        if (core_compute_row_num == 0) {
            core_compute_row_num = 1;
            use_core_num = remain_row_num;
            remain_row_num = 0;
        }

        if (vec_idx < use_core_num) {
            for (uint32_t col_block_idx = 0; col_block_idx < row_block_idx + 1; col_block_idx++) {
                uint32_t outOffset = row_block_idx * max_data_count * n + col_block_idx * max_data_count;

                uint32_t x_offset = row_block_idx * max_data_count;

                uint32_t start_row_idx =
                    vec_idx < remain_row_num ?
                        (core_compute_row_num + 1) * vec_idx :
                        (core_compute_row_num + 1) * remain_row_num + (vec_idx - remain_row_num) * core_compute_row_num;

                uint32_t x_copy_num = vec_idx < remain_row_num ? core_compute_row_num + 1 : core_compute_row_num;

                copy_vec_gm2ub(ub_x, x[x_offset + start_row_idx], x_copy_num);

                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
                SET_FLAG(MTE2, V, EVENT_ID2);
                WAIT_FLAG(MTE2, V, EVENT_ID2);

                uint32_t repeat_num = (x_copy_num + 63) / 64;

                muls_v<ArchType::ASCEND_V220, float>(
                    ub_x, ub_x, alpha,
                    repeat_num,
                    1,
                    1,
                    8,
                    8
                );
                PIPE_BARRIER(V);

                uint32_t y_offset = col_block_idx * max_data_count;
                uint32_t y_copy_num = (row_block_idx == col_block_idx) ? start_row_idx + x_copy_num : max_data_count;
                copy_vec_gm2ub(ub_y, y[y_offset], y_copy_num);

                PIPE_BARRIER(ALL);

                SET_FLAG(MTE3, V, EVENT_ID0);
                SET_FLAG(MTE3, V, EVENT_ID2);
                if (row_block_idx == col_block_idx) {
                    for (uint32_t i = 0; i < x_copy_num; i++) {
                        auto ub_out = ping_flag ? ub_out_ping : ub_out_pong;
                        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID2;

                        WAIT_FLAG(MTE3, V, event_id);
                        vec_single_core_compute_lower(ub_x, ub_y, ub_out, A, i, outOffset + (start_row_idx + i) * n,
                                                      start_row_idx + i + 1, event_id);

                        SET_FLAG(MTE3, V, event_id);
                        ping_flag = 1 - ping_flag;
                    }
                } else {
                    for (uint32_t i = 0; i < x_copy_num; i++) {
                        auto ub_out = ping_flag ? ub_out_ping : ub_out_pong;
                        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID2;

                        WAIT_FLAG(MTE3, V, event_id);

                        vec_single_core_compute_lower(ub_x, ub_y, ub_out, A, i, outOffset + (start_row_idx + i) * n,
                                                      y_copy_num, event_id);

                        SET_FLAG(MTE3, V, event_id);
                        ping_flag = 1 - ping_flag;
                    }
                }
                WAIT_FLAG(MTE3, V, EVENT_ID0);
                WAIT_FLAG(MTE3, V, EVENT_ID2);
            }
        }
    }
    PIPE_BARRIER(ALL);
}

__aicore__ __inline__ __attribute__((always_inline)) void vec_single_core_compute_upper(
   AscendC::LocalTensor<float> ub_x,
   AscendC::LocalTensor<float> ub_y,
   AscendC::LocalTensor<float> ub_out,
   AscendC::LocalTensor<float> ub_mask,
   AscendC::GlobalTensor<float> gm_A,
    uint32_t total_n, uint32_t ub_offset, uint32_t gm_offset, uint32_t len, uint32_t event_id)
{
    uint32_t offset = (total_n - len) / 8 * 8;
    uint32_t aligned_len = (total_n - offset + 7) / 8 * 8;
    uint32_t ignore_num = (total_n - len) % 8;
    uint32_t repeat_num = (aligned_len + 63) / 64;

    float temp_x = ub_x.GetValue(ub_offset);

    PIPE_BARRIER(ALL);

    muls_v<ArchType::ASCEND_V220, float>(
        ub_out, ub_y[offset], temp_x,
        repeat_num,
        1,
        1,
        8,
        8
    );

    if (ignore_num > 0) {
        PIPE_BARRIER(V);
        mul_v<ArchType::ASCEND_V220, float>(
            ub_out, ub_out, ub_mask[64 * ignore_num],
             1,
             1,
             1,
             1,
             8,
             8,
             8
        );
    }
    SET_FLAG(V, MTE3, event_id);
    WAIT_FLAG(V, MTE3, event_id);

    copy_vec_ub2gm(gm_A[gm_offset + offset], ub_out, len + ignore_num);
}

__aicore__ __inline__ __attribute__((always_inline)) void ssyr_upper_aiv(
    uint32_t vec_idx,
    AscendC::GlobalTensor<float> x,
    AscendC::GlobalTensor<float> y,
    AscendC::GlobalTensor<float> A,
    float alpha, uint32_t n, uint32_t core_num)
{
    if (core_num == 0) {
        core_num = 1;
    }

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> ub_x = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    AscendC::LocalTensor<float> ub_y = buf.GetBuffer<BufferType::ASCEND_UB, float>(47.5 * 1024);
    AscendC::LocalTensor<float> ub_out_ping = buf.GetBuffer<BufferType::ASCEND_UB, float>(95 * 1024);
    AscendC::LocalTensor<float> ub_out_pong = buf.GetBuffer<BufferType::ASCEND_UB, float>(142.5 * 1024);

    AscendC::LocalTensor<float> ub_mask = buf.GetBuffer<BufferType::ASCEND_UB, float>(190 * 1024);

    uint32_t max_data_count = 12160;

    AscendC::Duplicate<float, false>(ub_mask, (float)1.0, 64, 8, 1, 8);
    PIPE_BARRIER(ALL);

    for (uint32_t i = 0; i < 8; i++) {
        for (uint32_t j = 0; j < i; j++) {
            ub_mask.SetValue(i * 64 + j, static_cast<float>(0));
        }
    }
    PIPE_BARRIER(ALL);

    uint32_t row_block_num;
    uint32_t row_remain_num;

    uint32_t ping_flag = 1;

    row_block_num = n / max_data_count;
    row_remain_num = n % max_data_count;

    if (row_remain_num > 0)
        row_block_num++;

    uint32_t curr_compute_row_num = max_data_count;
    uint32_t curr_compute_col_num = max_data_count;

    for (uint32_t row_block_idx = 0; row_block_idx < row_block_num; row_block_idx++) {
        if (row_block_idx == row_block_num - 1 && row_remain_num > 0)
            curr_compute_row_num = row_remain_num;
        else
            curr_compute_row_num = max_data_count;

        uint32_t core_compute_row_num = curr_compute_row_num / core_num;
        uint32_t remain_row_num = curr_compute_row_num % core_num;

        uint32_t use_core_num = core_num;
        if (core_compute_row_num == 0) {
            core_compute_row_num = 1;
            use_core_num = remain_row_num;
            remain_row_num = 0;
        }

        if (vec_idx < use_core_num) {
            for (uint32_t col_block_idx = row_block_idx; col_block_idx < row_block_num; col_block_idx++) {
                if (col_block_idx == row_block_num - 1 && row_remain_num > 0)
                    curr_compute_col_num = row_remain_num;
                else
                    curr_compute_col_num = max_data_count;

                uint32_t outOffset = row_block_idx * max_data_count * n + col_block_idx * max_data_count;

                uint32_t x_offset = row_block_idx * max_data_count;

                uint32_t start_row_idx =
                    vec_idx < remain_row_num ?
                        (core_compute_row_num + 1) * vec_idx :
                        (core_compute_row_num + 1) * remain_row_num + (vec_idx - remain_row_num) * core_compute_row_num;

                uint32_t x_copy_num = vec_idx < remain_row_num ? core_compute_row_num + 1 : core_compute_row_num;
                copy_vec_gm2ub(ub_x, x[x_offset + start_row_idx], x_copy_num);

                SET_FLAG(MTE2, V, EVENT_ID0);
                WAIT_FLAG(MTE2, V, EVENT_ID0);
                SET_FLAG(MTE2, V, EVENT_ID2);
                WAIT_FLAG(MTE2, V, EVENT_ID2);

                uint32_t repeat_num = (x_copy_num + 63) / 64;

                muls_v<ArchType::ASCEND_V220, float>(
                    ub_x, ub_x, alpha,
                    repeat_num,
                    1,
                    1,
                    8,
                    8
                );
                PIPE_BARRIER(V);

                uint32_t y_offset = col_block_idx * max_data_count;

                uint32_t y_copy_num = curr_compute_col_num;

                copy_vec_gm2ub(ub_y, y[y_offset], y_copy_num);

                PIPE_BARRIER(ALL);

                SET_FLAG(MTE3, V, EVENT_ID0);
                SET_FLAG(MTE3, V, EVENT_ID2);
                if (row_block_idx == col_block_idx) {
                    for (uint32_t i = 0; i < x_copy_num; i++) {
                        auto ub_out = ping_flag ? ub_out_ping : ub_out_pong;
                        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID2;

                        WAIT_FLAG(MTE3, V, event_id);

                        vec_single_core_compute_upper(ub_x, ub_y, ub_out, ub_mask, A, curr_compute_col_num, i,
                                                      outOffset + (start_row_idx + i) * n,
                                                      y_copy_num - start_row_idx - i, event_id);

                        SET_FLAG(MTE3, V, event_id);
                        ping_flag = 1 - ping_flag;
                    }
                } else {
                    for (uint32_t i = 0; i < x_copy_num; i++) {
                        auto ub_out = ping_flag ? ub_out_ping : ub_out_pong;
                        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID2;

                        WAIT_FLAG(MTE3, V, event_id);

                        vec_single_core_compute_lower(ub_x, ub_y, ub_out, A, i, outOffset + (start_row_idx + i) * n,
                                                      y_copy_num, event_id);

                        SET_FLAG(MTE3, V, event_id);
                        ping_flag = 1 - ping_flag;
                    }
                }
                WAIT_FLAG(MTE3, V, EVENT_ID0);
                WAIT_FLAG(MTE3, V, EVENT_ID2);
            }
        }
    }
    PIPE_BARRIER(ALL);
}
#endif

extern "C" __global__ __aicore__ __vector__ void ssyr(__gm__ float *__restrict__ d_x, __gm__ float *__restrict__ d_A,
                                           __gm__ float *__restrict__ workspace, __gm__ float *__restrict__ tiling_gm)
{
#if __DAV_C220_VEC__
    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(tiling_gm);

    uint32_t uplo = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf));
    uint32_t n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 4));
    float alpha = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 8));
    uint32_t coreNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 12));

    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    AscendC::SetAtomicAdd<float>();

    uint32_t vec_idx = AscendC::GetBlockIdx();

    AscendC::GlobalTensor<float> d_x_tensor;
    AscendC::GlobalTensor<float> d_A_tensor;
    d_x_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_x));
    d_A_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(d_A));

    if (uplo == 0) {
        ssyr_lower_aiv(vec_idx, d_x_tensor, d_x_tensor, d_A_tensor, alpha, n, coreNum);
    } else {
        ssyr_upper_aiv(vec_idx, d_x_tensor, d_x_tensor, d_A_tensor, alpha, n, coreNum);
    }

    PIPE_BARRIER(ALL);
    AscendC::SetAtomicNone();
#endif
}

void ssyr_kernel_do(GM_ADDR gm_x, GM_ADDR gm_A, GM_ADDR workSpace, GM_ADDR tilingGm,
                    uint32_t numBlocks, void *stream)
{
    ssyr<<<numBlocks, nullptr, stream>>>((float *)gm_x, (float *)gm_A,
                                          (float *)workSpace, (float *)tilingGm);
}