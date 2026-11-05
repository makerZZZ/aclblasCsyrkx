/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifdef __CCE_KT_TEST__
#undef __aicore__
#define __aicore__
#else
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#endif

#include "kernel_operator.h"

#if __DAV_C220_VEC__
__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub_uint32(__ubuf__ uint32_t *dst,
                                                                                __gm__ uint32_t *src, uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(uint32_t);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    copy_gm_to_ubuf_align_b32(dst, src,
                              0,
                              nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_gm2ub(__ubuf__ float *dst, __gm__ float *src,
                                                                          uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    copy_gm_to_ubuf_align_b32(dst, src,
                              0,
                              nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_vec_ub2gm(__gm__ float *dst, __ubuf__ float *src,
                                                                          uint32_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    copy_ubuf_to_gm_align_b32(dst, src,
                              0,
                              nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void caxpy_compute_aiv(
    __gm__ float *gm_in, __gm__ float *gm_out, __ubuf__ float *ub_in, __ubuf__ float *ub_out,
    __ubuf__ uint32_t *ub_offset, float s_real, float s_imag, uint32_t copy_len, uint32_t len, uint32_t event_id)
{
    uint32_t repeatTime = (len + 63) / 64;
    uint32_t computeRepeat = (len / 2 + 63) / 64;

    uint32_t real_offset = 0;
    uint32_t imag_offset = 38 * 1024 / sizeof(float) / 2;

    auto ub_out_real = ub_out;
    auto ub_out_imag = ub_out + imag_offset;

    copy_vec_gm2ub(ub_in, gm_in, copy_len);

    set_flag(PIPE_MTE2, PIPE_V, event_id);
    wait_flag(PIPE_MTE2, PIPE_V, event_id);

    vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(ub_out_real),
              reinterpret_cast<__ubuf__ uint32_t *>(ub_in),
              nullptr, repeatTime,
              1,
              1,
              8, 8);

    vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(ub_out_imag),
              reinterpret_cast<__ubuf__ uint32_t *>(ub_in),
              nullptr, repeatTime,
              1,
              2,
              8, 8);

    pipe_barrier(PIPE_V);

    vmuls(ub_in, ub_out_real, s_real, computeRepeat, 1, 1, 8, 8);
    vmuls(ub_in + imag_offset, ub_out_real, s_imag, computeRepeat, 1, 1, 8, 8);

    vmuls(ub_out_real, ub_out_imag, s_imag, computeRepeat, 1, 1, 8, 8);

    pipe_barrier(PIPE_V);
    vsub(ub_in, ub_in, ub_out_real, computeRepeat, 1, 1, 1, 8, 8, 8);

    vmuls(ub_out_imag, ub_out_imag, s_real, computeRepeat, 1, 1, 8, 8);

    pipe_barrier(PIPE_V);
    vadd(ub_in + imag_offset, ub_out_imag, ub_in + imag_offset, computeRepeat, 1, 1, 1, 8, 8, 8);

    pipe_barrier(PIPE_V);

    vgather(reinterpret_cast<__ubuf__ uint32_t *>(ub_out), reinterpret_cast<__ubuf__ uint32_t *>(ub_offset),
            (uintptr_t)(ub_in),
            8,
            repeatTime
    );

    set_flag(PIPE_V, PIPE_MTE3, event_id);
    wait_flag(PIPE_V, PIPE_MTE3, event_id);

    copy_vec_ub2gm(gm_out, ub_out, copy_len);
}

__aicore__ __inline__ __attribute__((always_inline)) void caxpy_aiv(__gm__ float *gm_in, __gm__ uint32_t *gm_aug,
                                                                    __gm__ float *gm_out, float s_real, float s_imag,
                                                                    uint32_t offset, uint32_t calNum)
{
    set_atomic_add();
    set_atomic_f32();

    auto ub_out_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)0 * 1024);
    auto ub_out_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)38 * 1024);
    auto ub_in_ping = reinterpret_cast<__ubuf__ float *>((uintptr_t)76 * 1024);
    auto ub_in_pong = reinterpret_cast<__ubuf__ float *>((uintptr_t)114 * 1024);
    auto ub_offset = reinterpret_cast<__ubuf__ uint32_t *>((uintptr_t)152 * 1024);

    uint32_t ping_flag = 1;

    uint32_t maxDataCount = 38 * 1024 / sizeof(float);

    uint32_t repeatTime = calNum / maxDataCount;
    uint32_t remainNum = calNum % maxDataCount;

    copy_vec_gm2ub_uint32(ub_offset, gm_aug, maxDataCount);

    uint32_t curr_offset = offset;

    if (repeatTime > 0) {
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        for (uint32_t i = 0; i < repeatTime; i++) {
            auto ub_in = ping_flag ? ub_in_ping : ub_in_pong;
            auto ub_out = ping_flag ? ub_out_ping : ub_out_pong;

            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);

            caxpy_compute_aiv(gm_in + curr_offset, gm_out + curr_offset, ub_in, ub_out, ub_offset, s_real, s_imag,
                              maxDataCount, maxDataCount, event_id);

            set_flag(PIPE_MTE3, PIPE_MTE2, event_id);

            curr_offset += maxDataCount;
            ping_flag = 1 - ping_flag;
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }

    if (remainNum > 0) {
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        auto ub_in = ping_flag ? ub_in_ping : ub_in_pong;
        auto ub_out = ping_flag ? ub_out_ping : ub_out_pong;

        auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

        wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);

        caxpy_compute_aiv(gm_in + curr_offset, gm_out + curr_offset, ub_in, ub_out, ub_offset, s_real, s_imag,
                          remainNum, maxDataCount, event_id);

        set_flag(PIPE_MTE3, PIPE_MTE2, event_id);

        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }

    pipe_barrier(PIPE_ALL);
    set_atomic_none();
}
#endif

extern "C" __global__ __aicore__ __vector__ void caxpy(__gm__ float *__restrict__ x, __gm__ uint32_t *__restrict__ aug,
                                            __gm__ float *__restrict__ y, __gm__ float *__restrict__ tiling_gm)
{
#if __DAV_C220_VEC__
    auto vec_idx = AscendC::GetBlockIdx();

    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(tiling_gm);

    uint32_t n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf));
    float alphaReal = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 4));
    float alphaImag = (*(__gm__ float *)((__gm__ uint8_t *)tiling_buf + 8));

    uint32_t offset = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 12 + 4 * vec_idx));
    uint32_t calNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 12 + 40 * 4 + 4 * vec_idx));

    if (calNum > 0) {
        caxpy_aiv(x, aug, y, alphaReal, alphaImag, offset, calNum);
    }
#endif
}

void caxpy_kernel_do(GM_ADDR x, GM_ADDR maskBuf, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream)
{
    caxpy<<<numBlocks, nullptr, stream>>>((float *)x,
                                           (uint32_t *)maskBuf,
                                           (float *)y, (float *)tilingGm);
}