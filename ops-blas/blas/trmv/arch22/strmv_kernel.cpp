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

namespace AclBlassKernel {
typedef enum {
    ACLBLASS_FILL_MODE_LOWER = 0,
    ACLBLASS_FILL_MODE_UPPER = 1,
    ACLBLASS_FILL_MODE_FULL = 2
} FillMode;

typedef enum {
    ACLBLASS_OP_N = 0,
    ACLBLASS_OP_T = 1,
    ACLBLASS_OP_C = 2
} Operation;

typedef enum {
    ACLBLASS_DIAG_NON_UNIT = 0,
    ACLBLASS_DIAG_UNIT = 1
} DiagType;
}

constexpr int BLOCK_DIM = 128;
constexpr int UB_MATRIX_SIZE = BLOCK_DIM * BLOCK_DIM;
constexpr int UB_VECTOR_SIZE = BLOCK_DIM;
constexpr int ELE_SIZE = sizeof(float);

#if __DAV_C220_VEC__
__aicore__ __inline__ __attribute__((always_inline)) void load_matrix_gm2ub(__ubuf__ float *dst, __gm__ float *src,
                                                                            int64_t m_real, int64_t n_real,
                                                                            int64_t m_real_pad, int64_t n_real_pad,
                                                                            int64_t stride)
{
    uint16_t nBurst = n_real;
    uint32_t lenBurst = m_real * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = (stride - m_real) * sizeof(float);
    uint32_t dstGap = (BLOCK_DIM - m_real_pad) / 8;
    copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void load_vector_gm2ub(__ubuf__ float *dst, __gm__ float *src,
                                                                            __ubuf__ float *wksp, int64_t len,
                                                                            int64_t inc)
{
    if (inc == 1) {
        uint16_t nBurst = 1;
        uint32_t lenBurst = len * sizeof(float);
        uint8_t leftPaddingNum = 0;
        uint8_t rightPaddingNum = 0;
        uint32_t srcGap = 0;
        uint32_t dstGap = 0;
        copy_gm_to_ubuf_align_b32(dst, src, 0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else {
        int32_t content = UB_MATRIX_SIZE / 2;
        int32_t loop = len * inc / content;
        int32_t remain = len * inc % content;
        int32_t start_posi = 0;
        int32_t iub = 0;

        for (int i = 0; i < loop; ++i) {
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);

            copy_gm_to_ubuf_align_b32(wksp, src + i * content, 0, 1, content * sizeof(float), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);

            int iwhile = start_posi;
            while (iwhile < content) {
                *(dst + iub) = *(wksp + iwhile);
                iwhile = iwhile + inc;
                iub = iub + 1;
            }

            start_posi = iwhile - content;
        }
        if (remain) {
            set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);

            copy_gm_to_ubuf_align_b32(wksp, src + loop * content, 0, 1, remain * sizeof(float), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);

            int iwhile = start_posi;
            while (iub < len && iwhile < content) {
                *(dst + iub) = *(wksp + iwhile);
                iwhile = iwhile + inc;
                iub = iub + 1;
            }
        }
        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
    }
}

__aicore__ __inline__ __attribute__((always_inline)) void mask_invalid(__ubuf__ float *matrix, __ubuf__ float *uplo,
                                                                       uint64_t row_num,
                                                                       AclBlassKernel::DiagType diag)
{
    vmul(matrix, matrix, uplo, row_num, 1, 1, 1, 16, 16, 16);
    pipe_barrier(PIPE_V);
    vmul(matrix + 64, matrix + 64, uplo + 64, row_num, 1, 1, 1, 16, 16, 16);

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

    if (diag == AclBlassKernel::ACLBLASS_DIAG_UNIT) {
        for (uint32_t i = 0; i < row_num; ++i) {
            *(matrix + BLOCK_DIM * i + i) = 1;
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

__aicore__ __inline__ __attribute__((always_inline)) void matrix_vector_muls_notrans(__ubuf__ float *dst,
                                                                                     __ubuf__ float *src0,
                                                                                     __ubuf__ float *src1,
                                                                                     int64_t m_real, int64_t n_real)
{
    for (int64_t n_idx = 0; n_idx < n_real; ++n_idx) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
        float t = *(src1 + n_idx);

        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
        vaxpy(dst, src0 + n_idx * BLOCK_DIM, t, 2, 1, 1, 8, 8);
    }
}

__aicore__ __inline__ __attribute__((always_inline)) void matrix_vector_muls_trans(
    __ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1,
    __ubuf__ float *wksp,
    int64_t m_real, int64_t n_real)
{
    int64_t loop = n_real / 64;
    int64_t remain = n_real % 64;

    if (loop) {
        for (int64_t idx = 0; idx < loop; idx++) {
            if (idx == 0) {
                vmul(wksp, src0 + 64 * idx, src1 + 64 * idx,
                     m_real, 1, 1, 1, 8, 16, 0);
            } else {
                pipe_barrier(PIPE_V);
                vmla(wksp, src0 + 64 * idx, src1 + 64 * idx,
                     m_real, 1, 1, 1, 8, 16, 0);
            }
        }
        if (remain) {
            set_mask_norm();
            set_vector_mask((uint64_t)0, (((uint64_t)1 << remain) - 1));
            pipe_barrier(PIPE_V);
            vmla(wksp, src0 + 64 * loop, src1 + 64 * loop,
                 m_real, 1, 1, 1, 8, 16, 0);
            set_mask_norm();
            set_vector_mask((uint64_t)0, (uint64_t)-1);
        }
        pipe_barrier(PIPE_V);
        vcadd(wksp, wksp, m_real, 1, 1, 8, false);
    } else {
        set_mask_norm();
        set_vector_mask((uint64_t)0, (((uint64_t)1 << remain) - 1));
        pipe_barrier(PIPE_V);
        vmul(wksp, src0, src1,
             m_real, 1, 1, 1, 8, 16, 0);
        pipe_barrier(PIPE_V);
        vcadd(wksp, wksp, m_real, 1, 1, 8, false);
        set_mask_norm();
        set_vector_mask((uint64_t)0, (uint64_t)-1);
    }
    pipe_barrier(PIPE_V);
    vadd(dst, wksp, dst, 2, 1, 1, 1, 8, 8, 8);
}

__aicore__ __inline__ __attribute__((always_inline)) void store_vector_ub2gm(__gm__ float *dst, __ubuf__ float *src,
                                                                             uint64_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = (uint32_t)len * sizeof(float);
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    copy_ubuf_to_gm_align_b32(dst, src, 0, nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_wksp_to_x(__gm__ float *__restrict__ gm_X,
                                                                         __gm__ float *__restrict__ gm_wksp,
                                                                         uint64_t len, uint64_t inc)
{
    if (get_block_idx() == 0 && get_subblockid() == 0) {
        auto ub_tmpw = reinterpret_cast<__ubuf__ float *>((uintptr_t)0);
        auto ub_tmpx = reinterpret_cast<__ubuf__ float *>((uintptr_t)128 * 128 * 4);
        int32_t cont_tmpw = 128 * 128;
        int32_t cont_tmpx = 128 * 128;

        int32_t loop_tmpw = (int32_t)len / cont_tmpw;
        int32_t remain_tmpw = (int32_t)len % cont_tmpw;

        if (inc == 1) {
            for (int32_t w_idx = 0; w_idx < loop_tmpw; ++w_idx) {
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                copy_gm_to_ubuf_align_b32(ub_tmpw, gm_wksp + w_idx * cont_tmpw, 0, 1, cont_tmpw * sizeof(float), 0, 0, 0, 0);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                copy_ubuf_to_gm_align_b32(gm_X + w_idx * cont_tmpw, ub_tmpw, 0, 1, cont_tmpw * sizeof(float), 0, 0, 0, 0);
            }
            if (remain_tmpw) {
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                copy_gm_to_ubuf_align_b32(ub_tmpw, gm_wksp + loop_tmpw * cont_tmpw, 0, 1, remain_tmpw * sizeof(float), 0, 0, 0, 0);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
                copy_ubuf_to_gm_align_b32(gm_X + loop_tmpw * cont_tmpw, ub_tmpw, 0, 1, remain_tmpw * sizeof(float), 0, 0, 0, 0);
            }
        } else {
            for (int32_t w_idx = 0; w_idx < loop_tmpw; ++w_idx) {
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                copy_gm_to_ubuf_align_b32(ub_tmpw, gm_wksp + w_idx * cont_tmpw, 0, 1, cont_tmpw * sizeof(float), 0, 0, 0, 0);

                int32_t loop_tmpx = (cont_tmpw * inc) / cont_tmpx;
                int32_t remain_tmpx = (cont_tmpw * inc) % cont_tmpx;
                int32_t start_posi = 0;
                int32_t iub_tmpw = 0;

                for (int32_t x_idx = 0; x_idx < loop_tmpx; ++x_idx) {
                    pipe_barrier(PIPE_MTE2);
                    copy_gm_to_ubuf_align_b32(ub_tmpx, gm_X + w_idx * cont_tmpw * inc + x_idx * cont_tmpx, 0, 1, cont_tmpx * sizeof(float), 0, 0, 0, 0);

                    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
                    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
                    int iwhile = start_posi;

                    while (iwhile < cont_tmpx) {
                        *(ub_tmpx + iwhile) = *(ub_tmpw + iub_tmpw);
                        iwhile = iwhile + inc;
                        iub_tmpw = iub_tmpw + 1;
                    }
                    start_posi = iwhile - cont_tmpx;
                    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
                    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);

                    copy_ubuf_to_gm_align_b32(gm_X + w_idx * cont_tmpw * inc + x_idx * cont_tmpx, ub_tmpx, 0, 1, cont_tmpx * sizeof(float), 0, 0, 0, 0);
                    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                }

                if (remain_tmpx) {
                    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                    copy_gm_to_ubuf_align_b32(ub_tmpx, gm_X + w_idx * cont_tmpw * inc + loop_tmpx * cont_tmpx, 0, 1, remain_tmpx * sizeof(float), 0, 0, 0, 0);

                    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
                    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
                    int iwhile = start_posi;

                    while (iub_tmpw < cont_tmpw && iwhile < cont_tmpx) {
                        *(ub_tmpx + iwhile) = *(ub_tmpw + iub_tmpw);
                        iwhile = iwhile + inc;
                        iub_tmpw = iub_tmpw + 1;
                    }
                    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
                    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);

                    copy_ubuf_to_gm_align_b32(gm_X + w_idx * cont_tmpw * inc + loop_tmpx * cont_tmpx, ub_tmpx, 0, 1, remain_tmpx * sizeof(float), 0, 0, 0, 0);
                }
            }

            if (remain_tmpw) {
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

                copy_gm_to_ubuf_align_b32(ub_tmpw, gm_wksp + loop_tmpw * cont_tmpw, 0, 1, remain_tmpw * sizeof(float), 0, 0, 0, 0);

                int32_t loop_tmpx = (remain_tmpw * inc) / cont_tmpx;
                int32_t remain_tmpx = (remain_tmpw * inc) % cont_tmpx;
                int32_t start_posi = 0;
                int32_t iub_tmpw = 0;

                for (int32_t x_idx = 0; x_idx < loop_tmpx; ++x_idx) {
                    pipe_barrier(PIPE_MTE2);

                    copy_gm_to_ubuf_align_b32(ub_tmpx, gm_X + loop_tmpw * cont_tmpw * inc + x_idx * cont_tmpx, 0, 1, cont_tmpx * sizeof(float), 0, 0, 0, 0);

                    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
                    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
                    int iwhile = start_posi;

                    while (iwhile < cont_tmpx) {
                        *(ub_tmpx + iwhile) = *(ub_tmpw + iub_tmpw);
                        iwhile = iwhile + inc;
                        iub_tmpw = iub_tmpw + 1;
                    }
                    start_posi = iwhile - cont_tmpx;
                    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
                    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);

                    copy_ubuf_to_gm_align_b32(gm_X + loop_tmpw * cont_tmpw * inc + x_idx * cont_tmpx, ub_tmpx, 0, 1, cont_tmpx * sizeof(float), 0, 0, 0, 0);
                    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                }
                if (remain_tmpx) {
                    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);

                    copy_gm_to_ubuf_align_b32(ub_tmpx, gm_X + loop_tmpw * cont_tmpw * inc + loop_tmpx * cont_tmpx, 0, 1, remain_tmpx * sizeof(float), 0, 0, 0, 0);

                    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
                    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID1);
                    int iwhile = start_posi;

                    while (iub_tmpw < remain_tmpw && iwhile < cont_tmpx) {
                        *(ub_tmpx + iwhile) = *(ub_tmpw + iub_tmpw);
                        iwhile = iwhile + inc;
                        iub_tmpw = iub_tmpw + 1;
                    }
                    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
                    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);

                    copy_ubuf_to_gm_align_b32(gm_X + loop_tmpw * cont_tmpw * inc + loop_tmpx * cont_tmpx, ub_tmpx, 0, 1, remain_tmpx * sizeof(float), 0, 0, 0, 0);
                }
            }
        }
    }
}

__aicore__ __inline__ __attribute__((always_inline)) void aclblassStrmv(
    __gm__ float *__restrict__ gm_A, __gm__ float *__restrict__ gm_X, __gm__ float *__restrict__ gm_wksp,
    __gm__ float *__restrict__ gm_uplo, AclBlassKernel::FillMode mode, AclBlassKernel::Operation trans, AclBlassKernel::DiagType diag,
    int64_t M, int64_t lda, int64_t incx, int64_t M0)
{
    if (M0 == 0) {
        M0 = 128;
    }
    auto ub_a_ptr = reinterpret_cast<__ubuf__ float *>((uintptr_t)0);
    auto ub_x_ptr = reinterpret_cast<__ubuf__ float *>((uintptr_t)UB_MATRIX_SIZE * ELE_SIZE);
    auto ub_uplo_matrix = reinterpret_cast<__ubuf__ float *>((uintptr_t)(UB_MATRIX_SIZE + UB_VECTOR_SIZE) * ELE_SIZE);
    auto ub_res_ptr = reinterpret_cast<__ubuf__ float *>((uintptr_t)(UB_MATRIX_SIZE * 2 + UB_VECTOR_SIZE) * ELE_SIZE);
    auto ub_wksp_ptr = reinterpret_cast<__ubuf__ float *>((uintptr_t)(UB_MATRIX_SIZE * 2 + UB_VECTOR_SIZE * 2) * ELE_SIZE);

    int64_t m_tiles = (M + M0 - 1) / M0;
    int64_t n_tiles = 1;
    int64_t k_loop = (M + M0 - 1) / M0;
    int64_t m_remain = M % M0;
    int64_t k_remain = M % M0;

    copy_gm_to_ubuf(ub_uplo_matrix, gm_uplo, 0, M0, M0 / 8, 0, (BLOCK_DIM - M0) / 8);

    int32_t sub_blocks_num = get_subblockdim();
    int32_t blocks_num = get_block_num() * sub_blocks_num;
    if (blocks_num == 0) {
        blocks_num = 1;
    }
    int64_t tiles_num = m_tiles * n_tiles;
    int64_t tiles_per_core = tiles_num / blocks_num;
    int64_t block_id = get_block_idx() * sub_blocks_num + get_subblockid();
    if (block_id < tiles_num % blocks_num) {
        ++tiles_per_core;
    }
    int32_t btrans = trans == AclBlassKernel::ACLBLASS_OP_N ? 0 : 1;
    int32_t bmode = mode == AclBlassKernel::ACLBLASS_FILL_MODE_UPPER ? 1 : 0;

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

    for (int64_t tiles_idx = 0; tiles_idx < tiles_per_core; ++tiles_idx) {
        int64_t block_index = tiles_idx * blocks_num + get_block_idx() * sub_blocks_num + get_subblockid();
        int64_t row = block_index / n_tiles;
        int64_t m_real = M0;
        if (row == m_tiles - 1 && m_remain > 0) {
            m_real = m_remain;
        }
        int64_t m_real_pad = m_real % 8 ? (m_real & 0xfffffff8) + 8 : m_real;

        __gm__ float *gm_wksp_ptr = gm_wksp + row * M0;

        int32_t k_idx = row;
        int32_t k_dst = k_loop;
        if (btrans - bmode == 0) {
            k_idx = 0;
            k_dst = row + 1;
        }

        for (; k_idx < k_dst; ++k_idx) {
            int32_t k_real = M0;
            if (k_idx == k_loop - 1 && k_remain > 0) {
                k_real = k_remain;
            }

            int64_t k_real_pad = k_real % 8 ? (k_real & 0xfffffff8) + 8 : k_real;
            __gm__ float *gm_x_ptr = gm_X + M0 * incx * k_idx;

            if (trans == AclBlassKernel::ACLBLASS_OP_N) {
                __gm__ float *gm_a_ptr = gm_A + M0 * row + k_idx * M0 * lda;
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

                load_matrix_gm2ub(ub_a_ptr, gm_a_ptr, m_real, k_real, m_real_pad, k_real_pad, lda);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

                if (k_idx == row) {
                    mask_invalid(ub_a_ptr, ub_uplo_matrix, m_real, diag);
                }
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

                load_vector_gm2ub(ub_x_ptr, gm_x_ptr, ub_wksp_ptr, k_real, incx);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

                if (k_idx == 0 || (((btrans - bmode) != 0) && k_idx == row)) {
                    vector_dup(ub_res_ptr, (float)0, 2, 1, 1, 8, 8);
                }
                set_flag(PIPE_V, PIPE_S, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_S, EVENT_ID1);

                matrix_vector_muls_notrans(ub_res_ptr, ub_a_ptr, ub_x_ptr, m_real, k_real);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

            } else {
                __gm__ float *gm_a_ptr = gm_A + M0 * k_idx + row * M0 * lda;

                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                load_matrix_gm2ub(ub_a_ptr, gm_a_ptr, k_real, m_real, k_real_pad, m_real_pad, lda);

                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if (k_idx == row) {
                    mask_invalid(ub_a_ptr, ub_uplo_matrix, k_real, diag);
                }
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

                load_vector_gm2ub(ub_x_ptr, gm_x_ptr, ub_wksp_ptr, k_real, incx);

                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                if (k_idx == 0 || (((btrans - bmode) != 0) && k_idx == row)) {
                    vector_dup(ub_res_ptr, (float)0, 2, 1, 1, 8, 8);
                }
                pipe_barrier(PIPE_V);

                matrix_vector_muls_trans(ub_res_ptr, ub_a_ptr, ub_x_ptr, ub_wksp_ptr, m_real, k_real);

                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            }
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);

        store_vector_ub2gm(gm_wksp_ptr, ub_res_ptr, m_real);
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

    uint64_t flag_id = 3;
    uint64_t mode_sync = 0;
    uint64_t config = 1 | (mode_sync << 4) | (flag_id << 8);
    ffts_cross_core_sync(PIPE_MTE3, config);
}
#endif

extern "C" __global__ __aicore__ __vector__ void strmv(__gm__ float *__restrict__ gm_A,
                                            __gm__ float *__restrict__ gm_X, __gm__ float *__restrict__ gm_uplo,
                                            __gm__ float *__restrict__ gm_output, __gm__ float *__restrict__ gm_wksp,
                                            __gm__ uint32_t *__restrict__ tiling_gm)
{
#if __DAV_C220_VEC__
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(tiling_gm);
    uint32_t M = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf));
    uint32_t uplo = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 4));
    uint32_t trans = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 8));
    uint32_t diag = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 12));
    uint32_t lda = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 16));
    uint32_t incx = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 20));
    uint32_t M0 = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 24));

    AclBlassKernel::FillMode mode = uplo == 1 ? AclBlassKernel::ACLBLASS_FILL_MODE_UPPER : AclBlassKernel::ACLBLASS_FILL_MODE_LOWER;
    AclBlassKernel::Operation trans_t = trans == 0 ? AclBlassKernel::ACLBLASS_OP_N : AclBlassKernel::ACLBLASS_OP_T;
    AclBlassKernel::DiagType diag_t = diag == 1 ? AclBlassKernel::ACLBLASS_DIAG_UNIT : AclBlassKernel::ACLBLASS_DIAG_NON_UNIT;

    aclblassStrmv(gm_A, gm_X, gm_wksp, gm_uplo, mode, trans_t, diag_t, M, lda, incx, M0);

    uint64_t flag_id = 3;
    wait_flag_dev(flag_id);

    copy_wksp_to_x(gm_output, gm_wksp, M, incx);
#endif
}

void strmv_kernel_do(GM_ADDR gm_A, GM_ADDR gm_X, GM_ADDR gm_uplo, GM_ADDR gm_output,
                     GM_ADDR gm_wksp, GM_ADDR workSpace, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream)
{
    strmv<<<numBlocks, nullptr, stream>>>((float *)gm_A, (float *)gm_X,
                                           (float *)gm_uplo, (float *)gm_output,
                                           (float *)gm_wksp, (uint32_t *)tilingGm);
}