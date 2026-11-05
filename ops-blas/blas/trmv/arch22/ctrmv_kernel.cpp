/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


/* !
 * \file ctrmv_kernel.cpp
 * \brief Kernel side implementation for ctrmv operator
 */

#include "kernel_operator.h"
#include "common/helper/kernel_utils.h"
#include "common/iterator/iterator.h"
#include "common/compute/simd.h"
#include "cann_ops_blas_common.h"  // Reuse enum definitions from host

#ifdef __CCE_KT_TEST__
#undef __aicore__
#define __aicore__
#else
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#endif

constexpr int BLOCK_DIM = 64;
constexpr int UB_MATRIX_SIZE = BLOCK_DIM * BLOCK_DIM;
constexpr int UB_VECTOR_SIZE = BLOCK_DIM;
constexpr int ELE_SIZE = sizeof(float);

__aicore__ __inline__ __attribute__((always_inline)) void store_vector_ub2gm(
    AscendC::GlobalTensor<float> dst,
    AscendC::LocalTensor<float> src,
    uint64_t len)
{
    uint16_t nBurst = 1;
    uint32_t lenBurst = len * sizeof(float) * 2;
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = 0;
    uint32_t dstGap = 0;
    ub_to_gm_align<ArchType::ASCEND_V220, float>(
        dst, src,
        0,  // sid
        nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}

__aicore__ __inline__ __attribute__((always_inline)) void loda_matrix_gm2ub(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    int64_t m_real, int64_t n_real,
    int64_t m_real_pad, int64_t n_real_pad,
    int64_t stride)
{
    m_real_pad = (m_real * 2 + 7) / 8 * 8;
    uint16_t nBurst = n_real;
    uint32_t lenBurst = (uint32_t)m_real * sizeof(float) * 2;
    uint8_t leftPaddingNum = 0;
    uint8_t rightPaddingNum = 0;
    uint32_t srcGap = (uint32_t)(stride - m_real) * sizeof(float) * 2;
    uint32_t dstGap =
        (uint32_t)(BLOCK_DIM * 2 - m_real_pad) / 8;  // (64 * 2 - 8) / 8 * 2 = 14    14 * 8 = 120 128 - 8 /8
    gm_to_ub_align<ArchType::ASCEND_V220, float>(
        dst, src,
        0,  // sid
        nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
}
__aicore__ __inline__ __attribute__((always_inline)) void loda_vector_gm2ub(
    AscendC::LocalTensor<float> dst,
    AscendC::GlobalTensor<float> src,
    AscendC::LocalTensor<float> wksp,
    int64_t len, int64_t inc)
{
    if (inc == 1) {
        uint16_t nBurst = 1;
        uint32_t lenBurst = len * sizeof(float) * 2;
        uint8_t leftPaddingNum = 0;
        uint8_t rightPaddingNum = 0;
        uint32_t srcGap = 0;
        uint32_t dstGap = 0;
        gm_to_ub_align<ArchType::ASCEND_V220, float>(
            dst, src,
            0,  // sid
            nBurst, lenBurst, leftPaddingNum, rightPaddingNum, srcGap, dstGap);
    } else {
        int32_t content = UB_MATRIX_SIZE;
        int32_t loop = len * inc * 2 / content;
        int32_t remain = len * inc * 2 % content;
        int32_t start_posi = 0;
        int32_t iub = 0;
        for (int i = 0; i < loop; ++i) {
            PIPE_BARRIER(ALL);
            gm_to_ub_align<ArchType::ASCEND_V220, float>(
                wksp, src[i * content],
                0,  // sid
                1, content * sizeof(float), 0, 0, 0, 0);
            PIPE_BARRIER(ALL);
            int iwhile = start_posi;
            while (iwhile < content) {
                dst.SetValue(iub,  wksp.GetValue(iwhile));
                dst.SetValue(iub + 1, wksp.GetValue(iwhile + 1));
                iwhile = iwhile + inc * 2;
                iub = iub + 2;
            }
            PIPE_BARRIER(ALL);
            start_posi = iwhile - content;
        }
        if (remain) {
            PIPE_BARRIER(ALL);
            gm_to_ub_align<ArchType::ASCEND_V220, float>(
                wksp, src[loop * content],
                0,  // sid
                1, remain * sizeof(float), 0, 0, 0, 0);
            PIPE_BARRIER(ALL);
            int iwhile = start_posi;
            while (iub < len * 2 && iwhile < content) {
                dst.SetValue(iub,  wksp.GetValue(iwhile));
                dst.SetValue(iub + 1, wksp.GetValue(iwhile + 1));
                iwhile = iwhile + inc * 2;
                iub = iub + 2;
            }
            PIPE_BARRIER(ALL);
        }
    }
}

__aicore__ __inline__ __attribute__((always_inline)) void complex_to_real_imag(
    AscendC::LocalTensor<float> dst_real,
    AscendC::LocalTensor<float> dst_imag,
    AscendC::LocalTensor<float> src,
    uint32_t block_len)
{
    vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(dst_real.GetPhyAddr()),
              reinterpret_cast<__ubuf__ uint32_t *>(src.GetPhyAddr()),
              nullptr, block_len * 2 / 64, 1, 1, 8, 8);
    vreducev2(reinterpret_cast<__ubuf__ uint32_t *>(dst_imag.GetPhyAddr()),
              reinterpret_cast<__ubuf__ uint32_t *>(src.GetPhyAddr()),
              nullptr, block_len * 2 / 64, 1, 2, 8, 8);
}

/*
* @brief ????????????4x64 
*/
__aicore__ __inline__ __attribute__((always_inline)) void mask_invalid(
    AscendC::LocalTensor<float> matrix,
    AscendC::LocalTensor<float> uplo,
    uint64_t row_num, uint64_t is_real, aclblasDiagType_t diag)
{
    mul_v<ArchType::ASCEND_V220, float>(
        matrix,   // dst,
        matrix,   // src0,
        uplo,     // src1,
        row_num,  // repeat,
        1,        // dstBlockStride,
        1,        // src0BlockStride,
        1,        // src1BlockStride,
        8,        // dstRepeatStride,
        8,        // src0RepeatStride,
        8);       // src1RepeatStride
    PIPE_BARRIER(ALL);

    PIPE_BARRIER(ALL);
    if (diag == ACLBLAS_UNIT) {
        if (is_real) {
            for (uint32_t i = 0; i < row_num; ++i) {
                matrix.SetValue(64 * i + i, 1.0f);
            }
        } else {
            for (uint32_t i = 0; i < row_num; ++i) {
                matrix.SetValue(64 * i + i, 0.0f);
            }
        }
    }
    PIPE_BARRIER(ALL);
}

__aicore__ __inline__ __attribute__((always_inline)) void matrix_vector_muls_notrans(
    AscendC::LocalTensor<float> dst,
    AscendC::LocalTensor<float> src0,
    AscendC::LocalTensor<float> src1,
    int64_t m_real, int64_t n_real,
    int64_t m_real_pad, int64_t n_real_pad,
    int64_t flag)
{
    for (int64_t n_idx = 0; n_idx < n_real; ++n_idx) {
        float t = src1.GetValue(n_idx);
        PIPE_BARRIER(ALL);
        if (flag)
            t = -t;
        PIPE_BARRIER(ALL);
        AscendC::UnaryRepeatParams repeatParams{1, 1, 0, 8};
        AscendC::Axpy<float, float, true>(dst, src0[n_idx * BLOCK_DIM], t, 64, 1, repeatParams);
    }
}

__aicore__ __inline__ __attribute__((always_inline)) void complex_vmul_notrans(
    AscendC::LocalTensor<float> real_dst, AscendC::LocalTensor<float> imag_dst, AscendC::LocalTensor<float> real_src0,
    AscendC::LocalTensor<float> imag_src0, AscendC::LocalTensor<float> real_src1, AscendC::LocalTensor<float> imag_src1,
    int64_t m_real, int64_t n_real, int64_t m_real_pad, int64_t n_real_pad)
{
    matrix_vector_muls_notrans(real_dst, real_src0, real_src1, m_real, n_real, m_real_pad, n_real_pad, 0);
    matrix_vector_muls_notrans(real_dst, imag_src0, imag_src1, m_real, n_real, m_real_pad, n_real_pad, 1);
    matrix_vector_muls_notrans(imag_dst, real_src0, imag_src1, m_real, n_real, m_real_pad, n_real_pad, 0);
    matrix_vector_muls_notrans(imag_dst, imag_src0, real_src1, m_real, n_real, m_real_pad, n_real_pad, 0);
}

__aicore__ __inline__ __attribute__((always_inline)) void matrix_vector_muls_trans(
    AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> src0, AscendC::LocalTensor<float> src1, AscendC::LocalTensor<float> wksp, int64_t m_real, int64_t n_real, int64_t m_real_pad, int64_t n_real_pad,
    int64_t flag)
{
    AscendC::Duplicate<float, false>(wksp, (float)0.0, 64, m_real, 1, 8);
    PIPE_BARRIER(ALL);
    mul_v<ArchType::ASCEND_V220, float>(
        wksp, src0, src1,
        m_real,  //repeat times
        1,       // dst block stride
        1,       // src0 block stride
        1,       // src1 block stride
        8,       // dst repeat stride
        8,       // src0 repeat stride
        0        // src1 repeat stride
    );

    PIPE_BARRIER(ALL);

    if (n_real % 64) {
        AscendC::SetMaskNorm();
        SetVectorMask<float>((uint64_t)0, ((uint64_t)1 << n_real) - 1);
    }
    PIPE_BARRIER(ALL);
    cadd_v<ArchType::ASCEND_V220, float>(wksp, wksp, m_real, 1, 1, 8);  // ??4???????????
    PIPE_BARRIER(ALL);
    if (n_real % 64) {
        AscendC::SetMaskNorm();
        SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);
    }
    PIPE_BARRIER(ALL);
    if (flag) {
        sub_v<ArchType::ASCEND_V220, float>(dst, dst, wksp, 1, 1, 1, 1, 8, 8, 8);
    } else {
        add_v<ArchType::ASCEND_V220, float>(dst, dst, wksp, 1, 1, 1, 1, 8, 8, 8);
    }

    PIPE_BARRIER(ALL);
}

__aicore__ __inline__ __attribute__((always_inline)) void complex_vmul_trans(
    AscendC::LocalTensor<float> real_dst, AscendC::LocalTensor<float> imag_dst, AscendC::LocalTensor<float> real_src0,
    AscendC::LocalTensor<float> imag_src0, AscendC::LocalTensor<float> real_src1, AscendC::LocalTensor<float> imag_src1,
    AscendC::LocalTensor<float> wksp, int64_t m_real, int64_t n_real, int64_t m_real_pad, int64_t n_real_pad)
{
    matrix_vector_muls_trans(real_dst, real_src0, real_src1, wksp, m_real, n_real, m_real_pad, n_real_pad, 0);
    matrix_vector_muls_trans(real_dst, imag_src0, imag_src1, wksp, m_real, n_real, m_real_pad, n_real_pad, 1);
    matrix_vector_muls_trans(imag_dst, real_src0, imag_src1, wksp, m_real, n_real, m_real_pad, n_real_pad, 0);
    matrix_vector_muls_trans(imag_dst, imag_src0, real_src1, wksp, m_real, n_real, m_real_pad, n_real_pad, 0);
}

__aicore__ __inline__ __attribute__((always_inline)) void copy_wksp_to_x(AscendC::GlobalTensor<float>gm_X,
                                                                         AscendC::GlobalTensor<float>gm_wksp,
                                                                         uint64_t len, uint64_t inc)
{
    if (AscendC::GetBlockIdx() == 0) {
        AsdopsBuffer<ArchType::ASCEND_V220> buf;
        AscendC::LocalTensor<float> ub_tmpw = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
        AscendC::LocalTensor<float> ub_tmpx = buf.GetBuffer<BufferType::ASCEND_UB, float>(128 * 128 * 4);
        int32_t cont_tmpw = 128 * 128;
        int32_t cont_tmpx = 128 * 128;

        int32_t loop_tmpw = len * 2 / cont_tmpw;
        int32_t remain_tmpw = len * 2 % cont_tmpw;

        if (inc == 1) {
            PIPE_BARRIER(ALL);
            for (int32_t w_idx = 0; w_idx < loop_tmpw; ++w_idx) {
                gm_to_ub_align<ArchType::ASCEND_V220, float>(
                    ub_tmpw, gm_wksp[w_idx * cont_tmpw],
                    0,  // sid
                    1, cont_tmpw * sizeof(float), 0, 0, 0, 0);
                PIPE_BARRIER(ALL);
                ub_to_gm_align<ArchType::ASCEND_V220, float>(
                    gm_X[ w_idx * cont_tmpw],ub_tmpw,
                    0,  // sid
                    1, cont_tmpw * sizeof(float), 0, 0, 0, 0);
            }
            if (remain_tmpw) {
                gm_to_ub_align<ArchType::ASCEND_V220, float>(
                    ub_tmpw, gm_wksp[loop_tmpw * cont_tmpw],
                    0,  // sid
                    1, remain_tmpw * sizeof(float), 0, 0, 0, 0);
                PIPE_BARRIER(ALL);
                ub_to_gm_align<ArchType::ASCEND_V220, float>(
                    gm_X[loop_tmpw * cont_tmpw], ub_tmpw,
                    0,  // sid
                    1, remain_tmpw * sizeof(float), 0, 0, 0, 0);
                PIPE_BARRIER(ALL);
            }
        } else {
            for (int32_t w_idx = 0; w_idx < loop_tmpw; ++w_idx) {
                PIPE_BARRIER(ALL);
                gm_to_ub_align<ArchType::ASCEND_V220, float>(
                    ub_tmpw, gm_wksp[w_idx * cont_tmpw],
                    0,  // sid
                    1, cont_tmpw * sizeof(float), 0, 0, 0, 0);

                PIPE_BARRIER(ALL);
                int32_t loop_tmpx = (cont_tmpw / 2 * (int32_t)inc * 2) / cont_tmpx;
                int32_t remain_tmpx = (cont_tmpw / 2 * (int32_t)inc * 2) % cont_tmpx;
                int32_t start_posi = 0;  // ???????????
                int32_t iub_tmpw = 0;    // ???????????
                for (int32_t x_idx = 0; x_idx < loop_tmpx; ++x_idx) {
                    PIPE_BARRIER(ALL);
                    gm_to_ub_align<ArchType::ASCEND_V220, float>(
                        ub_tmpx, gm_X[w_idx * cont_tmpw * (int32_t)inc + x_idx * cont_tmpx],
                        0,  // sid
                        1, cont_tmpx * sizeof(float), 0, 0, 0, 0);
                    PIPE_BARRIER(ALL);
                    int iwhile = start_posi;
                    while (iwhile < cont_tmpx) {
                        ub_tmpx.SetValue(iwhile, ub_tmpw.GetValue(iub_tmpw));
                        ub_tmpx.SetValue(iwhile + 1, ub_tmpw.GetValue(iub_tmpw + 1));
                        iwhile = iwhile + inc * 2;
                        iub_tmpw = iub_tmpw + 2;
                    }
                    start_posi = iwhile - cont_tmpx;
                    PIPE_BARRIER(ALL);
                    ub_to_gm_align<ArchType::ASCEND_V220, float>(
                        gm_X[w_idx * cont_tmpw * inc + x_idx * cont_tmpx], ub_tmpx,
                        0,  // sid
                        1, cont_tmpx * sizeof(float), 0, 0, 0, 0);
                }
                if (remain_tmpx) {
                    PIPE_BARRIER(ALL);
                    gm_to_ub_align<ArchType::ASCEND_V220, float>(
                        ub_tmpx, gm_X[w_idx * cont_tmpw * inc + loop_tmpx * cont_tmpx],
                        0,  // sid
                        1, remain_tmpx * sizeof(float), 0, 0, 0, 0);
                    PIPE_BARRIER(ALL);
                    int iwhile = start_posi;
                    while (iub_tmpw < cont_tmpw && iwhile < cont_tmpx) {
                        ub_tmpx.SetValue(iwhile, ub_tmpw.GetValue(iub_tmpw));
                        ub_tmpx.SetValue(iwhile + 1, ub_tmpw.GetValue(iub_tmpw + 1));
                        iwhile = iwhile + inc * 2;
                        iub_tmpw = iub_tmpw + 2;
                    }
                    PIPE_BARRIER(ALL);
                    ub_to_gm_align<ArchType::ASCEND_V220, float>(
                        gm_X[w_idx * cont_tmpw * inc + loop_tmpx * cont_tmpx], ub_tmpx,
                        0,  // sid
                        1, remain_tmpx * sizeof(float), 0, 0, 0, 0);
                    PIPE_BARRIER(ALL);
                }
            }

            if (remain_tmpw) {
                PIPE_BARRIER(ALL);
                gm_to_ub_align<ArchType::ASCEND_V220, float>(
                    ub_tmpw, gm_wksp[loop_tmpw * cont_tmpw],
                    0,  // sid
                    1, remain_tmpw * sizeof(float), 0, 0, 0, 0);
                PIPE_BARRIER(ALL);
                int32_t loop_tmpx = (remain_tmpw * (int32_t)inc) / cont_tmpx;
                int32_t remain_tmpx = (remain_tmpw * (int32_t)inc) % cont_tmpx;
                int32_t start_posi = 0;  // ???????????
                int32_t iub_tmpw = 0;
                for (int32_t x_idx = 0; x_idx < loop_tmpx; ++x_idx) {
                    PIPE_BARRIER(ALL);
                    gm_to_ub_align<ArchType::ASCEND_V220, float>(
                        ub_tmpx, gm_X[loop_tmpw * cont_tmpw * inc + x_idx * cont_tmpx],
                        0,  // sid
                        1, cont_tmpx * sizeof(float), 0, 0, 0, 0);
                    PIPE_BARRIER(ALL);
                    int iwhile = start_posi;
                    while (iwhile < cont_tmpx) {
                        ub_tmpx.SetValue(iwhile, ub_tmpw.GetValue(iub_tmpw));
                        ub_tmpx.SetValue(iwhile + 1, ub_tmpw.GetValue(iub_tmpw + 1));
                        iwhile = iwhile + inc * 2;
                        iub_tmpw = iub_tmpw + 2;
                    }
                    start_posi = iwhile - cont_tmpx;

                    PIPE_BARRIER(ALL);
                    ub_to_gm_align<ArchType::ASCEND_V220, float>(
                        gm_X[loop_tmpw * cont_tmpw * inc + x_idx * cont_tmpx], ub_tmpx,
                        0,  // sid
                        1, cont_tmpx * sizeof(float), 0, 0, 0, 0);
                }
                if (remain_tmpx) {
                    PIPE_BARRIER(ALL);
                    gm_to_ub_align<ArchType::ASCEND_V220, float>(
                        ub_tmpx, gm_X[loop_tmpw * cont_tmpw * inc + loop_tmpx * cont_tmpx],
                        0,  // sid
                        1, remain_tmpx * sizeof(float), 0, 0, 0, 0);
                    PIPE_BARRIER(ALL);
                    int iwhile = start_posi;
                    while (iub_tmpw < remain_tmpw && iwhile < cont_tmpx) {
                        ub_tmpx.SetValue(iwhile, ub_tmpw.GetValue(iub_tmpw));
                        ub_tmpx.SetValue(iwhile + 1, ub_tmpw.GetValue(iub_tmpw + 1));
                        iwhile = iwhile + inc * 2;
                        iub_tmpw = iub_tmpw + 2;
                    }
                    PIPE_BARRIER(ALL);
                    ub_to_gm_align<ArchType::ASCEND_V220, float>(
                        gm_X[loop_tmpw * cont_tmpw * inc + loop_tmpx * cont_tmpx], ub_tmpx,
                        0,  // sid
                        1, remain_tmpx * sizeof(float), 0, 0, 0, 0);
                    PIPE_BARRIER(ALL);
                }
            }
        }
    }
}

__aicore__ __inline__ __attribute__((always_inline)) void ctrmv(
    AscendC::GlobalTensor<float> gm_A, AscendC::GlobalTensor<float> gm_X, AscendC::GlobalTensor<float> gm_wksp,
    AscendC::GlobalTensor<float> gm_uplo, aclblasFillMode_t mode, aclblasOperation_t trans, aclblasDiagType_t diag,
    int64_t M, int64_t lda, int64_t incx, int64_t M0)
{
    if (M0 == 0) {
        M0 = 128;
    }

    int32_t now_ub = 0;
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::LocalTensor<float> ub_a_ptr = buf.GetBuffer<BufferType::ASCEND_UB, float>(now_ub);
    now_ub += UB_MATRIX_SIZE * 2 * ELE_SIZE;  // 32KB
    AscendC::LocalTensor<float> ub_a_real_ptr = buf.GetBuffer<BufferType::ASCEND_UB, float>(now_ub);
    now_ub += UB_MATRIX_SIZE * ELE_SIZE;  // 16KB
    AscendC::LocalTensor<float> ub_a_imag_ptr = buf.GetBuffer<BufferType::ASCEND_UB, float>(now_ub);
    now_ub += UB_MATRIX_SIZE * ELE_SIZE;  // 16KB
    AscendC::LocalTensor<float> ub_x_ptr = buf.GetBuffer<BufferType::ASCEND_UB, float>(now_ub);
    now_ub += UB_VECTOR_SIZE * 2 * ELE_SIZE;  // 0.5KB
    AscendC::LocalTensor<float> ub_x_real_ptr = buf.GetBuffer<BufferType::ASCEND_UB, float>(now_ub);
    now_ub += UB_VECTOR_SIZE * ELE_SIZE;  // 0.25KB
    AscendC::LocalTensor<float> ub_x_imag_ptr = buf.GetBuffer<BufferType::ASCEND_UB, float>(now_ub);
    now_ub += UB_VECTOR_SIZE * ELE_SIZE;  // 0.25KB
    AscendC::LocalTensor<float> ub_uplo_matrix = buf.GetBuffer<BufferType::ASCEND_UB, float>(now_ub);
    now_ub += UB_MATRIX_SIZE * ELE_SIZE;  // 16KB
    int32_t now_ub_real = now_ub;
    AscendC::LocalTensor<float> ub_res_real_ptr = buf.GetBuffer<BufferType::ASCEND_UB, float>(now_ub);
    now_ub += UB_VECTOR_SIZE * ELE_SIZE;  // 0.25KB
    int32_t now_ub_imag = now_ub;
    AscendC::LocalTensor<float> ub_res_imag_ptr = buf.GetBuffer<BufferType::ASCEND_UB, float>(now_ub);
    now_ub += UB_VECTOR_SIZE * ELE_SIZE;  // 0.25KB
    AscendC::LocalTensor<float> ub_res_complex_ptr = buf.GetBuffer<BufferType::ASCEND_UB, float>(now_ub);
    now_ub += UB_VECTOR_SIZE * 2 * ELE_SIZE;  // 0.5KB
    AscendC::LocalTensor<uint32_t> ub_gather_mask_ptr = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(now_ub);
    now_ub += UB_VECTOR_SIZE * 2 * ELE_SIZE;  // 0.5KB
    AscendC::LocalTensor<float> ub_wksp_ptr = buf.GetBuffer<BufferType::ASCEND_UB, float>(now_ub);
    now_ub += UB_MATRIX_SIZE * ELE_SIZE;  // 16KB
    AscendC::LocalTensor<float> ub_carry_buf_ptr = buf.GetBuffer<BufferType::ASCEND_UB, float>(now_ub);

    int64_t m_tiles = (M + M0 - 1) / M0;
    int64_t n_tiles = 1;
    int64_t k_loop = (M + M0 - 1) / M0;
    int64_t m_remain = M % M0;
    int64_t k_remain = M % M0;

    PIPE_BARRIER(ALL);
    uint64_t offset = 0;
    for (int i = 0; i < UB_VECTOR_SIZE; ++i) {
        ub_gather_mask_ptr.SetValue(offset, static_cast<uint32_t>(now_ub_real + i * sizeof(float)));
        ++offset;
        ub_gather_mask_ptr.SetValue(offset, static_cast<uint32_t>(now_ub_imag + i * sizeof(float)));
        ++offset;
    }

    PIPE_BARRIER(ALL);

    gm_to_ub<ArchType::ASCEND_V220, float>(
        ub_uplo_matrix,         // dst
        gm_uplo,                // src
        0,                      // sid
        M0,                     // nBurst
        M0 / 8,                 // lenBurst
        0,                      // srcStride
        (BLOCK_DIM - M0) / 8);  // dstStride
    PIPE_BARRIER(ALL);

    int32_t blocks_num = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    if (blocks_num == 0) {
        blocks_num = 1;
    }
    int64_t tiles_num = m_tiles * n_tiles;
    int64_t tiles_per_core = (int64_t)tiles_num / (int64_t)blocks_num;
    int32_t block_id = AscendC::GetBlockIdx();
    if (block_id < tiles_num % blocks_num) {
        ++tiles_per_core;
    }

    int32_t btrans = trans == ACLBLAS_OP_N ? 0 : 1;
    int32_t bmode = mode == ACLBLAS_UPPER ? 1 : 0;
    int32_t bconj = trans == ACLBLAS_OP_C ? 1 : 0;

    for (int64_t tiles_idx = 0; tiles_idx < tiles_per_core; ++tiles_idx) {
        int64_t block_index = tiles_idx * blocks_num + AscendC::GetBlockIdx();
        int64_t row = block_index / n_tiles;
        int64_t m_real = M0;
        if (row == m_tiles - 1 && m_remain > 0) {
            m_real = m_remain;
        }
        int64_t m_real_pad = m_real % 8 ? (m_real & 0xfffffff8) + 8 : m_real;

        AscendC::GlobalTensor<float> gm_wksp_ptr = gm_wksp[row * M0 * 2];

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
            AscendC::GlobalTensor<float> gm_x_ptr = gm_X[M0 * incx * k_idx * 2];

            if (trans == ACLBLAS_OP_N) {
                AscendC::GlobalTensor<float> gm_a_ptr = gm_A[M0 * row * 2 + k_idx * M0 * lda * 2];

                loda_matrix_gm2ub(ub_a_ptr, gm_a_ptr, m_real, k_real, m_real_pad, k_real_pad, lda);
                PIPE_BARRIER(ALL);

                PIPE_BARRIER(ALL);
                loda_vector_gm2ub(ub_x_ptr, gm_x_ptr, ub_carry_buf_ptr, k_real, incx);
                PIPE_BARRIER(ALL);

                complex_to_real_imag(ub_a_real_ptr, ub_a_imag_ptr, ub_a_ptr, UB_MATRIX_SIZE);
                PIPE_BARRIER(ALL);

                complex_to_real_imag(ub_x_real_ptr, ub_x_imag_ptr, ub_x_ptr, UB_VECTOR_SIZE);
                PIPE_BARRIER(ALL);

                if (k_idx == row) {
                    mask_invalid(ub_a_real_ptr, ub_uplo_matrix, m_real, 1, diag);
                    mask_invalid(ub_a_imag_ptr, ub_uplo_matrix, m_real, 0, diag);
                }

                PIPE_BARRIER(ALL);
                if (k_idx == 0 || (((btrans - bmode) != 0) && k_idx == row)) {
                    AscendC::Duplicate<float, false>(ub_res_real_ptr, (float)0.0, 64, 1, 1, 8);
                    AscendC::Duplicate<float, false>(ub_res_imag_ptr, (float)0.0, 64, 1, 1, 8);
                }

                PIPE_BARRIER(ALL);
                complex_vmul_notrans(ub_res_real_ptr, ub_res_imag_ptr, ub_a_real_ptr, ub_a_imag_ptr, ub_x_real_ptr,
                                     ub_x_imag_ptr, m_real, k_real, m_real_pad, k_real_pad);
                PIPE_BARRIER(ALL);
            } else {
                AscendC::GlobalTensor<float> gm_a_ptr = gm_A[M0 * row * lda * 2 + k_idx * M0 * 2];
                PIPE_BARRIER(ALL);
                loda_matrix_gm2ub(ub_a_ptr, gm_a_ptr, k_real, m_real, k_real_pad, m_real_pad, lda);
                PIPE_BARRIER(ALL);
                loda_vector_gm2ub(ub_x_ptr, gm_x_ptr, ub_carry_buf_ptr, k_real, incx);
                PIPE_BARRIER(ALL);
                complex_to_real_imag(ub_a_real_ptr, ub_a_imag_ptr, ub_a_ptr, UB_MATRIX_SIZE);
                PIPE_BARRIER(ALL);
                complex_to_real_imag(ub_x_real_ptr, ub_x_imag_ptr, ub_x_ptr, UB_VECTOR_SIZE);
                if (k_idx == row) {
                    mask_invalid(ub_a_real_ptr, ub_uplo_matrix, m_real, 1, diag);
                    mask_invalid(ub_a_imag_ptr, ub_uplo_matrix, m_real, 0, diag);
                }
                PIPE_BARRIER(ALL);
                if (k_idx == 0 || (((btrans - bmode) != 0) && k_idx == row)) {
                    AscendC::Duplicate<float, false>(ub_res_real_ptr, (float)0.0, 64, 1, 1, 8);
                    AscendC::Duplicate<float, false>(ub_res_imag_ptr, (float)0.0, 64, 1, 1, 8);
                }
                PIPE_BARRIER(ALL);

                if (bconj) {
                    muls_v<ArchType::ASCEND_V220, float>(
                        ub_a_imag_ptr,  // dst,
                        ub_a_imag_ptr,  // src0,
                        -1.0f,          // src1,
                        m_real,         // repeat,
                        1,              // dstBlockStride,
                        1,              // srcBlockStride,
                        8,              // dstRepeatStride,
                        8);             // srcRepeatStride
                }

                PIPE_BARRIER(ALL);
                complex_vmul_trans(ub_res_real_ptr, ub_res_imag_ptr, ub_a_real_ptr, ub_a_imag_ptr, ub_x_real_ptr,
                                   ub_x_imag_ptr, ub_wksp_ptr, m_real, k_real, m_real_pad, k_real_pad);
            }
        }

        PIPE_BARRIER(ALL);
        AscendC::Gather(ub_res_complex_ptr, ub_a_ptr, ub_gather_mask_ptr, (uint32_t)0, 2 * 64);
        PIPE_BARRIER(ALL);
        store_vector_ub2gm(gm_wksp_ptr, ub_res_complex_ptr, m_real);
    }
    PIPE_BARRIER(ALL);

    FftsCrossCoreSync<PIPE_MTE3, 0>(0);
}

__global__ __aicore__ __vector__ void ctrmv(GM_ADDR gm_A, GM_ADDR gm_X, GM_ADDR gm_uplo,
                                  GM_ADDR gm_wksp, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if ASCEND_IS_AIC {
        return;
    }
    AscendC::SetAtomicNone();
    AscendC::SetMaskNorm();
    SetVectorMask<float>((uint64_t)-1, (uint64_t)-1);

    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);

    int64_t _mode = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf));
    int64_t _trans = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 8));
    int64_t _diag = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 16));

    aclblasFillMode_t mode = _mode == 0 ? ACLBLAS_LOWER : ACLBLAS_UPPER;
    aclblasOperation_t trans = ACLBLAS_OP_N;
    if (_trans == 1) {
        trans = ACLBLAS_OP_T;
    } else if (_trans == 2) {
        trans = ACLBLAS_OP_C;
    }
    aclblasDiagType_t diag = _diag == 0 ? ACLBLAS_NON_UNIT : ACLBLAS_UNIT;

    int64_t M = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 24));
    int64_t lda = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 32));
    int64_t incx = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 40));
    int64_t M0 = (*(__gm__ int64_t *)((__gm__ uint8_t *)tiling_buf + 48));

    AscendC::GlobalTensor<float> gm_A_tensor;
    AscendC::GlobalTensor<float> gm_X_tensor;
    AscendC::GlobalTensor<float> gm_wksp_tensor;
    AscendC::GlobalTensor<float> gm_uplo_tensor;
    gm_A_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_A));
    gm_X_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_X));
    gm_wksp_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_wksp));
    gm_uplo_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gm_uplo));

    PIPE_BARRIER(ALL);

    ctrmv(gm_A_tensor, gm_X_tensor, gm_wksp_tensor, gm_uplo_tensor, mode, trans, diag, M, lda, incx, M0);

    uint64_t flag_id = 0;
    WaitFlagDev(flag_id);

    copy_wksp_to_x(gm_X_tensor, gm_wksp_tensor, M, incx);
}

// Wrapper function for host to call
void ctrmv_kernel_do(GM_ADDR gm_A, GM_ADDR gm_X, GM_ADDR gm_uplo,
                     GM_ADDR gm_wksp, GM_ADDR tilingGm,
                     uint32_t numBlocks, void *stream)
{
    ctrmv<<<numBlocks, nullptr, stream>>>(gm_A, gm_X, gm_uplo, gm_wksp, tilingGm);
}
