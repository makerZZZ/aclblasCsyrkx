/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "stbsv_tiling_data.h"
#include "common/helper/kernel_constant.h"

using namespace AscendC;

enum class TbsvUplo { UPPER, LOWER };
enum class TbsvTrans { NO_TRANS, TRANS };
enum class TbsvDiag { UNIT, NON_UNIT };

__aicore__ inline int64_t TbsvBandIdxUpper(uint32_t row, uint32_t col, uint32_t k, uint32_t lda)
{
    return static_cast<int64_t>(k) + static_cast<int64_t>(row) - static_cast<int64_t>(col)
           + static_cast<int64_t>(col) * static_cast<int64_t>(lda);
}

__aicore__ inline int64_t TbsvBandIdxLower(uint32_t row, uint32_t col, uint32_t lda)
{
    return static_cast<int64_t>(row) - static_cast<int64_t>(col)
           + static_cast<int64_t>(col) * static_cast<int64_t>(lda);
}

template <TbsvUplo UPLO, TbsvTrans TRANS>
__aicore__ inline int64_t TbsvBandIdx(uint32_t row, uint32_t col, uint32_t k, uint32_t lda)
{
    if constexpr (UPLO == TbsvUplo::UPPER) {
        if constexpr (TRANS == TbsvTrans::NO_TRANS) {
            return TbsvBandIdxUpper(row, col, k, lda);
        } else {
            return TbsvBandIdxUpper(col, row, k, lda);
        }
    } else {
        if constexpr (TRANS == TbsvTrans::NO_TRANS) {
            return TbsvBandIdxLower(row, col, lda);
        } else {
            return TbsvBandIdxLower(col, row, lda);
        }
    }
}

__aicore__ inline int64_t TbsvXOffset(uint32_t idx, uint32_t n, int32_t incx)
{
    if (incx >= 0) {
        return static_cast<int64_t>(idx) * static_cast<int64_t>(incx);
    } else {
        return static_cast<int64_t>(n - 1 - idx) * static_cast<int64_t>(-incx);
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
class StbsvKernel {
public:
    __aicore__ inline StbsvKernel() {}
    __aicore__ inline void Init(const StbsvTilingData& tiling);
    __aicore__ inline void Process(TPipe& pipe);

private:
    __aicore__ inline void ProcessRows(TPipe& pipe);
    __aicore__ inline float GetDiag(uint32_t row) const;
    __aicore__ inline float GetElemOffDiag(uint32_t row, uint32_t col) const;
    __aicore__ inline int64_t XOffset(uint32_t idx) const;
    __aicore__ inline void ComputeRow(uint32_t row, LocalTensor<float>& xUb);

    GlobalTensor<float> aGM;
    GlobalTensor<float> xGM;

    static constexpr bool kForward =
        (UPLO == TbsvUplo::LOWER && TRANS == TbsvTrans::NO_TRANS) ||
        (UPLO == TbsvUplo::UPPER && TRANS == TbsvTrans::TRANS);

    uint32_t n;
    uint32_t k;
    uint32_t lda;
    int32_t incx;
};

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline int64_t StbsvKernel<UPLO, TRANS, DIAG>::XOffset(uint32_t idx) const
{
    return TbsvXOffset(idx, n, incx);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::Init(const StbsvTilingData& tiling)
{
    this->n = tiling.n;
    this->k = tiling.k;
    this->lda = tiling.lda;
    this->incx = tiling.incx;

    int64_t aCount = static_cast<int64_t>(tiling.lda) * static_cast<int64_t>(n);
    int64_t absIncx = incx >= 0 ? static_cast<int64_t>(incx) : -static_cast<int64_t>(incx);
    int64_t xCount = (n > 0) ? (absIncx * static_cast<int64_t>(n - 1) + 1) : 0;
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.a), aCount);
    xGM.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tiling.x), xCount);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline float StbsvKernel<UPLO, TRANS, DIAG>::GetDiag(uint32_t row) const
{
    if constexpr (UPLO == TbsvUplo::UPPER) {
        return aGM.GetValue(static_cast<int64_t>(k) + static_cast<int64_t>(row) * static_cast<int64_t>(lda));
    } else {
        return aGM.GetValue(static_cast<int64_t>(row) * static_cast<int64_t>(lda));
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline float StbsvKernel<UPLO, TRANS, DIAG>::GetElemOffDiag(uint32_t row, uint32_t col) const
{
    return aGM.GetValue(TbsvBandIdx<UPLO, TRANS>(row, col, k, lda));
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::ComputeRow(uint32_t row, LocalTensor<float>& xUb)
{
    float sum = xGM.GetValue(XOffset(row));

    if constexpr (kForward) {
        uint32_t jStart = (row >= k) ? (row - k) : 0;
        for (uint32_t j = jStart; j < row; ++j) {
            sum -= GetElemOffDiag(row, j) * xUb.GetValue(j);
        }
    } else {
        uint32_t jEnd = (row + k + 1 < n) ? (row + k + 1) : n;
        for (uint32_t j = row + 1; j < jEnd; ++j) {
            sum -= GetElemOffDiag(row, j) * xUb.GetValue(j);
        }
    }

    if constexpr (DIAG == TbsvDiag::NON_UNIT) {
        sum = sum / GetDiag(row);
    }

    xUb.SetValue(row, sum);
    xGM.SetValue(XOffset(row), sum);
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::ProcessRows(TPipe& pipe)
{
    TBuf<TPosition::VECCALC> xUbBuf;
    pipe.InitBuffer(xUbBuf, n * sizeof(float));
    LocalTensor<float> xUb = xUbBuf.Get<float>();
    if constexpr (kForward) {
        for (uint32_t i = 0; i < n; ++i) {
            ComputeRow(i, xUb);
        }
    } else {
        for (uint32_t i = n; i-- > 0; ) {
            ComputeRow(i, xUb);
        }
    }
}

template <TbsvUplo UPLO, TbsvTrans TRANS, TbsvDiag DIAG>
__aicore__ inline void StbsvKernel<UPLO, TRANS, DIAG>::Process(TPipe& pipe)
{
    ProcessRows(pipe);
}

#define DEFINE_TBSV_KERNEL(uplo, trans, diag, name)                                         \
    __global__ __aicore__ void name(StbsvTilingData tiling)                                \
    {                                                                                       \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                                     \
        TPipe pipe;                                                                         \
        StbsvKernel<TbsvUplo::uplo, TbsvTrans::trans, TbsvDiag::diag> op;                  \
        op.Init(tiling);                                                                    \
        op.Process(pipe);                                                                   \
    }

DEFINE_TBSV_KERNEL(LOWER, NO_TRANS, NON_UNIT, stbsv_kernel_lower_no_trans_non_unit)
DEFINE_TBSV_KERNEL(LOWER, NO_TRANS, UNIT, stbsv_kernel_lower_no_trans_unit)
DEFINE_TBSV_KERNEL(UPPER, NO_TRANS, NON_UNIT, stbsv_kernel_upper_no_trans_non_unit)
DEFINE_TBSV_KERNEL(UPPER, NO_TRANS, UNIT, stbsv_kernel_upper_no_trans_unit)
DEFINE_TBSV_KERNEL(LOWER, TRANS, NON_UNIT, stbsv_kernel_lower_trans_non_unit)
DEFINE_TBSV_KERNEL(LOWER, TRANS, UNIT, stbsv_kernel_lower_trans_unit)
DEFINE_TBSV_KERNEL(UPPER, TRANS, NON_UNIT, stbsv_kernel_upper_trans_non_unit)
DEFINE_TBSV_KERNEL(UPPER, TRANS, UNIT, stbsv_kernel_upper_trans_unit)

#undef DEFINE_TBSV_KERNEL

void stbsv_simt_kernel_do(const StbsvTilingData &tiling, void* stream);

void stbsv_kernel_do(const StbsvTilingData &tiling, void* stream)
{
    if (tiling.numThreads > 0) {
        stbsv_simt_kernel_do(tiling, stream);
        return;
    }

    if (tiling.uplo == ACLBLAS_LOWER) {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                stbsv_kernel_lower_no_trans_non_unit<<<1, nullptr, stream>>>(tiling);
            } else {
                stbsv_kernel_lower_no_trans_unit<<<1, nullptr, stream>>>(tiling);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                stbsv_kernel_lower_trans_non_unit<<<1, nullptr, stream>>>(tiling);
            } else {
                stbsv_kernel_lower_trans_unit<<<1, nullptr, stream>>>(tiling);
            }
        }
    } else {
        if (tiling.trans == ACLBLAS_OP_N) {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                stbsv_kernel_upper_no_trans_non_unit<<<1, nullptr, stream>>>(tiling);
            } else {
                stbsv_kernel_upper_no_trans_unit<<<1, nullptr, stream>>>(tiling);
            }
        } else {
            if (tiling.diag == ACLBLAS_NON_UNIT) {
                stbsv_kernel_upper_trans_non_unit<<<1, nullptr, stream>>>(tiling);
            } else {
                stbsv_kernel_upper_trans_unit<<<1, nullptr, stream>>>(tiling);
            }
        }
    }
}
