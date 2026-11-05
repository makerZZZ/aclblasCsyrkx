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
 * \file csyrk_cube_kernel.cpp
 * \brief Csyrk 4M cube GEMM kernel for arch35 (DAV_3510), tensor_api based.
 *
 *   temp (4n x tempLdc) = [Q0|Q1|Q2|Q3] stacked quads:
 *     Q0 = Ar*Ar^T  Q1 = Ai*Ai^T  Q2 = Ar*Ai^T  Q3 = Ai*Ar^T
 *   (trans=T/C computes the transposed variant; OP_C maps to OP_T, no conjugation)
 *
 *   Each quad is a full-matrix GEMM left*right, reusing the proven
 *   SyrkGemmKernelImpl (syrk_gemm_arch35.h) with the same GM layout strategy
 *   as ssyrk (DNExt/NDExt on-the-fly transpose). Ar/Ai are column-major
 *   (Fortran, lda = arLdc) after the deinterleave phase.
 *
 *   The combine stage only consumes the uplo/lower triangle, so computing the
 *   full matrix wastes ~2x cube work; a triangular tile skip can be added later
 *   for performance.
 */

#include "kernel_operator.h"
#include "csyrk_tiling_data.h"
#include "csyrk_kernel.h"
#include "csyrk_gemm_fused.h"
#include "common/helper/syrk_gemm_arch35.h"
#include "ssyrk_tiling_data.h"

using namespace AscendC;

constexpr int64_t CSYRK_ARCH35_L1_SIZE = static_cast<int64_t>(SYRK_ARCH35_L1_SIZE_BYTES);

namespace {

// Quadrant-to-source selectors for the 4M decomposition
// (Q0 = Ar*Ar^T, Q1 = Ai*Ai^T, Q2 = Ar*Ai^T, Q3 = Ai*Ar^T):
// the left operand reads Ar for Q0/Q2 and Ai for Q1/Q3; the right operand
// reads Ar for Q0/Q3 and Ai for Q1/Q2.
__aicore__ inline bool QuadUsesArLeft(uint32_t quad) { return (quad == 0 || quad == 2); }

__aicore__ inline bool QuadUsesArRight(uint32_t quad) { return (quad == 0 || quad == 3); }

// Generic full-matrix per-quad GEMM (shared syrk helper). Used only when fewer
// than 4 AIC cores are available, where the fused all-quad driver cannot split.
template <typename TensorA, typename TensorB, typename TensorTemp>
__aicore__ inline void CsyrkProcessQuad(
    TensorA gmLeftTensor, TensorB gmRightTensor, TensorTemp gmTempTensor, const CsyrkCubeTilingData& tiling)
{
    SyrkGemmKernelImpl<TensorA, TensorB, TensorTemp, CsyrkCubeTilingData>(
        gmLeftTensor, gmRightTensor, gmTempTensor, tiling, SYRK_ARCH35_BASE_K, SYRK_ARCH35_L1_BUF_NUM,
        CSYRK_ARCH35_L1_SIZE);
}

} // namespace

// Fallback (fewer than 4 AIC cores): one core computes all four quads
// serially via the generic GEMM driver.
__aicore__ inline void CsyrkCubeFallbackQuads(
    __gm__ float* pAr, __gm__ float* pAi, __gm__ float* pTemp, const CsyrkCubeTilingData& tiling, uint32_t arLdc,
    uint32_t k, uint64_t quadStride)
{
    const uint32_t n = tiling.n;
    const auto tempLayout = te::MakeFrameLayout<te::DNExtLayoutPtn>(n, tiling.tempLdc);
    // trans=N: C = A*A^T (left X n x k col-major, right Y^T k x n);
    // trans=T/C: C = A^T*A (left X^T, right Y col-major).
    if (tiling.isTransN != 0) {
        const auto leftLayout = te::MakeFrameLayout<te::DNExtLayoutPtn>(arLdc, k);
        const auto rightLayout = te::MakeFrameLayout<te::NDExtLayoutPtn>(k, arLdc);
        for (uint32_t q = 0; q < 4; q++) {
            auto gmLeft = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(QuadUsesArLeft(q) ? pAr : pAi), leftLayout);
            auto gmRight =
                te::MakeTensor(te::MakeMemPtr<te::Location::GM>(QuadUsesArRight(q) ? pAr : pAi), rightLayout);
            auto gmTempQ = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(pTemp + q * quadStride), tempLayout);
            CsyrkProcessQuad(gmLeft, gmRight, gmTempQ, tiling);
        }
        return;
    }
    const auto leftLayout = te::MakeFrameLayout<te::NDExtLayoutPtn>(n, arLdc);
    const auto rightLayout = te::MakeFrameLayout<te::DNExtLayoutPtn>(arLdc, n);
    for (uint32_t q = 0; q < 4; q++) {
        auto gmLeft = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(QuadUsesArLeft(q) ? pAr : pAi), leftLayout);
        auto gmRight = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(QuadUsesArRight(q) ? pAr : pAi), rightLayout);
        auto gmTempQ = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(pTemp + q * quadStride), tempLayout);
        CsyrkProcessQuad(gmLeft, gmRight, gmTempQ, tiling);
    }
}

// Wrap a GM base pointer with a frame layout into a tensor handle.
template <typename TLayout>
__aicore__ inline auto CsyrkMkT(__gm__ float* base, const TLayout& layout)
{
    return te::MakeTensor(te::MakeMemPtr<te::Location::GM>(base), layout);
}

// Build the four quad panels (op(A) M-side at `left`, op(A)^T N-side at `right`)
// and run the fused driver. HF32x3 reads the residual buffers alongside ar/ai.
template <bool UseHf32, typename TLeft, typename TRight, typename TQ>
__aicore__ inline void CsyrkQuadCall(
    TLeft left, TRight right, TQ q0, TQ q1, TQ q2, TQ q3, const CsyrkCubeTilingData& tiling, __gm__ float* pAr,
    __gm__ float* pAi, __gm__ float* pArLow, __gm__ float* pAiLow, uint32_t coreIdx, uint32_t coreNum)
{
    if (UseHf32) {
        SyrkGemmQuadFusedTileKernelImpl64Hf32x3(
            CsyrkMkT(pAr, left), CsyrkMkT(pAi, left), CsyrkMkT(pArLow, left), CsyrkMkT(pAiLow, left),
            CsyrkMkT(pAr, right), CsyrkMkT(pAi, right), CsyrkMkT(pArLow, right), CsyrkMkT(pAiLow, right), q0, q1, q2,
            q3, tiling, tiling.triangleMode, coreIdx, coreNum);
    } else {
        SyrkGemmQuadFusedTileKernelImpl64(
            CsyrkMkT(pAr, left), CsyrkMkT(pAi, left), CsyrkMkT(pAr, right), CsyrkMkT(pAi, right), q0, q1, q2, q3,
            tiling, tiling.triangleMode, coreIdx, coreNum);
    }
}

// Fused 4M: single-owner 64x64 tiles, four quads back to back. UseHf32 selects
// the compensated HF32x3 driver (three mmads per quad) vs the plain fp32 one.
template <bool UseHf32>
__aicore__ inline void CsyrkCubeQuadFused(
    __gm__ float* pAr, __gm__ float* pAi, __gm__ float* pArLow, __gm__ float* pAiLow, __gm__ float* pTemp,
    const CsyrkCubeTilingData& tiling, uint32_t arLdc, uint32_t k, uint64_t quadStride, uint32_t coreIdx,
    uint32_t coreNum)
{
    const uint32_t n = tiling.n;
    const auto tempLayout = te::MakeFrameLayout<te::DNExtLayoutPtn>(n, tiling.tempLdc);
    auto q0 = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(pTemp + 0 * quadStride), tempLayout);
    auto q1 = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(pTemp + 1 * quadStride), tempLayout);
    auto q2 = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(pTemp + 2 * quadStride), tempLayout);
    auto q3 = te::MakeTensor(te::MakeMemPtr<te::Location::GM>(pTemp + 3 * quadStride), tempLayout);
    // trans=N: left = op(A) (n x k col-major); trans=T/C: left = op(A)^T.
    if (tiling.isTransN != 0) {
        CsyrkQuadCall<UseHf32>(
            te::MakeFrameLayout<te::DNExtLayoutPtn>(arLdc, k), te::MakeFrameLayout<te::NDExtLayoutPtn>(k, arLdc), q0,
            q1, q2, q3, tiling, pAr, pAi, pArLow, pAiLow, coreIdx, coreNum);
    } else {
        CsyrkQuadCall<UseHf32>(
            te::MakeFrameLayout<te::NDExtLayoutPtn>(n, arLdc), te::MakeFrameLayout<te::DNExtLayoutPtn>(arLdc, n), q0,
            q1, q2, q3, tiling, pAr, pAi, pArLow, pAiLow, coreIdx, coreNum);
    }
}

extern "C" __global__ __aicore__ void csyrk_cube_kernel(
    __gm__ uint8_t* ar, __gm__ uint8_t* ai, __gm__ uint8_t* arLow, __gm__ uint8_t* aiLow, __gm__ uint8_t* temp,
    CsyrkCubeTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    // HF32 is a global PIPE_S state that may be left enabled by the runtime / a
    // prior kernel on this core; when set, every fp32 L0A/L0B operand is rounded
    // to HF32 (10-bit mantissa) before the mmad, degrading the result. Pin the
    // mode per the tiling flag.
    AscendC::SetHF32Mode(tiling.hf32 ? AscendC::HF32Mode::ENABLE : AscendC::HF32Mode::DISABLE);
    const uint32_t n = tiling.n;
    const uint32_t k = tiling.k;
    const uint32_t arLdc = tiling.arLdc;
    if (n == 0 || k == 0 || tiling.usedCoreNum == 0) {
        return;
    }
    __gm__ float* pAr = reinterpret_cast<__gm__ float*>(ar);
    __gm__ float* pAi = reinterpret_cast<__gm__ float*>(ai);
    __gm__ float* pArLow = (arLow != nullptr) ? reinterpret_cast<__gm__ float*>(arLow) : nullptr;
    __gm__ float* pAiLow = (aiLow != nullptr) ? reinterpret_cast<__gm__ float*>(aiLow) : nullptr;
    __gm__ float* pTemp = reinterpret_cast<__gm__ float*>(temp);
    const uint64_t quadStride = static_cast<uint64_t>(n) * tiling.tempLdc;
    const uint32_t totalCores = AscendC::GetBlockNum();
    if (totalCores < 4) {
        CsyrkCubeFallbackQuads(pAr, pAi, pTemp, tiling, arLdc, k, quadStride);
        return;
    }
    const uint32_t coreIdx = AscendC::GetBlockIdx();
    if (tiling.hf32 != 0 && pArLow != nullptr) {
        CsyrkCubeQuadFused<true>(pAr, pAi, pArLow, pAiLow, pTemp, tiling, arLdc, k, quadStride, coreIdx, totalCores);
        return;
    }
    CsyrkCubeQuadFused<false>(pAr, pAi, nullptr, nullptr, pTemp, tiling, arLdc, k, quadStride, coreIdx, totalCores);
}

void csyrk_cube_kernel_do(
    GM_ADDR ar, GM_ADDR ai, GM_ADDR temp, const CsyrkCubeTilingData& tiling, uint32_t numBlocks, void* stream,
    GM_ADDR arLow, GM_ADDR aiLow)
{
    csyrk_cube_kernel<<<numBlocks, nullptr, stream>>>(ar, ai, arLow, aiLow, temp, tiling);
}
// (default args live on the declaration in csyrk_kernel.h)
