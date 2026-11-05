/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

// Included after cgemm_fused: reuse the validated FP32 multiply and output primitives.
namespace cgemm_onchip {
using namespace AscendC;
constexpr uint32_t TILE = CGEMM_FUSED_TILE;
constexpr uint32_t OUTPUT_BYTES = cgemm_fused::Storage<TILE>::UB_STAGE_BYTES * cgemm_fused::Storage<TILE>::STAGES;
constexpr uint32_t INPUT_PLANES = CGEMM_COMPLEX_COMPONENTS + CGEMM_THREE_PRODUCTS;
constexpr uint32_t MAX_DEPTH = (CGEMM_UB_WORKSPACE_BYTES - OUTPUT_BYTES) / (INPUT_PLANES * TILE * sizeof(float));
constexpr uint32_t PANEL = MAX_DEPTH / CGEMM_CUBE_FRACTAL * CGEMM_CUBE_FRACTAL;
constexpr uint32_t ELEMENTS = TILE * PANEL;
constexpr uint32_t BYTES = ELEMENTS * sizeof(float);
constexpr uint32_t RAW_STRIDE = CGEMM_COMPLEX_COMPONENTS * TILE;
constexpr uint32_t VECTOR_FLOATS = CGEMM_VECTOR_BYTES / sizeof(float);
constexpr uint32_t C0_SHIFT = __builtin_ctz(GEMM_FP32_C0);
constexpr uint16_t INPUT_FREE = 0;
constexpr uint16_t INPUT_READY = INPUT_FREE + CGEMM_L1_PING_PONG_STAGES;
static_assert(PANEL == TILE, "Input packing currently uses equal row and column capacity");
static_assert(OUTPUT_BYTES + INPUT_PLANES * BYTES <= CGEMM_UB_WORKSPACE_BYTES);
static_assert(CGEMM_L1_PING_PONG_STAGES * CGEMM_COMPLEX_COMPONENTS * CGEMM_THREE_PRODUCTS * BYTES <= CGEMM_L1_BYTES);

__simd_vf__ inline void ClearInput(__ubuf__ float* raw)
{
    namespace V = AscendC::MicroAPI;
    V::RegTensor<float> zero;
    V::MaskReg all = V::CreateMask<float, V::MaskPattern::ALL>();
    V::Duplicate(zero, 0.0f);
    for (uint32_t i = 0; i < CGEMM_COMPLEX_COMPONENTS * ELEMENTS; i += VECTOR_FLOATS) {
        V::StoreAlign(raw + i, zero, all);
    }
}

__simd_vf__ inline void TransformNz(__ubuf__ float* raw, __ubuf__ float* nz, uint32_t rows, float sign, bool isB)
{
    namespace V = AscendC::MicroAPI;
    V::MaskReg all = V::CreateMask<float, V::MaskPattern::ALL>();
    V::RegTensor<uint32_t> lane, block, within, bitmask, index;
    V::Arange(reinterpret_cast<V::RegTensor<int32_t>&>(lane), int32_t(0));
    V::Duplicate(bitmask, uint32_t(GEMM_FP32_C0 - 1));
    V::And(within, lane, bitmask, all);
    V::ShiftRights(block, lane, int16_t(C0_SHIFT), all);
    V::Muls(block, block, rows * GEMM_FP32_C0, all);
    V::Add(index, block, within, all);
    for (uint32_t row = 0; row < rows; ++row) {
        V::RegTensor<float> real, imag, first, second;
        V::LoadAlign<float, V::LoadDist::DIST_DINTLV_B32>(real, imag, raw + row * RAW_STRIDE);
        V::Muls(imag, imag, sign, all);
        V::Add(first, real, imag, all);
        if (isB) {
            V::Sub(second, imag, real, all);
            V::Scatter(nz + row * GEMM_FP32_C0, real, index, all);
            V::Scatter(nz + ELEMENTS + row * GEMM_FP32_C0, second, index, all);
            V::Scatter(nz + 2 * ELEMENTS + row * GEMM_FP32_C0, first, index, all);
        } else {
            V::Scatter(nz + row * GEMM_FP32_C0, first, index, all);
            V::Scatter(nz + ELEMENTS + row * GEMM_FP32_C0, real, index, all);
            V::Scatter(nz + 2 * ELEMENTS + row * GEMM_FP32_C0, imag, index, all);
        }
    }
}

__aicore__ inline DataCopyExtParams InputCopyParams(uint32_t rows, uint32_t cols, uint32_t ld)
{
    const uint32_t rowBytes = CGEMM_COMPLEX_COMPONENTS * cols * sizeof(float);
    return {
        static_cast<uint16_t>(rows), rowBytes,
        static_cast<uint32_t>((ld - cols) * CGEMM_COMPLEX_COMPONENTS * sizeof(float)),
        static_cast<uint32_t>(
            (RAW_STRIDE * sizeof(float) - RoundUp<uint32_t>(rowBytes, CGEMM_DATA_BLOCK_BYTES)) /
            CGEMM_DATA_BLOCK_BYTES),
        0};
}

__aicore__ inline void ProducePanel(
    GM_ADDR a, GM_ADDR b, const CgemmOnchipTilingData& original, const cgemm_fused::TileShape& s, uint32_t start,
    uint32_t depth, uint32_t stage)
{
    const bool isB = GetSubBlockIdx() == 0;
    const bool trans = isB ? original.transB : original.transA;
    const uint32_t extent = isB ? s.rows : s.cols;
    const uint32_t origin = isB ? s.row : s.col;
    const uint32_t ld = isB ? original.ldb : original.lda;
    const bool rowIsOutput = isB ? !trans : trans;
    const uint32_t rows = rowIsOutput ? extent : depth;
    const uint32_t cols = rowIsOutput ? depth : extent;
    const uint64_t offset =
        rowIsOutput ? static_cast<uint64_t>(origin) * ld + start : static_cast<uint64_t>(start) * ld + origin;
    LocalTensor<float> raw(TPosition::VECCALC, OUTPUT_BYTES, CGEMM_COMPLEX_COMPONENTS * ELEMENTS);
    LocalTensor<float> nz(
        TPosition::VECCALC, OUTPUT_BYTES + CGEMM_COMPLEX_COMPONENTS * BYTES, CGEMM_THREE_PRODUCTS * ELEMENTS);
    WaitFlag<HardEvent::MTE3_MTE2>(GEMM_ZERO_FLAG);
    // Only incomplete Cube fragments need zero fill before copying the input.
    if (rows % CGEMM_CUBE_FRACTAL != 0 || cols % CGEMM_CUBE_FRACTAL != 0) {
        asc_vf_call<ClearInput>(reinterpret_cast<__ubuf__ float*>(raw.GetPhyAddr()));
        SetFlag<HardEvent::V_MTE2>(GEMM_ZERO_FLAG);
        WaitFlag<HardEvent::V_MTE2>(GEMM_ZERO_FLAG);
    }
    GlobalTensor<float> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(isB ? b : a));
    const auto copy = InputCopyParams(rows, cols, ld);
    DataCopyPadExtParams<float> pad{true, 0, 0, 0.0f};
    DataCopyPad(raw, source[CGEMM_COMPLEX_COMPONENTS * offset], copy, pad);
    SetFlag<HardEvent::MTE2_V>(GEMM_ZERO_FLAG);
    WaitFlag<HardEvent::MTE2_V>(GEMM_ZERO_FLAG);
    const bool conjugate = isB ? original.conjugateB : original.conjugateA;
    asc_vf_call<TransformNz>(
        reinterpret_cast<__ubuf__ float*>(raw.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(nz.GetPhyAddr()),
        RoundUp<uint32_t>(rows, CGEMM_CUBE_FRACTAL), conjugate ? -1.0f : 1.0f, isB);
    SetFlag<HardEvent::V_MTE3>(GEMM_ZERO_FLAG);
    WaitFlag<HardEvent::V_MTE3>(GEMM_ZERO_FLAG);
    CrossCoreWaitFlag<cgemm_fused::SYNC_MODE, PIPE_MTE3>(INPUT_FREE + stage);
    const uint32_t l1Base = (stage * CGEMM_COMPLEX_COMPONENTS + (isB ? 0 : 1)) * CGEMM_THREE_PRODUCTS * BYTES;
    LocalTensor<float> destination(TPosition::A1, l1Base, CGEMM_THREE_PRODUCTS * ELEMENTS);
    DataCopy(
        destination, nz,
        DataCopyParams{1, static_cast<uint16_t>(CGEMM_THREE_PRODUCTS * BYTES / CGEMM_DATA_BLOCK_BYTES), 0, 0});
    CrossCoreSetFlag<cgemm_fused::SYNC_MODE, PIPE_MTE3>(INPUT_READY + stage);
    SetFlag<HardEvent::MTE3_MTE2>(GEMM_ZERO_FLAG);
}

template <bool TransA, bool TransB>
__aicore__ inline void MultiplyPanel(
    const cgemm_fused::TileShape& s, uint32_t depth, bool first, uint32_t stage, uint32_t& sequence)
{
    CrossCoreWaitFlag<cgemm_fused::SYNC_MODE, PIPE_MTE1>(INPUT_READY + stage);
    CrossCoreWaitFlag<cgemm_fused::SYNC_MODE, PIPE_MTE1>(INPUT_READY + stage + cgemm_fused::PEER_OFFSET);
    const uint32_t l1Base = stage * CGEMM_COMPLEX_COMPONENTS * CGEMM_THREE_PRODUCTS * BYTES;
    for (uint32_t product = 0; product < CGEMM_THREE_PRODUCTS; ++product, ++sequence) {
        LocalTensor<float> left(TPosition::A1, l1Base + product * BYTES, ELEMENTS);
        LocalTensor<float> right(TPosition::A1, l1Base + (product + CGEMM_THREE_PRODUCTS) * BYTES, ELEMENTS);
        const uint32_t slot = sequence % CGEMM_OUTPUT_STAGES;
        WaitFlag<HardEvent::M_MTE1>(slot);
        const uint32_t base = slot * CGEMM_HALF_L0_FLOAT_ELEMENTS * sizeof(float);
        LocalTensor<float> a2(TPosition::A2, base, ELEMENTS), b2(TPosition::B2, base, ELEMENTS);
        CgemmCopyInA2(a2, left, s.rows, depth, depth, TransA);
        CgemmCopyInB2(b2, right, s.cols, depth, depth, TransB);
        SetFlag<HardEvent::MTE1_M>(slot);
        WaitFlag<HardEvent::MTE1_M>(slot);
        if (first && product == 0) {
            WaitFlag<HardEvent::FIX_M>(GEMM_ZERO_FLAG);
        }
        LocalTensor<float> accum(TPosition::CO1, product * TILE * TILE * sizeof(float), TILE * TILE);
        CgemmMmad(accum, a2, b2, s.rows, s.cols, depth, first);
        SetFlag<HardEvent::M_MTE1>(slot);
    }
    CrossCoreSetFlag<cgemm_fused::SYNC_MODE, PIPE_MTE1>(INPUT_FREE + stage);
    CrossCoreSetFlag<cgemm_fused::SYNC_MODE, PIPE_MTE1>(INPUT_FREE + stage + cgemm_fused::PEER_OFFSET);
}

__aicore__ inline uint32_t InitializePipeline(uint32_t block)
{
    if ASCEND_IS_AIC {
        SetFlag<HardEvent::FIX_M>(GEMM_ZERO_FLAG);
        SetFlag<HardEvent::M_MTE1>(GEMM_ZERO_FLAG);
        SetFlag<HardEvent::M_MTE1>(GEMM_FIRST_FLAG);
        for (uint32_t stage = 0; stage < CGEMM_L1_PING_PONG_STAGES; ++stage) {
            CrossCoreSetFlag<cgemm_fused::SYNC_MODE, PIPE_MTE1>(INPUT_FREE + stage);
            CrossCoreSetFlag<cgemm_fused::SYNC_MODE, PIPE_MTE1>(INPUT_FREE + stage + cgemm_fused::PEER_OFFSET);
        }
    }
    if ASCEND_IS_AIV {
        block /= GetTaskRation();
        SetFlag<HardEvent::MTE3_MTE2>(GEMM_ZERO_FLAG);
        for (uint32_t stage = 0; stage < cgemm_fused::Storage<TILE>::STAGES; ++stage) {
            CrossCoreSetFlag<cgemm_fused::SYNC_MODE, PIPE_MTE3>(cgemm_fused::FREE_FLAG + stage);
        }
    }
    return block;
}

__aicore__ inline void DrainPipeline()
{
    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::FIX_M>(GEMM_ZERO_FLAG);
        WaitFlag<HardEvent::M_MTE1>(GEMM_ZERO_FLAG);
        WaitFlag<HardEvent::M_MTE1>(GEMM_FIRST_FLAG);
        for (uint32_t stage = 0; stage < cgemm_fused::Storage<TILE>::STAGES; ++stage) {
            CrossCoreWaitFlag<cgemm_fused::SYNC_MODE, PIPE_FIX>(cgemm_fused::FREE_FLAG + stage);
            CrossCoreWaitFlag<cgemm_fused::SYNC_MODE, PIPE_FIX>(
                cgemm_fused::FREE_FLAG + stage + cgemm_fused::PEER_OFFSET);
        }
    }
    if ASCEND_IS_AIV {
        WaitFlag<HardEvent::MTE3_MTE2>(GEMM_ZERO_FLAG);
        for (uint32_t stage = 0; stage < CGEMM_L1_PING_PONG_STAGES; ++stage) {
            CrossCoreWaitFlag<cgemm_fused::SYNC_MODE, PIPE_MTE3>(INPUT_FREE + stage);
        }
    }
}

template <bool TransA, bool TransB>
__aicore__ inline void Run(GM_ADDR a, GM_ADDR b, GM_ADDR c, const CgemmOnchipTilingData& original)
{
    GemmTilingData t{};
    t.m = original.n;
    t.n = original.m;
    t.k = original.k;
    t.ldc = original.ldc;
    t.usedCoreNum = GetBlockNum();
    t.cgemmKPartitions = 1;
    uint32_t block = GetBlockIdx();
    block = InitializePipeline(block);
    const uint64_t tiles = static_cast<uint64_t>(CeilDiv<uint32_t>(t.m, TILE)) * CeilDiv<uint32_t>(t.n, TILE);
    uint32_t sequence = 0, panelSequence = 0, iteration = 0;
    for (uint64_t tile = block; tile < tiles; tile += t.usedCoreNum, ++iteration) {
        const auto s = cgemm_fused::Shape<TILE>(t, tile);
        for (uint32_t start = 0; start < original.k; start += PANEL, ++panelSequence) {
            const uint32_t depth = Min<uint32_t>(PANEL, original.k - start);
            const uint32_t stage = panelSequence % CGEMM_L1_PING_PONG_STAGES;
            if ASCEND_IS_AIV {
                ProducePanel(a, b, original, s, start, depth, stage);
            }
            if ASCEND_IS_AIC {
                MultiplyPanel<TransA, TransB>(s, depth, start == 0, stage, sequence);
            }
        }
        const uint32_t stage = iteration % cgemm_fused::Storage<TILE>::STAGES;
        if ASCEND_IS_AIC {
            cgemm_fused::PublishProducts<TILE>(s, stage);
        }
        if ASCEND_IS_AIV {
            // Produce the next tile before consuming the previous result, overlapping input work with Cube.
            if (iteration > 0) {
                const auto previous = cgemm_fused::Shape<TILE>(t, tile - t.usedCoreNum);
                const uint32_t previousStage = (iteration - 1) % cgemm_fused::Storage<TILE>::STAGES;
                cgemm_fused::ConsumePart<TILE, true>(c, t, previous, true, true, previousStage, a, b, &original);
            }
        }
    }
    if ASCEND_IS_AIV {
        if (iteration > 0) {
            const uint64_t lastTile = block + static_cast<uint64_t>(iteration - 1) * t.usedCoreNum;
            const auto last = cgemm_fused::Shape<TILE>(t, lastTile);
            cgemm_fused::ConsumePart<TILE, true>(
                c, t, last, true, true, (iteration - 1) % cgemm_fused::Storage<TILE>::STAGES, a, b, &original);
        }
    }
    DrainPipeline();
}

extern "C" __global__ __aicore__ void gemm_cgemm_onchip_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, CgemmOnchipTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    InitSocState();
    if ASCEND_IS_AIC {
        SetHF32Mode(HF32Mode::DISABLE);
    }
    if (t.transB && t.transA) {
        Run<true, true>(a, b, c, t);
    } else if (t.transB) {
        Run<true, false>(a, b, c, t);
    } else if (t.transA) {
        Run<false, true>(a, b, c, t);
    } else {
        Run<false, false>(a, b, c, t);
    }
}
} // namespace cgemm_onchip

void gemm_cgemm_onchip_do(
    uint32_t blocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, const CgemmOnchipTilingData& t)
{
    cgemm_onchip::gemm_cgemm_onchip_kernel<<<blocks, nullptr, stream>>>(a, b, c, t);
}
