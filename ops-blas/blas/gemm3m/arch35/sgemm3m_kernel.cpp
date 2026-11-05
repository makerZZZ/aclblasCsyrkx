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
 * \file sgemm3m_kernel.cpp
 * \brief GEMM3M kernel for arch35 (DAV_3510).
 *
 * C = alpha * (A1*B1 + A2*B2 + A3*B3) + beta * C (column-major)
 * A contains A1/A2/A3 merged along K; B contains B1/B2/B3 merged along K.
 * After swap: C^T = B1^T*A1^T + B2^T*A2^T + B3^T*A3^T (row-major)
 *
 * Cube Kernel (AIC, tensor_api): triple GEMM with L0C accumulation
 * Alpha/Beta Kernel (AIV, SIMD): post-processing C = alpha*temp + beta*C
 */

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#include "sgemm3m_kernel.h"

using namespace AscendC;
using namespace AscendC::Te;

// ============================================================================
// Constants for FP32 GEMM3M on arch35
// ============================================================================
constexpr uint64_t BASE_M = 128;
constexpr uint64_t BASE_N = 128;
constexpr uint64_t BASE_K = 64;
constexpr uint64_t K_L1 = 64;
constexpr uint64_t L1_BUF_NUM = 2;
constexpr uint64_t L1_SIZE = 512 * 1024;
constexpr uint64_t L0_HALF_SIZE = 32 * 1024;
constexpr uint64_t FIRST_BUF_SIZE = BASE_M * K_L1 * sizeof(float);
constexpr uint64_t SECOND_BUF_SIZE = K_L1 * BASE_N * sizeof(float);
constexpr uint64_t PAIR_STRIDE = FIRST_BUF_SIZE + SECOND_BUF_SIZE;
constexpr uint64_t L1_HALF_SIZE = L1_SIZE / L1_BUF_NUM;

// C0 block sizes for FP32 on arch35 (must match tensor_api layout expectations)
constexpr uint64_t FP32_C0 = 8;
constexpr uint64_t L0C_C0 = 16;

// ============================================================================
// Cube Kernel (AIC, tensor_api)
//
// Stack overflow fix: the 5 tensor_api atoms and 7 GM tensors are constructed
// ONCE per core (before the tile loop) and passed by const reference to each
// ProcessTileT call. This prevents the AIC compiler from allocating stack
// space for these objects across multiple tile iterations, which exceeded the
// ~12KB AIC stack limit when a core processed 2+ tiles.
//
// Performance optimization: L1/L0A/L0B/L0C tensor objects are still constructed
// per-tile (they depend on curM/curN), but reused via Slice inside the K-L1
// loop to eliminate Scalar-bound overhead (P9: Hot Loop 内不构造对象).
// ============================================================================

// Bundle of shared resources created once per core and reused across tiles.
// GmBT and GmAT use separate type parameters because their layout patterns
// (NDExt vs DNExt) differ depending on IS_TRANS_A/IS_TRANS_B.
template <typename CopyGM2L1T, typename CopyL12L0AT, typename CopyL12L0BT,
          typename CopyL0C2GMT, typename MmadT, typename GmBT, typename GmAT, typename GmCTT>
struct Gemm3MShared {
    CopyGM2L1T copyGM2L1Atom;
    CopyL12L0AT copyL12L0AAtom;
    CopyL12L0BT copyL12L0BAtom;
    CopyL0C2GMT copyL0C2GMAtom;
    MmadT mmadAtom;
    GmBT gmB1;
    GmBT gmB2;
    GmBT gmB3;
    GmAT gmA1;
    GmAT gmA2;
    GmAT gmA3;
    GmCTT gmCT;
};

__aicore__ inline uint64_t GetFirstOpL1Offset(uint32_t pair, uint32_t slot)
{
    return slot * L1_HALF_SIZE + pair * PAIR_STRIDE;
}

__aicore__ inline uint64_t GetSecondOpL1Offset(uint32_t pair, uint32_t slot)
{
    return GetFirstOpL1Offset(pair, slot) + FIRST_BUF_SIZE;
}

// ── L1 tensor factory helpers ──
// L1 layouts are unified regardless of transpose: the transpose is handled at
// GM→L1 via NDExt/DNExt layout choice (see MakeGmBTensor/MakeGmATensor), so
// L1→L0A/L0B is always NORMAL (no LoadDataTrait{transposed=true} needed).
// First op (B): L1 = NZ(curM, K_L1), matches L0A = NZ(curM, BASE_K).
// Second op (A): L1 = ZN(K_L1, curN), matches L0B = ZN(BASE_K, curN).
__aicore__ inline auto MakeL1BTensor(uint64_t offset, uint64_t curM)
{
    return Te::MakeTensor(Te::MakeMemPtr<Te::Location::L1, float>(offset),
        Te::MakeFrameLayout<Te::NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(curM, K_L1));
}

__aicore__ inline auto MakeL1ATensor(uint64_t offset, uint64_t curN)
{
    return Te::MakeTensor(Te::MakeMemPtr<Te::Location::L1, float>(offset),
        Te::MakeFrameLayout<Te::ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(K_L1, curN));
}

// ── GM tensor factory helpers ──
// Standard blaze pattern: trans=false → NDExtLayoutPtn (row-major read),
// trans=true → DNExtLayoutPtn (column-major read). CopyGM2L1 converts:
//   first op:  ND→NZ or DN→NZ  (both produce NZ at L1)
//   second op: ND→ZN or DN→ZN  (both produce ZN at L1)
// so L1 layout is transpose-independent and L1→L0A/L0B is always NORMAL.
template <bool IS_TRANS>
__aicore__ inline auto MakeGmBTensor(__gm__ float* ptr, uint64_t m, uint64_t lda, uint64_t k)
{
    if constexpr (IS_TRANS) {
        return Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(ptr),
            Te::MakeFrameLayout<Te::DNExtLayoutPtn>(lda, k));
    } else {
        return Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(ptr),
            Te::MakeFrameLayout<Te::NDExtLayoutPtn>(m, lda));
    }
}

template <bool IS_TRANS>
__aicore__ inline auto MakeGmATensor(__gm__ float* ptr, uint64_t k, uint64_t ldb, uint64_t n)
{
    if constexpr (IS_TRANS) {
        return Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(ptr),
            Te::MakeFrameLayout<Te::DNExtLayoutPtn>(ldb, n));
    } else {
        return Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(ptr),
            Te::MakeFrameLayout<Te::NDExtLayoutPtn>(k, ldb));
    }
}

// ============================================================================
// TileTensors: pre-constructed L0C/L0A/L0B/L1 tensors bundled for passing to
// helper functions.  Selector methods replace the former L1B_SEL/L1A_SEL macros,
// moving branch complexity out of ProcessTileT into separate methods.
// ============================================================================
template <typename L0CT, typename L0AT, typename L0BT, typename L1BT, typename L1AT>
struct TileTensors {
    L0CT l0C;
    L0AT l0A0, l0A1;
    L0BT l0B0, l0B1;
    L1BT l1B_00, l1B_01, l1B_02, l1B_10, l1B_11, l1B_12;
    L1AT l1A_00, l1A_01, l1A_02, l1A_10, l1A_11, l1A_12;

    __aicore__ inline L1BT& SelL1B(uint64_t slot, uint32_t pair)
    {
        if (slot == 0) {
            if (pair == 0) return l1B_00;
            if (pair == 1) return l1B_01;
            return l1B_02;
        }
        if (pair == 0) return l1B_10;
        if (pair == 1) return l1B_11;
        return l1B_12;
    }

    __aicore__ inline L1AT& SelL1A(uint64_t slot, uint32_t pair)
    {
        if (slot == 0) {
            if (pair == 0) return l1A_00;
            if (pair == 1) return l1A_01;
            return l1A_02;
        }
        if (pair == 0) return l1A_10;
        if (pair == 1) return l1A_11;
        return l1A_12;
    }

    __aicore__ inline L0AT& SelL0A(uint64_t bufId) { return (bufId == 0) ? l0A0 : l0A1; }
    __aicore__ inline L0BT& SelL0B(uint64_t bufId) { return (bufId == 0) ? l0B0 : l0B1; }
};

// Factory: construct all 17 tile-local tensors (1 L0C + 4 L0A/L0B + 12 L1).
// Returned by value; NRVO / Guaranteed Copy Elision avoids extra stack overhead.
__aicore__ inline auto MakeTileTensors(uint64_t curM, uint64_t curN)
{
    auto l0C = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0C, float>(0),
        Te::MakeFrameLayout<Te::NZLayoutPtn, AscendC::Std::Int<L0C_C0>>(curM, curN));
    auto l0A0 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0A, float>(0),
        Te::MakeFrameLayout<Te::NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(curM, BASE_K));
    auto l0A1 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0A, float>(L0_HALF_SIZE),
        Te::MakeFrameLayout<Te::NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(curM, BASE_K));
    auto l0B0 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0B, float>(0),
        Te::MakeFrameLayout<Te::ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(BASE_K, curN));
    auto l0B1 = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0B, float>(L0_HALF_SIZE),
        Te::MakeFrameLayout<Te::ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(BASE_K, curN));

    auto l1B_00 = MakeL1BTensor(GetFirstOpL1Offset(0, 0), curM);
    auto l1B_01 = MakeL1BTensor(GetFirstOpL1Offset(1, 0), curM);
    auto l1B_02 = MakeL1BTensor(GetFirstOpL1Offset(2, 0), curM);
    auto l1B_10 = MakeL1BTensor(GetFirstOpL1Offset(0, 1), curM);
    auto l1B_11 = MakeL1BTensor(GetFirstOpL1Offset(1, 1), curM);
    auto l1B_12 = MakeL1BTensor(GetFirstOpL1Offset(2, 1), curM);
    auto l1A_00 = MakeL1ATensor(GetSecondOpL1Offset(0, 0), curN);
    auto l1A_01 = MakeL1ATensor(GetSecondOpL1Offset(1, 0), curN);
    auto l1A_02 = MakeL1ATensor(GetSecondOpL1Offset(2, 0), curN);
    auto l1A_10 = MakeL1ATensor(GetSecondOpL1Offset(0, 1), curN);
    auto l1A_11 = MakeL1ATensor(GetSecondOpL1Offset(1, 1), curN);
    auto l1A_12 = MakeL1ATensor(GetSecondOpL1Offset(2, 1), curN);

    using L0CT = decltype(l0C);
    using L0AT = decltype(l0A0);
    using L0BT = decltype(l0B0);
    using L1BT = decltype(l1B_00);
    using L1AT = decltype(l1A_00);

    return TileTensors<L0CT, L0AT, L0BT, L1BT, L1AT>{
        l0C, l0A0, l0A1, l0B0, l0B1,
        l1B_00, l1B_01, l1B_02, l1B_10, l1B_11, l1B_12,
        l1A_00, l1A_01, l1A_02, l1A_10, l1A_11, l1A_12
    };
}

// GM -> L1 copy for one K-L1 iteration.
// Common case: pre-constructed L1 tensors + GM Slice only.
// Tail K-L1: on-the-fly construction with curKL1 (rare, once per tile).
template <typename SharedT, typename TileT>
__aicore__ inline void CopyGm2L1Tile(
    const SharedT& shared, TileT& tensors,
    uint64_t mOff, uint64_t nOff, uint64_t curM, uint64_t curN,
    uint64_t kOff, uint64_t curKL1, uint64_t l1Slot)
{
    const auto& copyGM2L1Atom = shared.copyGM2L1Atom;
    bool isTailKL1 = (curKL1 < K_L1);

    if (!isTailKL1) {
        auto gmBCoord = Te::MakeCoord(mOff, kOff);
        auto gmBShape = Te::MakeShape(curM, curKL1);
        Te::Copy(copyGM2L1Atom, tensors.SelL1B(l1Slot, 0), shared.gmB1.Slice(gmBCoord, gmBShape));
        Te::Copy(copyGM2L1Atom, tensors.SelL1B(l1Slot, 1), shared.gmB2.Slice(gmBCoord, gmBShape));
        Te::Copy(copyGM2L1Atom, tensors.SelL1B(l1Slot, 2), shared.gmB3.Slice(gmBCoord, gmBShape));

        auto gmACoord = Te::MakeCoord(kOff, nOff);
        auto gmAShape = Te::MakeShape(curKL1, curN);
        Te::Copy(copyGM2L1Atom, tensors.SelL1A(l1Slot, 0), shared.gmA1.Slice(gmACoord, gmAShape));
        Te::Copy(copyGM2L1Atom, tensors.SelL1A(l1Slot, 1), shared.gmA2.Slice(gmACoord, gmAShape));
        Te::Copy(copyGM2L1Atom, tensors.SelL1A(l1Slot, 2), shared.gmA3.Slice(gmACoord, gmAShape));
    } else {
        auto layoutB = Te::MakeFrameLayout<Te::NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(curM, curKL1);
        auto coordB = Te::MakeCoord(mOff, kOff);
        auto shapeB = Te::MakeShape(curM, curKL1);
        for (uint32_t p = 0; p < GEMM3M_NUM_PAIRS; ++p) {
            auto l1B = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L1, float>(GetFirstOpL1Offset(p, l1Slot)), layoutB);
            auto* gmB = (p == 0) ? &shared.gmB1 : (p == 1) ? &shared.gmB2 : &shared.gmB3;
            Te::Copy(copyGM2L1Atom, l1B, gmB->Slice(coordB, shapeB));
        }
        auto layoutA = Te::MakeFrameLayout<Te::ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(curKL1, curN);
        auto coordA = Te::MakeCoord(kOff, nOff);
        auto shapeA = Te::MakeShape(curKL1, curN);
        for (uint32_t p = 0; p < GEMM3M_NUM_PAIRS; ++p) {
            auto l1A = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L1, float>(GetSecondOpL1Offset(p, l1Slot)), layoutA);
            auto* gmA = (p == 0) ? &shared.gmA1 : (p == 1) ? &shared.gmA2 : &shared.gmA3;
            Te::Copy(copyGM2L1Atom, l1A, gmA->Slice(coordA, shapeA));
        }
    }
}

// L1->L0A + L1->L0B copy for one pair (common case uses pre-constructed
// tensors; tail K-L0 constructs on-the-fly with curKL0).
template <typename SharedT, typename TileT>
__aicore__ inline void CopyL1ToL0(
    const SharedT& shared, TileT& tensors,
    uint64_t curM, uint64_t curN, uint64_t curKL1, uint64_t curKL0,
    uint64_t kL0Off, uint64_t l1Slot, uint32_t pairIdx,
    bool isTailKL0, uint64_t l0BufId, uint64_t l0Off)
{
    const auto& copyL12L0AAtom = shared.copyL12L0AAtom;
    const auto& copyL12L0BAtom = shared.copyL12L0BAtom;

    if (!isTailKL0) {
        Te::Copy(copyL12L0AAtom, tensors.SelL0A(l0BufId),
            tensors.SelL1B(l1Slot, pairIdx).Slice(Te::MakeCoord(0, kL0Off), Te::MakeShape(curM, BASE_K)));
        Te::Copy(copyL12L0BAtom, tensors.SelL0B(l0BufId),
            tensors.SelL1A(l1Slot, pairIdx).Slice(Te::MakeCoord(kL0Off, 0), Te::MakeShape(BASE_K, curN)));
    } else {
        auto l0ATail = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0A, float>(l0Off),
            Te::MakeFrameLayout<Te::NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(curM, curKL0));
        auto l1BTail = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L1, float>(GetFirstOpL1Offset(pairIdx, l1Slot)),
            Te::MakeFrameLayout<Te::NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(curM, curKL1));
        Te::Copy(copyL12L0AAtom, l0ATail,
            l1BTail.Slice(Te::MakeCoord(0, kL0Off), Te::MakeShape(curM, curKL0)));
        auto l0BTail = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0B, float>(l0Off),
            Te::MakeFrameLayout<Te::ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(curKL0, curN));
        auto l1ATail = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L1, float>(GetSecondOpL1Offset(pairIdx, l1Slot)),
            Te::MakeFrameLayout<Te::ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(curKL1, curN));
        Te::Copy(copyL12L0BAtom, l0BTail,
            l1ATail.Slice(Te::MakeCoord(kL0Off, 0), Te::MakeShape(curKL0, curN)));
    }
}

// K-L0 inner loop body: CopyL1ToL0 + Mmad for one pair.
template <typename SharedT, typename TileT>
__aicore__ inline void ProcessKL0Pair(
    const SharedT& shared, TileT& tensors,
    uint64_t curM, uint64_t curN, uint64_t curKL1, uint64_t curKL0,
    uint64_t kL0Off, uint64_t l1Slot, uint32_t pairIdx,
    uint64_t iter0, uint64_t iter1, bool isTailKL0, uint64_t& l0PingPong)
{
    const auto& mmadAtom = shared.mmadAtom;

    uint64_t l0BufId = l0PingPong & 0x1;
    uint64_t l0Off = l0BufId * L0_HALF_SIZE;
    WaitFlag<HardEvent::M_MTE1>(l0BufId);

    CopyL1ToL0(shared, tensors, curM, curN, curKL1, curKL0,
               kL0Off, l1Slot, pairIdx, isTailKL0, l0BufId, l0Off);

    SetFlag<HardEvent::MTE1_M>(l0BufId);
    WaitFlag<HardEvent::MTE1_M>(l0BufId);

    bool isFirstK = (iter0 == 0 && iter1 == 0 && pairIdx == 0);
    Te::MmadParams mmadParams{
        static_cast<uint16_t>(curM), static_cast<uint16_t>(curN),
        static_cast<uint16_t>(curKL0), 0, isFirstK};

    if (!isTailKL0) {
        Te::Mmad(mmadAtom.with(mmadParams), tensors.l0C, tensors.SelL0A(l0BufId), tensors.SelL0B(l0BufId));
    } else {
        auto l0ATail = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0A, float>(l0Off),
            Te::MakeFrameLayout<Te::NZLayoutPtn, AscendC::Std::Int<FP32_C0>>(curM, curKL0));
        auto l0BTail = Te::MakeTensor(Te::MakeMemPtr<Te::Location::L0B, float>(l0Off),
            Te::MakeFrameLayout<Te::ZNLayoutPtn, AscendC::Std::Int<FP32_C0>>(curKL0, curN));
        Te::Mmad(mmadAtom.with(mmadParams), tensors.l0C, l0ATail, l0BTail);
    }

    SetFlag<HardEvent::M_MTE1>(l0BufId);
    l0PingPong++;
}

// K-L0 inner loop: iterate over K-L0 chunks and 3 pairs, delegating each pair
// to ProcessKL0Pair.
template <typename SharedT, typename TileT>
__aicore__ inline void ProcessKL0Loop(
    const SharedT& shared, TileT& tensors,
    uint64_t curM, uint64_t curN, uint64_t curKL1,
    uint64_t l1Slot, uint64_t iter0, uint64_t& l0PingPong)
{
    uint64_t kL0Iter = CeilDiv<uint64_t>(curKL1, BASE_K);
    for (uint64_t iter1 = 0; iter1 < kL0Iter; ++iter1) {
        uint64_t kL0Off = iter1 * BASE_K;
        uint64_t curKL0 = Min(curKL1 - kL0Off, BASE_K);
        bool isTailKL0 = (curKL0 < BASE_K);

        for (uint32_t pairIdx = 0; pairIdx < GEMM3M_NUM_PAIRS; ++pairIdx) {
            ProcessKL0Pair(shared, tensors, curM, curN, curKL1, curKL0,
                           kL0Off, l1Slot, pairIdx, iter0, iter1, isTailKL0, l0PingPong);
        }
    }
}

// Process one tile: K-L1 outer loop + K-L0 inner loop + L0C->GM writeback.
//
// Shared resources (atoms + GM tensors) are created ONCE per core before the
// tile loop and passed in via `shared`. Tile-local tensors (L1, L0A, L0B, L0C)
// are constructed by MakeTileTensors and bundled in a TileTensors struct.
// GM->L1 copy and K-L0 inner loop are delegated to CopyGm2L1Tile and
// ProcessKL0Loop respectively.
//
// Note: tiling values are after the column-major swap:
//   kernel's isTransA = API's isTransB, kernel's isTransB = API's isTransA
//   kernel's lda = API's ldb, kernel's ldb = API's lda
template <bool IS_TRANS_A, bool IS_TRANS_B, typename SharedT>
__aicore__ inline void ProcessTileT(
    const SharedT& shared,
    int32_t k,
    uint64_t mOff, uint64_t nOff, uint64_t curM, uint64_t curN,
    uint64_t& l0PingPong)
{
    auto tensors = MakeTileTensors(curM, curN);

    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

    uint64_t kL1Iter = CeilDiv<uint64_t>(k, K_L1);
    for (uint64_t iter0 = 0; iter0 < kL1Iter; ++iter0) {
        uint64_t l1Slot = iter0 % L1_BUF_NUM;
        uint64_t kOff = iter0 * K_L1;
        uint64_t curKL1 = Min(static_cast<uint64_t>(k) - kOff, K_L1);

        WaitFlag<HardEvent::MTE1_MTE2>(l1Slot);
        CopyGm2L1Tile(shared, tensors, mOff, nOff, curM, curN, kOff, curKL1, l1Slot);
        SetFlag<HardEvent::MTE2_MTE1>(l1Slot);
        WaitFlag<HardEvent::MTE2_MTE1>(l1Slot);

        ProcessKL0Loop(shared, tensors, curM, curN, curKL1, l1Slot, iter0, l0PingPong);

        SetFlag<HardEvent::MTE1_MTE2>(l1Slot);
    }

    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    auto gmCSlice = shared.gmCT.Slice(Te::MakeCoord(mOff, nOff), Te::MakeShape(curM, curN));
    Te::Copy(shared.copyL0C2GMAtom, gmCSlice, tensors.l0C);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
}


// Compute sub-matrix pointers from merged A and B (3 pairs each).
// b pointer (API's B, kernel's first operand):
//   IS_TRANS_A=true  (API transB=T): B is Nx(3K), column offsets with stride lda
//   IS_TRANS_A=false (API transB=N): B is (3K)xN, row offsets
// a pointer (API's A, kernel's second operand):
//   IS_TRANS_B=true  (API transA=T): A is (3K)xM, row offsets
//   IS_TRANS_B=false (API transA=N): A is Mx(3K), column offsets with stride ldb
template <bool IS_TRANS_A, bool IS_TRANS_B>
__aicore__ inline void ComputeSubMatrixPtrs(
    __gm__ float* a, __gm__ float* b, int32_t k, int32_t lda, int32_t ldb,
    __gm__ float*& b1, __gm__ float*& b2, __gm__ float*& b3,
    __gm__ float*& a1, __gm__ float*& a2, __gm__ float*& a3)
{
    if constexpr (IS_TRANS_A) {
        b1 = b;
        b2 = b + static_cast<uint64_t>(k) * lda;
        b3 = b + 2 * static_cast<uint64_t>(k) * lda;
    } else {
        b1 = b;
        b2 = b + static_cast<uint64_t>(k);
        b3 = b + 2 * static_cast<uint64_t>(k);
    }
    if constexpr (IS_TRANS_B) {
        a1 = a;
        a2 = a + static_cast<uint64_t>(k);
        a3 = a + 2 * static_cast<uint64_t>(k);
    } else {
        a1 = a;
        a2 = a + static_cast<uint64_t>(k) * ldb;
        a3 = a + 2 * static_cast<uint64_t>(k) * ldb;
    }
}

// Create shared resources (5 atoms + 7 GM tensors) ONCE per core, then run
// the tile loop. Sub-matrix pointers are computed by ComputeSubMatrixPtrs
// and baked into the GM tensors. The Gemm3MShared struct bundles them for
// passing to ProcessTileT by const reference.
template <bool IS_TRANS_A, bool IS_TRANS_B>
__aicore__ inline void ProcessAllTiles(
    __gm__ float* a, __gm__ float* b,
    __gm__ float* c, const Gemm3MTilingData& tiling,
    uint64_t mOff, uint64_t nOff, uint64_t actualM, uint64_t actualN)
{
    int32_t m = tiling.m;
    int32_t n = tiling.n;
    int32_t k = tiling.k;
    int32_t lda = tiling.lda;
    int32_t ldb = tiling.ldb;
    int32_t ldc = tiling.ldc;

    __gm__ float* b1;
    __gm__ float* b2;
    __gm__ float* b3;
    __gm__ float* a1;
    __gm__ float* a2;
    __gm__ float* a3;
    ComputeSubMatrixPtrs<IS_TRANS_A, IS_TRANS_B>(a, b, k, lda, ldb, b1, b2, b3, a1, a2, a3);

    auto copyGM2L1Atom  = Te::MakeCopy(Te::CopyGM2L1{},   Te::CopyGM2L1TraitDefault{});
    auto copyL12L0AAtom = Te::MakeCopy(Te::CopyL12L0A{},  Te::CopyL12L0ATraitDefault{});
    auto copyL12L0BAtom = Te::MakeCopy(Te::CopyL12L0B{},  Te::CopyL12L0BTraitDefault{});
    auto copyL0C2GMAtom = Te::MakeCopy(Te::CopyL0C2GM{},  Te::CopyL0C2GMTraitDefault{});
    auto mmadAtom       = Te::MakeMmad(Te::MmadOperation{}, Te::MmadTraitDefault{});

    auto gmB1 = MakeGmBTensor<IS_TRANS_A>(b1, m, lda, k);
    auto gmB2 = MakeGmBTensor<IS_TRANS_A>(b2, m, lda, k);
    auto gmB3 = MakeGmBTensor<IS_TRANS_A>(b3, m, lda, k);
    auto gmA1 = MakeGmATensor<IS_TRANS_B>(a1, k, ldb, n);
    auto gmA2 = MakeGmATensor<IS_TRANS_B>(a2, k, ldb, n);
    auto gmA3 = MakeGmATensor<IS_TRANS_B>(a3, k, ldb, n);
    auto gmCT = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(c),
                               Te::MakeFrameLayout<Te::NDExtLayoutPtn>(static_cast<uint64_t>(m), ldc));

    using SharedT = Gemm3MShared<decltype(copyGM2L1Atom), decltype(copyL12L0AAtom),
                                 decltype(copyL12L0BAtom), decltype(copyL0C2GMAtom),
                                 decltype(mmadAtom), decltype(gmB1), decltype(gmA1), decltype(gmCT)>;
    SharedT shared{copyGM2L1Atom, copyL12L0AAtom, copyL12L0BAtom, copyL0C2GMAtom, mmadAtom,
                   gmB1, gmB2, gmB3, gmA1, gmA2, gmA3, gmCT};

    uint64_t l0PingPong = 0;
    for (uint64_t mi = 0; mi < actualM; mi += BASE_M) {
        uint64_t curM = Min(actualM - mi, BASE_M);
        for (uint64_t ni = 0; ni < actualN; ni += BASE_N) {
            uint64_t curN = Min(actualN - ni, BASE_N);
            ProcessTileT<IS_TRANS_A, IS_TRANS_B>(shared, k,
                                                  mOff + mi, nOff + ni, curM, curN, l0PingPong);
        }
    }
}


// Dispatch tile processing based on runtime transpose flags
__aicore__ inline void DispatchTiles(
    __gm__ float* a, __gm__ float* b,
    __gm__ float* c, const Gemm3MTilingData& tiling,
    uint64_t mOff, uint64_t nOff, uint64_t actualM, uint64_t actualN)
{
    if (tiling.isTransA) {
        if (tiling.isTransB) {
            ProcessAllTiles<true, true>(a, b, c, tiling, mOff, nOff, actualM, actualN);
        } else {
            ProcessAllTiles<true, false>(a, b, c, tiling, mOff, nOff, actualM, actualN);
        }
    } else {
        if (tiling.isTransB) {
            ProcessAllTiles<false, true>(a, b, c, tiling, mOff, nOff, actualM, actualN);
        } else {
            ProcessAllTiles<false, false>(a, b, c, tiling, mOff, nOff, actualM, actualN);
        }
    }
}

extern "C" __global__ __aicore__ __cube__ void gemm3m_kernel(
    __gm__ uint8_t* a, __gm__ uint8_t* b,
    __gm__ uint8_t* c, Gemm3MTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::InitSocState();
    if (GetBlockIdx() >= static_cast<uint32_t>(tiling.usedCoreNum)) {
        return;
    }
    if (tiling.mBlocks == 0 || tiling.nBlocks == 0) {
        return;
    }

    uint32_t mBlockIdx = GetBlockIdx() % static_cast<uint32_t>(tiling.mBlocks);
    uint32_t nBlockIdx = GetBlockIdx() / static_cast<uint32_t>(tiling.mBlocks);
    uint64_t mOff = static_cast<uint64_t>(mBlockIdx) * tiling.singleCoreM;
    uint64_t nOff = static_cast<uint64_t>(nBlockIdx) * tiling.singleCoreN;

    // Pre-set FIX_M flag so first tile can start immediately
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    // Init L1/L0 event flags ONCE (matching matmul_fp32 reference pattern)
    for (uint32_t i = 0; i < L1_BUF_NUM; ++i) {
        SetFlag<HardEvent::MTE1_MTE2>(i);
    }
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);

    // Bounds check prevents uint64 underflow when mOff/nOff exceed tiling.m/n
    // (can happen due to singleCoreM rounding to baseM)
    uint64_t actualM = (mOff < static_cast<uint64_t>(tiling.m))
        ? Min(static_cast<uint64_t>(tiling.m) - mOff, static_cast<uint64_t>(tiling.singleCoreM))
        : 0;
    uint64_t actualN = (nOff < static_cast<uint64_t>(tiling.n))
        ? Min(static_cast<uint64_t>(tiling.n) - nOff, static_cast<uint64_t>(tiling.singleCoreN))
        : 0;

    auto aGm = reinterpret_cast<__gm__ float*>(a);
    auto bGm = reinterpret_cast<__gm__ float*>(b);
    auto cGm = reinterpret_cast<__gm__ float*>(c);

    // Shared resources (atoms + GM tensors) are created ONCE inside DispatchTiles,
    // before the tile loop. This prevents stack overflow when a core processes
    // 2+ tiles (the compiler no longer allocates stack for per-tile objects
    // across loop iterations).
    DispatchTiles(aGm, bGm, cGm, tiling, mOff, nOff, actualM, actualN);

    // Final drain: consume all pending flags (matching matmul_fp32 reference)
    for (uint32_t i = 0; i < L1_BUF_NUM; ++i) {
        WaitFlag<HardEvent::MTE1_MTE2>(i);
    }
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
}

// ============================================================================
// Alpha/Beta Vector Kernel (AIV, SIMD) — FP32 post-processing
// ============================================================================

namespace gemm3m_ab {

constexpr int32_t AB_TILE_SIZE = 4096;
constexpr int32_t AB_BUF_NUM = 2;

template <bool HAS_BETA>
class Gemm3MAlphaBetaKernel {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut,
        Gemm3MTilingData tiling, TPipe* pipe)
    {
        m_ = tiling.m;
        n_ = tiling.n;
        ldcTemp_ = tiling.ldc;
        ldcOrig_ = tiling.ldcOrig;
        alpha_ = tiling.alpha;
        beta_ = tiling.beta;
        tempGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tempAB),
                                    static_cast<uint64_t>(ldcTemp_) * n_);
        cOrigGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cOrig),
                                     static_cast<uint64_t>(ldcOrig_) * n_);
        cOutGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cOut),
                                    static_cast<uint64_t>(ldcOrig_) * n_);
        pipe->InitBuffer(tempQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        pipe->InitBuffer(outQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        if constexpr (HAS_BETA) {
            pipe->InitBuffer(cOrigQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
            pipe->InitBuffer(calcQue_, AB_BUF_NUM, AB_TILE_SIZE * sizeof(float));
        }
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        int32_t colsPerCore = (n_ + blockNum - 1) / blockNum;
        startCol_ = blockIdx * colsPerCore;
        endCol_ = Min(startCol_ + colsPerCore, n_);
    }

    __aicore__ inline void Process()
    {
        for (int32_t col = startCol_; col < endCol_; ++col) {
            for (int32_t row = 0; row < m_; row += AB_TILE_SIZE) {
                int32_t count = Min(AB_TILE_SIZE, m_ - row);
                ProcessTile(col, row, count);
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        LocalTensor<float> tempTile = tempQue_.AllocTensor<float>();
        DataCopyPad(tempTile, tempGlobal_[(uint64_t)col * ldcTemp_ + rowOffset], copyParams, padParams);
        tempQue_.EnQue(tempTile);
        LocalTensor<float> tempLocal = tempQue_.DeQue<float>();

        LocalTensor<float> result = outQue_.AllocTensor<float>();
        if constexpr (HAS_BETA) {
            LocalTensor<float> cOrigTile = cOrigQue_.AllocTensor<float>();
            DataCopyPad(cOrigTile, cOrigGlobal_[(uint64_t)col * ldcOrig_ + rowOffset], copyParams, padParams);
            cOrigQue_.EnQue(cOrigTile);
            LocalTensor<float> cOrigLocal = cOrigQue_.DeQue<float>();

            if (alpha_ == 1.0f) {
                // alpha=1: result = tempLocal + beta*C, skip redundant Muls by 1.0
                LocalTensor<float> scaledC = calcQue_.AllocTensor<float>();
                Muls(scaledC, cOrigLocal, beta_, count);
                Add(result, tempLocal, scaledC, count);
                calcQue_.FreeTensor(scaledC);
            } else {
                Muls(result, tempLocal, alpha_, count);
                LocalTensor<float> scaledC = calcQue_.AllocTensor<float>();
                Muls(scaledC, cOrigLocal, beta_, count);
                Add(result, result, scaledC, count);
                calcQue_.FreeTensor(scaledC);
            }
            cOrigQue_.FreeTensor(cOrigLocal);
        } else {
            // beta=0 (needPostProcess implies alpha!=1.0): result = alpha*tempLocal
            Muls(result, tempLocal, alpha_, count);
        }
        // Free input before EnQue(result) so the next iteration's MTE2 can start
        // (enables DoubleBuffer pipeline overlap)
        tempQue_.FreeTensor(tempLocal);
        outQue_.EnQue(result);
        LocalTensor<float> outTile = outQue_.DeQue<float>();
        DataCopyPad(cOutGlobal_[(uint64_t)col * ldcOrig_ + rowOffset], outTile, copyParams);
        outQue_.FreeTensor(outTile);
    }

    TQue<QuePosition::VECIN, AB_BUF_NUM> tempQue_;
    TQue<QuePosition::VECIN, AB_BUF_NUM> cOrigQue_;
    TQue<QuePosition::VECOUT, AB_BUF_NUM> outQue_;
    TQue<QuePosition::VECCALC, AB_BUF_NUM> calcQue_;
    GlobalTensor<float> tempGlobal_;
    GlobalTensor<float> cOrigGlobal_;
    GlobalTensor<float> cOutGlobal_;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldcTemp_ = 0;
    int32_t ldcOrig_ = 0;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
};

} // namespace gemm3m_ab

template <bool HAS_BETA>
__aicore__ inline void RunAlphaBetaKernel(
    __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut, Gemm3MTilingData tiling)
{
    TPipe pipe;
    gemm3m_ab::Gemm3MAlphaBetaKernel<HAS_BETA> op;
    op.Init(tempAB, cOrig, cOut, tiling, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void gemm3m_alpha_beta_kernel(
    __gm__ uint8_t* tempAB, __gm__ uint8_t* cOrig, __gm__ uint8_t* cOut, Gemm3MTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.hasBeta) {
        RunAlphaBetaKernel<true>(tempAB, cOrig, cOut, tiling);
    } else {
        RunAlphaBetaKernel<false>(tempAB, cOrig, cOut, tiling);
    }
}

// ============================================================================
// kernel_do implementations
// ============================================================================

void gemm3m_kernel_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR a, GM_ADDR b,
    GM_ADDR c, const Gemm3MTilingData& tilingData)
{
    gemm3m_kernel<<<numBlocks, nullptr, stream>>>(
        a, b, c, tilingData);
}

void gemm3m_alpha_beta_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR tempAB, GM_ADDR cOrig, GM_ADDR cOut,
    const Gemm3MTilingData& abTilingData)
{
    gemm3m_alpha_beta_kernel<<<numBlocks, nullptr, stream>>>(
        tempAB, cOrig, cOut, abTilingData);
}
