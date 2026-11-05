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
 * \file matrix_transform_tiling_data.h
 * \brief Host-to-device POD tiling payload for aclblasLtMatrixTransform (arch35).
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

// Order codes shared between Host routing and Kernel branch selection.
// Values mirror aclblasLtOrder_t (COL=0, ROW=1, COL32=2, COL4_4R2_8C=3, COL32_2R_4R4=4).
constexpr uint8_t MT_ORDER_COL = 0;
constexpr uint8_t MT_ORDER_ROW = 1;
constexpr uint8_t MT_ORDER_COL32 = 2;
constexpr uint8_t MT_ORDER_COL4_4R2_8C = 3;
constexpr uint8_t MT_ORDER_COL32_2R_4R4 = 4;

// op codes: N keeps the matrix, T transposes it (C is treated as T, no complex dtype).
constexpr uint8_t MT_OP_N = 0;
constexpr uint8_t MT_OP_T = 1;

// dtype codes shared between the Host dtype mapping and the Kernel dtype-key routing. This is the
// single definition site; the Host switch, the scale-path predicates and the Kernel MT_DTYPE_LIST
// X-macro all reference these constants so the encoding cannot drift. Unsupported low-precision
// dtypes map to MT_DTYPE_INVALID and are intercepted with ACLBLAS_STATUS_NOT_SUPPORTED.
constexpr uint8_t MT_DTYPE_FP32 = 0;
constexpr uint8_t MT_DTYPE_FP16 = 1;
constexpr uint8_t MT_DTYPE_BF16 = 2;
constexpr uint8_t MT_DTYPE_INT8 = 3;
constexpr uint8_t MT_DTYPE_INT32 = 4;
constexpr uint8_t MT_DTYPE_FP8_E4M3 = 5;
constexpr uint8_t MT_DTYPE_FP8_E5M2 = 6;
constexpr uint8_t MT_DTYPE_FP4 = 7;
constexpr uint8_t MT_DTYPE_INVALID = 0xFF;

// scale path codes: the compute domain a tensor's values flow through before the output cast.
//   FP32  -> {FP32, FP16, BF16} compute in float
//   INT32 -> {INT8, INT32} compute in int32
//   FP8   -> {FP8_E4M3FN, FP8_E5M2} compute in float (1 byte/element, byte-for-byte like INT8)
//   FP4   -> {FP4_E2M1} compute in bfloat16_t (packed fp4x2, 2 elements/byte; unpack/permute/repack)
constexpr uint8_t MT_SCALE_PATH_FP32 = 0;
constexpr uint8_t MT_SCALE_PATH_INT32 = 1;
constexpr uint8_t MT_SCALE_PATH_FP8 = 2;
constexpr uint8_t MT_SCALE_PATH_FP4 = 3;

// Launch phase: the main transform, or the de-layout-only pre-pass that stages a complex op=T
// input into a column-major ND workspace (two host-synchronised launches, no device cross-core
// sync). Linear-only and complex-input-with-op=N cases run a single MT_PHASE_MAIN launch.
constexpr uint8_t MT_PHASE_MAIN = 0;
constexpr uint8_t MT_PHASE_DELAYOUT = 1;
// FP4 packed pipeline phases. FP4 (fp4x2, 2 elements/byte) cannot be moved or permuted at byte
// granularity for non-contiguous layouts, so the whole transform runs in the bf16 domain:
//   UNPACK  -> read packed fp4x2 inputs, Cast fp4x2->bf16, stage logical column-major bf16 ND inputs
//   MAIN    -> the existing bf16 engine reads the bf16 ND inputs and writes a bf16 ND output
//   REPACK  -> read the bf16 ND output, Cast bf16->fp4x2, write the packed fp4x2 C layout
// (design 1.3.A section 3.8.3). UNPACK / REPACK run on the fp4 template instance; MAIN runs on the
// bf16 instance. The phases are separate launches serialised by stream order (no device cross-core
// sync); the host synchronises once before freeing the workspaces.
constexpr uint8_t MT_PHASE_FP4_UNPACK = 2;
constexpr uint8_t MT_PHASE_FP4_REPACK = 3;

#pragma pack(push, 8)
struct alignas(8) MatrixTransformTilingData {
    // Logical dimensions of C after op (rows x cols).
    uint32_t rows{0};
    uint32_t cols{0};

    // Leading dimensions (element counts) of A / B / C.
    uint32_t lda{0};
    uint32_t ldb{0};
    uint32_t ldc{0};

    // Scalars carried as bit patterns; Kernel reinterprets per the template dtype
    // (FP32 path -> float bits, INT32 path -> int32 bits).
    uint32_t alphaBits{0};
    uint32_t betaBits{0};

    // Branch encoding consumed inside the Kernel (dtype is fixed at the template layer).
    uint8_t orderA{0};
    uint8_t orderB{0};
    uint8_t orderC{0};
    uint8_t opA{0};
    uint8_t opB{0};
    uint8_t hasB{0};          // beta != 0 and B / Bdesc valid

    // De-layout staging flags (complex input + op=T). When set, the Kernel first de-layouts the
    // complex physical input into a plain column-major ND GM workspace (op=N semantics), then the
    // main pass reads that workspace as a COL linear input and applies op=T, matching the golden
    // de-layout->applyOp(transpose) order. Single-tile transpose cannot cover the op=T cross-tile
    // remap, so the transpose is organised at the logical ND level (design 1.3.A section 4.1 step 2).
    uint8_t needDelayoutA{0};
    uint8_t needDelayoutB{0};

    // Launch phase (MT_PHASE_MAIN / MT_PHASE_DELAYOUT).
    uint8_t phase{0};
    uint8_t reserved[1]{0};   // keep struct size 8-byte aligned

    // Physical (pre-op) dimensions of A / B; the de-layout pass walks the physical extent and the
    // de-layouted ND workspace is laid out column-major with leading dimension = physical rows.
    uint32_t physRowsA{0};
    uint32_t physColsA{0};
    uint32_t physRowsB{0};
    uint32_t physColsB{0};

    // FP4 packed pipeline (used only on the FP4 scale path). The unpack / repack passes walk
    // numBlocks contiguous leading-dim blocks; packed ld (bytes) is lda / ldb / ldc, the unpacked
    // bf16 ld is 2*packedLd. fp4IsA / fp4IsB / fp4IsC flag which tensors are fp4x2.
    uint32_t fp4NumBlocksA{0};
    uint32_t fp4NumBlocksB{0};
    uint32_t fp4NumBlocksC{0};
    uint8_t fp4IsA{0};
    uint8_t fp4IsB{0};
    uint8_t fp4IsC{0};
    uint8_t reserved2[1]{0};   // keep struct size 8-byte aligned
};
#pragma pack(pop)
