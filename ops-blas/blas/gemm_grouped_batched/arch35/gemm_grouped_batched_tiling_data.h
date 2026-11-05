/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_GROUPED_BATCHED_TILING_DATA_H
#define GEMM_GROUPED_BATCHED_TILING_DATA_H

#include <cstdint>

// FP32 alignment: 32B / sizeof(float) = 8 elements
constexpr uint32_t GEMM_GROUPED_FP32_ALIGN = 8;
constexpr uint32_t GEMM_GROUPED_UB_SIZE = 253952;  // 248KB

// Per-group parameters stored in GM, indexed by groupOffset
#pragma pack(push, 4)
struct GroupParam {
    uint32_t transa;        // 0=N, 1=T, 2=C
    uint32_t transb;        // 0=N, 1=T, 2=C
    uint32_t m;             // output rows
    uint32_t n;             // output columns
    uint32_t k;             // inner-product dimension
    int32_t  lda;           // A leading dimension (column-major)
    int32_t  ldb;           // B leading dimension (column-major)
    int32_t  ldc;           // C leading dimension (column-major)
    uint32_t groupSize;     // batch count in this group
    uint32_t groupOffset;   // flat batch index of this group's first batch
    float    alphaReal;     // alpha real part (S: alpha value; C: alpha.real)
    float    alphaImag;     // alpha imaginary part (S: 0; C: alpha.imag)
    float    betaReal;      // beta real part (S: beta value; C: beta.real)
    float    betaImag;      // beta imaginary part (S: 0; C: beta.imag)
    // Host-precomputed UB tile parameters
    uint32_t tileM;
    uint32_t tileK;
    uint32_t tileN;
    uint32_t tileM_aligned;  // CeilAlign(tileM, 8)
    uint32_t tileK_aligned;  // CeilAlign(tileK, 8)
    uint32_t tileN_aligned;  // CeilAlign(tileN, 8)
    // Host-precomputed 256B-aligned buffer sizes for this group
    uint32_t bufSizeA;
    uint32_t bufSizeB;
    uint32_t bufSizeC;
    uint32_t bufSizeCIn;
    uint32_t bufSizeMulTmp;
    uint32_t bufSizeVecTmp;
};
#pragma pack(pop)

// Global TilingData (R4: no arrays; GroupParam referenced via GM address)
#pragma pack(push, 4)
struct GemmGroupedBatchedTilingData {
    // GM addresses for per-group and pointer-array data
    uint64_t groupParamsGmAddr;   // GM address of GroupParam array
    uint64_t aPtrArrayGmAddr;     // GM address of A pointer array (uint64_t[])
    uint64_t bPtrArrayGmAddr;     // GM address of B pointer array (uint64_t[])
    uint64_t cPtrArrayGmAddr;     // GM address of C pointer array (uint64_t[])
    // Global parameters
    uint32_t groupCount;          // number of groups
    uint32_t totalBatchCount;     // sum of groupSize[g] across all groups
    uint32_t dtype;               // 0=S(FP32)
    uint32_t coreNum;             // total VectorCore count (dynamic, R2)
    uint32_t usedCoreNum;         // actually used Core count
    uint32_t batchPerCore;        // batches per Core (evenly distributed)
    uint32_t batchTail;           // remaining batches for last Core
    // Max buffer sizes across all groups (for one-shot InitBuffer)
    uint32_t maxBufSizeA;
    uint32_t maxBufSizeB;
    uint32_t maxBufSizeC;
    uint32_t maxBufSizeCIn;
    uint32_t maxBufSizeMulTmp;
    uint32_t maxBufSizeVecTmp;
};
#pragma pack(pop)

#endif  // GEMM_GROUPED_BATCHED_TILING_DATA_H
