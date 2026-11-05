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
 * \file complex_blas3_host_utils.h
 * \brief Host-side helpers shared by the complex BLAS-3 operators on arch22
 *        (chemm / csymm / cherk / csyrk / cher2k): scalar read-back, workspace
 *        planning for the packed/temp buffers, and GEMM tiling construction.
 */

#ifndef COMPLEX_BLAS3_HOST_UTILS_ARCH22_H
#define COMPLEX_BLAS3_HOST_UTILS_ARCH22_H

#include <algorithm>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/complex_blas3_tiling_data.h"
#include "common/helper/host_utils.h"

// Pad temp GEMM rows to 512B (128 floats) so every row start is aligned for MTE2
// burst reads.
constexpr uint32_t CBLAS3_TEMP_ALIGN = 128;
constexpr size_t CBLAS3_GM_ALIGN = 512;

// Checked 64-bit multiply and add, mirroring ssymm's TryMulU64.
//
// A buffer size is a product of two dimensions and an element size, and the
// dimensions are only bounded by the public int parameters. A dimension of
// INT_MAX already pads to 2^31, at which point d * d * sizeof(float) is exactly
// 2^64 and wraps to zero -- a zero request sails past the 2GiB cap in
// EnsureDefaultWorkspace and the kernel then writes past the end of a workspace
// sized for something far smaller. Every size below is therefore accumulated
// through these helpers, and an overflow is reported as an unsatisfiable
// allocation instead of being truncated.
static inline bool CBlas3TryMulU64(uint64_t lhs, uint64_t rhs, uint64_t* out)
{
    if (out == nullptr) {
        return false;
    }
    if (lhs == 0U || rhs == 0U) {
        *out = 0U;
        return true;
    }
    if (lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

static inline bool CBlas3TryAddU64(uint64_t lhs, uint64_t rhs, uint64_t* out)
{
    if (out == nullptr) {
        return false;
    }
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

// Round `bytes` up to the next CBLAS3_GM_ALIGN boundary, refusing the rounding
// when the carry itself would wrap.
static inline bool CBlas3TryAlignGm(uint64_t bytes, uint64_t* out)
{
    uint64_t bumped = 0U;
    if (!CBlas3TryAddU64(bytes, CBLAS3_GM_ALIGN - 1U, &bumped)) {
        return false;
    }
    *out = bumped / CBLAS3_GM_ALIGN * CBLAS3_GM_ALIGN;
    return true;
}

// A complex element is two floats wide; host-side mirror of COMPLEX_ELENUM.
constexpr uint32_t CBLAS3_CPLX_FLOATS = 2;

// Phase 0 divides one operand by 2^CBLAS3_RANGE_SCALE_EXP and Phase 2 multiplies
// the sums back. Both factors are exact powers of two, so every value that stays
// normal on the way down comes back bit-identical -- the transform exists only to
// keep the Cube's products and running sums away from the top of the fp32 range.
//
// Without it an operand near FLT_MAX overflows inside the accumulator even where
// the result is perfectly representable: on the RANDOM_EXTREME shape only 15 of
// the 342 in-range outputs survive. Eight binades of headroom cover a product of
// a FLT_MAX-sized entry with an operand up to 8 and an accumulation of a few
// hundred such terms, and recover all 342.
//
// The cost is at the bottom of the range: an input below 2^-118 lands in the
// denormals after scaling and loses mantissa bits. Anything that small only
// matters when every contribution to an output is that small, in which case the
// unscaled result underflows as well.
constexpr int CBLAS3_RANGE_SCALE_EXP = 8;
constexpr float CBLAS3_RANGE_SCALE_DOWN = 1.0F / 256.0F; // 2^-8
constexpr float CBLAS3_RANGE_SCALE_UP = 256.0F;          // 2^8

// Read alpha and beta back from device memory.
//
// Works for both the real-scalar operators (T = float: cherk, and the beta of
// cher2k) and the complex-scalar ones (T = aclblasComplex). The read is required
// because the zero/one fast paths are decided on the host, and it is the reason
// these APIs synchronise the stream once per call.
template <typename T>
static inline aclblasStatus_t CBlas3ReadScalarsFromDevice(
    const T* alpha, const T* beta, T& alphaVal, T& betaVal, aclrtStream stream, const char* opTag)
{
    aclError aclRet = aclrtMemcpyAsync(&alphaVal, sizeof(T), alpha, sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(opTag, "aclrtMemcpyAsync alpha D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtMemcpyAsync(&betaVal, sizeof(T), beta, sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(opTag, "aclrtMemcpyAsync beta D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE(opTag, "aclrtSynchronizeStream for D2H failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Logical shape of the rank-K operators' A operand, plus which GEMM operand has
// to be transposed. trans='N' keeps A as n x k and transposes the left operand;
// trans='T'/'C' stores A as k x n and transposes the right one. See the layout
// invariants in complex_blas3_arch22.h.
struct CBlas3RankKShape {
    uint32_t n;
    uint32_t k;
    uint32_t aRows;     // rows of A as stored, also the packed row stride
    uint32_t aCols;     // cols of A as stored
    uint32_t transMode; // CBlas3GemmTransMode
};

static inline CBlas3RankKShape CBlas3ResolveRankKShape(bool transposed, uint32_t n, uint32_t k)
{
    CBlas3RankKShape s{};
    s.n = n;
    s.k = k;
    if (!transposed) {
        s.aRows = n;
        s.aCols = k;
        s.transMode = CBLAS3_GEMM_TRANS_LEFT;
    } else {
        s.aRows = k;
        s.aCols = n;
        s.transMode = CBLAS3_GEMM_TRANS_RIGHT;
    }
    return s;
}

// Workspace layout for a rank-K operator: `tempCount` temp buffers of
// tempLdc x n floats, followed by `packedCount` packed real buffers of
// aRows x aCols floats.
//
// cherk:  tempCount = 4 (trans='C' path) or 2 (trans='N' interleaved path),
//         packedCount = 2 (Ar, Ai)
// csyrk:  tempCount = 2 (Pr, Pi), packedCount = 3 (P, Q, R)
// cher2k: tempCount = 4 (Mr, Mr^T, Mi, Mi^T), packedCount = 4 (Ar, Ai, Br, Bi)
struct CBlas3Workspace {
    uint8_t* temp[8] = {nullptr};
    uint8_t* packed[4] = {nullptr};
    uint32_t tempLdc = 0;
    size_t tempBytes = 0;
    size_t packedBytes = 0;
};

// Sub-buffer sizes for a rank-K operator, all of them checked. `tempStride` and
// `packedStride` are the per-buffer sizes rounded up to CBLAS3_GM_ALIGN so that
// every sub-buffer starts aligned, not only the ones whose natural size happens
// to be a multiple; `need` is the total, or 0 when the temps are skipped.
struct CBlas3RankKBytes {
    uint64_t tempBytes = 0U;
    uint64_t packedBytes = 0U;
    uint64_t tempStride = 0U;
    uint64_t packedStride = 0U;
    uint64_t need = 0U;
};

static inline bool CBlas3ComputeRankKBytes(
    const CBlas3RankKShape& shape, uint32_t tempLdc, uint32_t tempCount, uint32_t packedCount, bool skipTemp,
    CBlas3RankKBytes& out)
{
    // skipTemp means neither the temp nor the packed buffers are ever touched
    // (CBlas3PrepareWorkspace short-circuits to a 0-byte request), so the size
    // math below -- which can itself overflow for extreme n/k -- must not run
    // and must not turn an unrelated shape into a spurious ALLOC_FAILED.
    if (skipTemp) {
        out = CBlas3RankKBytes{};
        return true;
    }
    bool ok = CBlas3TryMulU64(shape.aRows, shape.aCols, &out.packedBytes) &&
              CBlas3TryMulU64(out.packedBytes, sizeof(float), &out.packedBytes) &&
              CBlas3TryMulU64(tempLdc, shape.n, &out.tempBytes) &&
              CBlas3TryMulU64(out.tempBytes, sizeof(float), &out.tempBytes) &&
              CBlas3TryAlignGm(out.tempBytes, &out.tempStride) && CBlas3TryAlignGm(out.packedBytes, &out.packedStride);
    if (!ok) {
        return false;
    }
    uint64_t tempTotal = 0U;
    uint64_t packedTotal = 0U;
    return CBlas3TryMulU64(out.tempStride, tempCount, &tempTotal) &&
           CBlas3TryMulU64(out.packedStride, packedCount, &packedTotal) &&
           CBlas3TryAddU64(tempTotal, packedTotal, &out.need);
}

static inline aclblasStatus_t CBlas3PrepareWorkspace(
    _aclblas_handle* h, const CBlas3RankKShape& shape, uint32_t tempCount, uint32_t packedCount, bool skipTemp,
    CBlas3Workspace& ws, const char* opTag)
{
    ws.tempLdc = CeilAlign<uint32_t>(shape.n, CBLAS3_TEMP_ALIGN);

    CBlas3RankKBytes bytes;
    if (!CBlas3ComputeRankKBytes(shape, ws.tempLdc, tempCount, packedCount, skipTemp, bytes)) {
        OP_LOGE(
            opTag, "workspace size overflows 64 bits: n=%u k=%u aRows=%u aCols=%u tempLdc=%u", shape.n, shape.k,
            shape.aRows, shape.aCols, ws.tempLdc);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    const uint64_t tempStride = bytes.tempStride;
    const uint64_t packedStride = bytes.packedStride;
    ws.packedBytes = static_cast<size_t>(bytes.packedBytes);
    ws.tempBytes = static_cast<size_t>(bytes.tempBytes);
    const size_t need = static_cast<size_t>(bytes.need);

    const aclblasStatus_t ret = EnsureDefaultWorkspace(h, need);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(
            opTag,
            "workspace ensure failed, required=%zu bytes; the library caps workspace at 2GiB, "
            "which for a square input limits n to about 9450",
            need);
        return ret;
    }
    if (need == 0U) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    uint8_t* cur = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    for (uint32_t i = 0; i < tempCount; ++i) {
        ws.temp[i] = cur;
        cur += tempStride;
    }
    for (uint32_t i = 0; i < packedCount; ++i) {
        ws.packed[i] = cur;
        cur += packedStride;
    }
    OP_LOGD(
        opTag, "workspace: tempBytes=%zu x%u packedBytes=%zu x%u tempLdc=%u total=%zu", ws.tempBytes, tempCount,
        ws.packedBytes, packedCount, ws.tempLdc, need);
    return ACLBLAS_STATUS_SUCCESS;
}

static inline CBlas3GemmTilingData CBlas3MakeRankKGemmTiling(
    const CBlas3RankKShape& shape, uint32_t tempLdc, uint32_t uploMode)
{
    CBlas3GemmTilingData t{};
    t.m = shape.n;
    t.n = shape.n;
    t.k = shape.k;
    t.packedLd = shape.aRows;
    t.ldc = tempLdc;
    t.mBlocks = CeilDiv<uint32_t>(shape.n, CBLAS3_TILE_M);
    t.nBlocks = CeilDiv<uint32_t>(shape.n, CBLAS3_TILE_N);
    t.uploMode = uploMode;
    t.transMode = shape.transMode;
    return t;
}

// Shape bookkeeping for the symm-family operators (chemm / csymm).
//
// A is d x d and triangular-stored, B and C are m x n; d = m for side='L' and
// d = n for side='R'. All column-major.
//
// The GEMM works on the transposed identity C^T = (A*B)^T = B^T*A^T, so its own
// extents are M = n, N = m, and K = d. Both packed operands are already the
// transposes a row-major Matmul wants, so neither is transposed -- see
// CBLAS3_GEMM_TRANS_NONE.
//
// aPadded is A's packed row stride rounded up to the expansion tile, because
// MirrorTriBody only handles whole tiles.
struct CBlas3SymmShape {
    uint32_t m;
    uint32_t n;
    uint32_t d;         // side='L' ? m : n
    uint32_t aPadded;   // d rounded up to CBLAS3_EXPAND_TILE
    uint32_t tileCount; // aPadded / CBLAS3_EXPAND_TILE
    bool sideLeft;
};

static inline CBlas3SymmShape CBlas3ResolveSymmShape(bool sideLeft, uint32_t m, uint32_t n)
{
    CBlas3SymmShape s{};
    s.m = m;
    s.n = n;
    s.d = sideLeft ? m : n;
    s.aPadded = CeilAlign<uint32_t>(s.d, CBLAS3_EXPAND_TILE);
    s.tileCount = s.aPadded / CBLAS3_EXPAND_TILE;
    s.sideLeft = sideLeft;
    return s;
}

// The four GEMMs of chemm / csymm differ only in which packed buffers they take, so
// one tiling describes all of them.
//
// side='L': left = packed B (seen as n x m, stride m), right = packed A (seen as
//           m x m, stride aPadded) -> K = m
// side='R': left = packed A (seen as n x n, stride aPadded), right = packed B (seen
//           as n x m, stride m) -> K = n
static inline CBlas3GemmTilingData CBlas3MakeSymmGemmTiling(const CBlas3SymmShape& shape, uint32_t tempLdc)
{
    CBlas3GemmTilingData t{};
    t.m = shape.n;
    t.n = shape.m;
    t.k = shape.d;
    t.ldc = tempLdc;
    t.mBlocks = CeilDiv<uint32_t>(shape.n, CBLAS3_TILE_M);
    t.nBlocks = CeilDiv<uint32_t>(shape.m, CBLAS3_TILE_N);
    t.uploMode = CBLAS3_UPLO_NONE;
    t.transMode = CBLAS3_GEMM_TRANS_NONE;
    t.ldA = shape.sideLeft ? shape.m : shape.aPadded;
    t.ldB = shape.sideLeft ? shape.aPadded : shape.m;
    t.packedLd = t.ldA;
    return t;
}

// Paired-K variant of the above (side='L' only). The contraction runs over 2*m
// slots: slot 2j carries the real parts, slot 2j+1 the imaginary ones.
//   left  = B itself, read straight from the caller's buffer. Its row i is source
//           column i, i.e. [Br_0, Bi_0, Br_1, Bi_1, ...], so the row stride is
//           twice B's leading dimension and no packing pass is needed.
//   right = one of the A buffers, whose planes sit one packed column apart, so the
//           stride between consecutive K slots is unchanged at aPadded.
static inline CBlas3GemmTilingData CBlas3MakeSymmPairedGemmTiling(
    const CBlas3SymmShape& shape, uint32_t ldb, uint32_t tempLdc)
{
    CBlas3GemmTilingData t = CBlas3MakeSymmGemmTiling(shape, tempLdc);
    t.k = shape.d * CBLAS3_CPLX_FLOATS;
    t.ldA = ldb * CBLAS3_CPLX_FLOATS;
    t.ldB = shape.aPadded;
    t.packedLd = t.ldA;
    return t;
}

// Workspace for the symm family. Two layouts, selected by CBlas3SymmUsesPairedK:
//
//   paired (side='L'): two temps and two A buffers, each holding a real and an
//       imaginary plane interleaved by packed column. B is not packed at all --
//       its source columns already carry the interleaved K order the GEMM wants.
//   plain  (side='R'): four temps, the Ar / Ai pair and the Br / Bi pair.
//
// tempLdc is the row stride of the transposed C^T output, so it spans m.
struct CBlas3SymmWorkspace {
    uint8_t* temp[4] = {nullptr};
    uint8_t* ar = nullptr;
    uint8_t* ai = nullptr;
    uint8_t* br = nullptr;
    uint8_t* bi = nullptr;
    // Paired layout only: aPair[0] = [Ar | -Ai], aPair[1] = [Ai | Ar], both
    // interleaved by packed column with the imaginary plane one column on.
    uint8_t* aPair[2] = {nullptr};
    uint32_t tempLdc = 0;
};

// Whether the paired-K layout applies. Only side='L' qualifies: there the GEMM's
// left operand is B with K running along a packed row, which is exactly how a
// column-major complex matrix already stores its real and imaginary parts, so B
// needs no packing and A only needs its two planes one packed column apart. For
// side='R' the roles swap and A -- the operand that must be mirrored out of its
// stored triangle -- would need the interleave along the fast axis, which the
// tile mirror cannot produce; that side keeps the four-GEMM layout.
static inline bool CBlas3SymmUsesPairedK(const CBlas3SymmShape& shape) { return shape.sideLeft; }

// Sub-buffer sizes for the symm family, all of them checked. Same reasoning as
// CBlas3ComputeRankKBytes: aPadded is m (or n) rounded up to the expansion tile,
// so at a dimension of INT_MAX the A plane alone is 2^31 * 2^31 * 4 == 2^64 and
// would wrap to zero. `total` is 0 when the temps are skipped.
struct CBlas3SymmBytes {
    uint64_t tempBytes = 0U;
    uint64_t aBytes = 0U;
    uint64_t bBytes = 0U;
    uint64_t total = 0U;
};

static inline bool CBlas3ComputeSymmBytes(
    const CBlas3SymmShape& shape, uint32_t tempLdc, bool skipTemp, CBlas3SymmBytes& out)
{
    // skipTemp means none of the temp/A/B buffers are ever touched
    // (CBlas3PrepareSymmWorkspace short-circuits to SUCCESS with no allocation),
    // so the size math below -- which can itself overflow for extreme m/n --
    // must not run and must not turn an unrelated shape into a spurious
    // ALLOC_FAILED.
    if (skipTemp) {
        out = CBlas3SymmBytes{};
        return true;
    }
    bool ok = CBlas3TryMulU64(tempLdc, shape.n, &out.tempBytes) &&
              CBlas3TryMulU64(out.tempBytes, sizeof(float), &out.tempBytes) &&
              CBlas3TryAlignGm(out.tempBytes, &out.tempBytes) &&
              CBlas3TryMulU64(shape.aPadded, shape.aPadded, &out.aBytes) &&
              CBlas3TryMulU64(out.aBytes, sizeof(float), &out.aBytes) && CBlas3TryAlignGm(out.aBytes, &out.aBytes) &&
              CBlas3TryMulU64(shape.m, shape.n, &out.bBytes) &&
              CBlas3TryMulU64(out.bBytes, sizeof(float), &out.bBytes) && CBlas3TryAlignGm(out.bBytes, &out.bBytes);
    if (!ok) {
        return false;
    }
    // Paired: 2 temps + 2 A buffers of twice the plain size, and no B buffers.
    // Plain: 4 temps + the Ar/Ai and Br/Bi pairs.
    const bool paired = CBlas3SymmUsesPairedK(shape);
    uint64_t tempTotal = 0U;
    uint64_t aTotal = 0U;
    uint64_t bTotal = 0U;
    return CBlas3TryMulU64(out.tempBytes, paired ? 2U : 4U, &tempTotal) &&
           CBlas3TryMulU64(out.aBytes, paired ? 4U : 2U, &aTotal) &&
           CBlas3TryMulU64(out.bBytes, paired ? 0U : 2U, &bTotal) && CBlas3TryAddU64(tempTotal, aTotal, &out.total) &&
           CBlas3TryAddU64(out.total, bTotal, &out.total);
}

static inline aclblasStatus_t CBlas3PrepareSymmWorkspace(
    _aclblas_handle* h, const CBlas3SymmShape& shape, bool skipTemp, CBlas3SymmWorkspace& ws, const char* tag)
{
    // C^T is (n x m) row-major, so its row stride spans m.
    ws.tempLdc = CeilAlign<uint32_t>(shape.m, CBLAS3_TEMP_ALIGN);

    CBlas3SymmBytes bytes;
    if (!CBlas3ComputeSymmBytes(shape, ws.tempLdc, skipTemp, bytes)) {
        OP_LOGE(
            tag, "workspace size overflows 64 bits: m=%u n=%u aPadded=%u tempLdc=%u", shape.m, shape.n, shape.aPadded,
            ws.tempLdc);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    if (skipTemp) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    // Paired: 2 temps + 2 A buffers of twice the plain size, and no B buffers.
    // Plain: 4 temps + the Ar/Ai and Br/Bi pairs.
    const bool paired = CBlas3SymmUsesPairedK(shape);
    const uint32_t tempCount = paired ? 2U : 4U;
    const size_t total = static_cast<size_t>(bytes.total);
    aclblasStatus_t ret = EnsureDefaultWorkspace(h, total);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(tag, "workspace allocation of %zu bytes failed", total);
        return ret;
    }
    uint8_t* base = static_cast<uint8_t*>(GetEffectiveWorkspace(h));
    if (base == nullptr) {
        OP_LOGE(tag, "GetEffectiveWorkspace returned nullptr");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    size_t off = 0;
    for (uint32_t i = 0; i < tempCount; ++i) {
        ws.temp[i] = base + off;
        off += bytes.tempBytes;
    }
    if (paired) {
        ws.aPair[0] = base + off;
        off += 2U * bytes.aBytes;
        ws.aPair[1] = base + off;
        return ACLBLAS_STATUS_SUCCESS;
    }
    ws.ar = base + off;
    off += bytes.aBytes;
    ws.ai = base + off;
    off += bytes.aBytes;
    ws.br = base + off;
    off += bytes.bBytes;
    ws.bi = base + off;
    return ACLBLAS_STATUS_SUCCESS;
}

// Issue one logical GEMM, splitting K so that no launch exceeds the kernel's
// compile-time singleK. The first chunk overwrites the temp buffer, later chunks
// accumulate atomically. Exceeding singleK is not diagnosed at runtime -- it
// silently mis-computes -- so the chunk size must come from the kernel itself.
template <typename LaunchFn>
static inline void CBlas3LaunchGemmChunked(
    LaunchFn launch, uint32_t blockDim, void* stream, uint8_t* left, uint8_t* right, uint8_t* out,
    CBlas3GemmTilingData tiling, uint32_t kChunk, bool forceAtomic = false)
{
    if (kChunk == 0U) {
        return;
    }
    // forceAtomic is for callers that add a second product into a temp another
    // launch already filled (CHER2K): then even the first K chunk must accumulate
    // instead of overwrite.
    for (uint32_t kBase = 0; kBase < tiling.k; kBase += kChunk) {
        tiling.kBase = kBase;
        tiling.kCount = std::min(kChunk, tiling.k - kBase);
        tiling.enAtomic = (kBase == 0U && !forceAtomic) ? 0U : 1U;
        launch(blockDim, stream, left, right, out, tiling);
    }
}

#endif // COMPLEX_BLAS3_HOST_UTILS_ARCH22_H
