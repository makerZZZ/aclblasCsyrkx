/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/devkit_version_compat.h"

#if ASC_DEVKIT_GE_9_1

#include "kernel_operator.h"
#include "csymm_kernel.h"
#include <cstdlib>

namespace csymm_prep {
using namespace AscendC;

// Max complex rows processed per vector chunk: larger chunks amortize the
// per-DataCopyPad fixed cost (UB usage stays small: in 8KB + re/im/sum 4KB).
constexpr int32_t PREP_CHUNK_ROWS = 2048;
// Deeper input queue: the per-chunk GM read DMA now overlaps the compute and
// GM write of up to PREP_BUF_NUM other chunks (the old code EnQue'd then
// immediately DeQue'd, exposing the full DMA latency per chunk).
constexpr int32_t PREP_BUF_NUM = 4;
// B columns merged into one kind-2 chunk when they are GM-adjacent (ldb==M):
// amortizes the per-chunk fixed DMA-launch cost for the B half of the prep.
// colsAvail is still capped by PREP_CHUNK_ROWS / M, so the UB footprint is
// unchanged for any value; 8 columns/chunk cuts the per-core chunk loop ~4x on
// the wide small-M shapes (256x1024 etc.) that dominated the prep phase.
constexpr int32_t PREP_B_COLS = 8;
// Rows per kind-3 (4-column mirror group) chunk; 4 complex per row keep the
// UB footprint identical to a kind-0 chunk (2048 complex = 16KB in buffer).
constexpr int32_t PREP_MIRROR_MAX_ROWS = PREP_CHUNK_ROWS / 4;
// The chunk list is streamed, never materialized: each generated descriptor is
// pushed straight into a PREP_BUF_NUM-deep ring and the oldest in-flight read
// is consumed (freeing its inQue_ slot) before a descriptor reuses its ring
// slot. The software pipeline (PREP_BUF_NUM GM reads always in flight) is
// therefore identical to the old build-then-run list, but without a fixed
// chunk array: on large S/M a core needs more chunks than the old 256-entry
// stack array held, which used to write past the array (AIV stack corruption,
// AI Core Error 81) and silently drop every column after the 256th.

struct PrepChunkDesc {
    uint32_t kind;    // 0 = contiguous interleave read, 1 = strided mirror read,
                      // 2 = packed B columns (cols x M contiguous rows in one read)
    uint32_t fromA;   // 1 = source is A (aGm_/arGm_), 0 = source is B (bGm_/brGm_)
    uint64_t srcOff;  // float offset into the source (bGm_ for B, aGm_ for A)
    int32_t rows;     // complex rows in this chunk (kind 2: cols * M)
    uint32_t dstOff;  // float offset in the (shared) re/im/sum output planes
};

class CsymmPrepKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR A, int32_t lda, int32_t uplo, GM_ADDR B, int32_t ldb,
        GM_ADDR Ar, GM_ADDR Ai, GM_ADDR Br, GM_ADDR Bi,
        GM_ADDR Aplus, GM_ADDR Bplus,
        int32_t M, int32_t N, int32_t S,
        int32_t aStart, int32_t aEnd, TPipe* pipe)
    {
        pipe_ = pipe;
        lda_ = lda;
        ldb_ = ldb;
        M_ = M;
        N_ = N;
        S_ = S;
        uplo_ = uplo;  // 121 = ACLBLAS_UPPER, 122 = ACLBLAS_LOWER
        aStart_ = aStart;
        aEnd_ = aEnd;

        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A));
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(B));
        arGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Ar));
        aiGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Ai));
        brGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Br));
        biGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Bi));
        apGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Aplus));
        bpGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Bplus));
    }

    __aicore__ inline void CsymmPrepSlice(int32_t dim, uint32_t& start, uint32_t& end)
    {
        // Split [0, dim) evenly across cores; cores whose index lands beyond the
        // range get an empty slice (start == end == 0).
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        start = 0;
        end = 0;
        if (dim > 0) {
            const uint32_t colsPer =
                (static_cast<uint32_t>(dim) + blockNum - 1) / blockNum;
            start = blockIdx * colsPer;
            if (start >= static_cast<uint32_t>(dim)) {
                start = 0;
                end = 0;
            } else {
                end = start + colsPer;
                if (end > static_cast<uint32_t>(dim)) {
                    end = static_cast<uint32_t>(dim);
                }
            }
        }
    }

    __aicore__ inline void Process()
    {
        // Columns: [0, N) are the packed matrix, [N, N+S) the symmetric
        // triangle (mirror) columns. The packed columns keep their contiguous
        // per-core slices so BuildBColumnChunks can batch multiple adjacent
        // columns into one chunk. The S triangle columns are sliced into
        // contiguous per-core ranges of their own so the mirror work spreads
        // over ALL cores: the old scheme appended [N, N+S) after the packed
        // range, which parked the whole triangle on the tail core(s) whenever
        // N was not a multiple of the core count (256x4096 put all 256 mirror
        // columns on one core -> 30us serialized).
        uint32_t bStart = 0;
        uint32_t bEnd = 0;
        uint32_t aStart = 0;
        uint32_t aEnd = 0;
        CsymmPrepSlice(N_, bStart, bEnd);
        CsymmPrepSlice(S_, aStart, aEnd);

        pipe_->InitBuffer(inQue_, PREP_BUF_NUM, PREP_CHUNK_ROWS * 2 * sizeof(float));
        pipe_->InitBuffer(reQue_, PREP_BUF_NUM, PREP_CHUNK_ROWS * sizeof(float));
        pipe_->InitBuffer(imQue_, PREP_BUF_NUM, PREP_CHUNK_ROWS * sizeof(float));
        pipe_->InitBuffer(sumQue_, PREP_BUF_NUM, PREP_CHUNK_ROWS * sizeof(float));

        // 1) Stream the per-core chunk list (packed columns first, then the
        //    core's triangle slice). A columns are filtered to [aStart_, aEnd_)
        //    so the host can split the prep into two launches (the A-col slice
        //    of the K-split GEMM). Descriptors go through Emit as they are
        //    generated, so a core whose chunk count exceeds any fixed array
        //    size still processes every column (see Emit/ConsumeOldest).
        emitted_ = 0;
        consumed_ = 0;
        uint32_t c = bStart;
        while (c < bEnd) {
            // A packed B chunk covers up to PREP_B_COLS columns: advance by the
            // number consumed so no column is handled twice.
            const uint32_t step = BuildBColumnChunks(c, bEnd);
            c += step;
        }
        for (int32_t a = static_cast<int32_t>(aStart);
             a < static_cast<int32_t>(aEnd); a++) {
            if (a >= aStart_ && a < aEnd_ && a >= mirrorGroupEnd_) {
                // Absolute A-column upper bound of this core's slice: kind-3
                // 4-column groups must not cross the slice boundary.
                const int32_t endColA = static_cast<int32_t>(aEnd);
                BuildAColumnChunks(static_cast<uint32_t>(a), endColA);
            }
        }

        // 2) Pipelined execution ran inline with the streaming above: keep up to
        //    PREP_BUF_NUM GM reads in flight so the per-chunk DMA latency
        //    overlaps the compute + GM write of the chunks already consumed.
        //    Drain whatever is still in flight.
        while (consumed_ < emitted_) {
            ConsumeOldest();
        }
    }

private:
    // Record one descriptor and launch its GM read. If PREP_BUF_NUM reads are
    // already in flight the oldest one is consumed first, which frees its
    // inQue_ slot *and* its ring slot: the ring address being reused here
    // (emitted_ % PREP_BUF_NUM) is exactly the slot of the chunk consumed by
    // that call, so no in-flight descriptor is ever overwritten.
    __aicore__ inline void Emit(uint32_t kind, uint32_t fromA, uint64_t srcOff,
                                int32_t rows, uint64_t dstOff)
    {
        if (emitted_ - consumed_ == static_cast<uint32_t>(PREP_BUF_NUM)) {
            ConsumeOldest();
        }
        PrepChunkDesc& d = ring_[emitted_ % static_cast<uint32_t>(PREP_BUF_NUM)];
        d.kind = kind;
        d.fromA = fromA;
        d.srcOff = srcOff;
        d.rows = rows;
        d.dstOff = static_cast<uint32_t>(dstOff);
        IssueRead(d);
        emitted_++;
    }

    // DeQue the oldest in-flight read and run its consume step. Only called
    // when at least one chunk is in flight, so the DeQue always has a matching
    // EnQue and never blocks on an empty queue.
    __aicore__ inline void ConsumeOldest()
    {
        LocalTensor<float> inLocal = inQue_.DeQue<float>();
        ConsumeChunk(ring_[consumed_ % static_cast<uint32_t>(PREP_BUF_NUM)], inLocal);
        inQue_.FreeTensor(inLocal);
        consumed_++;
    }

    // Issue one GM read into the input queue (async; the DMA overlaps the
    // consume of the already-issued chunks).
    __aicore__ inline void IssueRead(const PrepChunkDesc& c)
    {
        LocalTensor<float> in = inQue_.AllocTensor<float>();
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        GlobalTensor<float>& gmSrc = (c.fromA != 0) ? aGm_ : bGm_;
        if (c.kind == 2) {
            // Packed B columns: blockLen = one column (2*M floats, contiguous),
            // gap between columns = 2*(ldb - M) floats. When ldb == M the gap
            // is 0: DataCopyPad's 2D engine then only transfers the first
            // block, so fall back to one contiguous copy.
            const uint32_t colBytes = static_cast<uint32_t>(M_) * 2 * sizeof(float);
            const int32_t cols = c.rows / M_;
            if (ldb_ == M_) {
                DataCopyExtParams ext{1, colBytes * static_cast<uint32_t>(cols), 0, 0, 0};
                DataCopyPad(in, gmSrc[c.srcOff], ext, pad);
            } else {
                DataCopyExtParams ext{
                    static_cast<uint16_t>(cols), colBytes,
                    static_cast<int64_t>(2) * static_cast<int64_t>(ldb_ - M_) *
                        static_cast<int64_t>(sizeof(float)), 0, 0};
                DataCopyPad(in, gmSrc[c.srcOff], ext, pad);
            }
        } else if (c.kind == 3) {
            // 4-column mirror group: 32B per row (4 complex), GM gap 8*lda - 32.
            const int32_t gRows = c.rows / 4;
            DataCopyExtParams ext{
                static_cast<uint16_t>(gRows), 32,
                static_cast<int64_t>(8) * lda_ - 32, 0, 0};
            DataCopyPad<float, PaddingMode::Compact>(in, gmSrc[c.srcOff], ext, pad);
        } else if (c.kind == 0) {
            DataCopyExtParams ext{1, static_cast<uint32_t>(c.rows * 2 * sizeof(float)), 0, 0, 0};
            DataCopyPad(in, gmSrc[c.srcOff], ext, pad);
        } else {
            // Strided mirror read: `rows` complex elements of A's row a, packed
            // contiguously (1 complex = 8 B per block, GM gap 8*lda - 8 B).
            DataCopyExtParams ext{
                static_cast<uint16_t>(c.rows), 8,
                static_cast<int64_t>(8) * lda_ - 8, 0, 0};
            DataCopyPad<float, PaddingMode::Compact>(in, gmSrc[c.srcOff], ext, pad);
        }
        inQue_.EnQue(in);
    }

    // DeInterleave -> write re/im planes -> add -> write sum plane.
    __aicore__ inline void ConsumeChunk(const PrepChunkDesc& c, LocalTensor<float>& inLocal)
    {
        GlobalTensor<float>* gmRe = (c.fromA != 0) ? &arGm_ : &brGm_;
        GlobalTensor<float>* gmIm = (c.fromA != 0) ? &aiGm_ : &biGm_;
        GlobalTensor<float>* gmSum = (c.fromA != 0) ? &apGm_ : &bpGm_;
        LocalTensor<float> re = reQue_.AllocTensor<float>();
        LocalTensor<float> im = imQue_.AllocTensor<float>();
        DataCopyExtParams outExt{1, static_cast<uint32_t>(c.rows * sizeof(float)), 0, 0, 0};
        if (c.kind == 3) {
            // 4-column mirror group: the 4 re values of one output row
            // (columns a..a+3) are contiguous in the de-interleaved buffer but
            // land in column-major Ar/Ai/Ap at row + (a+k)*S -- S floats apart.
            // A single strided 16B-per-row write is therefore wrong: it would
            // spill across 4 rows of one column. Instead, write each of the 4
            // columns as one contiguous gRows run, reading every 4th element
            // of the de-interleaved buffer (16B source stride on MTE3).
            const int32_t gRows = c.rows / 4;
            DeInterleave(re, im, inLocal, c.rows * 2);
            reQue_.EnQue(re);
            imQue_.EnQue(im);
            LocalTensor<float> reLocal = reQue_.DeQue<float>();
            LocalTensor<float> imLocal = imQue_.DeQue<float>();
            // MTE3 strided 4B-block UB->GM writes measured ~10000x slower
            // than contiguous runs (336ms vs 50us on 256^2), so fan the
            // four interleaved columns out to contiguous UB runs first
            // (two-level DeInterleave turns the [row][col0..3] layout into
            // four [col][row] runs) and issue one contiguous write each.
            // sumQue_ provides the scratch (4 x PREP_CHUNK_ROWS floats);
            // kind3 is the only user here and chunks are consumed serially.
            // Write Ai first, then fold re+im in place and write Aplus.
            FanoutStore4(reLocal, arGm_, c.dstOff, gRows);
            FanoutStore4(imLocal, aiGm_, c.dstOff, gRows);
            Add(imLocal, reLocal, imLocal, c.rows);
            FanoutStore4(imLocal, apGm_, c.dstOff, gRows);
            reQue_.FreeTensor(reLocal);
            imQue_.FreeTensor(imLocal);
            return;
        }
        DeInterleave(re, im, inLocal, c.rows * 2);
        reQue_.EnQue(re);
        imQue_.EnQue(im);
        LocalTensor<float> reLocal = reQue_.DeQue<float>();
        LocalTensor<float> imLocal = imQue_.DeQue<float>();
        DataCopyPad((*gmRe)[c.dstOff], reLocal, outExt);
        DataCopyPad((*gmIm)[c.dstOff], imLocal, outExt);
        // Route the 3m sum plane through a VECOUT queue: the EnQue/DeQue pair
        // carries the producer->consumer flag that makes the MTE3 read of the
        // Add result safe. A raw vector->MTE3 (un-queued) dependency is a race
        // that -O0 happened to hide and -O1+ exposes (Aplus/Bplus garbage).
        LocalTensor<float> sum = sumQue_.AllocTensor<float>();
        Add(sum, reLocal, imLocal, c.rows);
        sumQue_.EnQue(sum);
        LocalTensor<float> sumLocal = sumQue_.DeQue<float>();
        DataCopyPad((*gmSum)[c.dstOff], sumLocal, outExt);
        sumQue_.FreeTensor(sumLocal);

        reQue_.FreeTensor(reLocal);
        imQue_.FreeTensor(imLocal);
    }

    // B columns -> Br, Bi, Bplus (float M x N, leading dim M).
    // B[i][c] lives at float offset 2*(i + c*ldb): one column is a contiguous
    // interleaved re/im run starting at 2*c*ldb. When ldb == M the columns
    // themselves are adjacent in GM, so packing B_PACK_COLS columns into one
    // chunk halves the GM->UB reads, the DeInterleave calls and the re/im/sum
    // writes (the de-interleaved re/im outputs of the packed columns are also
    // contiguous in the M-leading output planes).
    // Returns the number of columns consumed (>= 1) so the caller can advance
    // past the packed group and never re-process a column.
    __aicore__ inline uint32_t BuildBColumnChunks(uint32_t c, uint32_t endCol)
    {
        uint64_t outBase = static_cast<uint64_t>(c) * M_;
        uint64_t srcBase = 2ULL * static_cast<uint64_t>(c) * static_cast<uint64_t>(ldb_);
        int32_t done = 0;
        while (done < M_) {
            // How many whole columns fit in one UB buffer (rows <= chunk rows)?
            int32_t colsAvail = PREP_CHUNK_ROWS / M_;
            if (colsAvail < 1) colsAvail = 1;
            if (colsAvail > PREP_B_COLS) colsAvail = PREP_B_COLS;
            // Multi-column packing reads one whole column (2*M floats) per MTE2
            // block. The strided 2D path (ldb > M) needs a 32B-aligned
            // blockLen, but the ldb == M path is one contiguous DataCopyPad of
            // colBytes*cols, which is fine at any length (kind-0 chunks already
            // move 2*M floats regardless of alignment). Restrict the fallback
            // to the strided path only so odd M (90/166/211/...) can pack too.
            if (ldb_ != M_ && (static_cast<uint32_t>(M_) * 2 * sizeof(float)) % 32 != 0) {
                colsAvail = 1;
            }
            const int32_t colsInCore = static_cast<int32_t>(endCol) - static_cast<int32_t>(c);
            if (colsAvail > colsInCore) colsAvail = colsInCore;
            // Never pack across the B/A boundary: colsInCore counts the A
            // columns that follow B in this core's range, so a pack starting
            // at the last B column would read A's GM as "B column N" (OOB read
            // past B) and skip the first A column entirely.
            const int32_t colsInB = static_cast<int32_t>(N_) - static_cast<int32_t>(c);
            if (colsAvail > colsInB) colsAvail = colsInB;
            int32_t rows = M_ - done;
            if (rows > PREP_CHUNK_ROWS) rows = PREP_CHUNK_ROWS;
            if (done == 0 && colsAvail > 1 && rows == M_) {
                // Pack `colsAvail` whole columns into one chunk (kind 2).
                Emit(2, 0, srcBase, colsAvail * M_, outBase);
                return static_cast<uint32_t>(colsAvail);
            }
            Emit(0, 0, srcBase + static_cast<uint64_t>(done) * 2, rows,
                 outBase + static_cast<uint64_t>(done));
            done += rows;
        }
        return 1;
    }

    // A column a (symmetric, complex S x S, leading dim lda) -> Ar, Ai
    // (float S x S, leading dim S). Only the uplo-referenced triangle is stored
    // in the input; the implied triangle is mirrored (A symmetric, no conj):
    //   UPPER (121): rows i<=a read A[i][a] (contiguous); rows i>a read A[a][i].
    //   LOWER (122): rows i<a read A[a][i]; rows i>=a read A[i][a] (contiguous).
    __aicore__ inline void BuildAColumnChunks(uint32_t a, int32_t endColA)
    {
        bool upper = (uplo_ == 121);
        uint64_t outBase = static_cast<uint64_t>(a) * S_;
        int32_t S32 = static_cast<int32_t>(S_);
        int32_t a32 = static_cast<int32_t>(a);
        if (upper) {
            // Kind-3 (4-column 32B/row read merge) pays off when each group's
            // mirror run is long (large S); on S <= 512 the fan-out's queue
            // round-trips outweigh the read savings, so small triangles keep
            // the per-column kind-1 path (verified on 256/222-S rows).
            if (S32 >= 320 && a32 % 4 == 0 && a32 + 4 <= S32 && a32 + 4 <= aEnd_ &&
                a32 + 4 <= endColA) {
                // 4-column mirror group: the four stored columns a..a+3 are
                // emitted as contiguous kind-0 chunks and their shared mirror
                // rows as strided kind-1 chunks (one per column).
                BuildAMirrorGroup(a32);
                return;
            }
            // stored: rows [0, a], physical column a A[i][a] (i=0..a, contiguous
            // in column-major; lives in the stored UPPER triangle).
            PushAChunk(0, 2ULL * static_cast<uint64_t>(a) * static_cast<uint64_t>(lda_),
                       a32 + 1, outBase);
            // mirrored: rows [a+1, S-1], symmetric mirror A[a][i] (i > a lies in
            // the stored UPPER triangle at row a). Column-major, so consecutive
            // i are lda complex elements apart: strided kind-1 read.
            PushAChunk(1,
                       2ULL * (static_cast<uint64_t>(a) +
                               static_cast<uint64_t>(a32 + 1) * static_cast<uint64_t>(lda_)),
                       S32 - a32 - 1, outBase + a + 1);
        } else {
            if (S32 >= 320 && a32 % 4 == 0 && a32 > 0 && a32 + 4 <= S32 && a32 + 4 <= aEnd_ &&
                a32 + 4 <= endColA) {
                // 4-column LOWER mirror group (mirror rows are the prefix
                // [0, a), shared by all four columns; kind-3 fan-out read).
                BuildAMirrorGroupLower(a32);
                return;
            }
            // mirrored: rows [0, a-1], symmetric mirror A[a][i] (i < a lies in
            // the stored LOWER triangle at row a). Column-major, so consecutive
            // i are lda complex elements apart: strided kind-1 read.
            PushAChunk(1,
                       2ULL * static_cast<uint64_t>(a),
                       a32, outBase);
            // stored: rows [a, S-1], source 2*(a + a*lda) (contiguous)
            PushAChunk(0,
                       2ULL * (static_cast<uint64_t>(a) +
                       static_cast<uint64_t>(a) * static_cast<uint64_t>(lda_)),
                       S32 - a32, outBase + a);
        }
    }

    // Four-column UPPER mirror group starting at column a (a % 4 == 0).
    // For UPPER, column a+k's implied rows are the symmetric mirror A[a+k][i],
    // i in [col+1, S) -- the stored triangle at row col. The four stored
    // columns (rows [0, col], physical column col, contiguous kind-0) plus the
    // boundary and group mirrors (strided kind-1, one chunk per column) are
    // emitted here; the caller then skips columns a+1..a+3 via mirrorGroupEnd_.
    __aicore__ inline void BuildAMirrorGroup(int32_t a)
    {
        const int32_t S32 = static_cast<int32_t>(S_);
        // stored: 4 columns, rows [0, a+k], contiguous source 2*(i + (a+k)*lda).
        for (int32_t k = 0; k < 4; k++) {
            const int64_t col = static_cast<int64_t>(a) + k;
            PushAChunk(0,
                       2ULL * col * static_cast<uint64_t>(lda_),
                       static_cast<int32_t>(col) + 1,
                       static_cast<uint64_t>(col) * S_);
        }
        // exclusive boundary mirrors: column col=a+k has rows [col+1, a+4),
        // k=0..2 (3, 2, 1 rows); symmetric mirror source A[col][col+1..a+4)
        // (strided kind-1 read, like the single-column mirror path).
        for (int32_t k = 0; k < 3; k++) {
            const int64_t col = static_cast<int64_t>(a) + k;
            const int32_t rows = a + 4 - (static_cast<int32_t>(col) + 1);
            if (rows <= 0) continue;
            PushAChunk(1,
                       2ULL * (static_cast<uint64_t>(col) +
                               (static_cast<uint64_t>(col) + 1) *
                                   static_cast<uint64_t>(lda_)),
                       rows,
                       static_cast<uint64_t>(col) * S_ + (static_cast<int64_t>(col) + 1));
        }
        // group mirror: rows [a+4, S), symmetric mirror source A[col][a+4..S)
        // (the stored-triangle rows a..a+3 of output column i). Kind-3 reads
        // column i's rows a..a+3 as one 32B block, so the GM read transactions
        // drop 4x vs the per-column 8B kind-1 reads it replaces. Kind-3 runs
        // cover full 8-column groups only; any trailing (<8) rows fall back to
        // one per-column kind-1 chunk.
        const int32_t gStart = a + 4;
        const int32_t next = PushKind3Chunk(a, gStart, S32);
        for (int32_t k = 0; k < 4 && next < S32; k++) {
            const int64_t col = static_cast<int64_t>(a) + k;
            PushAChunk(1,
                       2ULL * (static_cast<uint64_t>(col) +
                               static_cast<uint64_t>(next) * static_cast<uint64_t>(lda_)),
                       S32 - next,
                       static_cast<uint64_t>(col) * S_ + static_cast<uint64_t>(next));
        }
        mirrorGroupEnd_ = a + 4;
    }

    // Four-column LOWER mirror group starting at column a (a % 4 == 0).
    // LOWER stores A[i][a] for i >= a (row >= column). The group mirrors the
    // shared implied rows [0, a) (symmetric source A[col][i], i < col, stored
    // at row col of output column i -> kind-3 reads rows a..a+3 of column i as
    // one 32B block) plus the small boundary rows [a, a+k) per column.
    __aicore__ inline void BuildAMirrorGroupLower(int32_t a)
    {
        const int32_t S32 = static_cast<int32_t>(S_);
        for (int32_t k = 0; k < 4; k++) {
            const int64_t col = static_cast<int64_t>(a) + k;
            // stored: rows [col, S), contiguous source 2*(col + col*lda).
            PushAChunk(0,
                       2ULL * (col + col * static_cast<uint64_t>(lda_)),
                       S32 - static_cast<int32_t>(col),
                       static_cast<uint64_t>(col) * S_ + static_cast<uint64_t>(col));
        }
        // boundary mirrors: column col=a+k needs implied rows [a, a+k)
        // (rows [0, a) come from the shared group mirror); source A[col][i],
        // i in [a, col), strided kind-1.
        for (int32_t k = 1; k < 4; k++) {
            const int64_t col = static_cast<int64_t>(a) + k;
            PushAChunk(1,
                       2ULL * (static_cast<uint64_t>(col) +
                               static_cast<uint64_t>(a) * static_cast<uint64_t>(lda_)),
                       k,
                       static_cast<uint64_t>(col) * S_ + static_cast<uint64_t>(a));
        }
        // group mirror: implied rows [0, a) of all four columns a..a+3.
        // Kind-3 runs cover full 8-column groups; trailing (<8) rows fall back
        // to one per-column kind-1 chunk.
        const int32_t next = PushKind3Chunk(a, 0, a);
        for (int32_t k = 0; k < 4 && next < a; k++) {
            const int64_t col = static_cast<int64_t>(a) + k;
            PushAChunk(1,
                       2ULL * (static_cast<uint64_t>(col) +
                               static_cast<uint64_t>(next) * static_cast<uint64_t>(lda_)),
                       a - next,
                       static_cast<uint64_t>(col) * S_ + static_cast<uint64_t>(next));
        }
        mirrorGroupEnd_ = a + 4;
    }

    // Kind-3 chunk: reads columns i in [iStart, iEnd), taking each column's
    // rows aCol..aCol+3 as one contiguous 32B GM block (strided by lda between
    // columns), and fans the 4 complex values out to the per-column outputs.
    // rows field is the complex count (4 per column); PREP_MIRROR_MAX_ROWS caps
    // the columns per chunk so the UB footprint matches kind-0 chunks.
    // Column counts are rounded down to a multiple of 8: FanoutStore4's
    // two-level DeInterleave writes the second column half at a 4*gRows*B
    // offset, which must stay 32B-aligned (gRows % 8 == 0) or the aligned
    // vector stores corrupt the data. Rows that do not fit a full 8-column
    // run are NOT emitted here: the caller receives the next unhandled row
    // and covers them with per-column kind-1 chunks.
    __aicore__ inline int32_t PushKind3Chunk(int32_t aCol, int32_t iStart, int32_t iEnd)
    {
        int32_t i = iStart;
        while (i + 8 <= iEnd) {
            int32_t cur = iEnd - i;
            if (cur > PREP_MIRROR_MAX_ROWS) cur = PREP_MIRROR_MAX_ROWS;
            cur &= ~7;  // keep the fan-out segment offsets 32B-aligned
            if (cur < 8) break;
            Emit(3, 1,
                 2ULL * (static_cast<uint64_t>(aCol) +
                     static_cast<uint64_t>(i) * static_cast<uint64_t>(lda_)),
                 4 * cur,
                 static_cast<uint64_t>(aCol) * S_ + static_cast<uint64_t>(i));
            i += cur;
        }
        return i;
    }

    // Kind-3 consume-side helper: src4 holds c.rows = 4*gRows floats in the
    // de-interleaved [row][col0 col1 col2 col3] layout (row-major per 4-wide
    // group). Two-level DeInterleave turns it into four contiguous gRows runs
    // (col k), each written with one contiguous DataCopyPad to column a+k of
    // the (column-major, leading dim S) output plane. sumQue_ buffers are the
    // scratch; kind3 is the only sumQue_ user and chunks consume serially.
    __aicore__ inline void FanoutStore4(
        const LocalTensor<float>& src4, GlobalTensor<float>& gm,
        uint32_t dstOff, int32_t gRows)
    {
        const uint32_t g = static_cast<uint32_t>(gRows);
        // Every vector-written buffer that an MTE3 read (DataCopyPad) consumes
        // must round-trip its VECOUT queue (EnQue/DeQue) to order the
        // producer->consumer access -- a raw vector->MTE3 dependency is a race
        // (-O0 hides it, -O1+ exposes it). s1/s2/s3 therefore all go through
        // the queue. sumQue_ (4 deep) is used only by kind-3 and chunks are
        // consumed serially, so 3 buffers in flight never deadlock.
        LocalTensor<float> s1 = sumQue_.AllocTensor<float>();
        LocalTensor<float> s2 = sumQue_.AllocTensor<float>();
        // level 1: [row][c0 c1 c2 c3] -> s1 = [c0 c2] rows, s2 = [c1 c3] rows
        DeInterleave(s1, s2, src4, 4 * g);
        sumQue_.EnQue(s1);
        sumQue_.EnQue(s2);
        LocalTensor<float> s1q = sumQue_.DeQue<float>();
        LocalTensor<float> s2q = sumQue_.DeQue<float>();
        // level 2, cols 0/2: s1 ([t][c0 c2]) -> s3[0,g)=col0, s3[g,2g)=col2
        LocalTensor<float> s3 = sumQue_.AllocTensor<float>();
        DeInterleave(s3, s3[g], s1q, 2 * g);
        sumQue_.EnQue(s3);
        LocalTensor<float> s3q = sumQue_.DeQue<float>();
        // level 2, cols 1/3: s2 ([t][c1 c3]) -> s1q[0,g)=col1, s1q[g,2g)=col3
        DeInterleave(s1q, s1q[g], s2q, 2 * g);
        sumQue_.EnQue(s1q);
        LocalTensor<float> s1q2 = sumQue_.DeQue<float>();
        DataCopyExtParams runExt{1,
            static_cast<uint32_t>(g * sizeof(float)), 0, 0, 0};
        const uint32_t sDim = static_cast<uint32_t>(S_);
        DataCopyPad(gm[dstOff], s3q, runExt);
        DataCopyPad(gm[dstOff + sDim], s1q2, runExt);
        DataCopyPad(gm[dstOff + 2 * sDim], s3q[g], runExt);
        DataCopyPad(gm[dstOff + 3 * sDim], s1q2[g], runExt);
        sumQue_.FreeTensor(s1q2);
        sumQue_.FreeTensor(s2q);
        sumQue_.FreeTensor(s3q);
    }

    // Split a (kind, srcOff, rows, dstOff) range into PREP_CHUNK_ROWS-sized
    // chunks and stream each one into the pipeline. Only the A column builders
    // call this, so the source is always A.
    __aicore__ inline void PushAChunk(
        uint32_t kind, uint64_t srcOff, int32_t rows, uint64_t dstOff)
    {
        if (rows <= 0) return;
        int32_t done = 0;
        while (done < rows) {
            int32_t cur = rows - done;
            if (cur > PREP_CHUNK_ROWS) cur = PREP_CHUNK_ROWS;
            // kind 0: consecutive complex elements (2 floats/elt), kind 1:
            // A's row a at column stride lda (2 floats apart per column).
            Emit(kind, 1,
                 srcOff +
                     static_cast<uint64_t>(done) * (kind == 0 ? 2ULL : 2ULL * static_cast<uint64_t>(lda_)),
                 cur,
                 dstOff + static_cast<uint64_t>(done));
            done += cur;
        }
    }

    TPipe* pipe_ = nullptr;
    GlobalTensor<float> aGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<float> arGm_;
    GlobalTensor<float> aiGm_;
    GlobalTensor<float> brGm_;
    GlobalTensor<float> biGm_;
    GlobalTensor<float> apGm_;
    GlobalTensor<float> bpGm_;
    TQue<QuePosition::VECIN, PREP_BUF_NUM> inQue_;
    TQue<QuePosition::VECOUT, PREP_BUF_NUM> reQue_;
    TQue<QuePosition::VECOUT, PREP_BUF_NUM> imQue_;
    TQue<QuePosition::VECOUT, PREP_BUF_NUM> sumQue_;
    // Streaming chunk pipeline: descriptors for the PREP_BUF_NUM reads
    // currently in flight, indexed by chunk sequence number modulo the depth.
    // Slots live on the AIV stack (PREP_BUF_NUM x sizeof(PrepChunkDesc)), not
    // in a growable array, so the chunk count is unbounded (see Emit).
    PrepChunkDesc ring_[PREP_BUF_NUM];
    uint32_t emitted_ = 0;   // chunks generated (and read issued)
    uint32_t consumed_ = 0;  // chunks dequeued and processed
    int32_t lda_ = 0;
    int32_t ldb_ = 0;
    int32_t M_ = 0;
    int32_t N_ = 0;
    int32_t S_ = 0;
    int32_t uplo_ = 0;
    int32_t aStart_ = 0;
    int32_t aEnd_ = 0;
    // When BuildAMirrorGroup packs columns a..a+3 into one chunk, the caller
    // skips the A processing of columns a+1..a+3 (mirrorGroupEnd_ = a+4).
    int32_t mirrorGroupEnd_ = 0;
};

extern "C" __global__ __aicore__ void csymm_prep_kernel(
    GM_ADDR A, int32_t lda, int32_t uplo, GM_ADDR B, int32_t ldb,
    GM_ADDR Ar, GM_ADDR Ai, GM_ADDR Br, GM_ADDR Bi,
    GM_ADDR Aplus, GM_ADDR Bplus,
    int32_t M, int32_t N, int32_t S,
    int32_t aStart, int32_t aEnd)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    csymm_prep::CsymmPrepKernel op;
    op.Init(A, lda, uplo, B, ldb, Ar, Ai, Br, Bi, Aplus, Bplus, M, N, S,
            aStart, aEnd, &pipe);
    op.Process();
}

} // namespace csymm_prep

void csymm_prep_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR A, int32_t lda, int32_t uplo, GM_ADDR B, int32_t ldb,
    GM_ADDR Ar, GM_ADDR Ai, GM_ADDR Br, GM_ADDR Bi,
    GM_ADDR Aplus, GM_ADDR Bplus,
    int32_t M, int32_t N, int32_t S,
    int32_t aStart, int32_t aEnd)
{
    csymm_prep::csymm_prep_kernel<<<numBlocks, nullptr, stream>>>(
        A, lda, uplo, B, ldb, Ar, Ai, Br, Bi, Aplus, Bplus, M, N, S,
        aStart, aEnd);
}

// ============================================================================
//  Small-shape single-launch direct kernel (m <= 64 && n <= 64).
//  The production path (prep + 3m cube + combine) costs three kernel launches,
//  ~2.9us of per-launch fixed overhead each -> ~15us end-to-end on shapes like
//  1x1..64x64 whose device work is a few microseconds. The task-book baseline
//  is 0.4x an H100 (which itself pays cuBLAS launch overhead): 1x1..16x16 need
//  <= 7.5us, 32x32 <= 10us, 64x64 <= 15us. A single AIV launch that computes
//  C = alpha*A*B + beta*C directly (symmetric A read with on-the-fly mirroring,
//  no 3m, no GM temp planes) lands at ~7-9us depending on m — the only path
//  that can meet those thresholds.
//  Column-parallel: each AIV core owns a range of output columns c. For each c
//  it accumulates the complex matrix-vector product over the K dimension and
//  writes the interleaved result back to C[:,c].
//    LEFT : C[i][c] = sum_k A[i][k] * B[k][c]   (A is SxS, S = m)
//    RIGHT: C[i][c] = sum_k B[i][k] * A[k][c]   (A is SxS, S = n)
//  A is half-stored symmetric (uplo); the mirrored half is read strided
//  (1 complex per 8B block, GM gap 8*lda - 8, the exact pattern the prep
//  kernel's kind-1 chunks use) and the stored half is contiguous per column.
// ============================================================================
namespace csymm_small {
using namespace AscendC;

// Host dispatches this kernel only for m,n <= 16, so the scratch layout only
// needs to cover SMAX = 16 (a 16-float column stride keeps every plane column
// 64B-aligned for vector ops). Sized this down from 64: the A-plane / queue
// InitBuffer region shrank 13x, and the UPPER lda==S fast path reads the whole
// square in one bulk DataCopyPad instead of 2*S per-column round trips.
constexpr int32_t SMALL_MAX = 16;
// VECIN queue depth for the column/chunk reads.
constexpr int32_t SMALL_BUF_NUM = 4;
// Scratch layout (floats) in the single VECCALC buffer, sized for SMALL_MAX:
//   [0, SMAX)          t0   (rank-1 real scratch)
//   [SMAX, 2*SMAX)     t1   (rank-1 imag scratch)
//   [2*SMAX, 3*SMAX)   cR   (LEFT: per-column B re;  RIGHT: per-column A re)
//   [3*SMAX, 4*SMAX)   cI
//   [4*SMAX, 5*SMAX)   accR
//   [5*SMAX, 6*SMAX)   accI
//   [6*SMAX, 7*SMAX)   outR
//   [7*SMAX, 8*SMAX)   outI
//   [8*SMAX, 9*SMAX)   bR   (RIGHT: per-k B-column re, must survive rank-1)
//   [9*SMAX, 10*SMAX)  bI
//   [10*SMAX, 10*SMAX+SMAX*SMAX)         aRe (LEFT full A plane)
//   [10*SMAX+SMAX*SMAX, 10*SMAX+2*SMAX*SMAX) aIm
// NOTE: every vector op must touch 32B-aligned (8-float) offsets inside UB.
// The scratch regions above all start at multiples of SMAX (64B); the A-plane
// copies are deliberately scalar so a non-aligned row offset can never feed a
// vector instruction (a UB misaligned vector access corrupts data AND makes the
// kernel pathologically slow).
constexpr int32_t SCR_T0 = 0;
constexpr int32_t SCR_T1 = SMALL_MAX;
constexpr int32_t SCR_C_R = 2 * SMALL_MAX;
constexpr int32_t SCR_C_I = 3 * SMALL_MAX;
constexpr int32_t SCR_ACC_R = 4 * SMALL_MAX;
constexpr int32_t SCR_ACC_I = 5 * SMALL_MAX;
constexpr int32_t SCR_OUT_R = 6 * SMALL_MAX;
constexpr int32_t SCR_OUT_I = 7 * SMALL_MAX;
constexpr int32_t SCR_B_R = 8 * SMALL_MAX;
constexpr int32_t SCR_B_I = 9 * SMALL_MAX;
constexpr int32_t SCR_A_RE = 10 * SMALL_MAX;
constexpr int32_t SCR_A_IM = 10 * SMALL_MAX + SMALL_MAX * SMALL_MAX;
// Second pair of rank-1 temporaries: the real and imaginary accumulator chains
// must not share scratch, otherwise the second chain stalls on a write-after-
// read hazard behind the first one and the two halves serialise.
constexpr int32_t SCR_T2 = 10 * SMALL_MAX + 2 * SMALL_MAX * SMALL_MAX;
constexpr int32_t SCR_T3 = SCR_T2 + SMALL_MAX;
// Raw (still interleaved) staging slots for the per-column GM read and the
// column result, so the hot column loop can bypass the VECIN/VECOUT queue
// handshake (Alloc/EnQue/DeQue/Free on every column costs real microseconds at
// these budgets) and DataCopyPad straight between GM and UB.
constexpr int32_t SCR_RAW_IN = SCR_T3 + SMALL_MAX;
constexpr int32_t SCR_RAW_OUT = SCR_RAW_IN + 2 * SMALL_MAX;
// Scratch for the 16x16 plane transpose used to build the mirrored half of A.
constexpr int32_t SCR_TR = SCR_RAW_OUT + 2 * SMALL_MAX;
constexpr int32_t SCR_TR_TMP = SCR_TR + SMALL_MAX * SMALL_MAX;
constexpr int32_t SCR_FLOATS = SCR_TR_TMP + SMALL_MAX * SMALL_MAX;

// Rank-1 update of acc (length m) with a complex column vec (re/im, length m)
// times the complex scalar (sr, si):
//   accR[i] += re[i]*sr - im[i]*si ; accI[i] += re[i]*si + im[i]*sr
// Vector instructions on this platform take 8-element-aligned counts reliably;
// the last m%8 elements are handled scalar-wise (they are few and cheap).
__aicore__ inline void Rank1Accumulate(
    LocalTensor<float>& accR, LocalTensor<float>& accI,
    LocalTensor<float> re, LocalTensor<float> im,
    float sr, float si, int32_t m, LocalTensor<float> t0, LocalTensor<float> t1,
    LocalTensor<float> t2, LocalTensor<float> t3)
{
    if (m <= 0) return;
    int32_t vc = m & ~7;
    if (vc == 0 && m < 8) {
        // Sub-8 row count: run one 8-wide vector op. Rows [m,8) read garbage
        // column padding but are never read back (Finalize writes [0,m)).
        vc = 8;
    }
    if (vc > 0) {
        // accR += re*sr - im*si ; accI += re*si + im*sr.
        // Axpy folds the scale into the accumulate (one op instead of Muls+Add)
        // and the two chains use disjoint temporaries, so they issue back to
        // back instead of serialising on a write-after-read hazard.
        Muls(t0, im, si, vc);   // im*si
        Axpy(accR, re, sr, vc); // accR += re*sr
        Sub(accR, accR, t0, vc);
        Muls(t2, im, sr, vc);   // im*sr
        Axpy(accI, re, si, vc); // accI += re*si
        Add(accI, accI, t2, vc);
        (void)t1;
        (void)t3;
    }
    for (int32_t i = vc; i < m; i++) {
        float arv = re.GetValue(i);
        float aiv = im.GetValue(i);
        float nr = accR.GetValue(i) + arv * sr - aiv * si;
        float ni = accI.GetValue(i) + arv * si + aiv * sr;
        accR.SetValue(i, nr);
        accI.SetValue(i, ni);
    }
}

// Zero-init an accumulator of length m (vec part via Duplicate, scalar tail).
__aicore__ inline void ZeroAcc(LocalTensor<float> acc, int32_t m)
{
    if (m <= 0) return;
    int32_t vc = m & ~7;
    if (vc == 0 && m < 8) {
        vc = 8;
    }
    if (vc > 0) {
        Duplicate(acc, static_cast<float>(0), vc);
    }
    for (int32_t i = vc; i < m; i++) {
        acc.SetValue(i, 0.0f);
    }
}

class CsymmSmallKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR A, int32_t lda, int32_t uplo, GM_ADDR B, int32_t ldb,
        GM_ADDR C, int32_t ldc, int32_t m, int32_t n, int32_t S, int32_t side,
        float ar, float ai, float br, float bi, TPipe* pipe)
    {
        pipe_ = pipe;
        lda_ = lda;
        ldb_ = ldb;
        ldc_ = ldc;
        m_ = m;
        n_ = n;
        S_ = S;
        uplo_ = uplo;  // 121 = ACLBLAS_UPPER, 122 = ACLBLAS_LOWER
        side_ = side;  // 0 = LEFT, 1 = RIGHT (host normalizes)
        ar_ = ar;
        ai_ = ai;
        br_ = br;
        bi_ = bi;
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A),
            static_cast<uint64_t>(lda) * lda * 2);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(B),
            static_cast<uint64_t>(ldb) * n_ * 2);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(C),
            static_cast<uint64_t>(ldc) * n_ * 2);
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        colsPerCore_ = (n_ + static_cast<int32_t>(blockNum) - 1) /
            static_cast<int32_t>(blockNum);
        startCol_ = static_cast<int32_t>(blockIdx) * colsPerCore_;
        endCol_ = startCol_ + colsPerCore_;
        if (endCol_ > n_) endCol_ = n_;
        if (startCol_ >= n_) {
            startCol_ = 0;
            endCol_ = 0;
        }
    }

    __aicore__ inline void Process()
    {
        if (startCol_ >= endCol_) return;
        pipe_->InitBuffer(calcBuf_, SCR_FLOATS * sizeof(float));
        calc_ = calcBuf_.Get<float>();
        // VECIN queue for GM reads. Frames are sized to also hold the full
        // contiguous A square (2*SMAX*SMAX floats) used by the UPPER lda==S
        // bulk-read fast path; the per-column chunk reads only touch the head
        // of a frame.
        pipe_->InitBuffer(inQue_, SMALL_BUF_NUM, 2 * SMALL_MAX * SMALL_MAX * sizeof(float));
        pipe_->InitBuffer(outQue_, SMALL_BUF_NUM, 2 * SMALL_MAX * sizeof(float));
        // Dedicated single-slot queue for the B-column read: issuing it before
        // the A-plane build lets the two GM transfers overlap. The A-plane build
        // is ~2us of pure UB work on 16x16, more than enough to hide one GM
        // latency, and the column loop needs B only after that.
        const bool prefetchB = (side_ == 0) && ((endCol_ - startCol_) == 1);
        if (prefetchB) {
            pipe_->InitBuffer(inQueB_, 1, 2 * SMALL_MAX * sizeof(float));
            StartBColumnRead(startCol_);
        }
        if (side_ == 0) {
            // LEFT: build the full mirrored A plane once (shared by all columns
            // this core owns), then rank-1 over the B column scalars.
            BuildAFullPlane();
        }
        for (int32_t c = startCol_; c < endCol_; c++) {
            if (side_ == 0) {
                ProcessColumnLeft(c, prefetchB);
            } else {
                ProcessColumnRight(c);
            }
        }
    }

private:
    // -------- A plane construction --------------------------------------
    // MirrorAChunk(seg): read one half of mirrored column `col` into
    // [rSeg, iSeg] (each `len` floats).
    //   seg == 0: LOWER mirror part   rows [0, col)      (from A row col)
    //   seg == 1: LOWER stored part   rows [col, S)      (from A column col)
    // (UPPER: seg0 = stored rows [0, col], seg1 = mirror rows [col+1, S).)
    __aicore__ inline void ReadAMirrorChunk(
        int32_t col, int32_t seg, int32_t& len,
        LocalTensor<float>& rSeg, LocalTensor<float>& iSeg, int32_t& outOff,
        bool& directToPlane)
    {
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        LocalTensor<float> in = inQue_.AllocTensor<float>();
        int64_t gmFloatOff = 0;
        if (uplo_ == 122) {  // LOWER
            if (seg == 0) {
                // mirror rows [0,col): A[col][i] i=0..col-1, strided.
                len = col;
                gmFloatOff = static_cast<int64_t>(2) * col;  // A[col][0]
                DataCopyExtParams ext{static_cast<uint16_t>(len), 8,
                    static_cast<int64_t>(8) * lda_ - 8, 0, 0};
                DataCopyPad<float, PaddingMode::Compact>(in, aGm_[gmFloatOff], ext, pad);
                outOff = 0;
            } else {
                // stored rows [col,S): A[i][col] i=col..S-1, contiguous.
                len = S_ - col;
                gmFloatOff = static_cast<int64_t>(2) * col * lda_ + 2 * col;
                DataCopyExtParams ext{1, static_cast<uint32_t>(len * 2 * sizeof(float)), 0, 0, 0};
                DataCopyPad(in, aGm_[gmFloatOff], ext, pad);
                outOff = col;
            }
        } else {             // UPPER
            if (seg == 0) {
                // stored rows [0,col]: A[i][col] i=0..col, contiguous.
                len = col + 1;
                gmFloatOff = static_cast<int64_t>(2) * col * lda_;
                DataCopyExtParams ext{1, static_cast<uint32_t>(len * 2 * sizeof(float)), 0, 0, 0};
                DataCopyPad(in, aGm_[gmFloatOff], ext, pad);
                outOff = 0;
            } else {
                // mirror rows [col+1,S): A[col][i] i=col+1..S-1, strided.
                len = S_ - col - 1;
                gmFloatOff = static_cast<int64_t>(2) * col + 2 * (col + 1) * lda_;
                DataCopyExtParams ext{static_cast<uint16_t>(len), 8,
                    static_cast<int64_t>(8) * lda_ - 8, 0, 0};
                DataCopyPad<float, PaddingMode::Compact>(in, aGm_[gmFloatOff], ext, pad);
                outOff = col + 1;
            }
        }
        inQue_.EnQue(in);
        LocalTensor<float> inLocal = inQue_.DeQue<float>();
        // When the destination row offset is 8-float aligned the DeInterleave
        // output can land straight in the A plane (both plane bases and
        // col*SMALL_MAX are 8-float aligned), skipping the rSeg/iSeg round
        // trip and the scalar copy. Otherwise (segment straddles a non-8
        // row offset) fall back to rSeg/iSeg + the caller's scalar scatter.
        // DeInterleave writes 16-float (64B) granules: the destination must
        // sit on a 64B boundary. Segments starting mid-row (outOff % 16 != 0)
        // can never be written directly. Column-leading segments (outOff == 0,
        // the only ones with outOff % 16 == 0) are safe to write directly:
        // any over-write past the logical segment length lands either in the
        // sibling half of the same column (overwritten by the second segment
        // which runs later) or in rows >= S that rank-1 never reads.
        directToPlane = ((outOff & 15) == 0);
        if (directToPlane) {
            LocalTensor<float> aRe = calc_[SCR_A_RE + col * SMALL_MAX + outOff];
            LocalTensor<float> aIm = calc_[SCR_A_IM + col * SMALL_MAX + outOff];
            DeInterleave(aRe, aIm, inLocal, 2 * len);
        } else {
            DeInterleave(rSeg, iSeg, inLocal, 2 * len);
        }
        inQue_.FreeTensor(inLocal);
    }

    // Write mirrored column `col` into the scratch planes aRe/aIm at
    // row-major-on-column layout offset col*S.
    __aicore__ inline void BuildAMirroredColumn(int32_t col)
    {
        // LOWER: seg0 = mirror rows [0,col), seg1 = stored rows [col,S).
        // UPPER: seg0 = stored rows [0,col], seg1 = mirror rows [col+1,S).
        const int32_t segs = 2;
        for (int32_t s = 0; s < segs; s++) {
            int32_t len = 0;
            int32_t outOff = 0;
            bool direct = false;
            LocalTensor<float> rS = calc_[SCR_T0];
            LocalTensor<float> iS = calc_[SCR_T1];
            ReadAMirrorChunk(col, s, len, rS, iS, outOff, direct);
            if (len <= 0) continue;
            if (direct) continue;  // already DeInterleaved straight into plane
            LocalTensor<float> aRe = calc_[SCR_A_RE + col * SMALL_MAX];
            LocalTensor<float> aIm = calc_[SCR_A_IM + col * SMALL_MAX];
            // Scalar copy into the plane at row outOff. outOff is NOT
            // guaranteed 8-float aligned, so a vector write here would touch a
            // misaligned UB address (wrong data + pathological slowness). len
            // is at most S <= 64, so plain scalar transfers are cheap.
            for (int32_t t = 0; t < len; t++) {
                aRe.SetValue(outOff + t, rS.GetValue(t));
                aIm.SetValue(outOff + t, iS.GetValue(t));
            }
        }
    }

    // Fill the mirrored (strictly lower) half of a stored column-major plane.
    // The mirror is exactly the transpose of the stored half -- element (i,c)
    // with i > c equals stored (c,i) -- so transposing the plane into scratch
    // turns the per-element scalar scatter into contiguous copies.
    // The transpose itself is done with DeInterleave: splitting a 256-element
    // sequence into its even-indexed and odd-indexed halves and concatenating
    // them rotates the element index right by one bit, so four rounds rotate the
    // 8-bit index by 4 and turn the [row|col] layout into [col|row].
    __aicore__ inline void FillMirrorFromTranspose(LocalTensor<float> p)
    {
        constexpr int32_t N = SMALL_MAX * SMALL_MAX;
        constexpr int32_t HALF = N / 2;
        LocalTensor<float> tr = calc_[SCR_TR];
        LocalTensor<float> tmp = calc_[SCR_TR_TMP];
        Adds(tr, p, 0.0f, N);
        LocalTensor<float>* cur = &tr;
        LocalTensor<float>* out = &tmp;
        for (int32_t r = 0; r < 4; r++) {
            DeInterleave((*out)[0], (*out)[HALF], *cur, N);
            LocalTensor<float>* t = cur;
            cur = out;
            out = t;
        }
        // Four (even) rounds leave the transposed plane back in `tr`.
        for (int32_t c = 0; c < SMALL_MAX; c++) {
            const int32_t lo = c + 1;
            if (lo >= SMALL_MAX) {
                continue;
            }
            if (lo <= 8) {
                // Rows [8, S) are 8-float aligned in both planes and are all
                // strictly lower (i >= 8 > c), so they copy as one vector op.
                Adds(p[c * SMALL_MAX + 8], tr[c * SMALL_MAX + 8], 0.0f, SMALL_MAX - 8);
                for (int32_t i = lo; i < 8; i++) {
                    p.SetValue(c * SMALL_MAX + i, tr.GetValue(c * SMALL_MAX + i));
                }
            } else {
                // Fewer than 8 elements left and an unaligned start: scalar.
                for (int32_t i = lo; i < SMALL_MAX; i++) {
                    p.SetValue(c * SMALL_MAX + i, tr.GetValue(c * SMALL_MAX + i));
                }
            }
        }
    }

    // UPPER + lda==S fast path: the stored complex SxS square is one contiguous
    // GM block (column-major, lda==S). Read it once into a single queue frame,
    // then rebuild the plane in UB: column c's stored rows [0..c] are the
    // contiguous head of GM column c, and the mirrored rows [c+1..S) copy
    // element (c,i) from stored element (i,c) at raw[c + i*S]. This removes the
    // 2*S per-column DataCopyPad round trips that dominate the small-shape
    // content time.
    // The stored head may go straight into the plane via DeInterleave only when
    // the source raw[2*S*c] is 32B aligned: byte offset 8*S*c is a multiple of
    // 32 for every c iff S%4==0. Other S values (2, 6, 10, 14, odd) fall back
    // to scalar copies (a misaligned DeInterleave source is pathologically
    // slow and corrupts data).
    __aicore__ inline void FastBuildAFullPlaneUpper()
    {
        LocalTensor<float> in = inQue_.AllocTensor<float>();
        DataCopyExtParams ext{1, static_cast<uint32_t>(2 * S_ * S_ * sizeof(float)),
            0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(in, aGm_[0], ext, pad);
        inQue_.EnQue(in);
        LocalTensor<float> raw = inQue_.DeQue<float>();
        // S == SMALL_MAX (16) fast path: one DeInterleave splits the whole
        // complex square straight into the A planes (here the plane stride
        // equals S, so the interleaved square and the plane layout coincide),
        // and the mirrored half is then filled in place -- element (i,c) with
        // i > c equals stored (c,i) == plane[c*S + i], and every write lands in
        // the strictly lower triangle so it can never clobber a source element.
        // This removes the 2*S per-column DeInterleave round trips that
        // dominated the small-shape content time.
        if (S_ == SMALL_MAX) {
            LocalTensor<float> aRe = calc_[SCR_A_RE];
            LocalTensor<float> aIm = calc_[SCR_A_IM];
            DeInterleave(aRe, aIm, raw, 2 * S_ * S_);
            // Mirrored half: element (i,c) with i > c equals stored (c,i).
            // Writes land in the strictly lower triangle, so they never clobber
            // a source element.
            for (int32_t c = 0; c < S_; c++) {
                for (int32_t i = c + 1; i < S_; i++) {
                    aRe.SetValue(c * SMALL_MAX + i, aRe.GetValue(i * SMALL_MAX + c));
                    aIm.SetValue(c * SMALL_MAX + i, aIm.GetValue(i * SMALL_MAX + c));
                }
            }
            inQue_.FreeTensor(raw);
            return;
        }
        const bool vecStored = ((S_ & 3) == 0);
        for (int32_t c = 0; c < S_; c++) {
            LocalTensor<float> aRe = calc_[SCR_A_RE + c * SMALL_MAX];
            LocalTensor<float> aIm = calc_[SCR_A_IM + c * SMALL_MAX];
            if (vecStored) {
                // Stored head of column c: rows [0..c], contiguous at raw[2*S*c].
                DeInterleave(aRe, aIm, raw[2 * S_ * c], 2 * (c + 1));
            } else {
                for (int32_t t = 0; t <= c; t++) {
                    aRe.SetValue(t, raw.GetValue(2 * (t + S_ * c)));
                    aIm.SetValue(t, raw.GetValue(2 * (t + S_ * c) + 1));
                }
            }
            // Mirrored tail of column c: element (i,c) := stored (c,i).
            for (int32_t i = c + 1; i < S_; i++) {
                aRe.SetValue(i, raw.GetValue(2 * (c + i * S_)));
                aIm.SetValue(i, raw.GetValue(2 * (c + i * S_) + 1));
            }
        }
        inQue_.FreeTensor(raw);
    }

    __aicore__ inline void BuildAFullPlane()
    {
        // UPPER symmetric square with contiguous storage (lda == S): one bulk
        // GM read + in-UB rebuild (see above). All other cases (LOWER, RIGHT,
        // padded lda) keep the per-column chunk path.
        if (uplo_ == 121 && lda_ == S_) {
            FastBuildAFullPlaneUpper();
            return;
        }
        for (int32_t col = 0; col < S_; col++) {
            BuildAMirroredColumn(col);
        }
    }

    // -------- per-column direct computation ------------------------------
    // Out-of-line result assembly shared by both sides:
    //   acc holds the raw product sum; apply alpha and add beta*C, interleave
    //   into outBuf (VECOUT) and copy to C[:,col].
    // Load complex C[:,col] (de-interleaved into cR/cI) when beta != 0.
    __aicore__ inline void CsymmLoadComplexC(
        int32_t col, int32_t m, LocalTensor<float>& cR, LocalTensor<float>& cI)
    {
        LocalTensor<float> cIn = inQue_.AllocTensor<float>();
        DataCopyExtParams ext{1, static_cast<uint32_t>(2 * m * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(cIn, cGm_[static_cast<int64_t>(2) * col * ldc_], ext, pad);
        inQue_.EnQue(cIn);
        LocalTensor<float> cLocal = inQue_.DeQue<float>();
        DeInterleave(cR, cI, cLocal, 2 * m);
        inQue_.FreeTensor(cLocal);
    }

    // Vector (8-aligned) + scalar tail alpha/beta accumulation into outBuf.
    __aicore__ inline void CsymmFinalizeGeneral(
        LocalTensor<float> accR, LocalTensor<float> accI,
        LocalTensor<float> outR, LocalTensor<float> outI,
        LocalTensor<float> t0, LocalTensor<float> t1,
        LocalTensor<float> cR, LocalTensor<float> cI,
        LocalTensor<float> outBuf, int32_t vc, int32_t m, bool betaZero)
    {
        if (vc > 0) {
            // outR = ar*accR - ai*accI
            Muls(outR, accR, ar_, vc);
            Muls(t0, accI, ai_, vc);
            Sub(outR, outR, t0, vc);
            // outI = ar*accI + ai*accR
            Muls(outI, accI, ar_, vc);
            Muls(t0, accR, ai_, vc);
            Add(outI, outI, t0, vc);
            if (!betaZero) {
                // beta term: t1 = br*cR - bi*cI -> outR; t0 = br*cI + bi*cR -> outI
                Muls(t0, cI, bi_, vc);   // bi*cI
                Muls(t1, cR, br_, vc);   // br*cR
                Sub(t1, t1, t0, vc);     // br*cR - bi*cI
                Add(outR, outR, t1, vc);
                Muls(t0, cI, br_, vc);   // br*cI
                Muls(t1, cR, bi_, vc);   // bi*cR (cR still original here)
                Add(t0, t0, t1, vc);     // br*cI + bi*cR
                Add(outI, outI, t0, vc);
            }
        }
        // Scalar tail [vc, m): identical formulas element-wise, writing the
        // interleaved positions of outBuf directly.
        for (int32_t i = vc; i < m; i++) {
            float avr = accR.GetValue(i);
            float avi = accI.GetValue(i);
            float or_ = ar_ * avr - ai_ * avi;
            float oi = ar_ * avi + ai_ * avr;
            if (!betaZero) {
                float cr = cR.GetValue(i);
                float ci = cI.GetValue(i);
                or_ += br_ * cr - bi_ * ci;
                oi += br_ * ci + bi_ * cr;
            }
            outBuf.SetValue(2 * i, or_);
            outBuf.SetValue(2 * i + 1, oi);
        }
        // Interleave the 8-aligned part: outBuf = [r0,i0,r1,i1,...]
        if (vc > 0) {
            LocalTensor<float> dst0 = outBuf[0];
            LocalTensor<float> dst1 = outBuf[vc];
            Interleave(dst0, dst1, outR, outI, vc);
        }
    }

    __aicore__ inline void FinalizeColumn(
        int32_t col, LocalTensor<float> accR, LocalTensor<float> accI)
    {
        const int32_t m = m_;
        int32_t vc = m & ~7;
        if (vc == 0 && m > 0) {
            vc = 8;  // sub-8 rows: one 8-wide op, garbage rows are not written
        }
        LocalTensor<float> outR = calc_[SCR_OUT_R];
        LocalTensor<float> outI = calc_[SCR_OUT_I];
        LocalTensor<float> t0 = calc_[SCR_T0];
        LocalTensor<float> t1 = calc_[SCR_T1];
        LocalTensor<float> cR = calc_[SCR_C_R];
        LocalTensor<float> cI = calc_[SCR_C_I];
        const bool betaZero = (br_ == 0.0f && bi_ == 0.0f);
        if (!betaZero) {
            CsymmLoadComplexC(col, m, cR, cI);
        }
        const bool alphaOne = (ar_ == 1.0f && ai_ == 0.0f);
        LocalTensor<float> outBuf = outQue_.AllocTensor<float>();
        if (alphaOne && betaZero) {
            // alpha==1, beta==0 (the steady-state perf-call shape): outBuf is
            // just the interleaved accumulator. Skips the alpha scaling
            // entirely (outR = ar*accR - ai*accI etc. reduce to a copy).
            if (vc > 0) {
                Interleave(outBuf[0], outBuf[vc], accR, accI, vc);
            }
            for (int32_t i = vc; i < m; i++) {
                outBuf.SetValue(2 * i, accR.GetValue(i));
                outBuf.SetValue(2 * i + 1, accI.GetValue(i));
            }
        } else {
            CsymmFinalizeGeneral(accR, accI, outR, outI, t0, t1, cR, cI, outBuf, vc, m, betaZero);
        }
        outQue_.EnQue(outBuf);
        LocalTensor<float> oLocal = outQue_.DeQue<float>();
        DataCopyExtParams ext{1, static_cast<uint32_t>(2 * m * sizeof(float)), 0, 0, 0};
        DataCopyPad<float, PaddingMode::Normal>(cGm_[static_cast<int64_t>(2) * col * ldc_],
                                                oLocal, ext);
        outQue_.FreeTensor(oLocal);
    }

    // Issue the GM read of B[:,col] into the dedicated queue so that it runs
    // concurrently with the A-plane build (see Process()).
    __aicore__ inline void StartBColumnRead(int32_t col)
    {
        LocalTensor<float> bIn = inQueB_.AllocTensor<float>();
        DataCopyExtParams ext{1, static_cast<uint32_t>(2 * S_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(bIn, bGm_[static_cast<int64_t>(2) * col * ldb_], ext, pad);
        inQueB_.EnQue(bIn);
    }

    __aicore__ inline void ProcessColumnLeft(int32_t col, bool bPrefetched)
    {
        const int32_t m = m_;  // S == m for LEFT
        LocalTensor<float> accR = calc_[SCR_ACC_R];
        LocalTensor<float> accI = calc_[SCR_ACC_I];
        LocalTensor<float> t0 = calc_[SCR_T0];
        LocalTensor<float> t1 = calc_[SCR_T1];
        ZeroAcc(accR, m);
        ZeroAcc(accI, m);
        // Read B[:,col] (S complex, contiguous) and de-interleave once into the
        // dedicated cR/cI region (must survive the per-k rank-1 scratch). When
        // prefetched, the copy was already issued before the A-plane build.
        LocalTensor<float> bLocal;
        if (bPrefetched) {
            bLocal = inQueB_.DeQue<float>();
        } else {
            LocalTensor<float> bIn = inQue_.AllocTensor<float>();
            DataCopyExtParams ext{1, static_cast<uint32_t>(2 * S_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(bIn, bGm_[static_cast<int64_t>(2) * col * ldb_], ext, pad);
            inQue_.EnQue(bIn);
            bLocal = inQue_.DeQue<float>();
        }
        LocalTensor<float> bR = calc_[SCR_C_R];
        LocalTensor<float> bI = calc_[SCR_C_I];
        DeInterleave(bR, bI, bLocal, 2 * S_);
        if (bPrefetched) {
            inQueB_.FreeTensor(bLocal);
        } else {
            inQue_.FreeTensor(bLocal);
        }
        // Hoist the B-column scalars out of the rank-1 loop: reading them inside
        // puts a UB scalar load on the critical path of every one of the S
        // accumulate steps.
        float bsr[SMALL_MAX];
        float bsi[SMALL_MAX];
        for (int32_t k = 0; k < S_; k++) {
            bsr[k] = bR.GetValue(k);
            bsi[k] = bI.GetValue(k);
        }
        LocalTensor<float> t2 = calc_[SCR_T2];
        LocalTensor<float> t3 = calc_[SCR_T3];
        for (int32_t k = 0; k < S_; k++) {
            LocalTensor<float> aRe = calc_[SCR_A_RE + k * SMALL_MAX];
            LocalTensor<float> aIm = calc_[SCR_A_IM + k * SMALL_MAX];
            Rank1Accumulate(accR, accI, aRe, aIm, bsr[k], bsi[k], m, t0, t1, t2, t3);
        }
        FinalizeColumn(col, accR, accI);
    }

    __aicore__ inline void ProcessColumnRight(int32_t col)
    {
        const int32_t m = m_;
        const int32_t S = S_;  // A is SxS with S == n for RIGHT
        LocalTensor<float> accR = calc_[SCR_ACC_R];
        LocalTensor<float> accI = calc_[SCR_ACC_I];
        LocalTensor<float> t0 = calc_[SCR_T0];
        LocalTensor<float> t1 = calc_[SCR_T1];
        ZeroAcc(accR, m);
        ZeroAcc(accI, m);
        // Build the mirrored A column `col` (length S) and copy it into the
        // dedicated cR/cI region (survives the per-k rank-1 scratch).
        BuildAMirroredColumn(col);
        LocalTensor<float> aR = calc_[SCR_C_R];
        LocalTensor<float> aI = calc_[SCR_C_I];
        for (int32_t i = 0; i < S; i++) {
            aR.SetValue(i, calc_[SCR_A_RE + col * SMALL_MAX].GetValue(i));
            aI.SetValue(i, calc_[SCR_A_IM + col * SMALL_MAX].GetValue(i));
        }
        // Hoist the A-column scalars out of the rank-1 loop (same reason as the
        // LEFT path: one UB scalar load less per accumulate step).
        float asr[SMALL_MAX];
        float asi[SMALL_MAX];
        for (int32_t k = 0; k < S; k++) {
            asr[k] = aR.GetValue(k);
            asi[k] = aI.GetValue(k);
        }
        LocalTensor<float> t2 = calc_[SCR_T2];
        LocalTensor<float> t3 = calc_[SCR_T3];
        for (int32_t k = 0; k < S; k++) {
            float sr = asr[k];
            float si = asi[k];
            // B[:,k] is a contiguous m-complex column.
            LocalTensor<float> bIn = inQue_.AllocTensor<float>();
            DataCopyExtParams ext{1, static_cast<uint32_t>(2 * m * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(bIn, bGm_[static_cast<int64_t>(2) * k * ldb_], ext, pad);
            inQue_.EnQue(bIn);
            LocalTensor<float> bLocal = inQue_.DeQue<float>();
            // bR/bI must live OUTSIDE the rank-1 scratch (SCR_T0/T1): the
            // rank-1 multiply chain overwrites t0/t1 while still reading the
            // B column on the next k iteration.
            LocalTensor<float> bR = calc_[SCR_B_R];
            LocalTensor<float> bI = calc_[SCR_B_I];
            DeInterleave(bR, bI, bLocal, 2 * m);
            inQue_.FreeTensor(bLocal);
            Rank1Accumulate(accR, accI, bR, bI, sr, si, m, t0, t1, t2, t3);
        }
        FinalizeColumn(col, accR, accI);
    }

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, SMALL_BUF_NUM> inQue_;
    TQue<QuePosition::VECIN, 1> inQueB_;
    TQue<QuePosition::VECOUT, SMALL_BUF_NUM> outQue_;
    TBuf<QuePosition::VECCALC> calcBuf_;
    LocalTensor<float> calc_;
    GlobalTensor<float> aGm_;
    GlobalTensor<float> bGm_;
    GlobalTensor<float> cGm_;
    int32_t lda_ = 0;
    int32_t ldb_ = 0;
    int32_t ldc_ = 0;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t S_ = 0;
    int32_t uplo_ = 0;
    int32_t side_ = 0;
    float ar_ = 1.0f;
    float ai_ = 0.0f;
    float br_ = 0.0f;
    float bi_ = 0.0f;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
    int32_t colsPerCore_ = 0;
};

} // namespace csymm_small

extern "C" __global__ __aicore__ void csymm_small_kernel(
    __gm__ uint8_t* A, int32_t lda, int32_t uplo,
    __gm__ uint8_t* B, int32_t ldb,
    __gm__ uint8_t* C, int32_t ldc,
    int32_t m, int32_t n, int32_t S, int32_t side,
    float ar, float ai, float br, float bi)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    csymm_small::CsymmSmallKernel op;
    op.Init(A, lda, uplo, B, ldb, C, ldc, m, n, S, side, ar, ai, br, bi, &pipe);
    op.Process();
}

void csymm_small_do(
    uint32_t numBlocks, void* stream,
    GM_ADDR A, int32_t lda, int32_t uplo,
    GM_ADDR B, int32_t ldb,
    GM_ADDR C, int32_t ldc,
    int32_t m, int32_t n, int32_t S, int32_t side,
    float ar, float ai, float br, float bi)
{
    csymm_small_kernel<<<numBlocks, nullptr, stream>>>(
        A, lda, uplo, B, ldb, C, ldc, m, n, S, side, ar, ai, br, bi);
}

#endif
