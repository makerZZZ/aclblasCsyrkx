/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "lib/matmul_intf.h"
#include "common/helper/kernel_utils.h"
#include "../ssymm_common_kernel.h"

namespace {
using namespace AscendC;
using namespace matmul;
using namespace fp32;

// 这些定义从原私有 ssymm 头文件折叠进来。
// 它们只服务 ssymm 实现本身，保留在 kernel.cpp 中满足单算子文件组织要求。

// 统一分类一个 ssymm tile 在对称矩阵中的位置。
// RIGHT 中 colBase/colCount 表示 A(k,j) tile，LEFT 中复用 colBase 表示 rowBase。
__aicore__ inline SsymmTileKind ClassifySsymmTile(aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    uint32_t rowCount,
    uint32_t colBase,
    uint32_t colCount,
    uint32_t kBase,
    uint32_t kCount,
    const SsymmTileClassifyShape &tileShape)
{
    return SSYMM_CLASSIFY_TILE_KIND(side, uplo, rowCount, colBase, colCount, kBase, kCount, tileShape);
}

// device 侧按基础 chunk 类型分类 RIGHT tile。
// 返回值用于判断 A tile 可 direct 读、mirror 转置读，还是需要 mixed/tail 慢路径。
__aicore__ inline RightCubeChunkKind ClassifyRightCubeChunkDevice(RightCubeUplo uplo,
    uint32_t rowCount,
    uint32_t colBase,
    uint32_t colCount,
    uint32_t kBase,
    uint32_t kCount,
    const RightCubeTileShape &tileShape)
{
    const SsymmTileClassifyShape classifyShape{tileShape.tm, tileShape.tn, tileShape.tk};
    switch (SSYMM_CLASSIFY_TILE_KIND(ACLBLAS_SIDE_RIGHT,
        (uplo == RightCubeUplo::LOWER) ? ACLBLAS_LOWER : ACLBLAS_UPPER,
        rowCount, colBase, colCount, kBase, kCount, classifyShape)) {
        case SsymmTileKind::DirectFull:
            return RightCubeChunkKind::DIRECT_FULL;
        case SsymmTileKind::MirrorFull:
            return RightCubeChunkKind::MIRROR_FULL;
        case SsymmTileKind::MixedDiag:
            return RightCubeChunkKind::MIXED_DIAG;
        case SsymmTileKind::Tail:
        default:
            return RightCubeChunkKind::TAIL;
    }
}

// device 侧按执行粒度分类 RIGHT tile。
// 完整对角 mixed tile 会被单独标识，使 chunk-local cube 路径可以处理它。
__aicore__ inline RightCubeExecuteChunkKind ClassifyRightCubeExecuteChunkKindDevice(RightCubeUplo uplo,
    uint32_t rowCount,
    uint32_t colBase,
    uint32_t colCount,
    uint32_t kBase,
    uint32_t kCount,
    const RightCubeTileShape &tileShape)
{
    (void)rowCount;
    if (colCount < tileShape.tn || kCount < tileShape.tk) {
        return RightCubeExecuteChunkKind::TAIL;
    }

    RightCubeExecuteGeometry geometry = RightCubeExecuteGeometry::MIXED_DIAG;
    if (uplo == RightCubeUplo::LOWER) {
        if (kBase >= colBase + colCount - 1) {
            geometry = RightCubeExecuteGeometry::DIRECT;
        } else if (kBase + kCount - 1 < colBase) {
            geometry = RightCubeExecuteGeometry::MIRROR;
        }
    } else {
        if (kBase + kCount - 1 <= colBase) {
            geometry = RightCubeExecuteGeometry::DIRECT;
        } else if (kBase > colBase + colCount - 1) {
            geometry = RightCubeExecuteGeometry::MIRROR;
        }
    }
    if (geometry == RightCubeExecuteGeometry::DIRECT) {
        return RightCubeExecuteChunkKind::DIRECT_FULL;
    }
    if (geometry == RightCubeExecuteGeometry::MIRROR) {
        return RightCubeExecuteChunkKind::MIRROR_FULL;
    }
    if (colBase == kBase) {
        return RightCubeExecuteChunkKind::MIXED_DIAG_FULL_TILE;
    }
    return RightCubeExecuteChunkKind::MIXED_DIAG;
}

// 判断 RIGHT tile 的几何关系：direct、mirror 或 mixed diagonal。
// pack kernel 根据该结果选择读取 A 的原始方向或镜像方向。
__aicore__ inline RightCubeExecuteGeometry ClassifyRightCubeExecuteGeometryDevice(RightCubeUplo uplo,
    uint32_t colBase,
    uint32_t colCount,
    uint32_t kBase,
    uint32_t kCount)
{
    if (uplo == RightCubeUplo::LOWER) {
        if (kBase >= colBase + colCount - 1) {
            return RightCubeExecuteGeometry::DIRECT;
        }
        if (kBase + kCount - 1 < colBase) {
            return RightCubeExecuteGeometry::MIRROR;
        }
        return RightCubeExecuteGeometry::MIXED_DIAG;
    }
    if (kBase + kCount - 1 <= colBase) {
        return RightCubeExecuteGeometry::DIRECT;
    }
    if (kBase > colBase + colCount - 1) {
        return RightCubeExecuteGeometry::MIRROR;
    }
    return RightCubeExecuteGeometry::MIXED_DIAG;
}

// 判断 RIGHT tile 是否存在列尾、K 尾或两者都有。
// 该信息和几何关系组合后决定 pack 时使用快速路径还是通用路径。
__aicore__ inline RightCubeExecuteExtent ClassifyRightCubeExecuteExtentDevice(uint32_t colCount,
    uint32_t kCount,
    const RightCubeTileShape &tileShape)
{
    const bool colTail = colCount < tileShape.tn;
    const bool kTail = kCount < tileShape.tk;
    if (!colTail && !kTail) {
        return RightCubeExecuteExtent::FULL_TILE;
    }
    if (colTail && kTail) {
        return RightCubeExecuteExtent::COLK_TAIL;
    }
    return colTail ? RightCubeExecuteExtent::COL_TAIL : RightCubeExecuteExtent::K_TAIL;
}

// 根据几何关系和尾块情况选择 A chunk 的 pack 模式。
// 完整块优先走 FAST_*，尾块或通用 mixed 情况走 GENERIC_*。
__aicore__ inline RightCubePackMode DetermineRightCubePackModeDevice(RightCubeExecuteGeometry geometry,
    RightCubeExecuteExtent extent)
{
    if (extent == RightCubeExecuteExtent::FULL_TILE) {
        if (geometry == RightCubeExecuteGeometry::DIRECT) {
            return RightCubePackMode::FAST_DIRECT;
        }
        if (geometry == RightCubeExecuteGeometry::MIRROR) {
            return RightCubePackMode::FAST_MIRROR;
        }
        return RightCubePackMode::FAST_MIXED_FULL_TILE;
    }
    if (geometry == RightCubeExecuteGeometry::DIRECT) {
        return RightCubePackMode::GENERIC_DIRECT;
    }
    if (geometry == RightCubeExecuteGeometry::MIRROR) {
        return RightCubePackMode::GENERIC_MIRROR;
    }
    return RightCubePackMode::GENERIC_SYMMETRIC;
}

// LEFT 和 RIGHT 复用的操作数不同，因此 tile 形状按方向分别保留。
// 当前最优取值虽然一致，但分开定义便于后续独立调优。
constexpr uint32_t SSYMM_LEFT_TILE_M = 8;
constexpr uint32_t SSYMM_LEFT_TILE_K = 64;
constexpr uint32_t SSYMM_LEFT_TILE_N = 128;
constexpr uint16_t SSYMM_SYNC_EVT = 0;

enum class SsymmSideMode { LEFT, RIGHT };
enum class SsymmFillMode { LOWER, UPPER };
enum class RightATileLoadKind { DIRECT, MIRROR_TRANSPOSED, MIXED };

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
class SsymmKernel {
public:
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR tilingGm);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);
    __aicore__ inline T ReadLeftA(uint32_t row, uint32_t k) const;
    __aicore__ inline T ReadRightA(uint32_t k, uint32_t col) const;
    __aicore__ inline uint32_t GetPaddingNum(uint32_t elementCount) const;
    __aicore__ inline void CopyInLeftOutputBlock(uint32_t rowBase, uint32_t rowBlockCount,
        uint32_t colBase, uint32_t colCount, LocalTensor<T> outLocal);
    __aicore__ inline void CopyInLeftOperands(uint32_t rowBase, uint32_t rowBlockCount,
        uint32_t colBase, uint32_t colCount, uint32_t kBase, uint32_t kCount,
        LocalTensor<T> aPanelLocal, LocalTensor<T> bTileLocal);
    __aicore__ inline void BuildLeftAPanelRowDirectContig(uint32_t row, uint32_t panelOffset,
        uint32_t kBase, uint32_t kCount, LocalTensor<T> aPanelLocal);
    __aicore__ inline void BuildLeftAPanelRowMirrorGather(uint32_t row, uint32_t panelOffset,
        uint32_t kBase, uint32_t kCount, LocalTensor<T> aPanelLocal);
    __aicore__ inline void BuildLeftAPanelRowMixedSplit(uint32_t row, uint32_t panelOffset,
        uint32_t kBase, uint32_t kCount, LocalTensor<T> aPanelLocal);
    __aicore__ inline void BuildLeftAPanelRow(uint32_t row, uint32_t panelOffset,
        uint32_t kBase, uint32_t kCount, LocalTensor<T> aPanelLocal);
    __aicore__ inline void ComputeLeftBlockScalar(uint32_t rowBlockCount, uint32_t colCount, uint32_t kCount,
        LocalTensor<T> outLocal, LocalTensor<T> aPanelLocal, LocalTensor<T> bTileLocal);
    __aicore__ inline void ComputeLeftBlock(uint32_t rowBlockCount, uint32_t colCount, uint32_t kCount,
        LocalTensor<T> outLocal, LocalTensor<T> aPanelLocal, LocalTensor<T> bTileLocal);
    __aicore__ inline void CopyOutLeftOutputBlock(uint32_t rowBase, uint32_t rowBlockCount,
        uint32_t colBase, uint32_t colCount, LocalTensor<T> outLocal);
    __aicore__ inline void ProcessLeftFallbackTile(uint32_t rowBase, uint32_t rowBlockCount,
        uint32_t colBase, uint32_t colCount, uint32_t kBase, uint32_t kCount, LocalTensor<T> outLocal);
    __aicore__ inline void ProcessLeft();
    __aicore__ inline void CopyInRightOutputBlock(uint32_t rowBase, uint32_t rowBlockCount,
        uint32_t colBase, uint32_t colCount, LocalTensor<T> outLocal);
    __aicore__ inline void CopyInRightOperands(uint32_t rowBase, uint32_t rowBlockCount,
        uint32_t colBase, uint32_t colCount, uint32_t kBase, uint32_t kCount,
        LocalTensor<T> bPanelLocal, LocalTensor<T> aTileLocal);
    __aicore__ inline RightATileLoadKind DetermineRightATileLoadKind(uint32_t colBase,
        uint32_t colCount, uint32_t kBase, uint32_t kCount) const;
    __aicore__ inline void ComputeRightBlock(uint32_t rowBlockCount, uint32_t colCount, uint32_t kCount,
        LocalTensor<T> outLocal, LocalTensor<T> bPanelLocal, LocalTensor<T> aTileLocal,
        RightATileLoadKind aTileLoadKind);
    __aicore__ inline void CopyOutRightOutputBlock(uint32_t rowBase, uint32_t rowBlockCount,
        uint32_t colBase, uint32_t colCount, LocalTensor<T> outLocal);
    __aicore__ inline void ProcessRight();

    GlobalTensor<T> aGM;
    GlobalTensor<T> bGM;
    GlobalTensor<T> cGM;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> segQueue;
    TQue<QuePosition::VECIN, 1> tileQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;

    uint32_t blockIdx = 0;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t lda = 0;
    uint32_t ldb = 0;
    uint32_t ldc = 0;
    uint32_t kDim = 0;
    uint32_t rowStart = 0;
    uint32_t rowCount = 0;
    uint32_t rightChunkMask = SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::DIRECT_FULL) |
        SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::MIRROR_FULL) |
        SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::MIXED_DIAG) |
        SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::TAIL) |
        SSYMM_RIGHT_CUBE_EXEC_CHUNK_BIT(RightCubeExecuteChunkKind::MIXED_DIAG_FULL_TILE);
    uint32_t rightChunkTn = 0;
    uint32_t rightChunkTk = 0;
    T alpha = static_cast<T>(1.0f);
    T beta = static_cast<T>(0.0f);
    uint32_t elementsPerBlock = static_cast<uint32_t>(NUM_ELE_PERBLOCK);
};

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 从 GM tiling 区解析当前 block 需要的 shape、行范围和 RIGHT chunk mask。
// host 已经按 block 写好 startRow/rowCount，device 这里只取本 block 的切片。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::ParseTilingData(GM_ADDR tilingGm)
{
    auto tiling = reinterpret_cast<__gm__ SsymmTilingData *>(tilingGm);
    m = tiling->m;
    n = tiling->n;
    lda = tiling->lda;
    ldb = tiling->ldb;
    ldc = tiling->ldc;
    kDim = tiling->kDim;
    rightChunkMask = tiling->rightChunkMask;
    rightChunkTn = tiling->rightChunkTn;
    rightChunkTk = tiling->rightChunkTk;
    alpha = static_cast<T>(tiling->alpha);
    beta = static_cast<T>(tiling->beta);

    // 根据 rowsPerCore 和 rowRemainder 计算当前核的行范围
    // 前 rowRemainder 个核分配 (rowsPerCore + 1) 行，其余核分配 rowsPerCore 行
    if (blockIdx < tiling->rowRemainder) {
        rowStart = blockIdx * (tiling->rowsPerCore + 1);
        rowCount = tiling->rowsPerCore + 1;
    } else if (blockIdx < tiling->useCoreNum) {
        rowStart = tiling->rowRemainder * (tiling->rowsPerCore + 1) +
                   (blockIdx - tiling->rowRemainder) * tiling->rowsPerCore;
        rowCount = tiling->rowsPerCore;
    } else {
        // 超出范围的 block 不处理任何行
        rowStart = 0;
        rowCount = 0;
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 初始化 fallback 模板 kernel 的 GM tensor 和 UB 队列。
// RIGHT/LEFT 使用不同 tile 形状，因此根据模板参数初始化不同的本地 buffer 容量。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::Init(GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR tilingGm)
{
    blockIdx = static_cast<uint32_t>(GetBlockIdx());
    ParseTilingData(tilingGm);

    uint32_t aDim = (SIDE == SsymmSideMode::LEFT) ? m : n;
    aGM.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(A), aDim * lda);
    bGM.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(B), m * ldb);
    cGM.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(C), m * ldc);

    if constexpr (SIDE == SsymmSideMode::RIGHT) {
        pipe.InitBuffer(segQueue, 1, SSYMM_RIGHT_TILE_M * SSYMM_RIGHT_TILE_K * sizeof(T));
        pipe.InitBuffer(tileQueue, 1, SSYMM_RIGHT_TILE_K * SSYMM_RIGHT_FALLBACK_BUFFER_STRIDE_N * sizeof(T));
        pipe.InitBuffer(outQueue, 1, SSYMM_RIGHT_TILE_M * SSYMM_RIGHT_FALLBACK_BUFFER_STRIDE_N * sizeof(T));
    } else {
        pipe.InitBuffer(segQueue, 1, SSYMM_LEFT_TILE_M * SSYMM_LEFT_TILE_K * sizeof(T));
        pipe.InitBuffer(tileQueue, 1, SSYMM_LEFT_TILE_K * SSYMM_LEFT_TILE_N * sizeof(T));
        pipe.InitBuffer(outQueue, 1, SSYMM_LEFT_TILE_M * SSYMM_LEFT_TILE_N * sizeof(T));
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 按 LEFT 语义读取对称矩阵 A(row,k)。
// 只有 uplo 指定的半边真实存储，另一半通过镜像索引读取。
__aicore__ inline T SsymmKernel<T, SIDE, UPLO>::ReadLeftA(uint32_t row, uint32_t k) const
{
    if constexpr (UPLO == SsymmFillMode::LOWER) {
        return (row >= k) ? aGM.GetValue(row * lda + k) : aGM.GetValue(k * lda + row);
    } else {
        return (row <= k) ? aGM.GetValue(row * lda + k) : aGM.GetValue(k * lda + row);
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 按 RIGHT 语义读取对称矩阵 A(k,col)。
// 与 LEFT 一样只依赖一个有效半边，另一半通过镜像索引恢复。
__aicore__ inline T SsymmKernel<T, SIDE, UPLO>::ReadRightA(uint32_t k, uint32_t col) const
{
    if constexpr (UPLO == SsymmFillMode::LOWER) {
        return (k >= col) ? aGM.GetValue(k * lda + col) : aGM.GetValue(col * lda + k);
    } else {
        return (k <= col) ? aGM.GetValue(k * lda + col) : aGM.GetValue(col * lda + k);
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 计算 DataCopyPad 需要补齐的元素个数。
// 结果让非整块尾部按 AscendC block 粒度对齐搬运。
__aicore__ inline uint32_t SsymmKernel<T, SIDE, UPLO>::GetPaddingNum(uint32_t elementCount) const
{
    return static_cast<uint32_t>(fp32::ROUND(elementCount, elementsPerBlock) - elementCount);
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 将 LEFT fallback 的 C 输出 tile 搬入 UB，并先乘 beta。
// 后续 K 维循环会把 alpha*A*B 的累加结果加到这个本地输出 tile 上。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::CopyInLeftOutputBlock(uint32_t rowBase,
    uint32_t rowBlockCount, uint32_t colBase, uint32_t colCount, LocalTensor<T> outLocal)
{
    for (uint32_t localRow = 0; localRow < rowBlockCount; ++localRow) {
        uint32_t row = rowBase + localRow;
        uint32_t outOffset = localRow * SSYMM_LEFT_TILE_N;
        if (colCount == SSYMM_LEFT_TILE_N) {
            DataCopy(outLocal[outOffset], cGM[row * ldc + colBase], colCount);
        } else {
            uint8_t paddingNum = static_cast<uint8_t>(GetPaddingNum(colCount));
            DataCopyExtParams copyParams{1, static_cast<uint16_t>(colCount * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
            DataCopyPad(outLocal[outOffset], cGM[row * ldc + colBase], copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();
        Muls(outLocal[outOffset], outLocal[outOffset], beta, colCount);
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 构造 LEFT A panel 中一行的 direct 连续片段。
// 当该片段完整落在已存储半边内时，可以直接 DataCopy 后统一乘 alpha。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::BuildLeftAPanelRowDirectContig(
    uint32_t row, uint32_t panelOffset, uint32_t kBase, uint32_t kCount, LocalTensor<T> aPanelLocal)
{
    if (kCount == SSYMM_LEFT_TILE_K) {
        DataCopy(aPanelLocal[panelOffset], aGM[row * lda + kBase], kCount);
    } else {
        const uint8_t paddingNum = static_cast<uint8_t>(GetPaddingNum(kCount));
        const DataCopyExtParams copyParams{1, static_cast<uint16_t>(kCount * sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
        DataCopyPad(aPanelLocal[panelOffset], aGM[row * lda + kBase], copyParams, padParams);
    }
    PipeBarrier<PIPE_ALL>();
    Muls(aPanelLocal[panelOffset], aPanelLocal[panelOffset], alpha, kCount);
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 构造 LEFT A panel 中一行的 mirror gather 片段。
// 逐元素从镜像位置读取 A(k,row)，再对整行乘 alpha。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::BuildLeftAPanelRowMirrorGather(
    uint32_t row, uint32_t panelOffset, uint32_t kBase, uint32_t kCount, LocalTensor<T> aPanelLocal)
{
    for (uint32_t t = 0; t < kCount; ++t) {
        const uint32_t k = kBase + t;
        aPanelLocal.SetValue(panelOffset + t, aGM.GetValue(k * lda + row));
    }
    PipeBarrier<PIPE_ALL>();
    Muls(aPanelLocal[panelOffset], aPanelLocal[panelOffset], alpha, kCount);
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 构造跨对角线的 LEFT A panel 行。
// direct 部分尽量连续搬运，mirror 部分逐元素补齐，最后整体乘 alpha。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::BuildLeftAPanelRowMixedSplit(
    uint32_t row, uint32_t panelOffset, uint32_t kBase, uint32_t kCount, LocalTensor<T> aPanelLocal)
{
    const SsymmLeftPanelUplo panelUplo = (UPLO == SsymmFillMode::LOWER)
        ? SsymmLeftPanelUplo::LOWER
        : SsymmLeftPanelUplo::UPPER;
    const uint32_t directOffset = SSYMM_LEFT_PANEL_DIRECT_OFFSET(panelUplo, row, kBase, kCount);
    const uint32_t directCount = SSYMM_LEFT_PANEL_DIRECT_COUNT(panelUplo, row, kBase, kCount);

    if (panelUplo == SsymmLeftPanelUplo::LOWER) {
        if (directCount > 0) {
            const uint8_t paddingNum = static_cast<uint8_t>(GetPaddingNum(directCount));
            const DataCopyExtParams copyParams{1, static_cast<uint16_t>(directCount * sizeof(T)), 0, 0, 0};
            const DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
            DataCopyPad(aPanelLocal[panelOffset], aGM[row * lda + kBase], copyParams, padParams);
        }
        for (uint32_t t = directCount; t < kCount; ++t) {
            const uint32_t k = kBase + t;
            aPanelLocal.SetValue(panelOffset + t, aGM.GetValue(k * lda + row));
        }
    } else {
        for (uint32_t t = 0; t < directOffset; ++t) {
            const uint32_t k = kBase + t;
            aPanelLocal.SetValue(panelOffset + t, aGM.GetValue(k * lda + row));
        }
        for (uint32_t t = directOffset; t < kCount; ++t) {
            const uint32_t k = kBase + t;
            aPanelLocal.SetValue(panelOffset + t, aGM.GetValue(row * lda + k));
        }
    }
    PipeBarrier<PIPE_ALL>();
    Muls(aPanelLocal[panelOffset], aPanelLocal[panelOffset], alpha, kCount);
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 根据当前 row 和 k 区间选择 LEFT A panel 的构造策略。
// direct、mirror、mixed 三种策略共享同一个 aPanelLocal 输出布局。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::BuildLeftAPanelRow(
    uint32_t row, uint32_t panelOffset, uint32_t kBase, uint32_t kCount, LocalTensor<T> aPanelLocal)
{
    const SsymmLeftPanelUplo panelUplo = (UPLO == SsymmFillMode::LOWER)
        ? SsymmLeftPanelUplo::LOWER
        : SsymmLeftPanelUplo::UPPER;
    const SsymmLeftPanelBuildKind buildKind =
        SSYMM_LEFT_PANEL_CLASSIFY_KIND(panelUplo, row, kBase, kCount);
    const SsymmLeftAPanelStrategy panelStrategy = SSYMM_LEFT_PANEL_SELECT_STRATEGY(buildKind);
    if (panelStrategy == SsymmLeftAPanelStrategy::DIRECT_CONTIG) {
        BuildLeftAPanelRowDirectContig(row, panelOffset, kBase, kCount, aPanelLocal);
        return;
    }
    if (panelStrategy == SsymmLeftAPanelStrategy::MIRROR_GATHER) {
        BuildLeftAPanelRowMirrorGather(row, panelOffset, kBase, kCount, aPanelLocal);
        return;
    }
    BuildLeftAPanelRowMixedSplit(row, panelOffset, kBase, kCount, aPanelLocal);
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 搬入 LEFT fallback 计算需要的 A panel 和 B tile。
// B 按 K 行连续搬运，A 通过对称半边分类构造成本地 panel。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::CopyInLeftOperands(uint32_t rowBase,
    uint32_t rowBlockCount, uint32_t colBase, uint32_t colCount, uint32_t kBase, uint32_t kCount,
    LocalTensor<T> aPanelLocal, LocalTensor<T> bTileLocal)
{
    for (uint32_t t = 0; t < kCount; ++t) {
        uint32_t k = kBase + t;
        uint32_t tileOffset = t * SSYMM_LEFT_TILE_N;
        if (colCount == SSYMM_LEFT_TILE_N) {
            DataCopy(bTileLocal[tileOffset], bGM[k * ldb + colBase], colCount);
        } else {
            uint8_t paddingNum = static_cast<uint8_t>(GetPaddingNum(colCount));
            DataCopyExtParams copyParams{1, static_cast<uint16_t>(colCount * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
            DataCopyPad(bTileLocal[tileOffset], bGM[k * ldb + colBase], copyParams, padParams);
        }
    }

    PipeBarrier<PIPE_ALL>();
    for (uint32_t localRow = 0; localRow < rowBlockCount; ++localRow) {
        uint32_t row = rowBase + localRow;
        uint32_t panelOffset = localRow * SSYMM_LEFT_TILE_K;
        BuildLeftAPanelRow(row, panelOffset, kBase, kCount, aPanelLocal);
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 计算一个 LEFT fallback tile 的局部乘加（纯标量路径）。
// 输入为本地 A panel、B tile 和已乘 beta 的 outLocal。
// 不在此函数内分配任何额外 UB buffer，避免与外层 segQueue/tileQueue 产生冲突。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::ComputeLeftBlockScalar(uint32_t rowBlockCount,
    uint32_t colCount, uint32_t kCount, LocalTensor<T> outLocal,
    LocalTensor<T> aPanelLocal, LocalTensor<T> bTileLocal)
{
    for (uint32_t localRow = 0; localRow < rowBlockCount; ++localRow) {
        const uint32_t outOffset = localRow * SSYMM_LEFT_TILE_N;
        const uint32_t panelOffset = localRow * SSYMM_LEFT_TILE_K;

        for (uint32_t j = 0; j < colCount; ++j) {
            T sum = static_cast<T>(0);
            for (uint32_t t = 0; t < kCount; ++t) {
                sum += aPanelLocal.GetValue(panelOffset + t) *
                       bTileLocal.GetValue(t * SSYMM_LEFT_TILE_N + j);
            }
            T outVal = outLocal.GetValue(outOffset + j);
            outLocal.SetValue(outOffset + j, outVal + sum);
        }
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// LEFT fallback 的计算入口。
// 当前实现直接复用标量路径，保留封装便于后续替换为更快实现。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::ComputeLeftBlock(uint32_t rowBlockCount,
    uint32_t colCount, uint32_t kCount, LocalTensor<T> outLocal,
    LocalTensor<T> aPanelLocal, LocalTensor<T> bTileLocal)
{
    ComputeLeftBlockScalar(rowBlockCount, colCount, kCount, outLocal, aPanelLocal, bTileLocal);
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 将 LEFT fallback 的本地输出 tile 写回 C。
// 完整列块走 DataCopy，尾列块走 DataCopyPad。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::CopyOutLeftOutputBlock(uint32_t rowBase,
    uint32_t rowBlockCount, uint32_t colBase, uint32_t colCount, LocalTensor<T> outLocal)
{
    for (uint32_t localRow = 0; localRow < rowBlockCount; ++localRow) {
        uint32_t row = rowBase + localRow;
        uint32_t outOffset = localRow * SSYMM_LEFT_TILE_N;
        if (colCount == SSYMM_LEFT_TILE_N) {
            DataCopy(cGM[row * ldc + colBase], outLocal[outOffset], colCount);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint16_t>(colCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(cGM[row * ldc + colBase], outLocal[outOffset], copyParams);
        }
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 处理 LEFT fallback 的一个 K tile。
// 函数内部临时申请 A/B 本地 tensor，完成搬入、计算和释放。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::ProcessLeftFallbackTile(uint32_t rowBase,
    uint32_t rowBlockCount, uint32_t colBase, uint32_t colCount, uint32_t kBase, uint32_t kCount,
    LocalTensor<T> outLocal)
{
    LocalTensor<T> aPanelLocal = segQueue.AllocTensor<T>();
    LocalTensor<T> bTileLocal = tileQueue.AllocTensor<T>();
    CopyInLeftOperands(rowBase, rowBlockCount, colBase, colCount, kBase, kCount, aPanelLocal, bTileLocal);
    ComputeLeftBlockScalar(rowBlockCount, colCount, kCount, outLocal, aPanelLocal, bTileLocal);
    segQueue.FreeTensor(aPanelLocal);
    tileQueue.FreeTensor(bTileLocal);
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 执行完整 LEFT fallback 流程。
// 外层遍历本 block 负责的 C 行和列块，内层遍历 K 维并累加到 outLocal。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::ProcessLeft()
{
    // LEFT fallback 保持通用标量路径，regular dense 优化形状已在 host 侧分派到 LeftCube。
    uint32_t rowEnd = rowStart + rowCount;
    for (uint32_t rowBase = rowStart; rowBase < rowEnd; rowBase += SSYMM_LEFT_TILE_M) {
        uint32_t rowBlockCount = SSYMM_LEFT_TILE_M;
        if (rowBase + rowBlockCount > rowEnd) {
            rowBlockCount = rowEnd - rowBase;
        }
        for (uint32_t colBase = 0; colBase < n; colBase += SSYMM_LEFT_TILE_N) {
            uint32_t colCount = SSYMM_LEFT_TILE_N;
            if (colBase + colCount > n) {
                colCount = n - colBase;
            }

            LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
            // 输出 tile 在完整 K 维归约结束前一直保留在 UB 中，最后再对齐写回。
            CopyInLeftOutputBlock(rowBase, rowBlockCount, colBase, colCount, outLocal);
            for (uint32_t kBase = 0; kBase < kDim; kBase += SSYMM_LEFT_TILE_K) {
                uint32_t kCount = SSYMM_LEFT_TILE_K;
                if (kBase + kCount > kDim) {
                    kCount = kDim - kBase;
                }
                ProcessLeftFallbackTile(rowBase, rowBlockCount, colBase, colCount, kBase, kCount, outLocal);
            }
            CopyOutLeftOutputBlock(rowBase, rowBlockCount, colBase, colCount, outLocal);
            // outLocal（深度 1 队列）会被下一个 tile 的 CopyInLeftOutputBlock 重新写入，
            // 必须先等本 tile 的 CopyOut（MTE3 读）完成，避免 MTE2 写与 MTE3 读竞争。
            SetFlag<HardEvent::MTE3_MTE2>(SSYMM_SYNC_EVT);
            WaitFlag<HardEvent::MTE3_MTE2>(SSYMM_SYNC_EVT);
            outQueue.FreeTensor(outLocal);
        }
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 将 RIGHT fallback 的 C 输出 tile 搬入 UB，并先乘 beta。
// 后续 K 维循环会把 alpha*B*A 的累加结果加到这个本地输出 tile 上。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::CopyInRightOutputBlock(uint32_t rowBase,
    uint32_t rowBlockCount, uint32_t colBase, uint32_t colCount, LocalTensor<T> outLocal)
{
    for (uint32_t localRow = 0; localRow < rowBlockCount; ++localRow) {
        uint32_t row = rowBase + localRow;
        uint32_t outOffset = localRow * SSYMM_RIGHT_FALLBACK_BUFFER_STRIDE_N;
        if (colCount == SSYMM_RIGHT_DISPATCH_TILE_N) {
            DataCopy(outLocal[outOffset], cGM[row * ldc + colBase], colCount);
        } else {
            uint8_t paddingNum = static_cast<uint8_t>(GetPaddingNum(colCount));
            DataCopyExtParams copyParams{1, static_cast<uint16_t>(colCount * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
            DataCopyPad(outLocal[outOffset], cGM[row * ldc + colBase], copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();
        Muls(outLocal[outOffset], outLocal[outOffset], beta, colCount);
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 判断 RIGHT A tile 在 fallback 中应如何搬入 UB。
// direct/mirror 完整块可连续搬运，mixed 块需要逐元素按对称语义读取。
__aicore__ inline RightATileLoadKind SsymmKernel<T, SIDE, UPLO>::DetermineRightATileLoadKind(
        uint32_t colBase, uint32_t colCount, uint32_t kBase, uint32_t kCount) const
{
    RightCubeTileShape tileShape{
        SSYMM_RIGHT_TILE_M,
        SSYMM_RIGHT_DISPATCH_TILE_N,
        SSYMM_RIGHT_TILE_K,
    };
    RightCubeUplo uplo = (UPLO == SsymmFillMode::LOWER) ? RightCubeUplo::LOWER : RightCubeUplo::UPPER;
    RightCubeChunkKind chunkKind = ClassifyRightCubeChunkDevice(
        uplo, SSYMM_RIGHT_TILE_M, colBase, colCount, kBase, kCount, tileShape);
    RightCubeLoadKind loadKind = SSYMM_RIGHT_CUBE_LOAD_KIND_FROM_CHUNK(chunkKind);
    if (loadKind == RightCubeLoadKind::DIRECT) {
        return RightATileLoadKind::DIRECT;
    }
    if (loadKind == RightCubeLoadKind::MIRROR_TRANSPOSED) {
        return RightATileLoadKind::MIRROR_TRANSPOSED;
    }
    return RightATileLoadKind::MIXED;
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 搬入 RIGHT fallback 计算需要的 B panel 和 A tile。
// B panel 会先乘 alpha，A tile 根据 direct/mirror/mixed 分类选择搬运方式。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::CopyInRightOperands(uint32_t rowBase,
    uint32_t rowBlockCount, uint32_t colBase, uint32_t colCount, uint32_t kBase, uint32_t kCount,
    LocalTensor<T> bPanelLocal, LocalTensor<T> aTileLocal)
{
    // 上一轮 K 迭代的 Muls(bPanelLocal, alpha) 是向量写，与本轮 DataCopy 的 MTE2 写
    // 竞争同一 bPanelLocal 缓冲，必须先等向量流水线完成再发起搬入。
    SetFlag<HardEvent::V_MTE2>(SSYMM_SYNC_EVT);
    WaitFlag<HardEvent::V_MTE2>(SSYMM_SYNC_EVT);

    for (uint32_t localRow = 0; localRow < rowBlockCount; ++localRow) {
        uint32_t row = rowBase + localRow;
        uint32_t panelOffset = localRow * SSYMM_RIGHT_TILE_K;
        if (kCount == SSYMM_RIGHT_TILE_K) {
            DataCopy(bPanelLocal[panelOffset], bGM[row * ldb + kBase], kCount);
        } else {
            uint8_t paddingNum = static_cast<uint8_t>(GetPaddingNum(kCount));
            DataCopyExtParams copyParams{1, static_cast<uint16_t>(kCount * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
            DataCopyPad(bPanelLocal[panelOffset], bGM[row * ldb + kBase], copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();
        Muls(bPanelLocal[panelOffset], bPanelLocal[panelOffset], alpha, kCount);
    }

    RightATileLoadKind aTileLoadKind = DetermineRightATileLoadKind(colBase, colCount, kBase, kCount);
    if (aTileLoadKind == RightATileLoadKind::DIRECT) {
        // 完整落在已存储半边内的 A tile 可以按行连续搬运。
        for (uint32_t t = 0; t < kCount; ++t) {
            uint32_t k = kBase + t;
            uint32_t tileOffset = t * SSYMM_RIGHT_FALLBACK_BUFFER_STRIDE_N;
            if (colCount == SSYMM_RIGHT_DISPATCH_TILE_N) {
                DataCopy(aTileLocal[tileOffset], aGM[k * lda + colBase], colCount);
            } else {
                uint8_t paddingNum = static_cast<uint8_t>(GetPaddingNum(colCount));
                DataCopyExtParams copyParams{1, static_cast<uint16_t>(colCount * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
                DataCopyPad(aTileLocal[tileOffset], aGM[k * lda + colBase], copyParams, padParams);
            }
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    if (aTileLoadKind == RightATileLoadKind::MIRROR_TRANSPOSED) {
        // 完整落在镜像半边的 A tile 可按 A[col,kRange] 连续搬运，并按转置布局消费。
        for (uint32_t j = 0; j < colCount; ++j) {
            uint32_t col = colBase + j;
            uint32_t tileOffset = j * SSYMM_RIGHT_TILE_K;
            if (kCount == SSYMM_RIGHT_TILE_K) {
                DataCopy(aTileLocal[tileOffset], aGM[col * lda + kBase], kCount);
            } else {
                uint8_t paddingNum = static_cast<uint8_t>(GetPaddingNum(kCount));
                DataCopyExtParams copyParams{1, static_cast<uint16_t>(kCount * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};
                DataCopyPad(aTileLocal[tileOffset], aGM[col * lda + kBase], copyParams, padParams);
            }
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }

    for (uint32_t t = 0; t < kCount; ++t) {
        uint32_t k = kBase + t;
        for (uint32_t j = 0; j < colCount; ++j) {
            aTileLocal.SetValue(t * SSYMM_RIGHT_FALLBACK_BUFFER_STRIDE_N + j, ReadRightA(k, colBase + j));
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 计算一个 RIGHT fallback tile 的局部乘加（纯标量路径）。
// mirror-transposed 布局需要按 j*K+t 读取 A，其余布局按 t*stride+j 读取。
// 不在此函数内分配任何额外 UB buffer，避免与外层 segQueue/tileQueue 产生冲突。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::ComputeRightBlock(uint32_t rowBlockCount,
    uint32_t colCount, uint32_t kCount, LocalTensor<T> outLocal,
    LocalTensor<T> bPanelLocal, LocalTensor<T> aTileLocal, RightATileLoadKind aTileLoadKind)
{
    for (uint32_t localRow = 0; localRow < rowBlockCount; ++localRow) {
        const uint32_t outOffset = localRow * SSYMM_RIGHT_FALLBACK_BUFFER_STRIDE_N;
        const uint32_t panelOffset = localRow * SSYMM_RIGHT_TILE_K;

        for (uint32_t j = 0; j < colCount; ++j) {
            T sum = static_cast<T>(0);
            for (uint32_t t = 0; t < kCount; ++t) {
                T aVal;
                if (aTileLoadKind == RightATileLoadKind::MIRROR_TRANSPOSED) {
                    aVal = aTileLocal.GetValue(j * SSYMM_RIGHT_TILE_K + t);
                } else {
                    aVal = aTileLocal.GetValue(t * SSYMM_RIGHT_FALLBACK_BUFFER_STRIDE_N + j);
                }
                sum += bPanelLocal.GetValue(panelOffset + t) * aVal;
            }
            T outVal = outLocal.GetValue(outOffset + j);
            outLocal.SetValue(outOffset + j, outVal + sum);
        }
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 将 RIGHT fallback 的本地输出 tile 写回 C。
// 完整列块走 DataCopy，尾列块走 DataCopyPad。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::CopyOutRightOutputBlock(uint32_t rowBase,
    uint32_t rowBlockCount, uint32_t colBase, uint32_t colCount, LocalTensor<T> outLocal)
{
    for (uint32_t localRow = 0; localRow < rowBlockCount; ++localRow) {
        uint32_t row = rowBase + localRow;
        uint32_t outOffset = localRow * SSYMM_RIGHT_FALLBACK_BUFFER_STRIDE_N;
        if (colCount == SSYMM_RIGHT_DISPATCH_TILE_N) {
            DataCopy(cGM[row * ldc + colBase], outLocal[outOffset], colCount);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint16_t>(colCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(cGM[row * ldc + colBase], outLocal[outOffset], copyParams);
        }
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 执行完整 RIGHT fallback 流程。
// 按 C 的行块和列块遍历，K 维 chunk 根据 rightChunkMask 决定是否参与本次计算。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::ProcessRight()
{
    // RIGHT 计算 C[rowBlock,colBlock] += B[rowBlock,kBlock] * A_sym[kBlock,colBlock]。
    // 对称 A tile 只依赖输出列和 K 块，因此同一 A tile 可被行块内所有行复用。
    uint32_t chunkColStep = (rightChunkTn == 0) ? SSYMM_RIGHT_DISPATCH_TILE_N : rightChunkTn;
    uint32_t chunkKStep = (rightChunkTk == 0) ? SSYMM_RIGHT_TILE_K : rightChunkTk;
    RightCubeTileShape chunkTileShape{
        SSYMM_RIGHT_TILE_M,
        chunkColStep,
        chunkKStep,
    };
    RightCubeUplo chunkUplo = (UPLO == SsymmFillMode::LOWER) ? RightCubeUplo::LOWER : RightCubeUplo::UPPER;
    uint32_t rowEnd = rowStart + rowCount;
    for (uint32_t rowBase = rowStart; rowBase < rowEnd; rowBase += SSYMM_RIGHT_TILE_M) {
        uint32_t rowBlockCount = SSYMM_RIGHT_TILE_M;
        if (rowBase + rowBlockCount > rowEnd) {
            rowBlockCount = rowEnd - rowBase;
        }
        for (uint32_t colBase = 0; colBase < n; colBase += chunkColStep) {
            uint32_t colCount = chunkColStep;
            if (colBase + colCount > n) {
                colCount = n - colBase;
            }

            LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
            // 输出 tile 在完整 K 维归约结束前一直保留在 UB 中，最后再对齐写回。
            CopyInRightOutputBlock(rowBase, rowBlockCount, colBase, colCount, outLocal);
            for (uint32_t kBase = 0; kBase < kDim; kBase += chunkKStep) {
                uint32_t kCount = chunkKStep;
                if (kBase + kCount > kDim) {
                    kCount = kDim - kBase;
                }
                RightCubeExecuteChunkKind executeKind = ClassifyRightCubeExecuteChunkKindDevice(
                    chunkUplo, rowBlockCount, colBase, colCount, kBase, kCount, chunkTileShape);
                if (!SSYMM_RIGHT_CUBE_EXEC_MASK_HAS(rightChunkMask, executeKind)) {
                    continue;
                }
                LocalTensor<T> bPanelLocal = segQueue.AllocTensor<T>();
                LocalTensor<T> aTileLocal = tileQueue.AllocTensor<T>();
                RightATileLoadKind aTileLoadKind = DetermineRightATileLoadKind(colBase, colCount, kBase, kCount);
                CopyInRightOperands(rowBase, rowBlockCount, colBase, colCount, kBase, kCount, bPanelLocal, aTileLocal);
                ComputeRightBlock(rowBlockCount, colCount, kCount, outLocal, bPanelLocal, aTileLocal, aTileLoadKind);
                segQueue.FreeTensor(bPanelLocal);
                tileQueue.FreeTensor(aTileLocal);
            }
            CopyOutRightOutputBlock(rowBase, rowBlockCount, colBase, colCount, outLocal);
            // 同 LEFT 路径：outLocal 单缓冲跨 tile 复用，先等 MTE3 读完成再回收。
            SetFlag<HardEvent::MTE3_MTE2>(SSYMM_SYNC_EVT);
            WaitFlag<HardEvent::MTE3_MTE2>(SSYMM_SYNC_EVT);
            outQueue.FreeTensor(outLocal);
        }
    }
}

template <typename T, SsymmSideMode SIDE, SsymmFillMode UPLO>
// 根据模板参数选择 LEFT 或 RIGHT fallback 主流程。
// 如果当前 block 没有负责的行，则直接返回。
__aicore__ inline void SsymmKernel<T, SIDE, UPLO>::Process()
{
    if (rowCount == 0) {
        return;
    }

    if constexpr (SIDE == SsymmSideMode::RIGHT) {
        ProcessRight();
        return;
    }
    ProcessLeft();
}

}  // anonymous namespace

// fallback kernel 的 host launch 入口声明。
// 具体定义在文件末尾，根据 side/uplo 分发到四个模板实例。
void ssymm_kernel_do(GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR tilingGm,
                     aclblasSideMode_t side, aclblasFillMode_t uplo,
                     uint32_t numBlocks, void *stream);

// Kernel 入口函数必须在全局命名空间中
__global__ __aicore__ void ssymm_left_lower_fallback_kernel(GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SsymmKernel<float, SsymmSideMode::LEFT, SsymmFillMode::LOWER> op;
    op.Init(A, B, C, tilingGm);
    op.Process();
}

// LEFT/UPPER 通用 fallback kernel 实例。
// 它只负责创建模板对象、初始化 GM/UB 状态并执行 Process。
__global__ __aicore__ void ssymm_left_upper_fallback_kernel(GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SsymmKernel<float, SsymmSideMode::LEFT, SsymmFillMode::UPPER> op;
    op.Init(A, B, C, tilingGm);
    op.Process();
}

// RIGHT/LOWER 通用 fallback kernel 实例。
// 它只负责创建模板对象、初始化 GM/UB 状态并执行 Process。
__global__ __aicore__ void ssymm_right_lower_kernel(GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SsymmKernel<float, SsymmSideMode::RIGHT, SsymmFillMode::LOWER> op;
    op.Init(A, B, C, tilingGm);
    op.Process();
}

// RIGHT/UPPER 通用 fallback kernel 实例。
// 它只负责创建模板对象、初始化 GM/UB 状态并执行 Process。
__global__ __aicore__ void ssymm_right_upper_kernel(GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR tilingGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SsymmKernel<float, SsymmSideMode::RIGHT, SsymmFillMode::UPPER> op;
    op.Init(A, B, C, tilingGm);
    op.Process();
}

using RightCubeUnifiedMatmulType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulShapeParams kRightCubeUnifiedShape{256, 256, 256, 128, 128, 64};
constexpr MatmulFuncParams kRightCubeUnifiedFunc{
    false, false, false, false, 0, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, true, true};
constexpr MatmulBiasParams kRightCubeUnifiedBias{false};
constexpr MatmulConfig kRightCubeUnifiedConfig =
    GetMMConfig<MatmulConfigMode::CONFIG_MDL>(kRightCubeUnifiedShape, kRightCubeUnifiedFunc, kRightCubeUnifiedBias);
constexpr MatmulApiStaticTiling kRightCubeUnifiedStatic =
    GetMatmulApiTiling<RightCubeUnifiedMatmulType, RightCubeUnifiedMatmulType,
        RightCubeUnifiedMatmulType, RightCubeUnifiedMatmulType>(kRightCubeUnifiedConfig);
using RightCubeUnifiedMatmul = MatmulImpl<RightCubeUnifiedMatmulType, RightCubeUnifiedMatmulType,
    RightCubeUnifiedMatmulType, RightCubeUnifiedMatmulType, kRightCubeUnifiedStatic>;

// RIGHT cube 的 chunk pack kernel。
// 它把当前 chunk 的 A 对称块和 B 面板打包到连续 workspace，供后续 cube dense 消费。
__global__ __aicore__ void ssymm_right_cube_unified_chunk_pack_kernel(
    GM_ADDR aSym, GM_ADDR b, GM_ADDR workspace, GM_ADDR configGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto config = reinterpret_cast<__gm__ SsymmRightCubeUnifiedConfig *>(configGm);
    GlobalTensor<float> aSymGlobal;
    GlobalTensor<float> bGlobal;
    aSymGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aSym), static_cast<uint64_t>(config->n) * config->lda);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(b), static_cast<uint64_t>(config->m) * config->ldb);

    __gm__ float *workspaceBase = reinterpret_cast<__gm__ float *>(workspace);
    const uint64_t aChunkCapacity = static_cast<uint64_t>(config->tk) * config->tn;
    GlobalTensor<float> aChunkGlobal;
    GlobalTensor<float> bChunkGlobal;
    aChunkGlobal.SetGlobalBuffer(workspaceBase, static_cast<uint64_t>(config->chunkKCount) * config->chunkColCount);
    bChunkGlobal.SetGlobalBuffer(workspaceBase + aChunkCapacity,
        static_cast<uint64_t>(config->m) * config->chunkKCount);

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    const RightCubeTileShape chunkTileShape{
        config->tm,
        config->tn,
        config->tk,
    };
    const RightCubeExecuteChunkKind executeKind = ClassifyRightCubeExecuteChunkKindDevice(
        static_cast<RightCubeUplo>(config->uplo),
        config->tm,
        config->chunkColBase,
        config->chunkColCount,
        config->chunkKBase,
        config->chunkKCount,
        chunkTileShape);
    const RightCubeExecuteGeometry geometry = ClassifyRightCubeExecuteGeometryDevice(
        static_cast<RightCubeUplo>(config->uplo),
        config->chunkColBase,
        config->chunkColCount,
        config->chunkKBase,
        config->chunkKCount);
    const RightCubeExecuteExtent extent = ClassifyRightCubeExecuteExtentDevice(
        config->chunkColCount,
        config->chunkKCount,
        chunkTileShape);
    const RightCubePackMode packMode = DetermineRightCubePackModeDevice(geometry, extent);

    // 使用 TPipe 和 TQue 进行向量化搬运
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> bQueue;
    const uint32_t alignedBSize = ((config->chunkKCount * sizeof(float) + 31) / 32) * 32;
    pipe.InitBuffer(bQueue, 1, alignedBSize);
    LocalTensor<float> bRowLocal = bQueue.AllocTensor<float>();

    // B 矩阵搬运：向量化复制每行的 K 维切片
    for (uint32_t row = blockIdx; row < config->m; row += blockNum) {
        const uint64_t bSrcOffset = static_cast<uint64_t>(row) * config->ldb + config->chunkKBase;
        const uint64_t bDstOffset = static_cast<uint64_t>(row) * config->chunkKCount;

        // 使用 DataCopy 批量搬运
        if (config->chunkKCount % 8 == 0) {
            // 32B 对齐，直接使用 DataCopy
            DataCopy(bRowLocal, bGlobal[bSrcOffset], config->chunkKCount);
            PipeBarrier<PIPE_ALL>();
            DataCopy(bChunkGlobal[bDstOffset], bRowLocal, config->chunkKCount);
        } else {
            // 非对齐，使用 DataCopy 读取 + DataCopyPad 写入
            DataCopy(bRowLocal, bGlobal[bSrcOffset], config->chunkKCount);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams copyParams{1, static_cast<uint16_t>(config->chunkKCount * sizeof(float)), 0, 0, 0};
            DataCopyPad(bChunkGlobal[bDstOffset], bRowLocal, copyParams);
        }
    }

    bQueue.FreeTensor(bRowLocal);

    // A 矩阵搬运：根据 packMode 选择向量化或标量路径
    TQue<QuePosition::VECIN, 1> aQueue;
    const uint32_t alignedASize = ((config->chunkColCount * sizeof(float) + 31) / 32) * 32;
    pipe.InitBuffer(aQueue, 1, alignedASize);
    LocalTensor<float> aRowLocal = aQueue.AllocTensor<float>();

    if (packMode == RightCubePackMode::FAST_DIRECT || packMode == RightCubePackMode::GENERIC_DIRECT) {
        // DIRECT 模式：A[k, col] 连续访问，使用 DataCopy
        for (uint32_t t = blockIdx; t < config->chunkKCount; t += blockNum) {
            const uint32_t k = config->chunkKBase + t;
            const uint64_t aSrcOffset = static_cast<uint64_t>(k) * config->lda + config->chunkColBase;
            const uint64_t aDstOffset = static_cast<uint64_t>(t) * config->chunkColCount;

            if (config->chunkColCount % 8 == 0) {
                DataCopy(aRowLocal, aSymGlobal[aSrcOffset], config->chunkColCount);
                PipeBarrier<PIPE_ALL>();
                DataCopy(aChunkGlobal[aDstOffset], aRowLocal, config->chunkColCount);
            } else {
                // 非对齐，使用 DataCopy 读取 + DataCopyPad 写入
                DataCopy(aRowLocal, aSymGlobal[aSrcOffset], config->chunkColCount);
                PipeBarrier<PIPE_ALL>();
                DataCopyExtParams copyParams{1, static_cast<uint16_t>(config->chunkColCount * sizeof(float)), 0, 0, 0};
                DataCopyPad(aChunkGlobal[aDstOffset], aRowLocal, copyParams);
            }
        }
    } else if (packMode == RightCubePackMode::FAST_MIRROR || packMode == RightCubePackMode::GENERIC_MIRROR) {
        // MIRROR 模式：A[col, k] 需要逐列读取后组成行
        for (uint32_t t = blockIdx; t < config->chunkKCount; t += blockNum) {
            const uint32_t k = config->chunkKBase + t;
            const uint64_t aDstOffset = static_cast<uint64_t>(t) * config->chunkColCount;

            // 从各列读取第 k 个元素，组成一行
            for (uint32_t j = 0; j < config->chunkColCount; ++j) {
                const uint32_t col = config->chunkColBase + j;
                aRowLocal.SetValue(j, aSymGlobal.GetValue(static_cast<uint64_t>(col) * config->lda + k));
            }

            // 写回 chunk
            PipeBarrier<PIPE_ALL>();
            if (config->chunkColCount % 8 == 0) {
                DataCopy(aChunkGlobal[aDstOffset], aRowLocal, config->chunkColCount);
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint16_t>(config->chunkColCount * sizeof(float)), 0, 0, 0};
                DataCopyPad(aChunkGlobal[aDstOffset], aRowLocal, copyParams);
            }
        }
    } else {
        // GENERIC_SYMMETRIC 或 MIXED_DIAG_FULL_TILE：需要逐元素条件判断，保持标量路径
        for (uint32_t t = blockIdx; t < config->chunkKCount; t += blockNum) {
            const uint32_t k = config->chunkKBase + t;
            for (uint32_t j = 0; j < config->chunkColCount; ++j) {
                const uint32_t col = config->chunkColBase + j;
                float value = 0.0f;
                if (static_cast<RightCubeUplo>(config->uplo) == RightCubeUplo::LOWER) {
                    value = (k >= col)
                        ? aSymGlobal.GetValue(static_cast<uint64_t>(k) * config->lda + col)
                        : aSymGlobal.GetValue(static_cast<uint64_t>(col) * config->lda + k);
                } else {
                    value = (k <= col)
                        ? aSymGlobal.GetValue(static_cast<uint64_t>(k) * config->lda + col)
                        : aSymGlobal.GetValue(static_cast<uint64_t>(col) * config->lda + k);
                }
                aChunkGlobal.SetValue(static_cast<uint64_t>(t) * config->chunkColCount + j, value);
            }
        }
    }

    aQueue.FreeTensor(aRowLocal);
}

// RIGHT cube 的 chunk dense kernel。
// 它读取 pack 后的 A/B chunk，调用 MatmulImpl 计算当前 chunk 的局部输出。
__global__ __aicore__ void ssymm_right_cube_unified_chunk_dense_kernel(GM_ADDR workspace, GM_ADDR configGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    auto config = reinterpret_cast<__gm__ SsymmRightCubeUnifiedConfig *>(configGm);

    TPipe pipe;
    __gm__ float *workspaceBase = reinterpret_cast<__gm__ float *>(workspace);
    const uint64_t aChunkCapacity = static_cast<uint64_t>(config->tk) * config->tn;
    const uint64_t bChunkCapacity = static_cast<uint64_t>(config->m) * config->tk;
    GlobalTensor<float> aChunkGlobal;
    GlobalTensor<float> bChunkGlobal;
    GlobalTensor<float> outChunkGlobal;
    aChunkGlobal.SetGlobalBuffer(workspaceBase, static_cast<uint64_t>(config->chunkKCount) * config->chunkColCount);
    bChunkGlobal.SetGlobalBuffer(workspaceBase + aChunkCapacity,
        static_cast<uint64_t>(config->m) * config->chunkKCount);
    outChunkGlobal.SetGlobalBuffer(workspaceBase + aChunkCapacity + bChunkCapacity,
        static_cast<uint64_t>(config->m) * config->chunkColCount);

    for (uint32_t rowBase = 0; rowBase < config->m; rowBase += SSYMM_RIGHT_CUBE_PARTIAL_MAX_M) {
        const uint32_t rowCount = (rowBase + SSYMM_RIGHT_CUBE_PARTIAL_MAX_M <= config->m)
            ? SSYMM_RIGHT_CUBE_PARTIAL_MAX_M
            : (config->m - rowBase);
        GlobalTensor<float> bChunkBandGlobal;
        GlobalTensor<float> outChunkBandGlobal;
        bChunkBandGlobal.SetGlobalBuffer(
            workspaceBase + aChunkCapacity + static_cast<uint64_t>(rowBase) * config->chunkKCount,
            static_cast<uint64_t>(rowCount) * config->chunkKCount);
        outChunkBandGlobal.SetGlobalBuffer(
            workspaceBase + aChunkCapacity + bChunkCapacity + static_cast<uint64_t>(rowBase) * config->chunkColCount,
            static_cast<uint64_t>(rowCount) * config->chunkColCount);

        RightCubeUnifiedMatmul matmulObj;
        matmulObj.SetSubBlockIdx(0);
        matmulObj.Init(static_cast<const TCubeTiling *>(nullptr), &pipe);
        matmulObj.DisableBias();
        matmulObj.SetOrgShape(static_cast<int>(rowCount), static_cast<int>(config->chunkColCount),
            static_cast<int>(config->chunkKCount));
        matmulObj.SetSingleShape(static_cast<int>(rowCount), static_cast<int>(config->chunkColCount),
            static_cast<int>(config->chunkKCount));
        matmulObj.SetTensorA(bChunkBandGlobal, false);
        matmulObj.SetTensorB(aChunkGlobal, false);
        matmulObj.IterateAll(outChunkBandGlobal, 0);
        matmulObj.End();
    }
}

// RIGHT cube 的 chunk 累加 kernel。
// 它把当前 chunk dense 输出累加到完整 m*n scratch 矩阵对应位置。
__global__ __aicore__ void ssymm_right_cube_unified_chunk_accum_kernel(
    GM_ADDR workspace, GM_ADDR scratch, GM_ADDR configGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto config = reinterpret_cast<__gm__ SsymmRightCubeUnifiedConfig *>(configGm);
    __gm__ float *workspaceBase = reinterpret_cast<__gm__ float *>(workspace);
    const uint64_t aChunkCapacity = static_cast<uint64_t>(config->tk) * config->tn;
    const uint64_t bChunkCapacity = static_cast<uint64_t>(config->m) * config->tk;
    GlobalTensor<float> outChunkGlobal;
    GlobalTensor<float> scratchGlobal;
    outChunkGlobal.SetGlobalBuffer(workspaceBase + aChunkCapacity + bChunkCapacity,
        static_cast<uint64_t>(config->m) * config->chunkColCount);
    scratchGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch), static_cast<uint64_t>(config->m) * config->n);

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t row = blockIdx; row < config->m; row += blockNum) {
        for (uint32_t j = 0; j < config->chunkColCount; ++j) {
            const uint32_t col = config->chunkColBase + j;
            const uint64_t scratchOffset = static_cast<uint64_t>(row) * config->n + col;
            const float accum = scratchGlobal.GetValue(scratchOffset);
            const float partial = outChunkGlobal.GetValue(static_cast<uint64_t>(row) * config->chunkColCount + j);
            scratchGlobal.SetValue(scratchOffset, accum + partial);
        }
    }
}

// LEFT cube 的 A panel pack kernel。
// 它按 row/k runtime config 从对称 A 中恢复有效值并写入连续 workspace。
__global__ __aicore__ void ssymm_left_cube_pack_kernel(GM_ADDR aSym, GM_ADDR aDense, GM_ADDR configGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto config = reinterpret_cast<__gm__ SsymmLeftCubeConfig *>(configGm);
    GlobalTensor<float> aSymGlobal;
    GlobalTensor<float> aDenseGlobal;
    aSymGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aSym), static_cast<uint64_t>(config->m) * config->lda);
    aDenseGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aDense), static_cast<uint64_t>(config->rowCount) * config->kCount);

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    const RightCubeUplo uplo = static_cast<RightCubeUplo>(config->uplo);
    for (uint32_t localRow = blockIdx; localRow < config->rowCount; localRow += blockNum) {
        const uint32_t row = config->rowBase + localRow;
        for (uint32_t t = 0; t < config->kCount; ++t) {
            const uint32_t col = config->kBase + t;
            const float value = (uplo == RightCubeUplo::LOWER)
                ? ((row >= col)
                    ? aSymGlobal.GetValue(static_cast<uint64_t>(row) * config->lda + col)
                    : aSymGlobal.GetValue(static_cast<uint64_t>(col) * config->lda + row))
                : ((row <= col)
                    ? aSymGlobal.GetValue(static_cast<uint64_t>(row) * config->lda + col)
                    : aSymGlobal.GetValue(static_cast<uint64_t>(col) * config->lda + row));
            aDenseGlobal.SetValue(static_cast<uint64_t>(localRow) * config->kCount + t, value);
        }
    }
}

// LEFT cube 的 dense kernel。
// 它将 packed A panel 与 B 的 K 分块送入 MatmulImpl，输出到 partial 区。
__global__ __aicore__ void ssymm_left_cube_dense_kernel(
    GM_ADDR workspace, GM_ADDR b, GM_ADDR configGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    auto config = reinterpret_cast<__gm__ SsymmLeftCubeConfig *>(configGm);

    TPipe pipe;
    RightCubeUnifiedMatmul matmulObj;
    GlobalTensor<float> aPackedGlobal;
    GlobalTensor<float> bChunkGlobal;
    GlobalTensor<float> partialGlobal;
    __gm__ float *workspaceBase = reinterpret_cast<__gm__ float *>(workspace);
    const uint64_t packedCount = static_cast<uint64_t>(config->rowCount) * config->kCount;
    // 对齐到 32B（8 个 float），确保 partial 区 GM 起始地址对齐。
    const uint64_t alignedPackedCount = ((packedCount + 7) / 8) * 8;
    __gm__ float *bChunkBase = reinterpret_cast<__gm__ float *>(b) +
        static_cast<uint64_t>(config->kBase) * config->ldb;
    aPackedGlobal.SetGlobalBuffer(workspaceBase, packedCount);
    bChunkGlobal.SetGlobalBuffer(bChunkBase, static_cast<uint64_t>(config->kCount) * config->ldb);
    partialGlobal.SetGlobalBuffer(workspaceBase + alignedPackedCount, static_cast<uint64_t>(config->rowCount) * config->n);

    matmulObj.SetSubBlockIdx(0);
    matmulObj.Init(static_cast<const TCubeTiling *>(nullptr), &pipe);
    matmulObj.DisableBias();
    matmulObj.SetOrgShape(static_cast<int>(config->rowCount), static_cast<int>(config->n), static_cast<int>(config->kCount));
    matmulObj.SetSingleShape(static_cast<int>(config->rowCount), static_cast<int>(config->n), static_cast<int>(config->kCount));
    matmulObj.SetTensorA(aPackedGlobal, false);
    matmulObj.SetTensorB(bChunkGlobal, false);
    matmulObj.IterateAll(partialGlobal, 0);
    matmulObj.End();
}

// LEFT cube 的 partial 清零 kernel。
// 每个 k 分块计算前先清空当前 partial 区，避免累加到上一次 launch 的旧数据。
__global__ __aicore__ void ssymm_left_cube_clear_partial_kernel(GM_ADDR workspace, GM_ADDR configGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto config = reinterpret_cast<__gm__ SsymmLeftCubeConfig *>(configGm);
    __gm__ float *workspaceBase = reinterpret_cast<__gm__ float *>(workspace);
    const uint64_t packedCount = static_cast<uint64_t>(config->rowCount) * config->kCount;
    // 对齐到 32B（8 个 float），确保 partial 区 GM 起始地址对齐。
    const uint64_t alignedPackedCount = ((packedCount + 7) / 8) * 8;
    GlobalTensor<float> partialGlobal;
    partialGlobal.SetGlobalBuffer(workspaceBase + alignedPackedCount, static_cast<uint64_t>(config->rowCount) * config->n);

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    const uint32_t total = config->rowCount * config->n;
    for (uint32_t idx = blockIdx; idx < total; idx += blockNum) {
        partialGlobal.SetValue(idx, 0.0f);
    }
}

// LEFT cube 的 partial 累加 kernel。
// 它将本次 k 分块的 partial 结果累加到完整 scratch 矩阵对应行范围。
__global__ __aicore__ void ssymm_left_cube_accum_kernel(GM_ADDR workspace, GM_ADDR scratch, GM_ADDR configGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto config = reinterpret_cast<__gm__ SsymmLeftCubeConfig *>(configGm);
    __gm__ float *workspaceBase = reinterpret_cast<__gm__ float *>(workspace);
    const uint64_t packedCount = static_cast<uint64_t>(config->rowCount) * config->kCount;
    // 对齐到 32B（8 个 float），确保 partial 区 GM 起始地址对齐。
    const uint64_t alignedPackedCount = ((packedCount + 7) / 8) * 8;
    GlobalTensor<float> partialGlobal;
    GlobalTensor<float> scratchGlobal;
    partialGlobal.SetGlobalBuffer(workspaceBase + alignedPackedCount, static_cast<uint64_t>(config->rowCount) * config->n);
    scratchGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch), static_cast<uint64_t>(config->m) * config->n);

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
    for (uint32_t localRow = blockIdx; localRow < config->rowCount; localRow += blockNum) {
        const uint32_t row = config->rowBase + localRow;
        for (uint32_t col = 0; col < config->n; ++col) {
            const uint64_t scratchOffset = static_cast<uint64_t>(row) * config->n + col;
            const float accum = scratchGlobal.GetValue(scratchOffset);
            const float partial = partialGlobal.GetValue(static_cast<uint64_t>(localRow) * config->n + col);
            scratchGlobal.SetValue(scratchOffset, accum + partial);
        }
    }
}

// postprocess 公共实现：对 [rowBase, rowBase+rowCount) 行执行 alpha*scratch + beta*C 并写回 C。
// LEFT 和 RIGHT postprocess kernel 共享此逻辑，仅行迭代范围不同。
__aicore__ inline void PostprocessRows(
    GlobalTensor<float> &scratchGlobal, GlobalTensor<float> &cGlobal,
    uint32_t rowBase, uint32_t rowCount, uint32_t n, uint32_t ldc,
    float alpha, float beta,
    TQue<QuePosition::VECIN, 1> &scratchQueue,
    TQue<QuePosition::VECIN, 1> &cQueue, TQue<QuePosition::VECOUT, 1> &outQueue)
{
    constexpr uint32_t maxChunkSize = 512;
    constexpr uint32_t elementsPerBlock = 8; // 32B / sizeof(float)

    LocalTensor<float> scratchLocal = scratchQueue.AllocTensor<float>();
    LocalTensor<float> cLocal = cQueue.AllocTensor<float>();
    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

    const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());

    for (uint32_t localRow = blockIdx; localRow < rowCount; localRow += blockNum) {
        const uint32_t row = rowBase + localRow;
        const uint64_t scratchRowOffset = static_cast<uint64_t>(row) * n;
        const uint64_t cRowOffset = static_cast<uint64_t>(row) * ldc;

        for (uint32_t colBase = 0; colBase < n; colBase += maxChunkSize) {
            const uint32_t colCount = (colBase + maxChunkSize <= n) ? maxChunkSize : (n - colBase);

            if (colCount % elementsPerBlock == 0) {
                DataCopy(scratchLocal, scratchGlobal[scratchRowOffset + colBase], colCount);
                DataCopy(cLocal, cGlobal[cRowOffset + colBase], colCount);
            } else {
                const uint8_t paddingNum = elementsPerBlock - (colCount % elementsPerBlock);
                DataCopyExtParams copyParams{1, static_cast<uint16_t>(colCount * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> padParams{true, 0, paddingNum, 0};
                DataCopyPad(scratchLocal, scratchGlobal[scratchRowOffset + colBase], copyParams, padParams);
                DataCopyPad(cLocal, cGlobal[cRowOffset + colBase], copyParams, padParams);
            }

            const uint32_t calcCount = ((colCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
            PipeBarrier<PIPE_ALL>();
            Muls(scratchLocal, scratchLocal, alpha, calcCount);
            Muls(cLocal, cLocal, beta, calcCount);
            Add(outLocal, scratchLocal, cLocal, calcCount);
            PipeBarrier<PIPE_ALL>();

            if (colCount % elementsPerBlock == 0) {
                DataCopy(cGlobal[cRowOffset + colBase], outLocal, colCount);
            } else {
                DataCopyExtParams copyParams{1, static_cast<uint16_t>(colCount * sizeof(float)), 0, 0, 0};
                DataCopyPad(cGlobal[cRowOffset + colBase], outLocal, copyParams);
            }
        }
    }

    scratchQueue.FreeTensor(scratchLocal);
    cQueue.FreeTensor(cLocal);
    outQueue.FreeTensor(outLocal);
}

// LEFT cube 的后处理 kernel（向量化实现）。
// 它从 scratch 读取完整结果，并融合 alpha 结果与 beta*C 后写回 C。
__global__ __aicore__ void ssymm_left_cube_postprocess_kernel(GM_ADDR scratch, GM_ADDR c, GM_ADDR configGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto config = reinterpret_cast<__gm__ SsymmLeftCubeConfig *>(configGm);
    GlobalTensor<float> scratchGlobal;
    GlobalTensor<float> cGlobal;
    scratchGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch), static_cast<uint64_t>(config->m) * config->n);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c), static_cast<uint64_t>(config->m) * config->ldc);

    constexpr uint32_t maxChunkSize = 512;
    constexpr uint32_t alignedChunkSize = ((maxChunkSize * sizeof(float) + 31) / 32) * 32;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> scratchQueue;
    TQue<QuePosition::VECIN, 1> cQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    pipe.InitBuffer(scratchQueue, 1, alignedChunkSize);
    pipe.InitBuffer(cQueue, 1, alignedChunkSize);
    pipe.InitBuffer(outQueue, 1, alignedChunkSize);
    PostprocessRows(scratchGlobal, cGlobal,
        config->rowBase, config->rowCount, config->n, config->ldc,
        config->alpha, config->beta,
        scratchQueue, cQueue, outQueue);
}

// RIGHT cube 的后处理 kernel（向量化实现）。
// 它从 scratch 读取完整结果，并融合 alpha 结果与 beta*C 后写回 C。
__global__ __aicore__ void ssymm_right_cube_unified_postprocess_kernel(GM_ADDR scratch, GM_ADDR c, GM_ADDR configGm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto config = reinterpret_cast<__gm__ SsymmRightCubeUnifiedConfig *>(configGm);
    GlobalTensor<float> scratchGlobal;
    GlobalTensor<float> cGlobal;
    scratchGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scratch), static_cast<uint64_t>(config->m) * config->n);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c), static_cast<uint64_t>(config->m) * config->ldc);

    constexpr uint32_t maxChunkSize = 512;
    constexpr uint32_t alignedChunkSize = ((maxChunkSize * sizeof(float) + 31) / 32) * 32;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> scratchQueue;
    TQue<QuePosition::VECIN, 1> cQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    pipe.InitBuffer(scratchQueue, 1, alignedChunkSize);
    pipe.InitBuffer(cQueue, 1, alignedChunkSize);
    pipe.InitBuffer(outQueue, 1, alignedChunkSize);
    PostprocessRows(scratchGlobal, cGlobal,
        0, config->m, config->n, config->ldc,
        config->alpha, config->beta,
        scratchQueue, cQueue, outQueue);
}

// host 侧 LEFT cube 组合 launch 入口。
// 单个 runtime config 会依次执行 clear、pack、dense、accum，postprocess 由外层按需触发。
void ssymm_left_cube_do(GM_ADDR aSym, GM_ADDR workspace, GM_ADDR b, GM_ADDR scratch,
                        GM_ADDR configGm, void *stream)
{
    constexpr uint32_t vecBlocks = 8;
    ssymm_left_cube_clear_partial_kernel<<<vecBlocks, nullptr, stream>>>(workspace, configGm);
    ssymm_left_cube_pack_kernel<<<vecBlocks, nullptr, stream>>>(aSym, workspace, configGm);
    ssymm_left_cube_dense_kernel<<<1, nullptr, stream>>>(workspace, b, configGm);
    ssymm_left_cube_accum_kernel<<<vecBlocks, nullptr, stream>>>(workspace, scratch, configGm);
}

// host 侧 LEFT cube 后处理 launch 入口。
// 它只启动 postprocess kernel，将 scratch 中的完整结果融合写回 C。
void ssymm_left_cube_postprocess_do(GM_ADDR scratch, GM_ADDR c, GM_ADDR configGm, void *stream)
{
    constexpr uint32_t vecBlocks = 8;
    ssymm_left_cube_postprocess_kernel<<<vecBlocks, nullptr, stream>>>(scratch, c, configGm);
}

// host 侧 RIGHT cube 组合 launch 入口。
// fallback 策略先调用通用 RIGHT kernel 写 scratch；partial cube 策略执行 pack/dense/accum。
void ssymm_right_cube_unified_do(GM_ADDR a, GM_ADDR aDense, GM_ADDR b, GM_ADDR c, GM_ADDR scratch,
                                 GM_ADDR tilingGm, GM_ADDR configGm,
                                 const SsymmRightCubeUnifiedConfig &config, uint32_t numBlocks, void *stream)
{
    RightCubeStrategy strategy = static_cast<RightCubeStrategy>(config.strategy);
    if (strategy == RightCubeStrategy::FALLBACK_TO_SCRATCH) {
        aclblasFillMode_t uplo = (static_cast<RightCubeUplo>(config.uplo) == RightCubeUplo::LOWER)
            ? ACLBLAS_LOWER
            : ACLBLAS_UPPER;
        constexpr uint32_t vecBlocks = 8;
        ssymm_kernel_do(a, b, scratch, tilingGm, ACLBLAS_SIDE_RIGHT, uplo, numBlocks, stream);
        ssymm_right_cube_unified_postprocess_kernel<<<vecBlocks, nullptr, stream>>>(scratch, c, configGm);
        return;
    }
    if (strategy == RightCubeStrategy::PARTIAL_CUBE) {
        ssymm_right_cube_unified_chunk_pack_kernel<<<8, nullptr, stream>>>(a, b, aDense, configGm);
        ssymm_right_cube_unified_chunk_dense_kernel<<<1, nullptr, stream>>>(aDense, configGm);
        ssymm_right_cube_unified_chunk_accum_kernel<<<8, nullptr, stream>>>(aDense, scratch, configGm);
        return;
    }

    aclblasFillMode_t uplo = (static_cast<RightCubeUplo>(config.uplo) == RightCubeUplo::LOWER)
        ? ACLBLAS_LOWER
        : ACLBLAS_UPPER;
    ssymm_kernel_do(a, b, c, tilingGm, ACLBLAS_SIDE_RIGHT, uplo, numBlocks, stream);
}

// host 侧 RIGHT cube postprocess 独立入口。
// 将 scratch 中的结果融合 alpha/beta 写回 C，仅在所有 chunks 完成后调用一次。
void ssymm_right_cube_unified_postprocess_do(GM_ADDR scratch, GM_ADDR c, GM_ADDR configGm, void *stream)
{
    constexpr uint32_t vecBlocks = 8;
    ssymm_right_cube_unified_postprocess_kernel<<<vecBlocks, nullptr, stream>>>(scratch, c, configGm);
}

// host 侧通用 fallback launch 入口。
// 根据 side/uplo 选择四个模板实例之一，保持外部调用点统一。
void ssymm_kernel_do(GM_ADDR A, GM_ADDR B, GM_ADDR C, GM_ADDR tilingGm,
                     aclblasSideMode_t side, aclblasFillMode_t uplo,
                     uint32_t numBlocks, void *stream)
{
    if (side == ACLBLAS_SIDE_LEFT) {
        if (uplo == ACLBLAS_LOWER) {
            ssymm_left_lower_fallback_kernel<<<numBlocks, nullptr, stream>>>(A, B, C, tilingGm);
        } else {
            ssymm_left_upper_fallback_kernel<<<numBlocks, nullptr, stream>>>(A, B, C, tilingGm);
        }
    } else {
        if (uplo == ACLBLAS_LOWER) {
            ssymm_right_lower_kernel<<<numBlocks, nullptr, stream>>>(A, B, C, tilingGm);
        } else {
            ssymm_right_upper_kernel<<<numBlocks, nullptr, stream>>>(A, B, C, tilingGm);
        }
    }
}
