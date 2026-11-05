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

#include <cstdint>

// UB 预算：硬件总 UB = 248KB (kernel_constant.h)
// SIMT 混合编程场景下编译期预留 32KB DCache + 8KB 保留区 = 40KB
// 可用动态内存 = 248KB - 40KB = 208KB
// 参考：https://gitcode.com/cann/cann-samples/tree/master/Samples/1_Features/hardware_features/simd_vf_constraints/
constexpr uint32_t GEMV_STRIDED_BATCHED_UBUF_SIZE = 208 * 1024;
constexpr uint32_t GEMV_STRIDED_BATCHED_BN_NORMAL = 1;

// 分核逻辑: 除最后一个 core 外，每个 core 处理 batchPerCore 个 batch；
//          最后一个 core 处理 batchTail 个（余数）。
// startMatId = coreIdx * batchPerCore
// calMatNum  = (coreIdx == usedCoreNum - 1) ? batchTail : batchPerCore
// pack(8): 确保 int64_t 字段 strideA/stridex/stridey 8 字节对齐，满足 GM 64-bit 访问最佳实践。
// 字段顺序: 所有 uint32_t/int32_t (4B) 在前，int64_t (8B) 在后，减少 padding。
// 注: 23 个 4B 字段合计 92B，非 8 的倍数，strideA 前有 4B 对齐 padding。
#pragma pack(push, 8)
struct GemvStridedBatchedTilingData {
    uint32_t dtype;           // 0=HSH, 1=S(FP32), 2=HSS, 3=TST, 4=TSS
    uint32_t trans;           // 0 = N, 1 = T/C
    float    alpha;           // 标量缩放因子（指针解引用后的值）
    float    beta;            // 标量缩放因子（指针解引用后的值）
    uint32_t m;               // 矩阵 A 行数
    uint32_t n;               // 矩阵 A 列数
    uint32_t outSize;         // 输出向量维度（Normal=m, Transpose=n）
    uint32_t dotSize;         // 点积向量维度（Normal=n, Transpose=m）
    uint32_t usedCoreNum;     // 实际使用的核数
    uint32_t batchCount;      // 批量大小
    uint32_t batchPerCore;    // 每核处理的 batch 数
    uint32_t batchTail;       // 尾核处理的 batch 数
    uint32_t dotTile;         // 点积轴切分大小（AIV 路径）
    uint32_t outTile;         // 输出轴切分大小（AIV 路径）
    uint32_t bufInA;          // UB: A 矩阵块大小（256B 对齐）
    uint32_t bufInx;          // UB: x 向量块大小（256B 对齐）
    uint32_t bufInY;          // UB: y 输入块大小（256B 对齐）
    uint32_t bufMatTmp;       // UB: 临时矩阵大小（256B 对齐）
    uint32_t bufVecTmp;       // UB: 临时向量大小（256B 对齐）
    uint32_t numThreads;      // SIMT 实际线程数（动态计算）
    int32_t  lda;             // A 前导维度
    int32_t  incx;            // x 向量内元素步长
    int32_t  incy;            // y 向量内元素步长
    int64_t  strideA;         // A 的 batch 间步长 = (int64_t)lda * n（显式，防溢出）
    int64_t  stridex;         // x 的 batch 间步长（显式参数）
    int64_t  stridey;         // y 的 batch 间步长（显式参数）
};
#pragma pack(pop)


