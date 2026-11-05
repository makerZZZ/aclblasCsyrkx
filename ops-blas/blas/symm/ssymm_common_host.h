/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ssymm_common_types.h"

inline constexpr uint32_t SSYMM_EXEC_PLAN_CORE_NUM = 8;
inline constexpr uint32_t SSYMM_EXEC_PLAN_LEFT_TILE_M = 8;
inline constexpr uint32_t SSYMM_EXEC_PLAN_LEFT_TILE_N = 128;
inline constexpr uint32_t SSYMM_EXEC_PLAN_LEFT_TILE_K = 64;
inline constexpr uint32_t SSYMM_EXEC_PLAN_SMALL_SHAPE_THRESHOLD = 128;

// 判断指定环境变量是否显式打开。
// 只接受 "1"、"true" 和 "TRUE"，避免调试开关被其他字符串误触发。
inline bool IsEnvEnabled(const char *name)
{
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0;
}

// 判断 A/B/C 是否使用规则 row-major dense 行跨度。
// 优化 cube backend 只处理这种紧凑布局，irregular ld 会回退到通用路径。
inline bool SsymmHasRegularLeadingDims(aclblasSideMode_t side,
    int64_t m, int64_t n, int64_t lda, int64_t ldb, int64_t ldc)
{
    const int64_t aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    return lda == aDim && ldb == n && ldc == n;
}

// 判断是否属于小 shape，需要留在通用 fallback 路径。
// 小矩阵用标量 fallback 可以避免 cube 初始化和 workspace 额外开销。
inline bool SsymmIsSmallShape(int64_t m, int64_t n)
{
    return m < SSYMM_EXEC_PLAN_SMALL_SHAPE_THRESHOLD || n < SSYMM_EXEC_PLAN_SMALL_SHAPE_THRESHOLD;
}

// 将已校验的公开 API 参数归一化为内部问题规格。
// planner 后续只消费这个值对象，避免反复直接处理原始入参。
inline SsymmProblemSpec NormalizeSsymmSpec(
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    int64_t m,
    int64_t n,
    const float *alpha,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    const float *beta)
{
    SsymmProblemSpec spec{};
    spec.side = side;
    spec.uplo = uplo;
    spec.m = static_cast<uint32_t>(m);
    spec.n = static_cast<uint32_t>(n);
    spec.lda = static_cast<uint32_t>(lda);
    spec.ldb = static_cast<uint32_t>(ldb);
    spec.ldc = static_cast<uint32_t>(ldc);
    spec.alpha = alpha == nullptr ? 0.0f : *alpha;
    spec.beta = beta == nullptr ? 0.0f : *beta;
    return spec;
}


// 根据归一化后的问题规格构造 host 执行计划。
// 先排除小 shape 和 irregular 布局，再按 LEFT/RIGHT 选择对应 cube 参数。
inline SsymmExecutionPlan BuildSsymmExecutionPlan(const SsymmProblemSpec &spec)
{
    SsymmExecutionPlan plan{};
    plan.spec = spec;
    plan.regularDense = SsymmHasRegularLeadingDims(spec.side, spec.m, spec.n, spec.lda, spec.ldb, spec.ldc);
    plan.smallShapeFallback = SsymmIsSmallShape(spec.m, spec.n);

    if (!plan.regularDense || plan.smallShapeFallback) {
        plan.backend = SsymmBackendKind::GenericFallback;
        return plan;
    }

    plan.coreNum = SSYMM_EXEC_PLAN_CORE_NUM;
    if (spec.side == ACLBLAS_SIDE_LEFT) {
        plan.backend = SsymmBackendKind::LeftCube;
        plan.tile = {SSYMM_EXEC_PLAN_LEFT_TILE_M, SSYMM_EXEC_PLAN_LEFT_TILE_N, SSYMM_EXEC_PLAN_LEFT_TILE_K};
        return plan;
    }

    plan.backend = SsymmBackendKind::RightCube;
    plan.tile = {SSYMM_RIGHT_TILE_M, SSYMM_RIGHT_DISPATCH_TILE_N, SSYMM_RIGHT_TILE_K};
    plan.debugFlags = 1;
    return plan;
}

// 将 side 枚举转换为稳定的小写调试字符串。
// 该函数只用于 plan dump 输出。
inline const char *ToSsymmSideString(aclblasSideMode_t side)
{
    return (side == ACLBLAS_SIDE_LEFT) ? "left" : "right";
}

// 将 uplo 枚举转换为稳定的小写调试字符串。
// 这里假设前置校验已经保证 uplo 只可能是 LOWER 或 UPPER。
inline const char *ToSsymmUploString(aclblasFillMode_t uplo)
{
    return (uplo == ACLBLAS_LOWER) ? "lower" : "upper";
}

// 将内部 backend 枚举转换为调试日志字符串。
// 未知值统一输出 GenericFallback，保证日志路径安全可读。
inline const char *ToSsymmBackendKindString(SsymmBackendKind backendKind)
{
    switch (backendKind) {
        case SsymmBackendKind::RightCube:
            return "RightCube";
        case SsymmBackendKind::LeftCube:
            return "LeftCube";
        case SsymmBackendKind::GenericFallback:
        default:
            return "GenericFallback";
    }
}

// 在 SSYMM_DEBUG_PLAN 打开时打印实际执行的 backend 信息。
// 仅输出 side/uplo/backend，不依赖已删除的诊断字段。
inline void DumpSsymmExecutionBackend(aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    SsymmBackendKind backendKind)
{
    if (!IsEnvEnabled("SSYMM_DEBUG_PLAN")) {
        return;
    }
    std::fprintf(stdout, "[ssymm-plan] side=%d uplo=%d backend=%d\n",
        static_cast<int>(side), static_cast<int>(uplo), static_cast<int>(backendKind));
}

