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
 * \file aclblaslt_handle_impl.h
 * \brief Internal handle struct, its algo LRU cache types, library version / hardware constants,
 *        and device-capability queries. Not installed.
 */

#pragma once

#include "cann_ops_blasLt.h"

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>

constexpr int ACLBLASLT_VERSION_MAJOR = 1;
constexpr int ACLBLASLT_VERSION_MINOR = 0;
constexpr int ACLBLASLT_VERSION_PATCH = 0;

constexpr uint32_t ACLBLASLT_HANDLE_MAGIC = 0xACBA1234;

// Ascend on-chip / peak defaults, used by handle setup and the algo heuristic.
constexpr size_t L1_SIZE = 512 * 1024;
constexpr size_t L0_SIZE = 256;
constexpr uint32_t DEFAULT_AI_CORES = 8;
constexpr double DEFAULT_PEAK_TFLOPS = 140.0;
constexpr double DEFAULT_PEAK_GBPS = 900.0;

struct AlgoKey {
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
    aclDataType aType = ACL_FLOAT;
    aclDataType bType = ACL_DT_UNDEFINED;
    aclDataType cType = ACL_DT_UNDEFINED;
    aclDataType dType = ACL_DT_UNDEFINED;
    aclblasComputeType_t computeType = ACLBLAS_COMPUTE_32F;
    aclblasLtEpilogue_t epilogue = ACLBLASLT_EPILOGUE_DEFAULT;
    bool transA = false;
    bool transB = false;

    bool operator==(const AlgoKey& other) const
    {
        return m == other.m && n == other.n && k == other.k && aType == other.aType && bType == other.bType &&
               cType == other.cType && dType == other.dType && computeType == other.computeType &&
               epilogue == other.epilogue && transA == other.transA && transB == other.transB;
    }
};

struct AlgoKeyHasher {
    size_t operator()(const AlgoKey& x) const
    {
        size_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { h ^= static_cast<size_t>(v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2)); };
        mix(x.m);
        mix(x.n);
        mix(x.k);
        mix(static_cast<uint64_t>(x.aType));
        mix(static_cast<uint64_t>(x.bType));
        mix(static_cast<uint64_t>(x.cType));
        mix(static_cast<uint64_t>(x.dType));
        mix(static_cast<uint64_t>(x.computeType));
        mix(static_cast<uint64_t>(x.epilogue));
        mix(static_cast<uint64_t>(x.transA));
        mix(static_cast<uint64_t>(x.transB));
        return h;
    }
};

struct CacheEntry {
    aclblasLtMatmulAlgo_t algo;
    std::list<AlgoKey>::iterator lruIter;
};

struct _aclblaslt_handle {
    uint32_t magic = ACLBLASLT_HANDLE_MAGIC;
    bool initialized = false;
    // version info
    int versionMajor = ACLBLASLT_VERSION_MAJOR;
    int versionMinor = ACLBLASLT_VERSION_MINOR;
    // AscendCL runtime
    aclrtContext context = nullptr;
    aclrtStream defaultStream = nullptr;
    int32_t deviceId = 0;
    // workspace
    void* internalWorkspace = nullptr;
    size_t workspaceSize = 0;
    // thread safety
    std::mutex* mutex = nullptr;
    // soc spec
    int npuArch = 0;
    size_t maxSharedMemory = 0;
    // algo cache
    std::unordered_map<AlgoKey, CacheEntry, AlgoKeyHasher>* algoCache = nullptr;
    size_t algoCacheMaxSize = 128;
    std::list<AlgoKey>* lruList = nullptr;
};

// Query the number of cube cores for a device, returning a safe fallback when the runtime query fails.
uint32_t QueryCubeCoreNum(int32_t deviceId);
