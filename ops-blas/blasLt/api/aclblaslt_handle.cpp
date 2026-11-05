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
 * \file aclblaslt_handle.cpp
 * \brief Public C API: handle lifecycle (create / destroy).
 */

#include "cann_ops_blasLt.h"

#include "aclblaslt_handle_impl.h"
#include "aclblaslt_layout_impl.h"
#include "host_utils.h"

#include <acl/acl.h>
#include <cstdlib>
#include <list>
#include <mutex>
#include <new>
#include <unordered_map>

extern "C" {

aclblasStatus_t aclblasLtCreate(aclblasLtHandle_t* lightHandle)
{
    if (lightHandle == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    _aclblaslt_handle* h = nullptr;
    auto st = AllocHandle(&h);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    int32_t deviceId = 0;
    aclError aclRet = aclrtGetDevice(&deviceId);
    if (aclRet != ACL_SUCCESS) {
        delete h;
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    aclrtContext currentCtx = nullptr;
    aclRet = aclrtGetCurrentContext(&currentCtx);
    if (aclRet != ACL_SUCCESS || currentCtx == nullptr) {
        delete h;
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    h->deviceId = deviceId;
    h->context = currentCtx;
    h->defaultStream = nullptr;
    h->workspaceSize = DEFAULT_WORKSPACE_SIZE;
    h->internalWorkspace = std::malloc(h->workspaceSize);
    if (h->internalWorkspace == nullptr) {
        delete h;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    h->mutex = new (std::nothrow) std::mutex();
    h->algoCache = new (std::nothrow) std::unordered_map<AlgoKey, CacheEntry, AlgoKeyHasher>();
    h->lruList = new (std::nothrow) std::list<AlgoKey>();
    if (h->mutex == nullptr || h->algoCache == nullptr || h->lruList == nullptr) {
        delete h->mutex;
        delete h->algoCache;
        delete h->lruList;
        std::free(h->internalWorkspace);
        delete h;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    h->npuArch = 2;
    h->maxSharedMemory = L1_SIZE;
    h->initialized = true;
    *lightHandle = h;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtDestroy(const aclblasLtHandle_t lightHandle)
{
    if (lightHandle == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = lightHandle;
    if (h->magic != ACLBLASLT_HANDLE_MAGIC) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    h->initialized = false;
    delete h->mutex;
    delete h->algoCache;
    delete h->lruList;
    std::free(h->internalWorkspace);
    h->mutex = nullptr;
    h->algoCache = nullptr;
    h->lruList = nullptr;
    h->internalWorkspace = nullptr;
    h->workspaceSize = 0;
    return FreeHandle(h);
}

} // extern "C"
