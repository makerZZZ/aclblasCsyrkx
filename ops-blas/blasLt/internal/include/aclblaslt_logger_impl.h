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
 * \file aclblaslt_logger_impl.h
 * \brief Internal logger manager for aclblasLt (not exposed publicly).
 */

#pragma once

#include <atomic>
#include <cstdio>
#include <mutex>

#include "cann_ops_blasLt.h"

namespace AclBlasLt {

class LoggerManager {
public:
    static LoggerManager& GetInstance();

    aclblasStatus_t SetFile(FILE* file);
    aclblasStatus_t SetLevel(int level);
    aclblasStatus_t SetMask(int mask);
    aclblasStatus_t ForceDisable();
    aclblasStatus_t SetCallback(aclblasLtLoggerCallback_t callback);
    aclblasStatus_t OpenFile(const char* logFile);

    bool ShouldLog(int category) const;
    void Log(int category, const char* functionName, const char* fmt, ...);

    void InitFromEnv();

private:
    LoggerManager();
    ~LoggerManager();
    LoggerManager(const LoggerManager&) = delete;
    LoggerManager& operator=(const LoggerManager&) = delete;

    mutable std::mutex mutex_;

    std::atomic<bool> forceDisabled_{false};
    std::atomic<int> level_{ACLBLASLT_LOG_LEVEL_OFF};
    std::atomic<int> mask_{ACLBLASLT_LOG_MASK_OFF};
    FILE* userFile_ = nullptr;
    FILE* ownedFile_ = nullptr;
    aclblasLtLoggerCallback_t callback_ = nullptr;

    void CloseOwnedFile();
    FILE* GetOutputFile() const;
};

} // namespace AclBlasLt
