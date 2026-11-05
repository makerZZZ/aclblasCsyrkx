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
 * \file aclblaslt_logger_impl.cpp
 * \brief Internal logger manager implementation for aclblasLt.
 */

#include "aclblaslt_logger_impl.h"

#include <cstdarg>
#include <cstdlib>

#include <securec.h>

namespace AclBlasLt {

LoggerManager::LoggerManager() { InitFromEnv(); }

LoggerManager::~LoggerManager() { CloseOwnedFile(); }

LoggerManager& LoggerManager::GetInstance()
{
    static LoggerManager instance;
    return instance;
}

void LoggerManager::InitFromEnv()
{
    const char* envLevel = std::getenv("ACLBLASLT_LOG_LEVEL");
    if (envLevel != nullptr) {
        int lvl = std::atoi(envLevel);
        if (lvl >= ACLBLASLT_LOG_LEVEL_OFF && lvl <= ACLBLASLT_LOG_LEVEL_API_TRACE) {
            level_.store(lvl, std::memory_order_relaxed);
        }
    }

    const char* envMask = std::getenv("ACLBLASLT_LOG_MASK");
    if (envMask != nullptr) {
        mask_.store(std::atoi(envMask), std::memory_order_relaxed);
    }

    int lvl = level_.load(std::memory_order_relaxed);
    int mask = mask_.load(std::memory_order_relaxed);
    if (lvl > ACLBLASLT_LOG_LEVEL_OFF && mask == ACLBLASLT_LOG_MASK_OFF) {
        if (lvl >= ACLBLASLT_LOG_LEVEL_ERROR) {
            mask |= ACLBLASLT_LOG_MASK_ERROR;
        }
        if (lvl >= ACLBLASLT_LOG_LEVEL_TRACE) {
            mask |= ACLBLASLT_LOG_MASK_TRACE;
        }
        if (lvl >= ACLBLASLT_LOG_LEVEL_HINTS) {
            mask |= ACLBLASLT_LOG_MASK_HINTS;
        }
        if (lvl >= ACLBLASLT_LOG_LEVEL_INFO) {
            mask |= ACLBLASLT_LOG_MASK_INFO;
        }
        if (lvl >= ACLBLASLT_LOG_LEVEL_API_TRACE) {
            mask |= ACLBLASLT_LOG_MASK_API_TRACE;
        }
        mask_.store(mask, std::memory_order_relaxed);
    }

    const char* envFile = std::getenv("ACLBLASLT_LOG_FILE");
    if (envFile != nullptr && envFile[0] != '\0') {
        FILE* f = std::fopen(envFile, "a");
        if (f != nullptr) {
            ownedFile_ = f;
        }
    }
}

aclblasStatus_t LoggerManager::SetFile(FILE* file)
{
    if (file == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (forceDisabled_.load(std::memory_order_relaxed)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    CloseOwnedFile();
    userFile_ = file;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LoggerManager::SetLevel(int level)
{
    if (level < ACLBLASLT_LOG_LEVEL_OFF || level > ACLBLASLT_LOG_LEVEL_API_TRACE) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (forceDisabled_.load(std::memory_order_relaxed)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    level_.store(level, std::memory_order_relaxed);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LoggerManager::SetMask(int mask)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (forceDisabled_.load(std::memory_order_relaxed)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    mask_.store(mask, std::memory_order_relaxed);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LoggerManager::ForceDisable()
{
    std::lock_guard<std::mutex> lock(mutex_);
    forceDisabled_.store(true, std::memory_order_relaxed);
    level_.store(ACLBLASLT_LOG_LEVEL_OFF, std::memory_order_relaxed);
    mask_.store(ACLBLASLT_LOG_MASK_OFF, std::memory_order_relaxed);
    CloseOwnedFile();
    userFile_ = nullptr;
    callback_ = nullptr;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LoggerManager::SetCallback(aclblasLtLoggerCallback_t callback)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (forceDisabled_.load(std::memory_order_relaxed)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    callback_ = callback;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LoggerManager::OpenFile(const char* logFile)
{
    if (logFile == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    FILE* f = std::fopen(logFile, "a");
    if (f == nullptr) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (forceDisabled_.load(std::memory_order_relaxed)) {
        std::fclose(f);
        return ACLBLAS_STATUS_SUCCESS;
    }
    CloseOwnedFile();
    ownedFile_ = f;
    return ACLBLAS_STATUS_SUCCESS;
}

bool LoggerManager::ShouldLog(int category) const
{
    if (forceDisabled_.load(std::memory_order_relaxed) ||
        level_.load(std::memory_order_relaxed) == ACLBLASLT_LOG_LEVEL_OFF) {
        return false;
    }
    return (mask_.load(std::memory_order_relaxed) & category) != 0;
}

void LoggerManager::Log(int category, const char* functionName, const char* fmt, ...)
{
    if (!ShouldLog(category)) {
        return;
    }

    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    if (vsnprintf_s(buffer, sizeof(buffer), sizeof(buffer) - 1, fmt, args) < 0) {
        va_end(args);
        return;
    }
    va_end(args);

    aclblasLtLoggerCallback_t cb = nullptr;
    FILE* out = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ShouldLog(category)) {
            return;
        }
        cb = callback_;
        if (cb == nullptr) {
            out = GetOutputFile();
        }
    }

    if (cb != nullptr) {
        cb(category, functionName, buffer);
        return;
    }

    std::fprintf(out, "[aclblasLt][%d] %s: %s\n", category, functionName, buffer);
    std::fflush(out);
}

FILE* LoggerManager::GetOutputFile() const
{
    if (userFile_ != nullptr) {
        return userFile_;
    }
    if (ownedFile_ != nullptr) {
        return ownedFile_;
    }
    return stdout;
}

void LoggerManager::CloseOwnedFile()
{
    if (ownedFile_ != nullptr) {
        std::fclose(ownedFile_);
        ownedFile_ = nullptr;
    }
}

} // namespace AclBlasLt
