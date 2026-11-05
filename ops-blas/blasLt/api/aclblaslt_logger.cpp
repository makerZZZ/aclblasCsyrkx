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
 * \file aclblaslt_logger.cpp
 * \brief Public C API: aclblasLt logger control functions (experimental).
 */

#include "cann_ops_blasLt.h"
#include "aclblaslt_logger_impl.h"

extern "C" {

aclblasStatus_t aclblasLtLoggerSetFile(FILE* file) { return AclBlasLt::LoggerManager::GetInstance().SetFile(file); }

aclblasStatus_t aclblasLtLoggerSetLevel(int level) { return AclBlasLt::LoggerManager::GetInstance().SetLevel(level); }

aclblasStatus_t aclblasLtLoggerSetMask(int mask) { return AclBlasLt::LoggerManager::GetInstance().SetMask(mask); }

aclblasStatus_t aclblasLtLoggerForceDisable(void) { return AclBlasLt::LoggerManager::GetInstance().ForceDisable(); }

aclblasStatus_t aclblasLtLoggerSetCallback(aclblasLtLoggerCallback_t callback)
{
    return AclBlasLt::LoggerManager::GetInstance().SetCallback(callback);
}

aclblasStatus_t aclblasLtLoggerOpenFile(const char* logFile)
{
    return AclBlasLt::LoggerManager::GetInstance().OpenFile(logFile);
}

} // extern "C"
