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
 * \file aclblaslt_handle_impl.cpp
 * \brief Device-capability queries backing the handle / matmul paths.
 */

#include "aclblaslt_handle_impl.h"

#include <acl/acl.h>

uint32_t QueryCubeCoreNum(int32_t deviceId)
{
    int64_t cubeCoreNum = 0;
    aclError aclRet = aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &cubeCoreNum);
    if (aclRet != ACL_SUCCESS || cubeCoreNum <= 0) {
        cubeCoreNum = 8; // Fallback minimum
    }
    return static_cast<uint32_t>(cubeCoreNum);
}
