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
 * \file aclblaslt_version.cpp
 * \brief Public C API: library version and property queries.
 */

#include "cann_ops_blasLt.h"

#include "aclblaslt_handle_impl.h"

#include <cstddef>

extern "C" {

aclblasStatus_t aclblasLtGetVersion(size_t* version)
{
    if (version == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    *version = (static_cast<size_t>(ACLBLASLT_VERSION_MAJOR) << 24) |
               (static_cast<size_t>(ACLBLASLT_VERSION_MINOR) << 16) | static_cast<size_t>(ACLBLASLT_VERSION_PATCH);
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasLtGetProperty(aclblasLtPropertyType_t type, int* value)
{
    if (value == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    switch (type) {
        case ACLBLASLT_PROPERTY_MAJOR_VERSION:
            *value = ACLBLASLT_VERSION_MAJOR;
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_PROPERTY_MINOR_VERSION:
            *value = ACLBLASLT_VERSION_MINOR;
            return ACLBLAS_STATUS_SUCCESS;
        case ACLBLASLT_PROPERTY_PATCH_LEVEL:
            *value = ACLBLASLT_VERSION_PATCH;
            return ACLBLAS_STATUS_SUCCESS;
        default:
            return ACLBLAS_STATUS_INVALID_VALUE;
    }
}

} // extern "C"
