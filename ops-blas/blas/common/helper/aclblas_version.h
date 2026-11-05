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
 * \file aclblas_version.h
 * \brief Internal ops-blas version macros (not exposed to users).
 */

#pragma once

#define ACLBLAS_VERSION_MAJOR 1
#define ACLBLAS_VERSION_MINOR 0
#define ACLBLAS_VERSION_PATCH 0
#define ACLBLAS_VERSION_STRING "1.0.0"

/**
 * @brief Encode version (major, minor, patch) into an integer.
 *
 * Encoding rule: MAJOR * 10000 + MINOR * 100 + PATCH,
 * e.g. 1.0.0 -> 10000, 1.2.3 -> 10203.
 */
#define ACLBLAS_MAKE_VERSION(major, minor, patch) ((major)*10000 + (minor)*100 + (patch))

#define ACLBLAS_VERSION ACLBLAS_MAKE_VERSION(ACLBLAS_VERSION_MAJOR, ACLBLAS_VERSION_MINOR, ACLBLAS_VERSION_PATCH)
