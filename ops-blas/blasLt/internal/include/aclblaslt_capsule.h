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
 * \file aclblaslt_capsule.h
 * \brief Helpers to pack an internal impl struct into its opaque public capsule, zero-padding the
 *        trailing reserved bytes. Not installed.
 */

#pragma once

#include "cann_ops_blasLt.h"

#include <cstddef>

// Copy implBytes from impl into capsule (capacity capsuleBytes) and zero the remaining bytes.
aclblasStatus_t MatPackTransformImpl(void* capsule, size_t capsuleBytes, const void* impl, size_t implBytes);
