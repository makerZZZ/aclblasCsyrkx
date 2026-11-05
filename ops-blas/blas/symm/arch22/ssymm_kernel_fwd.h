/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include "cann_ops_blas_common.h"

#define GM_ADDR uint8_t*

struct SsymmRightCubeUnifiedConfig;
struct SsymmLeftCubeConfig;

void ssymm_kernel_do(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tilingGm,
                     aclblasSideMode_t side, aclblasFillMode_t uplo,
                     uint32_t numBlocks, void *stream);

void ssymm_right_cube_unified_do(GM_ADDR a, GM_ADDR aDense, GM_ADDR b, GM_ADDR c, GM_ADDR scratch,
                                 GM_ADDR tilingGm, GM_ADDR configGm,
                                 const SsymmRightCubeUnifiedConfig &config, uint32_t numBlocks, void *stream);

void ssymm_right_cube_unified_postprocess_do(GM_ADDR scratch, GM_ADDR c, GM_ADDR configGm, void *stream);

void ssymm_left_cube_do(GM_ADDR aSym, GM_ADDR workspace, GM_ADDR b, GM_ADDR scratch,
                        GM_ADDR configGm, void *stream);

void ssymm_left_cube_postprocess_do(GM_ADDR scratch, GM_ADDR c, GM_ADDR configGm, void *stream);

