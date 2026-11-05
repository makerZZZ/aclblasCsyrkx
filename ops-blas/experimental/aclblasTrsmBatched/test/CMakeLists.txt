# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set(TRSM_BATCHED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/trsm_batched_test.cpp
    ${CMAKE_SOURCE_DIR}/op_host/trsm_batched_host.cpp
    ${CMAKE_SOURCE_DIR}/op_kernel/trsm_batched_kernel.cpp)

add_executable(trsm_batched_test ${TRSM_BATCHED_SOURCES})

# The ASC toolchain only auto-detects the .asc extension; force ASC language on the .cpp sources
# so they are compiled by the ASC (bisheng) compiler with the NPU arch flags below.
set_source_files_properties(${TRSM_BATCHED_SOURCES} PROPERTIES LANGUAGE ASC)

target_include_directories(trsm_batched_test PRIVATE
    ${CMAKE_SOURCE_DIR}/op_kernel
    ${CMAKE_SOURCE_DIR}/op_host
    ${CMAKE_SOURCE_DIR}/../../include
    ${CMAKE_SOURCE_DIR}/../../blas)

target_link_libraries(trsm_batched_test PRIVATE
    tiling_api register platform unified_dlog dl m graph_base)

target_compile_options(trsm_batched_test PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
