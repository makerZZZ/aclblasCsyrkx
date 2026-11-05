# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

ops_blas_add_gtest_tests(${OPS_BLAS} ssymm_test)

if(SOC_ARCH_DIRS MATCHES "arch35")
    target_compile_definitions(ssymm_test PRIVATE SSYMM_ARCH35=1)
elseif(SOC_ARCH_DIRS MATCHES "arch22")
    target_compile_definitions(ssymm_test PRIVATE SSYMM_ARCH22=1)
else()
    message(FATAL_ERROR "Unsupported SOC_ARCH_DIRS='${SOC_ARCH_DIRS}' for ssymm_test (expected arch35 or arch22)")
endif()
