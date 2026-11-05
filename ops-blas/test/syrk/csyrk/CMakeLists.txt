# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# GTest CSV-driven precision ST target (always built when the operator is enabled).
ops_blas_add_gtest_tests(${OPS_BLAS} csyrk_test)

# The CPU golden shares the operator's Netlib reference (csyrk_netlib_reference.h)
# with the host-side special-value fallback so the two stay bit-identical.
target_include_directories(csyrk_test PRIVATE ${CMAKE_SOURCE_DIR}/blas/syrk/arch35)

# This target inherits no optimization flag (CXX_FLAGS comes out empty even with
# CMAKE_BUILD_TYPE=Release), so its host code runs at -O0. The Netlib reference
# loop is the one hot spot: it costs ~6x more at -O0 than at -O2 and dominates a
# full-CSV run (n=2048 alone: 16.9 s vs 2.7 s). -O2 leaves the results bit-identical
# -- the loop's inner body is elementwise, and the x86-64 baseline has no FMA, so
# no reassociation or contraction happens.
target_compile_options(csyrk_test PRIVATE -O2)
