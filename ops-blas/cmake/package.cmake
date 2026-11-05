# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#### CPACK to package run #####

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cmake)

# makeself 依赖的路径（与 build.sh 中 CANN_3RD_LIB_PATH 对应）
if(NOT DEFINED CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH "${CMAKE_BINARY_DIR}/third_party")
endif()
# download makeself package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/makeself-fetch.cmake)

# Ascend 安装路径，用于生成 version.info 中的 compiler 版本
if(NOT DEFINED ASCEND_INSTALL_DIR)
  if(DEFINED ASCEND_CANN_PACKAGE_PATH AND ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_INSTALL_DIR "${ASCEND_CANN_PACKAGE_PATH}")
  else()
    set(ASCEND_INSTALL_DIR "$ENV{ASCEND_HOME_PATH}")
    if(NOT ASCEND_INSTALL_DIR)
      set(ASCEND_INSTALL_DIR "$ENV{ASCEND_TOOLKIT_HOME}")
    endif()
    if(NOT ASCEND_INSTALL_DIR)
      set(ASCEND_INSTALL_DIR "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
  endif()
endif()

# 生成 version.info 到 build 目录，供 install 使用
set(OPS_BLAS_VERSION_INFO "${CMAKE_BINARY_DIR}/version.info")
add_custom_target(gen_ops_blas_version_info ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}"
  COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/util/gen_version_info.sh"
    "${ASCEND_INSTALL_DIR}"
    "${CMAKE_BINARY_DIR}"
    "${OPS_BLAS_RUN_VERSION}"
  BYPRODUCTS "${OPS_BLAS_VERSION_INFO}"
  COMMENT "Generating version.info for ops_blas run package"
)

# 安装 version.info 到 run 包的 share/info/ops_blas 目录
install(FILES "${OPS_BLAS_VERSION_INFO}"
  DESTINATION share/info/ops_blas
  PERMISSIONS OWNER_READ GROUP_READ WORLD_READ
)

# 安装 install.sh 和 help.info 到 run 包，供 --install 使用
install(PROGRAMS "${CMAKE_SOURCE_DIR}/scripts/package/install.sh"
  DESTINATION share/info/ops_blas
)
install(FILES "${CMAKE_SOURCE_DIR}/scripts/package/help.info"
  DESTINATION share/info/ops_blas
)

function(pack _soc_version)
  # 打印路径
  message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
  message(STATUS "CMAKE_BINARY_DIR = ${CMAKE_BINARY_DIR}")
  # ============= CPack =============
  # arc 通过 uname -m 获取
  execute_process(COMMAND uname -m OUTPUT_VARIABLE PACK_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
  # SOC_VERSION -> CPACK_CHIP_SHORT 映射（参考 ops-nn remove_ascend）
  string(TOLOWER "${_soc_version}" _soc_lower)
  if(_soc_lower MATCHES "^ascend910b")
    set(CPACK_CHIP_SHORT "910b")
  elseif(_soc_lower MATCHES "^ascend950")
    set(CPACK_CHIP_SHORT "950")
  elseif(_soc_lower MATCHES "^ascend910_93")
    set(CPACK_CHIP_SHORT "A3")
  elseif(_soc_lower MATCHES "^ascend310p")
    set(CPACK_CHIP_SHORT "310p")
  else()
    set(CPACK_CHIP_SHORT "950")
  endif()
  set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
  set(CPACK_PACKAGE_VERSION "${OPS_BLAS_RUN_VERSION}")
  set(CPACK_PACKAGE_FILE_NAME "cann-${CPACK_CHIP_SHORT}-ops-blas_${OPS_BLAS_RUN_VERSION}_linux-${PACK_ARCH}")
  message(STATUS "CPACK_PACKAGE_FILE_NAME=${CPACK_PACKAGE_FILE_NAME} (SOC=${_soc_version} -> ${CPACK_CHIP_SHORT})")

  set(CPACK_INSTALL_PREFIX "/")

  set(CPACK_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
  set(CPACK_CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}")
  set(CPACK_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
  set(CPACK_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  set(CPACK_MAKESELF_PATH "${MAKESELF_PATH}")
  set(CPACK_SOC "${_soc_version}")
  set(CPACK_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
  set(CPACK_SET_DESTDIR ON)
  set(CPACK_GENERATOR External)
  set(CPACK_EXTERNAL_PACKAGE_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/makeself.cmake")
  set(CPACK_EXTERNAL_ENABLE_STAGING true)
  set(CPACK_PACKAGE_DIRECTORY "${CMAKE_SOURCE_DIR}/build_out")

  message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
  include(CPack)
endfunction()