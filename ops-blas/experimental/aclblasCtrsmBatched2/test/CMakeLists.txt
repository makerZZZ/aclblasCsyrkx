set(CTRSM_BATCHED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ctrsm_batched_test.cpp
    ${CMAKE_SOURCE_DIR}/op_host/ctrsm_batched_host.cpp
    ${CMAKE_SOURCE_DIR}/op_kernel/ctrsm_batched_kernel.cpp)

add_executable(ctrsm_batched_test ${CTRSM_BATCHED_SOURCES})

set_source_files_properties(${CTRSM_BATCHED_SOURCES} PROPERTIES LANGUAGE ASC)

target_include_directories(ctrsm_batched_test PRIVATE
    ${CMAKE_SOURCE_DIR}/op_kernel
    ${CMAKE_SOURCE_DIR}/op_host
    ${CMAKE_SOURCE_DIR}/../../include
    ${CMAKE_SOURCE_DIR}/../../blas
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(ctrsm_batched_test PRIVATE
    tiling_api register platform unified_dlog dl m graph_base)

target_compile_options(ctrsm_batched_test PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
