/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstring>
#include <initializer_list>

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace {
struct Resources {
    bool initialized = false;
    bool deviceSet = false;
    aclblasHandle_t handle = nullptr;
    aclrtStream stream = nullptr;
    void* x = nullptr;
    void* y = nullptr;
    void* a = nullptr;
    ~Resources()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        for (void* allocation : {x, y, a}) {
            if (allocation != nullptr) {
                aclrtFree(allocation);
            }
        }
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(0);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};
} // namespace

#define CHECK(call, success) do { \
    const auto status = (call); \
    if (status != (success)) { \
        std::fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(status)); \
        return 1; \
    } \
} while (false)

int main()
{
    Resources r;
    CHECK(aclInit(nullptr), ACL_SUCCESS);
    r.initialized = true;
    CHECK(aclrtSetDevice(0), ACL_SUCCESS);
    r.deviceSet = true;
    CHECK(aclblasCreate(&r.handle), ACLBLAS_STATUS_SUCCESS);
    CHECK(aclrtCreateStream(&r.stream), ACL_SUCCESS);
    CHECK(aclblasSetStream(r.handle, r.stream), ACLBLAS_STATUS_SUCCESS);

    const aclblasComplex alpha{1.0f, 1.0f};
    const aclblasComplex x[]{{1.0f, 2.0f}, {-1.0f, 1.0f}};
    const aclblasComplex y[]{{2.0f, -1.0f}, {3.0f, 2.0f}};
    // Column-major storage, n=lda=2. Only UPPER is updated.
    aclblasComplex a[]{{1.0f, 1.0f}, {2.0f, 3.0f}, {2.0f, 3.0f}, {-1.0f, 4.0f}};
    const aclblasComplex expected[]{{3.0f, 15.0f}, {2.0f, 3.0f}, {-11.0f, 12.0f}, {-13.0f, -4.0f}};
    CHECK(aclrtMalloc(&r.x, sizeof(x), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    CHECK(aclrtMalloc(&r.y, sizeof(y), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    CHECK(aclrtMalloc(&r.a, sizeof(a), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    CHECK(aclrtMemcpy(r.x, sizeof(x), x, sizeof(x), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    CHECK(aclrtMemcpy(r.y, sizeof(y), y, sizeof(y), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    CHECK(aclrtMemcpy(r.a, sizeof(a), a, sizeof(a), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    CHECK(aclblasCsyr2(r.handle, ACLBLAS_UPPER, 2, &alpha,
        static_cast<const aclblasComplex*>(r.x), 1, static_cast<const aclblasComplex*>(r.y), 1,
        static_cast<aclblasComplex*>(r.a), 2), ACLBLAS_STATUS_SUCCESS);
    CHECK(aclrtSynchronizeStream(r.stream), ACL_SUCCESS);
    CHECK(aclrtMemcpy(a, sizeof(a), r.a, sizeof(a), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (int index = 0; index < 4; ++index) {
        std::printf("A[%d,%d] = (%g,%g)\n", index % 2, index / 2, a[index].real, a[index].imag);
    }
    CHECK(std::memcmp(a, expected, sizeof(a)), 0);
    std::puts("PASS: complex symmetric UPPER update; lower element preserved");
    return 0;
}
