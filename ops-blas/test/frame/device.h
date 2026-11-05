/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"

class DeviceBuffer {
    void* devPtr_ = nullptr;
    size_t size_ = 0;

public:
    explicit DeviceBuffer(size_t bytes) : size_(bytes) {
        aclError ret = aclrtMalloc(&devPtr_, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMalloc failed: " + std::to_string(ret));
        }
    }

    ~DeviceBuffer() {
        if (devPtr_) {
            aclrtFree(devPtr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : devPtr_(other.devPtr_), size_(other.size_) {
        other.devPtr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (devPtr_) aclrtFree(devPtr_);
            devPtr_ = other.devPtr_;
            size_ = other.size_;
            other.devPtr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void copyFromHost(const void* hostData, size_t bytes) {
        aclError ret = aclrtMemcpy(devPtr_, size_, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMemcpy H2D failed: " + std::to_string(ret));
        }
    }

    void copyToHost(void* hostData, size_t bytes) {
        aclError ret = aclrtMemcpy(hostData, bytes, devPtr_, size_, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMemcpy D2H failed: " + std::to_string(ret));
        }
    }

    void* ptr() const { return devPtr_; }
    uint8_t* bytePtr() const { return static_cast<uint8_t*>(devPtr_); }
    float* floatPtr() const { return static_cast<float*>(devPtr_); }
    size_t size() const { return size_; }
};

inline std::vector<std::unique_ptr<DeviceBuffer>> allocAndCopyToDevice(
    const std::vector<std::vector<float>>& hostData) {
    std::vector<std::unique_ptr<DeviceBuffer>> devBufs;
    devBufs.reserve(hostData.size());
    for (const auto& hData : hostData) {
        size_t bytes = hData.empty() ? sizeof(float) : hData.size() * sizeof(float);
        auto buf = std::make_unique<DeviceBuffer>(bytes);
        if (!hData.empty()) {
            buf->copyFromHost(hData.data(), bytes);
        }
        devBufs.push_back(std::move(buf));
    }
    return devBufs;
}

// BLAS negative stride: first logical element is at the end of the allocated buffer.
inline float* adjustStridedBase(float* base, int64_t count, int64_t stride) {
    if (stride < 0 && count > 0) {
        const int64_t absStride = -stride;
        return base + (count - 1) * absStride;
    }
    return base;
}

inline const float* adjustStridedBase(const float* base, int64_t count, int64_t stride) {
    return adjustStridedBase(const_cast<float*>(base), count, stride);
}

