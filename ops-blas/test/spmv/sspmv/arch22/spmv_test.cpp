/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file spmv_test.cpp
 * \brief
 */

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

enum class SpmvCompareMode {
    FullMatrix,
    LowerTriangleOnly,
    SymmetricOnly,
};

// Change only this line to switch the verification mode.
constexpr SpmvCompareMode kCompareMode = SpmvCompareMode::FullMatrix;

static const char* GetCompareModeName()
{
    switch (kCompareMode) {
        case SpmvCompareMode::FullMatrix:
            return "full-matrix";
        case SpmvCompareMode::LowerTriangleOnly:
            return "lower-triangle-only";
        case SpmvCompareMode::SymmetricOnly:
            return "symmetric-only";
        default:
            return "unknown";
    }
}

static std::vector<float> BuildGolden(
    const std::vector<float>& aPacked, const std::vector<float>& x, const std::vector<float>& y, uint32_t n,
    int64_t incx, int64_t incy, float alpha, float beta)
{
    auto packedIndex = [n](uint32_t i, uint32_t j) {
        if (i < j) {
            std::swap(i, j);
        }
        return static_cast<size_t>(j) + (static_cast<size_t>(i) * (i + 1ULL)) / 2ULL;
    };

    std::vector<float> golden(n, 0.0f);
    switch (kCompareMode) {
        case SpmvCompareMode::FullMatrix:
            for (uint32_t i = 0; i < n; ++i) {
                float acc = 0.0f;
                for (uint32_t j = 0; j < n; ++j) {
                    uint32_t row = i >= j ? i : j;
                    uint32_t col = i >= j ? j : i;
                    size_t idx = packedIndex(row, col);
                    acc += aPacked[idx] * x[j * incx];
                }
                golden[i] = alpha * acc + beta * y[i * incy];
            }
            break;
        case SpmvCompareMode::LowerTriangleOnly:
            for (uint32_t i = 0; i < n; ++i) {
                float acc = 0.0f;
                for (uint32_t j = 0; j <= i; ++j) {
                    size_t idx = packedIndex(i, j);
                    acc += aPacked[idx] * x[j * incx];
                }
                golden[i] = alpha * acc + beta * y[i * incy];
            }
            break;
        case SpmvCompareMode::SymmetricOnly:
            for (uint32_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (uint32_t i = j + 1; i < n; ++i) {
                    size_t idx = packedIndex(i, j);
                    acc += aPacked[idx] * x[i * incx];
                }
                golden[j] = alpha * acc + beta * y[j * incy];
            }
            break;
        default:
            break;
    }

    return golden;
}

uint32_t VerifyResult(std::vector<float>& output, std::vector<float>& golden)
{
    std::cout << std::fixed << std::setprecision(6);

    auto printTensor = [](std::vector<float>& tensor, const char* name) {
        constexpr size_t maxPrintSize = 20;
        std::cout << name << ": ";
        std::copy(
            tensor.begin(), tensor.begin() + std::min(tensor.size(), maxPrintSize),
            std::ostream_iterator<float>(std::cout, " "));
        if (tensor.size() > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(output, "Output");
    printTensor(golden, "Golden");

    constexpr float absTol = 1e-3f;
    constexpr float relTol = 1e-4f;
    auto closeEnough = [&](float a, float b) {
        float diff = std::abs(a - b);
        float scale = std::max(std::abs(a), std::abs(b));
        return diff <= absTol || diff <= relTol * scale;
    };

    for (size_t i = 0; i < output.size(); ++i) {
        if (!closeEnough(output[i], golden[i])) {
            std::cout << "[Failed] Case accuracy is verification failed at index " << i << " (" << output[i] << " vs "
                      << golden[i] << ")" << std::endl;
            return 1;
        }
    }

    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

static int RunCase(uint32_t n)
{
    int32_t deviceId = 0;

    constexpr float alpha = 1.2f;
    constexpr float beta = 0.8f;
    std::vector<float> x(n, 0.0f);
    std::vector<float> y(n, 0.0f);
    std::mt19937 rng(20260319U + n);
    std::uniform_real_distribution<float> dist(0.0f, 0.5f);
    for (uint32_t i = 0; i < n; ++i) {
        x[i] = dist(rng);
        y[i] = dist(rng);
    }
    int64_t incx = 1;
    int64_t incy = 1;

    // Build packed lower-triangular stored matrix A with deterministic values.
    const size_t packedSize = (static_cast<size_t>(n) * (static_cast<size_t>(n) + 3U) - 2U) / 2U;
    std::vector<float> aPacked(packedSize, 0.0f);
    auto packedIndex = [n](uint32_t i, uint32_t j) {
        if (i < j) {
            std::swap(i, j);
        }
        return static_cast<size_t>(j) + (static_cast<size_t>(i) * (i + 1ULL)) / 2ULL;
    };
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            aPacked[packedIndex(i, j)] = dist(rng);
        }
    }

    aclrtStream stream = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);
    aclblasSetStream(handle, stream);

    std::vector<float> output = y;
    aclblasStatus_t ret = aclblasSspmv(
        handle, ACLBLAS_LOWER, static_cast<int>(n), &alpha, aPacked.data(), x.data(), static_cast<int>(incx), &beta,
        output.data(), static_cast<int>(incy));
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSspmv failed. ERROR: %d\n", ret); return ret);

    std::vector<float> golden = BuildGolden(aPacked, x, y, n, incx, incy, alpha, beta);
    int status = VerifyResult(output, golden);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return status;
}

int32_t main(int32_t argc, char* argv[])
{
    int ret = RunCase(2048); // aligned smoke case
    if (ret != 0) {
        return ret;
    }
    return RunCase(1537); // tail-block smoke case
}