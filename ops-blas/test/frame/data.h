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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

class DataGenerator {
public:
    static void fillConstant(std::vector<float>& data, float value) {
        std::fill(data.begin(), data.end(), value);
    }

    static void fillRandom(std::vector<float>& data, uint32_t seed, float minVal = -10.0f, float maxVal = 10.0f) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(minVal, maxVal);
        for (auto& elem : data) elem = dist(rng);
    }

    static void fillDeterministic(std::vector<float>& data, float start = 1.0f, float step = 1.0f) {
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = start + static_cast<float>(i) * step;
        }
    }

    static void fillBandedMatrix(float* A, int64_t m, int64_t n, int64_t kl, int64_t ku,
                                  int64_t lda, std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        std::fill(A, A + static_cast<size_t>(lda) * static_cast<size_t>(n), 0.0f);
        for (int64_t j = 0; j < n; j++) {
            int64_t iStart = (j > ku) ? (j - ku) : 0;
            int64_t iEnd = (j + kl < m - 1) ? (j + kl) : (m - 1);
            for (int64_t i = iStart; i <= iEnd; i++) {
                int64_t bandedRow = ku + i - j;
                A[bandedRow + j * lda] = dist(rng);
            }
        }
    }

    static float* fillStridedVector(float* v, int64_t len, int64_t inc, std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        int64_t absInc = (inc < 0) ? -inc : inc;
        std::fill(v, v + static_cast<size_t>((len - 1) * absInc + 1), 0.0f);
        for (int64_t i = 0; i < len; i++) {
            v[i * absInc] = dist(rng);
        }
        return (inc < 0) ? (v + (len - 1) * absInc) : v;
    }

    static void fillPackedMatrix(float* ap, int64_t n, bool upper, std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        if (upper) {
            for (int64_t j = 0; j < n; j++) {
                for (int64_t i = 0; i <= j; i++) {
                    int64_t idx = j * n + i - j * (j + 1) / 2;
                    ap[idx] = dist(rng);
                }
            }
        } else {
            for (int64_t j = 0; j < n; j++) {
                for (int64_t i = j; i < n; i++) {
                    int64_t idx = i + j * (2 * n - j - 1) / 2;
                    ap[idx] = dist(rng);
                }
            }
        }
    }

    static void fillTriangularMatrix(float* A, int64_t n, int64_t lda,
                                      bool upper, bool unitDiag, std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        std::fill(A, A + static_cast<size_t>(lda) * static_cast<size_t>(n), 0.0f);
        if (upper) {
            for (int64_t j = 0; j < n; j++) {
                for (int64_t i = 0; i <= j; i++) {
                    if (i == j && unitDiag) {
                        A[i + j * lda] = 1.0f;
                    } else {
                        A[i + j * lda] = dist(rng);
                    }
                }
            }
        } else {
            for (int64_t j = 0; j < n; j++) {
                for (int64_t i = j; i < n; i++) {
                    if (i == j && unitDiag) {
                        A[i + j * lda] = 1.0f;
                    } else {
                        A[i + j * lda] = dist(rng);
                    }
                }
            }
        }
    }

    static void fillRandomComplex(std::vector<float>& data, uint32_t seed,
                                   float minVal = -10.0f, float maxVal = 10.0f) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(minVal, maxVal);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = dist(rng);
        }
    }
};

