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

#include <cmath>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fill.h"

namespace cgemv_test {

struct FillMode {
    BlasFillMode base;
    bool isGaussian = false;
};

inline std::vector<std::string> splitFill(const std::string& text)
{
    std::vector<std::string> parts;
    std::istringstream input(text);
    std::string token;
    while (std::getline(input, token, '_')) {
        if (!token.empty()) {
            parts.push_back(token);
        }
    }
    return parts;
}

inline std::string joinFill(const std::vector<std::string>& parts)
{
    std::string text;
    for (const auto& part : parts) {
        if (!text.empty()) {
            text += '_';
        }
        text += part;
    }
    return text;
}

inline FillMode parseFill(const std::string& text)
{
    FillMode fill;
    auto parts = splitFill(text);
    if (parts.size() < 2 || parts[1] != "GAUSSIAN") {
        fill.base = BlasFillMode(text);
        return fill;
    }

    // Reuse the shared numeric parser without extending its pattern enum.
    // Keep numeric tokens verbatim to preserve float parsing and exceptions.
    parts[1] = "NORM";
    fill.isGaussian = parts[0] == "RANDOM";
    if (fill.isGaussian) {
        if (parts.size() == 2) {
            parts.push_back("0");
        }
        if (parts.size() == 3) {
            parts.push_back("1");
        }
    }
    fill.base = BlasFillMode(joinFill(parts));
    return fill;
}

class GaussianGenerator : public ValueGenerator {
    std::mt19937& rng_;
    std::normal_distribution<float> dist_;

    static float checkStddev(float stddev)
    {
        if (!(stddev > 0.0f) || !std::isfinite(stddev)) {
            throw std::invalid_argument("Gaussian standard deviation must be finite and greater than zero");
        }
        return stddev;
    }

public:
    GaussianGenerator(std::mt19937& rng, float mean, float stddev) : rng_(rng), dist_(mean, checkStddev(stddev)) {}
    float at(size_t) override { return dist_(rng_); }
};

inline std::unique_ptr<ValueGenerator> createGenerator(const FillMode& fill, std::mt19937& rng)
{
    if (fill.isGaussian) {
        return std::make_unique<GaussianGenerator>(rng, fill.base.val1, fill.base.val2);
    }
    return ::createGenerator(fill.base, rng);
}

} // namespace cgemv_test
