/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "blas_test.h"
#include "cgerc_golden.h"
#include "cgerc_param.h"
#include "csv_loader.h"
#include "verify.h"

namespace {

constexpr uint32_t IMAGINARY_SEED_OFFSET = 1000U;
constexpr uint32_t Y_SEED_OFFSET = 2000U;
constexpr uint32_t A_SEED_OFFSET = 4000U;
constexpr uint64_t COMPLEX_COMPONENT_COUNT = 2U;
constexpr int ZERO_Y_REGRESSION_DIMENSION = 512;
constexpr float ZERO_Y_INITIAL_REAL = 1.25F;
constexpr float ZERO_Y_INITIAL_IMAG = -2.5F;
constexpr uintptr_t DEVICE_ADDRESS_SENTINEL = 1U;
constexpr size_t MATRIX_GUARD_ELEMENTS = 4U;
constexpr int VECTOR_TEST_MIN_ROWS = 64;
constexpr int VECTOR_TEST_TAIL_ROWS = 65;
constexpr int VECTOR_TEST_MULTIPLE_TAIL_ROWS = 257;
constexpr int VECTOR_TEST_PIPELINED_COLUMNS = 1001;
constexpr int VECTOR_TEST_UB_LIMIT_ROWS = 7872;
constexpr int VECTOR_TEST_ABOVE_UB_LIMIT_ROWS = VECTOR_TEST_UB_LIMIT_ROWS + 1;
constexpr int VECTOR_TEST_UNEVEN_COLUMNS = 73;
constexpr int VECTOR_TEST_FEW_COLUMNS = 3;
constexpr float SPECIAL_VALUE_FINITE_REAL = 1.25F;
constexpr float SPECIAL_VALUE_FINITE_IMAG = -2.5F;
constexpr char PERFORMANCE_CASE_PREFIX[] = "TC_PF_";
constexpr size_t PERFORMANCE_CASE_PREFIX_LENGTH = sizeof(PERFORMANCE_CASE_PREFIX) - 1U;
constexpr int64_t FIRST_PERFORMANCE_CASE_ORDINAL = 1001;
constexpr size_t EXPECTED_PERFORMANCE_CASE_COUNT = 200U;
constexpr double MILLISECONDS_TO_MICROSECONDS = 1000.0;
constexpr double PERFORMANCE_THRESHOLD = 0.4;
constexpr int PERFORMANCE_WARMUP_COUNT = 20;
constexpr int PERFORMANCE_BATCH_COUNT = 5;
constexpr int PERFORMANCE_SAMPLES_PER_BATCH = 100;
constexpr size_t MEDIAN_INDEX_DIVISOR = 2U;

uint64_t VectorSpan(int count, int stride)
{
    if (count <= 0 || stride == 0) {
        return 0;
    }
    const uint64_t absStride =
        stride > 0 ? static_cast<uint64_t>(stride) : static_cast<uint64_t>(-static_cast<int64_t>(stride));
    return 1U + static_cast<uint64_t>(count - 1) * absStride;
}

std::vector<aclblasComplex> MakeComplexVector(uint64_t count, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || count == 0) {
        return {};
    }
    std::vector<float> real = makeBlasArray(static_cast<int64_t>(count), fill, seed);
    std::vector<float> imag = makeBlasArray(static_cast<int64_t>(count), fill, seed + IMAGINARY_SEED_OFFSET);
    std::vector<aclblasComplex> result(count);
    for (uint64_t i = 0; i < count; ++i) {
        result[i] = aclblasComplex{real[i], imag[i]};
    }
    return result;
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer()
    {
        if (ptr_ != nullptr) {
            const aclError status = aclrtFree(ptr_);
            if (status != ACL_SUCCESS) {
                std::cerr << "aclrtFree failed with status " << status << std::endl;
            }
        }
    }

    aclError CopyFrom(const void* host, size_t bytes)
    {
        if (host == nullptr || bytes == 0) {
            return ACL_SUCCESS;
        }
        aclError status = aclrtMalloc(&ptr_, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (status != ACL_SUCCESS) {
            return status;
        }
        return aclrtMemcpy(ptr_, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    aclError CopyTo(void* host, size_t bytes) const
    {
        if (host == nullptr || ptr_ == nullptr || bytes == 0) {
            return ACL_SUCCESS;
        }
        return aclrtMemcpy(host, bytes, ptr_, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    template <typename T>
    T* As() const
    {
        return static_cast<T*>(ptr_);
    }

private:
    void* ptr_ = nullptr;
};

class DeviceEvent {
public:
    DeviceEvent() = default;
    DeviceEvent(const DeviceEvent&) = delete;
    DeviceEvent& operator=(const DeviceEvent&) = delete;
    ~DeviceEvent()
    {
        if (event_ != nullptr) {
            const aclError status = aclrtDestroyEvent(event_);
            if (status != ACL_SUCCESS) {
                std::cerr << "aclrtDestroyEvent failed with status " << status << std::endl;
            }
        }
    }

    aclError Create() { return aclrtCreateEvent(&event_); }

    aclrtEvent Get() const { return event_; }

private:
    aclrtEvent event_ = nullptr;
};

struct CgercCaseData {
    explicit CgercCaseData(const CgercParam& param)
        : xHost(MakeComplexVector(VectorSpan(param.m, param.incx), param.x, param.randomSeed)),
          yHost(MakeComplexVector(VectorSpan(param.n, param.incy), param.y, param.randomSeed + Y_SEED_OFFSET)),
          aHost(MakeComplexVector(
              (param.n > 0 && param.lda > 0) ? static_cast<uint64_t>(param.lda) * static_cast<uint64_t>(param.n) : 0U,
              param.A, param.randomSeed + A_SEED_OFFSET)),
          golden(aHost),
          alpha{param.alphaReal, param.alphaImag}
    {}

    std::vector<aclblasComplex> xHost;
    std::vector<aclblasComplex> yHost;
    std::vector<aclblasComplex> aHost;
    std::vector<aclblasComplex> golden;
    aclblasComplex alpha;
};

struct CgercPerfBaseline {
    explicit CgercPerfBaseline(const csv_map& row)
        : id(ReadMap(row, std::string("id"))),
          m(parseInt(ReadMap(row, std::string("m")))),
          n(parseInt(ReadMap(row, std::string("n")))),
          incx(parseInt(ReadMap(row, std::string("incx")))),
          incy(parseInt(ReadMap(row, std::string("incy")))),
          gpuUs(parseDouble(ReadMap(row, std::string("gpu_ms"))) * MILLISECONDS_TO_MICROSECONDS)
    {}

    std::string id;
    int m = 0;
    int n = 0;
    int incx = 0;
    int incy = 0;
    double gpuUs = 0.0;
};

const std::vector<CgercPerfBaseline>& GetGpuBaselines()
{
    static const std::vector<CgercPerfBaseline> baselines = [] {
        std::string path = __FILE__;
        const size_t slash = path.find_last_of("/\\");
        path = (slash == std::string::npos ? std::string{} : path.substr(0, slash + 1U)) + "gpu_baseline.csv";
        return GetCasesFromCsv<CgercPerfBaseline>(path);
    }();
    return baselines;
}

const CgercPerfBaseline* FindGpuBaseline(const std::string& caseName)
{
    if (caseName.rfind(PERFORMANCE_CASE_PREFIX, 0U) != 0U) {
        return nullptr;
    }
    const int64_t ordinal = std::stoll(caseName.substr(PERFORMANCE_CASE_PREFIX_LENGTH));
    const auto& baselines = GetGpuBaselines();
    if (ordinal < FIRST_PERFORMANCE_CASE_ORDINAL) {
        throw std::runtime_error("No GPU baseline row for " + caseName);
    }
    const uint64_t baselineIndex = static_cast<uint64_t>(ordinal - FIRST_PERFORMANCE_CASE_ORDINAL);
    if (baselineIndex >= baselines.size()) {
        throw std::runtime_error("No GPU baseline row for " + caseName);
    }
    return &baselines[static_cast<size_t>(baselineIndex)];
}

} // namespace

class CgercArch35Test : public BlasTest<CgercParam> {
protected:
    void VerifyGuardedMatrix(
        int m, int n, const std::vector<aclblasComplex>& xHost, const std::vector<aclblasComplex>& yHost,
        const aclblasComplex& initial, const std::string& name)
    {
        const aclblasComplex alpha{1.0F, 0.0F};
        const size_t matrixElements = static_cast<size_t>(m) * n;
        std::vector<aclblasComplex> actual(matrixElements + MATRIX_GUARD_ELEMENTS + MATRIX_GUARD_ELEMENTS, initial);
        std::vector<aclblasComplex> golden(actual);
        DeviceBuffer xDevice;
        DeviceBuffer yDevice;
        DeviceBuffer aDevice;
        ASSERT_EQ(xDevice.CopyFrom(xHost.data(), xHost.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
        ASSERT_EQ(yDevice.CopyFrom(yHost.data(), yHost.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
        ASSERT_EQ(aDevice.CopyFrom(actual.data(), actual.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
        ASSERT_EQ(
            aclblasCgerc(
                handle_, m, n, &alpha, xDevice.As<const aclblasComplex>(), 1, yDevice.As<const aclblasComplex>(), 1,
                aDevice.As<aclblasComplex>() + MATRIX_GUARD_ELEMENTS, m),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_EQ(aDevice.CopyTo(actual.data(), actual.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
        ASSERT_EQ(
            aclblasCgercCpu(
                handle_, m, n, &alpha, xHost.data(), 1, yHost.data(), 1, golden.data() + MATRIX_GUARD_ELEMENTS, m),
            ACLBLAS_STATUS_SUCCESS);
        VerifyConfig config;
        applyMixedTolerance(
            config, ACL_FLOAT, reinterpret_cast<const float*>(golden.data()), golden.size() * COMPLEX_COMPONENT_COUNT);
        config.mixedRequiredMatchedRatio = 1.0;
        const bool pass = Verifier::verifyVector(
            reinterpret_cast<const float*>(actual.data()), reinterpret_cast<const float*>(golden.data()),
            golden.size() * COMPLEX_COMPONENT_COUNT, 1U, config, name);
        EXPECT_TRUE(pass);
        for (size_t i = 0U; i < MATRIX_GUARD_ELEMENTS; ++i) {
            EXPECT_FLOAT_EQ(actual[i].real, initial.real);
            EXPECT_FLOAT_EQ(actual[i].imag, initial.imag);
            EXPECT_FLOAT_EQ(actual[MATRIX_GUARD_ELEMENTS + matrixElements + i].real, initial.real);
            EXPECT_FLOAT_EQ(actual[MATRIX_GUARD_ELEMENTS + matrixElements + i].imag, initial.imag);
        }
    }

    aclblasStatus_t LaunchCgerc(
        const CgercParam& param, const aclblasComplex* alpha, const DeviceBuffer& xDevice, const DeviceBuffer& yDevice,
        const DeviceBuffer& aDevice)
    {
        return aclblasCgerc(
            handle_, param.m, param.n, alpha, xDevice.As<const aclblasComplex>(), param.incx,
            yDevice.As<const aclblasComplex>(), param.incy, aDevice.As<aclblasComplex>(), param.lda);
    }

    void VerifyAccuracyResult(const CgercParam& param, const CgercCaseData& data)
    {
        if (data.aHost.empty()) {
            return;
        }
        VerifyConfig config;
        if (param.alphaReal == 0.0F && param.alphaImag == 0.0F) {
            config.mode = PrecisionMode::EXACT;
        } else {
            applyMixedTolerance(
                config, ACL_FLOAT, reinterpret_cast<const float*>(data.golden.data()),
                data.golden.size() * COMPLEX_COMPONENT_COUNT);
        }
        const bool pass = Verifier::verifyVector(
            reinterpret_cast<const float*>(data.aHost.data()), reinterpret_cast<const float*>(data.golden.data()),
            data.golden.size() * COMPLEX_COMPONENT_COUNT, 1U, config, param.caseName);
        EXPECT_TRUE(pass);
    }

    void RunAccuracyCase(
        const CgercParam& param, CgercCaseData& data, DeviceBuffer& xDevice, DeviceBuffer& yDevice,
        DeviceBuffer& aDevice)
    {
        const aclblasComplex* alpha = param.nullAlpha ? nullptr : &data.alpha;
        const aclblasComplex* xHost = data.xHost.empty() ? nullptr : data.xHost.data();
        const aclblasComplex* yHost = data.yHost.empty() ? nullptr : data.yHost.data();
        aclblasComplex* aHost = data.aHost.empty() ? nullptr : data.aHost.data();
        ASSERT_EQ(xDevice.CopyFrom(xHost, data.xHost.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
        ASSERT_EQ(yDevice.CopyFrom(yHost, data.yHost.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
        ASSERT_EQ(aDevice.CopyFrom(aHost, data.aHost.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
        ASSERT_EQ(LaunchCgerc(param, alpha, xDevice, yDevice, aDevice), param.expectResult);
        if (param.expectResult != ACLBLAS_STATUS_SUCCESS) {
            return;
        }
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
        ASSERT_EQ(aDevice.CopyTo(aHost, data.aHost.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
        ASSERT_EQ(
            aclblasCgercCpu(
                handle_, param.m, param.n, alpha, xHost, param.incx, yHost, param.incy,
                data.golden.empty() ? nullptr : data.golden.data(), param.lda),
            ACLBLAS_STATUS_SUCCESS);
        VerifyAccuracyResult(param, data);
    }

    void WarmUpPerformanceCase(
        const CgercParam& param, const aclblasComplex* alpha, const DeviceBuffer& xDevice, const DeviceBuffer& yDevice,
        const DeviceBuffer& aDevice)
    {
        for (int iteration = 0; iteration < PERFORMANCE_WARMUP_COUNT; ++iteration) {
            ASSERT_EQ(LaunchCgerc(param, alpha, xDevice, yDevice, aDevice), ACLBLAS_STATUS_SUCCESS);
        }
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    }

    void MeasurePerformanceBatch(
        const CgercParam& param, const aclblasComplex* alpha, const DeviceBuffer& xDevice, const DeviceBuffer& yDevice,
        const DeviceBuffer& aDevice, std::vector<double>& batchAverageUs)
    {
        DeviceEvent start;
        DeviceEvent end;
        ASSERT_EQ(start.Create(), ACL_SUCCESS);
        ASSERT_EQ(end.Create(), ACL_SUCCESS);
        ASSERT_EQ(aclrtRecordEvent(start.Get(), stream_), ACL_SUCCESS);
        for (int sample = 0; sample < PERFORMANCE_SAMPLES_PER_BATCH; ++sample) {
            ASSERT_EQ(LaunchCgerc(param, alpha, xDevice, yDevice, aDevice), ACLBLAS_STATUS_SUCCESS);
        }
        ASSERT_EQ(aclrtRecordEvent(end.Get(), stream_), ACL_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeEvent(end.Get()), ACL_SUCCESS);
        float totalMs = 0.0F;
        ASSERT_EQ(aclrtEventElapsedTime(&totalMs, start.Get(), end.Get()), ACL_SUCCESS);
        batchAverageUs.push_back(
            static_cast<double>(totalMs) * MILLISECONDS_TO_MICROSECONDS / PERFORMANCE_SAMPLES_PER_BATCH);
    }

    void RunPerformanceCase(
        const CgercParam& param, const CgercCaseData& data, const DeviceBuffer& xDevice, const DeviceBuffer& yDevice,
        const DeviceBuffer& aDevice)
    {
        const CgercPerfBaseline* baseline = FindGpuBaseline(param.caseName);
        if (baseline == nullptr) {
            return;
        }
        ASSERT_EQ(GetGpuBaselines().size(), EXPECTED_PERFORMANCE_CASE_COUNT);
        ASSERT_EQ(param.m, baseline->m);
        ASSERT_EQ(param.n, baseline->n);
        ASSERT_EQ(param.incx, baseline->incx);
        ASSERT_EQ(param.incy, baseline->incy);
        ASSERT_GT(baseline->gpuUs, 0.0);
        const double limitUs = baseline->gpuUs / PERFORMANCE_THRESHOLD;
        const aclblasComplex* alpha = param.nullAlpha ? nullptr : &data.alpha;
        ASSERT_NO_FATAL_FAILURE(WarmUpPerformanceCase(param, alpha, xDevice, yDevice, aDevice));
        std::vector<double> batchAverageUs;
        batchAverageUs.reserve(PERFORMANCE_BATCH_COUNT);
        for (int batch = 0; batch < PERFORMANCE_BATCH_COUNT; ++batch) {
            ASSERT_NO_FATAL_FAILURE(MeasurePerformanceBatch(param, alpha, xDevice, yDevice, aDevice, batchAverageUs));
        }
        std::sort(batchAverageUs.begin(), batchAverageUs.end());
        const double averageUs = batchAverageUs[PERFORMANCE_BATCH_COUNT / MEDIAN_INDEX_DIVISOR];
        const uint64_t deviceBytes =
            (data.xHost.size() + data.yHost.size() + data.aHost.size()) * sizeof(aclblasComplex);
        std::cout << "CGERC_PERF case=" << param.caseName << " baseline_id=" << baseline->id << " m=" << param.m
                  << " n=" << param.n << " warmup=" << PERFORMANCE_WARMUP_COUNT
                  << " batches=" << PERFORMANCE_BATCH_COUNT << " samples=" << PERFORMANCE_SAMPLES_PER_BATCH
                  << " avg_us=" << averageUs << " gpu_us=" << baseline->gpuUs << " threshold=" << PERFORMANCE_THRESHOLD
                  << " limit_us=" << limitUs << " criterion=gpu_div_0.4"
                  << " device_bytes=" << deviceBytes << " verdict=" << (averageUs <= limitUs ? "PASS" : "FAIL")
                  << std::endl;
        EXPECT_LE(averageUs, limitUs);
    }
};

TEST_F(CgercArch35Test, HostParameterValidation)
{
    constexpr int rows = 2;
    constexpr int columns = 3;
    const aclblasComplex alpha{1.0F, 0.0F};
    const aclblasComplex zeroAlpha{0.0F, 0.0F};
    std::vector<aclblasComplex> storage(static_cast<size_t>(rows) * columns, zeroAlpha);
    DeviceBuffer buffer;
    ASSERT_EQ(buffer.CopyFrom(storage.data(), storage.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
    const auto* input = buffer.As<const aclblasComplex>();
    auto* output = buffer.As<aclblasComplex>();

    // A null handle wins even when every other parameter is invalid.
    EXPECT_EQ(
        aclblasCgerc(nullptr, rows, columns, nullptr, input, 0, input, 0, output, 0), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasCgerc(handle_, rows, columns, nullptr, input, 1, input, 1, output, rows), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgerc(handle_, rows, columns, &alpha, input, 0, input, 1, output, rows), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgerc(handle_, rows, columns, &alpha, input, 1, input, 0, output, rows), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgerc(handle_, rows, columns, &zeroAlpha, input, 1, input, 1, output, rows - 1),
        ACLBLAS_STATUS_INVALID_VALUE);
    // This wide column-major matrix has lda=m<n; negative strides are valid even on a no-op path.
    EXPECT_EQ(
        aclblasCgerc(handle_, rows, columns, &zeroAlpha, input, -1, input, -1, output, rows), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CgercArch35Test, ZeroYSkipsInfiniteX)
{
    constexpr int m = ZERO_Y_REGRESSION_DIMENSION;
    constexpr int n = ZERO_Y_REGRESSION_DIMENSION;
    const aclblasComplex alpha{1.0F, 0.0F};
    const aclblasComplex infinite{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
    const aclblasComplex zero{0.0F, 0.0F};
    const aclblasComplex initial{ZERO_Y_INITIAL_REAL, ZERO_Y_INITIAL_IMAG};
    std::vector<aclblasComplex> xHost(m, infinite);
    std::vector<aclblasComplex> yHost(n, zero);
    std::vector<aclblasComplex> aHost(static_cast<size_t>(m) * n, initial);

    DeviceBuffer xDevice;
    DeviceBuffer yDevice;
    DeviceBuffer aDevice;
    ASSERT_EQ(xDevice.CopyFrom(xHost.data(), xHost.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
    ASSERT_EQ(yDevice.CopyFrom(yHost.data(), yHost.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
    ASSERT_EQ(aDevice.CopyFrom(aHost.data(), aHost.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
    ASSERT_EQ(
        aclblasCgerc(
            handle_, m, n, &alpha, xDevice.As<const aclblasComplex>(), 1, yDevice.As<const aclblasComplex>(), 1,
            aDevice.As<aclblasComplex>(), m),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    ASSERT_EQ(aDevice.CopyTo(aHost.data(), aHost.size() * sizeof(aclblasComplex)), ACL_SUCCESS);
    for (const aclblasComplex& value : aHost) {
        ASSERT_FLOAT_EQ(value.real, initial.real);
        ASSERT_FLOAT_EQ(value.imag, initial.imag);
    }
}

TEST_F(CgercArch35Test, AddressRangeOverflow)
{
    const aclblasComplex alpha{1.0F, 0.0F};
    const auto* input = reinterpret_cast<const aclblasComplex*>(DEVICE_ADDRESS_SENTINEL);
    auto* output = reinterpret_cast<aclblasComplex*>(DEVICE_ADDRESS_SENTINEL);
    constexpr int maxInt = std::numeric_limits<int>::max();
    EXPECT_EQ(
        aclblasCgerc(handle_, maxInt, maxInt, &alpha, input, 1, input, 1, output, maxInt),
        ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CgercArch35Test, VectorSpecialValues)
{
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<aclblasComplex> values{
        {SPECIAL_VALUE_FINITE_REAL, SPECIAL_VALUE_FINITE_IMAG},
        {infinity, 1.0F},
        {-infinity, -1.0F},
        {1.0F, infinity},
        {1.0F, -infinity},
        {infinity, infinity},
        {nan, 1.0F},
        {1.0F, nan},
        {std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()}};
    const std::vector<int> rows{VECTOR_TEST_MIN_ROWS, VECTOR_TEST_TAIL_ROWS, VECTOR_TEST_MULTIPLE_TAIL_ROWS};
    const aclblasComplex initial{SPECIAL_VALUE_FINITE_REAL, SPECIAL_VALUE_FINITE_IMAG};
    for (int m : rows) {
        const int n = VECTOR_TEST_PIPELINED_COLUMNS;
        for (size_t special = 0U; special < values.size(); ++special) {
            SCOPED_TRACE("m=" + std::to_string(m) + " special=" + std::to_string(special));
            std::vector<aclblasComplex> xHost(m, values.front());
            std::vector<aclblasComplex> yHost(n, values[special]);
            ASSERT_NO_FATAL_FAILURE(VerifyGuardedMatrix(m, n, xHost, yHost, initial, "SIMD_special_y"));
            std::fill(xHost.begin(), xHost.end(), values[special]);
            std::fill(yHost.begin(), yHost.end(), values.front());
            ASSERT_NO_FATAL_FAILURE(VerifyGuardedMatrix(m, n, xHost, yHost, initial, "SIMD_special_x"));
        }
    }
}

TEST_F(CgercArch35Test, VectorPartitionAndUbBoundaries)
{
    const std::vector<int> rows{
        VECTOR_TEST_MIN_ROWS, VECTOR_TEST_TAIL_ROWS, VECTOR_TEST_UB_LIMIT_ROWS, VECTOR_TEST_ABOVE_UB_LIMIT_ROWS};
    const std::vector<int> columns{
        1, VECTOR_TEST_FEW_COLUMNS, VECTOR_TEST_UNEVEN_COLUMNS, VECTOR_TEST_PIPELINED_COLUMNS};
    const aclblasComplex initial{SPECIAL_VALUE_FINITE_REAL, SPECIAL_VALUE_FINITE_IMAG};
    for (int m : rows) {
        for (int n : columns) {
            SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n));
            const std::vector<aclblasComplex> xHost(m, initial);
            const std::vector<aclblasComplex> yHost(n, initial);
            ASSERT_NO_FATAL_FAILURE(VerifyGuardedMatrix(m, n, xHost, yHost, initial, "SIMD_partition_UB_boundary"));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Cgerc, CgercArch35Test, ::testing::ValuesIn(GetCasesFromCsv<CgercParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgercParam>);

TEST_P(CgercArch35Test, CsvDriven)
{
    const CgercParam& param = GetParam();
    CgercCaseData data(param);
    DeviceBuffer xDevice;
    DeviceBuffer yDevice;
    DeviceBuffer aDevice;
    ASSERT_NO_FATAL_FAILURE(RunAccuracyCase(param, data, xDevice, yDevice, aDevice));
    ASSERT_NO_FATAL_FAILURE(RunPerformanceCase(param, data, xDevice, yDevice, aDevice));
}
