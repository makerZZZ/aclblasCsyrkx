/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_strided_batched_ex_host.cpp
 * \brief Independent aclblasGemmStridedBatchedEx for Ascend 950/DAV-3510.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/dtype_cast.h"
#include "common/helper/host_utils.h"
#include "gemm_strided_batched_ex_kernel.h"

namespace {

constexpr char OP_NAME[] = "aclblasGemmStridedBatchedEx";
constexpr int32_t MAX_SPLIT_K = 4;
constexpr int32_t MAX_INT8_K_WITH_I32_ACCUMULATION = std::numeric_limits<int32_t>::max() / (128 * 128);
static_assert(sizeof(aclblasComplex) == 2 * sizeof(float), "aclblasComplex must be an interleaved pair of FP32 values");

static uint32_t GetEpilogueBlockCount(int64_t batchCount, int64_t m, int64_t n, uint32_t blockNum)
{
    constexpr uint64_t ELEMENTS_PER_CORE = 16384;
    if (blockNum == 0 || batchCount <= 0 || m <= 0 || n <= 0) {
        return 0;
    }

    const uint64_t maxValue = std::numeric_limits<uint64_t>::max();
    const uint64_t batch = static_cast<uint64_t>(batchCount);
    const uint64_t rows = static_cast<uint64_t>(m);
    const uint64_t cols = static_cast<uint64_t>(n);
    if (batch > maxValue / rows || batch * rows > maxValue / cols) {
        return blockNum;
    }

    const uint64_t elements = batch * rows * cols;
    const uint64_t needed = elements / ELEMENTS_PER_CORE + (elements % ELEMENTS_PER_CORE != 0 ? 1 : 0);
    return static_cast<uint32_t>(std::min<uint64_t>(blockNum, needed));
}

struct GemmStridedProblem {
    aclblasOperation_t transA;
    aclblasOperation_t transB;
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    int32_t batchCount;
    int64_t strideA;
    int64_t strideB;
    int64_t strideC;
    aclDataType aType;
    aclDataType bType;
    aclDataType cType;
    aclblasComputeType_t computeType;
    aclblasGemmAlgo_t requestedAlgo;
    GemmStridedBatchedExDtypeCase dtypeCase;
    GemmStridedBatchedExScalarKind scalarKind;
    float alpha;
    float beta;
    float alphaImag;
    float betaImag;
    int32_t alphaInt;
    int32_t betaInt;
};

static bool IsFp8(aclDataType type) { return type == ACL_FLOAT8_E4M3FN || type == ACL_FLOAT8_E5M2; }

static size_t ElementSize(aclDataType type)
{
    switch (type) {
        case ACL_FLOAT8_E4M3FN:
        case ACL_FLOAT8_E5M2:
        case ACL_INT8:
            return 1;
        case ACL_FLOAT16:
        case ACL_BF16:
            return 2;
        case ACL_FLOAT:
        case ACL_INT32:
            return 4;
        case ACL_COMPLEX64:
            return 8;
        default:
            return 0;
    }
}

static bool IsCompute16F(aclblasComputeType_t computeType)
{
    return computeType == ACLBLAS_COMPUTE_16F || computeType == ACLBLAS_COMPUTE_16F_PEDANTIC;
}

static bool IsCompute32F(aclblasComputeType_t computeType)
{
    return computeType == ACLBLAS_COMPUTE_32F || computeType == ACLBLAS_COMPUTE_32F_PEDANTIC;
}

static bool IsFastCompute32F(aclblasComputeType_t computeType)
{
    return computeType == ACLBLAS_COMPUTE_32F_FAST_16F || computeType == ACLBLAS_COMPUTE_32F_FAST_16BF ||
           computeType == ACLBLAS_COMPUTE_32F_FAST_TF32;
}

static bool IsCompute32I(aclblasComputeType_t computeType)
{
    return computeType == ACLBLAS_COMPUTE_32I || computeType == ACLBLAS_COMPUTE_32I_PEDANTIC;
}

template <typename T>
static bool CheckedMultiply(T lhs, T rhs, T& result)
{
    static_assert(std::numeric_limits<T>::is_integer, "integer required");
    if (lhs == 0 || rhs == 0) {
        result = 0;
        return true;
    }
    if (lhs > std::numeric_limits<T>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool CheckedSignedMultiply(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs == 0 || rhs == 0) {
        result = 0;
        return true;
    }

    const int64_t minValue = std::numeric_limits<int64_t>::min();
    const int64_t maxValue = std::numeric_limits<int64_t>::max();
    const bool positiveOverflow =
        (lhs > 0 && rhs > 0 && lhs > maxValue / rhs) || (lhs < 0 && rhs < 0 && rhs < maxValue / lhs);
    const bool negativeOverflow =
        (lhs > 0 && rhs < 0 && rhs < minValue / lhs) || (lhs < 0 && rhs > 0 && lhs < minValue / rhs);
    if (positiveOverflow || negativeOverflow) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool CheckedAddressRange(
    const void* base, int64_t stride, int32_t batchCount, int64_t matrixLastElement, size_t elementSize)
{
    int64_t lastBatch = 0;
    int64_t minElement = 0;
    int64_t maxElement = matrixLastElement;
    if (batchCount > 1 && !CheckedSignedMultiply(stride, static_cast<int64_t>(batchCount - 1), lastBatch)) {
        return false;
    }
    if (lastBatch < 0) {
        minElement = lastBatch;
    } else {
        if (matrixLastElement > std::numeric_limits<int64_t>::max() - lastBatch) {
            return false;
        }
        maxElement += lastBatch;
    }
    int64_t minBytes = 0;
    int64_t maxBytes = 0;
    if (!CheckedSignedMultiply(minElement, static_cast<int64_t>(elementSize), minBytes) ||
        !CheckedSignedMultiply(maxElement, static_cast<int64_t>(elementSize), maxBytes)) {
        return false;
    }
    const uintptr_t address = reinterpret_cast<uintptr_t>(base);
    if (minBytes < 0 && static_cast<uint64_t>(-(minBytes + 1)) + 1U > address) {
        return false;
    }
    return maxBytes < 0 || static_cast<uint64_t>(maxBytes) <= std::numeric_limits<uintptr_t>::max() - address;
}

static bool CheckedMatrixLastElement(int32_t rows, int32_t cols, int32_t ld, int64_t& last)
{
    if (rows <= 0 || cols <= 0) {
        last = 0;
        return true;
    }
    int64_t colOffset = 0;
    if (!CheckedSignedMultiply(static_cast<int64_t>(cols - 1), static_cast<int64_t>(ld), colOffset) ||
        colOffset > std::numeric_limits<int64_t>::max() - (rows - 1)) {
        return false;
    }
    last = colOffset + rows - 1;
    return true;
}

static GemmStridedBatchedExDtypeCase ResolveFp16Case(
    aclDataType aType, aclDataType bType, aclDataType cType, aclblasComputeType_t computeType)
{
    if (aType != ACL_FLOAT16 || bType != ACL_FLOAT16) {
        return GEMM_STRIDED_DTYPE_INVALID;
    }
    if (cType == ACL_FLOAT16 && (IsCompute16F(computeType) || IsCompute32F(computeType))) {
        return GEMM_STRIDED_DTYPE_FP16_F16;
    }
    return cType == ACL_FLOAT && IsCompute32F(computeType) ? GEMM_STRIDED_DTYPE_FP16_F32 : GEMM_STRIDED_DTYPE_INVALID;
}

static GemmStridedBatchedExDtypeCase ResolveBf16Case(
    aclDataType aType, aclDataType bType, aclDataType cType, aclblasComputeType_t computeType)
{
    if (aType != ACL_BF16 || bType != ACL_BF16 || !IsCompute32F(computeType)) {
        return GEMM_STRIDED_DTYPE_INVALID;
    }
    if (cType == ACL_BF16) {
        return GEMM_STRIDED_DTYPE_BF16_BF16;
    }
    return cType == ACL_FLOAT ? GEMM_STRIDED_DTYPE_BF16_F32 : GEMM_STRIDED_DTYPE_INVALID;
}

static GemmStridedBatchedExDtypeCase ResolveFp32Case(
    aclDataType aType, aclDataType bType, aclDataType cType, aclblasComputeType_t computeType)
{
    if (aType == ACL_FLOAT && bType == ACL_FLOAT && cType == ACL_FLOAT &&
        (IsCompute32F(computeType) || IsFastCompute32F(computeType))) {
        return GEMM_STRIDED_DTYPE_FP32_F32;
    }
    return GEMM_STRIDED_DTYPE_INVALID;
}

static GemmStridedBatchedExDtypeCase ResolveComplexCase(
    aclDataType aType, aclDataType bType, aclDataType cType, aclblasComputeType_t computeType)
{
    if (aType == ACL_COMPLEX64 && bType == ACL_COMPLEX64 && cType == ACL_COMPLEX64 &&
        (IsCompute32F(computeType) || IsFastCompute32F(computeType))) {
        return GEMM_STRIDED_DTYPE_COMPLEX32_F32;
    }
    return GEMM_STRIDED_DTYPE_INVALID;
}

static GemmStridedBatchedExDtypeCase ResolveInt8Case(
    aclDataType aType, aclDataType bType, aclDataType cType, aclblasComputeType_t computeType)
{
    if (aType != ACL_INT8 || bType != ACL_INT8) {
        return GEMM_STRIDED_DTYPE_INVALID;
    }
    if (cType == ACL_INT32 && IsCompute32I(computeType)) {
        return GEMM_STRIDED_DTYPE_INT8_I32;
    }
    return cType == ACL_FLOAT && IsCompute32F(computeType) ? GEMM_STRIDED_DTYPE_INT8_F32 : GEMM_STRIDED_DTYPE_INVALID;
}

static GemmStridedBatchedExDtypeCase ResolveFp8Pair(aclDataType aType, aclDataType bType, bool outF32)
{
    if (aType == ACL_FLOAT8_E4M3FN && bType == ACL_FLOAT8_E4M3FN) {
        return outF32 ? GEMM_STRIDED_DTYPE_FP8_E4M3_F32 : GEMM_STRIDED_DTYPE_FP8_E4M3_F16;
    }
    if (aType == ACL_FLOAT8_E5M2 && bType == ACL_FLOAT8_E5M2) {
        return outF32 ? GEMM_STRIDED_DTYPE_FP8_E5M2_F32 : GEMM_STRIDED_DTYPE_FP8_E5M2_F16;
    }
    if (aType == ACL_FLOAT8_E4M3FN) {
        return outF32 ? GEMM_STRIDED_DTYPE_FP8_E4M3_E5M2_F32 : GEMM_STRIDED_DTYPE_FP8_E4M3_E5M2_F16;
    }
    return outF32 ? GEMM_STRIDED_DTYPE_FP8_E5M2_E4M3_F32 : GEMM_STRIDED_DTYPE_FP8_E5M2_E4M3_F16;
}

static GemmStridedBatchedExDtypeCase ResolveFp8Case(
    aclDataType aType, aclDataType bType, aclDataType cType, aclblasComputeType_t computeType)
{
    if (!IsFp8(aType) || !IsFp8(bType) || !IsCompute32F(computeType) || (cType != ACL_FLOAT16 && cType != ACL_FLOAT)) {
        return GEMM_STRIDED_DTYPE_INVALID;
    }
    return ResolveFp8Pair(aType, bType, cType == ACL_FLOAT);
}

static GemmStridedBatchedExDtypeCase ResolveDtypeCase(
    aclDataType aType, aclDataType bType, aclDataType cType, aclblasComputeType_t computeType)
{
    using Resolver = GemmStridedBatchedExDtypeCase (*)(aclDataType, aclDataType, aclDataType, aclblasComputeType_t);
    static const Resolver resolvers[] = {ResolveFp16Case,    ResolveBf16Case, ResolveFp32Case,
                                         ResolveComplexCase, ResolveInt8Case, ResolveFp8Case};
    for (Resolver resolver : resolvers) {
        const GemmStridedBatchedExDtypeCase result = resolver(aType, bType, cType, computeType);
        if (result != GEMM_STRIDED_DTYPE_INVALID) {
            return result;
        }
    }
    return GEMM_STRIDED_DTYPE_INVALID;
}

static GemmStridedBatchedExScalarKind ResolveScalarKind(aclblasComputeType_t computeType)
{
    if (IsCompute16F(computeType)) {
        return GEMM_STRIDED_SCALAR_F16;
    }
    if (IsCompute32I(computeType)) {
        return GEMM_STRIDED_SCALAR_I32;
    }
    return GEMM_STRIDED_SCALAR_F32;
}

static float ReadScalar(const void* value, GemmStridedBatchedExScalarKind kind)
{
    if (value == nullptr) {
        return 0.0f;
    }
    if (kind == GEMM_STRIDED_SCALAR_F16) {
        return blas_common::HalfToFloat(*static_cast<const uint16_t*>(value));
    }
    if (kind == GEMM_STRIDED_SCALAR_I32) {
        return static_cast<float>(*static_cast<const int32_t*>(value));
    }
    return *static_cast<const float*>(value);
}

static bool IsIntegerCompute(const GemmStridedProblem& p) { return p.dtypeCase == GEMM_STRIDED_DTYPE_INT8_I32; }

static bool IsComplexCompute(const GemmStridedProblem& p) { return p.dtypeCase == GEMM_STRIDED_DTYPE_COMPLEX32_F32; }

static bool IsAlphaZero(const GemmStridedProblem& p)
{
    if (IsIntegerCompute(p)) {
        return p.alphaInt == 0;
    }
    return p.alpha == 0.0f && (!IsComplexCompute(p) || p.alphaImag == 0.0f);
}

static bool IsBetaZero(const GemmStridedProblem& p)
{
    if (IsIntegerCompute(p)) {
        return p.betaInt == 0;
    }
    return p.beta == 0.0f && (!IsComplexCompute(p) || p.betaImag == 0.0f);
}

static bool IsBetaOne(const GemmStridedProblem& p)
{
    if (IsIntegerCompute(p)) {
        return p.betaInt == 1;
    }
    return p.beta == 1.0f && (!IsComplexCompute(p) || p.betaImag == 0.0f);
}

static bool IsValidOperation(aclblasOperation_t operation)
{
    return operation == ACLBLAS_OP_N || operation == ACLBLAS_OP_T || operation == ACLBLAS_OP_C;
}

static aclblasStatus_t ValidateShapeAndOperation(const GemmStridedProblem& p)
{
    if (p.m < 0 || p.n < 0 || p.k < 0 || p.batchCount < 0) {
        OP_LOGE(OP_NAME, "negative shape: m=%d n=%d k=%d batch=%d", p.m, p.n, p.k, p.batchCount);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!IsValidOperation(p.transA) || !IsValidOperation(p.transB)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateLeadingDimensions(const GemmStridedProblem& p)
{
    const int32_t minLda = p.transA == ACLBLAS_OP_N ? std::max(1, p.m) : std::max(1, p.k);
    const int32_t minLdb = p.transB == ACLBLAS_OP_N ? std::max(1, p.k) : std::max(1, p.n);
    if (p.lda < minLda || p.ldb < minLdb || p.ldc < std::max(1, p.m)) {
        OP_LOGE(
            OP_NAME, "invalid leading dimensions lda=%d/%d ldb=%d/%d ldc=%d/%d", p.lda, minLda, p.ldb, minLdb, p.ldc,
            std::max(1, p.m));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateDtype(const GemmStridedProblem& p)
{
    if (p.dtypeCase == GEMM_STRIDED_DTYPE_INVALID) {
        OP_LOGE(
            OP_NAME, "unsupported dtype/compute A=%d B=%d C=%d compute=%d", static_cast<int>(p.aType),
            static_cast<int>(p.bType), static_cast<int>(p.cType), static_cast<int>(p.computeType));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if ((p.dtypeCase == GEMM_STRIDED_DTYPE_INT8_I32 || p.dtypeCase == GEMM_STRIDED_DTYPE_INT8_F32) &&
        p.k > MAX_INT8_K_WITH_I32_ACCUMULATION) {
        OP_LOGE(
            OP_NAME, "INT8 k=%d exceeds INT32 Cube accumulation limit=%d", p.k,
            MAX_INT8_K_WITH_I32_ACCUMULATION);
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateBasic(
    aclblasHandle_t handle, const GemmStridedProblem& p, const void* alpha, const void* beta)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    const aclblasStatus_t shapeStatus = ValidateShapeAndOperation(p);
    if (shapeStatus != ACLBLAS_STATUS_SUCCESS) {
        return shapeStatus;
    }
    if (alpha == nullptr || beta == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const aclblasStatus_t layoutStatus = ValidateLeadingDimensions(p);
    return layoutStatus == ACLBLAS_STATUS_SUCCESS ? ValidateDtype(p) : layoutStatus;
}

static bool RequiresInputMatrices(const GemmStridedProblem& p) { return !IsAlphaZero(p) && p.k != 0; }

static bool HasAlignedInt8Inputs(const GemmStridedProblem& p, const void* a, const void* b)
{
    return (reinterpret_cast<uintptr_t>(a) & 3U) == 0 && (reinterpret_cast<uintptr_t>(b) & 3U) == 0 &&
           (p.lda & 3) == 0 && (p.ldb & 3) == 0 && (p.strideA & 3) == 0 && (p.strideB & 3) == 0;
}

static bool HasValidLogicalElementCount(const GemmStridedProblem& p)
{
    uint64_t logicalElements = 0;
    uint64_t allLogicalElements = 0;
    return CheckedMultiply<uint64_t>(static_cast<uint64_t>(p.m), static_cast<uint64_t>(p.n), logicalElements) &&
           CheckedMultiply<uint64_t>(logicalElements, static_cast<uint64_t>(p.batchCount), allLogicalElements) &&
           allLogicalElements <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
}

static aclblasStatus_t ValidateStorageMetadata(const GemmStridedProblem& p, const void* a, const void* b, const void* c)
{
    const bool usesMultipleBatches = p.batchCount > 1;
    if (usesMultipleBatches && p.strideC == 0) {
        OP_LOGE(OP_NAME, "strideC must be nonzero when batchCount is greater than one");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (usesMultipleBatches && RequiresInputMatrices(p) && (p.strideA == 0 || p.strideB == 0)) {
        OP_LOGE(OP_NAME, "strideA and strideB must be nonzero when multiple input batches are read");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!HasValidLogicalElementCount(p)) {
        OP_LOGE(OP_NAME, "logical element count overflows");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (c == nullptr || (RequiresInputMatrices(p) && (a == nullptr || b == nullptr))) {
        OP_LOGE(OP_NAME, "null storage for a non-empty operation");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (IsIntegerCompute(p) && RequiresInputMatrices(p) && !HasAlignedInt8Inputs(p, a, b)) {
        OP_LOGE(OP_NAME, "INT8 to INT32 requires 4-byte aligned A/B batch addresses, lda/ldb and strides");
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateStorageAddressRanges(
    const GemmStridedProblem& p, const void* a, const void* b, const void* c)
{
    int64_t aLast = 0;
    int64_t bLast = 0;
    int64_t cLast = 0;
    const int32_t aRows = p.transA == ACLBLAS_OP_N ? p.m : p.k;
    const int32_t aCols = p.transA == ACLBLAS_OP_N ? p.k : p.m;
    const int32_t bRows = p.transB == ACLBLAS_OP_N ? p.k : p.n;
    const int32_t bCols = p.transB == ACLBLAS_OP_N ? p.n : p.k;
    if (!CheckedMatrixLastElement(aRows, aCols, p.lda, aLast) ||
        !CheckedMatrixLastElement(bRows, bCols, p.ldb, bLast) || !CheckedMatrixLastElement(p.m, p.n, p.ldc, cLast)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (RequiresInputMatrices(p) && (!CheckedAddressRange(a, p.strideA, p.batchCount, aLast, ElementSize(p.aType)) ||
                                     !CheckedAddressRange(b, p.strideB, p.batchCount, bLast, ElementSize(p.bType)))) {
        OP_LOGE(OP_NAME, "A/B address calculation overflows");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!CheckedAddressRange(c, p.strideC, p.batchCount, cLast, ElementSize(p.cType))) {
        OP_LOGE(OP_NAME, "C address calculation overflows");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ValidateStorage(const GemmStridedProblem& p, const void* a, const void* b, const void* c)
{
    if (p.m == 0 || p.n == 0 || p.batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclblasStatus_t metadataStatus = ValidateStorageMetadata(p, a, b, c);
    return metadataStatus == ACLBLAS_STATUS_SUCCESS ? ValidateStorageAddressRanges(p, a, b, c) : metadataStatus;
}

static bool IsFp16Input(const GemmStridedProblem& p) { return p.aType == ACL_FLOAT16 && p.bType == ACL_FLOAT16; }

static aclblasStatus_t SelectAlgorithm(
    const GemmStridedProblem& p, uint32_t coreCount, int32_t& selected, int32_t& splitK)
{
    selected = GEMM_STRIDED_KERNEL_BALANCED;
    splitK = 1;
    if (p.requestedAlgo == ACLBLAS_GEMM_DEFAULT) {
        if (IsFp16Input(p) && p.m <= 64 && p.n <= 64) {
            selected = GEMM_STRIDED_KERNEL_SMALL;
        } else if (IsFp16Input(p) && p.k >= 1024) {
            selected = GEMM_STRIDED_KERNEL_DEEP_K;
        } else if (IsFp16Input(p) && static_cast<uint32_t>(p.batchCount) > coreCount * 2U) {
            selected = GEMM_STRIDED_KERNEL_PERSISTENT;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }
    const int32_t algoValue = static_cast<int32_t>(p.requestedAlgo);
    if (algoValue < static_cast<int32_t>(ACLBLAS_GEMM_ALGO0) || algoValue > static_cast<int32_t>(ACLBLAS_GEMM_ALGO7)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    selected = algoValue - static_cast<int32_t>(ACLBLAS_GEMM_ALGO0);
    if (!IsFp16Input(p) && selected != GEMM_STRIDED_KERNEL_BALANCED) {
        OP_LOGE(OP_NAME, "explicit ALGO%d is not supported for dtypeCase=%d", selected, static_cast<int>(p.dtypeCase));
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (selected == GEMM_STRIDED_KERNEL_SMALL && (p.m > 256 || p.n > 256)) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (selected == GEMM_STRIDED_KERNEL_DEEP_K && p.k < 32) {
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (selected == GEMM_STRIDED_KERNEL_SPLIT_K) {
        if (p.k < 32) {
            return ACLBLAS_STATUS_NOT_SUPPORTED;
        }
        splitK = std::min(MAX_SPLIT_K, std::max(2, p.k / 64));
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static const char* KernelName(int32_t selected)
{
    static const char* const names[] = {
        "balanced_2d", "batch_first",           "m_major",         "n_major", "small_matrix",
        "deep_k_bk32", "split_k_deterministic", "persistent_batch"};
    return selected >= 0 && selected < 8 ? names[selected] : "invalid";
}

static void SelectBaseTile(
    const GemmStridedProblem& p, int32_t selected, int32_t& bm, int32_t& bn, int32_t& bk, int32_t& c0)
{
    if (p.aType == ACL_FLOAT) {
        bm = 32;
        bn = 16;
        bk = 8;
        c0 = 8;
    } else if (p.aType == ACL_INT8) {
        bm = 128;
        bn = 128;
        bk = 32;
        c0 = 32;
    } else if (IsFp8(p.aType)) {
        bm = 32;
        bn = 16;
        bk = 32;
        c0 = 32;
    } else if (selected == GEMM_STRIDED_KERNEL_SMALL) {
        bm = 64;
        bn = 64;
        bk = 16;
        c0 = 16;
    } else if (selected == GEMM_STRIDED_KERNEL_DEEP_K) {
        bm = 64;
        bn = 64;
        bk = 32;
        c0 = 16;
    } else {
        bm = 128;
        bn = 128;
        bk = 16;
        c0 = 16;
    }
}

static bool BuildTiling(
    const GemmStridedProblem& p, int32_t selected, int32_t splitK, uint32_t coreCount, bool directOutput,
    GemmStridedBatchedExTilingData& t)
{
    t.logicalM = p.m;
    t.logicalN = p.n;
    t.m = p.n;
    t.n = p.m;
    t.k = p.k;
    t.lda = p.ldb;
    t.ldb = p.lda;
    t.ldc = directOutput ? p.ldc : p.m;
    t.isTransA = p.transB == ACLBLAS_OP_N ? 0 : 1;
    t.isTransB = p.transA == ACLBLAS_OP_N ? 0 : 1;
    t.batchCount = p.batchCount;
    t.selectedAlgo = selected;
    t.splitK = splitK;
    t.hasBeta = IsBetaZero(p) ? 0 : 1;
    t.scalarKind = p.scalarKind;
    t.dtypeCase = p.dtypeCase;
    t.alpha = p.alpha;
    t.beta = p.beta;
    t.alphaImag = p.alphaImag;
    t.betaImag = p.betaImag;
    t.alphaInt = p.alphaInt;
    t.betaInt = p.betaInt;
    t.strideA = p.strideB;
    t.strideB = p.strideA;
    t.strideC = p.strideC;
    SelectBaseTile(p, selected, t.baseM, t.baseN, t.baseK, t.c0Size);
    const int64_t mBlocks = (static_cast<int64_t>(t.m) + t.baseM - 1) / t.baseM;
    const int64_t nBlocks = (static_cast<int64_t>(t.n) + t.baseN - 1) / t.baseN;
    if (mBlocks <= 0 || nBlocks <= 0 || mBlocks > std::numeric_limits<int32_t>::max() ||
        nBlocks > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    t.mBlocks = static_cast<int32_t>(mBlocks);
    t.nBlocks = static_cast<int32_t>(nBlocks);
    t.singleCoreM = t.baseM;
    t.singleCoreN = t.baseN;

    uint64_t tasks = 0;
    uint64_t spatialTasks = 0;
    uint64_t temp = 0;
    if (!CheckedMultiply<uint64_t>(static_cast<uint64_t>(t.mBlocks), static_cast<uint64_t>(t.nBlocks), spatialTasks) ||
        spatialTasks > std::numeric_limits<uint32_t>::max() ||
        !CheckedMultiply<uint64_t>(static_cast<uint64_t>(p.batchCount), spatialTasks, temp) ||
        !CheckedMultiply<uint64_t>(temp, static_cast<uint64_t>(splitK), tasks)) {
        return false;
    }
    t.totalTasks = tasks;
    t.usedCoreNum = static_cast<int32_t>(std::min<uint64_t>(coreCount, tasks));
    return t.usedCoreNum > 0;
}

static bool WorkspaceSize(const GemmStridedProblem& p, int32_t splitK, size_t& matrixElements, size_t& requiredBytes)
{
    size_t elements = 0;
    if (!CheckedMultiply<size_t>(static_cast<size_t>(p.m), static_cast<size_t>(p.n), matrixElements) ||
        !CheckedMultiply<size_t>(matrixElements, static_cast<size_t>(p.batchCount), elements) ||
        !CheckedMultiply<size_t>(elements, static_cast<size_t>(splitK), elements) ||
        !CheckedMultiply<size_t>(elements, sizeof(float), requiredBytes)) {
        return false;
    }
    return true;
}

static GemmStridedBatchedExDtypeCase ResolveCubeOutputCase(const GemmStridedProblem& p, bool directOutput)
{
    if (directOutput) {
        return p.dtypeCase;
    }
    switch (p.dtypeCase) {
        case GEMM_STRIDED_DTYPE_FP16_F16:
            return GEMM_STRIDED_DTYPE_FP16_F32;
        case GEMM_STRIDED_DTYPE_BF16_BF16:
            return GEMM_STRIDED_DTYPE_BF16_F32;
        case GEMM_STRIDED_DTYPE_FP8_E4M3_F16:
            return GEMM_STRIDED_DTYPE_FP8_E4M3_F32;
        case GEMM_STRIDED_DTYPE_FP8_E5M2_F16:
            return GEMM_STRIDED_DTYPE_FP8_E5M2_F32;
        case GEMM_STRIDED_DTYPE_FP8_E4M3_E5M2_F16:
            return GEMM_STRIDED_DTYPE_FP8_E4M3_E5M2_F32;
        case GEMM_STRIDED_DTYPE_FP8_E5M2_E4M3_F16:
            return GEMM_STRIDED_DTYPE_FP8_E5M2_E4M3_F32;
        case GEMM_STRIDED_DTYPE_INT8_F32:
            return GEMM_STRIDED_DTYPE_INT8_I32;
        default:
            return p.dtypeCase;
    }
}

static aclblasStatus_t LaunchEarlyExit(aclblasHandle_t handle, const GemmStridedProblem& p, void* c)
{
    if (IsBetaOne(p)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    GemmStridedBatchedExTilingData tiling{};
    tiling.logicalM = p.m;
    tiling.logicalN = p.n;
    tiling.ldc = p.ldc;
    tiling.batchCount = p.batchCount;
    tiling.strideC = p.strideC;
    tiling.beta = p.beta;
    tiling.betaImag = p.betaImag;
    tiling.hasBeta = IsBetaZero(p) ? 0 : 1;
    tiling.scalarKind = p.scalarKind;
    tiling.alphaInt = p.alphaInt;
    tiling.betaInt = p.betaInt;
    const uint32_t blocks = GetEpilogueBlockCount(p.batchCount, p.m, p.n, cores);
    auto* h = handle;
    gemm_strided_batched_ex_epilogue_do(
        blocks, h->stream, nullptr, reinterpret_cast<uint8_t*>(c), tiling, p.dtypeCase, true);
    OP_LOGI(
        OP_NAME, "requestedAlgo=%d selected=early_exit kernel=aiv_scale blocks=%u workspace=0",
        static_cast<int>(p.requestedAlgo), blocks);
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t LaunchComplex(
    aclblasHandle_t handle, const GemmStridedProblem& p, const void* a, const void* b, void* c)
{
    if (p.requestedAlgo != ACLBLAS_GEMM_DEFAULT && p.requestedAlgo != ACLBLAS_GEMM_ALGO0) {
        OP_LOGE(OP_NAME, "only DEFAULT and ALGO0 are supported for Complex-FP32");
        return ACLBLAS_STATUS_NOT_SUPPORTED;
    }
    if (IsAlphaZero(p) && IsBetaOne(p)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    GemmStridedBatchedExTilingData tiling{};
    tiling.logicalM = p.m;
    tiling.logicalN = p.n;
    tiling.k = IsAlphaZero(p) ? 0 : p.k;
    tiling.lda = p.lda;
    tiling.ldb = p.ldb;
    tiling.ldc = p.ldc;
    tiling.isTransA = p.transA == ACLBLAS_OP_N ? 0 : (p.transA == ACLBLAS_OP_C ? 2 : 1);
    tiling.isTransB = p.transB == ACLBLAS_OP_N ? 0 : (p.transB == ACLBLAS_OP_C ? 2 : 1);
    tiling.batchCount = p.batchCount;
    tiling.strideA = p.strideA;
    tiling.strideB = p.strideB;
    tiling.strideC = p.strideC;
    tiling.alpha = p.alpha;
    tiling.beta = p.beta;
    tiling.alphaImag = p.alphaImag;
    tiling.betaImag = p.betaImag;
    const uint32_t blocks = GetEpilogueBlockCount(p.batchCount, p.m, p.n, cores);
    auto* h = handle;
    gemm_strided_batched_ex_complex_do(
        blocks, h->stream, reinterpret_cast<uint8_t*>(const_cast<void*>(a)),
        reinterpret_cast<uint8_t*>(const_cast<void*>(b)), reinterpret_cast<uint8_t*>(c), tiling);
    OP_LOGI(
        OP_NAME, "requestedAlgo=%d selected=complex_simt blocks=%u workspace=0", static_cast<int>(p.requestedAlgo),
        blocks);
    return ACLBLAS_STATUS_SUCCESS;
}

static bool UsesDirectOutput(const GemmStridedProblem& p, int32_t splitK)
{
    const bool directInteger = IsIntegerCompute(p) && p.alphaInt == 1 && p.betaInt == 0;
    const bool directFloat =
        p.dtypeCase != GEMM_STRIDED_DTYPE_INT8_F32 && !IsIntegerCompute(p) && p.alpha == 1.0f && p.beta == 0.0f;
    return splitK == 1 && (directInteger || directFloat);
}

static aclblasStatus_t PrepareCubeOutput(
    aclblasHandle_t handle, const GemmStridedProblem& p, int32_t splitK, bool directOutput,
    GemmStridedBatchedExTilingData& tiling, size_t& workspaceBytes, uint8_t*& cubeOutput)
{
    if (splitK <= 0) {
        OP_LOGE(OP_NAME, "invalid splitK=%d", splitK);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (directOutput) {
        tiling.workspaceBatchStride = p.strideC;
        tiling.workspaceSplitStride = 0;
        return ACLBLAS_STATUS_SUCCESS;
    }
    size_t matrixElements = 0;
    int64_t workspaceBatchStride = 0;
    if (!WorkspaceSize(p, splitK, matrixElements, workspaceBytes) ||
        matrixElements > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        !CheckedSignedMultiply(
            static_cast<int64_t>(matrixElements), static_cast<int64_t>(splitK), workspaceBatchStride)) {
        OP_LOGE(OP_NAME, "workspace calculation overflow");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    auto* h = handle;
    const aclblasStatus_t status = EnsureDefaultWorkspace(h, workspaceBytes);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE(OP_NAME, "workspace=%zu is unavailable, status=%d", workspaceBytes, static_cast<int>(status));
        return status;
    }
    cubeOutput = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));
    if (cubeOutput == nullptr) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    tiling.workspaceBatchStride = workspaceBatchStride;
    tiling.workspaceSplitStride = static_cast<int64_t>(matrixElements);
    return ACLBLAS_STATUS_SUCCESS;
}

static void LaunchCubeKernel(
    aclblasHandle_t handle, const GemmStridedProblem& p, const void* a, const void* b, uint8_t* cubeOutput,
    int32_t selected, int32_t splitK, size_t workspaceBytes, bool directOutput,
    const GemmStridedBatchedExTilingData& tiling)
{
    auto* h = handle;
    OP_LOGI(
        OP_NAME,
        "requestedAlgo=%d selectedAlgo=%d kernel=%s BM=%d BN=%d BK=%d "
        "batch=%d mBlocks=%d nBlocks=%d splitK=%d tasks=%llu blocks=%d workspace=%zu",
        static_cast<int>(p.requestedAlgo), selected, KernelName(selected), tiling.baseM, tiling.baseN, tiling.baseK,
        p.batchCount, tiling.mBlocks, tiling.nBlocks, splitK, static_cast<unsigned long long>(tiling.totalTasks),
        tiling.usedCoreNum, workspaceBytes);

    const GemmStridedBatchedExDtypeCase cubeCase = ResolveCubeOutputCase(p, directOutput);
    // Column-major mapping: Cube receives B as its left input and A as its right input.
    gemm_strided_batched_ex_kernel_do(
        static_cast<uint32_t>(tiling.usedCoreNum), h->stream, reinterpret_cast<uint8_t*>(const_cast<void*>(b)),
        reinterpret_cast<uint8_t*>(const_cast<void*>(a)), cubeOutput, tiling, cubeCase);
}

static void LaunchMainEpilogue(
    aclblasHandle_t handle, const GemmStridedProblem& p, void* c, uint8_t* cubeOutput, uint32_t aivCores,
    const GemmStridedBatchedExTilingData& tiling)
{
    const uint32_t blocks = GetEpilogueBlockCount(p.batchCount, p.m, p.n, aivCores);
    GemmStridedBatchedExTilingData epilogueTiling = tiling;
    epilogueTiling.ldc = p.ldc;
    auto* h = handle;
    gemm_strided_batched_ex_epilogue_do(
        blocks, h->stream, cubeOutput, reinterpret_cast<uint8_t*>(c), epilogueTiling, p.dtypeCase, false);
}

static aclblasStatus_t LaunchMain(
    aclblasHandle_t handle, const GemmStridedProblem& p, const void* a, const void* b, void* c)
{
    const uint32_t cubeCores = GetAicCoreCount();
    const uint32_t aivCores = GetAivCoreCount();
    if (cubeCores == 0 || aivCores == 0) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    int32_t selected = 0;
    int32_t splitK = 1;
    aclblasStatus_t status = SelectAlgorithm(p, cubeCores, selected, splitK);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    const bool directOutput = UsesDirectOutput(p, splitK);
    GemmStridedBatchedExTilingData tiling{};
    if (!BuildTiling(p, selected, splitK, cubeCores, directOutput, tiling)) {
        OP_LOGE(OP_NAME, "task count overflow or invalid zero-task tiling");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    size_t workspaceBytes = 0;
    uint8_t* cubeOutput = reinterpret_cast<uint8_t*>(c);
    status = PrepareCubeOutput(handle, p, splitK, directOutput, tiling, workspaceBytes, cubeOutput);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    LaunchCubeKernel(handle, p, a, b, cubeOutput, selected, splitK, workspaceBytes, directOutput, tiling);
    if (!directOutput) {
        LaunchMainEpilogue(handle, p, c, cubeOutput, aivCores, tiling);
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static GemmStridedProblem MakeProblem(
    aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, int lda, int ldb, int ldc,
    int batchCount, int64_t strideA, int64_t strideB, int64_t strideC, aclDataType aType, aclDataType bType,
    aclDataType cType, aclblasComputeType_t computeType, aclblasGemmAlgo_t algo)
{
    return {
        transa,
        transb,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        batchCount,
        strideA,
        strideB,
        strideC,
        aType,
        bType,
        cType,
        computeType,
        algo,
        ResolveDtypeCase(aType, bType, cType, computeType),
        ResolveScalarKind(computeType),
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        0,
        0};
}

static void LoadScalars(GemmStridedProblem& p, const void* alpha, const void* beta)
{
    if (p.scalarKind == GEMM_STRIDED_SCALAR_I32) {
        p.alphaInt = *static_cast<const int32_t*>(alpha);
        p.betaInt = *static_cast<const int32_t*>(beta);
        p.alpha = static_cast<float>(p.alphaInt);
        p.beta = static_cast<float>(p.betaInt);
        return;
    }
    if (IsComplexCompute(p)) {
        const auto* alphaComplex = static_cast<const aclblasComplex*>(alpha);
        const auto* betaComplex = static_cast<const aclblasComplex*>(beta);
        p.alpha = alphaComplex->real;
        p.alphaImag = alphaComplex->imag;
        p.beta = betaComplex->real;
        p.betaImag = betaComplex->imag;
        return;
    }
    p.alpha = ReadScalar(alpha, p.scalarKind);
    p.beta = ReadScalar(beta, p.scalarKind);
}

} // namespace

aclblasStatus_t aclblasGemmStridedBatchedEx(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k,
    const void* alpha, const void* a, aclDataType aType, int lda, int64_t strideA, const void* b, aclDataType bType,
    int ldb, int64_t strideB, const void* beta, void* c, aclDataType cType, int ldc, int64_t strideC, int batchCount,
    aclblasComputeType_t computeType, aclblasGemmAlgo_t algo)
{
    GemmStridedProblem p = MakeProblem(
        transa, transb, m, n, k, lda, ldb, ldc, batchCount, strideA, strideB, strideC, aType, bType, cType, computeType,
        algo);
    aclblasStatus_t status = ValidateBasic(handle, p, alpha, beta);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    LoadScalars(p, alpha, beta);
    if (m == 0 || n == 0 || batchCount == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    status = ValidateStorage(p, a, b, c);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (IsComplexCompute(p)) {
        return LaunchComplex(handle, p, a, b, c);
    }
    if (k == 0 || IsAlphaZero(p)) {
        return LaunchEarlyExit(handle, p, c);
    }
    return LaunchMain(handle, p, a, b, c);
}
