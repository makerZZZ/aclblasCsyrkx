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
#include <string>

#include "acl/acl.h"
#include "cann_ops_blasLt.h"
#include "csv_loader.h"

// ── dtype string → aclDataType (transform-supported set) ──
inline aclDataType parseTransformDtype(const std::string& s)
{
    if (s == "FP32" || s == "ACL_FLOAT" || s == "0")        return ACL_FLOAT;
    if (s == "FP16" || s == "ACL_FLOAT16" || s == "1")      return ACL_FLOAT16;
    if (s == "INT8" || s == "ACL_INT8" || s == "2")         return ACL_INT8;
    if (s == "INT32" || s == "ACL_INT32" || s == "3")       return ACL_INT32;
    if (s == "BF16" || s == "ACL_BF16" || s == "27")        return ACL_BF16;
    // FP8 / FP4 low-precision dtypes (transform FP8 path scaleType=FP32, FP4 path scaleType=BF16).
    // FP8_E5M2 is enum N-1 of FP8_E4M3FN (35 vs 36); keep both spellings explicit so the CSV string
    // and the numeric id resolve to the same dtype without relying on enum adjacency.
    if (s == "FP8_E4M3FN" || s == "ACL_FLOAT8_E4M3FN" || s == "36") return ACL_FLOAT8_E4M3FN;
    if (s == "FP8_E5M2" || s == "ACL_FLOAT8_E5M2" || s == "35")     return ACL_FLOAT8_E5M2;
    if (s == "FP4_E2M1" || s == "ACL_FLOAT4_E2M1" || s == "40")     return ACL_FLOAT4_E2M1;
    if (s == "FP64" || s == "ACL_DOUBLE" || s == "11")      return ACL_DOUBLE;       // unsupported (E case)
    if (s == "COMPLEX64" || s == "ACL_COMPLEX64" || s == "16") return ACL_COMPLEX64; // unsupported (E case)
    // Unsupported low-precision dtypes (E-26~28 intercept cases): HiF8 / FP4_E1M2 / FP8_E8M0 / FP6.
    if (s == "HIFLOAT8" || s == "ACL_HIFLOAT8" || s == "34")    return ACL_HIFLOAT8;
    if (s == "FP4_E1M2" || s == "ACL_FLOAT4_E1M2" || s == "41") return ACL_FLOAT4_E1M2;
    if (s == "E8M0" || s == "FP8_E8M0" || s == "ACL_FLOAT8_E8M0" || s == "37") return ACL_FLOAT8_E8M0;
    if (s == "FP6" || s == "FP6_E2M3" || s == "ACL_FLOAT6_E2M3" || s == "39")  return ACL_FLOAT6_E2M3;
    return ACL_FLOAT;  // fallback
}

// ── order string → aclblasLtOrder_t ──
// Supported orders: COL/ROW/COL32/COL4_4R2_8C/COL32_2R_4R4 (requirement §2.2).
inline aclblasLtOrder_t parseTransformOrder(const std::string& s)
{
    if (s == "COL" || s == "ACLBLASLT_ORDER_COL" || s == "0")                return ACLBLASLT_ORDER_COL;
    if (s == "ROW" || s == "ACLBLASLT_ORDER_ROW" || s == "1")                return ACLBLASLT_ORDER_ROW;
    if (s == "COL32" || s == "ACLBLASLT_ORDER_COL32" || s == "2")            return ACLBLASLT_ORDER_COL32;
    if (s == "COL4_4R2_8C" || s == "ACLBLASLT_ORDER_COL4_4R2_8C" || s == "3") return ACLBLASLT_ORDER_COL4_4R2_8C;
    if (s == "COL32_2R_4R4" || s == "ACLBLASLT_ORDER_COL32_2R_4R4" || s == "4") return ACLBLASLT_ORDER_COL32_2R_4R4;
    // Illegal/out-of-range enum injection (E-14 / TC_L1_63): a malformed (non-numeric) string maps
    // to an invalid sentinel so the API order validation intercepts it.
    try {
        return static_cast<aclblasLtOrder_t>(std::stoi(s));
    } catch (...) {
        return static_cast<aclblasLtOrder_t>(-1);
    }
}

// ── scaleType string → aclDataType ──
inline aclDataType parseScaleType(const std::string& s)
{
    if (s == "INT32" || s == "ACL_INT32" || s == "3") return ACL_INT32;
    if (s == "BF16" || s == "ACL_BF16" || s == "27")  return ACL_BF16;  // FP4 path scaleType
    return ACL_FLOAT;  // FP32 default (float path / FP8 path)
}

// ── is FP8 transform dtype (E4M3FN / E5M2; FP32 scale-type domain, 1 byte/element) ──
inline bool isFp8TransformDtype(aclDataType dt)
{
    return dt == ACL_FLOAT8_E4M3FN || dt == ACL_FLOAT8_E5M2;
}

// ── is FP4 transform dtype (E2M1; BF16 scale-type domain, packed 2 elements/byte) ──
inline bool isFp4TransformDtype(aclDataType dt)
{
    return dt == ACL_FLOAT4_E2M1;
}

// ── dtype element size in bytes (FP4 is sub-byte: 0.5 byte/element, 2 elements packed/byte) ──
// FP4 returns 1 here would over-count; callers that size packed byte buffers must use the packed
// byte formula (ceil(count/2)). transformDtypeSize is kept as the *logical* element stride helper
// for the linear-layout byte math; FP4 packed byte sizing is done in the npu wrapper via packedLd.
inline size_t transformDtypeSize(aclDataType dt)
{
    switch (dt) {
        case ACL_FLOAT:   return 4;
        case ACL_INT32:   return 4;
        case ACL_FLOAT16: return 2;
        case ACL_BF16:    return 2;
        case ACL_INT8:    return 1;
        case ACL_FLOAT8_E4M3FN:
        case ACL_FLOAT8_E5M2: return 1;   // FP8: 1 byte/element
        case ACL_FLOAT4_E2M1: return 1;   // FP4: logical 0.5 byte; packed byte count = ceil(N/2)
        default:          return 4;
    }
}

// ── FP4 packed leading dim: 2 logical elements per byte (ld in packed bytes = ceil(logicalLd/2)) ──
inline int fp4PackedLd(int logicalLd)
{
    return (logicalLd + 1) / 2;
}

// ── FP4 packed byte count for an element count (ceil(count/2)) ──
inline int64_t fp4PackedBytes(int64_t elemCount)
{
    return (elemCount + 1) / 2;
}

// ── is integer-path dtype ──
inline bool isIntTransformDtype(aclDataType dt)
{
    return dt == ACL_INT8 || dt == ACL_INT32;
}

struct LtMatrixTransformParam : public BlasTestParamBase {
    aclDataType dtypeA = ACL_FLOAT;
    aclDataType dtypeB = ACL_FLOAT;
    aclDataType dtypeC = ACL_FLOAT;
    aclDataType scaleType = ACL_FLOAT;
    aclblasLtOrder_t orderA = ACLBLASLT_ORDER_COL;
    aclblasLtOrder_t orderB = ACLBLASLT_ORDER_COL;
    aclblasLtOrder_t orderC = ACLBLASLT_ORDER_COL;
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    // logical (op-applied) dims of A/B/C; for L0 these are kept equal & relation-compatible.
    int rowsA = 0, colsA = 0;
    int rowsB = 0, colsB = 0;
    int rowsC = 0, colsC = 0;
    int lda = 0, ldb = 0, ldc = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    int batchCount = 1;            // BATCH_COUNT layout attr; >1 expected NOT_SUPPORTED (first batch)

    // null / conflict injection flags
    bool BIsNull = false;          // B device ptr / data omitted (beta=0 single-input)
    bool handleNull = false;       // lightHandle == nullptr (TEST_F parity)
    bool transformDescNull = false;
    bool AdescNull = false;
    bool BdescNull = false;        // Bdesc nullptr while beta≠0 (E-09 / TC_L1_60)
    bool CdescNull = false;
    bool alphaNull = false;
    bool betaNull = false;
    bool ANull = false;            // A device ptr nullptr while Adesc present (conflict path)
    bool CIsNull = false;          // C device ptr nullptr

    LtMatrixTransformParam(const csv_map& m) : BlasTestParamBase(m)
    {
        dtypeA = parseTransformDtype(ReadMap(m, "dtypeA", "FP32"));
        dtypeB = parseTransformDtype(ReadMap(m, "dtypeB", "FP32"));
        dtypeC = parseTransformDtype(ReadMap(m, "dtypeC", "FP32"));
        scaleType = parseScaleType(ReadMap(m, "scale_type", "FP32"));
        orderA = parseTransformOrder(ReadMap(m, "orderA", "COL"));
        orderB = parseTransformOrder(ReadMap(m, "orderB", "COL"));
        orderC = parseTransformOrder(ReadMap(m, "orderC", "COL"));
        transA = parseOpTrans(ReadMap(m, "transA", "N"));
        transB = parseOpTrans(ReadMap(m, "transB", "N"));
        rowsA = parseInt(ReadMap(m, "rowsA", "0"));
        colsA = parseInt(ReadMap(m, "colsA", "0"));
        rowsB = parseInt(ReadMap(m, "rowsB", "0"));
        colsB = parseInt(ReadMap(m, "colsB", "0"));
        rowsC = parseInt(ReadMap(m, "rowsC", "0"));
        colsC = parseInt(ReadMap(m, "colsC", "0"));
        lda = parseInt(ReadMap(m, "lda", "0"));
        ldb = parseInt(ReadMap(m, "ldb", "0"));
        ldc = parseInt(ReadMap(m, "ldc", "0"));
        alpha = parseFloat(ReadMap(m, "alpha", "1.0"));
        beta = parseFloat(ReadMap(m, "beta", "0.0"));
        batchCount = parseInt(ReadMap(m, "batch_count", "1"));

        BIsNull = (ReadMap(m, "B_is_null", "false") == "true");
        handleNull = (ReadMap(m, "handle_null", "false") == "true");
        transformDescNull = (ReadMap(m, "transformDesc_null", "false") == "true");
        AdescNull = (ReadMap(m, "Adesc_null", "false") == "true");
        BdescNull = (ReadMap(m, "Bdesc_null", "false") == "true");
        CdescNull = (ReadMap(m, "Cdesc_null", "false") == "true");
        alphaNull = (ReadMap(m, "alpha_null", "false") == "true");
        betaNull = (ReadMap(m, "beta_null", "false") == "true");
        ANull = (ReadMap(m, "A_null", "false") == "true");
        CIsNull = (ReadMap(m, "C_is_null", "false") == "true");

        // default B dims mirror A if omitted
        if (rowsB == 0 && colsB == 0 && !BIsNull) { rowsB = rowsA; colsB = colsA; }
        if (rowsC == 0 && colsC == 0) { rowsC = rowsA; colsC = colsA; }
    }
};

