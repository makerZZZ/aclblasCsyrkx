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

static constexpr uint8_t BUFFER_NUM = 2;
static constexpr uint32_t TMPBUF_NUM = 3;
static constexpr uint32_t ELENUM_LINE_ALIGNED = 32;  // max elements in line
static constexpr uint32_t UBUF_SIZE = 190 * 1024;    // 190kb
static constexpr uint32_t BYTENUM_BLOCK = 32;
static constexpr uint32_t BYTENUM_REPEAT = 256;
static constexpr uint32_t COMPLEX_ELENUM = 2;
static constexpr uint32_t FP16_DTYPE_SIZE = sizeof(float) / 2;  // 2b

enum class aclDataType_t {
    ACL_C_32F = 0,
    ACL_C_64F
};

static uint32_t CalMaxMatNum(bool isTrans, bool dataType, uint32_t m)
{
    uint32_t matNum = 0;
    if (isTrans) {
        if (dataType) {
            // trans && fp32
            matNum = (UBUF_SIZE / COMPLEX_ELENUM - (3 * BUFFER_NUM + TMPBUF_NUM) * BYTENUM_REPEAT) /
                     (BUFFER_NUM * sizeof(float) * m * ELENUM_LINE_ALIGNED +
                         2 * BUFFER_NUM * sizeof(float) * ELENUM_LINE_ALIGNED + sizeof(uint32_t) * ELENUM_LINE_ALIGNED +
                         TMPBUF_NUM * sizeof(float) * m * ELENUM_LINE_ALIGNED + ELENUM_LINE_ALIGNED * BYTENUM_BLOCK);
        } else {
            // trans && fp16
            matNum =
                (UBUF_SIZE / COMPLEX_ELENUM - (3 * BUFFER_NUM + TMPBUF_NUM + 2) * BYTENUM_REPEAT) /
                (BUFFER_NUM * FP16_DTYPE_SIZE * m * ELENUM_LINE_ALIGNED +
                    2 * BUFFER_NUM * FP16_DTYPE_SIZE * ELENUM_LINE_ALIGNED + sizeof(uint32_t) * ELENUM_LINE_ALIGNED +
                    TMPBUF_NUM * sizeof(float) * m * ELENUM_LINE_ALIGNED + ELENUM_LINE_ALIGNED * BYTENUM_BLOCK +
                    m * ELENUM_LINE_ALIGNED * FP16_DTYPE_SIZE + ELENUM_LINE_ALIGNED * sizeof(float));
        }
    } else {
        if (dataType) {
            // no trans && fp32
            matNum = (UBUF_SIZE / COMPLEX_ELENUM - (3 * BUFFER_NUM + TMPBUF_NUM + 1) * BYTENUM_REPEAT) /
                     (BUFFER_NUM * sizeof(float) * m * ELENUM_LINE_ALIGNED +
                         2 * BUFFER_NUM * sizeof(float) * ELENUM_LINE_ALIGNED + sizeof(uint32_t) * ELENUM_LINE_ALIGNED +
                         TMPBUF_NUM * sizeof(float) * m * ELENUM_LINE_ALIGNED + ELENUM_LINE_ALIGNED * sizeof(float));
        } else {
            // no trans && fp16
            matNum =
                (UBUF_SIZE / COMPLEX_ELENUM - (3 * BUFFER_NUM + TMPBUF_NUM + 3) * BYTENUM_REPEAT) /
                (BUFFER_NUM * FP16_DTYPE_SIZE * m * ELENUM_LINE_ALIGNED +
                    2 * BUFFER_NUM * FP16_DTYPE_SIZE * ELENUM_LINE_ALIGNED + sizeof(uint32_t) * ELENUM_LINE_ALIGNED +
                    TMPBUF_NUM * sizeof(float) * m * ELENUM_LINE_ALIGNED + ELENUM_LINE_ALIGNED * sizeof(float) +
                    m * ELENUM_LINE_ALIGNED * FP16_DTYPE_SIZE + ELENUM_LINE_ALIGNED * FP16_DTYPE_SIZE);
        }
    }
    matNum = matNum > 0 ? matNum : 1;
    return matNum;
}

uint32_t *CreateCgemvBatchedMask(uint32_t m, uint32_t dtype, uint32_t trans)
{
    if (m <= 0) {
        m = 1;
    }

    bool isTrans = trans == 0 ? false : true;    // 0: ACLBLAS_OP_N
    bool dataType = (dtype == (uint32_t)aclDataType_t::ACL_C_64F) ? true : false;
    uint32_t dtypeSize = dataType ? sizeof(float) : FP16_DTYPE_SIZE;

    uint32_t maxMatNum = 1;
    maxMatNum = CalMaxMatNum(isTrans, dataType, static_cast<uint32_t>(m));
    maxMatNum = maxMatNum > 0 ? maxMatNum : 1;

    uint32_t eleNumPerRepeat = BYTENUM_REPEAT / dtypeSize;
    uint32_t maskSize = maxMatNum * ELENUM_LINE_ALIGNED * COMPLEX_ELENUM;
    uint32_t *maskData = nullptr;

    maskData = new uint32_t[maskSize];

    uint32_t realOffset = 0;
    uint32_t imagOffset = maxMatNum * ELENUM_LINE_ALIGNED + eleNumPerRepeat;

    int32_t k = 0;
    for (uint32_t i = 0; i < (maskSize / COMPLEX_ELENUM); i++) {
        maskData[k++] = (realOffset + i) * dtypeSize;
        maskData[k++] = (imagOffset + i) * dtypeSize;
    }

    return maskData;
}

