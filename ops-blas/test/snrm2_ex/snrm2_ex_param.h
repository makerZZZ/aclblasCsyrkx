/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SNRM2_EX_PARAM_H
#define SNRM2_EX_PARAM_H

#include <string>
#include <unordered_map>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

inline aclDataType parseXtype(const std::string& s)
{
    static const std::unordered_map<std::string, aclDataType> t = {
        {"ACL_FLOAT", ACL_FLOAT},
        {"ACL_FLOAT16", ACL_FLOAT16},
        {"FP32", ACL_FLOAT},
        {"FP16", ACL_FLOAT16},
        {"INVALID", static_cast<aclDataType>(0xFF)}};
    return parseEnum(s, t, ACL_FLOAT);
}

struct Snrm2ExParam : public BlasTestParamBase {
    aclDataType xtype = ACL_FLOAT;
    int64_t n = 0;
    int64_t incx = 1;
    BlasFillMode x = parseFill("RANDOM");

    Snrm2ExParam(const csv_map& map) : BlasTestParamBase(map)
    {
        xtype = parseXtype(ReadMap(map, "xtype", "ACL_FLOAT"));
        n = parseInt64(ReadMap(map, "n", "0"));
        incx = parseInt64(ReadMap(map, "incx", "1"));
        x = parseFill(ReadMap(map, "x", "RANDOM"));
    }
};

#endif // SNRM2_EX_PARAM_H
