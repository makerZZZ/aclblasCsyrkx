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

#include <math.h>
#include "cann_ops_blas_common.h"

#ifdef __cplusplus

/* ── aclblasComplex ── */

inline aclblasComplex operator+(aclblasComplex a, aclblasComplex b) {
    return {a.real + b.real, a.imag + b.imag};
}
inline aclblasComplex operator-(aclblasComplex a, aclblasComplex b) {
    return {a.real - b.real, a.imag - b.imag};
}
inline aclblasComplex operator*(aclblasComplex a, aclblasComplex b) {
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}
inline aclblasComplex operator/(aclblasComplex a, aclblasComplex b) {
    float denom = b.real * b.real + b.imag * b.imag;
    if (denom == 0.0f) {
        return {NAN, NAN};
    }
    return {(a.real * b.real + a.imag * b.imag) / denom,
            (a.imag * b.real - a.real * b.imag) / denom};
}
inline bool operator==(aclblasComplex a, aclblasComplex b) {
    return a.real == b.real && a.imag == b.imag;
}
inline bool operator!=(aclblasComplex a, aclblasComplex b) { return !(a == b); }

inline aclblasComplex& operator+=(aclblasComplex& a, aclblasComplex b) {
    a.real += b.real; a.imag += b.imag; return a;
}
inline aclblasComplex& operator-=(aclblasComplex& a, aclblasComplex b) {
    a.real -= b.real; a.imag -= b.imag; return a;
}
inline aclblasComplex& operator*=(aclblasComplex& a, aclblasComplex b) {
    a = a * b; return a;
}
inline aclblasComplex& operator/=(aclblasComplex& a, aclblasComplex b) {
    float denom = b.real * b.real + b.imag * b.imag;
    if (denom == 0.0f) { a = {NAN, NAN}; return a; }
    float r = (a.real * b.real + a.imag * b.imag) / denom;
    float i = (a.imag * b.real - a.real * b.imag) / denom;
    a = {r, i}; return a;
}

inline aclblasComplex operator*(aclblasComplex a, float s) {
    return {a.real * s, a.imag * s};
}
inline aclblasComplex operator*(float s, aclblasComplex a) {
    return {s * a.real, s * a.imag};
}

inline float blasComplexAbs(aclblasComplex z) { return hypotf(z.real, z.imag); }

/* ── aclblasDoubleComplex ── */

inline aclblasDoubleComplex operator+(aclblasDoubleComplex a, aclblasDoubleComplex b) {
    return {a.real + b.real, a.imag + b.imag};
}
inline aclblasDoubleComplex operator-(aclblasDoubleComplex a, aclblasDoubleComplex b) {
    return {a.real - b.real, a.imag - b.imag};
}
inline aclblasDoubleComplex operator*(aclblasDoubleComplex a, aclblasDoubleComplex b) {
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}
inline aclblasDoubleComplex operator/(aclblasDoubleComplex a, aclblasDoubleComplex b) {
    double denom = b.real * b.real + b.imag * b.imag;
    if (denom == 0.0) {
        return {NAN, NAN};
    }
    return {(a.real * b.real + a.imag * b.imag) / denom,
            (a.imag * b.real - a.real * b.imag) / denom};
}
inline bool operator==(aclblasDoubleComplex a, aclblasDoubleComplex b) {
    return a.real == b.real && a.imag == b.imag;
}
inline bool operator!=(aclblasDoubleComplex a, aclblasDoubleComplex b) { return !(a == b); }

inline aclblasDoubleComplex& operator+=(aclblasDoubleComplex& a, aclblasDoubleComplex b) {
    a.real += b.real; a.imag += b.imag; return a;
}
inline aclblasDoubleComplex& operator-=(aclblasDoubleComplex& a, aclblasDoubleComplex b) {
    a.real -= b.real; a.imag -= b.imag; return a;
}
inline aclblasDoubleComplex& operator*=(aclblasDoubleComplex& a, aclblasDoubleComplex b) {
    a = a * b; return a;
}
inline aclblasDoubleComplex& operator/=(aclblasDoubleComplex& a, aclblasDoubleComplex b) {
    double denom = b.real * b.real + b.imag * b.imag;
    if (denom == 0.0) { a = {NAN, NAN}; return a; }
    double r = (a.real * b.real + a.imag * b.imag) / denom;
    double i = (a.imag * b.real - a.real * b.imag) / denom;
    a = {r, i}; return a;
}

inline aclblasDoubleComplex operator*(aclblasDoubleComplex a, double s) {
    return {a.real * s, a.imag * s};
}
inline aclblasDoubleComplex operator*(double s, aclblasDoubleComplex a) {
    return {s * a.real, s * a.imag};
}

inline double blasComplexAbs(aclblasDoubleComplex z) { return hypot(z.real, z.imag); }

#endif
