/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "csyrk_kernel.h"
#include "csyrk_reference_device.h"

using namespace AscendC;
constexpr uint32_t CSYRK_DIRECT_THREADS = 512;

__simt_callee__ __aicore__ inline CsyrkValue CsyrkDirectProduct(
    __gm__ float* a, uint32_t row, uint32_t col, uint32_t k, uint32_t lda, uint32_t transposed, uint32_t rowLanes,
    bool skip, bool checked, uint32_t& unsafe)
{
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t kLanes = 32 / rowLanes;
    float sumReal = 0.0f;
    float sumImag = 0.0f;
    float correctionReal = 0.0f;
    float correctionImag = 0.0f;

    if (!skip) {
        for (uint32_t l = lane / rowLanes; l < k; l += kLanes) {
            const uint64_t x =
                (transposed != 0 ? static_cast<uint64_t>(row) * lda + l : static_cast<uint64_t>(l) * lda + row) * 2;
            const uint64_t y =
                (transposed != 0 ? static_cast<uint64_t>(col) * lda + l : static_cast<uint64_t>(l) * lda + col) * 2;
            const float xr = a[x];
            const float xi = a[x + 1];
            const float yr = a[y];
            const float yi = a[y + 1];
            if (!checked) {
                unsafe |= CsyrkUnsafeValue(xr) | CsyrkUnsafeValue(xi) | CsyrkUnsafeValue(yr) | CsyrkUnsafeValue(yi);
            }
            const float vr = (xr * yr - xi * yi) - correctionReal;
            const float vi = (xr * yi + xi * yr) - correctionImag;
            const float nr = sumReal + vr;
            const float ni = sumImag + vi;
            correctionReal = (nr - sumReal) - vr;
            correctionImag = (ni - sumImag) - vi;
            sumReal = nr;
            sumImag = ni;
        }
    }
    for (uint32_t step = 16; step >= rowLanes; step /= 2) {
        unsafe |= asc_shfl_down(unsafe, step, 32);
        sumReal += asc_shfl_down(sumReal, step, 32);
        sumImag += asc_shfl_down(sumImag, step, 32);
    }
    return {sumReal, sumImag};
}

__simt_callee__ __aicore__ inline void CsyrkStoreDirect(
    __gm__ float* a, __gm__ float* c, uint32_t row, uint32_t col, uint32_t k, uint32_t lda, uint32_t ldc,
    uint32_t transposed, float alphaReal, float alphaImag, float betaReal, float betaImag, bool skip, bool zeroBeta,
    uint32_t unsafe, const CsyrkValue& sum)
{
    const uint64_t output = (static_cast<uint64_t>(col) * ldc + row) * 2;
    float real = skip ? 0.0f : alphaReal * sum.real - alphaImag * sum.imag;
    float imag = skip ? 0.0f : alphaReal * sum.imag + alphaImag * sum.real;
    if (!zeroBeta) {
        const float oldReal = c[output];
        const float oldImag = c[output + 1];
        real += betaReal * oldReal - betaImag * oldImag;
        imag += betaReal * oldImag + betaImag * oldReal;
    }
    if (unsafe || !__builtin_isfinite(real) || !__builtin_isfinite(imag)) {
        CsyrkReferenceElement(a, c, row, col, k, lda, ldc, transposed, alphaReal, alphaImag, betaReal, betaImag);
    } else {
        c[output] = real;
        c[output + 1] = imag;
    }
}

__simt_vf__ __aicore__ void CsyrkDirect(
    __gm__ float* alpha, __gm__ float* a, __gm__ float* beta, __gm__ float* c, uint32_t n, uint32_t k, uint32_t lda,
    uint32_t ldc, uint32_t transposed, uint32_t upper, uint32_t core, uint32_t cores, bool checked)
{
    const float alphaReal = alpha[0];
    const float alphaImag = alpha[1];
    const float betaReal = beta[0];
    const float betaImag = beta[1];
    const bool skip = k == 0 || (alphaReal == 0.0f && alphaImag == 0.0f);
    const bool zeroBeta = betaReal == 0.0f && betaImag == 0.0f;
    if (skip && betaReal == 1.0f && betaImag == 0.0f)
        return;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t warps = blockDim.x / 32;
    const uint32_t rowLanes = transposed == 0 && n >= 64 ? 4 : 1;
    const uint32_t rowGroups = (n + rowLanes - 1) / rowLanes;
    for (uint32_t index = core * warps + threadIdx.x / 32; index < rowGroups * n; index += cores * warps) {
        const uint32_t row = index % rowGroups * rowLanes + lane % rowLanes;
        const uint32_t col = index / rowGroups;
        if (row >= n || (upper != 0 && row > col) || (upper == 0 && row < col))
            continue;
        uint32_t unsafe = CsyrkUnsafeValue(alphaReal) | CsyrkUnsafeValue(alphaImag) | CsyrkUnsafeValue(betaReal) |
                          CsyrkUnsafeValue(betaImag);
        const auto sum = CsyrkDirectProduct(a, row, col, k, lda, transposed, rowLanes, skip, checked, unsafe);
        if (lane >= rowLanes)
            continue;
        CsyrkStoreDirect(
            a, c, row, col, k, lda, ldc, transposed, alphaReal, alphaImag, betaReal, betaImag, skip, zeroBeta, unsafe,
            sum);
    }
}

__simt_vf__ __aicore__ void CsyrkStrided(
    __gm__ float* alpha, __gm__ float* a, __gm__ float* beta, __gm__ float* c, uint32_t n, uint32_t k, uint32_t lda,
    uint32_t ldc, uint32_t transposed, uint32_t upper, uint32_t core, uint32_t cores, bool checked)
{
    (void)checked;
    const float ar = alpha[0];
    const float ai = alpha[1];
    const float br = beta[0];
    const float bi = beta[1];
    for (uint32_t col = core; col < n; col += cores) {
        const uint32_t begin = upper != 0 ? 0 : col;
        const uint32_t end = upper != 0 ? col + 1 : n;
        for (uint32_t row = begin + threadIdx.x; row < end; row += blockDim.x) {
            CsyrkReferenceElement(a, c, row, col, k, lda, ldc, transposed, ar, ai, br, bi);
        }
    }
}

extern "C" __global__ __aicore__ void csyrk_direct_kernel(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, CsyrkDirectTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tiling.strided != 0) {
        asc_vf_call<CsyrkStrided>(
            dim3{CSYRK_DIRECT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(alpha),
            reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(beta),
            reinterpret_cast<__gm__ float*>(c), tiling.n, tiling.k, tiling.lda, tiling.ldc, tiling.transposed,
            tiling.upper, GetBlockIdx(), GetBlockNum(), tiling.checked != 0);
    } else {
        asc_vf_call<CsyrkDirect>(
            dim3{CSYRK_DIRECT_THREADS, 1, 1}, reinterpret_cast<__gm__ float*>(alpha),
            reinterpret_cast<__gm__ float*>(a), reinterpret_cast<__gm__ float*>(beta),
            reinterpret_cast<__gm__ float*>(c), tiling.n, tiling.k, tiling.lda, tiling.ldc, tiling.transposed,
            tiling.upper, GetBlockIdx(), GetBlockNum(), tiling.checked != 0);
    }
}

void csyrk_direct_kernel_do(
    GM_ADDR alpha, GM_ADDR a, GM_ADDR beta, GM_ADDR c, const CsyrkDirectTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    csyrk_direct_kernel<<<numBlocks, nullptr, stream>>>(alpha, a, beta, c, tiling);
}
