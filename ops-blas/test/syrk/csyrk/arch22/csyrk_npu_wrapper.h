/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CSYRK_NPU_WRAPPER_H
#define CSYRK_NPU_WRAPPER_H

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <thread>

#include "acl/acl.h"
#include "cann_ops_blas.h"

// Repeat-window target for performance measurement. Sizing the window by a
// probe keeps short shapes from being dominated by launch jitter while long
// shapes still finish in one window.
constexpr double kCsyrkPerfWindowMs = 30.0;
constexpr int kCsyrkPerfMaxIters = 2000;
// The Cube clock drops under sustained load, so an idle gap precedes the timed
// window to keep back-to-back cases from measuring a throttled frequency.
constexpr int kCsyrkPerfCooldownMs = 300;

struct CsyrkDeviceBuffers {
    void* dAlpha = nullptr;
    void* dA = nullptr;
    void* dBeta = nullptr;
    void* dC = nullptr;

    ~CsyrkDeviceBuffers()
    {
        if (dAlpha != nullptr) {
            aclrtFree(dAlpha);
        }
        if (dA != nullptr) {
            aclrtFree(dA);
        }
        if (dBeta != nullptr) {
            aclrtFree(dBeta);
        }
        if (dC != nullptr) {
            aclrtFree(dC);
        }
    }
};

static aclblasStatus_t CsyrkAllocAndCopy(void*& devPtr, const void* hostPtr, size_t bytes)
{
    aclError aclRet = aclrtMalloc(&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        devPtr = nullptr;
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    aclRet = aclrtMemcpy(devPtr, bytes, hostPtr, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

// Allocate+copy only when hostPtr is non-null and bytes > 0; leaves devPtr
// null otherwise so the caller can pass nullptr straight through to the API
// (this is how the null_alpha / nullA / nullC cases reach the operator).
static aclblasStatus_t CsyrkTryAllocAndCopy(const void* hostPtr, size_t bytes, void*& devPtr)
{
    if (hostPtr == nullptr || bytes == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    return CsyrkAllocAndCopy(devPtr, hostPtr, bytes);
}

// NPU wrapper for aclblasCsyrk. Same shape as the CHERK wrapper, with two
// differences: alpha and beta are *complex* (so the device scalars are 8 bytes
// each), and a nullBeta flag exists because beta is a Device pointer too.
// When handle is null or n<=0 it forwards directly so the operator's own
// validation paths are exercised.
//
// perfMode selects how opMs is obtained: false measures a single call, which is
// what an accuracy case needs because C must hold exactly one operator result;
// true runs one warm-up call and then reports the average over a repeat window,
// leaving C holding the last iteration, so a perfMode call must not be
// precision-checked. The warm-up matters because a first call on a shape also
// pays workspace growth and kernel load.
//
// opMs, when non-null, receives the wall time of the operator call plus device
// synchronisation. Host buffer allocation and H2D/D2H transfers are excluded,
// so it is the cost a caller pays for the operator itself on already-resident
// device data — the quantity the performance baseline is stated against.
// Stages alpha / A / beta / C onto the device. A null host pointer leaves the
// matching device pointer null, which the caller forwards as-is.
static aclblasStatus_t CsyrkUploadInputs(
    const aclblasComplex* effAlpha, const aclblasComplex* A, const aclblasComplex* effBeta, const aclblasComplex* C,
    size_t scalarBytes, size_t aBytes, size_t cBytes, CsyrkDeviceBuffers& bufs)
{
    aclblasStatus_t st = CsyrkTryAllocAndCopy(effAlpha, scalarBytes, bufs.dAlpha);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = CsyrkTryAllocAndCopy(A, aBytes, bufs.dA);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    st = CsyrkTryAllocAndCopy(effBeta, scalarBytes, bufs.dBeta);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    return CsyrkTryAllocAndCopy(C, cBytes, bufs.dC);
}

// Times a repeat window: one probe call sizes the window, then the average over
// `iters` back-to-back calls is reported. A cooldown precedes both so a throttled
// frequency from the previous case does not leak into this measurement.
template <typename RunFn, typename SyncFn, typename ElapsedFn>
static aclblasStatus_t CsyrkMeasurePerfWindow(
    RunFn&& runOnce, SyncFn&& syncDevice, ElapsedFn&& elapsedMs, double& measuredMs, int& iters)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(kCsyrkPerfCooldownMs));
    const auto probeStart = std::chrono::steady_clock::now();
    aclblasStatus_t ret = runOnce();
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    if (!syncDevice()) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const double probeMs = elapsedMs(probeStart);
    iters =
        (probeMs > 0.0) ? std::max(1, std::min(static_cast<int>(kCsyrkPerfWindowMs / probeMs), kCsyrkPerfMaxIters)) : 1;

    std::this_thread::sleep_for(std::chrono::milliseconds(kCsyrkPerfCooldownMs));
    const auto windowStart = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        ret = runOnce();
        if (ret != ACLBLAS_STATUS_SUCCESS) {
            return ret;
        }
    }
    if (!syncDevice()) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    // std::max above already floors iters at 1; the floor is restated here because
    // the static analyser cannot see through max/min and flags the division.
    if (iters < 1) {
        iters = 1;
    }
    measuredMs = elapsedMs(windowStart) / iters;
    return ACLBLAS_STATUS_SUCCESS;
}

// One warm-up call (always) plus, in perfMode, the averaged repeat window.
template <typename RunFn>
static aclblasStatus_t CsyrkRunAndTime(RunFn&& runOnce, bool perfMode, double& measuredMs, int& iters)
{
    const auto syncDevice = []() { return aclrtSynchronizeDevice() == ACL_SUCCESS; };
    const auto elapsedMs = [](std::chrono::steady_clock::time_point since) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
    };

    const auto opStart = std::chrono::steady_clock::now();
    const aclblasStatus_t ret = runOnce();
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }
    if (!syncDevice()) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    measuredMs = elapsedMs(opStart);
    iters = 1;

    if (perfMode) {
        return CsyrkMeasurePerfWindow(runOnce, syncDevice, elapsedMs, measuredMs, iters);
    }
    return ret;
}

// Copies C back to the host. A null host or device pointer means the case never
// staged C, so there is nothing to retrieve.
static aclblasStatus_t CsyrkDownloadResult(aclblasComplex* C, const void* dC, size_t cBytes)
{
    if (C == nullptr || dC == nullptr) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclError aclRet = aclrtMemcpy(C, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    return (aclRet == ACL_SUCCESS) ? ACLBLAS_STATUS_SUCCESS : ACLBLAS_STATUS_INTERNAL_ERROR;
}

inline aclblasStatus_t aclblasCsyrk_npu(
    aclblasHandle handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* beta, aclblasComplex* C, int ldc, bool nullAlpha = false,
    bool nullBeta = false, double* opMs = nullptr, bool perfMode = false, int* perfIters = nullptr)
{
    if (handle == nullptr || n <= 0) {
        return aclblasCsyrk(
            handle, uplo, trans, n, k, (nullAlpha ? nullptr : alpha), A, lda, (nullBeta ? nullptr : beta), C, ldc);
    }

    const int aCols = (trans == ACLBLAS_OP_N) ? k : n;
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(aCols) * sizeof(aclblasComplex);
    const size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);
    constexpr size_t scalarBytes = sizeof(aclblasComplex);

    // When nullAlpha / nullBeta are set, force the corresponding pointer seen by
    // the API to null so its INVALID_VALUE path is exercised.
    const aclblasComplex* effAlpha = nullAlpha ? nullptr : alpha;
    const aclblasComplex* effBeta = nullBeta ? nullptr : beta;

    CsyrkDeviceBuffers bufs;
    const aclblasStatus_t upRet = CsyrkUploadInputs(effAlpha, A, effBeta, C, scalarBytes, aBytes, cBytes, bufs);
    if (upRet != ACLBLAS_STATUS_SUCCESS) {
        return upRet;
    }

    const auto runOnce = [&]() {
        return aclblasCsyrk(
            handle, uplo, trans, n, k, bufs.dAlpha ? static_cast<const aclblasComplex*>(bufs.dAlpha) : effAlpha,
            bufs.dA ? static_cast<const aclblasComplex*>(bufs.dA) : A, lda,
            bufs.dBeta ? static_cast<const aclblasComplex*>(bufs.dBeta) : effBeta,
            bufs.dC ? static_cast<aclblasComplex*>(bufs.dC) : C, ldc);
    };
    double measuredMs = 0.0;
    int iters = 1;
    const aclblasStatus_t ret = CsyrkRunAndTime(runOnce, perfMode, measuredMs, iters);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return ret;
    }

    if (opMs != nullptr) {
        *opMs = measuredMs;
    }
    if (perfIters != nullptr) {
        *perfIters = iters;
    }

    const aclblasStatus_t downRet = CsyrkDownloadResult(C, bufs.dC, cBytes);
    if (downRet != ACLBLAS_STATUS_SUCCESS) {
        return downRet;
    }
    return ret;
}

#endif // CSYRK_NPU_WRAPPER_H
