// CPU-only pre-verification harness for aclblasCsyrkx (no NPU required).
//
// For every CSV case it checks:
//   1. the CPU golden (aclblasCsyrkx_cpu, cblas_cgemm based) agrees with an
//      independent double-precision reference implementation;
//   2. a float simulation of the operator's kernel math (deinterleave + 4M
//      GEMM products + complex combine, i.e. P_r=t1-t2, P_i=t3+t4 and the
//      complex alpha/beta cross-scaling) agrees with the golden within the
//      task-doc tolerance;
//   3. return codes match expect_result, and the non-uplo triangle is never
//      modified.
//
// Build (see local_verify/README.md) and run:
//   ./csyrkx_cpu_check ../test_cases/csyrkx_test.csv

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "csyrkx_param.h"
#include "csyrkx_golden.h"
#include "fill.h"
#include "csv_loader.h"

namespace {

constexpr double kAtol = 1.220703125e-4; // 2^-13
constexpr double kRtol = 1.220703125e-4; // 2^-13
constexpr double kFixedLimit = 1e-2;
constexpr double kRequiredRatio = 0.99;

bool IsNonFinite(float v) { return std::isnan(v) || std::isinf(v); }

bool NonFiniteEquivalent(float a, float g)
{
    if (std::isnan(a) && std::isnan(g)) {
        return true;
    }
    if (std::isinf(a) && std::isinf(g)) {
        return a == g;
    }
    return (std::isnan(a) && std::isinf(g)) || (std::isinf(a) && std::isnan(g));
}

double UlpAt(double magnitude, int mantissaBits = 23, int emin = -126)
{
    const double absMag = std::abs(magnitude);
    if (absMag == 0.0) {
        return std::ldexp(1.0, emin - mantissaBits);
    }
    int exp = 0;
    std::frexp(absMag, &exp);
    return std::ldexp(1.0, std::max(exp - 1, emin) - mantissaBits);
}

struct CompareResult {
    size_t total = 0;
    size_t finiteFail = 0;
    size_t nonFinite = 0;
    size_t nonFiniteFail = 0;
    double maxAbsErr = 0.0;
    bool maxAbsErrExceeded = false;
};

CompareResult CompareComponents(const std::vector<float>& actual, const std::vector<float>& golden)
{
    CompareResult r;
    for (size_t i = 0; i < actual.size(); i++) {
        const float a = actual[i];
        const float g = golden[i];
        r.total++;
        if (IsNonFinite(a) || IsNonFinite(g)) {
            r.nonFinite++;
            if (!NonFiniteEquivalent(a, g)) {
                r.nonFiniteFail++;
            }
            continue;
        }
        const double diff = std::abs(static_cast<double>(a) - static_cast<double>(g));
        r.maxAbsErr = std::max(r.maxAbsErr, diff);
        if (diff > kAtol + kRtol * std::abs(static_cast<double>(g))) {
            r.finiteFail++;
        }
        if (diff > std::max(kFixedLimit, 32.0 * UlpAt(g))) {
            r.maxAbsErrExceeded = true;
        }
    }
    return r;
}

bool CompareOk(const CompareResult& r)
{
    const size_t finiteTotal = r.total - r.nonFinite;
    const double ratio = finiteTotal == 0 ? 1.0 : 1.0 - static_cast<double>(r.finiteFail) / finiteTotal;
    return ratio >= kRequiredRatio && !r.maxAbsErrExceeded && r.nonFiniteFail == 0;
}

// NaN-aware bit comparison (used for the non-uplo triangle, which must keep
// the input-old values including NaN/Inf payloads).
bool SameValues(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i]))) {
            continue;
        }
        return false;
    }
    return true;
}

// EXTREME / Inf / NaN fills make the float golden and a double-precision
// reference legitimately disagree (intermediate float overflow vs double
// cancellation), so the reference cross-check is skipped for those inputs.
bool HasUnstableInput(const std::vector<aclblasComplex>& v)
{
    for (const auto& x : v) {
        const float parts[2] = {x.real, x.imag};
        for (float p : parts) {
            if (IsNonFinite(p) || std::abs(static_cast<double>(p)) > 1e30) {
                return true;
            }
        }
    }
    return false;
}

inline size_t CplxIndex(int i, int j, int ld) { return static_cast<size_t>(i) + static_cast<size_t>(j) * ld; }

// Independent double-precision reference (upholds the task-doc formula).
void ReferenceDouble(
    const CsyrkxParam& p, const std::vector<aclblasComplex>& A, const std::vector<aclblasComplex>& B,
    const std::vector<aclblasComplex>& cOld, std::vector<aclblasComplex>& out)
{
    out = cOld;
    const bool alphaZero = (p.alphaReal == 0.0f && p.alphaImag == 0.0f);
    const bool betaZero = (p.betaReal == 0.0f && p.betaImag == 0.0f);
    for (int j = 0; j < p.n; j++) {
        for (int i = 0; i < p.n; i++) {
            const bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (!isUplo) {
                continue;
            }
            double prodRe = 0.0;
            double prodIm = 0.0;
            if (!alphaZero && p.k > 0) {
                for (int t = 0; t < p.k; t++) {
                    double ar, ai, br, bi;
                    if (p.trans == ACLBLAS_OP_N) {
                        const aclblasComplex av = A[CplxIndex(i, t, p.lda)];
                        const aclblasComplex bv = B[CplxIndex(j, t, p.ldb)];
                        ar = av.real; ai = av.imag; br = bv.real; bi = bv.imag;
                    } else {
                        const aclblasComplex av = A[CplxIndex(t, i, p.lda)];
                        const aclblasComplex bv = B[CplxIndex(t, j, p.ldb)];
                        ar = av.real; ai = av.imag; br = bv.real; bi = bv.imag;
                    }
                    // (ar + i ai) * (br + i bi)
                    prodRe += ar * br - ai * bi;
                    prodIm += ar * bi + ai * br;
                }
            }
            const size_t idx = CplxIndex(i, j, p.ldc);
            const double orv = cOld[idx].real;
            const double oiv = cOld[idx].imag;
            double re = p.alphaReal * prodRe - p.alphaImag * prodIm;
            double im = p.alphaReal * prodIm + p.alphaImag * prodRe;
            if (!betaZero) {
                re += p.betaReal * orv - p.betaImag * oiv;
                im += p.betaReal * oiv + p.betaImag * orv;
            }
            out[idx] = aclblasComplex{static_cast<float>(re), static_cast<float>(im)};
        }
    }
}

// Float simulation of the kernel path for large shapes: the four real GEMM
// products are evaluated with cblas_sgemm (blocked, like the NPU cube unit),
// then the complex combine formulas are applied element-wise.
void KernelModelFast(
    const CsyrkxParam& p, const std::vector<aclblasComplex>& A, const std::vector<aclblasComplex>& B,
    const std::vector<aclblasComplex>& cOld, std::vector<aclblasComplex>& out)
{
    out = cOld;
    const int n = p.n;
    const int k = p.k;
    const int physRows = (p.trans == ACLBLAS_OP_N) ? n : k;
    const size_t realElems = static_cast<size_t>(physRows) * (p.trans == ACLBLAS_OP_N ? k : n);
    std::vector<float> ar(realElems), ai(realElems), br(realElems), bi(realElems);
    const int physCols = (p.trans == ACLBLAS_OP_N) ? k : n;
    for (int j = 0; j < physCols; j++) {
        for (int i = 0; i < physRows; i++) {
            const size_t dst = static_cast<size_t>(i) + static_cast<size_t>(j) * physRows;
            const size_t src = CplxIndex(i, j, p.lda);
            ar[dst] = A[src].real;
            ai[dst] = A[src].imag;
            const size_t srcB = CplxIndex(i, j, p.ldb);
            br[dst] = B[srcB].real;
            bi[dst] = B[srcB].imag;
        }
    }
    std::vector<float> t1(static_cast<size_t>(n) * n), t2(t1.size()), t3(t1.size()), t4(t1.size());
    const CBLAS_TRANSPOSE opA = (p.trans == ACLBLAS_OP_N) ? CblasNoTrans : CblasTrans;
    const CBLAS_TRANSPOSE opB = (p.trans == ACLBLAS_OP_N) ? CblasTrans : CblasNoTrans;
    cblas_sgemm(CblasColMajor, opA, opB, n, n, k, 1.0f, ar.data(), physRows, br.data(), physRows, 0.0f, t1.data(), n);
    cblas_sgemm(CblasColMajor, opA, opB, n, n, k, 1.0f, ai.data(), physRows, bi.data(), physRows, 0.0f, t2.data(), n);
    cblas_sgemm(CblasColMajor, opA, opB, n, n, k, 1.0f, ar.data(), physRows, bi.data(), physRows, 0.0f, t3.data(), n);
    cblas_sgemm(CblasColMajor, opA, opB, n, n, k, 1.0f, ai.data(), physRows, br.data(), physRows, 0.0f, t4.data(), n);

    const bool betaZero = (p.betaReal == 0.0f && p.betaImag == 0.0f);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            const bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (!isUplo) {
                continue;
            }
            const size_t idx = CplxIndex(i, j, p.ldc);
            const size_t tidx = static_cast<size_t>(i) + static_cast<size_t>(j) * n;
            const float pr = t1[tidx] - t2[tidx];
            const float pi = t3[tidx] + t4[tidx];
            float cr = p.alphaReal * pr - p.alphaImag * pi;
            float ci = p.alphaReal * pi + p.alphaImag * pr;
            if (!betaZero) {
                const float orv = cOld[idx].real;
                const float oiv = cOld[idx].imag;
                cr += p.betaReal * orv;
                cr -= p.betaImag * oiv;
                ci += p.betaReal * oiv;
                ci += p.betaImag * orv;
            }
            out[idx] = aclblasComplex{cr, ci};
        }
    }
}

// Float simulation of the kernel path: deinterleaved 4M products + combine.
void KernelModel(
    const CsyrkxParam& p, const std::vector<aclblasComplex>& A, const std::vector<aclblasComplex>& B,
    const std::vector<aclblasComplex>& cOld, std::vector<aclblasComplex>& out)
{
    out = cOld;
    const uint32_t n = static_cast<uint32_t>(p.n);
    const uint32_t k = static_cast<uint32_t>(p.k);
    const bool alphaZero = (p.alphaReal == 0.0f && p.alphaImag == 0.0f);
    const bool kZero = (p.k == 0);
    const bool betaZero = (p.betaReal == 0.0f && p.betaImag == 0.0f);

    for (uint32_t j = 0; j < n; j++) {
        for (uint32_t i = 0; i < n; i++) {
            const bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (!isUplo) {
                continue;
            }
            const size_t idx = CplxIndex(static_cast<int>(i), static_cast<int>(j), p.ldc);
            float cr = 0.0f;
            float ci = 0.0f;
            if (!alphaZero && !kZero) {
                float t1 = 0.0f;
                float t2 = 0.0f;
                float t3 = 0.0f;
                float t4 = 0.0f;
                for (uint32_t t = 0; t < k; t++) {
                    float ar, ai, br, bi;
                    if (p.trans == ACLBLAS_OP_N) {
                        ar = A[CplxIndex(static_cast<int>(i), static_cast<int>(t), p.lda)].real;
                        ai = A[CplxIndex(static_cast<int>(i), static_cast<int>(t), p.lda)].imag;
                        br = B[CplxIndex(static_cast<int>(j), static_cast<int>(t), p.ldb)].real;
                        bi = B[CplxIndex(static_cast<int>(j), static_cast<int>(t), p.ldb)].imag;
                    } else {
                        ar = A[CplxIndex(static_cast<int>(t), static_cast<int>(i), p.lda)].real;
                        ai = A[CplxIndex(static_cast<int>(t), static_cast<int>(i), p.lda)].imag;
                        br = B[CplxIndex(static_cast<int>(t), static_cast<int>(j), p.ldb)].real;
                        bi = B[CplxIndex(static_cast<int>(t), static_cast<int>(j), p.ldb)].imag;
                    }
                    t1 += ar * br;
                    t2 += ai * bi;
                    t3 += ar * bi;
                    t4 += ai * br;
                }
                const float pr = t1 - t2;
                const float pi = t3 + t4;
                cr = p.alphaReal * pr - p.alphaImag * pi;
                ci = p.alphaReal * pi + p.alphaImag * pr;
            }
            if (!betaZero) {
                const float orv = cOld[idx].real;
                const float oiv = cOld[idx].imag;
                cr += p.betaReal * orv;
                cr -= p.betaImag * oiv;
                ci += p.betaReal * oiv;
                ci += p.betaImag * orv;
            }
            out[idx] = aclblasComplex{cr, ci};
        }
    }
}

bool CheckModelAgainstGolden(
    const CsyrkxParam& p, const std::vector<aclblasComplex>& model, const std::vector<aclblasComplex>& golden,
    const std::vector<aclblasComplex>& cOld, const char* tag)
{
    std::vector<float> modelUploRe, modelUploIm, gRe, gIm;
    std::vector<float> modelNon, oldNon;
    for (int j = 0; j < p.n; j++) {
        for (int i = 0; i < p.n; i++) {
            const size_t idx = CplxIndex(i, j, p.ldc);
            const bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo) {
                modelUploRe.push_back(model[idx].real);
                modelUploIm.push_back(model[idx].imag);
                gRe.push_back(golden[idx].real);
                gIm.push_back(golden[idx].imag);
            } else {
                modelNon.push_back(model[idx].real);
                modelNon.push_back(model[idx].imag);
                oldNon.push_back(cOld[idx].real);
                oldNon.push_back(cOld[idx].imag);
            }
        }
    }
    const CompareResult mRe = CompareComponents(modelUploRe, gRe);
    const CompareResult mIm = CompareComponents(modelUploIm, gIm);
    if (!CompareOk(mRe) || !CompareOk(mIm) || !SameValues(modelNon, oldNon)) {
        std::cout << "[" << tag << "] " << p.caseName << " model-vs-golden mismatch"
                  << " reFail=" << mRe.finiteFail << "/" << (mRe.total - mRe.nonFinite)
                  << " imFail=" << mIm.finiteFail << "/" << (mIm.total - mIm.nonFinite)
                  << " maxAbsRe=" << mRe.maxAbsErr << " maxAbsIm=" << mIm.maxAbsErr
                  << " nonUploOk=" << SameValues(modelNon, oldNon) << std::endl;
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <csyrkx_test.csv>" << std::endl;
        return 2;
    }
    std::vector<CsyrkxParam> cases;
    try {
        cases = GetCasesFromCsv<CsyrkxParam>(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "failed to load CSV: " << e.what() << std::endl;
        return 2;
    }
    std::cout << "loaded " << cases.size() << " cases from " << argv[1] << std::endl;

    size_t retFail = 0;
    size_t goldenVsRefFail = 0;
    size_t modelVsGoldenFail = 0;
    size_t nonUploFail = 0;
    size_t checked = 0;
    size_t checkedFast = 0;
    size_t skippedBig = 0;
    // Cap the O(n^2*k) references; larger perf cases are covered by the
    // performance harness rather than this CPU cross-check.
    constexpr uint64_t kMaxReferenceWork = 60000000ULL;

    for (const auto& p : cases) {
        const aclblasComplex alpha{p.alphaReal, p.alphaImag};
        const aclblasComplex beta{p.betaReal, p.betaImag};
        const aclblasComplex* alphaPtr = p.nullAlpha ? nullptr : &alpha;
        const aclblasComplex* betaPtr = p.nullBeta ? nullptr : &beta;

        const int abCols = (p.k > 0) ? ((p.trans == ACLBLAS_OP_N) ? p.k : p.n) : 0;
        std::vector<aclblasComplex> A;
        std::vector<aclblasComplex> B;
        std::vector<aclblasComplex> C;
        if (p.n > 0 && p.ldc > 0) {
            C = p.nullC ? std::vector<aclblasComplex>{}
                        : makeBlasComplexMatrix(p.n, p.n, p.ldc, p.cFill, p.randomSeed + 2U);
        }
        if (p.n > 0 && p.k > 0 && abCols > 0 && p.lda > 0 && p.ldb > 0) {
            A = p.nullA ? std::vector<aclblasComplex>{} : makeBlasComplexMatrix(p.lda, abCols, p.lda, p.aFill, p.randomSeed);
            B = p.nullB ? std::vector<aclblasComplex>{} : makeBlasComplexMatrix(p.ldb, abCols, p.ldb, p.bFill, p.randomSeed + 1U);
        }
        const aclblasComplex* aPtr = A.empty() ? nullptr : A.data();
        const aclblasComplex* bPtr = B.empty() ? nullptr : B.data();
        std::vector<aclblasComplex> golden = C;
        aclblasComplex* cPtr = C.empty() ? nullptr : golden.data();

        const aclblasStatus_t goldenRet = aclblasCsyrkx_cpu(
            reinterpret_cast<aclblasHandle>(static_cast<uintptr_t>(1)), p.uplo, p.trans, p.n, p.k, alphaPtr, aPtr,
            p.lda, bPtr, p.ldb,
            betaPtr, cPtr, p.ldc, C.empty() ? nullptr : C.data());

        if (goldenRet != p.expectResult) {
            std::cout << "[RET ] " << p.caseName << " expected=" << static_cast<int>(p.expectResult)
                      << " golden=" << static_cast<int>(goldenRet) << std::endl;
            retFail++;
            continue;
        }
        if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0 || cPtr == nullptr) {
            continue;
        }

        const uint64_t work = static_cast<uint64_t>(p.n) * p.n * static_cast<uint64_t>(p.k);
        if (work > kMaxReferenceWork) {
            // Large shapes: validate the combine formulas with a blocked
            // cblas_sgemm-based model; the very largest perf cases are covered
            // by the performance harness instead.
            constexpr uint64_t kFastModelWorkCap = 20000000000ULL;
            if (work <= kFastModelWorkCap && !HasUnstableInput(A) && !HasUnstableInput(B) && !HasUnstableInput(C)) {
                std::vector<aclblasComplex> fastModel;
                KernelModelFast(p, A, B, C, fastModel);
                checkedFast++;
                if (!CheckModelAgainstGolden(p, fastModel, golden, C, "FST")) {
                    modelVsGoldenFail++;
                }
            } else {
                skippedBig++;
            }
            continue;
        }
        checked++;

        // (a) golden vs independent double reference
        const bool unstableInput = HasUnstableInput(A) || HasUnstableInput(B) || HasUnstableInput(C);
        if (!unstableInput) {
            std::vector<aclblasComplex> ref;
            ReferenceDouble(p, A, B, C, ref);
            std::vector<float> goldenUploRe, goldenUploIm, refUploRe, refUploIm;
            std::vector<float> goldenNon, refNon;
            for (int j = 0; j < p.n; j++) {
                for (int i = 0; i < p.n; i++) {
                    const size_t idx = CplxIndex(i, j, p.ldc);
                    const bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
                    if (isUplo) {
                        goldenUploRe.push_back(golden[idx].real);
                        goldenUploIm.push_back(golden[idx].imag);
                        refUploRe.push_back(ref[idx].real);
                        refUploIm.push_back(ref[idx].imag);
                    } else {
                        goldenNon.push_back(golden[idx].real);
                        goldenNon.push_back(golden[idx].imag);
                        refNon.push_back(C[idx].real);
                        refNon.push_back(C[idx].imag);
                    }
                }
            }
            const CompareResult rRe = CompareComponents(goldenUploRe, refUploRe);
            const CompareResult rIm = CompareComponents(goldenUploIm, refUploIm);
            const bool nonUploOk = SameValues(goldenNon, refNon);
            if (!CompareOk(rRe) || !CompareOk(rIm) || !nonUploOk) {
                std::cout << "[GOLD] " << p.caseName << " golden-vs-reference mismatch"
                          << " reFail=" << rRe.finiteFail << " imFail=" << rIm.finiteFail
                          << " nonFinReFail=" << rRe.nonFiniteFail << " nonFinImFail=" << rIm.nonFiniteFail
                          << " maxAbsRe=" << rRe.maxAbsErr << " maxAbsIm=" << rIm.maxAbsErr
                          << " nonUploOk=" << nonUploOk << std::endl;
                goldenVsRefFail++;
                if (!nonUploOk) {
                    nonUploFail++;
                }
            }
        }

        // (b) kernel float model vs golden
        std::vector<aclblasComplex> model;
        KernelModel(p, A, B, C, model);
        if (!CheckModelAgainstGolden(p, model, golden, C, "MODL")) {
            modelVsGoldenFail++;
        }
    }

    std::cout << "---------------- summary ----------------" << std::endl;
    std::cout << "cases=" << cases.size() << " numerically-checked=" << checked
              << " fast-model-checked=" << checkedFast << " skipped-large=" << skippedBig << std::endl;
    std::cout << "return-code mismatches : " << retFail << std::endl;
    std::cout << "golden vs reference    : " << goldenVsRefFail << std::endl;
    std::cout << "kernel model vs golden : " << modelVsGoldenFail << std::endl;
    const bool ok = (retFail == 0 && goldenVsRefFail == 0 && modelVsGoldenFail == 0);
    std::cout << (ok ? "RESULT: PASS" : "RESULT: FAIL") << std::endl;
    return ok ? 0 : 1;
}
