// CPU-only negative-path check for the real aclblasCsyrkx host entry.
//
// Only parameter-validation paths are exercised (no device, no stream). These
// return before any ACL runtime call, so a fake non-null handle is safe here.
// Covers the CSV negative rows TC_ED_371..385 plus the n == 0 quick-return
// cases (validation order: enums/dims -> ld -> scalar pointers -> n == 0).

#include <cstdint>
#include <cstdio>

#include "cann_ops_blas.h"

namespace {
int gFails = 0;

void Expect(const char* name, aclblasStatus_t got, aclblasStatus_t want)
{
    const bool ok = (static_cast<int>(got) == static_cast<int>(want));
    std::printf("%-34s got=%d want=%d %s\n", name, static_cast<int>(got), static_cast<int>(want), ok ? "OK" : "FAIL");
    if (!ok) {
        gFails++;
    }
}
} // namespace

int main()
{
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    aclblasComplex dummy{0.0f, 0.0f};
    // Fake, non-null handle: never dereferenced on the paths below.
    aclblasHandle_t h = reinterpret_cast<aclblasHandle_t>(static_cast<uintptr_t>(1));
    const aclblasComplex* A = &dummy;
    const aclblasComplex* B = &dummy;
    aclblasComplex* C = &dummy;

    Expect("null handle", aclblasCsyrkx(nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, A, 4, B, 4, &beta, C, 4),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    Expect("invalid uplo", aclblasCsyrkx(h, static_cast<aclblasFillMode_t>(999), ACLBLAS_OP_N, 4, 4, &alpha, A, 4, B,
                                      4, &beta, C, 4),
        ACLBLAS_STATUS_INVALID_VALUE);
    Expect("invalid trans", aclblasCsyrkx(h, ACLBLAS_UPPER, static_cast<aclblasOperation_t>(999), 4, 4, &alpha, A, 4,
                                         B, 4, &beta, C, 4),
        ACLBLAS_STATUS_INVALID_VALUE);
    Expect("negative n", aclblasCsyrkx(h, ACLBLAS_UPPER, ACLBLAS_OP_N, -1, 16, &alpha, A, 1, B, 1, &beta, C, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
    Expect("negative k", aclblasCsyrkx(h, ACLBLAS_UPPER, ACLBLAS_OP_N, 16, -1, &alpha, A, 16, B, 16, &beta, C, 16),
        ACLBLAS_STATUS_INVALID_VALUE);
    Expect("lda < n (trans=N)", aclblasCsyrkx(h, ACLBLAS_UPPER, ACLBLAS_OP_N, 16, 16, &alpha, A, 8, B, 16, &beta, C, 16),
        ACLBLAS_STATUS_INVALID_VALUE);
    Expect("lda < k (trans=T)", aclblasCsyrkx(h, ACLBLAS_UPPER, ACLBLAS_OP_T, 16, 16, &alpha, A, 8, B, 16, &beta, C, 16),
        ACLBLAS_STATUS_INVALID_VALUE);
    Expect("ldb < n (trans=N)", aclblasCsyrkx(h, ACLBLAS_UPPER, ACLBLAS_OP_N, 16, 16, &alpha, A, 16, B, 8, &beta, C, 16),
        ACLBLAS_STATUS_INVALID_VALUE);
    Expect("ldc < n", aclblasCsyrkx(h, ACLBLAS_UPPER, ACLBLAS_OP_N, 16, 16, &alpha, A, 16, B, 16, &beta, C, 8),
        ACLBLAS_STATUS_INVALID_VALUE);
    Expect("null alpha", aclblasCsyrkx(h, ACLBLAS_UPPER, ACLBLAS_OP_N, 16, 16, nullptr, A, 16, B, 16, &beta, C, 16),
        ACLBLAS_STATUS_INVALID_VALUE);
    Expect("null beta", aclblasCsyrkx(h, ACLBLAS_UPPER, ACLBLAS_OP_N, 16, 16, &alpha, A, 16, B, 16, nullptr, C, 16),
        ACLBLAS_STATUS_INVALID_VALUE);

    // n == 0 quick return: all parameter checks run, no compute, no C access.
    Expect("n0 trans=N quick return",
        aclblasCsyrkx(h, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 16, &alpha, nullptr, 1, nullptr, 1, &beta, C, 1),
        ACLBLAS_STATUS_SUCCESS);
    Expect("n0 trans=T quick return",
        aclblasCsyrkx(h, ACLBLAS_LOWER, ACLBLAS_OP_T, 0, 16, &alpha, A, 16, B, 16, &beta, C, 1),
        ACLBLAS_STATUS_SUCCESS);
    Expect("n0 null C success",
        aclblasCsyrkx(h, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 16, &alpha, A, 1, B, 1, &alpha, nullptr, 1),
        ACLBLAS_STATUS_SUCCESS);
    Expect("n0 invalid ldc (checked first)",
        aclblasCsyrkx(h, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 16, &alpha, A, 1, B, 1, &beta, C, 0),
        ACLBLAS_STATUS_INVALID_VALUE);
    Expect("n0 null alpha (checked first)",
        aclblasCsyrkx(h, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 16, nullptr, A, 1, B, 1, &beta, C, 1),
        ACLBLAS_STATUS_INVALID_VALUE);

    std::printf("---- %s (%d failure(s)) ----\n", gFails == 0 ? "PASS" : "FAIL", gFails);
    return gFails == 0 ? 0 : 1;
}
