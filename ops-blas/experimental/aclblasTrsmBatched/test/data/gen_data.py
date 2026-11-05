# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import os
import sys
from collections import namedtuple

import numpy as np
import scipy.linalg

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

SIDE_LEFT = 0
SIDE_RIGHT = 1
UPLO_UPPER = 0
UPLO_LOWER = 1
TRANS_N = 0
TRANS_T = 1
DIAG_NONUNIT = 0
DIAG_UNIT = 1

# 三角求解问题的标量参数（矩阵操作数 A/B 单独传入）
TrsmSpec = namedtuple("TrsmSpec", ["m", "n", "side", "uplo", "transa", "diag", "alpha"])


def compute_golden(matrix_a, matrix_b, spec):
    lower = (spec.uplo == UPLO_LOWER)
    unit_diagonal = (spec.diag == DIAG_UNIT)

    if spec.side == SIDE_LEFT:
        trans = 'T' if spec.transa == TRANS_T else 'N'
        matrix_x = scipy.linalg.solve_triangular(
            matrix_a, spec.alpha * matrix_b,
            lower=lower, trans=trans, unit_diagonal=unit_diagonal)
    else:
        trans = 'N' if spec.transa == TRANS_T else 'T'
        matrix_x = scipy.linalg.solve_triangular(
            matrix_a, (spec.alpha * matrix_b).T,
            lower=lower, trans=trans, unit_diagonal=unit_diagonal).T
    return matrix_x.astype(np.float32)


def gen_triangular_matrix(k_dim, uplo, diag):
    # Scale off-diagonals down for larger matrices so the triangular factor stays
    # well-conditioned. The condition number of a triangular matrix grows with size
    # (inv = I - N + N^2 - ...), and an fp32 solve on an ill-conditioned draw can
    # diverge from the fp32 LAPACK golden by more than the verify tolerance on a few
    # components. Keeping it well-conditioned makes the test measure correctness, not
    # fp32 rounding luck. (~0.3 for k_dim<=64, ~0.15 for 128.)
    off_scale = 0.3 * min(1.0, 64.0 / k_dim)
    matrix_a = (np.random.randn(k_dim, k_dim).astype(np.float32) * off_scale)
    if uplo == UPLO_UPPER:
        matrix_a = np.triu(matrix_a)
    else:
        matrix_a = np.tril(matrix_a)
    for i in range(k_dim):
        if diag == DIAG_UNIT:
            matrix_a[i, i] = 1.0
        else:
            matrix_a[i, i] = (np.random.uniform(1.5, 3.0)) * (1 if np.random.rand() > 0.5 else -1)
    return matrix_a


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    batch_count = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    side = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    uplo = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    transa = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    diag = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    alpha = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0

    np.random.seed(2026)
    k_dim = m if side == 0 else n

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    for batch in range(batch_count):
        matrix_a = gen_triangular_matrix(k_dim, uplo, diag)
        matrix_b = np.random.randn(m, n).astype(np.float32)

        matrix_a.tofile(f"input/input_a_{batch}.bin")
        matrix_b.tofile(f"input/input_b_{batch}.bin")

        golden = compute_golden(matrix_a, matrix_b,
                                TrsmSpec(m, n, side, uplo, transa, diag, alpha))
        golden.tofile(f"output/golden_b_{batch}.bin")

    logging.info(f"Generated {batch_count} batches: m={m}, n={n}, k_dim={k_dim}")
    logging.info(f"  side={side}, uplo={uplo}, transa={transa}, diag={diag}, alpha={alpha}")


if __name__ == "__main__":
    main()
