#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
CtrsmBatched 测试数据生成脚本。
生成输入矩阵 A/B 和 golden 参考结果 X。

使用方式:
    python gen_data.py <m> <n> <batch> <side> <uplo> <transa> <diag> [alpha_re] [alpha_im]
"""
import sys
import os
import logging
from collections import namedtuple
import numpy as np
import scipy.linalg

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

np.random.seed(2026)

# 三角求解问题的标量参数（矩阵操作数 A/B 单独传入）
TrsmSpec = namedtuple("TrsmSpec", ["m", "n", "side", "uplo", "transa", "diag", "alpha"])


def gen_complex_tri_matrix(k, uplo, diag, rng):
    off_scale = 0.3 * min(1.0, 64.0 / k)
    a_real = rng.randn(k, k).astype(np.float32) * off_scale
    a_imag = rng.randn(k, k).astype(np.float32) * off_scale
    matrix_a = (a_real + 1j * a_imag).astype(np.complex64)
    matrix_a = np.triu(matrix_a) if uplo == 0 else np.tril(matrix_a)
    for i in range(k):
        if diag == 1:
            matrix_a[i, i] = 1.0 + 0j
        else:
            row_sum = np.sum(np.abs(matrix_a[i, :])) - np.abs(matrix_a[i, i])
            diag_mag = max(row_sum + rng.uniform(1.0, 3.0), 1.5)
            sign = 1.0 if rng.rand() > 0.5 else -1.0
            matrix_a[i, i] = complex(diag_mag * sign, rng.uniform(-0.3, 0.3))
    return matrix_a.astype(np.complex64)


def compute_golden(matrix_a, matrix_b, spec):
    trans_map = {0: 'N', 1: 'T', 2: 'C'}
    if spec.transa not in trans_map:
        raise ValueError(f"invalid transa={spec.transa}, expected one of {sorted(trans_map)}")
    t = trans_map[spec.transa]
    lower = (spec.uplo == 1)
    unit = (spec.diag == 1)
    alpha = complex(spec.alpha)
    a_f64 = matrix_a.astype(np.complex128)
    b_f64 = matrix_b.astype(np.complex128)
    if spec.side == 0:
        matrix_x = scipy.linalg.solve_triangular(a_f64, alpha * b_f64, lower=lower, unit_diagonal=unit, trans=t)
    else:
        if t == 'N':
            matrix_x = scipy.linalg.solve_triangular(
                a_f64, (alpha * b_f64).T,
                lower=lower, unit_diagonal=unit, trans='T').T
        elif t == 'T':
            matrix_x = scipy.linalg.solve_triangular(
                a_f64, (alpha * b_f64).T,
                lower=lower, unit_diagonal=unit, trans='N').T
        else:
            matrix_x = scipy.linalg.solve_triangular(
                a_f64.conj(), (alpha * b_f64).T,
                lower=lower, unit_diagonal=unit, trans='N').T
    return matrix_x.astype(np.complex64)


def parse_args():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    side = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    uplo = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    transa = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    diag = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    alpha_re = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0
    alpha_im = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
    spec = TrsmSpec(m, n, side, uplo, transa, diag, complex(alpha_re, alpha_im))
    return spec, batch


def gen_one_batch(b, spec, k, rng):
    matrix_a = gen_complex_tri_matrix(k, spec.uplo, spec.diag, rng)
    matrix_b = (rng.randn(spec.m, spec.n).astype(np.float32) +
         1j * rng.randn(spec.m, spec.n).astype(np.float32)).astype(np.complex64)
    golden = compute_golden(matrix_a, matrix_b, spec)

    matrix_a.view(np.float32).tofile(f"./test/data/input/input_a_{b}.bin")
    matrix_b.view(np.float32).tofile(f"./test/data/input/input_b_{b}.bin")
    golden = np.ascontiguousarray(golden)
    golden.view(np.float32).tofile(f"./test/data/output/golden_{b}.bin")


def main():
    spec, batch = parse_args()

    rng = np.random.RandomState(2026)
    k = spec.m if spec.side == 0 else spec.n

    os.makedirs("./test/data/input", exist_ok=True)
    os.makedirs("./test/data/output", exist_ok=True)

    for b in range(batch):
        gen_one_batch(b, spec, k, rng)

    logging.info(f"Generated {batch} batches: m={spec.m}, n={spec.n}, k={k}")


if __name__ == "__main__":
    main()
