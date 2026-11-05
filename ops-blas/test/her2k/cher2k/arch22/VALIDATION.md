<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Cher2k arch22 validation

Source the installed CANN `set_env.sh` and build `cher2k_test` for the installed
910B SOC, with `BUILD_TEST=ON` and `TEST_NAMES=cher2k`. The golden implementation
requires OpenBLAS/CBLAS headers and libraries. Pass `REFBLAS_INCLUDE_DIR` and
`REFBLAS_LIB` to CMake if the system installs them outside its search paths.

Run from the repository root, replacing the absolute binary and report paths:

```bash
bash test/her2k/cher2k/arch22/run_validation.sh /build/test/her2k/cher2k/cher2k_test /reports/cher2k full
bash test/her2k/cher2k/arch22/run_validation.sh /build/test/her2k/cher2k/cher2k_test /reports/cher2k guards
bash test/her2k/cher2k/arch22/run_validation.sh /build/test/her2k/cher2k/cher2k_test /reports/cher2k perf
```

`full` checks all 1214 CSV cases against the CPU reference, including all 200
performance shapes, and the five standalone tests. It clears precision-skipping
and benchmark environment settings. The CSV includes invalid enums, dimensions,
leading dimensions, null pointers, no-op inputs and geometry boundaries.
`perf` separately measures the 200 performance cases with 10 warmups and 60
iterations. Run it alone on an otherwise idle device.

`guards` reuses the same device scalar addresses across unit, non-unit, zero and
complex alpha values and zero/nonzero beta values. It covers both triangles,
N/C transpose modes, all four host/device scalar combinations, and tile-boundary
dimensions and small-path shapes. Selected values use a 1e-3 absolute tolerance on bounded deterministic
inputs. Input matrices, unused triangle, leading-dimension padding and 32-complex
guards around C must remain byte-for-byte unchanged. For no-op calls (beta=1
and zero alpha or zero K), all C bytes must remain unchanged, including the
diagonal imaginary values. Other calls must set diagonal imaginary values to zero.
The scalar sequence includes both positive and negative zero alpha before
returning to the unit-alpha path at the same device addresses.

For device memory checking, create a separate build with
`-DCMAKE_ASC_FLAGS_RELEASE='-O2 -g --cce-enable-sanitizer'`, compile `cher2k_test`,
then run the instrumented binary with the matching CANN `mssanitizer`:

```bash
bash test/her2k/cher2k/arch22/run_validation.sh /sanitize-build/test/her2k/cher2k/cher2k_test /reports/cher2k memcheck
```

Inspect both GTest XML and `memcheck-report.log`; a successful test exit alone
does not prove that device memory checking ran or reported zero errors. Guard
tests complement the instrumented checker and are not a substitute for it.
The instrumented `MemorySanitizerPaths` test covers direct/packed OP_C, small
Cube, generic tail and small vector paths through 28 scalar-transition calls.
The larger 420-call host/device, boundary and small-path matrix remains in `guards`.
All sanitizer warnings and errors fail validation, including register-state
warnings such as `FFTS_BASE_ADDR`. Inspect kernel exit cleanup and rerun the
instrumented test before accepting a register-state finding as resolved.

For community checks, install pre-commit and `oat-py` with its dependencies
(including `lxml` and `chardet`), run `pre-commit install`, then use
`pre-commit run --files ...` with every file changed by the MR. The repository
configuration uses clang-format 18.1.8 from the GitCode mirror and serial OAT hooks. Review OAT's scanned
file count and report as well as the hook status; an environment-related skip
is not a completed compliance scan.
