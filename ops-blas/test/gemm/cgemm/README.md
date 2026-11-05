# aclblasCgemm test

The tests in this directory validate the `aclblasCgemm` interface on Ascend 950PR (`arch35`).

## Directory structure

```text
test/gemm/cgemm/
├── CMakeLists.txt
├── README.md
├── cgemm_golden.h
├── cgemm_param.h
└── arch35/
    ├── cgemm_npu_wrapper.h
    ├── cgemm_test.cpp
    ├── cgemm_test.csv
    └── cgemm_test_full.csv
```

The full CSV covers column-major complex matrix multiplication, all `N/T/C` transpose combinations,
non-contiguous leading dimensions, complex `alpha` and `beta`, boundary values, and invalid parameters. The CPU
reference result is computed with CBLAS and compared with the NPU result by the common test verifier.

`arch35/cgemm_test.csv` contains the first 50 cases from `arch35/cgemm_test_full.csv`,
in their original order. The test executable runs these 50 CSV cases and the separate null-handle test.
The full CSV retains all 1000 cases for reference and is not loaded by the default test entry.

## Build and run

From the repository root, build the operator and its test target:

```bash
bash build.sh --soc=ascend950 --ops=cgemm
```

Run the accuracy cases from the generated test executable in the build output directory:

```bash
./build/test/gemm/cgemm/cgemm_test
```
