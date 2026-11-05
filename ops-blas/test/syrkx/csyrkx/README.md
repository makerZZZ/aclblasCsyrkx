# aclblasCsyrkx 测试工程

CSV 驱动的 GTest 精度测试，覆盖 `aclblasCsyrkx`（COMPLEX64，Atlas 950PR / arch35）。

## 文件

| 文件 | 作用 |
|------|------|
| `csyrkx_param.h` | 解析 `arch35/csyrkx_test.csv` |
| `csyrkx_golden.h` | CPU golden：参数校验语义 + `cblas_cgemm` |
| `arch35/csyrkx_npu_wrapper.h` | Host→Device 搬运、返回结果回拷 |
| `arch35/csyrkx_test.cpp` | GTest 用例：返回值、uplo 精度、非 uplo 逐位不变、ldc padding 逐位不变 |
| `arch35/csyrkx_test.csv` | 1200 条用例（1000 精度 + 200 性能） |

## CSV 列

```
case_name,description,uplo,trans,n,k,alpha_real,alpha_imag,a_fill,lda,
b_fill,ldb,beta_real,beta_imag,c_fill,ldc,expect_result,
mere_threshold,mare_multiplier,random_seed
```

- `null`（alpha_real / beta_real）= 空标量指针；`NULLPTR`（a_fill/b_fill/c_fill）= 空矩阵指针；
- 非法枚举写 `999`，负维度写 `-1`；
- 填充 token：`RANDOM_NORM_5_5`、`VALUE_NORM_0`、`RANDOM_ALTER`、`RANDOM_EXTREME`、
  `VALUE_NORM_INF`、`VALUE_NORM_NAN`。

## 构建与运行（NPU 机器）

在 ops-blas 仓库根目录：

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash build.sh --ops=csyrkx --soc=ascend950 --run
```

或分步执行：

```bash
bash build.sh --ops=csyrkx --soc=ascend950
./build/test/syrkx/csyrkx/csyrkx_test                 # 全部 CSV 用例
./build/test/syrkx/csyrkx/csyrkx_test --gtest_filter=*TC_L0*
./build/test/syrkx/csyrkx/csyrkx_test --gtest_filter=*TC_ED*   # 边界/负向
```

构建依赖：`libblas-dev`（或 OpenBLAS，提供 `cblas.h`）、GTest，见
`cmake/test.cmake` 的 `_ops_blas_ensure_refblas_found`。

## 判定口径

1. **返回值**：与 `expect_result` 整数相等。
2. **uplo 三角**：实部/虚部分别按生态精度标准判定（FLOAT32 分量：
   atol = rtol = 2⁻¹³，matched_ratio ≥ 0.99，max_abs_error ≤ max(1e-2, 32×ULP)）。
   非有限值按任务书 §3.2 第 4 条单独判定（NaN/NaN、同号 Inf、NaN↔Inf 等效），
   不计入 matched_ratio 与 max_abs_error。
3. **非 uplo 三角**：与调用前 C 逐位一致（含 NaN/Inf）。
4. **ldc padding 行**（ldc > n）：与 Host 预填值逐位一致。

> 性能用例（`TC_PF_*`）只做功能/精度抽查；正式性能数据用 `msprof op` 采集，
> 见任务书 §3.3 与仓库根目录 `test_cases/verify_performance.py`。

## 无 NPU 时的本地预验证

见 `local_verify/README.md`：在纯 CPU 上运行 golden 与独立双精度参考、以及
kernel 4M 合成公式的浮点模拟，用于上机前排除公式/边界错误。
