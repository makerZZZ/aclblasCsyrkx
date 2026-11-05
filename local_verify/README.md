# aclblasCsyrkx 无 NPU 本地预验证

本目录提供**纯 CPU** 的预验证工具，用于在申请昇腾机时之前排除
公式错误、边界语义错误和 golden 口径错误。它不替代 NPU 上的端到端测试。

## 依赖准备

本机无系统级 BLAS/GTest，已下载到 `../local_deps/root`（免 root）：

```bash
cd /home/zmaker/lab/aclblasCsyrkx
mkdir -p local_deps/debs local_deps/root
cd local_deps/debs
apt-get download libopenblas-pthread-dev libopenblas0-pthread libgtest-dev libgfortran5 libquadmath0
cd ..
for d in debs/*.deb; do dpkg-deb -x "$d" root; done
```

## 编译

```bash
cd /home/zmaker/lab/aclblasCsyrkx
D=local_deps/root
g++ -std=c++17 -O2 -o local_verify/csyrkx_cpu_check local_verify/csyrkx_cpu_check.cpp \
  -Iops-blas/include -Iops-blas/test/frame -Iops-blas/test/utils \
  -Iops-blas/blas/common/helper -Iops-blas/test/syrkx/csyrkx \
  -I$D/usr/include -I$D/usr/include/x86_64-linux-gnu/openblas-pthread \
  -I/home/zmaker/Ascend/cann-9.2.1/include \
  -L$D/usr/lib/x86_64-linux-gnu/openblas-pthread -lopenblas \
  $D/usr/lib/x86_64-linux-gnu/libgtest.a -lpthread \
  -Wl,-rpath,$PWD/$D/usr/lib/x86_64-linux-gnu/openblas-pthread \
  -Wl,-rpath,$PWD/$D/usr/lib/x86_64-linux-gnu
```

## 运行

```bash
export LD_LIBRARY_PATH=$PWD/local_deps/root/usr/lib/x86_64-linux-gnu:$PWD/local_deps/root/usr/lib/x86_64-linux-gnu/openblas-pthread
./local_verify/csyrkx_cpu_check test_cases/csyrkx_test.csv
```

也可直接运行 `./local_verify/run_cpu_check.sh`。

## 检查内容

对 `test_cases/csyrkx_test.csv` 的 1200 条用例：

1. **返回值**：`aclblasCsyrkx_cpu` 的参数校验结果与 `expect_result` 逐条一致
   （handle / 枚举 / 负维度 / 前导维 / 空标量 / n=0 / 条件空矩阵指针）。
2. **golden 正确性**：`cblas_cgemm` golden 与一份独立双精度参考循环逐元素对比
   （同任务书容差；规模受控的 925 条）。
3. **kernel 公式正确性**：用 float 模拟 arch35 kernel 的 4M 分解与复数合成
   （`P_r=t1-t2`、`P_i=t3+t4`、复数 alpha/beta 交叉缩放、仅写 uplo 三角），
   与 golden 对比；中小规模 925 条逐元素、较大规模 226 条用 `cblas_sgemm`
   计算四个实乘积后同样对比。
4. 非 uplo 三角逐位不变（NaN 感知比较）。

预期输出：

```
cases=1200 numerically-checked=925 fast-model-checked=226 skipped-large=30
return-code mismatches : 0
golden vs reference    : 0
kernel model vs golden : 0
RESULT: PASS
```

`skipped-large=30` 是 n≈4000 的性能用例（O(n³) 参考在 CPU 上不可行），
其正确性由较小规模用例覆盖；性能/功能最终以 NPU 实测为准。

## 真实 host 入口的负向/quick-return 验证

`csyrkx_host_negative.cpp` 直接链接 `build/libops_blas.so`，调用**真实的**
`aclblasCsyrkx`，覆盖所有在触达 ACL runtime 之前返回的路径：

- 空 handle、非法 uplo/trans、负 n/k、非法 lda/ldb/ldc、空 alpha/beta；
- n==0 quick return（含 A/B/C 为空指针、ldc 非法、alpha 为空时"校验先于 no-op"）；
- 期望错误码与任务书 §2.4 一致。

```bash
g++ -std=c++17 -O2 -o local_verify/csyrkx_host_negative local_verify/csyrkx_host_negative.cpp \
  -Iops-blas/include -I/home/zmaker/Ascend/cann-9.2.1/include \
  -Lops-blas/build -lops_blas -L/home/zmaker/Ascend/cann-9.2.1/x86_64-linux/lib64 -lascendcl \
  -Wl,-rpath,$PWD/ops-blas/build -Wl,-rpath,/home/zmaker/Ascend/cann-9.2.1/x86_64-linux/lib64 \
  -Wl,--allow-shlib-undefined
LD_LIBRARY_PATH=/home/zmaker/Ascend/cann-9.2.1/x86_64-linux/lib64:/home/zmaker/Ascend/cann-9.2.1/lib64:/home/zmaker/Ascend/cann-9.2.1/x86_64-linux/devlib/linux/x86_64 \
  ./local_verify/csyrkx_host_negative
```

> 需要先完成 `bash build.sh --ops=csyrkx --soc=ascend950`（或本机等价的 cmake 构建）
> 生成 `ops-blas/build/libops_blas.so`；脚本使用 `build` 目录。

## 局限

- 不编译、不执行 Ascend C kernel，仅验证公式与边界；
- 不覆盖 tiling/多核划分、UB/L1 同步、真实 GEMM 累加顺序；
- 不支持 Device 标量指针路径（需 ACL runtime）。
