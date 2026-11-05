# aclblasCsymv 自测

## 范围

- 硬件：Ascend 950PR（arch35）
- CANN：9.1.0
- 数据类型：`COMPLEX64`
- 语义：`y = alpha * A * x + beta * y`，A 为列主序复数对称矩阵，未存储三角按转置读取且不取共轭

`arch35/csymv_test.csv` 共 1202 条：1002 条功能/精度用例和 200 条性能参考用例。功能用例按固定种子奇偶选择均匀或正态输入，比例各 50%；未使用的矩阵三角填入 NaN，用于检测误读和错误共轭。

## 构建

```bash
bash build.sh --soc=ascend950 --ops=csymv
```

在 950PR 上使用仓库的测试入口运行全部用例：

```bash
bash build.sh --soc=ascend950 --ops=csymv --run --device=0
```

若需分别保存精度和性能原始日志，构建后可直接调用 GTest 二进制：

```bash
build/test/symv/csymv/csymv_test --gtest_filter='-*TC_PF*' | tee accuracy.log
build/test/symv/csymv/csymv_test --gtest_filter='*TC_PF*' | tee performance.log
```

`--device` 指定逻辑设备索引；资源映射不同时可先设置 `ASCEND_RT_VISIBLE_DEVICES`。

## 测试口径

- 精度：实部、虚部分别使用 `rtol=2^-10`、`atol=2^-16`、匹配率至少 0.99，并限制逐元素最大绝对误差为 `max(1e-2, 32*ULP)`。
- 特殊值：golden 为 NaN 时要求结果为 NaN；golden 为正负 Inf 时要求符号一致。
- 性能：全部 200 条 `TC_PF` 用例逐条执行；每条先预热 10 次，再连续调用 60 次并同步 stream，使用平均耗时判定。
- 判定：`NPU_ms <= GPU_ms / 0.4`，其中 `GPU_ms` 来自同目录 `gpu_baseline.csv`；前三条基线分别对应任务书的三项硬门槛。

| uplo | n | 最大平均耗时 |
|---|---:|---:|
| UPPER | 512 | 22.75 us |
| UPPER | 2048 | 55.62 us |
| LOWER | 2048 | 40.25 us |

性能日志为每条用例输出 case、uplo、n、NPU 平均耗时、GPU 基线、性能比、门槛和 PASS/FAIL。
验收时应核对 200 条 `[PERF]` 均为 PASS，以及 GTest 的通过总数；不要只依赖末尾汇总。
