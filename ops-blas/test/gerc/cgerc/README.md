# aclblasCgerc Ascend 950PR 自测说明

`arch35/cgerc_test.csv` 是任务附件提供的 1200 条原始用例，SHA-256 为
`2fd8c6fa4bd89bd5a938a3a8c8bbf82f062f39c2e9eb1e64e7454818bd331f83`。
`arch35/gpu_baseline.csv` 是任务附件提供的 200 条 GPU 性能基线，SHA-256 为
`6f2c9ea86dc3a3c6c70c31a259b0d1d57f83e110686e8fa5523ad12c2146f904`（仅将附件换行统一为 LF）。

## 覆盖范围

| 用例组 | 数量 | 覆盖内容 |
|--------|-----:|----------|
| TC_L0 | 6 | 4、8 阶基础方阵及正负步长 |
| TC_SQ | 23 | 1 至 2048 的方阵尺寸扫描 |
| TC_AB | 10 | 零、一、负数、纯虚数、一般复数及大/小 alpha |
| TC_RC | 12 | 宽、窄非方阵 |
| TC_LD / TC_INC | 39 | lda padding，incx/incy 为 ±1、±2、±3 的组合 |
| TC_FL | 18 | 随机、全零、交替、极端值、Inf、NaN |
| TC_ED | 14 | quick return、空指针及非法参数 |
| TC_EX | 878 | 尺寸、标量、步长和 padding 扩展组合 |
| TC_PF | 200 | 连续布局性能与大规模内存场景 |

有限值精度使用测试框架链接的参考 BLAS `cblas_cgerc` 生成 golden；当 `alpha=(1,0)` 且输入含
Inf、NaN 或可能溢出的 FP32 极值时，使用固定运算顺序的标量 golden，消除不同 CBLAS 实现对
特殊值 fast path 的语义差异。复数的实部、虚部分别按 FP32 标准比对。`TC_PF_1001` 至
`TC_PF_1200` 与 GPU 基线附件的 200 行按顺序一一对应，测试还会
逐项核验 `m`、`n`、`incx`、`incy`，防止基线错配。每组均先预热 20 次，再执行 5 批、
每批 100 次采样，使用 ACL stream event 统计每批平均设备耗时并取中位数，以排除系统调度
造成的单次离群值；随后分别执行
`NPU_time <= 对应 GPU_time / 0.4` 门禁。前四组 512、1024、2048、4096 方阵的门槛分别为
8.675 us、12.850 us、44.6325 us、241.085 us；其余门槛由附件逐行计算。

## 运行验收

在装有 CANN 9.1.0、Ascend 950PR 且已加载 CANN 环境变量的机器上，从仓库根目录执行：

```bash
bash build.sh --soc=ascend950 --ops=cgerc
./build/test/gerc/cgerc/cgerc_test --gtest_color=no
```

测试二进制执行全部 1205 项 GTest（1200 条 CSV 用例、参数校验、`y=0`/`x=Inf` Netlib
语义、地址溢出、SIMD 特殊值及分核/UB 边界回归），并逐项检查 200 组附件性能门槛。
可通过 `ASCEND_DEVICE_ID` 选择设备。

补充回归使用 64、65、257 行分别覆盖 SIMD 整寄存器及尾部 mask，逐项验证 x/y 的正负 Inf、
单分量 NaN、FP32 极值。1001 列覆盖三列融合与多 tile 流水，1、3、73 列覆盖小列数及
非整除分核；7872/7873 行覆盖当前 UB 容量下 SIMD 上界及 SIMT 回退。A 前后设置独立哨兵，
验证计算结果的同时检查矩阵边界未被写入。上述尺寸只用于测试，算子仍按运行时维度和资源分配。
