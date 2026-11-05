# aclblasCsyrkx 自测报告（骨架）

> 按任务书模板填写：
> https://docs.qq.com/sheet/DUmVWWndaUE12WGFB?tab=BB08J2
> 需要包含：用例参数、精度对比结果及截图、性能数据及截图。

## 1 环境

| 项 | 值 |
|----|----|
| 硬件 | Atlas 950PR |
| CANN | 9.1.0 |
| 算子仓 | `<个人仓>/ops-blas`，分支/commit：`<填写>` |
| 编译命令 | `bash build.sh --ops=csyrkx --soc=ascend950` |
| 测试命令 | `./build/test/syrkx/csyrkx/csyrkx_test` |
| cblas golden | Netlib BLAS 3.10.0（`libblas-dev` / OpenBLAS 的 cblas_cgemm） |

## 2 用例与参数

- CSV：`test/syrkx/csyrkx/arch35/csyrkx_test.csv`（1200 条）
- 覆盖矩阵：

| 维度 | 覆盖 |
|------|------|
| uplo | UPPER / LOWER / 非法 999 |
| trans | OP_N / OP_T / OP_C / 非法 999 |
| n, k | 0、1、小质数、2 的幂 ±1、非对齐、大规模（至 4096） |
| alpha/beta | 复数随机、0、1、纯虚、大值、null |
| 前导维 | 紧凑 / +4 / +8 padding |
| 特殊值 | 交替、极端值、Inf、NaN |
| 边界 | n=0、k=0、负维度、空指针、非法前导维 |

## 3 精度结果

填写格式（每个用例一行，或按类别汇总 + 附件原始日志）：

| 用例 | 返回值 | matched_ratio(real/imag) | max_abs_err(real/imag) | 非uplo逐位 | padding逐位 | 结论 |
|------|--------|--------------------------|------------------------|------------|-------------|------|
| TC_L0_001 | PASS | 1.0 / 1.0 | ... | PASS | - | PASS |

汇总：

- 精度用例通过数 / 总数：`<填写>`
- 失败用例及原因：`<填写或无>`
- 非有限值用例单独统计：`<填写>`

附件：

- gtest 完整输出：`<日志文件>`
- 精度截图：`<截图>`

## 4 性能结果

采集方式：msprof op，5 次 warmup，10 次有效采样取算术平均。

| case | n | k | uplo/trans | GPU 标杆(us) | NPU 实测(us) | 倍率(GPU/NPU) |
|------|---|---|------------|--------------|--------------|----------------|
| csyrkx-base-000 | 256 | 256 | U/N | 13.094 | `<填写>` | `<填写>` |
| csyrkx-base-001 | 1024 | 1024 | L/N | 168.113 | | |
| csyrkx-base-002 | 2048 | 1024 | U/T | 471.258 | | |
| csyrkx-base-003 | 4096 | 2048 | L/T | 2744.080 | | |

汇总：

- 单 case 最小倍率：`<填写>`（要求 ≥0.4）
- 全部 TC_PF 倍率算术平均：`<填写>`（要求 ≥0.5）
- 结论：`<通过/不通过>`

附件：

- `OpBasicInfo.csv` / `summary.csv`
- 性能截图

## 5 结论

`<填写：功能/精度/性能是否全部达标；遗留问题>`
