# aclblasCsyrkx 算子设计文档（草案）

> 提交前请按模板
> https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
> 调整章节编号与格式，并放入 `cann-competitions/04_tasks/01_community-task-2026/tasklist/<任务目录>/`。

## 1 需求背景

`aclblasCsyrkx` 是 BLAS Level-3 的复数对称秩-k 扩展更新算子，对标
`cublasCsyrkx`（legacy）。与 `aclblasCsyrk` 的区别是 A、B 为独立矩阵；
与 `aclblasCherkx` 的区别是**不取共轭**，结果是对称（非 Hermitian）矩阵。

- 适配硬件：Atlas 950PR（arch35 / DAV_3510）
- 数据类型：COMPLEX64
- 公式：`C = alpha * op(A) * op(B)^T + beta * C`
  - trans=N：A、B 为 n×k，`C = alpha*A*B^T + beta*C`
  - trans=T/C：A、B 为 k×n，`C = alpha*A^T*B + beta*C`
  - **OP_C 等价 OP_T**（syrkx 是对称矩阵，共轭转置会破坏对称性；cuBLAS 同样按转置执行）
- C 为 n×n 列主序，仅更新 uplo 指定三角；alpha、beta 均为复数

## 2 接口设计

```cpp
aclblasStatus_t aclblasCsyrkx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C, int ldc);
```

参数序列与 `cublasCsyrkx` 一一对应，声明在 `include/cann_ops_blas.h`，不引入
950PR 私有接口。

返回值：

| 条件 | 返回值 |
|------|--------|
| handle == nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo / trans 非法 | `ACLBLAS_STATUS_INVALID_VALUE` |
| n < 0 / k < 0 | `ACLBLAS_STATUS_INVALID_VALUE` |
| lda/ldb < max(1, trans=N?n:k)，ldc < max(1,n) | `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha / beta == nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| n>0 且 C == nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha != 0 且 k>0 且 A/B == nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| n == 0（以上校验通过后） | `ACLBLAS_STATUS_SUCCESS`，不写 C |
| workspace 超 2GiB / 分配失败 | `ACLBLAS_STATUS_ALLOC_FAILED` |

alpha/beta 支持 Host 或 Device 指针（两者须同侧），Device 指针经 stream
D2H 读回后再做值相关分支。

## 3 总体方案

采用 4M 分解把复数 GEMM 拆成 4 个实数 GEMM，复用仓内
`gemm_kernel_do`（AIC）、AIV deinterleave 与 combine：

```
Phase 0 (AIV × 2): A -> Ar,Ai；B -> Br,Bi
Phase 1 (AIC × 4): 4 个实数 GEMM
    trans=N : t1=Ar*Br^T, t2=Ai*Bi^T, t3=Ar*Bi^T, t4=Ai*Br^T
    trans=T : t1=Ar^T*Br, t2=Ai^T*Bi, t3=Ar^T*Bi, t4=Ai^T*Br
Phase 2 (AIV): P_r=t1-t2, P_i=t3+t4
    Cr = alpha_r*P_r - alpha_i*P_i
    Ci = alpha_r*P_i + alpha_i*P_r
    C  = (Cr + i*Ci) + beta*C_old        （仅 uplo 三角）
```

数学推导（trans=N）：

```
A*B^T = (Ar+iAi)(Br^T+iBi^T) = (Ar Br^T - Ai Bi^T) + i(Ar Bi^T + Ai Br^T)
```

trans=T 同理：`A^T*B = (Ar^T Br - Ai^T Bi) + i(Ar^T Bi + Ai^T Br)`。

### 3.1 为什么复用 gemm_kernel_do

- 仓内 `cherk/cher2k/cherkx` 已验证 950PR 上的 4M 实数 GEMM 路径；
- 本算子 C 恒为 n×n，column-major 存储约定下的 fixpipe 转置只影响
  transpose 标志，按 cher2k 的 operand 对应表即可得到目标乘积；
- Phase 2 只需一次 AIV 合并，避免自行实现 cube 流水。

### 3.2 各阶段细节

**Phase 0（deinterleave）**：按行分核，64×64 tile，GM→UB 用
`te::Copy` 二维跨步搬运，`DeInterleave` 拆分实虚部，再 `te::Copy` 写回
两片连续 float 矩阵；tile 间用 V/MTE2/MTE3 事件避免 WAR/RAW。

**Phase 1（4 个 GEMM）**：Ar/Ai/Br/Bi 紧密存储，lda=ldb=physRows；
tiling 的 M=N=n，K=k；按 `_GEMM_KERNEL_DEP_OPS` 与 gemm 源码一起编译。

**Phase 2（combine）**：按 64×64 tile 只遍历 uplo 三角：

- UPPER：`jBase >= iBase`，完整在三角内的块用一次二维 copy；
  跨对角块逐列只写 `min(j-i+1, rows)` 个元素；
- LOWER：`jBase < iBase+rows`；跨对角块为“整列写回”，先把该列非 uplo 段
  用 C 旧值在 UB 内恢复，再一次 MTE3 写整列，保证非 uplo 逐位不变；
- alpha 实部/虚部、beta 实部/虚部分支：`alpha_i==0`/`beta_i==0` 时走
  无交叉项的快路径（与实数标量历史行为逐位一致）；
- `k==0` 或 `alpha==0`：不启动 deinterleave/GEMM，只做 `C=beta*C`；
- `beta==0`：不读取 C 旧值，避免 `0*NaN/Inf` 污染。

## 4 内存与 workspace

GEMM 路径需要：

- 4 × `n × align(n,16) × 4B` 临时实矩阵；
- 4 × `physRows × physCols × 4B`（Ar/Ai/Br/Bi）。

统一从 handle workspace 分配（512B 对齐），乘法用 64 位溢出检查，
超过 2GiB 返回 `ALLOC_FAILED`。n=k=4096 时约 537MB。
`alpha==0` 或 `k==0` 路径不申请 workspace。

## 5 精度与确定性

- 4M 分解 + 单精度 GEMM/合成，误差量级与 cuBLAS 的实数化实现相当；
- 精度判定（COMPLEX64 按 FLOAT32 分量）：atol=rtol=2⁻¹³，
  matched_ratio ≥ 0.99，max_abs_error ≤ max(1e-2, 32×ULP)；
- 非有限值按任务书 §3.2 第 4 条：NaN/NaN、同号 Inf/Inf、NaN↔Inf 等效，
  不计入统计；
- 确定性：tiling 与累加路径固定，同输入多次执行逐位一致。

## 6 测试方案

1. `test/syrkx/csyrkx/arch35/csyrkx_test.csv`：1200 条（1000 精度 + 200 性能），
   覆盖 uplo×trans、尺寸扫描、标量特殊值、前导维 padding、Inf/NaN、边界/负向；
2. GTest 判定：返回值、uplo 三角混合容差、非 uplo 与 ldc padding 逐位不变；
3. 无 NPU 预验证（`local_verify/`）：
   - `cblas_cgemm` golden vs 独立双精度参考；
   - kernel 4M 公式的 float 模拟 vs golden（含大规模 cblas_sgemm 版本）；
   - 真实 host 入口的负向/quick-return 路径；
4. 上机后：全量 GTest、`verify_accuracy.py`、`msprof op` 性能采集、
   重复执行确定性检查。

## 7 性能预期与优化方向

当前版本 4 个 GEMM 计算完整 n×n（仅 Phase 2 限制三角），
kernel 数 = 2 deinterleave + 4 GEMM + 1 combine = 7。

可优化项（按收益）：

1. GEMM 只计算 uplo 三角（省约一半 cube 算力）；
2. 4 个 GEMM 融合为单 kernel（减少 launch 与 temp 读写）；
3. 小 n（≤64）走单 kernel SIMT 直算，规避固定 launch 开销；
4. combine 小 n 时降低 AIV 核数。

是否达标以 950PR 实测 `TC_PF_*` 为准（单 case ≥0.4×、平均 ≥0.5× H100）。

## 8 风险与待确认

1. CANN 9.1.0 与本地 9.2.1 的 asc-devkit 头文件差异，需上机重编确认；
2. `test_cases/README.md` 中 OP_C 的旧说明与任务书冲突，以任务书为准；
3. 小尺寸性能受 7 次 launch 固定开销限制，可能需要 SIMT 直算路径；
4. 超大 k 时 4M 的实数化和误差需用全量精度用例确认。
