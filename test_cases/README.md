# aclblasCsyrkx 测试用例及测试指导参考

> **任务书**: [aclblasCsyrkx_Atlas950PR_task_doc.md](../aclblasCsyrkx_Atlas950PR_task_doc.md)
> **适配硬件**: 昇腾 Atlas 950PR（arch35）
> **CANN 版本**: CANN 9.1.0
> **数据类型**: complex64（单精度复数）
> **内存预算**: 单用例 host 侧 ≤ 512MB（设计选择，非硬件限制；2×n×k×8B + n×n×8B ≤ 512MB，性能用例 n ≤ 4096）
> **性能标杆**: NVIDIA H100（CUDA 12.4、驱动 550.104.05），接口 cublasCsyrkx

## 算子说明

**接口**: `aclblasCsyrkx`（列主序稠密存储，C 为 n×n 对称矩阵）
**公式**: `C = alpha*op(A)*op(B)^T + beta*C`
- trans=N：A、B 均为 n×k，`C = alpha*A*B^T + beta*C`
- trans=T：A、B 均为 k×n，`C = alpha*A^T*B + beta*C`
- trans=C：**与 trans=T 等价**，`C = alpha*A^T*B + beta*C`（syrkx 输出为对称而非 Hermitian，不取共轭；cuBLAS `cublasCsyrkx` 对 `CUBLAS_OP_C` 同样按转置执行，P102/CUDA 12.5 实测 T 与 C 输出逐位相同）
**三角语义**: 仅更新 uplo 指定三角（ACLBLAS_UPPER 只写上三角、ACLBLAS_LOWER 只写下三角），另一三角不引用、不写回
**标量**: alpha/beta 均为复数（Host 内存指针）
**维度**: lda/ldb ≥ max(1, trans=N ? n : k)，ldc ≥ max(1, n)，k ≥ 0

## CSV 列格式

列序按 FORMAT_BASELINE.md 决议（§2.3 签名顺序去 handle）：

```
case_name,description,uplo,trans,n,k,alpha_real,alpha_imag,a_fill,lda,
b_fill,ldb,beta_real,beta_imag,c_fill,ldc,expect_result,
mere_threshold,mare_multiplier,random_seed
```

约定：

- tensor 入参以 `a_fill`/`b_fill`/`c_fill` 表达填充 token（`RANDOM_NORM_5_5`、`VALUE_NORM_0`、`RANDOM_ALTER`、`RANDOM_EXTREME`、`VALUE_NORM_INF`、`VALUE_NORM_NAN`）；
- 空指针：标量 alpha/beta 写 `null`，tensor A/B/C 写 `NULLPTR`；
- 枚举写 `ACLBLAS_UPPER`/`ACLBLAS_LOWER`、`ACLBLAS_OP_N/T/C` 全名，非法枚举写 `999`；负维度写 `-1`；非法前导维写具体数值。

## 用例规模

`gen_csv.py` 使用固定随机种子 `20261028` 生成 **1200 条**用例（同种子逐字节可复现）：

- 精度：1000 条；
- 性能：200 条（`TC_PF` 前缀），前 4 条与任务书 §3.3 表格逐参数精确一致，其余为小尺寸、规模对数扫描（n 256→4096，k=n 或 n/2）、预算内随机组合，均为连续访存。

生成/重新生成用例：

```bash
python3 gen_csv.py                            # 默认 1000 精度 + 200 性能
python3 gen_csv.py --accuracy 1500 --perf 300 # 扩展条数
python3 gen_csv.py --seed 12345               # 更换随机种子
```

生成文件：

- `csyrkx_test.csv`：1200 条 CSV 驱动用例。
- `gpu_baseline.csv`：200 条性能基线（`gpu_ms` 已由 H100 实测回填，前 4 条 `csyrkx-base-000~003` 对应任务书 §3.3 典型 case）。

## 用例分类

| 类别 | 前缀 | 条数 | 说明 |
|------|------|------|------|
| L0 基础 | TC_L0 | 12 | uplo × trans 3 × 尺寸 (4,8) |
| L1 方阵尺寸 | TC_SQ | 162 | 27 种尺寸 × trans 3 × uplo 2 |
| L1 矩形形状 | TC_MN | 60 | 10 组 n≠k（含 1000×999、2048×1024）× trans 3 × uplo 2 |
| L2 标量 | TC_AB | 48 | 8 组 alpha/beta × trans 3 × uplo 2 |
| L4 前导维 | TC_LD | 24 | lda/ldb/ldc padding（+4/+8）× trans 3 × uplo 2 × 4 种尺寸 |
| L5 填充 | TC_FL | 18 | 均匀随机 / 全零 / 交替 / 极端值 / Inf / NaN（A、B、C 分别） |
| L5b 覆盖 | TC_CV | 32 | 中等尺寸（10~400）× uplo × trans(N/T) |
| L6 边界 | TC_ED | 31 | n=0 quick return / k=0 仅 uplo 三角 beta 缩放（UPPER/LOWER 各一组）/ 条件空指针叠加 / 非法枚举 / 非法前导维 / 负维度 / 特殊标量 |
| EX 扩展 | TC_EX | 613 | 尺寸池 × uplo × trans × 标量 × padding 确定性采样 |
| PF 性能 | TC_PF | 200 | 4 条任务书典型 case + 小尺寸 + 规模对数扫描 + 随机组合 |

### 边界与负向口径（§2.4/§2.5）

- `n=0`：quick return `ACLBLAS_STATUS_SUCCESS`，不写 C；
- `k=0`：仅对 uplo 三角执行 `C = beta*C`（beta=0 置零、beta=1 不变），不引用 A/B，**另一三角不得改写**（golden 同时校验非 uplo 三角逐位不变）；
- A/B 仅在「n>0 且 k>0」时报空指针错误、C 仅在「n>0」时报错 —— 故 `n0_a_b_null`、`k0_a_b_null`、`alpha0_a_null` 期望 SUCCESS；
- 空指针 alpha/beta、非法 uplo/trans、非法前导维、负 n/k：`ACLBLAS_STATUS_INVALID_VALUE`；
- 参数校验先于 no-op 判定。

## 精度验收

```bash
python3 verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --csv ./csyrkx_test.csv
python3 verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --skip-build --device 1
```

用例安装路径：`test/syrkx/csyrkx/<arch>/csyrkx_test.csv`（arch35 对应 950PR）。仓内当前**无** `test/syrkx/` 工程，需按本 CSV 列序新建 `csyrkx_param.h` 与 golden 后接入。

### 精度阈值（标准口径 + 测试工程列配置）

任务书口径（生态算子开源精度标准，COMPLEX64 按 FLOAT32 分量判定）：

- atol = 2⁻¹³ (1.22e-4)，rtol = 2⁻¹³ (1.22e-4)
- matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32×ULP

测试工程口径（MERE/MARE，CSV 按列配置）：

> 阈值列 `mere_threshold` / `mare_multiplier` 仅用于测试工程 MERE/MARE 配置（取值等于 opbase 精度标准同 dtype 档位），**不参与生态精度标准比对**；标准比对按 dtype 固定阈值执行，不随用例列变化。

- mere_threshold = 2⁻¹³ ≈ 1.2207e-4，mare_multiplier = 10.0

### 验证策略

- **比对面**：仅比对 uplo 指定三角（含对角）的实部/虚部；非 uplo 三角须与调用前逐位一致。
- **golden**：`cblas_cgemm`（trans=N 时 opA=NoTrans、opB=Trans；trans=T 时 opA=Trans、opB=NoTrans；trans=C 时按共轭），计算后把非 uplo 三角恢复为调用前原值。
- **边界语义**：beta=0 不读取 C 旧值（uplo 三角）；k=0 仅做 `beta*C`。

## 性能验收

```bash
python3 verify_performance.py --repo /path/to/ops-blas --soc ascend950 --timeout 3600
python3 verify_performance.py --repo /path/to/ops-blas --soc ascend950 --skip-build --device 1
```

### 精确 kernel 耗时获取方式（msprof）

```bash
msprof op --application="./build/test/syrkx/csyrkx/csyrkx_test --gtest_filter=*TC_PF*" --output=./prof_csyrkx
```

在 `--output` 目录下 `OPPROF_*/OpBasicInfo.csv` 中按 kernel 名统计 `Task Duration(us)` 求平均。

- **判定口径**：NPU 平均单次耗时 ≤ H100 标杆耗时 / 0.4（任务书 §3.3：不高于标杆耗时的 2.5 倍）；`gpu_ms` 已由 H100 实测回填（锁频 + warmup 5 次后有效采样 >50 次取均值，逐条与任务书 §3.3 典型 case 对齐）；P102 参考值仅存档于 `1028_work/03_gpu_verify/<平台>/<算子>/summary/gpu_baseline.csv`，不进入本文件。
- **平均性能要求（L3 算子）**：全部 `TC_PF` 用例的倍率（`H100 标杆耗时 / NPU 实测耗时`）算术平均值须 ≥ **0.5**；脚本以 `AVG_THRESHOLD = 0.5` 单独输出平均倍率判定，与逐 case 的 `PERF_THRESHOLD = 0.4` 同时满足才算性能通过。
- 性能测试须先 warmup 5 次再有效采样 >50 次取平均。

## 补充说明

- 性能用例均为连续访存：lda/ldb/ldc 紧凑，β 取 (0,0)、α 取 (1,0)。
- 上库时需同步提交 `test/syrkx/csyrkx/arch35/` 测试工程代码。
- **待研发确认（任务书 §2.5 备注）**：复数域 `trans=OP_C` 的语义（取 A^H/B^H 还是与 OP_T 等价）尚未最终确认；本用例集按共轭转置构造用例，若研发确认等价 OP_T，需同步调整 TC_ED/TC_EX 中 OP_C 用例的 golden（case 参数本身不变）。
- 部分特殊场景（Inf/NaN 填充、beta=0 配 NaN 旧值）可能导致标杆异常，可过滤并说明。
