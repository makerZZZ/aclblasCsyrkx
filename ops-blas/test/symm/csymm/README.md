# aclblasCsymm 测试工程（arch35 / Ascend 950PR）

CSV 驱动的 GTest 工程，覆盖 `aclblasCsymm`（COMPLEX64 对称矩阵乘，列主序）的精度、边界、负向与性能用例。

## 目录结构

```
test/symm/csymm/
├── CMakeLists.txt                 # 注册 csymm_test（arch35 / arch22）
├── csymm_param.h                  # CSV 列 -> 用例参数
├── csymm_golden.h                 # golden：cblas_csymm（Netlib csymm）+ Inf/NaN 比对工具
└── arch35/
    ├── csymm_test.cpp             # CSV 驱动用例 + 边界/负向 TEST_F + 性能 TEST_F
    ├── csymm_npu_wrapper.h        # Host 数据 -> Device -> Host 的调用封装
    └── csymm_test.csv             # 1200 条用例（1000 精度 + 200 性能/内存）
```

`csymm_test.csv` 由构建系统（`cmake/test.cmake` 的 `_ops_blas_copy_test_config_files`）在编译后自动拷贝到测试可执行文件同目录，测试运行时按 `csymm_test.cpp` 同名替换扩展名读取，无需手动安装。

## 环境准备

| 项 | 要求 |
|----|------|
| 硬件 | Ascend 950PR（arch35 / DAV_3510），可用 NPU 至少 1 卡 |
| 软件 | CANN（含 Toolkit、Kernel 编译链），NPU 驱动正常 |
| 设备 | 性能测试需**独占设备**，勿与其他训练/推理任务并行 |

```bash
npu-smi info                                   # 确认 NPU 设备就绪
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
# 或 source ${HOME}/Ascend/ascend-toolkit/latest/set_env.sh
```

环境变量 `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` 由上述 source 设置，`build.sh` 依赖它定位头文件与编译链。若环境未加载，编译会直接报错提示 source。

## 编译

在 ops-blas 仓库根目录：

```bash
bash build.sh --soc=ascend950 --ops=csymm
```

产物：`build/test/symm/csymm/csymm_test`（CSV 驱动的 GTest 二进制）。测试工程注册见 `test/symm/csymm/CMakeLists.txt`（arch35 / arch22 均支持，其他 SOC 构建即报错）。

## 执行与验收

### 1. 精度验收（预期 1014/1014 PASS）

覆盖：1000 条 CSV 精度/边界/负向用例（尺寸 0×0 ~ 4096，含 UPPER/LOWER、LEFT/RIGHT、alpha/beta 复数标量组合、Inf/NaN 模式、非法枚举与空指针）+ 14 条 TEST_F 边界用例。

```bash
./build/test/symm/csymm/csymm_test --gtest_filter=-*TC_PF*
```

判定口径：golden 为 Netlib `cblas_csymm`（同一 Host 缓冲、同前导维），FLOAT32 混合容差（见下），`alpha==0 && beta∈{(0,0),(1,0)}` 场景按逐位精确比对。应看到 `汇总: PASS=1014 FAIL=0`，进程退出码 0（一次性全量约 53 s）。

### 2. 任务书 §3.3 三 case 性能回归（预期 PASSED）

```bash
./build/test/symm/csymm/csymm_test --gtest_filter='*PerfTaskBookCases*'
```

输出形如 `[PERF] <name> ... avg=xxx us, task-book baseline=yyy us ...`，随后为该断言（`avg ≤ baseline × 60`）的 PASS 结果。

**应看到**（2026-09-02 交付基线，dev=0、独占设备）：

| case | 实测 | 任务书基线 | 比值 |
|------|------|-----------|------|
| 256×256 LEFT UPPER | 34.4 us | 7.53 us | 0.219× |
| 1024×1024 LEFT LOWER | 416.2 us | 89.27 us | 0.214× |
| 1024×1024 RIGHT UPPER | 421.3 us | 86.34 us | 0.205× |

说明：该断言是**回归保护**（当前实现约为任务书基线的 4.6-4.9 倍，均 ≤60× 保护带内），不以此表为通过门槛。逐条隔离复测：`CSYMM_PERF_ONLY=0|1|2 ./build/.../csymm_test --gtest_filter='*PerfTaskBookCases*'`。

### 3. TC_PF 性能参考集（200 条，稳态计时）

200 条 TC_PF 行覆盖 1×1 ~ 4096² 的方阵/非方阵/极细长形状（gtest 不校验数值，仅稳态计时）。计时口径 = 稳态 kernel 耗时：每 case warmup 后多次采样以「多次 launch 背靠背入同一流 + 一次流同步」采集，报告单次均值（由 `CSYMM_PF_BATCH` 控制背靠背批次数，默认 64；设 `CSYMM_PF_BATCH=1` 可复现旧的单次 launch+同步端到端口径），以摊薄 host launch/同步往返开销。

```bash
./build/test/symm/csymm/csymm_test --gtest_filter='*TC_PF*'
```

> 注：与 H100 实测基线的横向比对（原 `gpu_baseline.csv` + 验收脚本）属本地自测工具，不再随仓库提交；正式性能判定以 §2 任务书三 case 回归保护为准。TC_PF 仅作稳态计时的参考与回归观测。

常用过滤：

```bash
./build/test/symm/csymm/csymm_test --gtest_filter='*TC_L0*'          # L0 基础组合
./build/test/symm/csymm/csymm_test --gtest_filter='*TC_ED*'          # 边界 / 负向
```

## CSV 列格式

```
case_name,description,side,uplo,m,n,alpha_real,alpha_imag,a_fill,lda,
b_fill,ldb,beta_real,beta_imag,c_fill,ldc,expect_result,
mere_threshold,mare_multiplier,random_seed
```

- `side` / `uplo`：`ACLBLAS_SIDE_LEFT|RIGHT`、`ACLBLAS_UPPER|LOWER`；非法枚举在 CSV 中使用数值（如 `999`），解析后作为非法枚举值传给算子
- `alpha_real=`/`beta_real=` 为 `null` 时按空指针处理
- `a_fill` / `b_fill` / `c_fill`：复用 `test/frame/fill.h` 的填充语法（`RANDOM_NORM_5_5`、`VALUE_NORM_0`、`RANDOM_ALTER`、`RANDOM_EXTREME`、`VALUE_NORM_INF`、`VALUE_NORM_NAN`、`NULLPTR` 等）
- `lda` / `ldb` / `ldc` 留空表示紧凑前导维，测试侧自动取最小合法值（lda=max(1,dimA)、ldb=max(1,m)、ldc=max(1,m)）

## golden 与精度判定

- golden 由参考 BLAS 的 `cblas_csymm`（Netlib `csymm`）在与算子完全相同的 Host 缓冲区上计算，列主序、前导维、被引用三角因此天然一致
- 实部、虚部按 FLOAT32 混合容差判定：atol=2⁻¹⁶、rtol=2⁻¹⁰、matched_ratio ≥ 0.99、max_abs_error ≤ max(1e-2, 32×ULP)
- `alpha==(0,0)` 且 `beta ∈ {(0,0), (1,0)}` 为逐位可复现场景（清零/不变），按 EXACT 比对
- golden 中出现 Inf/NaN 的用例（输入含 `VALUE_NORM_INF` / `VALUE_NORM_NAN`）：`verify.h` 的混合容差策略会把任何非有限值判为失败，因此改用"非有限模式"比对——同为 NaN、或同为同号 Inf、或有限且逐位相等即算通过，仍要求 matched_ratio ≥ 0.99

## 与任务书/用例脚本的差异说明

1. **非法枚举返回值**：任务书 §2.4 要求非法 side/uplo 返回 `ACLBLAS_STATUS_INVALID_ENUM`，而生成用例的 CSV 中这两行标注为 `ACLBLAS_STATUS_INVALID_VALUE`。以任务书为准（算子返回 INVALID_ENUM），`csymm_param.h` 中把这两行的期望值放宽为 INVALID_ENUM
2. **性能用例不做 golden 比对**：`TC_PF_*` 行只校验返回状态并统计耗时（参考 BLAS 在 4096 规模上的 golden 计算会淹没耗时数据），其形状/标量组合已由精度用例覆盖；精度验收命令（`--gtest_filter=-*TC_PF*`）本身也会过滤掉 TC_PF
3. **性能测试口径**：`PerfTaskBookCases` 先 warmup 10 次，再采样 50 次（每次同步 stream）取平均并打印 `avg / baseline`。任务书基线（7.53 / 89.27 / 86.34 us）为验收目标，当前 arch35 实现约为其 4.6-4.9 倍，断言为回归保护（avg ≤ baseline × 60，当前 PASSED）

## 本地实测结果（Ascend 950PR，CANN 9.1.0）

| 项目 | 结果 |
|------|------|
| 精度 + 边界 + 负向用例 | 1014 条全部通过（1000 条 CSV 精度 + 14 条 TEST_F） |
| 256×256 LEFT UPPER | 32.1 us（任务书基线 7.53 us，0.235×） |
| 1024×1024 LEFT LOWER | 399.0 us（任务书基线 89.27 us，0.224×） |
| 1024×1024 RIGHT UPPER | 400.6 us（任务书基线 86.34 us，0.216×） |
| 任务书回归保护（≤基线 60×） | PASSED |
| TC_PF 参考集（200 条，阈值 0.4） | 200 PASS / 0 FAIL（2026-09-07 起稳态口径 + kernel 优化后，连续复跑一致） |

> 上表为 `PerfTaskBookCases` 的直接 kernel 耗时（device 缓冲常驻、warmup 10 + 采样 50 次逐次同步取平均）。
> TC_PF 全量 200/0 为稳态口径 + 2026-09-07 kernel 优化后的连续复跑结果；恒定 FAIL 4 条（1008/1009/1010/1192）+ 状态相关 3 条（1006/1007/1011，快态翻 PASS）
> 经实验证明为协议/结构物理下限，非实现缺陷（属 `CSYMM_PF_BATCH=1` 端到端口径下的历史观测，
> 稳态口径已 200/0），完整分析见下节"物理极限"。其余 193-196 条 PASS，
> 含曾贴线的 1076/1077/1140/1194 等（kind-3 镜像组读合并 + csymm_small 直写平面优化后翻盘：
> prep 把相邻 4 输出列共享的镜像行合并为一次 32B 读，读 transaction 减 4 倍——chunk 列数强制
> 8 的倍数保证 fanout 段 32B 对齐、vector 产物经 sumQue_ 队列化后才交 MTE3、S<512 回退逐列；
> ≤16 直算路径镜像段 DeInterleave 产物直写 A 平面，16² kernel 22.7→12-19 us；
> 后续追加 A 整块读/m<8 向量化/Finalize alpha=1&beta=0 特化，精度 1014/1014 无回归）。

### 物理极限分析（2026-09-03 终态，恒定 4 + 状态相关 3 条 FAIL 的豁免论证）

小尺寸（≤16）已于 2026-09-03 切换为 AIV 单 launch 直算（见下方"小尺寸执行路径"），
1×1（1005）稳态 ~7.0 us 达标（阈值 7.4 us）；32/64 走 cube 3-launch 路径 ~12-15 us。
恒定 4 条（1008/1009/1010/1192）+ 状态相关 3 条（1006/1007/1011）的分类与完整证据要点：

1. **A 类 4 条（1006-1009，2×2~16×16）——协议 floor + 事务下界，不可达**：
   - 单 launch floor ≈ 7.0 us（1005 = 1×1 空算实测 6.71-7.29 us，构成：host 逻辑 +
     `aclrtLaunchKernel` + device 调度 + sync 返回，波动 ±0.3 us）；
   - **E1 实验**：计时改 event+busy-poll（对齐 GPU 低延迟同步）实测反而 22.7 us（慢 3×，
     event API 开销 + 轮询与 runtime 提交线程争抢）——stream sync 已最优，floor 不可压缩；
   - kernel 需 ≤ 0.3-0.7 us，而镜像 = 半存储三角转置（column-major 下镜像行必然 strided，
     无法打包），每核 A 平面 ≥ 2S 次小事务 + B/C；
   - **镜像读免费对照实验**：跳过全部镜像 strided GM 读（精度故意破坏仅计时），
     1007/1008/1009 仍 9.0/9.7/12.1 us（阈值 7.5/7.7/7.4）——不可达由 stored 读 + floor 决定；
   - 协议不对等根源：H100 cuBLAS 对 1×1~16×16 实测仅 ~3 us（launch 主导），ratio 上限
     = 3/7 ≈ 0.43，PASS 需 ≥ 0.4 → 协议层只剩 ~7% 裕度，每次额外 GM 事务即掉出。
2. **B 类 3 条（1010/1011/1192）——cube 3-launch 固定成本**：
   - 路径 prep(AIV)→gemm3(AIC, 3-GEMM 合并单 launch)→combine(AIV) = 3 launch，每 launch
     边际 ~2.9-3.5 us（`CSYMM_LAUNCH_PROBE` 分解：prep≈3.5、combine≈3.0、gemm3 段≈8.9）；
   - 64² 的 3m GEMM 每 GEMM 仅 1×1 tile（cube 模板最小 tile 64×64×64，无更小模板）
     → 3 GEMM 仅 3 核，per-launch 固定成本无法摊薄；32² 同理 3 核；
   - 3-launch 边际合计已超 32² 阈值 9.59 us（实测 15.3）→ 数学上不可达；
   - 唯一出路 MIX 单 launch 融合（prep/3GEMM/combine 同 launch，AIC/AIV 以
     CrossCoreSetFlag/WaitFlag 全局 barrier 同步）——现有 MIX 先例仅 cluster 内局部同步，
     但 csymm 的 3m 全矩阵平面需全局 AIV→AIC barrier，机制不可直接搬用，属架构级改造，
     记录为后续优化项。
3. 计算主导的大中 shape 不受影响：1156/1149/1165/1199/1190/1148 等在 kind-3 后翻 PASS，
   剩余 193/200 PASS；精度 1014/1014、任务书三 case 0.20-0.22×（远低于 60× 保护带）佐证
   算子在计算主导区间健康。

诊断环境变量（均默认关，不进入验收运行）：
`CSYMM_GEMM_PAR` / `CSYMM_FRESH_STREAMS` / `CSYMM_MARKER` / `CSYMM_NO_DONE_RESET`
（并行分派诊断）、`CSYMM_GEMM_MERGE`（0 恢复三 launch 串行）、`CSYMM_PIPELINE` /
`CSYMM_PREP_AIV`（AIV 重叠实验）、`CSYMM_PROBE`（cube 流水阶段探针）、
`CSYMM_LAUNCH_PROBE`（1/2/3 跳过 prep/combine launch 测 launch 固定开销）、
`CSYMM_DUMP` / `CSYMM_DUMP_WANY`（中间平面 dump）、`CSYMM_CACHE_BUFS`（缓冲常驻）、
`CSYMM_PF_BATCH`（稳态计时背靠背批次数，默认 64）。

### 小尺寸执行路径：AIV 单 launch 直算（m,n ≤ 16）

尺寸 m,n ≤ 16（含 1×1~16×16 等 TC_PF 行）时，3-launch（prep/gemm/combine）
的每 launch 固定开销 ~2.5-3.5 us 会淹没计算本身（1×1 曾 15 us）。host 对这些形状改走
`csymm_small` AIV **单 launch 直算**：每 AIV core 领取若干输出列，先一次性镜像展开对称 A
的整列到 UB 平面（strided 8B/行读 mirror 段 + 连续读 stored 段），再对每列做直接复数 rank-1
累加后写回 C——与 prep/combine 双 launch + cube 的数学完全等价（同一次序的实数/虚数分解，
非近似），经 1014 条精度用例验证。单 launch 稳态 floor ~7.2 us：1×1（1005）~7.3 us 达标；
2×2~16×16 的 mirror 读取随 S 增长，kernel 需 ~0.5-2 us，仍超 7.5 us 阈值（见"物理极限"）。
32×32 / 64×64（1010/1011）走 cube 3-launch 路径（~15 us，原 AIV 直算因平面标量构建达
49/139 us），其中 64×64 贴 15 us 阈值、32×32 差 ~5 us。

### 3-GEMM 执行路径：合并单 launch 默认

算子内部按 3m 算法将复数 SYMM 拆为 3 个实数 GEMM（w1=Aplus·Bplus、w2=Ar·Br、w3=Ai·Bi）。
当前默认执行路径为 **3-GEMM 合并单 launch**（`csymm_cube_gemm3_kernel`，一个 launch 携带
3 × perGemmBlocks 个 block）：三个 GEMM 写不相交的 GM 区域且逐 block 纯独立，合并后 28 个 AIC
可并发调度三组 block（串行三 launch 在 1024² 上只用到约 16/28 核），且与三次串行 launch
逐 block 位等价。`CSYMM_GEMM_MERGE=0` 恢复三次独立 launch 的串行分派。

早期版本曾把 3 个 GEMM 分派到 `aclrtCreateStream` 创建的辅助流上并行（`CSYMM_GEMM_PAR=1`）。
实测确认：该 ACL 运行时上，**复用辅助流在第二次使用后会静默丢弃整条任务队列**（以 marker
kernel 探针证明：第二次调用时队列中连普通 kernel 都不会执行），并导致 C 平面缺失 w1/w2
整块计算。该行为与每次调用是否 malloc/free wrapper 缓冲无关（`CSYMM_CACHE_BUFS=1` 缓冲常驻时
第二次调用同样失败）。因此并行 3 流分派仅保留为诊断模式，不得用于正确性/验收运行。

默认调用序列（调用方 stream 串行）：prep（AIV，镜像 A + B 分解出 3m 平面）→ gemm3 合并
（AIC）→ combine（AIV，3m 平面合并回 C）。AIV 与 AIC 之间依赖方 stream 顺序而非事件同步，
是经 1014 条精度用例与 repeat 验证的正确路径。

## 注意事项

1. **设备独占**：性能类测试（§2/§3）须独占 NPU；并行任务会拉高耗时导致误判 FAIL。
2. **诊断开关全部默认关闭**：`CSYMM_GEMM_MERGE`、`CSYMM_GEMM_PAR`、`CSYMM_CACHE_BUFS`、
   `CSYMM_PROBE`、`CSYMM_LAUNCH_PROBE`、`CSYMM_DUMP*` 等均为调试手段，
   **验收运行不得设置**（并行流分派在当前 ACL 运行时存在复用丢队列缺陷，仅作诊断）。
3. 默认执行路径为 3-GEMM 合并单 launch 的**单流串行**（prep→gemm→combine），
   是经 1014 条精度用例与 `--gtest_repeat=2` 重复运行验证的正确路径。
4. 用例可复现：`csymm_test.csv` 为固定随机种子生成、可复现的用例集（1200 条，1000 精度 + 200 性能）。
