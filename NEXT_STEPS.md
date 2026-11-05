# aclblasCsyrkx 后续工作指引

本文件说明：本机已完成什么、上 NPU 机时前后要跑哪些测试、验收要交哪些材料。

---

## 0. 当前状态（本机，无 NPU）

已完成：

- 算子实现：`ops-blas/blas/syrkx/arch35/csyrkx_{host,kernel}.cpp`、
  `csyrkx_kernel.h`、`csyrkx_tiling_data.h`
- 接口声明：`include/cann_ops_blas.h` 新增 `aclblasCsyrkx`
- 构建注册：`cmake/asc_devkit_version.cmake`（CSYRKX 门控）、
  `blas/CMakeLists.txt`（`_GEMM_KERNEL_DEP_OPS` 增加 csyrkx）
- 测试工程：`test/syrkx/csyrkx/`（param/golden/wrapper/test.cpp/CSV/README）
- 文档：`blas/syrkx/README.md` 增加 Csyrkx 章节；`docs/zh/api_list.md` 增加条目
- 编译验证（ascend950 / dav-3510）：
  - `build.sh --ops=csyrkx --soc=ascend950` 通过（含 device kernel 编译与安装）
  - GTest 目标 `csyrkx_test`、`blas_all_test` 链接通过
- CPU 预验证：`local_verify/run_cpu_check.sh`
  - 1200 条 CSV：返回值 0 失配；golden vs 独立双精度参考 0 失配；
    kernel 4M 公式浮点模拟 vs golden 0 失配（925 小中规模 + 226 大规模，
    仅 30 条 n≈4000 性能用例跳过）
  - 真实 host 入口的负向/quick-return 校验 16/16 通过

尚未完成（必须有 NPU）：

- kernel 真实执行、精度、性能
- Device 标量指针、真实多核/同步时序
- 最终性能调优（若未达 0.4×/0.5× GPU）

---

## 1. 上机前：把仓库和依赖带到云端

1. 将整个 `ops-blas` 仓库（或你的个人 fork 分支）推到 gitcode 私仓；
2. 云端装好 CANN 9.1.0（任务书要求）与 `libblas-dev`（或 OpenBLAS）+ GTest：
   ```bash
   sudo apt-get install libblas-dev libgtest-dev   # 或 ./install_deps.sh
   ```
3. 确认 CANN 环境与 NPU：
   ```bash
   source ${ASCEND_HOME_PATH}/set_env.sh
   npu-smi info
   ```
4. 注意：本机 CANN 是 9.2.1；任务书验收环境是 9.1.0。上机后应先在 9.1.0
   上重新编译一遍，确认 asc-devkit 版本门控/头文件兼容（仓库已按 >=9.1 设计）。

---

## 2. NPU 端测试流程（按顺序跑，省机时）

### 2.1 编译

```bash
cd ops-blas
source ${ASCEND_HOME_PATH}/set_env.sh
bash build.sh --ops=csyrkx --soc=ascend950
```

单算子构建会同时编译 `gemm_kernel.cpp`（已在 `_GEMM_KERNEL_DEP_OPS` 登记）。

### 2.2 冒烟：单元/负向用例先跑

```bash
./build/test/syrkx/csyrkx/csyrkx_test --gtest_filter='*NullHandle*'
./build/test/syrkx/csyrkx/csyrkx_test --gtest_filter='*TC_L0*'
./build/test/syrkx/csyrkx/csyrkx_test --gtest_filter='*TC_ED*'
```

重点看：

- `TC_ED_370 alpha0_a_null`、`TC_ED_367 n0_a_b_null`、`TC_ED_368 n0_c_null`
  是否 SUCCESS（host 已用真实入口验证，上机确认 Device 路径）；
- `TC_ED_376/377` 非法枚举是否 `INVALID_VALUE`；
- `TC_ED_366 beta0_c_nan`：C 为 NaN、beta=0 时 uplo 三角应为有限值。

### 2.3 全量精度

```bash
./build/test/syrkx/csyrkx/csyrkx_test                 # 1200 条
# 或按前缀分批，便于定位：
./build/test/syrkx/csyrkx/csyrkx_test --gtest_filter='*TC_SQ*:*TC_MN*:*TC_AB*'
./build/test/syrkx/csyrkx/csyrkx_test --gtest_filter='*TC_LD*:*TC_FL*:*TC_CV*'
./build/test/syrkx/csyrkx/csyrkx_test --gtest_filter='*TC_EX*'
```

也可用任务提供的脚本（它按 CSV 列序驱动并汇总）：

```bash
cd test_cases
python3 verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --csv ./csyrkx_test.csv
```

> 注意：`test_cases/README.md` 写的安装路径是 `test/syrkx/csyrkx/<arch>/`，
> 与仓库现行布局 `test/syrkx/csyrkx/arch35/` 一致。
> `test_cases/README.md` 的“补充说明”称 OP_C 语义待确认——任务书已明确
> OP_C == OP_T（不取共轭），本实现按此执行；若脚本的 golden 仍按共轭构造，
> 需要把脚本里 trans=C 的 `CblasConjTrans` 改成 `CblasTrans`。

### 2.4 性能采集

按任务书 §3.3：msprof op、5 次 warmup、10 次有效采样取平均。

```bash
# 单算子 task duration
msprof op --application="./build/test/syrkx/csyrkx/csyrkx_test --gtest_filter=*TC_PF*" \
          --output=./prof_csyrkx
# 在 prof_csyrkx/OPPROF_*/OpBasicInfo.csv 中按 kernel 名统计 Task Duration(us)
```

或使用任务脚本：

```bash
python3 verify_performance.py --repo /path/to/ops-blas --soc ascend950 --timeout 3600
```

判定：

- 每个 case：`GPU 标杆耗时 / NPU 实测耗时 >= 0.4`；
- 全部 `TC_PF_*` 倍率算术平均 `>= 0.5`；
- 标杆文件：`test_cases/gpu_baseline.csv`（H100 / CUDA 12.4）。

若性能不达标，优先优化点（按收益排序）：

1. **只算 uplo 三角**：当前 4 个 GEMM 都算完整 n×n。可在 GEMM 侧按三角分块
   （参考 `blas/syrk/arch35/csyrk_cube_kernel.cpp` 的 `triangleMode`）省约一半算力；
2. **融合 4M**：把 4 次 GEMM 合成一次 kernel（参考 `blas/her2k` 的
   `cher2k_fused4_mix_do`），减少 3 次 launch 与 temp 读写；
3. **小尺寸 SIMT 直算**：n 很小（≤64）时 7 次 launch 开销占主导，可参考
   `blas/syrk/arch35/csyrk_direct_kernel.cpp` 用单 kernel SIMT 直接算 uplo；
4. **combine 核数**：小 n 时减少 AIV 核数（参考 cher2k 的 TuneCombineCoreNum）。

### 2.5 回归与确定性

- 同一 case 连续跑两次，`--gtest_repeat=2` 或对比输出，确认逐位一致；
- 跑一遍 `blas_all_test`（若开启）确认新增 TENSOR_API_OPS 条目未影响其他算子：
  ```bash
  bash build.sh --run --blas-all --smoke
  ```

---

## 3. 提交验收前要准备的材料

任务书 §4 清单：

| 序号 | 交付件 | 本仓库对应产物 / 待办 |
|------|--------|------------------------|
| 1 | 算子设计文档 | `deliverables/design_doc.md`（草案）；按模板
https://gitcode.com/cann/cann-competitions/blob/master/04_tasks/01_community-task-2026/resources/design_template.md
整理后，向 `cann-competitions` 的 `04_tasks/01_community-task-2026/tasklist` 提 PR |
| 2 | 自测用例及测试代码 | `test/syrkx/csyrkx/`（含 README 复现步骤）+ `test_cases/`（任务提供）+ `local_verify/`（CPU 预验证） |
| 3 | 自测报告 | 按腾讯文档模板填写：用例参数、精度对比结果及截图、性能数据及截图。骨架见 `deliverables/self_test_report.md` |
| 4 | 待验收代码地址 | 个人 gitcode 仓 + 分支 + 算子目录；邀请 `Ascend-CANN` 为开发者；算子 README 见 `blas/syrkx/README.md` |

提交包（任务书给的模板 zip）通常还需要：

- 自验证步骤说明（本文件 §2 + `test/syrkx/csyrkx/README.md`）
- 精度/性能日志原始文件（gtest 输出、msprof OpBasicInfo.csv）
- 代码 commit 链接（如字符受限，放到 .txt 附件）
- 私仓开发者邀请截图

### 设计文档 PR 流程（简要）

1. fork `cann-competitions`；
2. 在 `04_tasks/01_community-task-2026/tasklist/<你的任务目录>/` 下放设计文档；
3. 按模板补齐：算子背景、接口、算法、tiling/流水、精度方案、测试方案、风险；
4. 提 PR，等待评审合入（合入后再提交验收）。

---

## 4. 常见坑（上机前先看）

1. **OP_C 语义**：必须是 `Trans`，不是 `ConjTrans`。任务提供的
   `test_cases/README.md` 有一句“待研发确认”的旧说明，以任务书 §2.1/§2.2 为准。
2. **beta 是复数**：`const aclblasComplex* beta`，与 cherkx（实数 beta）不同；
   CSV 有 `beta_real`/`beta_imag` 两列。
3. **A/B 空指针条件**：`alpha==0` 或 `k==0` 时允许为空（CSV `alpha0_a_null`）；
   不要照搬“一律非空”的写法，否则该用例会失败。
4. **C 空指针**：`n>0` 时一律 `INVALID_VALUE`（任务书口径），
   与 cherkx 的“beta==0 且 C 为空可 SUCCESS”不同。
5. **n==0**：先做完全部参数校验（枚举、前导维、标量指针）再 quick return。
6. **beta=0 的 NaN 语义**：uplo 三角不得读取 C 旧值（`beta0_c_nan`）。
7. **非 uplo 三角**：逐位不变，包括 NaN/Inf；LOWER 对角列的整列写回必须先恢复
   非 uplo 段（kernel 已处理）。
8. **workspace**：4 个 temp + 4 个实矩阵；n=4096、k=4096 时约 537MB，
   未超 2GiB；若用户设置 workspace 过小会返回 `ALLOC_FAILED`。
9. **CANN 9.1.0 兼容**：本机用 9.2.1 编译通过；9.1.0 若 `asc-config.cmake`
   路径不同，构建时设置 `ASC_DIR`，或按仓库 README 的 `install_deps.sh` 处理。
