# Gerc 算子

## 算子概述

`aclblasCgerc` 完成单精度复数矩阵的共轭秩 1 更新：

```text
A := alpha * x * conj(y)^T + A
```

矩阵 `A` 使用列主序存储。对每个 `0 <= i < m`、`0 <= j < n`，更新公式为：

```text
A(i,j) := A(i,j) + alpha * x(i) * conj(y(j))
```

## 算子执行接口

### aclblasCgerc

#### 产品支持情况

- Ascend 950PR：支持
- Ascend 950DT：暂未验证
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCgerc(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* x, int incx, const aclblasComplex* y, int incy,
    aclblasComplex* A, int lda);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|-----------|----------|------|
| handle | 输入 | aclblasHandle_t | aclBLAS 句柄，不能为 `nullptr` |
| m | 输入 | int | 矩阵 `A` 的行数，要求 `m >= 0` |
| n | 输入 | int | 矩阵 `A` 的列数，要求 `n >= 0` |
| alpha | 输入 | const aclblasComplex* | Host 侧复数标量，不能为 `nullptr` |
| x | 输入 | const aclblasComplex* | Device 侧复数向量，逻辑长度为 `m` |
| incx | 输入 | int | `x` 的元素步长，支持任意非零正、负整数 |
| y | 输入 | const aclblasComplex* | Device 侧复数向量，逻辑长度为 `n` |
| incy | 输入 | int | `y` 的元素步长，支持任意非零正、负整数 |
| A | 输入/输出 | aclblasComplex* | Device 侧列主序复数矩阵，原地更新 |
| lda | 输入 | int | `A` 的主维度，要求 `lda >= max(1, m)` |

当步长为负数时，接口遵循 BLAS 语义，从物理数组中对应的反向逻辑起点遍历。向量最少需要分配
`1 + (length - 1) * abs(inc)` 个复数元素。

#### 约束与行为

- 参数校验先于 quick return。
- `handle == nullptr` 优先返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；其余非法参数返回
  `ACLBLAS_STATUS_INVALID_VALUE`，即使随后本可 quick return 也一样。
- `incx/incy == 0` 非法；负步长不是 no-op，而是按 BLAS 反向逻辑起点参与计算。
- `lda` 的下限来自列主序矩阵每列的 `m` 个有效行，故为 `max(1,m)`；它与列数 `n` 无关。
- `m == 0`、`n == 0` 或 `alpha == (0, 0)` 时不启动计算并返回成功。
- 非空计算中，`x`、`y`、`A` 均不能为 `nullptr`。
- 计算异步提交到 `handle` 绑定的 stream；读取输出前调用 `aclrtSynchronizeStream`。
- 输入输出数据类型为 `COMPLEX64`，即相邻两个 FP32 分量依次表示实部和虚部。

#### Ascend 950PR 实现说明

- 本次新增 `arch35` 实现，复用公共头文件 `include/cann_ops_blas.h` 中已有的 `aclblasCgerc` 声明；
  接口没有 `uplo`、`trans`、`diag` 或 `side` 参数。
- 各核更新互斥的矩阵区域，直接原地写回 `A`，不使用额外 GM 工作区。SIMD 路径的临时数据由
  `TPipe` 在核内 UB 分配，SIMT 路径使用线程局部标量，因此无需申请或读取 handle 工作区。
- 连续布局且 `alpha=(1,0)`、行数满足向量对齐与 UB 容量约束时选择 SIMD 路径；其余输入走
  通用 SIMT 路径，包含负步长和 `lda` padding。选择条件由输入、设备核数与 UB 容量推导，
  不依赖测试用例编号或特定矩阵尺寸。
- SIMT 复数外积保留 `volatile` FP32 乘积中间量，在分量加减前形成独立求值边界，以保持
  非融合运算及溢出、Inf/NaN 传播语义；这与文件级 `-ffp-contract=off` 编译选项同时保留。

#### 调用示例

以下示例省略 Host/Device 数据拷贝和错误处理：

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);
    aclblasSetStream(handle, stream);

    constexpr int m = 4;
    constexpr int n = 4;
    constexpr int lda = m;
    aclblasComplex alpha{1.0F, 0.0F};
    aclblasComplex* dx = nullptr;
    aclblasComplex* dy = nullptr;
    aclblasComplex* dA = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&dx), m * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&dy), n * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&dA), lda * n * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);

    aclblasCgerc(handle, m, n, &alpha, dx, 1, dy, 1, dA, lda);
    aclrtSynchronizeStream(stream);

    aclrtFree(dx);
    aclrtFree(dy);
    aclrtFree(dA);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
