# Hpr算子

## 算子概述

hpr (Hermitian Packed Rank-1 Update) 实现 Hermitian 矩阵 packed 格式的秩-1 更新。该算子将 `alpha * x * x^H` 加到 packed Hermitian 矩阵的指定三角区域，并将对角线虚部置零。

数学表达式：

```
A := alpha * x * x^H + A
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasChpr | 单精度复数 Hermitian packed 秩-1 更新 |

## 算子执行接口

### aclblasChpr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasChpr(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float *alpha, const aclblasComplex *x, int incx, aclblasComplex *ap)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，指定更新的三角区域，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 实数标量乘数指针，Host 内存 |
| x | 输入 | const aclblasComplex* | 输入向量指针，长度至少 1 + (n-1) * abs(incx)，Device 内存 |
| incx | 输入 | int | x 的元素间步长，incx != 0，Host 内存 |
| ap | 输入/输出 | aclblasComplex* | packed Hermitian 矩阵指针，长度 n*(n+1)/2，Device 内存 |

#### 约束说明

- n >= 0
- incx != 0
- alpha 为实数标量

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    const int n = 4;
    float alpha = 1.0f;
    aclblasComplex x[4] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}};
    aclblasComplex ap[10] = {};

    aclblasComplex* dX = nullptr;
    aclblasComplex* dAp = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&dX), sizeof(x), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&dAp), sizeof(ap), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dX, sizeof(x), x, sizeof(x), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dAp, sizeof(ap), ap, sizeof(ap), ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasChpr(handle, ACLBLAS_UPPER, n, &alpha, dX, 1, dAp);

    aclrtFree(dX);
    aclrtFree(dAp);
    aclblasDestroy(handle);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
