# Rotm算子

## 算子概述

Modified Givens Rotation 算子，对向量 x 和 y 应用 modified Givens 旋转。

数学表达式：

```
[x[i]; y[i]] := H * [x[i]; y[i]]
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSrotm | 实数向量 Modified Givens 旋转 |

## 算子执行接口

### aclblasSrotm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSrotm(aclblasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *param)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量长度，Host 内存 |
| x | 输入/输出 | float*（FP32） | 输入/输出向量 x，Device 内存 |
| incx | 输入 | int | x 的步长（可正可负），Host 内存 |
| y | 输入/输出 | float*（FP32） | 输入/输出向量 y，Device 内存 |
| incy | 输入 | int | y 的步长（可正可负），Host 内存 |
| param | 输入 | const float*（FP32） | 5 个元素的旋转参数数组：[flag, h11, h21, h12, h22]，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0
- incy != 0