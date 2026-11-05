# Symv算子

## 算子概述

symv (Symmetric Matrix-Vector Multiplication) 实现对称矩阵与向量的乘法运算。该算子针对对称矩阵的存储特性进行优化，仅存储上三角或下三角部分，通过对称性推断未存储部分，高效完成矩阵与向量的乘加运算。

数学表达式：

```
y = alpha * A * x + beta * y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsymv | 单精度对称矩阵-向量乘法 |
| aclblasCsymv | 单精度复数对称矩阵-向量乘法 |

## 算子执行接口

### aclblasSsymv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsymv(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定矩阵 A 存储上三角（ACLBLAS_UPPER）或下三角（ACLBLAS_LOWER），Host 内存 |
| n | 输入 | int | 对称矩阵 A 的行数和列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 用于乘法的 float 标量，Host/Device 内存 |
| A | 输入 | const float*（FP32） | 对称矩阵 float 数组，维度为 lda x n，仅存储 uplo 指定的三角部分，Device 内存 |
| lda | 输入 | int | 用于存储矩阵 A 的二维数组的主维，lda >= max(1, n)，Host 内存 |
| x | 输入 | const float*（FP32） | float 向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，incx != 0，Host 内存 |
| beta | 输入 | const float*（FP32） | 用于乘法的 float 标量。如果 beta == 0，则 y 不必是有效输入，Host/Device 内存 |
| y | 输入/输出 | float*（FP32） | float 向量，包含 n 个元素。输入为初始 y 值，输出为计算结果，Device 内存 |
| incy | 输入 | int | y 中连续元素之间的步长，incy != 0，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0
- incy != 0
- lda >= max(1, n)

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

class AclContext {
public:
    explicit AclContext(int deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct AclrtMemDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct AclblasHandleDeleter {
    void operator()(aclblasHandle_t handle) const
    {
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
    }
};

int aclblasSsymvTest(AclContext& ctx)
{
    constexpr int n = 3;
    constexpr int lda = 3;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t aSize = lda * n * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);
    constexpr size_t ySize = n * sizeof(float);
    float alpha = 1.0f;
    float beta = 1.0f;

    // A 按列主序存储，此处存储上三角部分。
    float hA[lda * n] = {
        1.0f, 0.0f, 0.0f,
        2.0f, 5.0f, 0.0f,
        3.0f, 6.0f, 9.0f
    };
    float hX[n] = {1.0f, 2.0f, 3.0f};
    float hY[n] = {10.0f, 20.0f, 30.0f};

    void *rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    void *rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dY(rawY);

    aclRet = aclrtMemcpy(dA.get(), aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dX.get(), xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dY.get(), ySize, hY, ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasSsymv(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_UPPER, n, &alpha,
        static_cast<float*>(dA.get()), lda, static_cast<float*>(dX.get()), incx, &beta,
        static_cast<float*>(dY.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hY, ySize, dY.get(), ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSsymvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

### aclblasCsymv

#### 产品支持情况

- Ascend 950PR：支持
- 其他产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCsymv(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 上下文句柄，携带执行 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | `ACLBLAS_UPPER` 使用 A 的上三角，`ACLBLAS_LOWER` 使用 A 的下三角；未存储部分按对称关系读取且不取共轭，Host 内存 |
| n | 输入 | int | A 的行列数以及 x、y 的逻辑长度，n >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex* | 复数乘数，Host/Device 内存，不可为空 |
| A | 输入 | const aclblasComplex* | lda x n 的列主序复数对称矩阵；alpha 非零时需有效，Device 内存 |
| lda | 输入 | int | A 的前导维度，lda >= max(1, n)，Host 内存 |
| x | 输入 | const aclblasComplex* | 逻辑长度为 n 的复数向量；alpha 非零时需有效，Device 内存 |
| incx | 输入 | int | x 的元素步长，支持正负值且不能为 0，Host 内存 |
| beta | 输入 | const aclblasComplex* | 复数乘数，Host/Device 内存，不可为空；beta 为零时不读取 y 的原值 |
| y | 输入/输出 | aclblasComplex* | 逻辑长度为 n 的复数向量，原地写回，Device 内存 |
| incy | 输入 | int | y 的元素步长，支持正负值且不能为 0，Host 内存 |

#### 约束说明

- 计算 `y = alpha * A * x + beta * y`，A 满足 `A = A^T`，不是厄米特矩阵。
- n 为 0 时直接返回成功，不访问 alpha、beta、A、x、y。
- alpha 为 `(0, 0)` 且 beta 为 `(1, 0)` 时直接返回成功且不写 y。
- alpha 为 `(0, 0)` 时不读取 A 和 x；beta 为 `(0, 0)` 时不读取 y 的输入值。
- uplo 非法时返回 `ACLBLAS_STATUS_INVALID_ENUM`；其他非法数值或必要空指针返回 `ACLBLAS_STATUS_INVALID_VALUE`。
