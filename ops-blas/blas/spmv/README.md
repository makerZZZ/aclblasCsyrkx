# Spmv算子

## 算子概述

Spmv（Symmetric Packed Matrix-Vector Multiplication）算子实现了对称压缩矩阵与向量的乘法运算。该算子针对对称矩阵的存储特性进行了优化，采用压缩存储格式以节省内存空间，并高效完成矩阵与向量的乘加运算。

数学表达式：

```
y = alpha * A * x + beta * y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSspmv | 单精度浮点对称压缩矩阵-向量乘法 |

## 算子执行接口

### aclblasSspmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSspmv(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，Host 内存 |
| n | 输入 | int | 方阵阶数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha 的指针，Host 内存 |
| AP | 输入 | const float*（FP32） | 对称压缩矩阵，共 n(n+1)/2 个元素，Device 内存 |
| x | 输入 | const float*（FP32） | 输入向量，n 个元素，Device 内存 |
| incx | 输入 | int | x 的步长，incx != 0（可正可负），Host 内存 |
| beta | 输入 | const float*（FP32） | 标量 beta 的指针，Host 内存 |
| y | 输入/输出 | float*（FP32） | 输入/输出向量，n 个元素，Device 内存 |
| incy | 输入 | int | y 的步长，incy != 0（可正可负），Host 内存 |

#### 约束说明

- n >= 0
- incx != 0, incy != 0

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

int aclblasSspmvTest(AclContext& ctx)
{
    constexpr int n = 3;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t apSize = n * (n + 1) / 2 * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);
    constexpr size_t ySize = n * sizeof(float);
    float alpha = 1.0f;
    float beta = 1.0f;

    float hAP[n * (n + 1) / 2] = {1.0f, 2.0f, 5.0f, 3.0f, 6.0f, 9.0f};
    float hX[n] = {1.0f, 2.0f, 3.0f};
    float hY[n] = {10.0f, 20.0f, 30.0f};

    void *rawAP = nullptr;
    auto aclRet = aclrtMalloc(&rawAP, apSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAP(rawAP);

    void *rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dY(rawY);

    aclRet = aclrtMemcpy(dAP.get(), apSize, hAP, apSize, ACL_MEMCPY_HOST_TO_DEVICE);
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

    blasRet = aclblasSspmv(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_UPPER, n, &alpha,
        static_cast<float*>(dAP.get()), static_cast<float*>(dX.get()), incx, &beta,
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

    ret = aclblasSspmvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```
