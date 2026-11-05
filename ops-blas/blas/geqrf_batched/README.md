# GeqrfBatched算子

## 算子概述

GeqrfBatched（批量 QR 分解）算子对一批矩阵使用 Householder 反射进行 QR 分解。对每个批次 j（j = 0, 1, ..., batchSize-1），将 m x n 实矩阵 A[j] 分解为 A[j] = Q[j] * R[j]。属于 LAPACK 风格的批量分解算子。

数学表达式：

```
A[j] = Q[j] * R[j],   j = 0, 1, ..., batchSize-1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgeqrfBatched | 单精度批量 QR 分解 |

## 算子执行接口

### aclblasSgeqrfBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgeqrfBatched(aclblasHandle_t handle, int m, int n, float *const Aarray[], int lda, float *const TauArray[], int *info, int batchSize)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| m | 输入 | int | 每个矩阵 Aarray[i] 的行数，要求 m >= 0，Host 内存 |
| n | 输入 | int | 每个矩阵 Aarray[i] 的列数，要求 n >= 0，Host 内存 |
| Aarray | 输入/输出 | float *const []（FP32） | 设备端指针数组，每个元素指向一个 m x n 列主序矩阵。输入时包含原始矩阵，输出时下三角存储 Householder 向量 v，上三角存储 R，Device 内存 |
| lda | 输入 | int | 矩阵 Aarray[i] 的前导维度，要求 lda >= max(1, m)，Host 内存 |
| TauArray | 输出 | float *const []（FP32） | 设备端指针数组，每个元素指向维度 min(m, n) 的向量，存储 Householder 标量因子 tau，Device 内存 |
| info | 输出 | int* | Host 端指针，指向单个 int。0 = 成功，Host 内存 |
| batchSize | 输入 | int | Aarray 中包含的指针数量（批次数），要求 batchSize >= 0，Host 内存 |

#### 约束说明

- m >= 0, n >= 0, batchSize >= 0
- lda >= max(1, m)

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
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

int aclblasSgeqrfBatchedTest(AclContext& ctx)
{
    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int lda = 2;
    constexpr int batchSize = 1;
    constexpr int minMN = m < n ? m : n;
    constexpr size_t matBytes = static_cast<size_t>(lda) * n * sizeof(float);
    constexpr size_t tauBytes = static_cast<size_t>(minMN) * sizeof(float);
    constexpr size_t ptrBytes = sizeof(float*) * batchSize;

    // A 按列主序存储：[[1,3],[2,4]]
    float hA[lda * n] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hTau[minMN] = {0.0f};
    int info = 0;

    void *rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    void *rawTau = nullptr;
    aclRet = aclrtMalloc(&rawTau, tauBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dTau(rawTau);

    void *rawAPtrs = nullptr;
    aclRet = aclrtMalloc(&rawAPtrs, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAPtrs(rawAPtrs);

    void *rawTauPtrs = nullptr;
    aclRet = aclrtMalloc(&rawTauPtrs, ptrBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dTauPtrs(rawTauPtrs);

    aclRet = aclrtMemcpy(dA.get(), matBytes, hA, matBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 构造设备端指针数组：将每个设备指针拷贝到 Device 上的指针数组
    float* hAPtr = static_cast<float*>(dA.get());
    aclRet = aclrtMemcpy(dAPtrs.get(), ptrBytes, &hAPtr, ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    float* hTauPtr = static_cast<float*>(dTau.get());
    aclRet = aclrtMemcpy(dTauPtrs.get(), ptrBytes, &hTauPtr, ptrBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasSgeqrfBatched(
        static_cast<aclblasHandle_t>(handle.get()), m, n,
        reinterpret_cast<float* const*>(dAPtrs.get()), lda,
        reinterpret_cast<float* const*>(dTauPtrs.get()), &info, batchSize);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hA, matBytes, dA.get(), matBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(hTau, tauBytes, dTau.get(), tauBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    printf("info = %d\n", info);
    for (int i = 0; i < lda * n; i++) {
        printf("A[%d] = %f\n", i, hA[i]);
    }
    for (int i = 0; i < minMN; i++) {
        printf("tau[%d] = %f\n", i, hTau[i]);
    }

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgeqrfBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```