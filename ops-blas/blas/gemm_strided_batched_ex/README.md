# GemmStridedBatchedEx算子

## 算子概述

GemmStridedBatchedEx算子实现一组参数相同、矩阵首地址等间隔的通用矩阵乘法，矩阵采用列主序存储。

数学表达式：

```text
C_i = alpha * op(A_i) * op(B_i) + beta * C_i, i = 0, 1, ..., batchCount - 1
```

其中，`A_i`、`B_i` 和 `C_i` 分别表示第 `i` 批矩阵，相邻批次矩阵首地址的元素间隔分别由 `strideA`、`strideB` 和 `strideC` 指定。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasGemmStridedBatchedEx | 支持多种数据类型、矩阵转置和等间隔批处理的通用矩阵乘法 |

## 算子执行接口

### aclblasGemmStridedBatchedEx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasGemmStridedBatchedEx(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, const void* alpha, const void* A, aclDataType Atype, int lda, int64_t strideA, const void* B, aclDataType Btype, int ldb, int64_t strideB, const void* beta, void* C, aclDataType Ctype, int ldc, int64_t strideC, int batchCount, aclblasComputeType_t computeType, aclblasGemmAlgo_t algo)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas库上下文句柄，携带stream和workspace，Host内存 |
| transa | 输入 | aclblasOperation_t | 矩阵A的操作类型，支持`ACLBLAS_OP_N`（不转置）、`ACLBLAS_OP_T`（转置）和`ACLBLAS_OP_C`（共轭转置），Host内存 |
| transb | 输入 | aclblasOperation_t | 矩阵B的操作类型，支持`ACLBLAS_OP_N`（不转置）、`ACLBLAS_OP_T`（转置）和`ACLBLAS_OP_C`（共轭转置），Host内存 |
| m | 输入 | int | 矩阵`op(A)`和C的行数，Host内存 |
| n | 输入 | int | 矩阵`op(B)`和C的列数，Host内存 |
| k | 输入 | int | 矩阵`op(A)`的列数和矩阵`op(B)`的行数，Host内存 |
| alpha | 输入 | const void* | 乘积项的缩放因子，数据类型由`computeType`及输入类型决定，Host内存 |
| A | 输入 | const void* | 第一批矩阵A的首地址，Device内存 |
| Atype | 输入 | aclDataType | 矩阵A的元素类型，Host内存 |
| lda | 输入 | int | 矩阵A的主维长度，Host内存 |
| strideA | 输入 | int64_t | 相邻批次矩阵A首地址的元素间隔，单位为A的元素个数，Host内存 |
| B | 输入 | const void* | 第一批矩阵B的首地址，Device内存 |
| Btype | 输入 | aclDataType | 矩阵B的元素类型，Host内存 |
| ldb | 输入 | int | 矩阵B的主维长度，Host内存 |
| strideB | 输入 | int64_t | 相邻批次矩阵B首地址的元素间隔，单位为B的元素个数，Host内存 |
| beta | 输入 | const void* | 原矩阵C的缩放因子，数据类型由`computeType`及输入类型决定，Host内存 |
| C | 输入/输出 | void* | 第一批矩阵C的首地址，输入为累加前矩阵，输出为计算结果，Device内存 |
| Ctype | 输入 | aclDataType | 矩阵C的元素类型，Host内存 |
| ldc | 输入 | int | 矩阵C的主维长度，Host内存 |
| strideC | 输入 | int64_t | 相邻批次矩阵C首地址的元素间隔，单位为C的元素个数，Host内存 |
| batchCount | 输入 | int | 批次数量，Host内存 |
| computeType | 输入 | aclblasComputeType_t | 计算精度类型，Host内存 |
| algo | 输入 | aclblasGemmAlgo_t | GEMM算法类型，Host内存 |

#### 约束说明

- `m`、`n`、`k`和`batchCount`必须大于等于0。
- `lda >= max(1, transa == ACLBLAS_OP_N ? m : k)`，`ldb >= max(1, transb == ACLBLAS_OP_N ? k : n)`，`ldc >= max(1, m)`。
- 当`batchCount > 1`时，`strideC`不能为0；需要读取多批矩阵A和B时，`strideA`和`strideB`不能为0。三个stride均以对应矩阵的数据类型元素个数为单位，支持负值。
- 当`m == 0`、`n == 0`或`batchCount == 0`时，接口直接返回成功；当`k == 0`或`alpha == 0`时，A和B允许为空指针。其他非空计算中，A、B和C必须为有效的Device指针，`alpha`和`beta`必须为有效的Host指针。
- 支持的数据类型组合如下。Complex-FP32组合的`alpha`和`beta`类型为`aclblasComplex`；`ACLBLAS_COMPUTE_16F`及其PEDANTIC模式对应FP16标量，`ACLBLAS_COMPUTE_32I`及其PEDANTIC模式对应INT32标量，其他非复数组合对应FP32标量。

  | computeType | Atype | Btype | Ctype |
  |-------------|-------|-------|-------|
  | `ACLBLAS_COMPUTE_16F` / `ACLBLAS_COMPUTE_16F_PEDANTIC` | `ACL_FLOAT16` | `ACL_FLOAT16` | `ACL_FLOAT16` |
  | `ACLBLAS_COMPUTE_32F` / `ACLBLAS_COMPUTE_32F_PEDANTIC` | `ACL_FLOAT16` | `ACL_FLOAT16` | `ACL_FLOAT16` / `ACL_FLOAT` |
  | `ACLBLAS_COMPUTE_32F` / `ACLBLAS_COMPUTE_32F_PEDANTIC` | `ACL_BF16` | `ACL_BF16` | `ACL_BF16` / `ACL_FLOAT` |
  | `ACLBLAS_COMPUTE_32F` / `ACLBLAS_COMPUTE_32F_PEDANTIC` / `ACLBLAS_COMPUTE_32F_FAST_16F` / `ACLBLAS_COMPUTE_32F_FAST_16BF` / `ACLBLAS_COMPUTE_32F_FAST_TF32` | `ACL_FLOAT` | `ACL_FLOAT` | `ACL_FLOAT` |
  | `ACLBLAS_COMPUTE_32I` / `ACLBLAS_COMPUTE_32I_PEDANTIC` | `ACL_INT8` | `ACL_INT8` | `ACL_INT32` |
  | `ACLBLAS_COMPUTE_32F` / `ACLBLAS_COMPUTE_32F_PEDANTIC` | `ACL_INT8` | `ACL_INT8` | `ACL_FLOAT` |
  | `ACLBLAS_COMPUTE_32F` / `ACLBLAS_COMPUTE_32F_PEDANTIC` / `ACLBLAS_COMPUTE_32F_FAST_16F` / `ACLBLAS_COMPUTE_32F_FAST_16BF` / `ACLBLAS_COMPUTE_32F_FAST_TF32` | `ACL_COMPLEX64` | `ACL_COMPLEX64` | `ACL_COMPLEX64` |
  | `ACLBLAS_COMPUTE_32F` / `ACLBLAS_COMPUTE_32F_PEDANTIC` | `ACL_FLOAT8_E4M3FN` / `ACL_FLOAT8_E5M2` | `ACL_FLOAT8_E4M3FN` / `ACL_FLOAT8_E5M2` | `ACL_FLOAT16` / `ACL_FLOAT` |

- 不支持FP64、Complex-FP64和复数INT8组合，不支持的组合返回`ACLBLAS_STATUS_NOT_SUPPORTED`。
- `INT8×INT8→INT32`要求A和B的每批地址以及`lda`、`ldb`、`strideA`、`strideB`均满足4字节对齐。
- `INT8×INT8→INT32/FP32`采用INT32 Cube累加，要求`k <= 131071`。
- FP16输入支持`ACLBLAS_GEMM_DEFAULT`和`ACLBLAS_GEMM_ALGO0`～`ACLBLAS_GEMM_ALGO7`；其他已支持类型仅支持`ACLBLAS_GEMM_DEFAULT`和`ACLBLAS_GEMM_ALGO0`。
- 通用后处理所需workspace大小为`batchCount * m * n * sizeof(float)`字节；`ACLBLAS_GEMM_ALGO6`还需乘以内部`splitK`。workspace不足时返回`ACLBLAS_STATUS_ALLOC_FAILED`。

#### 调用示例

以下示例使用FP32数据完成两个2×2矩阵批次的乘法。示例代码仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

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
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

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
    int32_t deviceId_;
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

int aclblasGemmStridedBatchedExTest(AclContext& ctx)
{
    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int k = 2;
    constexpr int lda = 2;
    constexpr int ldb = 2;
    constexpr int ldc = 2;
    constexpr int batchCount = 2;
    constexpr int64_t strideA = 4;
    constexpr int64_t strideB = 4;
    constexpr int64_t strideC = 4;
    constexpr size_t matrixElements = static_cast<size_t>(batchCount) * 4;
    constexpr size_t matrixBytes = matrixElements * sizeof(float);

    // 每四个连续元素表示一个列主序2×2矩阵。
    const std::vector<float> hA = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f};
    const std::vector<float> hB = {1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 7.0f, 6.0f, 8.0f};
    std::vector<float> hC(matrixElements, 0.0f);

    void* rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, matrixBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    void* rawB = nullptr;
    aclRet = aclrtMalloc(&rawB, matrixBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dB(rawB);

    void* rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, matrixBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dC(rawC);

    aclRet = aclrtMemcpy(dA.get(), matrixBytes, hA.data(), matrixBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dB.get(), matrixBytes, hB.data(), matrixBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dC.get(), matrixBytes, hC.data(), matrixBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<_aclblas_handle, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(handle.get(), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    blasRet = aclblasGemmStridedBatchedEx(
        handle.get(), ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k, &alpha, dA.get(), ACL_FLOAT, lda, strideA,
        dB.get(), ACL_FLOAT, ldb, strideB, &beta, dC.get(), ACL_FLOAT, ldc, strideC, batchCount,
        ACLBLAS_COMPUTE_32F, ACLBLAS_GEMM_DEFAULT);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hC.data(), matrixBytes, dC.get(), matrixBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    for (int batch = 0; batch < batchCount; ++batch) {
        const size_t offset = static_cast<size_t>(batch) * static_cast<size_t>(strideC);
        std::printf("batch %d: [%f, %f; %f, %f]\n", batch, hC[offset], hC[offset + 2], hC[offset + 1],
                    hC[offset + 3]);
    }
    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasGemmStridedBatchedExTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```
