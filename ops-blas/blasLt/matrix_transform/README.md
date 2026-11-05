# MatrixTransform算子

## 算子概述

MatrixTransform 算子实现了 BLAS Lt 矩阵变换，核心运算为 C = alpha * op(A) + beta * op(B)。支持矩阵转置、标量缩放、加法，以及多种内存布局（order）与数据类型之间的转换。采用 aclBLASLt 描述符风格接口，复用 `aclblasLtHandle_t` / `aclblasLtMatrixLayout_t`，并配合独立的变换描述符 `aclblasLtMatrixTransformDesc_t` 使用。

数学表达式：

```
C = alpha * op(A) + beta * op(B)
```

- `op(·)` 由变换描述符的 `TRANSA` / `TRANSB` 指定：`N`（不转置）、`T`（转置）、`C`（共轭转置，本算子无复数类型，等价于 `T`）
- `alpha` / `beta` 为主机侧标量（pointer mode = HOST），按 `scaleType` 解释：浮点/FP8 路径为 `float`，整数路径为 `int32_t`，FP4 路径为 `bfloat16_t`
- `beta = 0` 且 `B = nullptr` 时退化为 `C = alpha * op(A)`；`alpha = 1` 且 `beta = 0` 时退化为纯布局/类型转换

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasLtMatrixTransform | 矩阵变换计算（转置 / 缩放 / 加法 / 布局与 dtype 转换） |

## 算子执行接口

### aclblasLtMatrixTransform

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasLtMatrixTransform(aclblasLtHandle_t lightHandle, aclblasLtMatrixTransformDesc_t transformDesc, const void* alpha, const void* A, aclblasLtMatrixLayout_t Adesc, const void* beta, const void* B, aclblasLtMatrixLayout_t Bdesc, void* C, aclblasLtMatrixLayout_t Cdesc, aclrtStream stream)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| lightHandle | 输入 | aclblasLtHandle_t | aclBLASLt 库上下文句柄，须经 `aclblasLtCreate` 创建，不可为 NULL，否则返回 `ACLBLAS_STATUS_NOT_INITIALIZED`，Host 内存 |
| transformDesc | 输入 | aclblasLtMatrixTransformDesc_t | 变换描述符，承载 scaleType / pointerMode / TRANSA / TRANSB，不可为 NULL，Host 内存 |
| alpha | 输入 | const void* | A 的缩放系数，按 scaleType 解释（浮点/FP8 路径为 float，整数路径为 int32_t，FP4 路径为 bfloat16_t），不可为 NULL，Host 内存 |
| A | 输入 | const void* | 输入矩阵 A，dtype/order/ld 由 Adesc 描述，Device 内存 |
| Adesc | 输入 | aclblasLtMatrixLayout_t | 矩阵 A 的 layout 描述符，Host 内存 |
| beta | 输入 | const void* | B 的缩放系数，按 scaleType 解释，不可为 NULL，beta=0 时 B/Bdesc 可为 NULL，Host 内存 |
| B | 输入 | const void* | 输入矩阵 B，beta=0 时可为 NULL，Device 内存 |
| Bdesc | 输入 | aclblasLtMatrixLayout_t | 矩阵 B 的 layout 描述符，beta=0 时可为 NULL，Host 内存 |
| C | 输出 | void* | 输出矩阵，dtype/order/ld 由 Cdesc 描述，Device 内存 |
| Cdesc | 输入 | aclblasLtMatrixLayout_t | 矩阵 C 的 layout 描述符，Host 内存 |
| stream | 输入 | aclrtStream | AscendCL 执行流，Host 内存 |

#### 约束说明

- rows、cols >= 0；rows=0 或 cols=0 时为空操作，直接返回 `ACLBLAS_STATUS_SUCCESS`
- 计算流程：输入按各自 dtype 读入，转换到 `scaleType` 域完成 `alpha·A + beta·B`，再转换为输出 dtype 并按 Cdesc 指定的 order 写回
- A / B / C 的 dtype 各自独立可设，支持 FP32 / FP16 / BF16 / INT8 / INT32 / FP8_E4M3FN / FP8_E5M2 / FP4_E2M1
- scaleType 须与 dtype 路径匹配：浮点路径与 FP8 路径传 `ACL_FLOAT`，整数路径传 `ACL_INT32`，FP4 路径传 `ACL_BF16`；跨 scaleType 路径转换（如浮点输入直接转整数输出）返回 `ACLBLAS_STATUS_NOT_SUPPORTED`
- 同 scaleType 路径内支持任意 in → out 转换；FP8 经 float 与 FP32/FP16/BF16 互转；FP4 经 bf16 与 BF16/FP32 互转
- order 支持 COL / ROW / COL32 / COL4_4R2_8C / COL32_2R_4R4；COL / ROW / COL32 支持全部 dtype 与全部 op（N/T，C 等价 T）
- 复杂量化布局 COL4_4R2_8C / COL32_2R_4R4 仅支持 INT8 / INT32 / FP8 / FP4 量化路径 dtype；纯浮点 dtype 与复杂量化布局组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`
- transA / transB 支持 N / T（C 等价 T）；op(A)、op(B) 与 C 的逻辑维度须相容
- batchCount 当前仅支持 1；batchCount > 1 返回 `ACLBLAS_STATUS_NOT_SUPPORTED`
- leading dimension（ld，元素数）下界：COL 时 ld >= rows；ROW 时 ld >= cols；COL32 / COL4_4R2_8C / COL32_2R_4R4 按对应复合 tile 步长计算
- FP4（fp4x2 packed）的 ld 按 packed 字节步长理解：`packedLd = (logicalLd + 1) / 2` 向上取整；逻辑维度为奇数时合法支持（尾字节高 nibble 零填充）
- 不支持 FP64、复数、HiF8、FP8_E8M0、FP6、FP4_E1M2
- 变换描述符通过 `aclblasLtMatrixTransformDescCreate` 创建并设置 scaleType；`SetAttribute` 可设置 TRANSA / TRANSB（默认 `ACLBLAS_OP_N`）
- layout 描述符通过 `aclblasLtMatrixLayoutCreate` 创建，`SetAttribute` 可设置 TYPE / ORDER / ROWS / COLS / LD / BATCH_COUNT

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。使用 BLASLt 接口时，需在 CMakeLists.txt 中额外链接 `libops_blasLt.so`。

```cpp
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blasLt.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct AclMemDeleter {
    void operator()(void* p) const { aclrtFree(p); }
};
struct LtHandleDeleter {
    void operator()(aclblasLtHandle_t h) const { aclblasLtDestroy(h); }
};
struct LtTransformDescDeleter {
    void operator()(aclblasLtMatrixTransformDesc_t d) const { aclblasLtMatrixTransformDescDestroy(d); }
};
struct LtLayoutDeleter {
    void operator()(aclblasLtMatrixLayout_t d) const { aclblasLtMatrixLayoutDestroy(d); }
};

int aclblasLtMatrixTransformTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();
    const uint64_t rows = 16;
    const uint64_t cols = 16;
    const int64_t ld = 16;
    float alpha = 2.0f;
    float beta = 1.0f;

    // 1. 创建 aclBLASLt 句柄与变换描述符
    aclblasLtHandle_t rawHandle = nullptr;
    auto blasRet = aclblasLtCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasLtCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasLtHandle_t>::type, LtHandleDeleter> handlePtr(rawHandle);

    aclblasLtMatrixTransformDesc_t rawTransformDesc = nullptr;
    blasRet = aclblasLtMatrixTransformDescCreate(&rawTransformDesc, ACL_FLOAT);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasLtMatrixTransformDescCreate failed. ERROR: %d\n", blasRet); return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasLtMatrixTransformDesc_t>::type, LtTransformDescDeleter>
        transformDescPtr(rawTransformDesc);

    int32_t opN = ACLBLAS_OP_N;
    blasRet = aclblasLtMatrixTransformDescSetAttribute(
        transformDescPtr.get(), ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opN, sizeof(int32_t));
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("SetAttribute TRANSA failed. ERROR: %d\n", blasRet);
              return blasRet);
    blasRet = aclblasLtMatrixTransformDescSetAttribute(
        transformDescPtr.get(), ACLBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &opN, sizeof(int32_t));
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("SetAttribute TRANSB failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 创建 A / B / C 的 layout 描述符
    aclblasLtMatrixLayout_t rawAdesc = nullptr;
    aclblasLtMatrixLayout_t rawBdesc = nullptr;
    aclblasLtMatrixLayout_t rawCdesc = nullptr;
    int32_t order = ACLBLASLT_ORDER_COL;

    blasRet = aclblasLtMatrixLayoutCreate(&rawAdesc, ACL_FLOAT, rows, cols, ld);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("MatrixLayoutCreate A failed. ERROR: %d\n", blasRet);
              return blasRet);
    blasRet = aclblasLtMatrixLayoutCreate(&rawBdesc, ACL_FLOAT, rows, cols, ld);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("MatrixLayoutCreate B failed. ERROR: %d\n", blasRet);
              return blasRet);
    blasRet = aclblasLtMatrixLayoutCreate(&rawCdesc, ACL_FLOAT, rows, cols, ld);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("MatrixLayoutCreate C failed. ERROR: %d\n", blasRet);
              return blasRet);

    std::unique_ptr<std::remove_pointer<aclblasLtMatrixLayout_t>::type, LtLayoutDeleter> aDescPtr(rawAdesc);
    std::unique_ptr<std::remove_pointer<aclblasLtMatrixLayout_t>::type, LtLayoutDeleter> bDescPtr(rawBdesc);
    std::unique_ptr<std::remove_pointer<aclblasLtMatrixLayout_t>::type, LtLayoutDeleter> cDescPtr(rawCdesc);

    aclblasLtMatrixLayoutSetAttribute(aDescPtr.get(), ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    aclblasLtMatrixLayoutSetAttribute(bDescPtr.get(), ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    aclblasLtMatrixLayoutSetAttribute(cDescPtr.get(), ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

    // 3. 准备 Host 数据
    std::vector<float> hA(rows * cols, 1.0f);
    std::vector<float> hB(rows * cols, 2.0f);
    size_t bytes = rows * cols * sizeof(float);

    // 4. 申请 Device 内存并拷贝数据
    void* rawA = nullptr;
    void* rawB = nullptr;
    void* rawC = nullptr;
    auto aclRet = aclrtMalloc(&rawA, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc(&rawB, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for B failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc(&rawC, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);

    std::unique_ptr<void, AclMemDeleter> aDevicePtr(rawA);
    std::unique_ptr<void, AclMemDeleter> bDevicePtr(rawB);
    std::unique_ptr<void, AclMemDeleter> cDevicePtr(rawC);

    aclRet = aclrtMemcpy(aDevicePtr.get(), bytes, hA.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(bDevicePtr.get(), bytes, hB.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);

    // 5. 调用 aclblasLtMatrixTransform：C = alpha * A + beta * B
    blasRet = aclblasLtMatrixTransform(
        handlePtr.get(), transformDescPtr.get(),
        &alpha, aDevicePtr.get(), aDescPtr.get(),
        &beta, bDevicePtr.get(), bDescPtr.get(),
        cDevicePtr.get(), cDescPtr.get(), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasLtMatrixTransform failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 6. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 7. 将结果从 Device 拷贝回 Host 并打印
    std::vector<float> hC(rows * cols, 0.0f);
    aclRet = aclrtMemcpy(hC.data(), bytes, cDevicePtr.get(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    LOG_PRINT("C[0] = %f (expected %f)\n", hC[0], alpha * hA[0] + beta * hB[0]);

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasLtMatrixTransformTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasLtMatrixTransformTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
