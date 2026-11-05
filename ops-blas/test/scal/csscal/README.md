# Csscal 测试说明

沿用 ops-blas 的 GTest/Netlib 测试框架，`arch35/csscal_test.csv` 包含任务要求的 1200 条用例：1000 条精度、200 条性能；另有 3 个参数测试。

在已编译的 ops-blas 根目录运行三个性能用例：

```bash
CSSCAL_PERF_SAMPLES=60 CSSCAL_PERF_WARMUP=5 ./build/test/scal/csscal/csscal_test --gtest_filter='*TC_PF_1001*:*TC_PF_1002*:*TC_PF_1003*'
```

每批预热 5 次、测量 60 次，每次恢复输入。`EVENT_TIME` 为平均 ACL 事件时间，单位 ms，乘 1000 后与标杆比较；msprof Task Duration 单独记录。

| n | incx | alpha | 平均耗时上限 |
|---:|---:|---:|---:|
| 1048576 | 1 | 2.5 | 13.57 微秒 |
| 2097152 | 1 | 2.5 | 21.05 微秒 |
| 4194304 | 1 | 2.5 | 43.02 微秒 |

精度按实部、虚部分别与 Netlib CBLAS 比较。最终验收需补齐实机结果、性能与内存截图。
