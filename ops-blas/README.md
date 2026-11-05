# ops-blas

## 🔥Latest News
- [2026/06] blas 目录新增 Level-3 混合精度 GEMM 接口：[aclblasGemmEx](blas/gemm_ex/)、[aclblasGemmBatchedEx](blas/gemm_batched_ex/)、[aclblasSgemmGroupedBatched](blas/gemm_grouped_batched/) 等。补齐 A5 批量 LAPACK 求解链路：[aclblasSgetrfBatched](blas/getrf_batched/)、[aclblasSgetriBatched](blas/getri_batched/)、[aclblasSgelsBatched](blas/gels_batched/)、[aclblasSgetrsBatched](blas/getrs_batched/) 等。
- [2026/05] blas目录新增[aclblasSgbmv](blas/gbmv/)、[aclblasSsbmv](blas/sbmv/)、[aclblasSspmv](blas/spmv/)、[aclblasStpsv](blas/tpsv/)等计算接口。
- [2026/03] ops-blas项目上线，提供BLAS计算的API以及现代灵活接口aclBLASLt。

## 🚀概述

ops-blas是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供高性能线性代数计算以及轻量化GEMM调用的算子库。

## 🛠️环境准备

[环境部署](docs/zh/install/quick_install.md)是体验本项目能力的前提，请先完成NPU驱动、CANN包安装等，确保环境正常。

## ⬇️源码下载

环境准备好后，下载与CANN版本配套的分支源码，命令如下，\$\{tag\_version\}替换为分支标签名。
 	 
> 说明：若环境中已存在配套分支源码，**可跳过本步骤**，例如CANNLab默认已提供最新商发版CANN对应的源码。

```bash
git clone -b ${tag_version} https://gitcode.com/cann/ops-blas.git
```

说明：对于CANNLab云开发环境，已默认提供最新商发CANN版本配套的源码，如需获取其他版本源码，参考上述命令获取。

## 📖学习教程

- [快速入门](docs/QUICKSTART.md)：从零开始快速体验项目核心基础能力，涵盖源码编译、算子调用、开发与调试等操作。

## 💬相关信息

- [目录结构](docs/zh/install/dir_structure.md)
- [接口列表](docs/zh/api_list.md)
- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-linear-algebra)

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/ops-blas/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/ops-blas/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/ops-blas/wiki)获取技术文章，如系列化教程、优秀实践等。