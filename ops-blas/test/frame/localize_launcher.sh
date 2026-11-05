#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
# 合并测试 Runner（blas_all_test，issue #457）的 CXX_COMPILER_LAUNCHER。
# 用法（由 CMake CXX_COMPILER_LAUNCHER 注入，元素依次拼接在真实编译器命令前）：
#   localize_launcher.sh <symbols_file> <real-compiler> <args...>
# 透传真实编译命令；编译成功后对产物 .o 将跨文件重名的全局符号
# （清单由 amalgamate_conflicts.py 生成）降级为局部符号，
# 避免合并链接时 multiple definition。仅 blas_all_test 目标启用。

symbols_file="$1"
shift

"$@"
status=$?
[ $status -eq 0 ] || exit $status

# 提取 -o <out> 参数
out=""
prev=""
for a in "$@"; do
    if [ "$prev" = "-o" ]; then
        out="$a"
        break
    fi
    prev="$a"
done
[ -n "$out" ] || exit 0
case "$out" in
    *.o) ;;
    *) exit 0 ;;
esac

if [ -z "${symbols_file}" ] || [ ! -s "${symbols_file}" ]; then
    exit 0
fi
if ! command -v objcopy >/dev/null 2>&1; then
    exit 0
fi

# --localize-symbols=<file>：文件内每行一个符号；失败不阻塞编译（交由链接期回退）
objcopy --localize-symbols="${symbols_file}" "${out}" 2>/dev/null || true
exit 0
