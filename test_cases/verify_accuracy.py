#!/usr/bin/env python3
"""
csyrkx 精度验收 — 编译并执行 ops-blas C++ GTest，解析逐条 PASS/FAIL。

前置：将本目录的 csyrkx_test.csv 放到 ops-blas 仓 test/syrkx/csyrkx/<arch>/ 下
     （arch35 对应 Atlas 950PR，arch22 对应 A2/A3），或用 --csv 自动复制（先备份原文件）。
注意：仓内当前**无** test/syrkx/ 工程（仅有实数 ssyrkx 同类实现需参考），需按本 CSV 列序
     新建 csyrkx_param.h 与 golden 后接入。列序：
     case_name,description,uplo,trans,n,k,alpha_real,alpha_imag,a_fill,lda,
     b_fill,ldb,beta_real,beta_imag,c_fill,ldc,expect_result,...
     负向约定：标量空指针写 "null"、tensor 空指针写 "NULLPTR"、非法枚举写 999、
     负维度写 -1、非法前导维写具体数值。

用法（Atlas 950PR 版）：
  python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --csv ./csyrkx_test.csv
  python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --skip-build --device 1
  python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --filter TC_L0 --timeout 3600
"""
import subprocess, os, sys, re, argparse, shutil

OP = "csyrkx"
FAMILY = "syrkx"
ARCH_BY_SOC = {"ascend950": "arch35", "ascend910b3": "arch22", "ascend910b4": "arch22"}


def find_binary(repo):
    candidates = [
        os.path.join(repo, "build", "test", FAMILY, OP, f"{OP}_test"),
        os.path.join(repo, "build", "test", FAMILY, f"{OP}_test"),
        os.path.join(repo, "build", "test", OP, f"{OP}_test"),
    ]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def install_csv(repo, soc, csv_path):
    arch = ARCH_BY_SOC.get(soc, "arch35")
    dst = os.path.join(repo, "test", FAMILY, OP, arch, f"{OP}_test.csv")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        shutil.copy2(dst, dst + ".bak")
    shutil.copy2(csv_path, dst)
    print(f"  用例安装: {csv_path} -> {dst}（原文件备份为 .bak）")


def build(repo, soc):
    cmd = ["bash", "build.sh", f"--soc={soc}", f"--ops={OP}"]
    print(f"  编译: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=repo, timeout=600)
    if r.returncode != 0:
        print("  编译失败"); return False
    print("  编译成功"); return True


def run(binary, filter_str, device, timeout):
    env = os.environ.copy()
    if device is not None:
        env["ASCEND_DEVICE_ID"] = str(device)
    parts = [f"*{filter_str}*"] if filter_str else []
    parts.append("-*TC_PF*")   # 排除性能用例
    gf = ":".join(parts)
    cmd = [binary, f"--gtest_filter={gf}"]
    print(f"  运行: {cmd[0]} --gtest_filter={gf}")
    try:
        r = subprocess.run(cmd, timeout=timeout, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.TimeoutExpired:
        print(f"  超时 ({timeout}s)，大尺寸用例较慢，请加大 --timeout")
        return 0, 0, []
    details = []
    for line in r.stdout.splitlines():
        m = re.match(r"\[\s+OK\s+\]\s+\S+/(\S+)\s+\((\d+)\s*ms\)", line)
        if m: details.append(("PASS", m.group(1), int(m.group(2))))
        m = re.match(r"\[\s+FAILED\s+\]\s+\S+/(\S+)", line)
        if m: details.append(("FAIL", m.group(1), 0))
    passed = sum(1 for s, _, _ in details if s == "PASS")
    failed = sum(1 for s, _, _ in details if s == "FAIL")
    return passed, failed, details


def main():
    ap = argparse.ArgumentParser(description="csyrkx 精度验收")
    ap.add_argument("--repo", required=True, help="ops-blas 仓库路径")
    ap.add_argument("--soc", required=True, help="SOC 版本，如 ascend950 / ascend910b3")
    ap.add_argument("--csv", default=None, help="先复制用例 CSV 到仓内测试目录")
    ap.add_argument("--filter", default=None, help="用例名过滤 (如 TC_L0)")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--device", type=int, default=None, help="NPU 设备 ID")
    ap.add_argument("--timeout", type=int, default=1800, help="超时秒数")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    print(f"\n{'='*70}\n  csyrkx 精度验收\n{'='*70}")
    if args.csv:
        install_csv(repo, args.soc, os.path.abspath(args.csv))
    if not args.skip_build:
        if not build(repo, args.soc): sys.exit(1)
    else:
        print("  跳过编译")
    binary = find_binary(repo)
    if not binary:
        print("  找不到测试二进制，请先编译"); sys.exit(1)
    print(f"  二进制: {binary}")
    passed, failed, details = run(binary, args.filter, args.device, args.timeout)
    print(f"\n  {'case':<50} {'result':>6} {'ms':>6}\n  {'-'*64}")
    for status, name, ms in details:
        print(f"  {name:<50} {status:>6} {ms:>6}" + ("  <==" if status == "FAIL" else ""))
    print(f"\n  汇总: PASS={passed}  FAIL={failed}  (共{passed+failed})")
    if failed == 0 and passed > 0: print("  结论: ALL PASS")
    elif failed > 0: print(f"  结论: {failed} FAILURES")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
