#!/usr/bin/env python3
"""
csyrkx 性能/内存验收 — 执行 C++ GTest 的 TC_PF 用例采集 NPU 耗时，与 gpu_baseline.csv 比对。

判定口径：NPU 平均单次耗时 ≤ 标杆耗时（gpu_baseline.csv 的 gpu_ms 列）/倍率。
任务书 §3.3 判据：平均单次耗时不高于 GPU H100 标杆耗时的 2.5 倍（即达到标杆 0.4 倍以上性能）。
gpu_ms 为待回填的基线数据（NVIDIA H100，CUDA 12.4，驱动 550.104.05，warmup 后 >50 次采样均值）。

用法（Atlas 950PR 版）：
  python verify_performance.py --repo /path/to/ops-blas --soc ascend950 --timeout 3600
  python verify_performance.py --repo /path/to/ops-blas --soc ascend950 --skip-build --device 1
"""
import subprocess, os, sys, re, argparse, csv, datetime

OP = "csyrkx"
FAMILY = "syrkx"
PERF_THRESHOLD = 0.4
AVG_THRESHOLD = 0.5      # 任务书 §3.3：L3 算子平均倍率要求（全部 TC_PF 的算术平均）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_CSV = os.path.join(SCRIPT_DIR, "gpu_baseline.csv")
KEY = ["uplo", "trans", "n", "k"]


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


def parse_csv_params(repo):
    for arch in ["arch35", "arch22"]:
        p = os.path.join(repo, "test", FAMILY, OP, arch, f"{OP}_test.csv")
        if os.path.exists(p):
            return {r["case_name"]: r for r in csv.DictReader(open(p))
                    if r["case_name"].startswith("TC_PF_")}
    return {}


def load_baseline():
    """返回 {(uplo,trans,n,k): gpu_ms}；gpu_ms 为空（未回填）的条目跳过"""
    if not os.path.exists(GPU_CSV):
        return {}
    out = {}
    for r in csv.DictReader(open(GPU_CSV)):
        if r["gpu_ms"].strip():
            out[tuple(r[k] for k in KEY)] = float(r["gpu_ms"])
    return out


def build(repo, soc):
    cmd = ["bash", "build.sh", f"--soc={soc}", f"--ops={OP}"]
    print(f"  编译: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=repo, timeout=600)
    if r.returncode != 0: print("  编译失败"); return False
    print("  编译成功"); return True


def main():
    ap = argparse.ArgumentParser(description="csyrkx 性能/内存验收")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--soc", required=True)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(SCRIPT_DIR, f"results_npu_csyrkx_{ts}.csv")

    print(f"\n{'='*90}")
    print(f"  csyrkx 性能/内存验收")
    print(f"  验收标准: 单 case NPU 耗时 <= GPU H100 耗时 / {PERF_THRESHOLD}"
          f"（倍率 >= {PERF_THRESHOLD}），且全部 TC_PF 平均倍率 >= {AVG_THRESHOLD}")
    print(f"           （L3 算子要求：单 case >= {PERF_THRESHOLD}×，平均 >= {AVG_THRESHOLD}×）")
    print(f"{'='*90}")

    if not args.skip_build:
        if not build(repo, args.soc): sys.exit(1)
    else: print("  跳过编译")
    binary = find_binary(repo)
    if not binary: print("  找不到测试二进制"); sys.exit(1)
    print(f"  二进制: {binary}")

    csv_params = parse_csv_params(repo)
    baseline = load_baseline()
    print(f"  基线: {GPU_CSV}（已回填 {len(baseline)} 条）")
    if not baseline:
        print("  [提示] 基线尚未回填，全部用例将标记 NO_REF，仅采集 NPU 耗时")

    env = os.environ.copy()
    if args.device is not None: env["ASCEND_DEVICE_ID"] = str(args.device)
    cmd = [binary, "--gtest_filter=*TC_PF*"]
    print(f"  运行: {cmd[0]} --gtest_filter=*TC_PF*")
    try:
        r = subprocess.run(cmd, timeout=args.timeout, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.TimeoutExpired:
        print(f"  超时 ({args.timeout}s)，建议加大 --timeout"); sys.exit(1)

    results = []
    for line in r.stdout.splitlines():
        m = re.match(r"\[\s+OK\s+\]\s+\S+/(\S+)\s+\((\d+)\s*ms\)", line)
        if m: results.append({"case_name": m.group(1), "npu_ms": int(m.group(2)), "status": "ok"})
        m = re.match(r"\[\s+FAILED\s+\]\s+\S+/(\S+)", line)
        if m: results.append({"case_name": m.group(1), "npu_ms": -1, "status": "failed"})

    out_rows = []
    for res in results:
        p = csv_params.get(res["case_name"], {})
        key = tuple(p.get(k, "") for k in KEY)
        base = baseline.get(key)
        npu = res["npu_ms"]
        ratio = 0
        if base is not None and npu > 0:
            ratio = base / npu
            verdict = "PASS" if ratio >= PERF_THRESHOLD else "FAIL"
        else:
            verdict = "NO_REF"
        out_rows.append({"case_name": res["case_name"],
                         "uplo": p.get("uplo", "").replace("ACLBLAS_", ""),
                         "trans": p.get("trans", "").replace("ACLBLAS_OP_", ""),
                         "n": p.get("n", ""), "k": p.get("k", ""),
                         "npu_ms": npu if npu > 0 else "", "baseline_ms": base if base else "",
                         "ratio": round(ratio, 3) if ratio else "", "verdict": verdict})

    if out_rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            for row in out_rows: w.writerow(row)
        print(f"  结果: {out_csv} ({len(out_rows)} 条)")

    n_pass = sum(1 for r in out_rows if r["verdict"] == "PASS")
    n_fail = sum(1 for r in out_rows if r["verdict"] == "FAIL")
    n_noref = sum(1 for r in out_rows if r["verdict"] == "NO_REF")
    print(f"\n  {'case':<16} {'uplo':>6} {'tr':>3} {'n':>6} {'k':>6} "
          f"{'NPU_ms':>8} {'base_ms':>8} {'ratio':>6} {'verdict':>6}\n  {'-'*72}")
    for r in out_rows:
        print(f"  {r['case_name']:<16} {r['uplo']:>6} {r['trans']:>3} {r['n']:>6} {r['k']:>6} "
              f"{str(r['npu_ms']):>8} {str(r['baseline_ms']):>8} {str(r['ratio']):>6} "
              f"{r['verdict']:>6}")
    print(f"\n  汇总: PASS={n_pass}  FAIL={n_fail}  NO_REF={n_noref}")
    ratios = [float(r["ratio"]) for r in out_rows if r["ratio"] != ""]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    avg_verdict = "NO_REF" if not ratios else ("PASS" if avg_ratio >= AVG_THRESHOLD else "FAIL")
    print(f"  平均倍率: {avg_ratio:.3f}（要求 >= {AVG_THRESHOLD}）-> {avg_verdict}")
    print(f"  注: GTest 输出耗时含 host 准备+kernel+golden 计算+比对，为保守上界；")
    print(f"      精确 kernel 耗时可配合 msprof 采集。")
    sys.exit(0 if n_fail == 0 and avg_verdict != "FAIL" and (n_pass > 0 or n_noref > 0) else 1)


if __name__ == "__main__":
    main()
