#!/usr/bin/env python3
"""Generate csyrkx test CSV: C = alpha*op(A)*op(B)^T + beta*C（complex64，C 为 n×n 对称，
仅更新 uplo 指定三角；op 取 N（A、B 均 n×k）/ T / C（A、B 均 k×n））

列格式（按 FORMAT_BASELINE.md 决议，§2.3 签名顺序去 handle）：
  case_name,description,uplo,trans,n,k,alpha_real,alpha_imag,a_fill,lda,
  b_fill,ldb,beta_real,beta_imag,c_fill,ldc,expect_result,
  mere_threshold,mare_multiplier,random_seed
  - tensor 入参 A/B/C 以 <参数名>_fill 小写列表达填充 token（空指针写 NULLPTR）；
  - 复数标量拆 alpha_real/alpha_imag、beta_real/beta_imag；
  - 枚举写 ACLBLAS_* 全名（uplo=ACLBLAS_UPPER/LOWER，trans=ACLBLAS_OP_N/T/C），
    非法枚举写 999、负维度写 -1、非法前导维写具体数值。

用法（条数可扩展）：
  python gen_csv.py                          # 默认 1000 精度 + 200 性能
  python gen_csv.py --accuracy 1500 --perf 300
  python gen_csv.py --seed 12345
"""
import csv, os, random, argparse

SEED = 20261028

SIZES = [1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65,96,127,128,129,255,256,257,512,1024,2048]
MED = [10,12,15,24,48,96,200,400]
RECT = [(10,20),(15,32),(32,64),(64,17),(128,64),(200,300),(256,128),(512,1024),
        (1000,999),(2048,1024)]
UPLO = ["ACLBLAS_UPPER", "ACLBLAS_LOWER"]
TRANS = ["ACLBLAS_OP_N", "ACLBLAS_OP_T", "ACLBLAS_OP_C"]
SCALARS = [(1,0,0,0),(1,0,1,0),(0.5,0.5,0.5,0.25),(0,0,1,0),
           (-1,0,1,0),(1,1,0,0),(0.5,-0.5,1,1),(0,0,0,0)]
FILL_RAND = "RANDOM_NORM_5_5"
FILLS = [("RANDOM_NORM_5_5","rand"),("VALUE_NORM_0","zero"),("RANDOM_ALTER","alter"),
         ("RANDOM_EXTREME","extreme"),("VALUE_NORM_INF","inf"),("VALUE_NORM_NAN","nan")]
MERE = "0.00012207031"
MARE = "10.0"
BUDGET = 512*1024**2
ELEM = 8
SUCCESS = "ACLBLAS_STATUS_SUCCESS"
INVALID = "ACLBLAS_STATUS_INVALID_VALUE"

# 任务书 §3.3 典型 case（逐参数精确一致：n,k,uplo,trans,lda,ldb,ldc）
DOC_CASES = [("ACLBLAS_UPPER",256,256,"ACLBLAS_OP_N"),
             ("ACLBLAS_LOWER",1024,1024,"ACLBLAS_OP_N"),
             ("ACLBLAS_UPPER",2048,1024,"ACLBLAS_OP_T"),
             ("ACLBLAS_LOWER",4096,2048,"ACLBLAS_OP_T")]


def rows_ab(trans, n, k):
    return n if trans == "ACLBLAS_OP_N" else k


def elems(trans, n, k, lda=None, ldb=None, ldc=None):
    n, k = max(n, 0), max(k, 0)
    lda = max(1, rows_ab(trans, n, k)) if lda is None else lda
    ldb = max(1, rows_ab(trans, n, k)) if ldb is None else ldb
    ldc = max(1, n) if ldc is None else ldc
    cols = k if trans == "ACLBLAS_OP_N" else n
    return lda * cols + ldb * cols + ldc * n


def fits(trans, n, k):
    return elems(trans, n, k) * ELEM <= BUDGET


class Gen:
    def __init__(self, seed):
        self.rows = []
        self.idx = 0
        self.rng = random.Random(seed)

    def add(self, nm, ds, uplo, trans, n, k, ar, ai, br, bi,
            exp=SUCCESS, lda=None, ldb=None, ldc=None,
            af=FILL_RAND, bf=FILL_RAND, cf=FILL_RAND, mere=MERE, mare=MARE):
        self.idx += 1
        self.rows.append({
            "case_name": nm, "description": ds, "uplo": uplo, "trans": trans,
            "n": n, "k": k,
            "alpha_real": ar, "alpha_imag": ai,
            "a_fill": af, "lda": max(1, rows_ab(trans, n, k)) if lda is None else lda,
            "b_fill": bf, "ldb": max(1, rows_ab(trans, n, k)) if ldb is None else ldb,
            "beta_real": br, "beta_imag": bi,
            "c_fill": cf, "ldc": max(1, n) if ldc is None else ldc,
            "expect_result": exp, "mere_threshold": mere, "mare_multiplier": mare,
            "random_seed": 20261028 + self.idx})

    # ---- 精度用例各类别 ----
    def l0_basic(self):
        for u in UPLO:
            for tr in TRANS:
                for sz in [4, 8]:
                    self.add(f"TC_L0_{self.idx+1:03d}", f"basic_{u[-5:]}_{tr[-1]}_n{sz}",
                             u, tr, sz, sz, 1, 0, 0, 0)

    def l1_size(self):
        for u in UPLO:
            for tr in TRANS:
                for sz in SIZES:
                    self.add(f"TC_SQ_{self.idx+1:03d}", f"sq_{sz}_{u[-5:]}_{tr[-1]}",
                             u, tr, sz, sz, 1, 0, 0, 0)

    def l1b_rect(self):
        for u in UPLO:
            for tr in TRANS:
                for (n, k) in RECT:
                    self.add(f"TC_MN_{self.idx+1:03d}", f"mn_{n}_{k}_{u[-5:]}_{tr[-1]}",
                             u, tr, n, k, 1, 0, 0, 0)

    def l2_scalar(self):
        for (ar, ai, br, bi) in SCALARS:
            for u in UPLO:
                for tr in TRANS:
                    self.add(f"TC_AB_{self.idx+1:03d}",
                             f"ab_{ar}_{ai}_{br}_{bi}_{u[-5:]}_{tr[-1]}",
                             u, tr, 32, 32, ar, ai, br, bi)

    def l4_ld(self):
        for u in UPLO:
            for tr in TRANS:
                for sz in [16, 32, 64, 128]:
                    self.add(f"TC_LD_{self.idx+1:03d}", f"ld_{sz}_{u[-5:]}_{tr[-1]}_pad",
                             u, tr, sz, sz, 1, 0, 0, 1,
                             lda=rows_ab(tr, sz, sz) + 4, ldb=rows_ab(tr, sz, sz) + 8,
                             ldc=sz + 4)

    def l5_fill(self):
        for f, l in FILLS:
            for tag in ["a", "b", "c"]:
                self.add(f"TC_FL_{self.idx+1:03d}", f"fill_{tag}_{l}",
                         "ACLBLAS_UPPER", "ACLBLAS_OP_N", 32, 32, 1, 0, 0, 1,
                         **{"af" if tag == "a" else "bf" if tag == "b" else "cf": f})

    def l5b_cover(self):
        for sz in MED:
            for u in UPLO:
                for tr in ["ACLBLAS_OP_N", "ACLBLAS_OP_T"]:
                    self.add(f"TC_CV_{self.idx+1:03d}", f"cv_{sz}_{u[-5:]}_{tr[-1]}",
                             u, tr, sz, sz, 1, 0, 0, 1)

    def l6_edge(self):
        U, L = "ACLBLAS_UPPER", "ACLBLAS_LOWER"
        N, T, C = "ACLBLAS_OP_N", "ACLBLAS_OP_T", "ACLBLAS_OP_C"
        # ---- 合法 no-op / 0 维（§2.5） ----
        self.add(f"TC_ED_{self.idx+1:03d}", "n0_quick_return", U, N, 0, 16, 1, 0, 0, 0)
        self.add(f"TC_ED_{self.idx+1:03d}", "n0_quick_return_lower", L, T, 0, 16, 1, 0, 0, 0)
        self.add(f"TC_ED_{self.idx+1:03d}", "k0_beta1_upper_unchanged", U, N, 16, 0, 1, 0, 1, 0)
        self.add(f"TC_ED_{self.idx+1:03d}", "k0_beta1_lower_unchanged", L, N, 16, 0, 1, 0, 1, 0)
        self.add(f"TC_ED_{self.idx+1:03d}", "k0_beta0_upper_zeroed", U, N, 16, 0, 1, 0, 0, 0)
        self.add(f"TC_ED_{self.idx+1:03d}", "k0_beta0_lower_zeroed", L, T, 16, 0, 1, 0, 0, 0)
        self.add(f"TC_ED_{self.idx+1:03d}", "n1k1", U, N, 1, 1, 1, 0, 0, 0)
        self.add(f"TC_ED_{self.idx+1:03d}", "alpha0_beta1_C_unchanged", U, N, 16, 16, 0, 0, 1, 0)
        self.add(f"TC_ED_{self.idx+1:03d}", "alpha0_beta_half", L, T, 16, 16, 0, 0, 0.5, 0.5)
        self.add(f"TC_ED_{self.idx+1:03d}", "beta0_c_nan", U, N, 32, 32, 1, 0, 0, 0,
                 cf="VALUE_NORM_NAN")
        # ---- §2.5 条件叠加：n=0 / k=0 时不校验对应空指针 ----
        self.add(f"TC_ED_{self.idx+1:03d}", "n0_a_b_null", U, N, 0, 16, 1, 0, 0, 0,
                 af="NULLPTR", bf="NULLPTR")
        self.add(f"TC_ED_{self.idx+1:03d}", "n0_c_null", U, N, 0, 16, 1, 0, 1, 0, cf="NULLPTR")
        self.add(f"TC_ED_{self.idx+1:03d}", "k0_a_b_null", U, N, 16, 0, 1, 0, 0, 0,
                 af="NULLPTR", bf="NULLPTR")
        self.add(f"TC_ED_{self.idx+1:03d}", "alpha0_a_null", U, N, 16, 16, 0, 0, 1, 0, af="NULLPTR")
        # ---- 负向：空指针 ----
        self.add(f"TC_ED_{self.idx+1:03d}", "null_alpha", U, N, 16, 16, "null", 0, 0, 0, exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "null_beta", U, N, 16, 16, 1, 0, "null", 0, exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "null_A", U, N, 16, 16, 1, 0, 0, 0, af="NULLPTR", exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "null_B", U, N, 16, 16, 1, 0, 0, 0, bf="NULLPTR", exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "null_C", U, N, 16, 16, 1, 0, 1, 0, cf="NULLPTR", exp=INVALID)
        # ---- 负向：非法枚举 / 非法前导维 / 负维度 ----
        self.add(f"TC_ED_{self.idx+1:03d}", "invalid_uplo", "999", N, 16, 16, 1, 0, 0, 0, exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "invalid_trans", U, "999", 16, 16, 1, 0, 0, 0, exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "invalid_lda_N", U, N, 16, 16, 1, 0, 0, 0,
                 lda=8, exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "invalid_lda_T", U, T, 16, 16, 1, 0, 0, 0,
                 lda=8, exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "invalid_ldb_N", U, N, 16, 16, 1, 0, 0, 0,
                 ldb=8, exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "invalid_ldb_T", U, T, 16, 16, 1, 0, 0, 0,
                 ldb=8, exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "invalid_ldc", U, N, 16, 16, 1, 0, 0, 0,
                 ldc=8, exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "neg_n", U, N, -1, 16, 1, 0, 0, 0, exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "neg_k", U, N, 16, -1, 1, 0, 0, 0, exp=INVALID)
        self.add(f"TC_ED_{self.idx+1:03d}", "neg_n_null_A", U, N, -1, 16, 1, 0, 0, 0,
                 af="NULLPTR", exp=INVALID)
        # ---- 特殊标量 ----
        self.add(f"TC_ED_{self.idx+1:03d}", "large_alpha", L, C, 32, 32, 100, 50, 0, 0)
        self.add(f"TC_ED_{self.idx+1:03d}", "pure_imag_scalars", U, C, 32, 32, 0, 1, 0, -1)

    def ex_extend(self, target):
        pool = [1,2,3,4,5,7,8,9,15,16,17,31,32,33,63,64,65,96,127,128,129,
                200,255,256,257,511,512,513,1000,1024]
        i = 0
        while len([r for r in self.rows if not r["case_name"].startswith("TC_PF")]) < target:
            n = pool[self.rng.randrange(len(pool))]
            k = pool[self.rng.randrange(len(pool))]
            u = UPLO[i % 2]
            tr = TRANS[i % 3]
            ar, ai, br, bi = SCALARS[i % len(SCALARS)]
            if not fits(tr, n, k):
                i += 1
                continue
            pad = (i % 5 == 4)
            self.add(f"TC_EX_{self.idx+1:04d}", f"ex_{n}_{k}_{u[-5:]}_{tr[-1]}",
                     u, tr, n, k, ar, ai, br, bi,
                     lda=rows_ab(tr, n, k) + 8 if pad else None,
                     ldb=rows_ab(tr, n, k) + 8 if pad else None,
                     ldc=n + 8 if pad else None)
            i += 1

    # ---- 性能用例 ----
    def perf(self, target):
        # 任务书 §3.3 的 4 条典型 case（逐参数精确一致）
        for u, n, k, tr in DOC_CASES:
            self.add(f"TC_PF_{self.idx+1:04d}", f"pf_doc_{n}_{k}_{u[-5:]}_{tr[-1]}",
                     u, tr, n, k, 1, 0, 0, 0)
        # 小尺寸延迟敏感区
        for sz in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            self.add(f"TC_PF_{self.idx+1:04d}", f"pf_small_{sz}", "ACLBLAS_UPPER",
                     "ACLBLAS_OP_N", sz, sz, 1, 0, 0, 0)
        # 规模对数扫描（512MB 预算内：2*n*k + n*n 个复数 ≤ 512MB）
        n_sq = min(50, target // 4)
        for i in range(n_sq):
            n = int(round(256 * (4096 / 256) ** (i / max(n_sq - 1, 1))))
            k = max(16, n // 2) if i % 2 == 0 else n
            tr = ["ACLBLAS_OP_N", "ACLBLAS_OP_T"][i % 2]
            u = UPLO[i % 2]
            if not fits(tr, n, k):
                continue
            self.add(f"TC_PF_{self.idx+1:04d}", f"pf_sq_{n}_{k}_{u[-5:]}_{tr[-1]}",
                     u, tr, n, k, 1, 0, 0, 0)
        # 补足到 target：预算内随机 n/k（连续访存：lda/ldb/ldc 紧凑）
        while len([r for r in self.rows if r["case_name"].startswith("TC_PF")]) < target:
            n = self.rng.randrange(16, 4097)
            k = self.rng.randrange(16, 4097)
            tr = TRANS[self.rng.randrange(2)]
            u = UPLO[self.rng.randrange(2)]
            if not fits(tr, n, k):
                continue
            self.add(f"TC_PF_{self.idx+1:04d}", f"pf_mix_{n}_{k}_{u[-5:]}_{tr[-1]}",
                     u, tr, n, k, 1, 0, 0, 0)


FIELDS = ["case_name","description","uplo","trans","n","k",
          "alpha_real","alpha_imag","a_fill","lda","b_fill","ldb",
          "beta_real","beta_imag","c_fill","ldc",
          "expect_result","mere_threshold","mare_multiplier","random_seed"]


def main():
    ap = argparse.ArgumentParser(description="csyrkx 用例生成（条数可扩展）")
    ap.add_argument("--accuracy", type=int, default=1000, help="精度用例条数（默认1000）")
    ap.add_argument("--perf", type=int, default=200, help="性能/内存用例条数（默认200）")
    ap.add_argument("--seed", type=int, default=SEED, help="随机种子（固定可复现）")
    ap.add_argument("--dist", choices=["uniform", "mixed"], default="uniform",
                    help="输入分布；mixed 依赖测试工程扩展，暂以均匀生成")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline-out", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(here, "csyrkx_test.csv")
    bl_out = args.baseline_out or os.path.join(here, "gpu_baseline.csv")
    if args.dist == "mixed":
        print("[提示] 正态分布依赖 ops-blas 测试工程扩展（另行任务），本次以均匀分布生成")

    g = Gen(args.seed)
    g.l0_basic(); g.l1_size(); g.l1b_rect(); g.l2_scalar(); g.l4_ld()
    g.l5_fill(); g.l5b_cover(); g.l6_edge()
    if args.accuracy > len(g.rows):
        g.ex_extend(args.accuracy)
    g.perf(args.perf)

    acc_rows = [r for r in g.rows if not r["case_name"].startswith("TC_PF")]
    pf_rows = [r for r in g.rows if r["case_name"].startswith("TC_PF")]
    for r in pf_rows:
        assert r["lda"] == max(1, rows_ab(r["trans"], r["n"], r["k"])), \
            f"性能用例含非紧凑 lda: {r['case_name']}"
        assert r["ldb"] == max(1, rows_ab(r["trans"], r["n"], r["k"])), \
            f"性能用例含非紧凑 ldb: {r['case_name']}"
        assert r["ldc"] == max(1, r["n"]), f"性能用例含非紧凑 ldc: {r['case_name']}"
        assert r["a_fill"] == FILL_RAND and r["b_fill"] == FILL_RAND and r["c_fill"] == FILL_RAND, \
            f"性能用例含非随机填充: {r['case_name']}"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for r in g.rows: w.writerow(r)

    with open(bl_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","uplo","trans","n","k","lda","ldb","ldc","gpu_ms"])
        for i, r in enumerate(pf_rows):
            prefix = "csyrkx-base" if i < len(DOC_CASES) else "csyrkx-perf"
            w.writerow([f"{prefix}-{i:03d}", r["uplo"], r["trans"], r["n"], r["k"],
                        r["lda"], r["ldb"], r["ldc"], ""])

    from collections import Counter
    cats = Counter(r["case_name"].split("_")[1] for r in acc_rows)
    print(f"{out}: 共 {len(g.rows)} 条（精度 {len(acc_rows)} + 性能/内存 {len(pf_rows)}）")
    print("精度用例分类:", dict(sorted(cats.items())))
    print(f"{bl_out}: {len(pf_rows)} 条基线占位（gpu_ms 待回填）")


if __name__ == "__main__":
    main()
