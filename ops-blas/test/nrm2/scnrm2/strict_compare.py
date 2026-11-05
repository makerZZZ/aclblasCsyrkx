# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Compare strict NPU event timings with the supplied GPU baseline."""
import csv
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log, baseline, out = map(Path, sys.argv[1:])
timed = {}
for line in log.read_text().splitlines():
    PATTERN = (r"\[STRICT_PERF\] case=(\S+) n=(\d+) "
               r"incx=(-?\d+).*?samples=(\d+) average_us=([0-9.eE+-]+)")
    m = re.search(PATTERN, line)
    if m:
        timed[m.group(1)] = (int(m.group(2)), int(m.group(3)),
                            int(m.group(4)), float(m.group(5)))
rows = []
for i, b in enumerate(csv.DictReader(baseline.open()), 1001):
    name = f"TC_PF_{i}"
    n, incx, samples, npu = timed[name]
    gpu_us = float(b["gpu_ms"]) * 1000.0
    ratio = gpu_us / npu if npu > 0 else 0.0
    row = dict(case_name=name, n=n, incx=incx,
               gpu_baseline_us=f"{gpu_us:.6f}", npu_avg_us=f"{npu:.6f}",
               samples=samples, ratio=f"{ratio:.6f}",
               verdict="PASS" if ratio >= 0.4 else "FAIL")
    rows.append(row)
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
logging.info("TOTAL=%d PASS=%d FAIL=%d", len(rows),
             sum(r["verdict"] == "PASS" for r in rows),
             sum(r["verdict"] == "FAIL" for r in rows))
