#!/bin/bash
# Build and run the CPU-only pre-verification for aclblasCsyrkx.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="$ROOT/local_deps/root"
CANN="${ASCEND_HOME_PATH:-/home/zmaker/Ascend/cann-9.2.1}"

if [ ! -d "$D/usr/include/x86_64-linux-gnu/openblas-pthread" ]; then
    echo "local_deps not prepared; see local_verify/README.md" >&2
    exit 1
fi

g++ -std=c++17 -O2 -o "$ROOT/local_verify/csyrkx_cpu_check" "$ROOT/local_verify/csyrkx_cpu_check.cpp" \
    -I"$ROOT/ops-blas/include" -I"$ROOT/ops-blas/test/frame" -I"$ROOT/ops-blas/test/utils" \
    -I"$ROOT/ops-blas/blas/common/helper" -I"$ROOT/ops-blas/test/syrkx/csyrkx" \
    -I"$D/usr/include" -I"$D/usr/include/x86_64-linux-gnu/openblas-pthread" \
    -I"$CANN/include" \
    -L"$D/usr/lib/x86_64-linux-gnu/openblas-pthread" -lopenblas \
    "$D/usr/lib/x86_64-linux-gnu/libgtest.a" -lpthread \
    -Wl,-rpath,"$D/usr/lib/x86_64-linux-gnu/openblas-pthread" \
    -Wl,-rpath,"$D/usr/lib/x86_64-linux-gnu"

export LD_LIBRARY_PATH="$D/usr/lib/x86_64-linux-gnu:$D/usr/lib/x86_64-linux-gnu/openblas-pthread:${LD_LIBRARY_PATH}"

echo "== CSV formula / golden / boundary check =="
"$ROOT/local_verify/csyrkx_cpu_check" "$ROOT/test_cases/csyrkx_test.csv"

echo
echo "== real host entry, negative / n==0 validation paths =="
g++ -std=c++17 -O2 -o "$ROOT/local_verify/csyrkx_host_negative" "$ROOT/local_verify/csyrkx_host_negative.cpp" \
    -I"$ROOT/ops-blas/include" -I"$CANN/include" \
    -L"$ROOT/ops-blas/build" -lops_blas \
    -L"$CANN/x86_64-linux/lib64" -lascendcl \
    -Wl,-rpath,"$ROOT/ops-blas/build" -Wl,-rpath,"$CANN/x86_64-linux/lib64" \
    -Wl,--allow-shlib-undefined

LD_LIBRARY_PATH="$CANN/x86_64-linux/lib64:$CANN/lib64:$CANN/x86_64-linux/devlib/linux/x86_64:${LD_LIBRARY_PATH}" \
    "$ROOT/local_verify/csyrkx_host_negative"
