#!/usr/bin/env bash
# Build and run the RMSNorm JIT kernel microbenchmark.
# Assumes OpenVINO has already been compiled.
set -euo pipefail

OV_ROOT="${OV_ROOT:-$HOME/openvino-workspace/build/openvino}"
KERNEL_SRC_DIR="${KERNEL_SRC_DIR:-$OV_ROOT/src/plugins/intel_cpu/src/nodes/kernels/x64}"
BENCH_CPU="${BENCH_CPU:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------- pre-flight ----------
if [ ! -d "$OV_ROOT/build" ]; then
    echo "[ERROR] OV build not found at $OV_ROOT/build"
    echo "        Set OV_ROOT to a built OpenVINO repo root."
    exit 1
fi
if [ ! -f "$KERNEL_SRC_DIR/rms_kernel.cpp" ]; then
    echo "[ERROR] Kernel source not found: $KERNEL_SRC_DIR/rms_kernel.cpp"
    echo "        Set KERNEL_SRC_DIR to a directory containing rms_kernel.cpp/.hpp."
    exit 1
fi

echo "================================================================"
echo "  RMSNorm JIT Kernel Microbenchmark"
echo "  OV_ROOT:        $OV_ROOT"
echo "  KERNEL_SRC_DIR: $KERNEL_SRC_DIR"
echo "  Pinned to CPU:  $BENCH_CPU"
echo "================================================================"
echo ""

# ---------- build ----------
cmake -B "$SCRIPT_DIR/build" \
    -DOV_ROOT="$OV_ROOT" \
    -DKERNEL_SRC_DIR="$KERNEL_SRC_DIR" \
    -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
cmake --build "$SCRIPT_DIR/build" --parallel "$(nproc)" 2>&1 | tail -3

# ---------- run ----------
taskset -c "$BENCH_CPU" "$SCRIPT_DIR/build/rms_kernel_bench"
