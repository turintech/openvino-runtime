#!/bin/bash
# GPU MVN 2-Pass Optimization: Randomized A/B Benchmark
#
# Orchestrates: model creation → randomized interleaved A/B benchmark → statistical comparison.
# Takes two pre-built OpenVINO directories (optimized vs baseline) — stateless, no branch switching.
#
# Usage:
#   ./experiments/30-gpu-mvn-2pass-benchmark/commands/benchmark.sh \
#       <build_optimized> <build_baseline> [options]
#
# Options:
#   --model-dir <path>   Skip model creation, use pre-existing IR files
#   --trials N           Randomized trials (default: 24)
#   --niter N            Iterations per benchmark_app run (default: 500)
#   --seed N             Random seed (default: 42)
#
# Example (using exp 12's existing models):
#   ./experiments/30-gpu-mvn-2pass-benchmark/commands/benchmark.sh \
#       /tmp/build_optimized /tmp/build_baseline \
#       --model-dir experiments/12-gpu-mvn-2pass/models

set -euo pipefail

# ── Resolve paths ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$EXPERIMENT_DIR/scripts"
RESULTS_DIR="$EXPERIMENT_DIR/results"

# ── Defaults ──────────────────────────────────────────────────────────
MODEL_DIR=""
TRIALS=24
NITER=500
SEED=42

# ── Helper ────────────────────────────────────────────────────────────
log() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
}

usage() {
    echo "Usage: $0 <build_optimized> <build_baseline> [options]"
    echo ""
    echo "Positional arguments:"
    echo "  build_optimized    Build dir with MVN 2-pass patch applied"
    echo "  build_baseline     Build dir with baseline (unpatched) OpenVINO"
    echo ""
    echo "Options:"
    echo "  --model-dir <path>   Skip model creation, use pre-existing IR files"
    echo "  --trials N           Randomized trials (default: 24)"
    echo "  --niter N            Iterations per benchmark_app run (default: 500)"
    echo "  --seed N             Random seed (default: 42)"
    exit 1
}

# ── Parse arguments ───────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    usage
fi

BUILD_OPTIMIZED="$(realpath "$1")"
BUILD_BASELINE="$(realpath "$2")"
shift 2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$(realpath "$2")"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --niter)
            NITER="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage
            ;;
    esac
done

# ── Validate builds ──────────────────────────────────────────────────
validate_build() {
    local label="$1"
    local build_dir="$2"
    local app="$build_dir/openvino/bin/intel64/Release/benchmark_app"

    if [[ ! -d "$build_dir" ]]; then
        echo "ERROR: $label build directory not found: $build_dir" >&2
        exit 1
    fi
    if [[ ! -x "$app" ]]; then
        echo "ERROR: $label benchmark_app not found: $app" >&2
        exit 1
    fi
}

validate_build "Optimized" "$BUILD_OPTIMIZED"
validate_build "Baseline" "$BUILD_BASELINE"

# ── Print configuration ──────────────────────────────────────────────
log "GPU MVN 2-Pass Optimization: Randomized A/B Benchmark"
echo "  Optimized build:  $BUILD_OPTIMIZED"
echo "  Baseline build:   $BUILD_BASELINE"
echo "  Trials:           $TRIALS"
echo "  Iterations:       $NITER"
echo "  Seed:             $SEED"
if [[ -n "$MODEL_DIR" ]]; then
    echo "  Model dir:        $MODEL_DIR (pre-existing)"
else
    echo "  Model dir:        $RESULTS_DIR/models (will create)"
fi
echo ""

# ── Phase 1: Create models (if needed) ───────────────────────────────
if [[ -z "$MODEL_DIR" ]]; then
    MODEL_DIR="$RESULTS_DIR/models"
    log "Phase 1: Creating synthetic models"
    /usr/bin/python3 "$SCRIPTS_DIR/create_models.py" \
        --build "$BUILD_OPTIMIZED" \
        --output-dir "$MODEL_DIR"
else
    log "Phase 1: Using pre-existing models"
    if [[ ! -d "$MODEL_DIR" ]] || ! ls "$MODEL_DIR"/*.xml >/dev/null 2>&1; then
        echo "ERROR: No .xml model files found in $MODEL_DIR" >&2
        exit 1
    fi
    echo "  Models: $(ls "$MODEL_DIR"/*.xml | wc -l) IR files in $MODEL_DIR"
fi

# ── Phase 2: Randomized interleaved A/B benchmark ────────────────────
BENCHMARK_CSV="$RESULTS_DIR/randomized_ab.csv"
log "Phase 2: Randomized interleaved A/B benchmark ($TRIALS trials)"
/usr/bin/python3 "$SCRIPTS_DIR/run_benchmark.py" \
    --build-optimized "$BUILD_OPTIMIZED" \
    --build-baseline "$BUILD_BASELINE" \
    --model-dir "$MODEL_DIR" \
    --trials "$TRIALS" \
    --niter "$NITER" \
    --seed "$SEED" \
    --output "$BENCHMARK_CSV"

# ── Phase 3: Statistical comparison ──────────────────────────────────
COMPARISON_CSV="$RESULTS_DIR/comparison.csv"
log "Phase 3: Statistical comparison (Welch's t-test)"
/usr/bin/python3 "$SCRIPTS_DIR/compare_results.py" \
    --input "$BENCHMARK_CSV" \
    --output "$COMPARISON_CSV"

# ── Done ──────────────────────────────────────────────────────────────
log "Benchmark Complete!"
echo "  Results:"
echo "    Raw data:    $BENCHMARK_CSV"
echo "    Comparison:  $COMPARISON_CSV"
echo "    Models:      $MODEL_DIR"
