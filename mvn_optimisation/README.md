# MVN 2-Pass Optimisation Benchmark

Randomized interleaved A/B benchmark for measuring the impact of the 2-pass MVN kernel optimisation in the OpenVINO Intel GPU plugin.

## Prerequisites

Two OpenVINO builds compiled from source:

- Optimised - built from a branch with the 2-pass MVN kernel patch applied
- Baseline - built from the same base commit without the patch

Both builds must contain `benchmark_app` at `<build>/openvino/bin/intel64/Release/benchmark_app`.

Python 3.11 with the OpenVINO Python bindings is required for model creation (the optimised build's bindings are used automatically).

## Quick Start

```bash
./benchmark.sh <build_optimized> <build_baseline>
```

This runs three phases end-to-end:

1. Create models - generates three synthetic IR models (UNet3D encoder, SegNet highres, SegNet control)
2. Benchmark - runs 24 randomized interleaved trials (12 per build), 500 iterations each
3. Compare - computes per-model Welch's t-test on latency and MVN kernel timing

Results are written to `results/`.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-dir <path>` | (create new) | Skip model creation, use pre-existing IR files |
| `--trials N` | 24 | Total randomized trials (split evenly between builds) |
| `--niter N` | 500 | Iterations per `benchmark_app` invocation |
| `--seed N` | 42 | Random seed for trial ordering |

## Individual Scripts

### `create_models.py`

Generates synthetic OpenVINO IR models designed to isolate MVN performance:

```bash
python3 create_models.py --build <build_dir> --output-dir models/
```

### `run_benchmark.py`

Runs the randomized interleaved A/B benchmark using `benchmark_app`:

```bash
python3 run_benchmark.py \
    --build-optimized <build_opt> --build-baseline <build_base> \
    --model-dir models/ --trials 24 --niter 500 --seed 42 \
    --output results/randomized_ab.csv
```

### `compare_results.py`

Computes statistical comparison (Welch's t-test) from benchmark CSV:

```bash
python3 compare_results.py \
    --input results/randomized_ab.csv \
    --output results/comparison.csv
```
