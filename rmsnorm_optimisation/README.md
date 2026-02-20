# RMSNorm JIT Kernel Microbenchmark

Builds and runs a standalone microbenchmark for the JIT-compiled RMSNorm scaling kernel (AVX2). Tests 8 shapes across F32 and BF16 precisions, validates correctness against a double-precision reference, and reports latency statistics with 95% confidence intervals.

## Prerequisites

- A built OpenVINO tree
- A Docker image built from the Dockerfile at `.github/dockerfiles/ov_build/ubuntu_22_04_x64/Dockerfile` in the OpenVINO repo root

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OV_ROOT` | `$HOME/openvino-workspace/build/openvino` | Path to a built OpenVINO repo root |
| `KERNEL_SRC_DIR` | `$OV_ROOT/src/plugins/intel_cpu/src/nodes/kernels/x64` | Directory containing the `rms_kernel.cpp`/`.hpp` to benchmark |
| `BENCH_CPU` | `0` | CPU core to pin the benchmark to |

## Usage

Build the Docker image (from the OpenVINO repo root):

```bash
docker build -t openvino-build \
  -f .github/dockerfiles/ov_build/ubuntu_22_04_x64/Dockerfile .
```

Run the benchmark inside Docker, mounting the built OpenVINO tree:

```bash
cd rmsnorm_optimisation

docker run --rm \
  --cpuset-cpus=0 \
  --user "$(id -u):$(id -g)" \
  -v /path/to/openvino:/openvino:ro \
  -v "$PWD":/bench \
  -w /bench \
  -e OV_ROOT=/openvino \
  -e BENCH_CPU=0 \
  openvino-build \
  bash run_bench.sh
```

To benchmark a different kernel variant, set `KERNEL_SRC_DIR`:

```bash
docker run --rm \
  --cpuset-cpus=0 \
  --user "$(id -u):$(id -g)" \
  -v /path/to/openvino:/openvino:ro \
  -v /path/to/variant:/variant:ro \
  -v "$PWD":/bench \
  -w /bench \
  -e OV_ROOT=/openvino \
  -e KERNEL_SRC_DIR=/variant \
  -e BENCH_CPU=0 \
  openvino-build \
  bash run_bench.sh
```

## Expected Output

The benchmark prints one line per shape, then writes `results.json`:

```
=== RMSNorm JIT Kernel Microbenchmark (AVX2) ===
Batched timing: 64 calls/batch, 200 batches/trial, 30 trials
IQR outlier trimming, 95% CI (t-distribution)

[BENCH] small_256_f32           data_size=  256  mean=   20.1 +/-  0.3 ns  min=   18.7 ns  p50=   20.3 ns  max_err=0.0e+00  trials=30
[BENCH] llama7b_f32             data_size= 4096  mean=  235.3 +/-  4.2 ns  min=  219.4 ns  p50=  235.3 ns  max_err=1.2e-07  trials=30
[BENCH] llama13b_f32            data_size= 5120  mean=  404.5 +/-  6.6 ns  min=  372.0 ns  p50=  410.0 ns  max_err=0.0e+00  trials=30
[BENCH] large_8192_f32          data_size= 8192  mean=  691.5 +/-  7.6 ns  min=  639.3 ns  p50=  691.6 ns  max_err=1.2e-07  trials=30
[BENCH] xlarge_16384_f32        data_size=16384  mean= 1581.5 +/- 11.7 ns  min= 1527.2 ns  p50= 1578.8 ns  max_err=2.4e-07  trials=30
[BENCH] llama7b_bf16            data_size= 4096  mean=  809.7 +/-  8.0 ns  min=  767.3 ns  p50=  811.8 ns  max_err=3.8e-03  trials=30
[BENCH] llama13b_bf16           data_size= 5120  mean= 1021.0 +/-  8.3 ns  min=  967.5 ns  p50= 1019.7 ns  max_err=3.8e-03  trials=30
[BENCH] large_8192_bf16         data_size= 8192  mean= 1784.8 +/- 14.9 ns  min= 1685.2 ns  p50= 1794.0 ns  max_err=3.8e-03  trials=30

[INFO] Results written to results.json
[OK] All benchmarks passed
```

## Output Format

`results.json` contains per-shape statistics:

```json
{
  "shapes": [
    {
      "name": "llama7b_bf16",
      "data_size": 4096,
      "precision": "bf16",
      "mean_ns": 809.7,
      "ci95_ns": 8.0,
      "min_ns": 767.3,
      "p50_ns": 811.8,
      "max_abs_error": 3.8e-03,
      "trials": 30
    }
  ]
}
```
