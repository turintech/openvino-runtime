// Standalone microbenchmark for jit_rms_kernel (AVX2).
// No gtest dependency - just a plain main().
//
// Build:
//   cmake -B build -DOV_ROOT=/path/to/openvino
//   cmake --build build --parallel $(nproc)
//
// Run:
//   taskset -c 0 ./build/rms_kernel_bench

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "nodes/kernels/x64/rms_kernel.hpp"

using namespace ov::intel_cpu::kernel;
namespace x64 = dnnl::impl::cpu::x64;

// ---------------------------------------------------------------------------
// BF16 conversion helpers
// ---------------------------------------------------------------------------
static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // Round to nearest even
    uint32_t rounding_bias = ((u >> 16) & 1) + 0x7FFF;
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

static inline float bf16_to_f32(uint16_t bf) {
    uint32_t u = static_cast<uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// ---------------------------------------------------------------------------
// Benchmark parameters
// ---------------------------------------------------------------------------
static constexpr int WARMUP            = 100;
static constexpr int BATCH_SIZE        = 64;   // calls per timed batch
static constexpr int BATCHES_PER_TRIAL = 200;  // batches per trial
static constexpr int N_TRIALS          = 30;   // independent trials
static constexpr double T_CRIT_29      = 2.045; // t_{29, 0.025} for 95% CI

struct BenchShape {
    size_t data_size;
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    const char* name;
};

static const BenchShape shapes[] = {
    // F32 shapes
    {256,   ov::element::f32, ov::element::f32, "small_256_f32"},
    {4096,  ov::element::f32, ov::element::f32, "llama7b_f32"},
    {5120,  ov::element::f32, ov::element::f32, "llama13b_f32"},
    {8192,  ov::element::f32, ov::element::f32, "large_8192_f32"},
    {16384, ov::element::f32, ov::element::f32, "xlarge_16384_f32"},
    // BF16 shapes - half the bytes, different cache pressure
    {4096,  ov::element::bf16, ov::element::bf16, "llama7b_bf16"},
    {5120,  ov::element::bf16, ov::element::bf16, "llama13b_bf16"},
    {8192,  ov::element::bf16, ov::element::bf16, "large_8192_bf16"},
};

struct BenchResult {
    const char* name;
    size_t data_size;
    const char* precision;
    double mean_ns;
    double ci95_ns;
    double min_ns;
    double p50_ns;
    double max_abs_error;
    int trials;
    bool valid;
};

// ---------------------------------------------------------------------------
// Reference RMSNorm in f32 (for correctness validation)
// ---------------------------------------------------------------------------
static void reference_rmsnorm(const float* src, const float* scale,
                              float* dst, size_t n, float eps) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum_sq += static_cast<double>(src[i]) * static_cast<double>(src[i]);
    }
    double mean_sq = sum_sq / static_cast<double>(n);
    double rsqrt = 1.0 / std::sqrt(mean_sq + static_cast<double>(eps));
    for (size_t i = 0; i < n; i++) {
        dst[i] = static_cast<float>(static_cast<double>(src[i]) * rsqrt * static_cast<double>(scale[i]));
    }
}

// ---------------------------------------------------------------------------
// IQR-trimmed mean (robust to OS scheduling spikes)
// ---------------------------------------------------------------------------
static double trimmed_mean(std::vector<double>& data) {
    std::sort(data.begin(), data.end());
    size_t n = data.size();
    double q1 = data[n / 4];
    double q3 = data[3 * n / 4];
    double iqr = q3 - q1;
    double upper = q3 + 1.5 * iqr;

    double sum = 0.0;
    int count = 0;
    for (double v : data) {
        if (v <= upper) {
            sum += v;
            count++;
        }
    }
    return count > 0 ? sum / count : data[n / 2];
}

// ---------------------------------------------------------------------------
// Run benchmark for a single shape
// ---------------------------------------------------------------------------
static bool run_bench(const BenchShape& shape, BenchResult& result) {
    result.name = shape.name;
    result.data_size = shape.data_size;
    result.precision = (shape.src_prc == ov::element::bf16) ? "bf16" : "f32";
    result.trials = N_TRIALS;
    result.valid = false;

    // --- kernel creation ---
    jit_rms_compile_params jcp{};
    jcp.src_prc    = shape.src_prc;
    jcp.dst_prc    = shape.dst_prc;
    jcp.data_size  = shape.data_size;
    jcp.scale_size = shape.data_size;
    jcp.eps        = 1e-6f;

    auto kernel = std::make_shared<jit_rms_kernel<x64::avx2>>(jcp);
    if (!kernel) {
        std::fprintf(stderr, "[ERROR] Failed to create kernel for %s\n", shape.name);
        return false;
    }
    auto status = kernel->create_kernel();
    if (status != dnnl::impl::status::success) {
        std::fprintf(stderr, "[ERROR] JIT compilation failed for %s (status=%d)\n",
                     shape.name, static_cast<int>(status));
        return false;
    }

    // --- allocate aligned buffers ---
    const size_t src_elem_size = shape.src_prc.size();
    const size_t dst_elem_size = shape.dst_prc.size();
    const size_t src_bytes   = shape.data_size * src_elem_size;
    const size_t dst_bytes   = shape.data_size * dst_elem_size;
    const size_t scale_bytes = shape.data_size * sizeof(float);

    uint8_t* src   = static_cast<uint8_t*>(std::aligned_alloc(64, src_bytes));
    uint8_t* dst   = static_cast<uint8_t*>(std::aligned_alloc(64, dst_bytes));
    float*   scale = static_cast<float*>(std::aligned_alloc(64, scale_bytes));
    // f32 reference buffers for correctness check
    float* src_f32 = static_cast<float*>(std::aligned_alloc(64, shape.data_size * sizeof(float)));
    float* ref_out = static_cast<float*>(std::aligned_alloc(64, shape.data_size * sizeof(float)));

    if (!src || !dst || !scale || !src_f32 || !ref_out) {
        std::fprintf(stderr, "[ERROR] Allocation failed for %s\n", shape.name);
        std::free(src); std::free(dst); std::free(scale);
        std::free(src_f32); std::free(ref_out);
        return false;
    }

    // Fill with realistic data
    for (size_t i = 0; i < shape.data_size; i++) {
        float val = 0.01f * (static_cast<float>(i % 100) - 50.0f);
        src_f32[i] = val;
        scale[i] = 1.0f;

        if (shape.src_prc == ov::element::bf16) {
            uint16_t bf = f32_to_bf16(val);
            std::memcpy(src + i * 2, &bf, sizeof(bf));
        } else {
            std::memcpy(src + i * 4, &val, sizeof(val));
        }
    }
    std::memset(dst, 0, dst_bytes);

    jit_rms_call_args args{};
    args.src   = src;
    args.scale = scale;
    args.dst   = dst;

    // --- warmup ---
    for (int i = 0; i < WARMUP; i++) {
        (*kernel)(&args);
    }

    // --- correctness validation ---
    // For bf16 input, the kernel loads bf16->f32 internally, so the reference
    // should use the bf16-roundtripped values as input
    if (shape.src_prc == ov::element::bf16) {
        for (size_t i = 0; i < shape.data_size; i++) {
            uint16_t bf;
            std::memcpy(&bf, src + i * 2, sizeof(bf));
            src_f32[i] = bf16_to_f32(bf);
        }
    }
    reference_rmsnorm(src_f32, scale, ref_out, shape.data_size, jcp.eps);

    double max_err = 0.0;
    for (size_t i = 0; i < shape.data_size; i++) {
        float kernel_val;
        if (shape.dst_prc == ov::element::bf16) {
            uint16_t bf;
            std::memcpy(&bf, dst + i * 2, sizeof(bf));
            kernel_val = bf16_to_f32(bf);
        } else {
            std::memcpy(&kernel_val, dst + i * 4, sizeof(kernel_val));
        }
        double err = std::fabs(static_cast<double>(kernel_val) - static_cast<double>(ref_out[i]));
        if (err > max_err) max_err = err;
    }
    result.max_abs_error = max_err;

    double err_threshold = (shape.dst_prc == ov::element::bf16) ? 1e-2 : 1e-5;
    if (max_err > err_threshold) {
        std::fprintf(stderr, "[WARN] %s: max_abs_error=%.2e exceeds threshold %.0e\n",
                     shape.name, max_err, err_threshold);
    }

    // --- batched timed runs with multiple trials ---
    std::vector<double> trial_means(N_TRIALS);

    for (int t = 0; t < N_TRIALS; t++) {
        std::vector<double> batch_latencies(BATCHES_PER_TRIAL);

        for (int b = 0; b < BATCHES_PER_TRIAL; b++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int k = 0; k < BATCH_SIZE; k++) {
                (*kernel)(&args);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            batch_latencies[b] = std::chrono::duration<double, std::nano>(t1 - t0).count()
                                 / BATCH_SIZE;
        }

        trial_means[t] = trimmed_mean(batch_latencies);
    }

    // --- statistics across trials ---
    std::sort(trial_means.begin(), trial_means.end());
    double sum  = std::accumulate(trial_means.begin(), trial_means.end(), 0.0);
    double mean = sum / N_TRIALS;

    double sq_diff_sum = 0.0;
    for (double tm : trial_means) {
        double d = tm - mean;
        sq_diff_sum += d * d;
    }
    double std_dev = std::sqrt(sq_diff_sum / (N_TRIALS - 1));
    double std_err = std_dev / std::sqrt(static_cast<double>(N_TRIALS));
    double ci95    = T_CRIT_29 * std_err;

    double min_val = trial_means[0];
    double p50     = trial_means[N_TRIALS / 2];

    result.mean_ns = mean;
    result.ci95_ns = ci95;
    result.min_ns  = min_val;
    result.p50_ns  = p50;
    result.valid   = true;

    std::printf("[BENCH] %-22s  data_size=%5zu  mean=%7.1f +/- %4.1f ns  "
                "min=%7.1f ns  p50=%7.1f ns  max_err=%.1e  trials=%d\n",
                shape.name, shape.data_size,
                mean, ci95, min_val, p50, max_err, N_TRIALS);

    std::free(src);
    std::free(dst);
    std::free(scale);
    std::free(src_f32);
    std::free(ref_out);
    return true;
}

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------
static void write_json(const std::vector<BenchResult>& results, const char* filename) {
    FILE* f = std::fopen(filename, "w");
    if (!f) {
        std::fprintf(stderr, "[WARN] Could not open %s for writing\n", filename);
        return;
    }

    // Collect valid results for clean comma handling
    std::vector<const BenchResult*> valid;
    for (const auto& r : results) {
        if (r.valid) valid.push_back(&r);
    }

    std::fprintf(f, "{\n  \"shapes\": [\n");
    for (size_t i = 0; i < valid.size(); i++) {
        const auto& r = *valid[i];
        std::fprintf(f, "    {\n");
        std::fprintf(f, "      \"name\": \"%s\",\n", r.name);
        std::fprintf(f, "      \"data_size\": %zu,\n", r.data_size);
        std::fprintf(f, "      \"precision\": \"%s\",\n", r.precision);
        std::fprintf(f, "      \"mean_ns\": %.1f,\n", r.mean_ns);
        std::fprintf(f, "      \"ci95_ns\": %.1f,\n", r.ci95_ns);
        std::fprintf(f, "      \"min_ns\": %.1f,\n", r.min_ns);
        std::fprintf(f, "      \"p50_ns\": %.1f,\n", r.p50_ns);
        std::fprintf(f, "      \"max_abs_error\": %.6e,\n", r.max_abs_error);
        std::fprintf(f, "      \"trials\": %d\n", r.trials);
        std::fprintf(f, "    }%s\n", (i + 1 < valid.size()) ? "," : "");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("=== RMSNorm JIT Kernel Microbenchmark (AVX2) ===\n");
    std::printf("Batched timing: %d calls/batch, %d batches/trial, %d trials\n",
                BATCH_SIZE, BATCHES_PER_TRIAL, N_TRIALS);
    std::printf("IQR outlier trimming, 95%% CI (t-distribution)\n\n");

    std::vector<BenchResult> results;
    int failures = 0;

    for (const auto& shape : shapes) {
        BenchResult r{};
        if (!run_bench(shape, r))
            failures++;
        results.push_back(r);
    }

    write_json(results, "results.json");
    std::printf("\n[INFO] Results written to results.json\n");
    std::printf("%s\n", failures ? "[FAIL] Some benchmarks failed" : "[OK] All benchmarks passed");
    return failures;
}
