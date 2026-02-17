#!/usr/bin/env python3
"""
Randomized interleaved A/B benchmark for GPU MVN 2-pass optimization.

Uses two pre-built OpenVINO directories (optimized vs baseline). On each trial,
randomly selects which build to benchmark, then runs benchmark_app on all models.
This eliminates systematic temporal bias present in sequential designs.

Usage:
    uv run python run_benchmark.py \
        --build-optimized /path/to/build \
        --build-baseline /path/to/build_reference \
        --model-dir ../models \
        --trials 24 --niter 500 --seed 42 \
        --output ../results/randomized_ab.csv
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def find_models(model_dir: Path) -> list[tuple[str, Path]]:
    """Find all .xml model files in directory, return (name, path) pairs."""
    models = []
    for xml_file in sorted(model_dir.glob("*.xml")):
        name = xml_file.stem
        models.append((name, xml_file))
    return models


def run_benchmark_app(benchmark_app: str, model_path: Path, ld_library_path: str,
                      niter: int) -> dict:
    """Run benchmark_app on a model and parse JSON output."""
    with tempfile.TemporaryDirectory() as report_dir:
        cmd = [
            benchmark_app,
            "-m", str(model_path),
            "-d", "GPU",
            "-hint", "latency",
            "-niter", str(niter),
            "-report_type", "average_counters",
            "-report_folder", report_dir,
            "-json_stats",
        ]

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = ld_library_path

        result = subprocess.run(
            cmd, env=env, timeout=600, capture_output=True, text=True,
        )

        if result.returncode != 0:
            print(f"  benchmark_app FAILED (rc={result.returncode})", file=sys.stderr)
            print(f"  stderr: {result.stderr[:500]}", file=sys.stderr)
            return {"error": result.stderr[:500]}

        report_path = Path(report_dir) / "benchmark_report.json"
        if not report_path.exists():
            return {"error": "benchmark_report.json not found"}

        with open(report_path) as f:
            report = json.load(f)

        counters_path = Path(report_dir) / "benchmark_average_counters_report.json"
        counters = None
        if counters_path.exists():
            with open(counters_path) as f:
                counters = json.load(f)

        return {
            "report": report,
            "counters": counters,
            "stdout": result.stdout,
        }


def extract_metrics(data: dict) -> dict:
    """Extract key metrics from benchmark_app JSON output."""
    report = data["report"]
    counters = data.get("counters")

    exec_results = report.get("execution_results", {})

    metrics = {
        "latency_median_ms": exec_results.get("latency_median", 0.0),
        "latency_avg_ms": exec_results.get("latency_avg", 0.0),
        "latency_min_ms": exec_results.get("latency_min", 0.0),
        "throughput_fps": exec_results.get("throughput", 0.0),
        "iterations": exec_results.get("iterations_num", 0),
    }

    mvn_total_ms = 0.0
    mvn_count = 0
    total_exec_ms = 0.0

    if counters and "avg_performance" in counters:
        nodes = counters["avg_performance"].get("nodes", [])
        for node in nodes:
            if node.get("status") == "EXECUTED":
                real_time = node.get("real_time", 0.0)
                total_exec_ms += real_time
                node_type = node.get("node_type", "").lower()
                node_name = node.get("name", "").lower()
                if "mvn" in node_type or "mvn" in node_name:
                    mvn_total_ms += real_time
                    mvn_count += 1

    metrics["mvn_total_ms"] = round(mvn_total_ms, 4)
    metrics["mvn_count"] = mvn_count
    metrics["total_exec_ms"] = round(total_exec_ms, 4)
    if total_exec_ms > 0:
        metrics["mvn_pct"] = round(mvn_total_ms / total_exec_ms * 100, 2)
    else:
        metrics["mvn_pct"] = 0.0

    return metrics


def get_build_info(build_dir: Path) -> str:
    """Get git branch/commit info for a build directory."""
    ov_dir = build_dir / "openvino"
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=ov_dir, capture_output=True, text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=ov_dir, capture_output=True, text=True,
        ).stdout.strip()
        return f"{branch} ({commit})"
    except Exception:
        return "unknown"


def resolve_build(build_dir: Path) -> tuple[str, str]:
    """Validate build dir and return (benchmark_app_path, ld_library_path)."""
    ov_dir = build_dir / "openvino"
    benchmark_app = str(ov_dir / "bin" / "intel64" / "Release" / "benchmark_app")
    if not Path(benchmark_app).exists():
        print(f"ERROR: benchmark_app not found: {benchmark_app}", file=sys.stderr)
        sys.exit(1)

    ld_library_path = ":".join([
        str(ov_dir / "bin" / "intel64" / "Release"),
        str(ov_dir / "temp" / "Linux_x86_64" / "tbb" / "lib"),
        str(ov_dir / "build" / "lib"),
        os.environ.get("LD_LIBRARY_PATH", ""),
    ])
    return benchmark_app, ld_library_path


def main():
    parser = argparse.ArgumentParser(
        description="Randomized interleaved A/B benchmark for GPU MVN 2-pass optimization"
    )
    parser.add_argument(
        "--build-optimized", required=True,
        help="Absolute path to optimized build directory"
    )
    parser.add_argument(
        "--build-baseline", required=True,
        help="Absolute path to baseline build directory"
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Directory containing model IR files (.xml/.bin)"
    )
    parser.add_argument(
        "--trials", type=int, default=24,
        help="Number of trials (each trial benchmarks all models with one build) (default: 24)"
    )
    parser.add_argument(
        "--niter", type=int, default=500,
        help="Number of iterations per benchmark_app invocation (default: 500)"
    )
    parser.add_argument(
        "--cooldown", type=int, default=5,
        help="Cooldown seconds between trials (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for trial schedule (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output CSV path"
    )
    args = parser.parse_args()

    build_opt = Path(args.build_optimized)
    build_base = Path(args.build_baseline)
    model_dir = Path(args.model_dir)

    # Validate builds
    opt_app, opt_ld = resolve_build(build_opt)
    base_app, base_ld = resolve_build(build_base)

    # Validate models
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    models = find_models(model_dir)
    if not models:
        print(f"ERROR: No .xml model files in {model_dir}", file=sys.stderr)
        sys.exit(1)

    # Build info
    opt_info = get_build_info(build_opt)
    base_info = get_build_info(build_base)

    # Generate balanced randomized trial schedule (equal split, then shuffle)
    rng = random.Random(args.seed)
    half = args.trials // 2
    schedule = ["optimized"] * half + ["baseline"] * (args.trials - half)
    rng.shuffle(schedule)

    # Count per build
    n_opt = schedule.count("optimized")
    n_base = schedule.count("baseline")

    csv_file = Path(args.output)

    print(f"{'=' * 70}")
    print(f"Randomized Interleaved A/B Benchmark")
    print(f"{'=' * 70}")
    print(f"Optimized: {build_opt}")
    print(f"           {opt_info}")
    print(f"Baseline:  {build_base}")
    print(f"           {base_info}")
    print(f"Models:    {model_dir} ({len(models)} models)")
    print(f"           {', '.join(n for n, _ in models)}")
    print(f"Trials:    {args.trials} (optimized={n_opt}, baseline={n_base})")
    print(f"Iters:     {args.niter} per run")
    print(f"Cooldown:  {args.cooldown}s between trials")
    print(f"Seed:      {args.seed}")
    print(f"Output:    {csv_file}")
    print(f"Schedule:  {' '.join('O' if s == 'optimized' else 'B' for s in schedule)}")
    print(f"{'=' * 70}\n")

    csv_file.parent.mkdir(parents=True, exist_ok=True)

    success_counts = defaultdict(int)

    fieldnames = [
        'trial', 'build_tag', 'model', 'latency_median_ms', 'latency_avg_ms',
        'latency_min_ms', 'throughput_fps', 'iterations', 'mvn_total_ms',
        'mvn_count', 'total_exec_ms', 'mvn_pct', 'timestamp',
    ]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial_idx, build_tag in enumerate(schedule, 1):
            if build_tag == "optimized":
                app, ld = opt_app, opt_ld
            else:
                app, ld = base_app, base_ld

            print(f"\n{'─' * 50}")
            print(f"Trial {trial_idx}/{args.trials} — {build_tag.upper()}")
            print(f"{'─' * 50}")

            for model_name, model_path in models:
                print(f"  {model_name}: benchmarking ({args.niter} iters)...")

                data = run_benchmark_app(app, model_path, ld, args.niter)

                if "error" in data:
                    print(f"  {model_name}: FAILED — {data['error'][:100]}", file=sys.stderr)
                    sys.exit(1)

                metrics = extract_metrics(data)

                row = {
                    'trial': trial_idx,
                    'build_tag': build_tag,
                    'model': model_name,
                    'timestamp': datetime.now().isoformat(),
                    **metrics,
                }
                writer.writerow(row)
                f.flush()
                success_counts[(build_tag, model_name)] += 1

                print(f"  {model_name}: median={metrics['latency_median_ms']:.2f}ms, "
                      f"throughput={metrics['throughput_fps']:.1f}fps, "
                      f"MVN={metrics['mvn_total_ms']:.3f}ms ({metrics['mvn_pct']:.1f}%)")

            # Cooldown between trials (skip after last)
            if trial_idx < args.trials:
                print(f"  cooldown {args.cooldown}s...")
                time.sleep(args.cooldown)

    # Check for models with no successful results (comparison impossible)
    missing = []
    for model_name, _ in models:
        for tag in ["optimized", "baseline"]:
            if success_counts[(tag, model_name)] == 0:
                missing.append((tag, model_name))

    if missing:
        print(f"\n{'=' * 70}", file=sys.stderr)
        print("ERROR: Some (build, model) pairs had 0 successful results:", file=sys.stderr)
        for tag, model_name in missing:
            print(f"  {tag:>10s} / {model_name}: 0 successes", file=sys.stderr)
        print("Comparison is impossible for these models.", file=sys.stderr)
        print(f"{'=' * 70}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"Complete! Results: {csv_file}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
