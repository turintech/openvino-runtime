#!/usr/bin/env python3
"""
Statistical comparison of randomized interleaved A/B benchmark results.

Reads a single CSV produced by run_benchmark.py (with build_tag column),
splits by build tag, and computes per-model Welch's t-test on latency and MVN timing.

Groups models into:
  - MVN models (expect improvement): unet3d_encoder, segnet_highres
  - Control model (expect no change): segnet_control

Usage:
    uv run python compare_results.py \
        --input ../results/randomized_ab.csv \
        --output ../results/comparison.csv
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_CONTROL_MODELS = ["segnet_control"]


def read_tagged_csv(csv_path: Path) -> dict:
    """Read tagged benchmark CSV, return {build_tag: {model: {metric: [values]}}}."""
    builds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row['build_tag']
            model = row['model']
            builds[tag][model]['latency_median_ms'].append(float(row['latency_median_ms']))
            builds[tag][model]['latency_avg_ms'].append(float(row['latency_avg_ms']))
            builds[tag][model]['throughput_fps'].append(float(row['throughput_fps']))
            builds[tag][model]['mvn_total_ms'].append(float(row['mvn_total_ms']))
            builds[tag][model]['total_exec_ms'].append(float(row['total_exec_ms']))
    return dict(builds)


def stats(values: list[float]) -> dict:
    """Compute mean, stdev, n."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "stdev": 0.0, "n": 0}
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        stdev = math.sqrt(variance)
    else:
        stdev = 0.0
    return {"mean": mean, "stdev": stdev, "n": n}


def welch_t_test(s1: dict, s2: dict) -> float:
    """Welch's t-test. Returns t-statistic (positive = s1 > s2 = improvement)."""
    if s1["n"] < 2 or s2["n"] < 2:
        return 0.0
    se1 = s1["stdev"] ** 2 / s1["n"]
    se2 = s2["stdev"] ** 2 / s2["n"]
    se = math.sqrt(se1 + se2)
    if se == 0:
        return 0.0
    return (s1["mean"] - s2["mean"]) / se


def main():
    parser = argparse.ArgumentParser(
        description="Compare randomized A/B benchmark results"
    )
    parser.add_argument("--input", required=True, help="Path to tagged CSV from run_benchmark.py")
    parser.add_argument("--output", type=str, default=None, help="Output comparison CSV")
    parser.add_argument("--control-models", nargs="*", default=DEFAULT_CONTROL_MODELS,
                        help="Model names to treat as controls (default: segnet_control)")
    args = parser.parse_args()

    CONTROL_MODELS = set(args.control_models)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input CSV not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = input_path.parent / "comparison.csv"

    data = read_tagged_csv(input_path)

    if "baseline" not in data or "optimized" not in data:
        print(f"ERROR: CSV must contain both 'baseline' and 'optimized' build_tag values", file=sys.stderr)
        print(f"  Found tags: {list(data.keys())}", file=sys.stderr)
        sys.exit(1)

    baseline_data = data["baseline"]
    optimized_data = data["optimized"]

    # Count trials per build
    n_baseline_trials = 0
    n_optimized_trials = 0
    with open(input_path) as f:
        reader = csv.DictReader(f)
        trials_seen = {"baseline": set(), "optimized": set()}
        for row in reader:
            trials_seen[row['build_tag']].add(int(row['trial']))
    n_baseline_trials = len(trials_seen["baseline"])
    n_optimized_trials = len(trials_seen["optimized"])

    # Merge model names preserving order
    all_models = list(baseline_data.keys())
    for k in optimized_data:
        if k not in all_models:
            all_models.append(k)

    mvn_models = [m for m in all_models if m not in CONTROL_MODELS]
    ctrl_models = [m for m in all_models if m in CONTROL_MODELS]

    print(f"{'=' * 95}")
    print(f"Randomized A/B Benchmark: Baseline vs Optimized")
    print(f"{'=' * 95}")
    print(f"Input:     {input_path}")
    print(f"Trials:    {n_baseline_trials} baseline, {n_optimized_trials} optimized")
    print(f"Models:    {len(all_models)} ({len(mvn_models)} MVN, {len(ctrl_models)} control)")

    header_latency = (
        f"{'Model':<20} "
        f"{'Base med':>10} "
        f"{'Opt med':>10} "
        f"{'Diff ms':>10} "
        f"{'Change%':>9} "
        f"{'t-stat':>8} "
        f"{'Sig?':>5} "
        f"{'n_b':>4} "
        f"{'n_o':>4}"
    )

    header_mvn = (
        f"{'Model':<20} "
        f"{'Base MVN':>10} "
        f"{'Opt MVN':>10} "
        f"{'Diff ms':>10} "
        f"{'Change%':>9} "
        f"{'MVN% base':>10} "
        f"{'MVN% opt':>10}"
    )

    rows = []

    def print_latency_group(group_name: str, model_list: list[str]):
        print(f"\n--- {group_name}: Median Latency ---")
        print(header_latency)
        print("-" * 95)

        improved = 0
        significant = 0

        for model in model_list:
            b_data = baseline_data.get(model, {})
            o_data = optimized_data.get(model, {})
            b_vals = b_data.get('latency_median_ms', [])
            o_vals = o_data.get('latency_median_ms', [])

            if not b_vals or not o_vals:
                print(f"{model:<20} {'N/A':>10} {'N/A':>10}")
                continue

            b_stats = stats(b_vals)
            o_stats = stats(o_vals)

            diff = o_stats["mean"] - b_stats["mean"]
            pct_change = diff / b_stats["mean"] * 100 if b_stats["mean"] != 0 else 0.0
            t = welch_t_test(b_stats, o_stats)
            is_sig = abs(t) > 2.0
            is_improved = diff < 0

            if is_improved:
                improved += 1
            if is_sig and is_improved:
                significant += 1

            print(
                f"{model:<20} "
                f"{b_stats['mean']:>9.2f}  "
                f"{o_stats['mean']:>9.2f}  "
                f"{diff:>+9.3f}  "
                f"{pct_change:>+7.2f}%  "
                f"{t:>7.2f}  "
                f"{'YES' if is_sig else 'no':>3}  "
                f"{b_stats['n']:>3} "
                f"{o_stats['n']:>3}"
            )

            # MVN timing
            b_mvn = stats(b_data.get('mvn_total_ms', []))
            o_mvn = stats(o_data.get('mvn_total_ms', []))
            b_exec = stats(b_data.get('total_exec_ms', []))
            o_exec = stats(o_data.get('total_exec_ms', []))

            mvn_diff = o_mvn["mean"] - b_mvn["mean"]
            mvn_pct_change = mvn_diff / b_mvn["mean"] * 100 if b_mvn["mean"] != 0 else 0.0
            mvn_t = welch_t_test(b_mvn, o_mvn)

            b_mvn_pct = b_mvn["mean"] / b_exec["mean"] * 100 if b_exec["mean"] != 0 else 0.0
            o_mvn_pct = o_mvn["mean"] / o_exec["mean"] * 100 if o_exec["mean"] != 0 else 0.0

            rows.append({
                'model': model,
                'group': group_name,
                'baseline_latency_median_ms': round(b_stats["mean"], 3),
                'baseline_latency_stdev_ms': round(b_stats["stdev"], 3),
                'optimized_latency_median_ms': round(o_stats["mean"], 3),
                'optimized_latency_stdev_ms': round(o_stats["stdev"], 3),
                'latency_diff_ms': round(diff, 4),
                'latency_change_pct': round(pct_change, 3),
                'latency_t_stat': round(t, 3),
                'latency_significant': is_sig,
                'baseline_mvn_ms': round(b_mvn["mean"], 4),
                'optimized_mvn_ms': round(o_mvn["mean"], 4),
                'mvn_diff_ms': round(mvn_diff, 4),
                'mvn_change_pct': round(mvn_pct_change, 3),
                'mvn_t_stat': round(mvn_t, 3),
                'baseline_mvn_pct_of_exec': round(b_mvn_pct, 2),
                'optimized_mvn_pct_of_exec': round(o_mvn_pct, 2),
                'n_baseline': b_stats["n"],
                'n_optimized': o_stats["n"],
            })

        return improved, significant, len(model_list)

    # Print latency comparison for each group
    m_improved, m_sig, m_total = print_latency_group(
        "MVN models (expect improvement)", mvn_models)
    c_improved, c_sig, c_total = print_latency_group(
        "Control models (expect no change)", ctrl_models)

    # Print MVN kernel timing breakdown
    print(f"\n--- MVN Kernel Timing Breakdown ---")
    print(header_mvn)
    print("-" * 95)

    for model in all_models:
        b_data = baseline_data.get(model, {})
        o_data = optimized_data.get(model, {})

        b_mvn = stats(b_data.get('mvn_total_ms', []))
        o_mvn = stats(o_data.get('mvn_total_ms', []))
        b_exec = stats(b_data.get('total_exec_ms', []))
        o_exec = stats(o_data.get('total_exec_ms', []))

        if b_mvn["n"] == 0 or o_mvn["n"] == 0:
            continue

        mvn_diff = o_mvn["mean"] - b_mvn["mean"]
        mvn_pct = mvn_diff / b_mvn["mean"] * 100 if b_mvn["mean"] != 0 else 0.0
        b_mvn_pct = b_mvn["mean"] / b_exec["mean"] * 100 if b_exec["mean"] != 0 else 0.0
        o_mvn_pct = o_mvn["mean"] / o_exec["mean"] * 100 if o_exec["mean"] != 0 else 0.0

        marker = " *" if model in CONTROL_MODELS else ""
        print(
            f"{model:<20} "
            f"{b_mvn['mean']:>9.3f}  "
            f"{o_mvn['mean']:>9.3f}  "
            f"{mvn_diff:>+9.4f}  "
            f"{mvn_pct:>+7.2f}%  "
            f"{b_mvn_pct:>8.1f}%  "
            f"{o_mvn_pct:>8.1f}%{marker}"
        )

    # Summary — verdict based on MVN kernel t-stat, not E2E latency
    mvn_kernel_improved = sum(
        1 for r in rows
        if r['group'].startswith('MVN')
        and abs(r['mvn_t_stat']) > 2.0
        and r['mvn_change_pct'] < 0
    )
    mvn_kernel_regressed = sum(
        1 for r in rows
        if r['group'].startswith('MVN')
        and abs(r['mvn_t_stat']) > 2.0
        and r['mvn_change_pct'] > 0
    )
    mvn_kernel_total = sum(1 for r in rows if r['group'].startswith('MVN'))
    ctrl_stable = c_sig == 0

    print(f"\n{'=' * 95}")
    print(f"SUMMARY")
    print(f"{'=' * 95}")
    print(f"  Design:         Randomized interleaved A/B")
    print(f"  Trials:         {n_baseline_trials} baseline, {n_optimized_trials} optimized")
    print(f"  MVN kernel:     {mvn_kernel_improved}/{mvn_kernel_total} significantly improved, "
          f"{mvn_kernel_regressed}/{mvn_kernel_total} significantly regressed")
    print(f"  E2E latency:    {m_improved}/{m_total} improved, {m_sig}/{m_total} statistically significant")
    print(f"  Control models: {c_improved}/{c_total} improved, {c_sig}/{c_total} statistically significant")
    print()

    if mvn_kernel_improved >= 1 and mvn_kernel_regressed == 0 and ctrl_stable:
        print("  VERDICT: Optimization VERIFIED")
        print("    - MVN kernel shows significant improvement (|t| > 2.0)")
        print("    - No MVN kernel regressions")
        print("    - Control model shows no significant change")
    elif mvn_kernel_regressed > 0:
        print("  VERDICT: REGRESSION detected")
        print(f"    - {mvn_kernel_regressed} MVN model(s) show significant kernel regression")
    elif mvn_kernel_improved >= 1:
        print("  VERDICT: Optimization LIKELY EFFECTIVE (but control shows unexpected change)")
        print(f"    - {mvn_kernel_improved} MVN model(s) show significant kernel improvement")
        print("    - Control model shows unexpected significant change")
    else:
        print("  VERDICT: INCONCLUSIVE")
        print("    - No MVN kernel metric reached significance (|t| > 2.0)")

    print(f"\n* Significant at p < 0.05 (Welch's t-test, |t| > 2.0)")
    print(f"  Negative diff = faster (improvement)")

    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model', 'group',
            'baseline_latency_median_ms', 'baseline_latency_stdev_ms',
            'optimized_latency_median_ms', 'optimized_latency_stdev_ms',
            'latency_diff_ms', 'latency_change_pct',
            'latency_t_stat', 'latency_significant',
            'baseline_mvn_ms', 'optimized_mvn_ms',
            'mvn_diff_ms', 'mvn_change_pct', 'mvn_t_stat',
            'baseline_mvn_pct_of_exec', 'optimized_mvn_pct_of_exec',
            'n_baseline', 'n_optimized',
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nResults saved: {output_csv}")


if __name__ == "__main__":
    main()
