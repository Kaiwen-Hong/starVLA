#!/usr/bin/env python3
"""
Analyze inference delay sweep results for PI (FM), DD soft mask, and DD hard mask.

Reads JSON files from the sweep directory and produces:
  1. Summary table (printed to stdout)
  2. Plot: total inference time vs inference_delay
  3. Plot: effective Hz (frequency) comparison
  4. Plot: per-step breakdown
  5. Plot: number of action model steps vs delay

Usage:
    python realworld/0405-analyze-inference-sweep.py
    python realworld/0405-analyze-inference-sweep.py --sweep_dir /path/to/results
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SWEEP_DIR = "/scratch/wangpc/starVLA/results/benchmark_sweep_0405"

# Series config: (key, file_prefix, label, color, marker)
SERIES = [
    ("pi",      "pi",      "PI (Flow-Matching, ΠGDM)", "#4C72B0", "o"),
    ("dd",      "dd",      "DD (MaskGIT, natural)",      "#DD8452", "s"),
    ("dd_hard", "dd_hard", "DD (MaskGIT, hard mask)",   "#55A868", "D"),
]


def load_sweep(sweep_dir: str) -> dict:
    """Load all JSON results from the sweep directory."""
    sweep_dir = Path(sweep_dir)
    results = {key: {} for key, *_ in SERIES}

    for path in sorted(sweep_dir.glob("*.json")):
        name = path.stem
        if name.startswith("sweep_"):
            continue

        with open(path) as f:
            data = json.load(f)

        # Match file prefix to series key
        for key, prefix, *_ in SERIES:
            if name.startswith(prefix + "_delay"):
                delay_str = name[len(prefix) + len("_delay"):]
                try:
                    delay = int(delay_str)
                except ValueError:
                    continue
                results[key][delay] = data
                break

    return results


def extract_timing(data: dict, delay: int) -> dict:
    """Extract key timing metrics from a single benchmark result."""
    out = {"delay": delay}

    if delay == 0:
        key = "predict_action"
    else:
        key = "predict_action_realtime" if "predict_action_realtime" in data else "predict_action"

    timing = data.get(key, {})
    total = timing.get("total_ms", {})

    out["total_mean_ms"] = total.get("mean_ms", float("nan"))
    out["total_median_ms"] = total.get("median_ms", float("nan"))
    out["total_p95_ms"] = total.get("p95_ms", float("nan"))
    out["total_std_ms"] = total.get("std_ms", float("nan"))

    # Per-step times
    step_keys = sorted(
        [k for k in timing if k.startswith("am_step_") and k.count("_") == 2],
        key=lambda k: int(k.rsplit("_", 1)[-1]),
    )
    step_means = [timing[k]["mean_ms"] for k in step_keys]
    out["num_steps"] = len(step_keys)
    out["per_step_mean_ms"] = np.mean(step_means) if step_means else float("nan")
    out["total_steps_ms"] = sum(step_means)

    vlm = timing.get("vlm_forward", {})
    out["vlm_forward_ms"] = vlm.get("mean_ms", float("nan"))

    am_init = timing.get("am_init", {})
    out["am_init_ms"] = am_init.get("mean_ms", float("nan"))

    out["num_inference_steps"] = data.get("num_inference_steps", 0)
    out["num_inference_steps_rtc"] = data.get("num_inference_steps_rtc", 0)
    out["action_horizon"] = data.get("action_horizon", 0)
    out["rtc_mode"] = data.get("rtc_mode", "N/A")
    out["mask_mode"] = data.get("mask_mode", "N/A")

    return out


def print_table(all_rows: dict):
    """Print comparison table for all series."""
    # Collect all delays
    all_delays = sorted(set(d for rows in all_rows.values() for r in rows for d in [r["delay"]]))

    # Build header
    col_headers = []
    for key, _, label, *_ in SERIES:
        if all_rows.get(key):
            short = label.split("(")[1].rstrip(")") if "(" in label else label
            col_headers.append((key, short))

    header_parts = [f"{'delay':>5}"]
    for _, short in col_headers:
        header_parts.append(f"{'total':>9} {'#steps':>6} {'Hz':>6}")
    header = "  |  ".join(header_parts)

    label_parts = [f"{'':>5}"]
    for _, short in col_headers:
        label_parts.append(f"{short:>23}")
    label_line = "  |  ".join(label_parts)

    sep = "-" * len(header)

    print(f"\n{sep}")
    print("  Inference Delay Sweep")
    print(sep)
    print(label_line)
    print(header)
    print(sep)

    rows_by_key = {}
    for key, _, *_ in SERIES:
        rows_by_key[key] = {r["delay"]: r for r in all_rows.get(key, [])}

    for d in all_delays:
        parts = [f"{d:>5}"]
        for key, _ in col_headers:
            r = rows_by_key[key].get(d)
            if r:
                total = f"{r['total_mean_ms']:.0f} ms"
                nsteps = f"{r['num_steps']}"
                hz = f"{1000/r['total_mean_ms']:.1f}" if r['total_mean_ms'] > 0 else "N/A"
                parts.append(f"{total:>9} {nsteps:>6} {hz:>6}")
            else:
                parts.append(f"{'N/A':>9} {'N/A':>6} {'N/A':>6}")
        print("  |  ".join(parts))

    print(sep)

    for key, _, label, *_ in SERIES:
        rows = all_rows.get(key, [])
        if rows:
            r = rows[0]
            info = f"  {label}: H={r['action_horizon']}, base_steps={r['num_inference_steps']}"
            if r.get("rtc_mode", "N/A") != "N/A":
                info += f", rtc_mode={r['rtc_mode']}"
            print(info)
    print()


def plot_total_time(all_rows, out_path):
    """Plot total inference time vs delay."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for key, _, label, color, marker in SERIES:
        rows = all_rows.get(key, [])
        if not rows:
            continue
        delays = [r["delay"] for r in rows]
        means = [r["total_mean_ms"] for r in rows]
        stds = [r["total_std_ms"] for r in rows]
        ax.errorbar(delays, means, yerr=stds, marker=marker, capsize=4,
                     label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Inference Delay (n_actions)", fontsize=13)
    ax.set_ylabel("Total Inference Time (ms)", fontsize=13)
    ax.set_title("Total Inference Time vs Execution Horizon", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 9))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_hz(all_rows, out_path):
    """Plot effective Hz vs delay."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for key, _, label, color, marker in SERIES:
        rows = all_rows.get(key, [])
        if not rows:
            continue
        delays = [r["delay"] for r in rows]
        hz = [1000 / r["total_mean_ms"] if r["total_mean_ms"] > 0 else 0 for r in rows]
        ax.plot(delays, hz, marker=marker, label=label, color=color,
                linewidth=2, markersize=8)

    ax.set_xlabel("Inference Delay (n_actions)", fontsize=13)
    ax.set_ylabel("Frequency (Hz)", fontsize=13)
    ax.set_title("Inference Frequency vs Execution Horizon", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 9))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_breakdown(all_rows, out_path):
    """Stacked bar: VLM forward + action model steps + other."""
    active = [(key, label, color) for key, _, label, color, _ in SERIES if all_rows.get(key)]
    n_panels = len(active)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (key, label, color) in zip(axes, active):
        rows = all_rows[key]
        delays = [r["delay"] for r in rows]
        vlm = [r["vlm_forward_ms"] for r in rows]
        steps = [r["total_steps_ms"] for r in rows]
        other = [max(0, r["total_mean_ms"] - r["vlm_forward_ms"] - r["total_steps_ms"])
                 for r in rows]

        x = np.arange(len(delays))
        w = 0.6

        ax.bar(x, vlm, w, label="VLM forward", color="#4C72B0")
        ax.bar(x, steps, w, bottom=vlm, label="Action model steps", color="#DD8452")
        ax.bar(x, other, w, bottom=[v + s for v, s in zip(vlm, steps)],
               label="Other (preproc, init, decode)", color="#55A868")

        ax.set_xticks(x)
        ax.set_xticklabels(delays)
        ax.set_xlabel("Inference Delay (n_actions)", fontsize=12)
        ax.set_ylabel("Time (ms)", fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        for i, r in enumerate(rows):
            ax.text(i, r["total_mean_ms"] + 2, f"{r['total_mean_ms']:.0f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Inference Time Breakdown", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_num_steps(all_rows, out_path):
    """Plot number of action model steps vs delay."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for key, _, label, color, marker in SERIES:
        rows = all_rows.get(key, [])
        if not rows:
            continue
        delays = [r["delay"] for r in rows]
        nsteps = [r["num_steps"] for r in rows]
        ax.plot(delays, nsteps, marker=marker, label=label, color=color,
                linewidth=2, markersize=8)

    ax.set_xlabel("Inference Delay (n_actions)", fontsize=13)
    ax.set_ylabel("Number of Action Model Steps", fontsize=13)
    ax.set_title("Action Model Steps vs Execution Horizon", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 9))
    ax.set_yticks(range(0, 10))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze inference delay sweep")
    parser.add_argument("--sweep_dir", type=str, default=DEFAULT_SWEEP_DIR)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Sweep directory not found: {sweep_dir}")
        print("Run the sweep first: python realworld/0405-sweep-inference-delay.py")
        return

    results = load_sweep(args.sweep_dir)

    all_rows = {}
    for key, *_ in SERIES:
        if results[key]:
            all_rows[key] = [extract_timing(results[key][d], d) for d in sorted(results[key])]

    if not any(all_rows.values()):
        print("No results found.")
        return

    # Print summary table
    print_table(all_rows)

    # Save combined JSON
    combined = {}
    for key, rows in all_rows.items():
        combined[key] = {r["delay"]: r for r in rows}
    combined_path = sweep_dir / "sweep_summary.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=float)
    print(f"  Saved summary: {combined_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_total_time(all_rows, sweep_dir / "plot_total_time.png")
    plot_hz(all_rows, sweep_dir / "plot_hz.png")
    plot_breakdown(all_rows, sweep_dir / "plot_breakdown.png")
    plot_num_steps(all_rows, sweep_dir / "plot_num_steps.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
