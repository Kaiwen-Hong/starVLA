#!/usr/bin/env python3
"""
Open-loop MSE evaluation across inference delays for PI and DD.

For delay=0 (sync): predict_action, evaluate all H positions.
For delay>0 (RTC):  predict_action_realtime with GT as prefix,
                    evaluate only the last execution_horizon positions
                    (the positions actually generated, not copied).

Loads each model ONCE, sweeps delay from 0 to max_delay.

Usage:
    python realworld/0405-openloop-mse-vs-delay.py
    python realworld/0405-openloop-mse-vs-delay.py --num_samples 200 --max_delay 8
    python realworld/0405-openloop-mse-vs-delay.py --skip_pi --num_samples 100
"""

import warnings
warnings.filterwarnings("ignore", message=".*video decoding and encoding.*torchvision.*")

import argparse
import json
import sys
import os
import time
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

# Import benchmark module for model/data loading and VLM forward
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "bench", str(REPO_DIR / "realworld" / "0403-benchmarking-inference.py")
)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT_PI = (
    "/scratch/wangpc/starVLA/results/Checkpoints/fastumi_pickandplace_qwenPI_329v4/"
    "checkpoints/steps_15000_pytorch_model.pt"
)
DEFAULT_CHECKPOINT_DD = (
    "/scratch/wangpc/starVLA/results/Checkpoints/fastumi_pickandplace_qwenDiscreteDiffusion_329v4/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
DEFAULT_OUT_DIR = "/scratch/wangpc/starVLA/results/openloop_mse_sweep_0405"

DIM_LABELS = [
    "pos_x", "pos_y", "pos_z",
    "rot6d_0", "rot6d_1", "rot6d_2",
    "rot6d_3", "rot6d_4", "rot6d_5",
    "gripper",
]


# ── Helpers ──────────────────────────────────────────────────────────

class _NoopTimer:
    @contextmanager
    def track(self, name):
        yield


def _valid_mask(gt, rot_slice=slice(3, 9)):
    """Detect padded timesteps: rotation dims all zero = padding."""
    rot = gt[:, rot_slice]
    return ~np.all(np.abs(rot) < 1e-6, axis=1)


def get_pos_real_scale(norm_stats: dict) -> np.ndarray:
    """Return per-dim half-range for position dims 0:3 from norm_stats."""
    key = next(iter(norm_stats.keys()))
    action_stats = norm_stats[key]["action"]
    pos_min = np.array(action_stats["min"][:3], dtype=np.float64)
    pos_max = np.array(action_stats["max"][:3], dtype=np.float64)
    return (pos_max - pos_min) / 2.0


def vlm_forward(model, framework_type, batch):
    """Run VLM forward, return (vl_embs_list, state)."""
    timer = _NoopTimer()
    if "PI" in framework_type:
        return bench._vlm_forward_pi(model, batch, timer)
    else:
        return bench._vlm_forward_dd(model, batch, timer)


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_delay(model, framework_type, dataloader, num_samples,
                   inference_delay, infer_kwargs, pos_scale=None,
                   hard_mask=False):
    """Evaluate open-loop MSE for a single delay setting.

    delay=0: predict_action, evaluate all positions.
    delay>0: predict_action_realtime with GT prefix,
             evaluate last execution_horizon positions only.
    """
    am = model.action_model
    H = am.action_horizon
    execution_horizon = inference_delay if inference_delay > 0 else H
    is_pi = "PI" in framework_type

    all_mse = []
    all_l1 = []
    all_per_dim_mse = []
    all_pos_rmse_cm = []

    count = 0
    from tqdm import tqdm
    for batch in tqdm(dataloader, total=min(num_samples, len(dataloader)),
                      desc=f"delay={inference_delay}", leave=False):
        if count >= num_samples:
            break

        sample = batch[0]
        gt_actions = np.array(sample["action"], dtype=np.float32)  # [chunk_len, D]

        if inference_delay == 0:
            # Sync mode: use model-level predict_action
            with torch.inference_mode():
                output = model.predict_action(examples=batch, **infer_kwargs)
            pred = output["normalized_actions"][0].astype(np.float32)
            pred_len = pred.shape[0]
            gt = gt_actions[-pred_len:]
            eval_pred = pred
            eval_gt = gt
        else:
            # RTC mode: VLM forward + action_model.predict_action_realtime
            vl_embs_list, state = vlm_forward(model, framework_type, batch)
            device = vl_embs_list[0].device
            dtype = vl_embs_list[0].dtype

            # Use GT as prev_action_chunk (aligned to action horizon)
            gt_aligned = gt_actions[-H:]
            prev_chunk_t = (
                torch.from_numpy(gt_aligned)
                .unsqueeze(0)
                .to(device, dtype=torch.float32)
            )

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float32):
                    if is_pi:
                        pred_t = am.predict_action_realtime(
                            vl_embs_list, state, prev_chunk_t,
                            inference_delay=inference_delay,
                            mode=infer_kwargs.get("rtc_mode", "pigdm"),
                        )
                    else:
                        pred_t = am.predict_action_realtime(
                            vl_embs_list, state, prev_chunk_t,
                            inference_delay=inference_delay,
                            execution_horizon=inference_delay,
                            hard_mask=hard_mask,
                            decode_temperature=infer_kwargs.get("decode_temperature", 0.0),
                            choice_temperature=infer_kwargs.get("choice_temperature", 0.1),
                        )

            pred = pred_t[0].detach().cpu().float().numpy()
            gt = gt_aligned

            # Evaluate last execution_horizon positions
            eval_pred = pred[-execution_horizon:]
            eval_gt = gt[-execution_horizon:]

        # Valid mask (exclude padded timesteps)
        valid = _valid_mask(eval_gt)
        if not valid.any():
            count += 1
            continue

        diff = eval_pred[valid] - eval_gt[valid]
        mse = np.mean(diff ** 2)
        l1 = np.mean(np.abs(diff))
        per_dim_mse = np.mean(diff ** 2, axis=0)

        all_mse.append(mse)
        all_l1.append(l1)
        all_per_dim_mse.append(per_dim_mse)

        # Position RMSE in cm
        if pos_scale is not None:
            pos_diff = diff[:, :3]
            pos_rmse = np.sqrt(np.mean(pos_diff ** 2, axis=0)) * pos_scale * 100
            all_pos_rmse_cm.append(pos_rmse)

        count += 1

    if not all_mse:
        return None

    all_per_dim_mse = np.array(all_per_dim_mse)
    per_dim_mse = np.mean(all_per_dim_mse, axis=0)

    result = {
        "delay": inference_delay,
        "execution_horizon": execution_horizon,
        "num_samples": len(all_mse),
        "mse_mean": float(np.mean(all_mse)),
        "mse_std": float(np.std(all_mse)),
        "l1_mean": float(np.mean(all_l1)),
        "l1_std": float(np.std(all_l1)),
        "per_dim_mse": {DIM_LABELS[i]: float(per_dim_mse[i])
                        for i in range(len(DIM_LABELS))},
        "pos_mse": float(np.mean(per_dim_mse[:3])),
        "rot_mse": float(np.mean(per_dim_mse[3:9])),
        "grip_mse": float(per_dim_mse[9]),
    }

    if all_pos_rmse_cm:
        pos_rmse_cm = np.mean(all_pos_rmse_cm, axis=0)  # [3,]
        result["pos_rmse_cm"] = {
            DIM_LABELS[i]: float(pos_rmse_cm[i]) for i in range(3)
        }
        result["pos_rmse_cm_avg"] = float(np.mean(pos_rmse_cm))

    return result


def sweep_model(model, framework_type, dataloader, delays, num_samples,
                infer_kwargs, pos_scale, hard_mask=False):
    """Sweep all delays for a single model config."""
    results = {}
    for delay in delays:
        print(f"\n  {framework_type} | delay={delay} | "
              f"{'hard_mask' if hard_mask else 'natural'}")
        res = evaluate_delay(
            model, framework_type, dataloader, num_samples,
            delay, infer_kwargs, pos_scale, hard_mask=hard_mask,
        )
        if res is not None:
            results[delay] = res
            print(f"    MSE={res['mse_mean']:.6f}  L1={res['l1_mean']:.6f}  "
                  f"pos_RMSE={res.get('pos_rmse_cm_avg', 0):.4f} cm  "
                  f"(eval last {res['execution_horizon']} of {model.action_model.action_horizon} positions)")
    return results


# ── Plotting ─────────────────────────────────────────────────────────

SERIES_CFG = [
    ("pi",      "PI (ΠGDM)",           "#4C72B0", "o"),
    ("dd",      "DD (natural mask)",    "#DD8452", "s"),
    ("dd_hard", "DD (hard mask)",       "#55A868", "D"),
]


def plot_mse_vs_delay(all_results, out_path):
    """Plot overall MSE vs delay."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for key, label, color, marker in SERIES_CFG:
        rows = all_results.get(key)
        if not rows:
            continue
        delays = sorted(rows.keys())
        mse = [rows[d]["mse_mean"] for d in delays]
        std = [rows[d]["mse_std"] for d in delays]
        ax.errorbar(delays, mse, yerr=std, marker=marker, capsize=4,
                     label=label, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Inference Delay (n_actions)", fontsize=13)
    ax.set_ylabel("MSE (normalized)", fontsize=13)
    ax.set_title("Open-Loop MSE vs Execution Horizon\n"
                 "(evaluating only generated positions, GT prefix)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, max(d for rows in all_results.values()
                                for d in rows.keys()) + 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_pos_rmse_vs_delay(all_results, out_path):
    """Plot position RMSE (cm) vs delay."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for key, label, color, marker in SERIES_CFG:
        rows = all_results.get(key)
        if not rows:
            continue
        delays = sorted(rows.keys())
        vals = [rows[d].get("pos_rmse_cm_avg") for d in delays]
        if any(v is None for v in vals):
            continue
        ax.plot(delays, vals, marker=marker, label=label, color=color,
                linewidth=2, markersize=8)

    ax.set_xlabel("Inference Delay (n_actions)", fontsize=13)
    ax.set_ylabel("Position RMSE (cm)", fontsize=13)
    ax.set_title("Open-Loop Position RMSE vs Execution Horizon\n"
                 "(evaluating only generated positions, GT prefix)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close(fig)


def plot_group_mse_vs_delay(all_results, out_path):
    """Plot MSE by group (position/rotation/gripper) vs delay."""
    groups = [("pos_mse", "Position"), ("rot_mse", "Rotation"), ("grip_mse", "Gripper")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (mse_key, group_name) in zip(axes, groups):
        for key, label, color, marker in SERIES_CFG:
            rows = all_results.get(key)
            if not rows:
                continue
            delays = sorted(rows.keys())
            vals = [rows[d][mse_key] for d in delays]
            ax.plot(delays, vals, marker=marker, label=label, color=color,
                    linewidth=2, markersize=7)
        ax.set_xlabel("Inference Delay", fontsize=12)
        ax.set_ylabel("MSE", fontsize=12)
        ax.set_title(group_name, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Group MSE vs Execution Horizon (GT prefix)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


def print_summary(all_results):
    """Print summary table."""
    all_delays = sorted(set(d for rows in all_results.values() for d in rows.keys()))

    active = [(k, l) for k, l, *_ in SERIES_CFG if k in all_results]
    col_w = 28

    header = f"{'delay':>5}  |  " + "  |  ".join(f"{l:>{col_w}}" for _, l in active)
    sub = f"{'':>5}  |  " + "  |  ".join(
        f"{'MSE':>9} {'L1':>8} {'pos cm':>9}" for _ in active
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("  Open-Loop MSE vs Delay (evaluating generated positions only)")
    print(sep)
    print(header)
    print(sub)
    print(sep)

    for d in all_delays:
        parts = [f"{d:>5}"]
        for key, _ in active:
            r = all_results[key].get(d)
            if r:
                mse = f"{r['mse_mean']:.5f}"
                l1 = f"{r['l1_mean']:.5f}"
                pos = f"{r.get('pos_rmse_cm_avg', 0):.3f}"
                parts.append(f"{mse:>9} {l1:>8} {pos:>9}")
            else:
                parts.append(f"{'N/A':>9} {'N/A':>8} {'N/A':>9}")
        print("  |  ".join(parts))

    print(sep)

    for key, label in active:
        rows = all_results[key]
        if rows:
            d0 = rows.get(0)
            if d0:
                print(f"  {label}: H={d0['execution_horizon']} (delay=0)")
    print()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Open-loop MSE vs inference delay sweep")
    parser.add_argument("--checkpoint_pi", type=str, default=DEFAULT_CHECKPOINT_PI)
    parser.add_argument("--checkpoint_dd", type=str, default=DEFAULT_CHECKPOINT_DD)
    parser.add_argument("--data_root_dir", type=str, default=bench.DATA_ROOT_DIR)
    parser.add_argument("--max_delay", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--rtc_mode", type=str, default="pigdm",
                        choices=["pigdm", "simulated_delay"])
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip_pi", action="store_true")
    parser.add_argument("--skip_dd", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delays = list(range(0, args.max_delay + 1))

    print(f"{'='*60}")
    print(f"  Open-Loop MSE Sweep: delays={delays}")
    print(f"  Samples: {args.num_samples}  Steps: {args.num_inference_steps}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    infer_kwargs = dict(
        decode_temperature=bench.DECODE_TEMPERATURE,
        choice_temperature=bench.CHOICE_TEMPERATURE,
        use_simple_max=False,
        rtc_mode=args.rtc_mode,
    )

    all_results = {}

    # ── PI ──
    if not args.skip_pi:
        print(f"\n{'='*60}")
        print(f"  Loading PI model...")
        print(f"{'='*60}")
        model_pi, fw_pi = bench.load_model(args.checkpoint_pi)
        am = model_pi.action_model
        if hasattr(am, "num_inference_timesteps"):
            am.num_inference_timesteps = args.num_inference_steps

        dataloader_pi = bench.load_dataset(
            model_pi, include_state=args.include_state,
            data_root_dir=args.data_root_dir,
        )

        try:
            pos_scale = get_pos_real_scale(model_pi.norm_stats)
        except Exception:
            pos_scale = None

        all_results["pi"] = sweep_model(
            model_pi, fw_pi, dataloader_pi, delays,
            args.num_samples, infer_kwargs, pos_scale,
        )

        for d, res in all_results["pi"].items():
            res["checkpoint"] = args.checkpoint_pi
        with open(out_dir / "pi_results.json", "w") as f:
            json.dump(all_results["pi"], f, indent=2)

        del model_pi, dataloader_pi
        torch.cuda.empty_cache()

    # ── DD ──
    if not args.skip_dd:
        print(f"\n{'='*60}")
        print(f"  Loading DD model...")
        print(f"{'='*60}")
        model_dd, fw_dd = bench.load_model(args.checkpoint_dd)
        am = model_dd.action_model
        if hasattr(am, "num_inference_steps"):
            am.num_inference_steps = args.num_inference_steps

        dataloader_dd = bench.load_dataset(
            model_dd, include_state=args.include_state,
            data_root_dir=args.data_root_dir,
        )

        try:
            pos_scale = get_pos_real_scale(model_dd.norm_stats)
        except Exception:
            pos_scale = None

        # DD natural mask
        all_results["dd"] = sweep_model(
            model_dd, fw_dd, dataloader_dd, delays,
            args.num_samples, infer_kwargs, pos_scale, hard_mask=False,
        )
        for d, res in all_results["dd"].items():
            res["checkpoint"] = args.checkpoint_dd
            res["mask_mode"] = "natural"
        with open(out_dir / "dd_results.json", "w") as f:
            json.dump(all_results["dd"], f, indent=2)

        # DD hard mask
        all_results["dd_hard"] = sweep_model(
            model_dd, fw_dd, dataloader_dd, delays,
            args.num_samples, infer_kwargs, pos_scale, hard_mask=True,
        )
        for d, res in all_results["dd_hard"].items():
            res["checkpoint"] = args.checkpoint_dd
            res["mask_mode"] = "hard"
        with open(out_dir / "dd_hard_results.json", "w") as f:
            json.dump(all_results["dd_hard"], f, indent=2)

        del model_dd, dataloader_dd
        torch.cuda.empty_cache()

    # ── Load any previously saved results for missing series ──
    for key, fname in [("pi", "pi_results.json"), ("dd", "dd_results.json"),
                       ("dd_hard", "dd_hard_results.json")]:
        if key not in all_results and (out_dir / fname).exists():
            with open(out_dir / fname) as f:
                loaded = json.load(f)
            all_results[key] = {int(k): v for k, v in loaded.items()}

    if not all_results:
        print("No results.")
        return

    # ── Summary + Plots ──
    print_summary(all_results)

    print("Generating plots...")
    plot_mse_vs_delay(all_results, out_dir / "plot_mse_vs_delay.png")
    plot_pos_rmse_vs_delay(all_results, out_dir / "plot_pos_rmse_vs_delay.png")
    plot_group_mse_vs_delay(all_results, out_dir / "plot_group_mse_vs_delay.png")

    # Save combined summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done. Results in: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
