#!/usr/bin/env python3
"""
Sweep inference_delay from 0..max_delay for PI and DD, loading each model ONCE.

Reuses the benchmark infrastructure from 0403-benchmarking-inference.py but
avoids the cost of reloading models for every delay value.

Usage:
    python realworld/0405-sweep-inference-delay.py
    python realworld/0405-sweep-inference-delay.py --max_delay 8 --num_samples 50
"""

import warnings
warnings.filterwarnings("ignore", message=".*video decoding and encoding.*torchvision.*")

import argparse
import json
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

# Import the benchmark script as a module (avoids rewriting shared code)
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
DEFAULT_OUT_DIR = "/scratch/wangpc/starVLA/results/benchmark_sweep_0405"


def sweep_one_model(model, framework_type, dataloader, delays, num_samples,
                    warmup, num_steps, infer_kwargs):
    """Run benchmark for a single model across all delay values."""
    all_results = {}

    for delay in delays:
        print(f"\n{'─'*60}")
        print(f"  {framework_type} | inference_delay={delay}")
        print(f"{'─'*60}")

        results = bench.benchmark(
            model, framework_type, dataloader,
            num_samples=num_samples,
            warmup=warmup,
            inference_delay=delay,
            infer_kwargs=infer_kwargs,
        )

        # Add metadata
        results["inference_delay"] = delay
        results["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        results["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        results["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)

        bench.print_results(results)
        all_results[delay] = results

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Sweep inference_delay for PI and DD (load each model once)")
    parser.add_argument("--checkpoint_pi", type=str, default=DEFAULT_CHECKPOINT_PI)
    parser.add_argument("--checkpoint_dd", type=str, default=DEFAULT_CHECKPOINT_DD)
    parser.add_argument("--data_root_dir", type=str, default=bench.DATA_ROOT_DIR)
    parser.add_argument("--max_delay", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--rtc_mode", type=str, default="pigdm",
                        choices=["pigdm", "simulated_delay"])
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip_pi", action="store_true", help="Skip PI benchmark")
    parser.add_argument("--skip_dd", action="store_true", help="Skip DD benchmark")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delays = list(range(0, args.max_delay + 1))

    print(f"{'='*60}")
    print(f"  Inference Delay Sweep: delays={delays}")
    print(f"  Samples: {args.num_samples}  Warmup: {args.warmup}")
    print(f"  Steps: {args.num_inference_steps}  RTC mode: {args.rtc_mode}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    infer_kwargs = dict(
        decode_temperature=bench.DECODE_TEMPERATURE,
        choice_temperature=bench.CHOICE_TEMPERATURE,
        use_simple_max=False,
        rtc_mode=args.rtc_mode,
    )

    # ── PI (Flow-Matching) ──
    if not args.skip_pi:
        print(f"\n{'='*60}")
        print(f"  Loading PI model (once)...")
        print(f"{'='*60}")
        model_pi, fw_pi = bench.load_model(args.checkpoint_pi)

        am = model_pi.action_model
        if hasattr(am, "num_inference_timesteps"):
            am.num_inference_timesteps = args.num_inference_steps
        if hasattr(am, "num_inference_steps"):
            am.num_inference_steps = args.num_inference_steps

        dataloader_pi = bench.load_dataset(
            model_pi, include_state=args.include_state,
            data_root_dir=args.data_root_dir,
        )

        pi_results = sweep_one_model(
            model_pi, fw_pi, dataloader_pi, delays,
            args.num_samples, args.warmup, args.num_inference_steps, infer_kwargs,
        )

        # Save individual JSONs
        for delay, res in pi_results.items():
            res["checkpoint"] = args.checkpoint_pi
            res["rtc_mode"] = args.rtc_mode
            path = out_dir / f"pi_delay{delay}.json"
            with open(path, "w") as f:
                json.dump(res, f, indent=2)

        # Free GPU memory
        del model_pi, dataloader_pi
        torch.cuda.empty_cache()

    # ── DD (Discrete Diffusion / MaskGIT) ──
    if not args.skip_dd:
        print(f"\n{'='*60}")
        print(f"  Loading DD model (once)...")
        print(f"{'='*60}")
        model_dd, fw_dd = bench.load_model(args.checkpoint_dd)

        am = model_dd.action_model
        if hasattr(am, "num_inference_timesteps"):
            am.num_inference_timesteps = args.num_inference_steps
        if hasattr(am, "num_inference_steps"):
            am.num_inference_steps = args.num_inference_steps

        dataloader_dd = bench.load_dataset(
            model_dd, include_state=args.include_state,
            data_root_dir=args.data_root_dir,
        )

        # DD natural mask (default: prefix = H - execution_horizon)
        dd_kwargs = dict(infer_kwargs, hard_mask=False)
        dd_results = sweep_one_model(
            model_dd, fw_dd, dataloader_dd, delays,
            args.num_samples, args.warmup, args.num_inference_steps, dd_kwargs,
        )
        for delay, res in dd_results.items():
            res["checkpoint"] = args.checkpoint_dd
            res["mask_mode"] = "natural"
            path = out_dir / f"dd_delay{delay}.json"
            with open(path, "w") as f:
                json.dump(res, f, indent=2)

        # DD hard mask (prefix = inference_delay)
        dd_hard_kwargs = dict(infer_kwargs, hard_mask=True)
        dd_hard_results = sweep_one_model(
            model_dd, fw_dd, dataloader_dd, delays,
            args.num_samples, args.warmup, args.num_inference_steps, dd_hard_kwargs,
        )
        for delay, res in dd_hard_results.items():
            res["checkpoint"] = args.checkpoint_dd
            res["mask_mode"] = "hard"
            path = out_dir / f"dd_hard_delay{delay}.json"
            with open(path, "w") as f:
                json.dump(res, f, indent=2)

        del model_dd, dataloader_dd
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  Sweep complete. Results in: {out_dir}")
    print(f"  Run analysis:")
    print(f"    python realworld/0405-analyze-inference-sweep.py --sweep_dir {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
