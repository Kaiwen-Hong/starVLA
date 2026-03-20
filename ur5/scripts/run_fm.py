#!/usr/bin/env python3
"""Unified closed-loop control with flow matching (continuous / QwenPI).

Supports three execution modes via --mode:
  sync  : observe -> infer -> execute (sequential, no overlap)
  async : gap-free 100Hz servo thread + async inference (no RTC prefix)
  rtc   : gap-free 100Hz servo thread + async inference + RTC prefix inpainting

Usage:
    python ur5/scripts/run_fm.py --mode sync
    python ur5/scripts/run_fm.py --mode async --n_actions 8
    python ur5/scripts/run_fm.py --mode rtc   --n_actions 8 --inference_delay 8
"""

import os
import sys
import argparse
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "ur5"))
os.chdir(REPO_DIR)

from scripts.robo_utils import load_model, get_action_stats, run_main

DEFAULT_CHECKPOINT = (
    "checkpoints/DiscreteRTC/"
    "fastumi_pickandplace_qwenPI_no_state_fixgripper/"
    "checkpoints/steps_25000_pytorch_model.pt"
)
SPLASH_TITLES = {
    "sync":  "Flow Matching\nSync Policy",
    "async": "Flow Matching\nAsync Policy",
    "rtc":   "Flow Matching\nRTC Policy",
}


def load_fm_model(checkpoint_path: str):
    model = load_model(checkpoint_path)
    if not getattr(model.config.datasets.vla_data, "image_size", None):
        model.config.datasets.vla_data.image_size = [224, 224]
        print("[FIX] Set image_size=[224,224] (was missing from checkpoint config)")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Unified FM closed-loop control")
    p.add_argument("--mode", choices=["sync", "async", "rtc"], default="rtc")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--arm", choices=["left", "right"], default="left")
    p.add_argument("--camera_dev", type=int, default=0)
    p.add_argument("--n_actions", type=int, default=8)
    p.add_argument("--inference_delay", type=int, default=8,
                   help="RTC prefix timesteps (-1 = same as n_actions)")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--instruction", type=str, default="pick up the building block")
    p.add_argument("--include_state", action="store_true", default=False)
    # Robot
    p.add_argument("--fix_rotation", action="store_true", default=True)
    p.add_argument("--no_fix_rotation", dest="fix_rotation", action="store_false")
    p.add_argument("--no_go_home", action="store_true", default=False)
    p.add_argument("--stop_when_grasping", action="store_true", default=True)
    p.add_argument("--no_stop_when_grasping", dest="stop_when_grasping", action="store_false")
    p.add_argument("--grasp_lift_threshold", type=float, default=0.04)
    # Logging
    p.add_argument("--save_rollout", action="store_true", default=False)
    p.add_argument("--no_save_rollout", dest="save_rollout", action="store_false")
    p.add_argument("--rollout_dir", type=str, default=None)
    p.add_argument("--stopwatch_port", type=int, default=8765)

    args = p.parse_args()
    if args.inference_delay < 0:
        args.inference_delay = args.n_actions
    return args


def main():
    args = parse_args()
    model = load_fm_model(args.checkpoint)
    action_stats, dataset_key = get_action_stats(model)

    run_main(args, model, action_stats, dataset_key,
             infer_kwargs={}, splash_title=SPLASH_TITLES[args.mode],
             method_name="FM")


if __name__ == "__main__":
    main()
