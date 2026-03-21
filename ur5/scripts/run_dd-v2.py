#!/usr/bin/env python3
"""Unified closed-loop control with discrete diffusion (v2).

Same as run_dd.py but uses servo_v2 + inferencer_v2 which fix:
  1. Gripper commands run on a separate thread (no servo-blocking pauses).
  2. Waypoint splicing uses buffer tail pose (no spatial snap-back).
  3. Servo timing resets after delays (no burst catch-up jitter).

Usage:
    python ur5/scripts/run_dd-v2.py --mode rtc --n_actions 8 --inference_delay 8
"""

import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "ur5"))
os.chdir(REPO_DIR)

# Patch servo and inferencer with v2 versions before anything imports them.
import scripts.servo as _servo_mod
import scripts.inferencer as _infer_mod
from scripts.servo_v2 import ServoRunner as _ServoRunnerV2
from scripts.inferencer_v2 import Inferencer as _InferencerV2
_servo_mod.ServoRunner = _ServoRunnerV2
_infer_mod.Inferencer = _InferencerV2

from scripts.robo_utils import load_model, get_action_stats, run_main

DEFAULT_CHECKPOINT = (
    "checkpoints/DiscreteRTC/"
    "fastumi_pickandplace_discrete_diffusion_real_0314_no_state_fixgripper/"
    "checkpoints/steps_25000_pytorch_model.pt"
)
SPLASH_TITLES = {
    "sync":  "Discrete Diffusion\nSync Policy",
    "async": "Discrete Diffusion\nAsync Policy",
    "rtc":   "Discrete Diffusion\nRTC Policy (v2)",
}


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Unified DD closed-loop control (v2)")
    p.add_argument("--mode", choices=["sync", "async", "rtc"], default="rtc")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--arm", choices=["left", "right"], default="left")
    p.add_argument("--camera_dev", type=int, default=0)
    p.add_argument("--n_actions", type=int, default=8)
    p.add_argument("--inference_delay", type=int, default=8,
                   help="RTC prefix timesteps (-1 = same as n_actions)")
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--instruction", type=str, default="pick up the building block")
    # DD-specific
    p.add_argument("--include_state", action="store_true", default=False)
    p.add_argument("--decode_temperature", type=float, default=0.0)
    p.add_argument("--choice_temperature", type=float, default=0.1)
    p.add_argument("--use_simple_max", action="store_true", default=False)
    p.add_argument("--fixed_steps", action="store_true", default=True)
    p.add_argument("--no_fixed_steps", dest="fixed_steps", action="store_false")
    # Robot
    p.add_argument("--fix_rotation", action="store_true", default=True)
    p.add_argument("--no_fix_rotation", dest="fix_rotation", action="store_false")
    p.add_argument("--no_go_home", action="store_true", default=False)
    p.add_argument("--stop_when_grasping", action="store_true", default=True)
    p.add_argument("--no_stop_when_grasping", dest="stop_when_grasping", action="store_false")
    p.add_argument("--grasp_lift_threshold", type=float, default=0.2)
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
    model = load_model(args.checkpoint)
    action_stats, dataset_key = get_action_stats(model)

    infer_kwargs = dict(
        decode_temperature=args.decode_temperature,
        choice_temperature=args.choice_temperature,
        use_simple_max=args.use_simple_max,
    )
    if args.mode == "rtc":
        infer_kwargs["fixed_steps"] = args.fixed_steps

    extra_config = {
        "decode_temperature": args.decode_temperature,
        "choice_temperature": args.choice_temperature,
    }
    if args.mode == "rtc":
        extra_config["fixed_steps"] = args.fixed_steps

    run_main(args, model, action_stats, dataset_key,
             infer_kwargs=infer_kwargs, splash_title=SPLASH_TITLES[args.mode], method_name="DD",
             extra_config=extra_config)


if __name__ == "__main__":
    main()
