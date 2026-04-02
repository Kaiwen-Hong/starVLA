#!/usr/bin/env python3
"""
Step 9: Closed-loop control with smoother motion.

Based on step8 (direct servoL) but with two improvements:
  1. Linear interpolation between predicted waypoints (20Hz → 100Hz servo)
  2. Z-floor clamping instead of emergency stop
  3. Tuned servoL parameters (lookahead=0.2, gain=200) for smoother motion

Usage:
    python ur5/step9-dd-closed-loop-smooth.py
    python ur5/step9-dd-closed-loop-smooth.py --n_actions 14 --arm left
    python ur5/step9-dd-closed-loop-smooth.py --n_actions 8 --arm left 
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "ur5"))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework

from scripts.log_utils import RolloutSaver, visualize_step

import importlib
_step2 = importlib.import_module("ur5.step2-replace-with-real-camera")
RealCamera = _step2.RealCamera

import modular_policy
project_root = Path(modular_policy.__file__).parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}
ROBOT_IPS = {'left': '192.168.0.3', 'right': '192.168.0.2'}
HOME_POSES_WORLD = {
    'left': [0.300311, -0.489314, 0.250303, -2.220294, 2.215871, 0.010386, 1],
    'right': [-0.1, -0.3, 0.25, 2.2419, -2.1984, 0.0166, 1],
}

INSTRUCTION = "pick up the block in the pot and place it in the red area on the turntable"
CONTROL_HZ = 20
INTERP_MULT = 5       # interpolation multiplier: 20Hz × 5 = 100Hz servo
SERVO_HZ = CONTROL_HZ * INTERP_MULT  # 100Hz
Z_MIN_WORLD = 0.001793 - 0.02


# ── Math utilities (same as step8) ────────────────────────────────

def base_to_world(pose_base, T_bw):
    p = list(pose_base)
    p[0] += T_bw[0, 3]; p[1] += T_bw[1, 3]; p[2] += T_bw[2, 3]
    return p


def world_to_base(pose_world, T_bw):
    p = list(pose_world)
    p[0] -= T_bw[0, 3]; p[1] -= T_bw[1, 3]; p[2] -= T_bw[2, 3]
    return p


def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    a1 = d6[:3].astype(np.float64)
    a2 = d6[3:].astype(np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def rot6d_to_axisangle(d6: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rot6d_to_mat(d6)).as_rotvec().astype(np.float32)


def axisangle_to_rot6d(rx, ry, rz):
    R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    return R[:2, :].flatten().astype(np.float32)


def ee_pose_to_state10d(pose_6d, gripper: float) -> np.ndarray:
    x, y, z, rx, ry, rz = pose_6d[:6]
    rot6d = axisangle_to_rot6d(rx, ry, rz)
    return np.array([x, y, z, *rot6d, gripper], dtype=np.float32)


def action_10d_to_delta7d(action_10d: np.ndarray) -> np.ndarray:
    delta = np.zeros(7, dtype=np.float32)
    delta[:3] = action_10d[:3]
    delta[3:6] = rot6d_to_axisangle(action_10d[3:9])
    delta[6] = action_10d[9]
    return delta


def _precise_wait(t_end: float, slack_time: float = 0.001):
    t_wait = t_end - time.monotonic()
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time.monotonic() < t_end:
            pass


# ── Model loading (same as step8) ────────────────────────────────

def _detect_attn_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_model(checkpoint_path: str):
    import torch

    print(f"Loading model from: {checkpoint_path}")
    t0 = time.time()

    checkpoint_pt = Path(checkpoint_path)
    model_config, norm_stats = read_mode_config(checkpoint_pt)
    config = dict_to_namespace(model_config)
    config.trainer.pretrained_checkpoint = None
    config.framework.qwenvl.attn_implementation = _detect_attn_implementation()

    model = build_framework(cfg=config)
    model.norm_stats = norm_stats

    state_dict = torch.load(checkpoint_pt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model = model.to("cuda").eval()
    print(f"Model loaded in {time.time() - t0:.1f}s ({config.framework.name})")
    return model


TRAIN_IMAGE_SIZE = (224, 224)


def build_example(image: Image.Image, instruction: str,
                  state_10d: np.ndarray = None) -> dict:
    image = image.resize(TRAIN_IMAGE_SIZE)
    example = {"image": [image], "lang": instruction}
    if state_10d is not None:
        example["state"] = state_10d.reshape(1, -1)
    return example


# ── Go home (same as step8) ──────────────────────────────────────

def go_home(rtde_c, rtde_r, arm, T_bw, robot_ip):
    home = HOME_POSES_WORLD[arm]
    home_base = world_to_base(home[:6], T_bw)
    target_joints = rtde_c.getInverseKinematics(home_base)
    print(f"Moving to home pose (world): {[round(x, 2) for x in home[:6]]}")
    rtde_c.moveJ(target_joints, 1.0, 1.0)

    current_base = rtde_r.getActualTCPPose()
    current_world = base_to_world(current_base, T_bw)
    print(f"Reached: {[round(x, 4) for x in current_world]}")

    from robotiq_gripper import RobotiqGripper
    gripper = RobotiqGripper()
    gripper.connect(hostname=robot_ip, port=63352)
    pos = int((1.0 - home[6]) * 255)
    gripper.move(pos, 255, 150)
    gripper.disconnect()
    print("Gripper opened.")
    return home[6]


# ── Interpolation ────────────────────────────────────────────────

def interpolate_waypoints(start_pose, waypoints, mult):
    """Linearly interpolate between waypoints.

    start_pose: (6,) current pose
    waypoints:  (N, 6) target poses
    mult:       number of sub-steps between each waypoint

    Returns (N*mult, 6) interpolated poses.
    """
    all_poses = []
    prev = start_pose
    for wp in waypoints:
        for j in range(1, mult + 1):
            alpha = j / mult
            interp = prev + alpha * (wp - prev)
            all_poses.append(interp)
        prev = wp
    return np.array(all_poses, dtype=np.float64)


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop control with smooth interpolated servoL")
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/discreteRTC/"
                "fastumi_pickandplace_qwenDiscreteDiffusion_329v2/"
                "checkpoints/steps_20000_pytorch_model.pt",
    )
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--n_actions", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--no_go_home", action="store_true", default=False)
    parser.add_argument("--instruction", type=str, default=INSTRUCTION)
    parser.add_argument("--decode_temperature", type=float, default=0.0)
    parser.add_argument("--choice_temperature", type=float, default=0.1)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    parser.add_argument("--fix_rotation", action="store_true", default=True)
    parser.add_argument("--no_fix_rotation", action="store_true", default=False)
    parser.add_argument("--save_rollout", action="store_true", default=True)
    parser.add_argument("--no_save_rollout", dest="save_rollout", action="store_false")
    parser.add_argument("--rollout_dir", type=str, default=None,
                        help="Directory to save rollout (default: auto)")
    args = parser.parse_args()
    if args.no_fix_rotation:
        args.fix_rotation = False

    robot_ip = ROBOT_IPS[args.arm]
    T_bw = BASE_IN_WORLD[args.arm]
    servo_dt = 1.0 / SERVO_HZ

    # ── Load model ───────────────────────────────────────────────
    model = load_model(args.checkpoint)
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]

    # ── Connect to robot ─────────────────────────────────────────
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    print(f"\nConnecting to {args.arm} arm at {robot_ip}...")
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    # ── Connect gripper ──────────────────────────────────────────
    from robotiq_gripper import RobotiqGripper
    print("Connecting to gripper...")
    gripper_hw = RobotiqGripper()
    gripper_hw.connect(hostname=robot_ip, port=63352)
    current_gripper = 0.0

    # ── Open camera ──────────────────────────────────────────────
    print(f"Opening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    print("Warming up camera (2s)...")
    t_warm = time.monotonic() + 2.0
    while time.monotonic() < t_warm:
        cam.grab_rgb()
    print("Camera ready.")

    # ── Go home ──────────────────────────────────────────────────
    if not args.no_go_home:
        go_home(rtde_c, rtde_r, args.arm, T_bw, robot_ip)
        current_gripper = 0.0

    # ── Print config ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Step 9-dd-dynamic: Closed-loop (DD, smooth {SERVO_HZ}Hz interpolated)")
    print(f"  Arm:                {args.arm}")
    print(f"  Instruction:        {args.instruction}")
    print(f"  Actions/step:       {args.n_actions}")
    print(f"  Waypoints:          {CONTROL_HZ}Hz × {INTERP_MULT} = {SERVO_HZ}Hz servo")
    print(f"  servoL params:      lookahead=0.2, gain=200")
    print(f"  Z safety:           clamp to {Z_MIN_WORLD:.4f}m")
    print(f"  Include state:      {args.include_state}")
    print(f"  fix_rotation:       {args.fix_rotation}")
    print(f"  Save rollout:       {args.save_rollout}")
    print(f"{'=' * 60}")

    # ── Rollout saver ────────────────────────────────────────────
    saver = None
    if args.save_rollout:
        ts = time.strftime("%Y%m%d_%H%M%S")
        rollout_dir = Path(args.rollout_dir) if args.rollout_dir else (
            Path("ur5") / "rollouts" / f"step9dd_dynamic_dd_{ts}")
        run_config = {
            "mode": "sync", "method": "step9dd-dynamic-dd",
            "checkpoint": args.checkpoint,
            "instruction": args.instruction,
            "arm": args.arm, "camera_dev": args.camera_dev,
            "n_actions": args.n_actions,
            "include_state": args.include_state,
            "fix_rotation": args.fix_rotation,
            "control_hz": CONTROL_HZ, "servo_hz": SERVO_HZ,
            "interp_mult": INTERP_MULT, "dataset_key": dataset_key,
        }
        saver = RolloutSaver(rollout_dir, run_config)

    input("\n>>> Press Enter to START (Ctrl+C to abort) <<<")

    # ── Control loop ─────────────────────────────────────────────
    step = 0
    try:
        while args.max_steps == 0 or step < args.max_steps:
            loop_t0 = time.monotonic()

            # 1. Read current robot state
            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            current_state_10d = ee_pose_to_state10d(pose_world, current_gripper)
            current_pos = np.array(pose_world, dtype=np.float64)

            # 2. Grab camera frame
            pil_img = cam.grab_pil()
            if pil_img is None:
                print("[WARN] Camera frame dropped, retrying...")
                continue

            # 3. Inference
            state_for_model = current_state_10d if args.include_state else None
            example = build_example(pil_img, args.instruction, state_10d=state_for_model)

            t_infer = time.monotonic()
            output = model.predict_action(
                examples=[example],
                decode_temperature=args.decode_temperature,
                choice_temperature=args.choice_temperature,
                use_simple_max=args.use_simple_max,
            )
            infer_ms = (time.monotonic() - t_infer) * 1000

            pred_normalized = output["normalized_actions"][0].astype(np.float32)
            pred_actions_10d = baseframework.unnormalize_actions(pred_normalized, action_stats)

            # 4. Convert local-frame actions to absolute world-frame waypoints
            #    Actions are in EE local frame (inv(base) @ target), so:
            #      position: world_delta = R_current @ local_delta
            #      rotation: R_new = R_current @ R_relative
            n_exec = min(args.n_actions, len(pred_actions_10d))
            waypoints = np.zeros((n_exec, 6), dtype=np.float64)
            pos = np.array(current_pos[:3], dtype=np.float64)
            R_cur = Rotation.from_rotvec(current_pos[3:6]).as_matrix().astype(np.float64)

            for i in range(n_exec):
                # Local-frame position delta → world frame
                local_dp = pred_actions_10d[i, :3].astype(np.float64)
                pos = pos + R_cur @ local_dp

                # Rotation: compose relative rotation
                if not args.fix_rotation:
                    R_rel = rot6d_to_mat(pred_actions_10d[i, 3:9])
                    R_cur = R_cur @ R_rel

                # Safety: clamp z
                if pos[2] < Z_MIN_WORLD:
                    print(f"  [SAFETY] z clamped: {pos[2]:.4f} → {Z_MIN_WORLD:.4f}")
                    pos[2] = Z_MIN_WORLD

                waypoints[i, :3] = pos
                waypoints[i, 3:6] = Rotation.from_matrix(R_cur).as_rotvec()

                # Handle gripper on first transition
                new_gripper = float(pred_actions_10d[i, 9])
                if (new_gripper > 0.5) != (current_gripper > 0.5):
                    grip_pos = int(new_gripper * 255)
                    label = "CLOSE" if new_gripper > 0.5 else "OPEN"
                    print(f"  Gripper -> {label} (pos={grip_pos})")
                    gripper_hw.move(grip_pos, 255, 150)
                    current_gripper = new_gripper

            # 5. Interpolate and execute at 100Hz
            interp_poses = interpolate_waypoints(current_pos, waypoints, INTERP_MULT)

            t_exec_start = time.monotonic()
            for i, pose_world_i in enumerate(interp_poses):
                target_base = world_to_base(pose_world_i.tolist(), T_bw)
                rtde_c.servoL(target_base, 0, 0, servo_dt, 0.2, 200)
                _precise_wait(t_exec_start + (i + 1) * servo_dt)

            rtde_c.servoStop()

            exec_ms = (time.monotonic() - t_exec_start) * 1000
            total_ms = (time.monotonic() - loop_t0) * 1000

            print(f"[step {step:4d}]  infer={infer_ms:5.0f}ms  exec={exec_ms:5.0f}ms  "
                  f"total={total_ms:5.0f}ms  "
                  f"pos=[{pose_world[0]:.3f}, {pose_world[1]:.3f}, {pose_world[2]:.3f}]  "
                  f"grip={'C' if current_gripper > 0.5 else 'O'}")

            # 6. Save rollout
            if saver:
                # Build absolute trajectory (local→world transform)
                abs_poses = np.zeros((len(pred_actions_10d), 7), dtype=np.float64)
                p = np.array(pose_world[:3], dtype=np.float64)
                R_viz = Rotation.from_rotvec(pose_world[3:6]).as_matrix().astype(np.float64)
                for t in range(len(pred_actions_10d)):
                    local_dp = pred_actions_10d[t, :3].astype(np.float64)
                    p = p + R_viz @ local_dp
                    R_rel = rot6d_to_mat(pred_actions_10d[t, 3:9])
                    R_viz = R_viz @ R_rel
                    abs_poses[t, :3] = p
                    abs_poses[t, 3:6] = Rotation.from_matrix(R_viz).as_rotvec()
                    abs_poses[t, 6] = pred_actions_10d[t, 9]

                # Save camera image
                img_path = saver.dir / "images" / f"step_{step:04d}.jpg"
                pil_img.save(img_path, quality=90)

                # Save visualization async
                viz_path = saver.dir / "images" / f"step_{step:04d}_viz.png"
                saver.submit_viz(
                    visualize_step, pil_img, pose_world, n_exec, step,
                    str(viz_path), args.instruction, "sync",
                    pred_poses=abs_poses,
                )

                # Save step data
                saver.save_step({
                    "step": step,
                    "wall_time": time.time(),
                    "pose_world": list(pose_world),
                    "gripper": current_gripper,
                    "infer_ms": round(infer_ms, 1),
                    "exec_ms": round(exec_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "n_exec": n_exec,
                    "pred_normalized": pred_normalized.tolist(),
                    "pred_actions_10d": pred_actions_10d.tolist(),
                })

            step += 1

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        print("Cleaning up...")
        try: rtde_c.servoStop()
        except Exception: pass
        try: rtde_c.stopScript()
        except Exception: pass
        try: gripper_hw.disconnect()
        except Exception: pass
        cam.close()
        if saver:
            saver.finalize()
        print(f"Done. Executed {step} inference steps.")


if __name__ == "__main__":
    main()
