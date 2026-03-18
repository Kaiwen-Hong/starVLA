#!/usr/bin/env python3
"""
Step 6: Closed-loop control with real robot (Discrete Diffusion).

Synchronous receding-horizon loop:
    1. Read current EE pose from robot (RTDE)
    2. Grab camera frame
    3. Run discrete diffusion inference -> predicted action chunk (T, 10)
    4. Denormalize
    5. Execute first N actions via servoL (default N=8)
    6. Repeat from step 1

Key differences from QwenPI version:
  - 8-step MaskGIT decode (vs 4-step flow matching) -> potentially slower inference
  - Supports temperature tuning for action quality
  - Actions quantized to 256 bins (~0.0078 resolution in normalized space)

The robot WILL move. Use Ctrl+C to stop at any time.

Usage:
    python ur5-dd/step6-closed-loop.py
    python ur5-dd/step6-closed-loop.py --n_actions 2 --arm left
    python ur5-dd/step6-closed-loop.py --use_simple_max  # faster, deterministic
    python ur5-dd/step6-closed-loop.py --decode_temperature 0.5
"""

import sys
import os
import time
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "ur5"))  # for robotiq_gripper
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework

# ── Import camera from ur5/step2 ───────────────────────────────────
import importlib
_step2 = importlib.import_module("ur5.step2-replace-with-real-camera")
RealCamera = _step2.RealCamera

# ── Robot extrinsics ────────────────────────────────────────────────
import modular_policy

_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}

ROBOT_IPS = {
    'left': '192.168.0.3',
    'right': '192.168.0.2',
}

# Home poses in WORLD frame [x, y, z, rx, ry, rz, gripper(1=open)]
HOME_POSES_WORLD = {
    'left': [0.30, -0.2, 0.25, -2.2192, 2.2148, 0.0091, 1],
    'right': [-0.1, -0.3, 0.25, 2.2419, -2.1984, 0.0166, 1],
}

INSTRUCTION = "pick up the building block"
CONTROL_HZ = 20


# ═══════════════════════════════════════════════════════════════════
#  Frame conversion & math utilities
# ═══════════════════════════════════════════════════════════════════

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
    R = rot6d_to_mat(d6)
    return Rotation.from_matrix(R).as_rotvec().astype(np.float32)


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


# ═══════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════

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
    print(f"  num_bins: {getattr(config.framework.action_model, 'num_bins', 'N/A')}")
    print(f"  num_inference_steps: {getattr(config.framework.action_model, 'num_inference_steps', 'N/A')}")
    return model


def build_example(image: Image.Image, instruction: str,
                  state_10d: np.ndarray = None) -> dict:
    example = {"image": [image], "lang": instruction}
    if state_10d is not None:
        example["state"] = state_10d.reshape(1, -1)
    return example


# ═══════════════════════════════════════════════════════════════════
#  Go home
# ═══════════════════════════════════════════════════════════════════

def go_home(rtde_c, rtde_r, arm, T_bw, robot_ip):
    home = HOME_POSES_WORLD[arm]
    ee_pose = home[:6]
    gripper_value = home[6]

    home_base = world_to_base(ee_pose, T_bw)
    target_joints = rtde_c.getInverseKinematics(home_base)
    print(f"Moving to home pose (world): {[round(x, 2) for x in ee_pose]}")
    rtde_c.moveJ(target_joints, 1.0, 1.0)

    current_base = rtde_r.getActualTCPPose()
    current_world = base_to_world(current_base, T_bw)
    print(f"Reached: {[round(x, 4) for x in current_world]}")

    from robotiq_gripper import RobotiqGripper
    gripper = RobotiqGripper()
    gripper.connect(hostname=robot_ip, port=63352)
    pos = int((1.0 - gripper_value) * 255)
    gripper.move(pos, 255, 150)
    gripper.disconnect()
    print("Gripper opened.")
    return gripper_value


# ═══════════════════════════════════════════════════════════════════
#  Main closed-loop
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop control with UR5 (Discrete Diffusion)")
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/DiscreteRTC/"
                "fastumi_pickandplace_discrete_diffusion_real_0314_no_state_fixgripper/"
                "checkpoints/steps_15000_pytorch_model.pt",
    )
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--n_actions", type=int, default=8,
                        help="Number of actions to execute per inference (default: 8)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max inference steps (0=unlimited, Ctrl+C to stop)")
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--no_go_home", action="store_true", default=False)
    parser.add_argument("--instruction", type=str, default=INSTRUCTION)
    # Discrete diffusion specific
    parser.add_argument("--decode_temperature", type=float, default=0.1,
                        help="Temperature for MaskGIT decode (default: 0.1)")
    parser.add_argument("--choice_temperature", type=float, default=0.1,
                        help="Temperature for token choice (default: 0.1)")
    parser.add_argument("--use_simple_max", action="store_true", default=False,
                        help="Use argmax (faster, deterministic, skips iterative decode)")
    args = parser.parse_args()

    robot_ip = ROBOT_IPS[args.arm]
    T_bw = BASE_IN_WORLD[args.arm]
    dt = 1.0 / CONTROL_HZ

    # ── Load model ───────────────────────────────────────────────────
    model = load_model(args.checkpoint)

    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', modes={action_stats.get('norm_modes', 'legacy')}")

    # ── Connect to robot ─────────────────────────────────────────────
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    print(f"\nConnecting to {args.arm} arm at {robot_ip}...")
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    # ── Connect to gripper ───────────────────────────────────────────
    from robotiq_gripper import RobotiqGripper

    print("Connecting to gripper...")
    gripper_hw = RobotiqGripper()
    gripper_hw.connect(hostname=robot_ip, port=63352)
    current_gripper = 0.0

    # ── Open camera ──────────────────────────────────────────────────
    print(f"Opening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    print("Warming up camera (2s)...")
    t_warm = time.monotonic() + 2.0
    while time.monotonic() < t_warm:
        cam.grab_rgb()
    print("Camera ready.")

    # ── Go home ──────────────────────────────────────────────────────
    if not args.no_go_home:
        go_home(rtde_c, rtde_r, args.arm, T_bw, robot_ip)
        current_gripper = 0.0

    # ── Print config ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Closed-loop control (Discrete Diffusion)")
    print(f"  Arm:                {args.arm}")
    print(f"  Instruction:        {args.instruction}")
    print(f"  Actions/step:       {args.n_actions}")
    print(f"  Control freq:       {CONTROL_HZ} Hz (dt={dt*1000:.0f}ms)")
    print(f"  Max steps:          {'unlimited' if args.max_steps == 0 else args.max_steps}")
    print(f"  Include state:      {args.include_state}")
    print(f"  decode_temperature: {args.decode_temperature}")
    print(f"  choice_temperature: {args.choice_temperature}")
    print(f"  use_simple_max:     {args.use_simple_max}")
    print(f"{'=' * 60}")

    input("\n>>> Press Enter to START (Ctrl+C to abort) <<<")

    # ── Control loop ─────────────────────────────────────────────────
    step = 0
    try:
        while args.max_steps == 0 or step < args.max_steps:
            loop_t0 = time.monotonic()

            # 1. Read current robot state
            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            current_state_10d = ee_pose_to_state10d(pose_world, current_gripper)

            # 2. Grab camera frame
            pil_img = cam.grab_pil()
            if pil_img is None:
                print("[WARN] Camera frame dropped, retrying...")
                continue

            # 3. Inference (discrete diffusion)
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

            # 4. Denormalize
            pred_actions_10d = baseframework.unnormalize_actions(pred_normalized, action_stats)

            # 5. Execute first N actions
            n_exec = min(args.n_actions, len(pred_actions_10d))
            current_pos = np.array(pose_world, dtype=np.float64)

            t_exec_start = time.monotonic()

            for i in range(n_exec):
                delta = action_10d_to_delta7d(pred_actions_10d[i])

                target_world = current_pos + delta[:6]
                target_base = world_to_base(target_world.tolist(), T_bw)

                rtde_c.servoL(target_base, 0, 0, dt, 0.1, 300)

                # Handle gripper
                new_gripper = float(delta[6])
                if (new_gripper > 0.5) != (current_gripper > 0.5):
                    grip_pos = int(new_gripper * 255)
                    label = "CLOSE" if new_gripper > 0.5 else "OPEN"
                    print(f"  Gripper -> {label} (pos={grip_pos})")
                    gripper_hw.move(grip_pos, 255, 150)
                    current_gripper = new_gripper

                current_pos = target_world

                _precise_wait(t_exec_start + (i + 1) * dt)

            rtde_c.servoStop()

            exec_ms = (time.monotonic() - t_exec_start) * 1000
            total_ms = (time.monotonic() - loop_t0) * 1000

            pos = pose_world[:3]
            print(f"[step {step:4d}]  infer={infer_ms:5.0f}ms  exec={exec_ms:5.0f}ms  "
                  f"total={total_ms:5.0f}ms  "
                  f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]  "
                  f"grip={'C' if current_gripper > 0.5 else 'O'}")

            step += 1

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        print("Cleaning up...")
        try:
            rtde_c.servoStop()
        except Exception:
            pass
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        try:
            gripper_hw.disconnect()
        except Exception:
            pass
        cam.close()
        print(f"Done. Executed {step} inference steps.")


if __name__ == "__main__":
    main()
