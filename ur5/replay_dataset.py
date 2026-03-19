#!/usr/bin/env python3
"""
Dataset Replay Script for UR5e Robot

Replays recorded demonstrations on real UR5e robot.
Supports session directories (raw SLAM data) and zarr datasets.
Control modes: EEF (end-effector) or joint.
"""

# ============ DATA PATH CONFIG (edit here) ============
DATA_BASE = "/home/kaiwen/Desktop/research/fastumipro-collection/data_collector_opt/DATA"
# DATA_SESSION = "left_hand_250801DR48FP25002960/pickandplace-314-v2/session_20260314_172507"
DATA_SESSION = "left_hand_250801DR48FP25002960/pickandplace-314/session_20260314_172507"
DATA_PATH = f"{DATA_BASE}/{DATA_SESSION}"
# ======================================================

import os
import sys
import argparse
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as Rot
import modular_policy

project_root = Path(modular_policy.__file__).parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

from modular_policy.real_world.real_ur5e_env import RealUR5eEnv
from modular_policy.common.trans_utils import are_joints_close
from modular_policy.common.precise_sleep import precise_wait
from robotiq_gripper import RobotiqGripper

_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}

# Home poses (EEF) in WORLD frame: [x, y, z, rx, ry, rz]
HOME_POSE_WORLD = {
    'left': [0.20, -0.2, 0.2, -2.2192, 2.2148, 0.0091],
}

ROBOT_IPS = {'left': '192.168.0.3', 'right': '192.168.0.2'}

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ActionData:
    actions: np.ndarray              # (N, 6): joint or eef poses
    gripper_actions: Optional[np.ndarray] = None  # (N, 1): gripper values


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_zarr_dataset(zarr_path: str, episode: int, ctrl_mode: str,
                      joint_source: str = 'robot_joint') -> ActionData:
    """Load actions from a zarr dataset for a given episode."""
    group = zarr.open_group(zarr_path, 'r')
    data = group['data']
    episode_ends = group['meta']['episode_ends'][:]

    # Flatten observations if raw format
    if 'observations' in data:
        data_dict = {}
        for k, v in data['observations'].items():
            data_dict[k] = v
        for k, v in data.items():
            if k != 'observations':
                data_dict[k] = v
        data = data_dict

    ep_start = 0 if episode == 0 else episode_ends[episode - 1]
    ep_end = episode_ends[episode]
    s = slice(ep_start, ep_end)
    logger.info(f"Episode {episode}: steps {ep_start}-{ep_end - 1}")

    if ctrl_mode == 'joint':
        actions = data[joint_source][s]
        gripper = actions[:, 6:7] if actions.shape[1] > 6 else None
        actions = actions[:, :6]
    else:
        actions = data['robot_eef_pose'][s]
        gripper = actions[:, 6:7] if actions.shape[1] >= 7 else None

    logger.info(f"Loaded {len(actions)} actions ({ctrl_mode}, shape {actions.shape})")
    return ActionData(actions=actions, gripper_actions=gripper)


def load_session(session_dir: str) -> ActionData:
    """Load EEF actions from a raw session directory (SLAM poses)."""
    slam_dir = os.path.join(session_dir, 'SLAM_Poses')
    pose_file = os.path.join(slam_dir, 'slam_raw_pose_worldframe_downsampled.txt')

    # Auto-generate if missing
    if not os.path.isfile(pose_file):
        slam_script = os.path.join(project_root, 'data_collector_opt', 'slam_analyze_and_vel.py')
        slam_raw = os.path.join(slam_dir, 'slam_raw.txt')
        logger.info(f"Running SLAM processing on {slam_raw}")
        result = subprocess.run([sys.executable, slam_script, '-i', slam_raw],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SLAM processing failed:\n{result.stderr}")

    # Parse: time x y z rx ry rz clamp_open
    rows = []
    with open(pose_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                rows.append([float(v) for v in parts[1:8]])

    data = np.array(rows, dtype=np.float64)
    actions = data[:, :6]
    gripper = data[:, 6:7]

    # Correct SLAM device rotation → gripper rotation
    T_inv = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # T_SLAM_GRIPPER^T
    for i in range(len(actions)):
        R = Rot.from_rotvec(actions[i, 3:6]).as_matrix() @ T_inv
        actions[i, 3:6] = Rot.from_matrix(R).as_rotvec()

    # Convert clamp_open → Robotiq: <0.8 means grasping → close (1.0)
    gripper = np.where(gripper < 0.8, 1.0, 0.0)
    logger.info(f"Loaded {len(actions)} poses, gripper closed: "
                f"{int((gripper > 0.5).sum())}/{len(gripper)} steps")
    return ActionData(actions=actions, gripper_actions=gripper)


def world_to_base(action_data: ActionData, arm: str) -> ActionData:
    """Convert EEF actions from world frame to robot base frame."""
    offset = BASE_IN_WORLD[arm][:3, 3]
    actions = action_data.actions.copy()
    actions[:, :3] -= offset
    logger.info(f"World→base frame (offset {offset})")
    return ActionData(actions=actions, gripper_actions=action_data.gripper_actions)


# ── Replay ────────────────────────────────────────────────────────────────────

def open_gripper(robot_ip: str):
    """Open gripper with standalone connection (clean, no stutter)."""
    gripper = RobotiqGripper()
    gripper.connect(hostname=robot_ip, port=63352)
    logger.info("Opening gripper...")
    gripper.move(0, 255, 150)
    gripper.disconnect()


def go_home(env, arm: str):
    """Move robot to home pose (world→base frame conversion)."""
    if arm not in HOME_POSE_WORLD:
        return
    pose_world = HOME_POSE_WORLD[arm]
    offset = BASE_IN_WORLD[arm][:3, 3]
    pose_base = [pose_world[i] - offset[i] for i in range(3)] + pose_world[3:] + [0.0]

    logger.info(f"Home pose (world): {pose_world}")
    env.exec_actions(
        joint_actions=np.zeros((1, 6)),
        eef_actions=np.array([pose_base], dtype=np.float64),
        timestamps=np.array([time.time() + 2.0]),
        mode='eef',
    )
    time.sleep(3.0)
    logger.info("Reached home pose.")


def add_gripper_to_actions(actions: np.ndarray, gripper: Optional[np.ndarray],
                           use_gripper: bool) -> np.ndarray:
    """Concatenate gripper column to actions if available."""
    if gripper is not None and use_gripper and actions.shape[1] < 7:
        actions = np.concatenate([actions, gripper], axis=1)
        logger.info(f"Actions with gripper: {actions.shape}")
    return actions


def replay(action_data: ActionData, env_config: Dict, frequency: int):
    """Main replay: open gripper → go home → execute trajectory."""
    arm = env_config['single_arm_type']
    robot_ip = ROBOT_IPS[arm]

    open_gripper(robot_ip)

    with RealUR5eEnv(**env_config) as env:
        go_home(env, arm)

        n = len(action_data.actions)
        timestamps = time.time() + np.arange(n) / frequency + 1.0
        logger.info(f"Replaying {n} actions at {frequency} Hz ({n / frequency:.1f}s)...")

        use_gripper = env_config.get('use_gripper', True)
        if env_config['ctrl_mode'] == 'joint':
            joints = add_gripper_to_actions(
                action_data.actions, action_data.gripper_actions, use_gripper)
            env.exec_actions(
                joint_actions=joints,
                eef_actions=np.zeros((n, 7)),
                timestamps=timestamps, mode='joint')
        else:
            eef = add_gripper_to_actions(
                action_data.actions, action_data.gripper_actions, use_gripper)
            env.exec_actions(
                joint_actions=np.zeros((n, 6)),
                eef_actions=eef,
                timestamps=timestamps, mode='eef')

        precise_wait(time.monotonic() + n / frequency + 3.0)
        logger.info("Replay completed.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Replay dataset on UR5e robot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # use DATA_PATH
  %(prog)s /path/to/session_dir               # raw session (auto SLAM)
  %(prog)s data/dataset.zarr                  # zarr dataset
  %(prog)s data/dataset.zarr -m joint -f 10   # joint mode at 10Hz
        """)
    parser.add_argument('dataset', nargs='?', default=DATA_PATH)
    parser.add_argument('-e', '--episode', type=int, default=0)
    parser.add_argument('-m', '--mode', choices=['joint', 'eef'], default='eef')
    parser.add_argument('-j', '--joint-source', choices=['robot_joint', 'joint_action'],
                        default='robot_joint')
    parser.add_argument('-s', '--speed', type=float, default=1.0,
                        help='Speed slider 0.0-1.0')
    parser.add_argument('-f', '--frequency', type=int, default=20,
                        help='Replay frequency in Hz')
    parser.add_argument('--arm', choices=['left', 'right'], default='left')
    parser.add_argument('--no-gripper', action='store_true')
    parser.add_argument('--robot-left-ip', default='192.168.0.3')
    parser.add_argument('--robot-right-ip', default='192.168.0.2')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    # Load data
    is_session = os.path.isdir(args.dataset) and os.path.isdir(
        os.path.join(args.dataset, 'SLAM_Poses'))

    if is_session:
        if args.mode != 'eef':
            logger.warning("Session data only supports EEF mode, switching.")
            args.mode = 'eef'
        action_data = load_session(args.dataset)
    else:
        action_data = load_zarr_dataset(
            args.dataset, args.episode, args.mode, args.joint_source)

    if args.mode == 'eef':
        action_data = world_to_base(action_data, args.arm)

    env_config = {
        'output_dir': '/tmp/ur5_replay',
        'ctrl_mode': args.mode,
        'speed_slider_value': args.speed,
        'single_arm_type': args.arm,
        'use_gripper': not args.no_gripper,
        'robot_left_ip': args.robot_left_ip,
        'robot_right_ip': args.robot_right_ip,
        'tactile_sensors': None,
        'max_pos_speed': 0.25,
        'max_rot_speed': 0.6,
    }

    logger.info(f"Config: mode={args.mode}, freq={args.frequency}Hz, "
                f"arm={args.arm}, speed={args.speed}, gripper={not args.no_gripper}")

    replay(action_data, env_config, args.frequency)


if __name__ == '__main__':
    main()
