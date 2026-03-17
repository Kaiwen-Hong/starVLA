#!/usr/bin/env python3
"""
Debug script: move UR5e to a single pose from SLAM data.

Edit POSE_STRING below, then run:
    python replay_dataset_debug.py

Only depends on: numpy, scipy, rtde_control, rtde_receive
"""

import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# ============ EDIT HERE ============
# Format: timestamp x y z rx ry rz clamp_open
POSE_STRING = "88906.448080000 0.327694 -0.194824 0.118280 -2.104816 0.010745 2.286732 0.989348"

ARM = 'left'
ROBOT_IP = '192.168.0.3'   # left robot
SPEED = 0.25      # m/s
ACCELERATION = 0.5  # m/s^2
# ===================================

# Robot extrinsics (base pose in world frame)
_EXTRINSICS_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'fastumipro-collection',
    'modular_policy', 'real_world', 'robot_extrinsics')
# Fallback: try installed modular_policy package
try:
    import modular_policy
    _EXTRINSICS_DIR = os.path.join(
        os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
except ImportError:
    pass

BASE_IN_WORLD = {
    'left': np.load(os.path.join(_EXTRINSICS_DIR, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_EXTRINSICS_DIR, 'right_base_pose_in_world.npy')),
}

# SLAM device -> gripper rotation mapping
T_SLAM_GRIPPER = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
])


def slam_to_gripper_rotation(rotvec: np.ndarray) -> np.ndarray:
    """Convert a single SLAM axis-angle to gripper axis-angle."""
    T_inv = T_SLAM_GRIPPER.T
    R_slam = Rot.from_rotvec(rotvec).as_matrix()
    R_eef = R_slam @ T_inv
    return Rot.from_matrix(R_eef).as_rotvec()


def parse_pose(pose_str: str):
    """Parse pose string -> (xyz_rotvec(6,), gripper(float))."""
    parts = pose_str.strip().split()
    # skip timestamp (col 0), take x y z rx ry rz clamp_open (cols 1-7)
    vals = [float(v) for v in parts[1:8]]
    xyz = np.array(vals[:3], dtype=np.float64)
    rotvec = np.array(vals[3:6], dtype=np.float64)
    clamp_open = vals[6]

    # SLAM -> gripper rotation correction
    rotvec = slam_to_gripper_rotation(rotvec)

    # Gripper: clamp_open < 0.8 means grasping -> 1.0 (close), else 0.0 (open)
    gripper = 1.0 if clamp_open < 0.8 else 0.0

    pose_world = np.concatenate([xyz, rotvec])  # (6,)
    return pose_world, gripper


def world_to_base(pose_world: np.ndarray, arm: str) -> np.ndarray:
    """Convert EEF pose from world frame to robot base frame (R_bw=I)."""
    T_bw = BASE_IN_WORLD[arm]
    pose_base = pose_world.copy()
    pose_base[0] -= T_bw[0, 3]
    pose_base[1] -= T_bw[1, 3]
    pose_base[2] -= T_bw[2, 3]
    return pose_base


def main():
    pose_world, gripper = parse_pose(POSE_STRING)
    pose_base = world_to_base(pose_world, ARM)

    print("=== Debug Pose ===")
    print(f"Input:        {POSE_STRING}")
    print(f"World  (6D):  {pose_world}")
    print(f"Base   (6D):  {pose_base}")
    print(f"Gripper:      {gripper} ({'closed' if gripper > 0.5 else 'open'})")
    print("==================")

    print(f"Connecting to robot at {ROBOT_IP}...")
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)

    current_pose = rtde_r.getActualTCPPose()
    print(f"Current pose: {np.round(current_pose, 4).tolist()}")
    print(f"Target  pose: {np.round(pose_base, 4).tolist()}")

    input("Press Enter to move robot (Ctrl+C to abort)...")

    print("Moving...")
    rtde_c.moveL(pose_base.tolist(), SPEED, ACCELERATION)
    print("Done.")

    final_pose = rtde_r.getActualTCPPose()
    print(f"Final   pose: {np.round(final_pose, 4).tolist()}")

    rtde_c.stopScript()


if __name__ == '__main__':
    main()
