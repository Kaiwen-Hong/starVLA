"""
Move robot to init position using EE (Cartesian) pose via RTDE.
Pose is defined in unified world frame.

Usage:
    python ur5n/go_init.py
    python ur5n/go_init.py --arm right
    python ur5n/go_init.py --speed 0.1
"""

import os
import click
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import modular_policy

# Load robot base poses in world frame
_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}

# Init position from SLAM file (SLAM device frame rotation)
# Units: meters and radians (axis-angle)
INIT_POSE_SLAM = [0.155666, -0.428459, 0.185173, 2.130869, 0.107971, -2.304919] # start

# INIT_POSE_SLAM = [0.212796, -0.475417, 0.145049, 2.133150, 0.037575, -2.308580]

def slam_to_gripper_rotation(rotvec):
    """Correct SLAM device rotation to UR5 gripper rotation.
    Same transform as replay_dataset.py load_session()."""
    T_inv = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # T_SLAM_GRIPPER^T
    R = Rot.from_rotvec(rotvec).as_matrix() @ T_inv
    return Rot.from_matrix(R).as_rotvec()


# Convert to gripper frame for UR5
INIT_POSE_WORLD = list(INIT_POSE_SLAM[:3]) + list(slam_to_gripper_rotation(INIT_POSE_SLAM[3:6]))

ROBOT_IPS = {
    'left': '192.168.0.3',
    'right': '192.168.0.2',
}


def world_to_base(pose_world, T_bw):
    """Convert [x,y,z,rx,ry,rz] from world frame to robot base frame.
    Since R_bw is identity, only translation changes."""
    pose_base = list(pose_world)
    pose_base[0] -= T_bw[0, 3]
    pose_base[1] -= T_bw[1, 3]
    pose_base[2] -= T_bw[2, 3]
    return pose_base


def base_to_world(pose_base, T_bw):
    """Convert [x,y,z,rx,ry,rz] from robot base frame to world frame."""
    pose_world = list(pose_base)
    pose_world[0] += T_bw[0, 3]
    pose_world[1] += T_bw[1, 3]
    pose_world[2] += T_bw[2, 3]
    return pose_world


@click.command()
@click.option('--arm', '-a', default='left',
              type=click.Choice(['left', 'right'], case_sensitive=False),
              help="Which arm to move (default: left)")
@click.option('--speed', '-s', default=0.25, type=float,
              help="Movement speed in m/s (default: 0.25)")
@click.option('--acceleration', '-acc', default=1.2, type=float,
              help="Acceleration in m/s^2 (default: 1.2)")
def main(arm, speed, acceleration):
    """Move robot arm to init position using Cartesian pose (world frame)"""
    robot_ip = ROBOT_IPS[arm]
    T_bw = BASE_IN_WORLD[arm]

    print(f"Connecting to {arm} arm at {robot_ip}...")
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    current_base = rtde_r.getActualTCPPose()
    current_world = base_to_world(current_base, T_bw)
    print(f"Current (world): {[round(x, 4) for x in current_world]}")
    print(f"Target  (world): {[round(x, 4) for x in INIT_POSE_WORLD]}")

    init_pose_base = world_to_base(INIT_POSE_WORLD, T_bw)
    target_joints = rtde_c.getInverseKinematics(init_pose_base)
    print("Moving to init position via moveJ...")
    rtde_c.moveJ(target_joints, speed * 4, acceleration)

    final_base = rtde_r.getActualTCPPose()
    final_world = base_to_world(final_base, T_bw)
    print(f"Final   (world): {[round(x, 4) for x in final_world]}")

    rtde_c.stopScript()
    print("Done!")


if __name__ == '__main__':
    main()
