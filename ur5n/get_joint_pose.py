"""
Get current joint angles from UR5 robot.

Usage:
    python ur5n/get_joint_pose.py
    python ur5n/get_joint_pose.py --arm right
"""

import os
import click
import numpy as np
from rtde_receive import RTDEReceiveInterface
import modular_policy

ROBOT_IPS = {
    'left': '192.168.0.3',
    'right': '192.168.0.2',
}

_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}


def base_to_world(pose_base, T_bw):
    """Convert [x,y,z,rx,ry,rz] from robot base frame to world frame.
    Since R_bw is identity, only translation changes."""
    pose_world = list(pose_base)
    pose_world[0] += T_bw[0, 3]
    pose_world[1] += T_bw[1, 3]
    pose_world[2] += T_bw[2, 3]
    return pose_world


@click.command()
@click.option('--arm', '-a', default='left',
              type=click.Choice(['left', 'right'], case_sensitive=False),
              help="Which arm to read: left or right")
def main(arm):
    """Read and print current joint angles and EE pose"""

    robot_ip = ROBOT_IPS[arm]
    print(f"Connecting to {arm} arm at {robot_ip}...")

    rtde_r = RTDEReceiveInterface(robot_ip)

    joints = rtde_r.getActualQ()
    pose_base = rtde_r.getActualTCPPose()
    pose_world = base_to_world(pose_base, BASE_IN_WORLD[arm])

    print(f"\n=== {arm.upper()} ARM ===")
    print(f"Joint angles (rad):")
    print(f"  {[round(x, 6) for x in joints]}")
    print(f"\nJoint angles (deg):")
    print(f"  {[round(x * 57.2958, 2) for x in joints]}")
    print(f"\nEE Pose (world frame) [x, y, z, rx, ry, rz]:")
    print(f"  {[round(x, 6) for x in pose_world]}")
    print(f"\nEE Pose (base frame)  [x, y, z, rx, ry, rz]:")
    print(f"  {[round(x, 6) for x in pose_base]}")


if __name__ == '__main__':
    main()
