"""
Get current EE pose from robot, displayed in unified world frame.

Usage:
    python get_ee_pose.py --arm right
    python get_ee_pose.py --arm left
"""

import os
import click
import numpy as np
from rtde_receive import RTDEReceiveInterface
from robotiq_gripper import RobotiqGripper
import modular_policy

ROBOT_IPS = {
    'left': '192.168.0.3',
    'right': '192.168.0.2',
}

# Load robot base poses in world frame
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
    """Read and print current EE pose in world frame"""

    robot_ip = ROBOT_IPS[arm]
    print(f"Connecting to {arm} arm at {robot_ip}...")

    rtde_r = RTDEReceiveInterface(robot_ip)

    pose_base = rtde_r.getActualTCPPose()
    joints = rtde_r.getActualQ()

    pose_world = base_to_world(pose_base, BASE_IN_WORLD[arm])

    print(f"\n=== {arm.upper()} ARM ===")
    print(f"EE Pose (world frame) [x, y, z, rx, ry, rz]:")
    print(f"  {[round(x, 6) for x in pose_world]}")
    print(f"\nEE Pose (base frame)  [x, y, z, rx, ry, rz]:")
    print(f"  {[round(x, 6) for x in pose_base]}")
    print(f"\nJoint angles (rad):")
    print(f"  {[round(x, 6) for x in joints]}")
    print(f"\nJoint angles (deg):")
    print(f"  {[round(x * 57.2958, 2) for x in joints]}")

    # Read gripper position
    gripper = RobotiqGripper()
    gripper.connect(robot_ip, 63352)
    gripper_pos = gripper.get_current_position()
    gripper.disconnect()

    print(f"\nGripper position: {gripper_pos} / 255 ({gripper_pos / 255 * 100:.1f}% closed)")

    print(f"\n# World frame pose for gohome_ee.py:")
    print(f"'{arm}': {[round(x, 4) for x in pose_world]},")


if __name__ == '__main__':
    main()
