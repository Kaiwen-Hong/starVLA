"""
Move robot to home position using EE (Cartesian) pose via RTDE.
Home poses are defined in unified world frame.

Usage:
    python gohome_ee.py --arm right
    python gohome_ee.py --arm left
    python gohome_ee.py --arm both
"""

import os
import click
import numpy as np
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

# Home positions in WORLD frame [x, y, z, rx, ry, rz, gripper]
# Units: meters and radians (axis-angle), gripper: 1=open, 0=closed
# Rotation is the same in both frames (R_bw = I)
HOME_POSES_WORLD = {
    'left': [0.30, -0.2, 0.25, -2.2192, 2.2148, 0.0091, 1],
    'right': [-0.1, -0.3, 0.25, 2.2419, -2.1984, 0.0166, 1],
}

# HOME_POSES_WORLD = {
#     'left': [0.150, -0.2, 0.25,  1.2092, -1.2092, -1.2092, 1],
#     'right': [-0.1, -0.3, 0.2, 2.2419, -2.1984, 0.0166, 1],
# }

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


def set_gripper(robot_ip, gripper_value):
    """Set Robotiq gripper position. 1=open, 0=closed."""
    from robotiq_gripper import RobotiqGripper
    gripper = RobotiqGripper()
    gripper.connect(hostname=robot_ip, port=63352)
    pos = int((1.0 - gripper_value) * 255)  # 1=open→pos=0, 0=closed→pos=255
    label = "Opening" if gripper_value > 0.5 else "Closing"
    print(f"{label} gripper (pos={pos})...")
    gripper.move(pos, 255, 150)
    print("Gripper done.")


def move_to_home(arm, robot_ip, home_pose_world, speed=0.25, acceleration=1.2):
    """Move robot to home EE pose (specified in world frame)"""
    print(f"Connecting to robot at {robot_ip}...")

    ee_pose = home_pose_world[:6]
    gripper_value = home_pose_world[6]

    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    T_bw = BASE_IN_WORLD[arm]

    # Get current pose in world frame
    current_base = rtde_r.getActualTCPPose()
    current_world = base_to_world(current_base, T_bw)

    print(f"Current (world): {[round(x, 4) for x in current_world]}")
    print(f"Target  (world): {[round(x, 4) for x in ee_pose]}")

    # Use moveJ (joint space) — no Cartesian singularity issues
    home_pose_base = world_to_base(ee_pose, T_bw)
    target_joints = rtde_c.getInverseKinematics(home_pose_base)
    print("Moving to home position via moveJ...")
    rtde_c.moveJ(target_joints, 1.0, 1.0)

    # Verify
    final_base = rtde_r.getActualTCPPose()
    final_world = base_to_world(final_base, T_bw)
    print(f"Final   (world): {[round(x, 4) for x in final_world]}")

    rtde_c.stopScript()

    # Set gripper
    set_gripper(robot_ip, gripper_value)
    print("Done!")


@click.command()
@click.option('--arm', '-a', default='left',
              type=click.Choice(['left', 'right', 'both'], case_sensitive=False),
              help="Which arm to move: left, right, or both")
@click.option('--speed', '-s', default=0.25, type=float,
              help="Movement speed in m/s (default: 0.25)")
@click.option('--acceleration', '-acc', default=1.2, type=float,
              help="Acceleration in m/s^2 (default: 1.2)")
def main(arm, speed, acceleration):
    """Move robot arm(s) to home position using Cartesian pose (world frame)"""

    if arm == 'both':
        for arm_name in ['left', 'right']:
            print(f"\n=== {arm_name.upper()} ARM ===")
            move_to_home(arm_name, ROBOT_IPS[arm_name], HOME_POSES_WORLD[arm_name], speed, acceleration)
    else:
        move_to_home(arm, ROBOT_IPS[arm], HOME_POSES_WORLD[arm], speed, acceleration)


if __name__ == '__main__':
    main()
