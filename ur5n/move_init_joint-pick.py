"""
Move robot to init position using joint angles via RTDE.

Usage:
    python ur5n/move_init_joint.py
    python ur5n/move_init_joint.py --arm right
    python ur5n/move_init_joint.py --speed 0.5
"""

import click
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

ROBOT_IPS = {
    'left': '192.168.0.3',
    'right': '192.168.0.2',
}

# Init joint angles in degrees
# left:  [-148.81, -114.47, -84.39, -68.65, 85.89, 208.29]
# right: TODO
INIT_JOINTS_DEG = {
    # 'left': [-148.81, -114.47, -84.39, -68.65, 85.89, 208.29],
    'left': [-153.45, -94.74, -102.94, -69.45, 86.09, 203.62],

}

INIT_JOINTS_RAD = {
    arm: [np.deg2rad(d) for d in degs]
    for arm, degs in INIT_JOINTS_DEG.items()
}


@click.command()
@click.option('--arm', '-a', default='left',
              type=click.Choice(['left', 'right'], case_sensitive=False),
              help="Which arm to move (default: left)")
@click.option('--speed', '-s', default=1.0, type=float,
              help="Joint speed in rad/s (default: 1.0)")
@click.option('--acceleration', '-acc', default=1.2, type=float,
              help="Acceleration in rad/s^2 (default: 1.2)")
def main(arm, speed, acceleration):
    """Move robot arm to init position using joint angles"""

    if arm not in INIT_JOINTS_RAD:
        print(f"No init joints defined for {arm} arm yet.")
        return

    robot_ip = ROBOT_IPS[arm]
    target_joints = INIT_JOINTS_RAD[arm]

    print(f"Connecting to {arm} arm at {robot_ip}...")
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    current_joints = rtde_r.getActualQ()
    print(f"Current (deg): {[round(np.rad2deg(j), 2) for j in current_joints]}")
    print(f"Target  (deg): {INIT_JOINTS_DEG[arm]}")

    print("Moving to init position via moveJ...")
    rtde_c.moveJ(target_joints, speed, acceleration)

    final_joints = rtde_r.getActualQ()
    print(f"Final   (deg): {[round(np.rad2deg(j), 2) for j in final_joints]}")

    rtde_c.stopScript()
    print("Done!")


if __name__ == '__main__':
    main()
