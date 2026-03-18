"""
Keyboard teleoperation for UR5 robot in Cartesian (EE) space.

Controls (world frame):
    W  → -Y          S  → +Y
    A  → +X          D  → -X
    Up → +Z        Down → -Z

    Step size: 1 cm per key press.
    Q / Esc → quit

Usage:
    python teleop_with_keyboard.py
    python teleop_with_keyboard.py --arm left
    python teleop_with_keyboard.py --step 0.02
"""

import os
import sys
import tty
import termios
import click
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import modular_policy

# ── robot config ──────────────────────────────────────────────────────
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

# ── coordinate transforms ────────────────────────────────────────────

def world_to_base(pose_world, T_bw):
    pose_base = list(pose_world)
    pose_base[0] -= T_bw[0, 3]
    pose_base[1] -= T_bw[1, 3]
    pose_base[2] -= T_bw[2, 3]
    return pose_base


def base_to_world(pose_base, T_bw):
    pose_world = list(pose_base)
    pose_world[0] += T_bw[0, 3]
    pose_world[1] += T_bw[1, 3]
    pose_world[2] += T_bw[2, 3]
    return pose_world


# ── keyboard reading ─────────────────────────────────────────────────

def get_key():
    """Read a single keypress (blocking). Returns string like 'w', 'q', 'UP', 'DOWN'."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':  # escape sequence
            ch2 = sys.stdin.read(1)
            if ch2 == '[':
                ch3 = sys.stdin.read(1)
                arrow_map = {'A': 'UP', 'B': 'DOWN', 'C': 'RIGHT', 'D': 'LEFT'}
                return arrow_map.get(ch3, '')
            return 'ESC'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── key → delta mapping (world frame) ────────────────────────────────

def key_to_delta(key, step):
    """Return (dx, dy, dz) in world frame for a given key, or None to quit."""
    key = key.upper()
    mapping = {
        'W': (0, -step, 0),      # W → -Y
        'S': (0, +step, 0),      # S → +Y
        'A': (+step, 0, 0),      # A → +X
        'D': (-step, 0, 0),      # D → -X
        'UP': (0, 0, +step),     # ↑ → +Z
        'DOWN': (0, 0, -step),   # ↓ → -Z
    }
    if key in ('Q', 'ESC'):
        return None
    return mapping.get(key)


# ── main ──────────────────────────────────────────────────────────────

@click.command()
@click.option('--arm', '-a', default='left',
              type=click.Choice(['left', 'right'], case_sensitive=False))
@click.option('--step', '-s', default=0.01, type=float,
              help='Step size in meters (default: 0.01 = 1 cm)')
@click.option('--speed', default=0.25, type=float,
              help='Movement speed in m/s')
@click.option('--acceleration', default=1.2, type=float,
              help='Acceleration in m/s^2')
def main(arm, step, speed, acceleration):
    """Keyboard teleoperation — move EE in world frame"""

    robot_ip = ROBOT_IPS[arm]
    T_bw = BASE_IN_WORLD[arm]

    print(f"Connecting to {arm} arm at {robot_ip}...")
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    print("\n── Keyboard Teleop ──")
    print(f"  Step size : {step * 100:.1f} cm")
    print("  W/S  → -Y/+Y    A/D  → +X/-X")
    print("  ↑/↓  → +Z/-Z")
    print("  Q / Esc → quit\n")

    try:
        while True:
            # show current pose
            cur_base = rtde_r.getActualTCPPose()
            cur_world = base_to_world(cur_base, T_bw)
            print(f"  EE (world): x={cur_world[0]:+.4f}  y={cur_world[1]:+.4f}  z={cur_world[2]:+.4f}", end='\r')

            key = get_key()
            delta = key_to_delta(key, step)

            if delta is None:  # quit
                print("\nExiting teleop.")
                break

            if isinstance(delta, tuple):
                dx, dy, dz = delta
                # compute new world pose (keep rotation unchanged)
                new_world = list(cur_world)
                new_world[0] += dx
                new_world[1] += dy
                new_world[2] += dz

                new_base = world_to_base(new_world, T_bw)
                target_joints = rtde_c.getInverseKinematics(new_base)
                rtde_c.moveJ(target_joints, 1.0, 1.0)

                final_base = rtde_r.getActualTCPPose()
                final_world = base_to_world(final_base, T_bw)
                print(f"  EE (world): x={final_world[0]:+.4f}  y={final_world[1]:+.4f}  z={final_world[2]:+.4f}")

    finally:
        rtde_c.stopScript()
        print("Robot script stopped.")


if __name__ == '__main__':
    main()
