"""
Keyboard teleoperation for UR5 robot in Cartesian (EE) space.
Same as teleop_with_keyboard.py but the position grid is 1 mm.

Controls (world frame):
    W  → -Y          S  → +Y
    A  → +X          D  → -X
    Up → +Z        Down → -Z

    Z/X → Roll  -/+
    C/V → Pitch -/+
    B/N → Yaw   -/+

    G → toggle gripper (open/close)
    Step size: 1 mm / 0.05 rad per key press.
    Q / Esc → quit

Usage:
    python ur5n/1mmgrid-teleop.py
    python ur5n/1mmgrid-teleop.py --arm right
    python ur5n/1mmgrid-teleop.py --step 0.002 --rot-step 0.1
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


# ── gripper ──────────────────────────────────────────────────────────

def set_gripper(robot_ip, open_gripper):
    """Set Robotiq gripper. open_gripper=True → open, False → close."""
    from robotiq_gripper import RobotiqGripper
    gripper = RobotiqGripper()
    gripper.connect(hostname=robot_ip, port=63352)
    pos = 0 if open_gripper else 255
    label = "Opening" if open_gripper else "Closing"
    print(f"\n  {label} gripper...")
    gripper.move(pos, 255, 150)


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

def key_to_delta(key, step, rot_step):
    """Return (dx, dy, dz, drx, dry, drz) in world frame, or 'GRIP' to toggle, or None to quit."""
    key_upper = key.upper()

    pos_map = {
        'W': (0, -step, 0, 0, 0, 0),
        'S': (0, +step, 0, 0, 0, 0),
        'A': (+step, 0, 0, 0, 0, 0),
        'D': (-step, 0, 0, 0, 0, 0),
        'UP': (0, 0, +step, 0, 0, 0),
        'DOWN': (0, 0, -step, 0, 0, 0),
    }

    rot_map = {
        'Z': (0, 0, 0, -rot_step, 0, 0),
        'X': (0, 0, 0, +rot_step, 0, 0),
        'C': (0, 0, 0, 0, -rot_step, 0),
        'V': (0, 0, 0, 0, +rot_step, 0),
        'B': (0, 0, 0, 0, 0, -rot_step),
        'N': (0, 0, 0, 0, 0, +rot_step),
    }

    if key_upper in ('Q', 'ESC'):
        return None
    if key_upper == 'G':
        return 'GRIP'
    if key_upper in pos_map:
        return pos_map[key_upper]
    if key_upper in rot_map:
        return rot_map[key_upper]
    return 'UNKNOWN'


# ── main ──────────────────────────────────────────────────────────────

@click.command()
@click.option('--arm', '-a', default='left',
              type=click.Choice(['left', 'right'], case_sensitive=False))
@click.option('--step', '-s', default=0.001, type=float,
              help='Position step size in meters (default: 0.001 = 1 mm)')
@click.option('--rot-step', '-r', default=0.05, type=float,
              help='Rotation step size in radians (default: 0.05 ≈ 2.9°)')
@click.option('--speed', default=0.25, type=float,
              help='Movement speed in m/s')
@click.option('--acceleration', default=1.2, type=float,
              help='Acceleration in m/s^2')
def main(arm, step, rot_step, speed, acceleration):
    """Keyboard teleoperation — move EE in world frame (position + rotation + gripper)"""

    robot_ip = ROBOT_IPS[arm]
    T_bw = BASE_IN_WORLD[arm]

    print(f"Connecting to {arm} arm at {robot_ip}...")
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    gripper_open = True

    print("\n── Keyboard Teleop (1 mm grid) ──")
    print(f"  Pos step : {step * 1000:.2f} mm")
    print(f"  Rot step : {rot_step:.3f} rad ({rot_step * 57.2958:.1f}°)")
    print("  W/S  → -Y/+Y    A/D  → +X/-X    ↑/↓ → +Z/-Z")
    print("  Z/X  → Roll-/+   C/V → Pitch-/+  B/N → Yaw-/+")
    print("  G → toggle gripper")
    print("  Q / Esc → quit\n")

    try:
        while True:
            cur_base = rtde_r.getActualTCPPose()
            cur_world = base_to_world(cur_base, T_bw)
            print(f"  pos=({cur_world[0]:+.4f}, {cur_world[1]:+.4f}, {cur_world[2]:+.4f})  "
                  f"rot=({cur_world[3]:+.4f}, {cur_world[4]:+.4f}, {cur_world[5]:+.4f})  "
                  f"grip={'open' if gripper_open else 'closed'}",
                  end='\r')

            key = get_key()
            delta = key_to_delta(key, step, rot_step)

            if delta is None:
                print("\nExiting teleop.")
                break

            if delta == 'GRIP':
                gripper_open = not gripper_open
                set_gripper(robot_ip, gripper_open)
                continue

            if delta == 'UNKNOWN':
                continue

            dx, dy, dz, drx, dry, drz = delta
            new_world = [
                cur_world[0] + dx,
                cur_world[1] + dy,
                cur_world[2] + dz,
                cur_world[3] + drx,
                cur_world[4] + dry,
                cur_world[5] + drz,
            ]

            new_base = world_to_base(new_world, T_bw)
            target_joints = rtde_c.getInverseKinematics(new_base)
            rtde_c.moveJ(target_joints, 1.0, 1.0)

            final_base = rtde_r.getActualTCPPose()
            final_world = base_to_world(final_base, T_bw)
            print(f"  pos=({final_world[0]:+.4f}, {final_world[1]:+.4f}, {final_world[2]:+.4f})  "
                  f"rot=({final_world[3]:+.4f}, {final_world[4]:+.4f}, {final_world[5]:+.4f})  "
                  f"grip={'open' if gripper_open else 'closed'}")

    finally:
        rtde_c.stopScript()
        print("Robot script stopped.")


if __name__ == '__main__':
    main()
