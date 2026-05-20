"""
Dual-arm keyboard teleoperation for UR5 robots in Cartesian (EE) space.
XYZ only — no rotation, no gripper.

Controls (world frame):
    LEFT  arm:  W → -Y    S → +Y    A → +X    D → -X    ↑ → +Z    ↓ → -Z
    RIGHT arm:  8 → -Y    5 → +Y    4 → +X    6 → -X    + → +Z    - → -Z

    Step size: 1 cm per key press.
    P → print full pose (joints + EE pose, both frames) for both arms
    Q / Esc → quit

Usage:
    python ur5n/dual-teleop_with_keyboard-only_xyz.py
    python ur5n/dual-teleop_with_keyboard-only_xyz.py --step 0.02
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


# ── key → (arm, dx, dy, dz) mapping ──────────────────────────────────

def key_to_action(key, step):
    """Return ('left'|'right', dx, dy, dz), 'PRINT', None to quit, or 'UNKNOWN'."""
    key_upper = key.upper() if len(key) > 1 else key  # preserve raw single char for +/-

    # Quit
    if key_upper in ('Q', 'ESC') or key == 'q':
        return None

    # Print full pose
    if key in ('p', 'P'):
        return 'PRINT'

    # Left arm: W/S/A/D/UP/DOWN
    left_map = {
        'w': ('left', 0, -step, 0),
        's': ('left', 0, +step, 0),
        'a': ('left', +step, 0, 0),
        'd': ('left', -step, 0, 0),
        'UP': ('left', 0, 0, +step),
        'DOWN': ('left', 0, 0, -step),
    }
    if key in left_map:
        return left_map[key]
    if key_upper in left_map:
        return left_map[key_upper]

    # Right arm: 8/5/4/6 and +/-
    right_map = {
        '8': ('right', 0, -step, 0),
        '5': ('right', 0, +step, 0),
        '4': ('right', +step, 0, 0),
        '6': ('right', -step, 0, 0),
        '+': ('right', 0, 0, +step),
        '=': ('right', 0, 0, +step),  # same key as '+' without shift
        '-': ('right', 0, 0, -step),
    }
    if key in right_map:
        return right_map[key]

    return 'UNKNOWN'


# ── pose tracking (mirrors get_joint_pose.py) ────────────────────────

def print_full_pose(arm, rtde_r_arm):
    joints = rtde_r_arm.getActualQ()
    pose_base = rtde_r_arm.getActualTCPPose()
    pose_world = base_to_world(pose_base, BASE_IN_WORLD[arm])
    print(f"\n=== {arm.upper()} ARM ===")
    print(f"Joint angles (rad): {[round(x, 6) for x in joints]}")
    print(f"Joint angles (deg): {[round(x * 57.2958, 2) for x in joints]}")
    print(f"EE Pose (world)   : {[round(x, 6) for x in pose_world]}")
    print(f"EE Pose (base)    : {[round(x, 6) for x in pose_base]}")


# ── safety: per-arm workspace bounds (world frame) ───────────────────
# Block only when a move would take the EE further outside the allowed region.
# If the EE is already outside (boundary mis-set), moves toward / staying are
# still allowed — we never force the robot back to the boundary.
WORKSPACE_BOUNDS = {
    'left': {
        'x': (0.2512, 0.49),
        'y': (-0.23, -0.0012),
    },
    'right': {
        'x': (-0.1873, -0.0179),
        'y': (-0.21, 0.014),
    },
}


def check_workspace(arm, cur_world, new_world):
    """Return (ok, msg). Block iff new pose violates an axis bound AND moves
    further out on that axis. Per-axis check is independent."""
    bounds = WORKSPACE_BOUNDS[arm]
    axis_idx = {'x': 0, 'y': 1, 'z': 2}
    for axis, (lo, hi) in bounds.items():
        i = axis_idx[axis]
        cur_v, new_v = cur_world[i], new_world[i]
        if new_v < lo and new_v < cur_v:
            return False, f"{arm} world {axis}={new_v:.4f} < {lo} (moving outward)"
        if new_v > hi and new_v > cur_v:
            return False, f"{arm} world {axis}={new_v:.4f} > {hi} (moving outward)"
    return True, ""


# ── safety: inter-EE distance check (world frame) ────────────────────

def check_collision(rtde_r, min_dist):
    """Return (ok, dist). ok=False if EEs closer than min_dist in world XYZ."""
    l = base_to_world(rtde_r['left'].getActualTCPPose(), BASE_IN_WORLD['left'])
    r = base_to_world(rtde_r['right'].getActualTCPPose(), BASE_IN_WORLD['right'])
    dist = float(np.linalg.norm(np.array(l[:3]) - np.array(r[:3])))
    return dist >= min_dist, dist


# ── main ──────────────────────────────────────────────────────────────

@click.command()
@click.option('--step', '-s', default=0.01, type=float,
              help='Position step size in meters (default: 0.01 = 1 cm)')
@click.option('--speed', default=1.0, type=float,
              help='moveJ speed')
@click.option('--acceleration', default=1.0, type=float,
              help='moveJ acceleration')
@click.option('--min-dist', default=0.15, type=float,
              help='Minimum allowed inter-EE distance in world frame (m)')
def main(step, speed, acceleration, min_dist):
    """Dual-arm keyboard teleop — XYZ only, world frame."""

    print(f"Connecting to LEFT  arm at {ROBOT_IPS['left']}...")
    rtde_c_left = RTDEControlInterface(ROBOT_IPS['left'])
    rtde_r_left = RTDEReceiveInterface(ROBOT_IPS['left'])

    print(f"Connecting to RIGHT arm at {ROBOT_IPS['right']}...")
    rtde_c_right = RTDEControlInterface(ROBOT_IPS['right'])
    rtde_r_right = RTDEReceiveInterface(ROBOT_IPS['right'])

    rtde_c = {'left': rtde_c_left, 'right': rtde_c_right}
    rtde_r = {'left': rtde_r_left, 'right': rtde_r_right}

    print("\n── Dual-Arm Keyboard Teleop (XYZ only, world frame) ──")
    print(f"  Pos step : {step * 100:.1f} cm")
    print(f"  Min dist : {min_dist * 100:.1f} cm (inter-EE world distance)")
    for arm_name in ('left', 'right'):
        b = WORKSPACE_BOUNDS[arm_name]
        print(f"  Workspace {arm_name.upper():5s}: "
              f"x∈[{b['x'][0]}, {b['x'][1]}]  y∈[{b['y'][0]}, {b['y'][1]}]")
    print("  LEFT : W/S → -Y/+Y    A/D → +X/-X    ↑/↓ → +Z/-Z")
    print("  RIGHT: 8/5 → -Y/+Y    4/6 → +X/-X    +/- → +Z/-Z")
    print("  P → print full pose    Q / Esc → quit\n")

    try:
        while True:
            l_world = base_to_world(rtde_r['left'].getActualTCPPose(), BASE_IN_WORLD['left'])
            r_world = base_to_world(rtde_r['right'].getActualTCPPose(), BASE_IN_WORLD['right'])
            dist = float(np.linalg.norm(np.array(l_world[:3]) - np.array(r_world[:3])))
            print(f"  L=({l_world[0]:+.4f}, {l_world[1]:+.4f}, {l_world[2]:+.4f})  "
                  f"R=({r_world[0]:+.4f}, {r_world[1]:+.4f}, {r_world[2]:+.4f})  "
                  f"d={dist:.3f}m",
                  end='\r')

            key = get_key()
            action = key_to_action(key, step)

            if action is None:
                print("\nExiting teleop.")
                break

            if action == 'UNKNOWN':
                continue

            if action == 'PRINT':
                print_full_pose('left', rtde_r['left'])
                print_full_pose('right', rtde_r['right'])
                print(f"Inter-EE world distance: {dist:.4f} m\n")
                continue

            arm, dx, dy, dz = action
            T_bw = BASE_IN_WORLD[arm]
            cur_world = base_to_world(rtde_r[arm].getActualTCPPose(), T_bw)
            new_world = [
                cur_world[0] + dx,
                cur_world[1] + dy,
                cur_world[2] + dz,
                cur_world[3],
                cur_world[4],
                cur_world[5],
            ]

            # Per-arm workspace bound (only block moves that go further out).
            ok, msg = check_workspace(arm, cur_world, new_world)
            if not ok:
                print(f"\n  [BLOCKED] {msg}")
                continue

            # Predict inter-EE distance after the move; block if too close.
            other = 'right' if arm == 'left' else 'left'
            other_world = base_to_world(rtde_r[other].getActualTCPPose(), BASE_IN_WORLD[other])
            predicted_dist = float(np.linalg.norm(
                np.array(new_world[:3]) - np.array(other_world[:3])))
            if predicted_dist < min_dist:
                print(f"\n  [BLOCKED] {arm} move would bring EEs to "
                      f"{predicted_dist:.3f}m < min {min_dist:.3f}m")
                continue

            new_base = world_to_base(new_world, T_bw)
            target_joints = rtde_c[arm].getInverseKinematics(new_base)
            rtde_c[arm].moveJ(target_joints, speed, acceleration)

    finally:
        rtde_c_left.stopScript()
        rtde_c_right.stopScript()
        print("Robot scripts stopped.")


if __name__ == '__main__':
    main()
