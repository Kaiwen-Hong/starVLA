"""
Dual-arm SpaceMouse teleoperation for UR5 in Cartesian space (XY only, world frame).

Smooth streaming control via RTDE servoJ. Z and orientation are held at the pose
captured at startup -- only world XY moves. Per-arm workspace bounds and inter-EE
collision distance are enforced every control tick.

At startup the script asks you to wiggle the SpaceMouse you want to control the
LEFT arm with -- the device with larger motion magnitude is assigned to LEFT,
the other to RIGHT. (The two SpaceMice expose no serial number, so we cannot
tell them apart by static USB info.)

Side-channel keyboard:
    P       -> print full pose for both arms
    Q / Esc -> quit

Mapping (matches modular_policy/real_world/bimanual_spacemouse.py tx_zup):
    SpaceMouse push forward (+sm_y) -> world +X
    SpaceMouse push right   (+sm_x) -> world -Y
    Z and rotation channels are dropped.

Usage:
    python ur5n/dual-teleop_with_spacemouse-only_xyz.py
    python ur5n/dual-teleop_with_spacemouse-only_xyz.py --max-vel 0.5 --rate 125
"""

import os
import sys
import time
import select
import termios
import tty
import threading
import queue

import click
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import modular_policy

# NOTE: We intentionally avoid `import hid` (cython-hidapi 0.15.0 ships with the
# libusb backend on Linux, which fails to detach the kernel `usbhid` driver and
# returns "open failed" for these devices). We instead read /dev/hidrawN
# directly -- works at native kernel speed, no extra dependency.


# -- robot config --------------------------------------------------------
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

# World-frame XY workspace bounds (matches dual-teleop_with_keyboard-only_xyz.py)
WORKSPACE_BOUNDS = {
    'left':  {'x': (0.2512, 0.49),    'y': (-0.23, -0.0012)},
    'right': {'x': (-0.1873, -0.0179),'y': (-0.21,  0.014)},
}


# -- coordinate transforms ---------------------------------------------
def world_to_base(pose_world, T_bw):
    p = list(pose_world)
    p[0] -= T_bw[0, 3]
    p[1] -= T_bw[1, 3]
    p[2] -= T_bw[2, 3]
    return p


def base_to_world(pose_base, T_bw):
    p = list(pose_base)
    p[0] += T_bw[0, 3]
    p[1] += T_bw[1, 3]
    p[2] += T_bw[2, 3]
    return p


# -- spacemouse low-level (raw /dev/hidrawN reader) --------------------
VENDOR_ID = 0x256f   # 9583
PRODUCT_ID = 0xc635  # 50741


def _to_int16(y1, y2):
    x = y1 | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def _scale(x, axis_scale=350.0):
    x = x / axis_scale
    return max(-1.0, min(1.0, x))


def _conv(b1, b2):
    return _scale(_to_int16(b1, b2))


def find_spacemouse_hidraw_nodes():
    """Scan /sys/class/hidraw and return /dev/hidrawN paths whose USB parent
    matches our SpaceMouse VID/PID."""
    paths = []
    for name in sorted(os.listdir('/sys/class/hidraw')):
        uevent = f'/sys/class/hidraw/{name}/device/uevent'
        if not os.path.exists(uevent):
            continue
        with open(uevent) as f:
            content = f.read()
        # HID_ID=0003:0000256F:0000C635
        for line in content.splitlines():
            if line.startswith('HID_ID='):
                parts = line.split('=', 1)[1].split(':')
                if len(parts) == 3:
                    vid = int(parts[1], 16)
                    pid = int(parts[2], 16)
                    if vid == VENDOR_ID and pid == PRODUCT_ID:
                        paths.append(f'/dev/{name}')
                break
    return paths


class SpaceMouseRaw:
    """Tiny non-blocking reader for one /dev/hidrawN node."""

    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)

    def close(self):
        try:
            os.close(self.fd)
        except Exception:
            pass

    def read_events(self):
        """Drain pending reports; return (motion[6], buttons[2], has_motion)."""
        motion = np.zeros(6, dtype=np.float32)
        buttons = [False, False]
        has_motion = False
        for _ in range(20):
            try:
                d = os.read(self.fd, 64)
            except BlockingIOError:
                break
            except OSError:
                break
            if not d:
                break
            if d[0] == 1 and len(d) >= 7:
                motion[0] = _conv(d[3], d[4])
                motion[1] = _conv(d[1], d[2])
                motion[2] = _conv(d[5], d[6]) * -1
                if len(d) >= 13:
                    motion[3] = _conv(d[7], d[8])
                    motion[4] = _conv(d[9], d[10])
                    motion[5] = _conv(d[11], d[12])
                has_motion = True
            elif d[0] == 2 and len(d) >= 7:
                motion[3] = _conv(d[1], d[2])
                motion[4] = _conv(d[3], d[4])
                motion[5] = _conv(d[5], d[6])
                has_motion = True
            elif d[0] == 3 and len(d) >= 2:
                buttons[0] = bool(d[1] & 1)
                buttons[1] = bool(d[1] & 2)
        return motion, buttons, has_motion


def open_spacemice():
    paths = find_spacemouse_hidraw_nodes()
    print(f"Found {len(paths)} SpaceMouse hidraw node(s):")
    for i, p in enumerate(paths):
        print(f"  [{i}] {p}")
    if len(paths) < 2:
        raise RuntimeError(
            f"Need 2 SpaceMice, found {len(paths)}. Check USB / udev / permissions.")
    return [SpaceMouseRaw(p) for p in paths[:2]]


def calibrate_left_right(handles, calib_seconds=1.0):
    """Ask user to wiggle the LEFT spacemouse; the more-active device wins."""
    print("\n-- Calibration --")
    print("  Wiggle ONLY the SpaceMouse you want to control the LEFT arm.")
    input(f"  Press Enter, then move it noticeably for {calib_seconds:.1f} seconds...")
    # Drain stale events
    for h in handles:
        for _ in range(40):
            try:
                if not os.read(h.fd, 64):
                    break
            except BlockingIOError:
                break
    mag = [0.0] * len(handles)
    t_end = time.time() + calib_seconds
    while time.time() < t_end:
        for i, h in enumerate(handles):
            m, _, has = h.read_events()
            if has:
                mag[i] += float(np.linalg.norm(m[:3]))
        time.sleep(0.005)
    print(f"  Motion magnitudes: {[round(x, 2) for x in mag]}")
    if max(mag) < 1.0:
        raise RuntimeError("No clear motion detected. Re-run and move harder.")
    left_idx = int(np.argmax(mag))
    right_idx = 1 - left_idx
    print(f"  -> {handles[left_idx].path} = LEFT, "
          f"{handles[right_idx].path} = RIGHT\n")
    return {'left': handles[left_idx], 'right': handles[right_idx]}


# -- keyboard side-channel (raw mode, background thread) ----------------
class KeyReader:
    def __init__(self):
        self.q = queue.Queue()
        self._stop = threading.Event()
        self.fd = sys.stdin.fileno()
        self._old = None
        self._thread = None

    def start(self):
        self._old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old)

    def _run(self):
        while not self._stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                # absorb optional CSI sequence
                r2, _, _ = select.select([sys.stdin], [], [], 0.001)
                if r2:
                    sys.stdin.read(1)
                    r3, _, _ = select.select([sys.stdin], [], [], 0.001)
                    if r3:
                        sys.stdin.read(1)
                self.q.put('ESC')
            else:
                self.q.put(ch.upper())

    def drain(self):
        keys = []
        while True:
            try:
                keys.append(self.q.get_nowait())
            except queue.Empty:
                break
        return keys


# -- pose printing (mirrors keyboard version) ---------------------------
def print_full_pose(arm, rtde_r_arm):
    joints = rtde_r_arm.getActualQ()
    pose_base = rtde_r_arm.getActualTCPPose()
    pose_world = base_to_world(pose_base, BASE_IN_WORLD[arm])
    print(f"\n=== {arm.upper()} ARM ===")
    print(f"Joint angles (rad): {[round(x, 6) for x in joints]}")
    print(f"Joint angles (deg): {[round(x * 57.2958, 2) for x in joints]}")
    print(f"EE Pose (world)   : {[round(x, 6) for x in pose_world]}")
    print(f"EE Pose (base)    : {[round(x, 6) for x in pose_base]}")


# -- main ---------------------------------------------------------------
@click.command()
@click.option('--max-vel', default=3.1, type=float,
              help='Max world-frame XY velocity at full SpaceMouse deflection (m/s).')
@click.option('--deadzone', default=0.05, type=float,
              help='SpaceMouse deadzone in normalized [-1,1] units.')
@click.option('--rate', default=125.0, type=float, help='Control loop rate (Hz).')
@click.option('--min-dist', default=0.15, type=float,
              help='Minimum allowed inter-EE distance in world frame (m).')
@click.option('--lookahead', default=0.1, type=float, help='servoJ lookahead time.')
@click.option('--gain', default=300, type=int, help='servoJ proportional gain.')
@click.option('--vel-cap', default=3.14, type=float,
              help='servoJ joint velocity cap (rad/s, max allowed by RTDE = pi).')
@click.option('--acc-cap', default=3.14, type=float,
              help='servoJ joint acceleration cap (rad/s^2, max allowed by RTDE = pi).')
@click.option('--calib-seconds', default=1.0, type=float,
              help='How long to listen during left/right calibration.')
def main(max_vel, deadzone, rate, min_dist, lookahead, gain,
         vel_cap, acc_cap, calib_seconds):
    """Dual-arm SpaceMouse streaming teleop (XY only, world frame)."""

    # -- spacemouse: open + calibrate (cooked tty: input() works) ------
    raw_handles = open_spacemice()
    sm = calibrate_left_right(raw_handles, calib_seconds=calib_seconds)

    # -- robots --------------------------------------------------------
    print(f"Connecting LEFT  arm @ {ROBOT_IPS['left']}...")
    rtde_c_l = RTDEControlInterface(ROBOT_IPS['left'])
    rtde_r_l = RTDEReceiveInterface(ROBOT_IPS['left'])
    print(f"Connecting RIGHT arm @ {ROBOT_IPS['right']}...")
    rtde_c_r = RTDEControlInterface(ROBOT_IPS['right'])
    rtde_r_r = RTDEReceiveInterface(ROBOT_IPS['right'])
    rtde_c = {'left': rtde_c_l, 'right': rtde_c_r}
    rtde_r = {'left': rtde_r_l, 'right': rtde_r_r}

    # SpaceMouse -> world frame transform (XY only used).
    # SpaceMouse axes after read_spacemouse(): X=right, Y=forward, Z=up.
    # Push SM forward -> world -X; push SM right -> world +Y (intuitive here).
    tx = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Snapshot starting pose for each arm; XY will move, Z + rotvec are frozen.
    target_world = {}
    frozen_zr = {}  # z and rotvec components held constant
    for arm in ('left', 'right'):
        cur_world = base_to_world(rtde_r[arm].getActualTCPPose(), BASE_IN_WORLD[arm])
        target_world[arm] = list(cur_world)
        frozen_zr[arm] = (cur_world[2], cur_world[3], cur_world[4], cur_world[5])

    # Banner
    print("\n-- Dual-arm SpaceMouse teleop (XY streaming, Z/rot frozen) --")
    print(f"  max_vel={max_vel} m/s   deadzone={deadzone}   rate={rate} Hz")
    print(f"  min inter-EE dist={min_dist*100:.1f} cm")
    for arm in ('left', 'right'):
        b = WORKSPACE_BOUNDS[arm]
        print(f"  ws {arm.upper():5s}: x in [{b['x'][0]}, {b['x'][1]}]  "
              f"y in [{b['y'][0]}, {b['y'][1]}]   z held @ {frozen_zr[arm][0]:+.4f}")
    print("  P -> print full pose    Q / Esc -> quit\n")

    # -- start raw-mode keyboard listener -----------------------------
    keys = KeyReader()
    keys.start()

    dt = 1.0 / rate
    last_motion_t = {'left': time.time(), 'right': time.time()}
    motion_timeout = 0.1
    next_t = time.time()
    block_warn_t = 0.0

    try:
        while True:
            next_t += dt

            # Side-channel keyboard
            for k in keys.drain():
                if k in ('Q', 'ESC'):
                    raise KeyboardInterrupt
                if k == 'P':
                    keys.stop()
                    print_full_pose('left', rtde_r['left'])
                    print_full_pose('right', rtde_r['right'])
                    l = target_world['left']; r = target_world['right']
                    d = float(np.linalg.norm(np.array(l[:3]) - np.array(r[:3])))
                    print(f"Inter-EE world distance: {d:.4f} m\n")
                    keys.start()

            # Read both SpaceMice
            motions = {}
            for arm in ('left', 'right'):
                m, _, has = sm[arm].read_events()
                if has:
                    motions[arm] = m
                    last_motion_t[arm] = time.time()
                elif time.time() - last_motion_t[arm] > motion_timeout:
                    motions[arm] = np.zeros(6, dtype=np.float32)
                else:
                    # Keep last; for safety we zero it -- velocity-style control.
                    motions[arm] = np.zeros(6, dtype=np.float32)

            # Integrate XY in world frame
            proposed = {arm: list(target_world[arm]) for arm in ('left', 'right')}
            for arm in ('left', 'right'):
                sm_xyz = motions[arm][:3].astype(np.float64)
                sm_xyz = np.where(np.abs(sm_xyz) < deadzone, 0.0, sm_xyz)
                world_vel = tx @ sm_xyz * max_vel  # m/s
                proposed[arm][0] += world_vel[0] * dt
                proposed[arm][1] += world_vel[1] * dt
                # Z, rotvec frozen
                z, rx, ry, rz = frozen_zr[arm]
                proposed[arm][2] = z
                proposed[arm][3] = rx
                proposed[arm][4] = ry
                proposed[arm][5] = rz

            # Workspace clamp (XY only)
            for arm in ('left', 'right'):
                b = WORKSPACE_BOUNDS[arm]
                proposed[arm][0] = float(np.clip(proposed[arm][0], b['x'][0], b['x'][1]))
                proposed[arm][1] = float(np.clip(proposed[arm][1], b['y'][0], b['y'][1]))

            # Inter-EE distance: if violated, freeze target this tick
            l_pos = np.array(proposed['left'][:3])
            r_pos = np.array(proposed['right'][:3])
            dist = float(np.linalg.norm(l_pos - r_pos))
            if dist < min_dist:
                proposed = {arm: list(target_world[arm]) for arm in ('left', 'right')}
                now = time.time()
                if now - block_warn_t > 0.5:
                    sys.stdout.write(f"\n  [BLOCKED] EEs would close to {dist:.3f}m "
                                     f"< min {min_dist:.3f}m -- freezing\n")
                    block_warn_t = now

            target_world = proposed

            # IK (qnear=current joints) + servoJ
            for arm in ('left', 'right'):
                base_pose = world_to_base(target_world[arm], BASE_IN_WORLD[arm])
                qnear = rtde_r[arm].getActualQ()
                try:
                    q = rtde_c[arm].getInverseKinematics(base_pose, qnear)
                except Exception as e:
                    sys.stdout.write(f"\n[IK fail {arm}] {e}\n")
                    continue
                rtde_c[arm].servoJ(q, vel_cap, acc_cap, dt, lookahead, gain)

            # Status line
            l = target_world['left']; r = target_world['right']
            sys.stdout.write(
                f"  L=({l[0]:+.4f},{l[1]:+.4f})  R=({r[0]:+.4f},{r[1]:+.4f})  "
                f"d={dist:.3f}m   \r")
            sys.stdout.flush()

            # Pace the loop
            sleep_t = next_t - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                # Fell behind -- resync to now to avoid runaway catchup.
                next_t = time.time()

    except KeyboardInterrupt:
        sys.stdout.write("\nExiting teleop.\n")
    finally:
        keys.stop()
        try:
            rtde_c_l.servoStop()
        except Exception:
            pass
        try:
            rtde_c_r.servoStop()
        except Exception:
            pass
        try:
            rtde_c_l.stopScript()
        except Exception:
            pass
        try:
            rtde_c_r.stopScript()
        except Exception:
            pass
        for h in raw_handles:
            h.close()
        print("Robot scripts stopped.")


if __name__ == '__main__':
    main()
