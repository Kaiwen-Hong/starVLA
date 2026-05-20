"""Standalone debug for the two SpaceMice. Does NOT touch any robot.

Tries multiple ways to enumerate / open / read the devices, prints
everything, and falls back gracefully so we can see exactly where the
existing teleop script breaks.

Usage:
    python ur5n/debug_spacemouse.py                 # default: try everything
    python ur5n/debug_spacemouse.py --listen-secs 8 # longer live read
"""

import os
import sys
import time
import struct
import click
import numpy as np

VENDOR_ID = 0x256f   # 9583  -- 3Dconnexion
PRODUCT_ID = 0xc635  # 50741 -- SpaceMouse Compact / Wireless receiver


def banner(s):
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def section(s):
    print("\n--- " + s + " ---")


# ----------------------------------------------------------------------
# Probe 1: list /dev/hidraw* and identify which two are SpaceMice via sysfs
# ----------------------------------------------------------------------
def probe_hidraw_nodes():
    section("hidraw nodes (matched against VID:PID via sysfs)")
    matches = []
    for i in range(64):
        node = f"/dev/hidraw{i}"
        if not os.path.exists(node):
            continue
        # sysfs: /sys/class/hidraw/hidrawN/device/uevent has HID_ID=...:VID:PID
        uevent = f"/sys/class/hidraw/hidraw{i}/device/uevent"
        vid = pid = None
        if os.path.exists(uevent):
            with open(uevent) as f:
                for line in f:
                    if line.startswith("HID_ID="):
                        # HID_ID=0003:0000256F:0000C635
                        parts = line.strip().split("=", 1)[1].split(":")
                        if len(parts) == 3:
                            vid = int(parts[1], 16)
                            pid = int(parts[2], 16)
        try:
            st = os.stat(node)
            perm = oct(st.st_mode & 0o777)
        except OSError as e:
            perm = f"stat-fail({e})"
        marker = " <-- SpaceMouse" if vid == VENDOR_ID and pid == PRODUCT_ID else ""
        print(f"  {node:16s}  vid=0x{(vid or 0):04x}  pid=0x{(pid or 0):04x}  "
              f"perm={perm}{marker}")
        if vid == VENDOR_ID and pid == PRODUCT_ID:
            matches.append(node)
    if not matches:
        print("  !! No SpaceMouse hidraw nodes found.")
    return matches


# ----------------------------------------------------------------------
# Probe 2: hidapi enumerate
# ----------------------------------------------------------------------
def probe_hidapi_enumerate():
    section("hidapi enumerate(VID, PID)")
    import hid
    devs = hid.enumerate(VENDOR_ID, PRODUCT_ID)
    print(f"  enumerate() returned {len(devs)} entries")
    for i, d in enumerate(devs):
        print(f"  [{i}] keys={list(d.keys())}")
        for k in ('path', 'vendor_id', 'product_id', 'serial_number',
                  'manufacturer_string', 'product_string',
                  'release_number', 'usage_page', 'usage', 'interface_number'):
            if k in d:
                print(f"        {k}: {d[k]!r}")
    return devs


# ----------------------------------------------------------------------
# Probe 3: try opening each enumerated device by path
# ----------------------------------------------------------------------
def probe_open_by_path(devs):
    section("hid.device().open_path(path)")
    import hid
    handles = []
    for i, d in enumerate(devs):
        h = hid.device()
        path = d['path']
        try:
            h.open_path(path)
            print(f"  [{i}] open_path({path!r}) -> OK")
            try:
                h.set_nonblocking(True)
            except Exception as e:
                print(f"        set_nonblocking failed: {e}")
            handles.append((i, h, path))
        except Exception as e:
            print(f"  [{i}] open_path({path!r}) -> FAIL: {type(e).__name__}: {e}")
    return handles


# ----------------------------------------------------------------------
# Probe 4: try opening by VID/PID + serial fallback
# ----------------------------------------------------------------------
def probe_open_by_vid_pid():
    section("hid.device().open(vid, pid)  (only opens FIRST matching device)")
    import hid
    h = hid.device()
    try:
        h.open(VENDOR_ID, PRODUCT_ID)
        print("  open(vid,pid) -> OK")
        try:
            h.set_nonblocking(True)
        except Exception as e:
            print(f"  set_nonblocking failed: {e}")
        return h
    except Exception as e:
        print(f"  open(vid,pid) -> FAIL: {type(e).__name__}: {e}")
        return None


# ----------------------------------------------------------------------
# Probe 5: raw os.open of the matched hidraw nodes (bypasses hidapi)
# ----------------------------------------------------------------------
def probe_raw_open(hidraw_paths):
    section("os.open() on the matched /dev/hidraw* nodes (raw kernel)")
    fds = []
    for p in hidraw_paths:
        try:
            fd = os.open(p, os.O_RDWR | os.O_NONBLOCK)
            print(f"  {p} -> os.open OK fd={fd}")
            fds.append((p, fd))
        except OSError as e:
            print(f"  {p} -> os.open FAIL: {e}")
    return fds


# ----------------------------------------------------------------------
# Live read via hidapi handles
# ----------------------------------------------------------------------
def live_read_hidapi(handles, listen_secs):
    section(f"Live read via hidapi (push the SpaceMice for {listen_secs}s)")
    if not handles:
        print("  no hidapi handles to read.")
        return
    print("  Move both SpaceMice to see motion samples below.")
    t_end = time.time() + listen_secs
    counts = {i: 0 for i, _, _ in handles}
    last_print = 0
    while time.time() < t_end:
        for i, h, path in handles:
            for _ in range(20):
                d = h.read(13)
                if not d:
                    break
                counts[i] += 1
                report_id = d[0]
                # Brief decode of report id 1 (translation+rotation) and id 3 (buttons)
                if report_id == 1 and len(d) >= 7:
                    def s16(a, b):
                        x = a | (b << 8)
                        return x - 65536 if x >= 32768 else x
                    tx = s16(d[1], d[2])
                    ty = s16(d[3], d[4])
                    tz = s16(d[5], d[6])
                    if abs(tx) + abs(ty) + abs(tz) > 50 and time.time() - last_print > 0.05:
                        print(f"  dev[{i}] {path!r}  trans=({tx:+5d},{ty:+5d},{tz:+5d})")
                        last_print = time.time()
                elif report_id == 3:
                    print(f"  dev[{i}] {path!r}  BUTTON byte={d[1]:#04x}")
        time.sleep(0.005)
    print(f"  total bytes-of-reports per device: {counts}")


# ----------------------------------------------------------------------
# Live read via raw fds (fallback if hidapi can't open)
# ----------------------------------------------------------------------
def live_read_raw(fds, listen_secs):
    section(f"Live read via os.read() on hidraw fds ({listen_secs}s)")
    if not fds:
        print("  no raw fds to read.")
        return
    print("  Move both SpaceMice to see motion samples below.")
    t_end = time.time() + listen_secs
    counts = {p: 0 for p, _ in fds}
    last_print = 0
    while time.time() < t_end:
        for p, fd in fds:
            for _ in range(20):
                try:
                    data = os.read(fd, 64)
                except BlockingIOError:
                    break
                except OSError as e:
                    print(f"  {p} read failed: {e}")
                    break
                if not data:
                    break
                counts[p] += 1
                if len(data) >= 7 and data[0] == 1:
                    def s16(a, b):
                        x = a | (b << 8)
                        return x - 65536 if x >= 32768 else x
                    tx = s16(data[1], data[2])
                    ty = s16(data[3], data[4])
                    tz = s16(data[5], data[6])
                    if abs(tx) + abs(ty) + abs(tz) > 50 and time.time() - last_print > 0.05:
                        print(f"  {p}  trans=({tx:+5d},{ty:+5d},{tz:+5d})")
                        last_print = time.time()
                elif len(data) >= 2 and data[0] == 3:
                    print(f"  {p}  BUTTON byte={data[1]:#04x}")
        time.sleep(0.005)
    print(f"  reports per device: {counts}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
@click.command()
@click.option('--listen-secs', default=5.0, type=float,
              help='Seconds to listen for live motion in each backend.')
@click.option('--skip-hidapi', is_flag=True, help='Skip hidapi probes.')
@click.option('--skip-raw', is_flag=True, help='Skip raw hidraw probes.')
def main(listen_secs, skip_hidapi, skip_raw):
    banner("SpaceMouse standalone debug -- NO ROBOT MOTION")

    hidraw_paths = probe_hidraw_nodes()

    if not skip_hidapi:
        try:
            devs = probe_hidapi_enumerate()
        except Exception as e:
            print(f"  enumerate FAILED: {type(e).__name__}: {e}")
            devs = []
        handles = probe_open_by_path(devs) if devs else []
        single_handle = probe_open_by_vid_pid() if not handles else None

        if handles:
            live_read_hidapi(handles, listen_secs)
        elif single_handle is not None:
            live_read_hidapi([(0, single_handle, b'(vid+pid)')], listen_secs)
            try:
                single_handle.close()
            except Exception:
                pass
        else:
            print("\n  hidapi could open zero devices.")

        for i, h, _ in handles:
            try:
                h.close()
            except Exception:
                pass

    if not skip_raw:
        fds = probe_raw_open(hidraw_paths)
        if fds:
            live_read_raw(fds, listen_secs)
        for _, fd in fds:
            try:
                os.close(fd)
            except Exception:
                pass

    banner("done")


if __name__ == '__main__':
    main()
