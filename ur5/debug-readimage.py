#!/usr/bin/env python3
"""
Debug version of readimage.py.

Diagnoses why /dev/videoN returns `select() timeout`. Enumerates the
camera's actual capabilities and tests several (FOURCC, resolution, FPS)
combinations to determine whether the problem is hardware or software.

Usage:
    python ur5/debug-readimage.py            # probe /dev/video0
    python ur5/debug-readimage.py --dev 2    # probe /dev/video2
"""

import argparse
import glob
import os
import shutil
import subprocess
import time

import cv2
import numpy as np


def header(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def run_cmd(cmd: str, timeout: int = 5) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=timeout)
        out = (r.stdout or "") + (r.stderr or "")
        return out.strip() or "<empty>"
    except FileNotFoundError:
        return "<command not found>"
    except subprocess.TimeoutExpired:
        return "<timed out>"
    except Exception as e:
        return f"<error: {e}>"


# ── 1. Environment / device enumeration ──────────────────────────────
def list_video_devices() -> list[str]:
    header("1. /dev/video* devices present")
    devs = sorted(glob.glob("/dev/video*"))
    if not devs:
        print("  (none — camera not detected by kernel)")
    for d in devs:
        # v4l2-ctl --info is concise; fall back to ls -l
        if shutil.which("v4l2-ctl"):
            info = run_cmd(f"v4l2-ctl -d {d} --info | head -n 8")
        else:
            info = run_cmd(f"ls -l {d}")
        print(f"\n  {d}")
        for line in info.splitlines():
            print(f"    {line}")
    return devs


def v4l2_capabilities(dev_path: str) -> None:
    header(f"2. Supported formats/resolutions/FPS for {dev_path}")
    if not shutil.which("v4l2-ctl"):
        print("  v4l2-ctl not installed — run:  sudo apt install v4l-utils")
        return
    out = run_cmd(f"v4l2-ctl -d {dev_path} --list-formats-ext")
    print(out)


def check_users(dev_path: str) -> None:
    header(f"3. Processes holding {dev_path} open")
    if shutil.which("fuser"):
        print("--- fuser ---")
        print(run_cmd(f"fuser -v {dev_path}"))
    if shutil.which("lsof"):
        print("--- lsof ---")
        print(run_cmd(f"lsof {dev_path}"))


def usb_topology() -> None:
    header("4. USB topology (lsusb -t)")
    print(run_cmd("lsusb -t"))
    print("\n--- lsusb (devices) ---")
    print(run_cmd("lsusb"))


# ── 5. Probe capture configurations ──────────────────────────────────
def try_config(dev: int, fourcc_str: str, w: int, h: int, fps: int,
               n_frames: int = 10, timeout_s: float = 4.0) -> tuple[bool, str, dict]:
    """Open the camera with a given config, try to grab n_frames within timeout."""
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return False, "VideoCapture.open() failed", {}

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    fcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc = "".join([chr((fcc_int >> (8 * i)) & 0xFF) for i in range(4)])
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    afps = cap.get(cv2.CAP_PROP_FPS)
    driver = {"fourcc": fcc, "w": aw, "h": ah, "fps": afps}

    n_ok = 0
    bytes_total = 0
    t0 = time.time()
    deadline = t0 + timeout_s
    first_frame_t = None
    while n_ok < n_frames and time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            if first_frame_t is None:
                first_frame_t = time.time()
            n_ok += 1
            try:
                bytes_total += frame.nbytes
            except Exception:
                pass
    dt = time.time() - t0
    cap.release()

    if n_ok >= n_frames:
        achieved = n_frames / max(dt, 1e-6)
        msg = (f"OK — got {n_frames} frames in {dt:.2f}s "
               f"({achieved:.1f} fps achieved, first frame at {first_frame_t - t0:.2f}s)")
        return True, msg, driver

    if n_ok == 0:
        msg = f"FAIL — no frames in {dt:.1f}s (select() timeout)"
    else:
        msg = (f"PARTIAL — only {n_ok}/{n_frames} frames in {dt:.1f}s "
               f"(first at {first_frame_t - t0:.2f}s)")
    return False, msg, driver


def probe(dev: int) -> list[tuple[tuple, bool, str, dict]]:
    header("5. Probing capture configurations")
    configs = [
        # original failing config from readimage.py
        ("YU12", 1920, 1080, 100),
        # Same format/res but progressively lower FPS
        ("YU12", 1920, 1080, 60),
        ("YU12", 1920, 1080, 30),
        ("YU12", 1920, 1080, 15),
        # MJPG (compressed) — UVC cams usually support high FPS only in MJPG
        ("MJPG", 1920, 1080, 100),
        ("MJPG", 1920, 1080, 60),
        ("MJPG", 1920, 1080, 30),
        # Lower resolutions in YU12 (less bandwidth)
        ("YU12", 1280, 720, 60),
        ("YU12", 640, 480, 30),
    ]
    results = []
    for cfg in configs:
        fcc, w, h, fps = cfg
        print(f"\n>>> {fcc} {w}x{h}@{fps}fps")
        ok, msg, drv = try_config(dev, *cfg)
        mark = "✓" if ok else "✗"
        print(f"    driver reports: FOURCC={drv.get('fourcc')} "
              f"{drv.get('w')}x{drv.get('h')}@{drv.get('fps', 0):.1f}fps")
        print(f"    {mark} {msg}")
        results.append((cfg, ok, msg, drv))
    return results


# ── 6. Diagnosis ─────────────────────────────────────────────────────
def diagnose(results: list[tuple[tuple, bool, str, dict]]) -> None:
    header("6. Summary")
    print(f"{'Config':<35} {'Result'}")
    print("-" * 70)
    for cfg, ok, msg, _ in results:
        fcc, w, h, fps = cfg
        cfg_str = f"{fcc} {w}x{h}@{fps}fps"
        status = "OK  " if ok else "FAIL"
        print(f"{cfg_str:<35} {status}  {msg}")

    header("7. Diagnosis")
    by_key = {cfg: ok for cfg, ok, _, _ in results}
    orig_ok = by_key[("YU12", 1920, 1080, 100)]
    yu12_lower_fps_ok = any(by_key[("YU12", 1920, 1080, f)] for f in (60, 30, 15))
    mjpg_any_ok = any(by_key[("MJPG", 1920, 1080, f)] for f in (100, 60, 30))
    yu12_low_res_ok = by_key[("YU12", 1280, 720, 60)] or by_key[("YU12", 640, 480, 30)]
    any_ok = any(ok for _, ok, _, _ in results)

    if orig_ok:
        print("→ The original config now works. Earlier failure may have been")
        print("  transient (another process held the camera, or the device was")
        print("  in a bad state). Re-run readimage.py.")
        return

    if not any_ok:
        print("→ HARDWARE / SYSTEM issue: every config failed.")
        print("  Likely causes (in order of likelihood):")
        print("    a) Another process is holding the camera open — see section 3.")
        print("    b) Cable / USB port problem — try a different USB3 port,")
        print("       a different cable, or another USB3 host controller.")
        print("    c) Wrong /dev/videoN — many UVC devices expose multiple nodes")
        print("       (video0/video1/etc.); only one is the streaming node.")
        print("       Check section 2 for the node with actual format support.")
        print("    d) Camera firmware crash — unplug+replug and retry.")
        return

    # Bandwidth math
    bw_yu12_100 = 1920 * 1080 * 1.5 * 100 / 1e6   # MB/s
    print(f"  Bandwidth math: YU12 1920x1080@100fps ≈ {bw_yu12_100:.0f} MB/s "
          f"(~{bw_yu12_100*8/1000:.1f} Gbps)")
    print("    USB 2.0 ceiling ≈ 60 MB/s  (480 Mbps)")
    print("    USB 3.0 ceiling ≈ 400 MB/s practical (5 Gbps theoretical)\n")

    if mjpg_any_ok and not yu12_lower_fps_ok:
        print("→ SOFTWARE / CONFIG issue: the camera does NOT support raw YU12")
        print("  at 1080p — it only streams 1080p in compressed MJPG.")
        print("  FIX: change readimage.py to use MJPG:")
        print("       cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))")
        print("       cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # let OpenCV decode")
        print("       # then cap.read() returns BGR directly; drop the I420 reshape.")
    elif yu12_lower_fps_ok and not mjpg_any_ok:
        print("→ SOFTWARE / CONFIG issue: YU12 1080p works at lower FPS — the")
        print("  100 fps request exceeds what the camera can actually deliver in")
        print("  raw format (bandwidth or sensor limit). FIX: drop FPS to the")
        print("  highest YU12 rate that succeeded above.")
    elif yu12_lower_fps_ok and mjpg_any_ok:
        print("→ SOFTWARE / CONFIG issue: 100 fps in YU12 at 1080p is unsupported.")
        print("  Two valid fixes:")
        print("    (a) keep YU12 but drop FPS to the highest rate that worked,")
        print("    (b) switch to MJPG to keep ~100 fps (compressed).")
    elif yu12_low_res_ok:
        print("→ SOFTWARE / CONFIG issue: full 1080p is unreachable on this link;")
        print("  lower resolution works. Use 1280x720 (or 640x480) in YU12, or")
        print("  switch to MJPG at 1080p.")
    else:
        print("→ Mixed result — see section 6 to pick a config that succeeded.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dev", type=int, default=0, help="/dev/videoN index")
    args = p.parse_args()
    dev_path = f"/dev/video{args.dev}"

    list_video_devices()
    v4l2_capabilities(dev_path)
    check_users(dev_path)
    usb_topology()
    results = probe(args.dev)
    diagnose(results)


if __name__ == "__main__":
    main()
