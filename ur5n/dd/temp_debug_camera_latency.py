#!/usr/bin/env python3
"""
Camera latency deep-dive diagnostic.

Tests each component of the observation pipeline independently:
  1. Raw cap.read() latency (single frame)
  2. flush(n) latency for different n values
  3. grab_rgb() latency (read + YUV decode)
  4. grab_pil() latency (flush + read + decode + crop + PIL)
  5. Back-to-back grab_pil() to see if latency is consistent
  6. Buffer staleness: what happens if we wait between reads?
  7. Alternative: MJPEG vs YU12 codec comparison

Usage:
    python ur5n/dd/temp_debug_camera_latency.py
    python ur5n/dd/temp_debug_camera_latency.py --dev 0 --trials 30
    python ur5n/dd/temp_debug_camera_latency.py --width 640 --height 480
"""

import argparse
import time
import numpy as np
import cv2
from PIL import Image


def fmt_stats(times_ms):
    a = np.array(times_ms)
    return (f"mean={a.mean():.1f}ms  std={a.std():.1f}ms  "
            f"min={a.min():.1f}ms  max={a.max():.1f}ms  "
            f"median={np.median(a):.1f}ms")


def open_camera(dev, width, height, fps, codec="YU12"):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open /dev/video{dev}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
    if codec == "YU12":
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))
    actual_buf = int(cap.get(cv2.CAP_PROP_BUFFERSIZE))

    print(f"  Opened /dev/video{dev}: {actual_w}x{actual_h}@{actual_fps}fps  "
          f"fourcc={fourcc_str}  buffersize={actual_buf}")
    return cap, actual_w, actual_h


def warmup(cap, n=10):
    print(f"  Warming up ({n} frames)...")
    for _ in range(n):
        cap.read()


# ─────────────────────────────────────────────────────────────────
#  Test 1: Raw cap.read() latency
# ─────────────────────────────────────────────────────────────────

def test_raw_read(cap, trials):
    print(f"\n{'='*60}")
    print(f"TEST 1: Raw cap.read() latency  ({trials} trials)")
    print(f"{'='*60}")

    times = []
    for i in range(trials):
        t0 = time.monotonic()
        ok, frame = cap.read()
        dt = (time.monotonic() - t0) * 1000
        times.append(dt)
        if i < 5:
            shape = frame.shape if ok else "FAIL"
            print(f"  [{i}] {dt:.2f}ms  ok={ok}  shape={shape}")

    print(f"  {fmt_stats(times)}")
    return times


# ─────────────────────────────────────────────────────────────────
#  Test 2: flush(n) latency for different n
# ─────────────────────────────────────────────────────────────────

def test_flush(cap, trials):
    print(f"\n{'='*60}")
    print(f"TEST 2: flush(n) latency for n=0..8  ({trials} trials each)")
    print(f"{'='*60}")

    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        times = []
        for _ in range(trials):
            t0 = time.monotonic()
            for _ in range(n):
                cap.read()
            dt = (time.monotonic() - t0) * 1000
            times.append(dt)
        print(f"  flush({n}): {fmt_stats(times)}")


# ─────────────────────────────────────────────────────────────────
#  Test 3: grab_rgb() — read + YUV decode
# ─────────────────────────────────────────────────────────────────

def test_grab_rgb(cap, width, height, trials):
    print(f"\n{'='*60}")
    print(f"TEST 3: grab_rgb() — read + YUV decode  ({trials} trials)")
    print(f"{'='*60}")

    read_times = []
    decode_times = []
    total_times = []

    for i in range(trials):
        t0 = time.monotonic()
        ok, raw = cap.read()
        t1 = time.monotonic()
        if not ok:
            print(f"  [{i}] read FAILED")
            continue

        yuv = np.ascontiguousarray(raw).reshape(height * 3 // 2, width)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t2 = time.monotonic()

        read_ms = (t1 - t0) * 1000
        decode_ms = (t2 - t1) * 1000
        total_ms = (t2 - t0) * 1000
        read_times.append(read_ms)
        decode_times.append(decode_ms)
        total_times.append(total_ms)

    print(f"  read:   {fmt_stats(read_times)}")
    print(f"  decode: {fmt_stats(decode_times)}")
    print(f"  total:  {fmt_stats(total_times)}")


# ─────────────────────────────────────────────────────────────────
#  Test 4: Full grab_pil() pipeline — flush + read + decode + crop + PIL
# ─────────────────────────────────────────────────────────────────

def test_grab_pil(cap, width, height, trials):
    print(f"\n{'='*60}")
    print(f"TEST 4: Full grab_pil() pipeline  ({trials} trials)")
    print(f"        flush(5) + read + YUV decode + center crop + PIL")
    print(f"{'='*60}")

    flush_times = []
    read_times = []
    decode_times = []
    crop_pil_times = []
    total_times = []

    for i in range(trials):
        t_start = time.monotonic()

        # flush
        for _ in range(5):
            cap.read()
        t_after_flush = time.monotonic()

        # read
        ok, raw = cap.read()
        t_after_read = time.monotonic()
        if not ok:
            print(f"  [{i}] read FAILED")
            continue

        # decode
        yuv = np.ascontiguousarray(raw).reshape(height * 3 // 2, width)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t_after_decode = time.monotonic()

        # crop + PIL
        h, w = rgb.shape[:2]
        s = min(h, w)
        left, top = (w - s) // 2, (h - s) // 2
        pil_img = Image.fromarray(rgb[top:top + s, left:left + s])
        t_end = time.monotonic()

        flush_ms = (t_after_flush - t_start) * 1000
        read_ms = (t_after_read - t_after_flush) * 1000
        decode_ms = (t_after_decode - t_after_read) * 1000
        crop_pil_ms = (t_end - t_after_decode) * 1000
        total_ms = (t_end - t_start) * 1000

        flush_times.append(flush_ms)
        read_times.append(read_ms)
        decode_times.append(decode_ms)
        crop_pil_times.append(crop_pil_ms)
        total_times.append(total_ms)

        if i < 3:
            print(f"  [{i}] flush={flush_ms:.1f}  read={read_ms:.1f}  "
                  f"decode={decode_ms:.1f}  crop+pil={crop_pil_ms:.1f}  "
                  f"TOTAL={total_ms:.1f}ms")

    print(f"\n  flush(5):  {fmt_stats(flush_times)}")
    print(f"  read:      {fmt_stats(read_times)}")
    print(f"  decode:    {fmt_stats(decode_times)}")
    print(f"  crop+pil:  {fmt_stats(crop_pil_times)}")
    print(f"  TOTAL:     {fmt_stats(total_times)}")


# ─────────────────────────────────────────────────────────────────
#  Test 5: Back-to-back grab_pil() — consistency check
# ─────────────────────────────────────────────────────────────────

def test_back_to_back(cap, width, height, trials):
    print(f"\n{'='*60}")
    print(f"TEST 5: Back-to-back grab_pil()  ({trials} calls)")
    print(f"        Checking if latency is consistent vs first-call spike")
    print(f"{'='*60}")

    times = []
    for i in range(trials):
        t0 = time.monotonic()

        # flush
        for _ in range(5):
            cap.read()
        # read + decode + crop
        ok, raw = cap.read()
        if ok:
            yuv = np.ascontiguousarray(raw).reshape(height * 3 // 2, width)
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            s = min(h, w)
            left, top = (w - s) // 2, (h - s) // 2
            Image.fromarray(rgb[top:top + s, left:left + s])

        dt = (time.monotonic() - t0) * 1000
        times.append(dt)

    # Print as a sequence to spot patterns
    for i, t in enumerate(times):
        marker = " <<<" if t > np.median(times) * 1.5 else ""
        print(f"  [{i:2d}] {t:.1f}ms{marker}")
    print(f"\n  {fmt_stats(times)}")


# ─────────────────────────────────────────────────────────────────
#  Test 6: Buffer staleness — what happens with idle gaps?
# ─────────────────────────────────────────────────────────────────

def test_staleness(cap, width, height):
    print(f"\n{'='*60}")
    print(f"TEST 6: Buffer staleness — idle gap before read")
    print(f"        Simulates real scenario: inference takes time,")
    print(f"        then we grab a frame. Is the frame stale?")
    print(f"{'='*60}")

    for delay_ms in [0, 50, 100, 200, 400, 800]:
        # Drain buffer first
        for _ in range(10):
            cap.read()

        # Simulate inference delay
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

        # Now measure: how long to get a fresh frame?
        # Option A: single read (might be stale)
        t0 = time.monotonic()
        cap.read()
        single_ms = (time.monotonic() - t0) * 1000

        # Drain again
        for _ in range(10):
            cap.read()
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

        # Option B: flush(5) + read
        t0 = time.monotonic()
        for _ in range(5):
            cap.read()
        cap.read()
        flush_ms = (time.monotonic() - t0) * 1000

        print(f"  idle={delay_ms:4d}ms  ->  single_read={single_ms:.1f}ms  "
              f"flush5+read={flush_ms:.1f}ms")


# ─────────────────────────────────────────────────────────────────
#  Test 7: Continuous background reader thread (alternative approach)
# ─────────────────────────────────────────────────────────────────

def test_threaded_reader(cap, width, height, trials):
    import threading

    print(f"\n{'='*60}")
    print(f"TEST 7: Threaded background reader — always-fresh frame")
    print(f"        A thread reads continuously; main grabs latest.")
    print(f"        This eliminates flush() entirely.")
    print(f"{'='*60}")

    latest_frame = [None]
    frame_lock = threading.Lock()
    running = [True]
    read_count = [0]

    def reader_loop():
        while running[0]:
            ok, raw = cap.read()
            if ok:
                with frame_lock:
                    latest_frame[0] = raw
                    read_count[0] += 1

    t = threading.Thread(target=reader_loop, daemon=True)
    t.start()

    # Let the reader fill in for a moment
    time.sleep(0.2)

    grab_times = []
    for i in range(trials):
        t0 = time.monotonic()

        with frame_lock:
            raw = latest_frame[0]

        if raw is not None:
            yuv = np.ascontiguousarray(raw).reshape(height * 3 // 2, width)
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            s = min(h, w)
            left, top = (w - s) // 2, (h - s) // 2
            Image.fromarray(rgb[top:top + s, left:left + s])

        dt = (time.monotonic() - t0) * 1000
        grab_times.append(dt)

        # Simulate inference gap (200ms)
        time.sleep(0.2)

    running[0] = False
    t.join(timeout=2.0)

    print(f"  Reader thread captured {read_count[0]} frames total")
    print(f"  Grab latency (decode+crop+PIL only, no flush/read wait):")
    print(f"  {fmt_stats(grab_times)}")


# ─────────────────────────────────────────────────────────────────
#  Test 8: MJPEG codec comparison
# ─────────────────────────────────────────────────────────────────

def test_mjpeg(dev, width, height, fps, trials):
    print(f"\n{'='*60}")
    print(f"TEST 8: MJPEG codec comparison  ({trials} trials)")
    print(f"{'='*60}")

    try:
        cap_mj = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap_mj.isOpened():
            print("  SKIP: cannot open camera for MJPEG test")
            return
        cap_mj.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap_mj.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap_mj.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap_mj.set(cv2.CAP_PROP_FPS, fps)
        try:
            cap_mj.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        actual_fourcc = int(cap_mj.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))
        print(f"  Opened with fourcc={fourcc_str}")

        warmup(cap_mj, 10)

        # Single read
        read_times = []
        for _ in range(trials):
            t0 = time.monotonic()
            ok, frame = cap_mj.read()
            dt = (time.monotonic() - t0) * 1000
            read_times.append(dt)
        print(f"  Single read:    {fmt_stats(read_times)}")

        # flush(5) + read
        full_times = []
        for _ in range(trials):
            t0 = time.monotonic()
            for _ in range(5):
                cap_mj.read()
            ok, frame = cap_mj.read()
            dt = (time.monotonic() - t0) * 1000
            full_times.append(dt)
        print(f"  flush(5)+read:  {fmt_stats(full_times)}")

        cap_mj.release()
    except Exception as e:
        print(f"  SKIP: MJPEG test failed: {e}")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Camera latency diagnostic")
    parser.add_argument("--dev", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    print(f"Camera latency diagnostic")
    print(f"  dev=/dev/video{args.dev}  res={args.width}x{args.height}  "
          f"fps={args.fps}  trials={args.trials}")
    print()

    # Open with YU12 (same as closedloop_rtc.py)
    print("Opening camera (YU12 codec, matching closedloop_rtc.py)...")
    cap, actual_w, actual_h = open_camera(
        args.dev, args.width, args.height, args.fps, codec="YU12")
    warmup(cap)

    test_raw_read(cap, args.trials)
    test_flush(cap, args.trials)
    test_grab_rgb(cap, actual_w, actual_h, args.trials)
    test_grab_pil(cap, actual_w, actual_h, args.trials)
    test_back_to_back(cap, actual_w, actual_h, min(args.trials, 15))
    test_staleness(cap, actual_w, actual_h)
    test_threaded_reader(cap, actual_w, actual_h, min(args.trials, 10))

    cap.release()

    test_mjpeg(args.dev, args.width, args.height, args.fps, args.trials)

    print(f"\n{'='*60}")
    print("DONE. Key things to look for:")
    print("  - Test 1: Is single read ~33ms (30fps) or much higher?")
    print("  - Test 2: flush(n) should scale linearly. If not, V4L2 buffering issue.")
    print("  - Test 4: Which stage dominates? flush vs decode vs crop?")
    print("  - Test 5: Any outlier spikes? First-call penalty?")
    print("  - Test 6: After idle, does single read return instantly (stale frame)?")
    print("  - Test 7: Threaded reader should show <10ms grab time (just decode+crop).")
    print("  - Test 8: MJPEG may be faster if USB bandwidth is the bottleneck.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
