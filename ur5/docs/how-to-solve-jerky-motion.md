# How to Solve Jerky Motion in Async/RTC Mode

`run_dd.py --mode rtc` uses a continuous 100Hz servo thread with async inference, which should be gap-free. However, small but visible pauses/stutters can still occur. This document explains the three root causes and their fixes, implemented in `run_dd-v2.py`.

## Quick Start

Replace:
```bash
python ur5/scripts/run_dd.py --mode rtc --n_actions 8 --inference_delay 8
```
with:
```bash
python ur5/scripts/run_dd-v2.py --mode rtc --n_actions 8 --inference_delay 8
```

All arguments are identical. The v2 script monkey-patches `ServoRunner` and `Inferencer` with their v2 versions before calling the same `run_main()`.

---

## Problem 1: Gripper Commands Block the Servo Thread

**Severity: High — causes 200-500ms full pauses.**

In `servo.py`, the gripper move is executed inside the 100Hz servo loop:

```python
# servo.py _loop()
if grip_cmd is not None:
    self._gripper_hw.move(grip_cmd, 255, 150)  # blocks until gripper reaches target
```

`gripper_hw.move()` is a blocking call to the Robotiq gripper. A full open/close takes 200-500ms. During this time, no `servoL` commands are sent and the robot freezes.

The logs confirm this — duplicate gripper commands each block the servo thread:
```
Gripper -> CLOSE (pos=255)
Gripper -> CLOSE (pos=255)   ← redundant, blocks again
Gripper -> OPEN (pos=0)
Gripper -> OPEN (pos=0)      ← redundant
Gripper -> OPEN (pos=0)      ← redundant
```

**Fix in `servo_v2.py`:** Gripper commands run on a separate daemon thread. The servo loop only posts a flag; the gripper thread picks it up asynchronously. Deduplication ensures the same command is never executed twice in a row.

```
servo.py:     servo loop ──[blocks 300ms on gripper]──> resumes
servo_v2.py:  servo loop ──[posts flag, continues]──>   (gripper thread handles it)
```

## Problem 2: Spatial Discontinuity at Buffer Splice Points

**Severity: Medium — causes ~30-50mm snap-back stutters.**

In `inferencer.py`, new waypoints are computed starting from `servo.last_pose` (the robot's current position):

```python
# inferencer.py _loop()
start_pos = self._servo.last_pose  # current robot position
waypoints, grip_trans = compute_waypoints(start_pos, actions_to_push, ...)
self._servo.push_waypoints(start_pos, waypoints, grip_trans)
```

But the buffer still contains ~50 unconsumed poses that will move the robot forward. The new waypoints are appended after those poses, yet they start from the current position (far behind the buffer's end). This creates a spatial jump:

```
Buffer:  [..., P_future_end] → [near_P_current + δ, ...]
                                ↑ snaps backward!
```

With `buf≈52` and `max_d≈6mm/action`, the snap-back is roughly 10 actions × 5mm ≈ 30-50mm.

**Fix in `inferencer_v2.py`:** Use `servo.tail_pose` (the last pose in the buffer) instead of `servo.last_pose`:

```python
# inferencer_v2.py _loop()
start_pos = self._servo.tail_pose  # end of buffer, not current position
```

**Fix in `servo_v2.py`:** New `tail_pose` property reads the last element of the buffer under lock:

```python
@property
def tail_pose(self):
    with self._lock:
        if self._buffer:
            return np.array(self._buffer[-1], dtype=np.float64)
    return self._last_pose.copy()
```

This guarantees spatial continuity — new waypoints start exactly where the existing buffer ends.

## Problem 3: Timing Burst After Delays

**Severity: Low — causes brief jitter after any hiccup.**

The servo loop advances its deadline with `t_next += servo_dt`. If an iteration takes longer than 10ms (e.g., GIL contention, OS scheduling), `t_next` falls behind. The loop then tries to "catch up" by firing `servoL` commands as fast as possible until it closes the gap, causing a burst of rapid-fire commands and jerky motion.

```
Normal:    servoL ... 10ms ... servoL ... 10ms ... servoL
After lag: servoL servoL servoL servoL servoL ... 10ms ... servoL   (burst)
```

**Fix in `servo_v2.py`:** Reset `t_next` if it falls more than one tick behind:

```python
now = time.monotonic()
if t_next < now - servo_dt:
    t_next = now + servo_dt
```

This caps catch-up to at most 1 tick (10ms), preventing burst-fire.

---

## Files

| File | Role |
|------|------|
| `scripts/servo_v2.py` | `ServoRunner` with fixes 1 + 2 + 3 |
| `scripts/inferencer_v2.py` | `Inferencer` with fix 2 |
| `scripts/run_dd-v2.py` | Entry point, patches v2 classes into `run_main()` |

No changes to `robo_utils.py`, `servo.py`, or `inferencer.py` — the originals are untouched.
