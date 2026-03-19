# What's New in Step 9 over Step 8

Step 9 (`step9-dd-closed-loop-smooth.py`) builds on Step 8 (`step8-dd-closed-loop-no_vis.py`) with three improvements aimed at smoother robot motion.

## 1. Linear Interpolation: 20Hz → 100Hz Servo

**Problem:** In Step 8, the model predicts N waypoints executed at 20Hz (one every 50ms). The robot "jumps" between waypoints, causing jerky motion.

**Solution:** Step 9 linearly interpolates between each pair of adjacent waypoints, inserting 4 intermediate points (configurable via `INTERP_MULT=5`). This increases the servo command rate from 20Hz to 100Hz (one every 10ms).

```
Step 8:  A ---------> B ---------> C        (20Hz, large steps)
Step 9:  A -> · -> · -> · -> · -> B -> ...  (100Hz, small steps)
```

Core logic (`interpolate_waypoints`):

```python
for j in range(1, mult + 1):            # mult=5, j=1,2,3,4,5
    alpha = j / mult                     # 0.2, 0.4, 0.6, 0.8, 1.0
    interp = prev + alpha * (wp - prev)  # linear interpolation
```

The robot glides smoothly along straight-line segments instead of jumping to each target.

## 2. Z-Floor Clamp Instead of Emergency Stop

**Step 8 behavior:** If a target pose's z drops below `Z_MIN_WORLD`, the program raises a `RuntimeError` and halts entirely. Safe but disruptive — requires a full restart.

**Step 9 behavior:** The z value is clamped to `Z_MIN_WORLD` and execution continues:

```python
if pos[2] < Z_MIN_WORLD:
    pos[2] = Z_MIN_WORLD  # clamp, don't stop
```

This lets the end-effector "slide along the table surface" rather than crashing the program. For pick tasks where the model occasionally predicts slightly-too-low targets, clamping is more practical than aborting.

Note: the Z threshold is also slightly more conservative in Step 9 (`-0.02` offset vs `-0.03` in Step 8).

## 3. Tuned `servoL` Parameters

`servoL(target, 0, 0, dt, lookahead_time, gain)` is UR's real-time servo command. Two key parameters were adjusted:

| Parameter | Step 8 | Step 9 | Effect |
|-----------|--------|--------|--------|
| `lookahead_time` | 0.1s | 0.2s | Looks further ahead, smoother transitions |
| `gain` | 300 | 200 | Softer tracking, less jitter |

- **`lookahead_time`** (seconds): How far ahead the robot plans its trajectory. Larger values produce smoother motion at the cost of slightly delayed response.
- **`gain`**: Proportional gain controlling how aggressively the robot tracks the target. Higher values = tighter tracking but more vibration; lower values = softer, more fluid motion.

Step 8's high-gain, short-lookahead settings prioritize precise tracking, but at 20Hz input the large gaps between targets cause jerky motion. Step 9 pairs the denser 100Hz interpolated targets with softer servo parameters for overall smoother behavior.

## Summary

All three changes work together:

1. **Interpolation** makes target points dense (100Hz)
2. **Z-clamp** prevents abrupt shutdowns on minor prediction errors
3. **Softer servoL tuning** lets the robot move fluidly through the dense waypoints
