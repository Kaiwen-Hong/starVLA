# Fix: Clamp Near-Constant rot6d Dims R00/R11 (2026-03-13)

## Quick Fix Reference

```
starVLA/dataloader/gr00t_lerobot/transform/state_action.py   (core logic)
starVLA/dataloader/gr00t_lerobot/data_config.py              (config)
starVLA/dataloader/gr00t_lerobot/datasets.py                 (stats saving)
```

Added `near_constant_threshold` parameter to `StateActionTransform`.
When a dimension's `max - min < threshold`, its statistics are clamped
to `min = max = mean`, causing the Normalizer to output 0 (forward)
and the constant value (inverse). No changes to normalization logic itself.

FastUMIDataConfig sets `near_constant_threshold=1e-3` on both state and
action transforms.

**Alternative:** See `0314-no-normalization-on-rotation.md` for applying
no normalization on rotation instead.

---

## Background

The 0312 investigation (`realworld/0312/` scripts 01-07) identified that
action dims R00 (idx 3) and R11 (idx 7) are near-constant diagonal
entries of a near-identity relative rotation matrix:

```
10D action: [dx, dy, dz, R00, R01, R02, R10, R11, R12, gripper]
             0    1   2    3    4    5    6    7    8      9
```

| dim | mean     | std      | min      | max      | range    |
|-----|----------|----------|----------|----------|----------|
| R00 | 0.999998 | 0.000052 | 0.999271 | 1.000000 | 0.000729 |
| R11 | 0.999999 | 0.000047 | 0.999451 | 1.000000 | 0.000549 |

All other action dims have ranges > 0.018 (next smallest: R12 = 0.041).

### Why this causes loss oscillation

Under `min_max` normalization, the tiny range (~0.0007) gets stretched
to [-1, 1], amplifying floating-point noise ~2700-3600x. The "signal"
in R00/R11 is pure noise. Different episodes have different noise
amplitudes, so:

- Batches with high-noise episodes → large MSE on R00/R11 → loss spike
- Batches with low-noise episodes → small MSE → loss dip
- Result: oscillating training loss

### Why mean_std didn't work (0312 Fix 1, reverted)

A previous attempt changed rot6d normalization to `mean_std`. This was
reverted because:

1. `mean_std` amplifies even more (1/std = 1/0.00005 = 20000x)
2. `mean_std` output is unbounded — can produce large values when data
   has outliers or offset, which is problematic for some architectures
3. While all 6 rot6d dims get unit variance by definition, R00/R11 still
   contain no real signal — their "signal" IS the noise

---

## Root Cause

The problem is not which normalization mode to use — it's that R00 and
R11 are fundamentally uninformative dimensions. For near-identity
relative rotations:

```
R = [[R00, R01, R02],    ≈  [[1,  0,  0],
     [R10, R11, R12],        [0,  1,  0],
     [R20, R21, R22]]        [0,  0,  1]]
```

R00 ≈ 1.0 and R11 ≈ 1.0 always, with only floating-point-level
variation. Any normalization that tries to make these "useful" just
amplifies noise.

---

## Fix: `near_constant_threshold`

### Key insight

The existing `Normalizer` already handles `min == max` correctly:

```python
# forward (line 161-174 of state_action.py):
mask = min != max
normalized[..., ~mask] = 0        # constant dims → always 0

# inverse (line 206-209):
return (x + 1) / 2 * (max - min) + min   # when max==min: → min (constant)
```

So if we force `min = max = mean` for near-constant dims in the
statistics, everything works with zero changes to normalization logic.

### Change 1: `state_action.py` — StateActionTransform

Added `near_constant_threshold` field (default 0.0, backward compatible).

In `set_metadata()`, after loading raw statistics from the dataset and
before creating Normalizer objects, dims with `max - min < threshold`
get clamped: `min[i] = max[i] = mean[i]`.

```python
near_constant_threshold: float = Field(
    default=0.0,
    description="Dims with (max - min) < threshold treated as constant.",
)
```

Clamping also applies to `q01`/`q99` if present (for q99 mode).

### Change 2: `data_config.py` — FastUMIDataConfig

Both action and state `StateActionTransform` now pass
`near_constant_threshold=1e-3`:

```python
StateActionTransform(
    apply_to=self.action_keys,
    near_constant_threshold=1e-3,
    normalization_modes={
        "action.eef_pos": "min_max",
        "action.eef_rot6d": "min_max",
        "action.gripper": "binary",
    },
),
```

Threshold 1e-3 cleanly separates:
- R00 range=0.000729 < 0.001 → clamped
- R11 range=0.000549 < 0.001 → clamped
- R12 range=0.041 > 0.001 → NOT clamped (smallest meaningful dim)

State rot6d is unaffected (state R00 range=0.65, R11 range=0.11, both
far above threshold).

### Change 3: `datasets.py` — Save clamped stats

Added two helpers:
- `_extract_near_constant_threshold(transforms)` — walks transform chain
- `_clamp_near_constant_dims(combined_stats, threshold)` — applies clamping

Both `_save_dataset_statistics_()` and `save_dataset_statistics()` now
apply the same clamping before writing `dataset_statistics.json`, so
inference denormalization uses consistent (clamped) statistics.

---

## Effect

After this fix, in normalized action space:

| dim  | norm mode | behavior                              |
|------|-----------|---------------------------------------|
| dx   | min_max   | unchanged, maps to [-1, 1]            |
| dy   | min_max   | unchanged                             |
| dz   | min_max   | unchanged                             |
| R00  | min_max   | **always 0** (min==max, trivial)      |
| R01  | min_max   | unchanged                             |
| R02  | min_max   | unchanged                             |
| R10  | min_max   | unchanged                             |
| R11  | min_max   | **always 0** (min==max, trivial)      |
| R12  | min_max   | unchanged                             |
| grip | binary    | unchanged                             |

- Model learns to predict 0 for R00/R11 trivially (near-zero loss)
- No cross-episode variance from these dims → no loss oscillation
- Denormalization maps any prediction back to ~1.0 (the constant)
- All other dims unaffected

---

## Files Changed

| File | Change |
|------|--------|
| `starVLA/dataloader/gr00t_lerobot/transform/state_action.py` | Added `near_constant_threshold` field to `StateActionTransform`; clamping logic in `set_metadata()` |
| `starVLA/dataloader/gr00t_lerobot/data_config.py` | Set `near_constant_threshold=1e-3` in `FastUMIDataConfig.transform()`; updated comment |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | Added `_extract_near_constant_threshold()`, `_clamp_near_constant_dims()`; applied in both save functions |

No changes needed to denormalization code (`base_framework.py`,
`model2fastumi_interface.py`) — the `min == max` case already produces
the correct constant output.

## Action Required

1. Delete old checkpoints (their `dataset_statistics.json` has unclamped stats)
2. Retrain from scratch — config changes are picked up automatically
3. Verify with `python realworld/0312/05_pipeline_normalization.py`
   — R00/R11 should show constant 0 in normalized space
