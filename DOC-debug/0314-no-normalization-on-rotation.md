# Fix: No Normalization on Rotation (rot6d R00/R11) (2026-03-14)

## Quick Fix Reference

```
starVLA/dataloader/gr00t_lerobot/transform/state_action.py   (core logic)
starVLA/dataloader/gr00t_lerobot/data_config.py              (config)
```

**Fix:** apply **no normalization** on `eef_rot6d`. Set `"action.eef_rot6d"` and
`"state.eef_rot6d"` to `"none"` in `normalization_modes`. This avoids amplifying
noise in near-constant dims (R00, R11) and keeps rotation values in their
natural range.

See also: `0313-fix-dataset-issue.md` for background and the alternative
`near_constant_threshold` approach.

---

## Background

The 0312 investigation identified that action dims R00 and R11 in near-identity
relative rotations have tiny ranges (~0.0007). Any min_max or mean_std
normalization amplifies floating-point noise ~2700-3600x, causing training loss
oscillation.

**Solution:** Do not normalize rotation at all.

---

## Fix: No normalization on rotation

### Key insight

Keep `eef_rot6d` un-normalized. Rotation values stay in their natural range;
no statistics or scaling needed.

### Change 1: `state_action.py` — add `"none"` mode (if needed)

If `Normalizer` does not yet support `"none"`, add a pass-through mode:

```python
valid_modes = ["q99", "mean_std", "min_max", "binary", "none"]
# none: forward(x) = x, inverse(x) = x
```

### Change 2: `data_config.py` — FastUMIDataConfig

Set `eef_rot6d` to `"none"` for both state and action:

```python
StateActionTransform(
    apply_to=self.action_keys,
    normalization_modes={
        "action.eef_pos": "min_max",
        "action.eef_rot6d": "none",
        "action.gripper": "binary",
    },
),
# and for state:
    "state.eef_rot6d": "none",
```

Position and gripper keep `min_max`; only rotation is un-normalized.

---

## Effect

| dim      | norm mode | behavior                          |
|----------|-----------|-----------------------------------|
| dx,dy,dz | min_max   | maps to [-1, 1]                   |
| R00–R12  | **none**  | **pass-through** (no scaling)     |
| grip     | binary    | unchanged                         |

- Rotation values stay in natural range (~[-1, 1] for rot6d)
- No amplification of floating-point noise in R00/R11
- No loss oscillation from near-constant dims

---

## Implementation Summary

### state_action.py — Normalizer

- Added `"none"` to `valid_modes`.
- **forward()**: `mode == "none"` → `return x.clone()` (pass-through).
- **inverse()**: `mode == "none"` → `return x.clone()` (pass-through).
- **validate_normalization_statistics()**: `mode == "none"` → skip stats checks.
- **set_metadata()**: For `mode == "none"`, use `statistics={}` when creating `Normalizer`; allow `"none"` alongside `"binary"` for non-continuous keys.

### data_config.py — FastUMIDataConfig

- Both state and action `StateActionTransform`: `"state.eef_rot6d": "none"`, `"action.eef_rot6d": "none"`.
- Removed `near_constant_threshold` (no longer used).
- `ACTION_NORM_MODES`: `["min_max"]*3 + ["none"]*6 + ["binary"]`.

### base_framework.py — unnormalize_actions

- Added explicit `elif mode == "none": pass` in per-dim denormalization loop.
- Docstring updated to include `"none"` in supported `norm_modes`.

### model2fastumi_interface.py — unnormalize_fastumi_actions

- Added `elif mode == "none": pass` for pass-through during inference.

### datasets.py

- No changes. `_extract_action_norm_modes()` reads `normalization_modes` from transforms and will include `"none"` for eef_rot6d dims automatically.

---

## Files Changed

| File | Change |
|------|--------|
| `starVLA/dataloader/gr00t_lerobot/transform/state_action.py` | Add `"none"` mode to `Normalizer`; handle in validator and set_metadata |
| `starVLA/dataloader/gr00t_lerobot/data_config.py` | Set eef_rot6d → `"none"`, remove near_constant_threshold, update ACTION_NORM_MODES |
| `starVLA/model/framework/base_framework.py` | Add `"none"` handling in `unnormalize_actions()` |
| `examples/FastUMI/eval_files/model2fastumi_interface.py` | Add `"none"` handling in `unnormalize_fastumi_actions()` |

No changes needed to `datasets.py`.

---

## Action Required

1. Delete old checkpoints (their `dataset_statistics.json` assumes min_max for rot6d)
2. Retrain from scratch
