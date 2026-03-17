# ur5/ Scripts Setup

## Dependency

Scripts in this directory depend on the `modular_policy` package from:
```
/home/kaiwen/Desktop/research/fastumipro-collection/
```

The package is NOT copied here. It is installed in **editable mode** so imports resolve to the original source.

## Install (one-time)

```bash
pip install -e /home/kaiwen/Desktop/research/fastumipro-collection/
```

This makes `modular_policy` importable from anywhere. Edits to the source in `fastumipro-collection/modular_policy/` take effect immediately (no reinstall needed).

## Verify

```bash
python -c "import modular_policy; print(modular_policy.__file__)"
# Should print: /home/kaiwen/Desktop/research/fastumipro-collection/modular_policy/__init__.py
```

## What was changed from the original scripts

The original scripts (from `fastumipro-collection/scripts/`) used relative paths like:
```python
os.path.join(os.path.dirname(__file__), '..', 'modular_policy', ...)
```

These were replaced with package-based resolution:
```python
import modular_policy
os.path.join(os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
```

This makes the scripts location-independent.

## Scripts origin

Copied from `/home/kaiwen/Desktop/research/fastumipro-collection/scripts/`:
- `get_ee_pose.py` - Read current EE pose
- `gohome_ee.py` - Move robot to home position
- `readimage.py` - Camera image utility (no external deps)
- `replay_dataset.py` - Replay recorded demonstrations
- `robotiq_gripper.py` - Gripper control (no external deps)
