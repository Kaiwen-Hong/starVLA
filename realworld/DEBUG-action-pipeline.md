# Debugging Action Values in QwenPI Training

Use these breakpoint locations to trace raw data and preprocessing before the next batch.

## Data flow

1. Parquet files → `get_state_or_action` (raw numpy)
2. `get_step_data` (dict of modalities)
3. `StateActionToTensor` (numpy → torch)
4. `StateActionTransform` (normalization)
5. `_pack_sample` → batch format
6. Training loop receives batch

## Breakpoint locations

### 1. Raw actions from parquet (before any transform)

**File:** `starVLA/dataloader/gr00t_lerobot/datasets.py`  
**Location:** End of `get_state_or_action`, around line ~1602

```python
# Breakpoint here – inspect the return value
return self.retrieve_data_and_pad(...)
```

Or inside `get_state_or_action`, right after:

```python
data_array = data_array[:, le_indices]  # raw action from parquet
```

**Inspect:** `data_array` – raw action values from disk.

---

### 2. Raw data before transforms (all modalities)

**File:** `starVLA/dataloader/gr00t_lerobot/datasets.py`  
**Location:** `__getitem__`, line ~1251–1252

```python
raw_data = self.get_step_data(trajectory_id, base_index)  # breakpoint AFTER this
data = self.transforms(raw_data)
```

**Inspect:** `raw_data` – keys like `action.eef_pos`, `action.eef_rot6d`, `action.gripper` (numpy arrays, raw values).

---

### 3. Before/after normalization

**File:** `starVLA/dataloader/gr00t_lerobot/transform/state_action.py`  
**Location:** `StateActionTransform.apply`, lines ~399–404

```python
# Breakpoint BEFORE this line – see raw (pre-normalization) values
state = self._normalizers[key].forward(state)  # breakpoint here
data[key] = state
```

**Inspect:**
- `key` – e.g. `action.eef_pos`, `action.eef_rot6d`, `action.gripper`
- `state` before `forward` – pre-normalization
- `state` after `forward` – post-normalization

Or inside `Normalizer.forward` (line ~108):

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # breakpoint here – x is pre-normalization, return is post
```

**Note:** `mean_std` does **not** clip; outliers stay large after normalization. `q99` and `min_max` clip to `[-1, 1]`.

---

### 4. Batch received by training loop

**File:** `starVLA/training/train_starvla.py`  
**Location:** Line ~284

```python
batch_vla = self._get_next_batch()  # breakpoint AFTER this
```

**Inspect:** `batch_vla` – list of sample dicts with `action`, `image`, `instruction`, etc.

---

## Quick reference (line numbers may shift)

| Purpose                | File                    | Function/line      |
|------------------------|-------------------------|--------------------|
| Raw actions            | `datasets.py`           | `get_state_or_action` ~1602 |
| Pre-transform dict     | `datasets.py`           | `__getitem__` ~1252 |
| Normalization in/out   | `state_action.py`       | `StateActionTransform.apply` ~399 |
| Batch in training      | `train_starvla.py`      | ~284               |

## VS Code launch.json

For attach-style debugging, add a config that runs training and attaches to the process, then set breakpoints at the above spots.
