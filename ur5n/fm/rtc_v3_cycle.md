# RTC v3 Inference-Execution Cycle

## Parameters

- `chunk_len` = 16 (total actions per model output)
- `n_actions` = `execution_horizon` = 8 (actions pushed to servo per cycle)
- `inference_delay` = 4 (fixed prefix length)
- `prev_len` = `chunk_len - n_actions` = 8 (prev_action_chunk size)

## Model Interface: `predict_action_realtime`

**Input**: `prev_action_chunk` of size `prev_len` = 8 actions:

| Index range | Count | Role |
|---|---|---|
| `[0 : inference_delay]` = `[0:4]` | 4 | Hard-fixed (copied verbatim to output) |
| `[inference_delay : prev_len]` = `[4:8]` | 4 | Soft guidance (may or may not influence output) |

**Output**: `chunk_len` = 16 actions:

| Index range | Count | Source |
|---|---|---|
| `[0 : inference_delay]` = `[0:4]` | 4 | Fixed from `prev[0:4]` |
| `[inference_delay : prev_len]` = `[4:8]` | 4 | Guided by `prev[4:8]` |
| `[prev_len : chunk_len]` = `[8:16]` | 8 | Fully generated from masked tokens |

## Init (no prefix)

1. `predict_action` (standard, no prefix) -> 16 actions
2. Push `output[0 : n_actions]` = `[0:8]` to servo
3. `prev_action_chunk` <- `output[n_actions:]` = `output[8:16]` (8 actions)

## Steady-State Cycle (uniform, including first cycle)

```
Servo buffer: |====consumed====|==remaining==|
              exec_start        trigger_t     exec_done
              |--- (n-d) ------|--- d --------|
```

1. **Trigger**: when servo has only `inference_delay` actions remaining
   - `trigger_t = exec_start + (n_actions - inference_delay)`
   - At this point, `prev_action_chunk[0:inference_delay]` = the `inference_delay` actions still in servo
2. **Capture**: grab camera image
3. **Infer**: `predict_action_realtime(prev_action_chunk)` -> 16 new actions
4. **Push**: `output[inference_delay : inference_delay + n_actions]` = `[4:12]` to servo
   - Standard: push after servo finishes current window (buffer empty)
   - Optimization: if inference finishes early, push immediately (appends seamlessly)
5. **Update prev**: `prev_action_chunk` <- `output[n_actions:]` = `output[8:16]`
6. **Advance**: `exec_start += n_actions`

## Why `output[n_actions:]` is the correct next `prev_action_chunk`

```
output:  [0:4]  [4:8]  [8:12]  [12:16]
          fixed  guided |--pushed [4:12]--|  
                        |--next prev [8:16]--|
```

- `output[n_actions : n_actions + inference_delay]` = `output[8:12]`
  - Last `inference_delay` actions of the pushed window
  - Still in servo buffer when next inference triggers -> becomes **fixed prefix**
- `output[n_actions + inference_delay :]` = `output[12:16]`
  - Actions beyond the pushed window -> becomes **soft guidance**

## First Cycle Walkthrough

```
Init:  predict_action -> output[0:16]
       Push output[0:8] to servo, exec_start=0
       prev = output[8:16]

Cycle 1:
  trigger_t = 0 + (8-4) = 4   (when 4 of 8 consumed, 4 remain)
  infer with prev = output[8:16]
  -> new_output[0:16]
  push new_output[4:12]
  prev = new_output[8:16]
  exec_start = 8

Cycle 2:
  trigger_t = 8 + (8-4) = 12
  infer with prev = new_output[8:16]
  -> ...same pattern...
```

No special first-cycle handling needed. The cycle is uniform.
