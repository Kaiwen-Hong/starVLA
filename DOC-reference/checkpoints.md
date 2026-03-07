# Checkpoint Summary

> Updated: 2026-02-22. Lists the four runs that have actual model weights saved.

---

## Overview

| Run ID | Dataset | Total Steps Saved | Latest Checkpoint | # .pt files |
|--------|---------|-------------------|-------------------|-------------|
| `robotwin_qwenOFT_4xH100` | RoboTwin (50 tasks, 25K ep) | 100K (complete) | `steps_100000` | 10 |
| `custom_qwenOFT_requeue` | Custom all (33 tasks, 3.3K ep) | 100K / 500K target | `steps_100000` | 2 |
| `custom_v0218_qwenOFT_h100-0` | Custom v0218 (27 tasks, 2.7K ep) | 400K / 500K target | `steps_400000` | 8 |
| `custom_v0218_qwenOFT_h100` | Custom v0218 (27 tasks, 2.7K ep) | 450K / 500K target | `steps_450000` | 9 |

Each `_pytorch_model.pt` file is ~9.2 GB.

---

## 1. `robotwin_qwenOFT_4xH100`

| Item | Value |
|------|-------|
| Dataset | RoboTwin-Randomized, 50 tasks, 25,000 episodes |
| Data mix | `robotwin` |
| Image format | MP4 video (AV1), 15 FPS |
| GPU | 4× H100 80GB |
| Target steps | 100,000 (complete) |
| Effective batch | 4 GPU × 8 × ? accum |
| Checkpoints | steps_10000 … steps_100000 (every 10K) |
| Path | `results/Checkpoints/robotwin_qwenOFT_4xH100/` |

---

## 2. `custom_qwenOFT_requeue`

| Item | Value |
|------|-------|
| Dataset | Custom all, 33 task variants, 3,300 episodes |
| Data mix | `custom_all` |
| Image format | Image-in-parquet, 50 FPS |
| GPU | 4× H200 80GB (kempner_requeue) |
| Target steps | 500,000 |
| Steps saved | 100,000 (training was not completed) |
| Checkpoints | steps_50000, steps_100000 |
| Path | `results/Checkpoints/custom_qwenOFT_requeue/` |

---

## 3. `custom_v0218_qwenOFT_h100-0`  *(old run)*

| Item | Value |
|------|-------|
| Dataset | Custom v0218, 27 task variants, 2,700 episodes |
| Data mix | `custom_v0218` |
| Image format | Image-in-parquet, 50 FPS |
| GPU | 4× H100 80GB (kempner_h100) |
| Target steps | 500,000 |
| Steps saved | 400,000 |
| Checkpoints | steps_50000 … steps_400000 (every 50K) |
| Path | `results/Checkpoints/custom_v0218_qwenOFT_h100-0/` |
| Note | Old run (run_id ends in `-0`). Superseded by the run below. |

---

## 4. `custom_v0218_qwenOFT_h100`  *(latest)*

| Item | Value |
|------|-------|
| Dataset | Custom v0218, 27 task variants, 2,700 episodes |
| Data mix | `custom_v0218` |
| Image format | Image-in-parquet, 50 FPS |
| GPU | 4× H100 80GB (kempner_h100) |
| Target steps | 500,000 |
| Steps saved | 450,000 |
| Checkpoints | steps_50000 … steps_450000 (every 50K) |
| Path | `results/Checkpoints/custom_v0218_qwenOFT_h100/` |
| Note | Current primary v0218 run. Still 50K steps from target. |

---

## Checkpoint file structure (per run)

```
results/Checkpoints/<run_id>/
├── config.yaml                           # full training config
├── dataset_statistics.json               # action/state normalization stats
├── summary.jsonl                         # one line per saved checkpoint
└── checkpoints/
    ├── steps_<N>/                        # full training state (resume only)
    │   └── (model shards, optimizer, scheduler, RNG states — DO NOT delete)
    └── steps_<N>_pytorch_model.pt        # standalone model weights (~9.2 GB each)
```

For inference/evaluation: use `steps_<N>_pytorch_model.pt`.
For resuming training: use `steps_<N>/` directories (via `accelerator.load_state()`).
