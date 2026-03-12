# 0307 - FastUMI pickandplace-real-0307 Training Setup

## What changed

Configured `realworld/0306-train-pickandplace.sh` to train on the new real-world dataset `pickandplace-real-0307` (250 episodes, 22224 frames) with 4 GPUs.

### Files modified

1. **`realworld/0306-train-pickandplace.sh`**
   - Added `DATASET_NAME` and `DATA_MIX` variables at the top (lines 24-25) for easy dataset switching
   - Default GPU count changed from auto-detect to **4**
   - Preflight check and training command now use the variables instead of hardcoded paths

2. **`starVLA/dataloader/gr00t_lerobot/mixtures.py`**
   - Added mixture entry `fastumi_pickandplace_real_0307` pointing to `pickandplace-real-0307` dataset

### How to switch datasets

Edit two variables at the top of the shell script:
```bash
DATASET_NAME=pickandplace-real-0307
DATA_MIX=fastumi_pickandplace_real_0307
```
The `DATA_MIX` value must have a matching entry in `starVLA/dataloader/gr00t_lerobot/mixtures.py`.

---

## Training configuration summary

| Parameter | Value |
|---|---|
| Script | `realworld/0306-train-pickandplace.sh` |
| Dataset | `playground/Datasets/FastUMI/pickandplace-real-0307` (250 eps, 22224 frames) |
| Pretrained model | `Qwen3-VL-4B-Instruct-Action` |
| Framework | QwenOFT (Qwen VLM + L1 regression MLP head) |
| Action/state dim | 10 (3 pos + 6 rot6d + 1 gripper) |
| GPUs | 4 |
| Per-device batch | 8 |
| Gradient accumulation | 2 |
| Effective batch size | 64 |
| Learning rate (VLM backbone) | 1e-05 |
| Learning rate (action head) | 1e-04 |
| LR scheduler | cosine with min lr |
| Max training steps | 100,000 |
| Checkpoint save interval | every 10,000 steps |
| Eval interval | every 2,000 steps |
| Log interval | every 100 steps |
| Precision | BFloat16 |
| DeepSpeed | ZeRO Stage 2 |
| Attention | flash_attention_2 (fallback: sdpa) |
| Run ID | `fastumi_pickandplace_qwenOFT` |
| W&B project | `starVLA_FastUMI` |

---

## Data pipeline verification

### Dataset structure
```
playground/Datasets/FastUMI/pickandplace-real-0307/
├── data/          # parquet files
├── meta/
│   ├── info.json          # 250 episodes, 22224 frames, fps=20
│   ├── modality.json      # key mapping (verified correct)
│   ├── episodes.jsonl
│   └── tasks.jsonl        # task: "pickandplace-vla- 0307"
└── videos/        # h264 mp4, 256x256 wrist camera
```

### Key mapping (modality.json)
The `FastUMIDataConfig` expects granular keys but the dataset stores flat arrays. The `modality.json` bridges this:

| Config key | Original key in dataset | Slice |
|---|---|---|
| `state.eef_pos` | `observation.state` | [0:3] |
| `state.eef_rot6d` | `observation.state` | [3:9] |
| `state.gripper` | `observation.state` | [9:10] |
| `action.eef_pos` | `action` | [0:3] |
| `action.eef_rot6d` | `action` | [3:9] |
| `action.gripper` | `action` | [9:10] |
| `video.wrist` | `observation.images.wrist` | full |
| `annotation.human.action.task_description` | `task_index` | full |

### Normalization stats
`stats_gr00t.json` does not exist yet. The training code auto-computes and saves it on rank 0 during the first run.

---

## Architecture overview

```
Input: [wrist image 256x256, task description]
  |
  v
Qwen3-VL-4B-Instruct-Action (VLM encoder, 2048-dim hidden)
  |
  v
Action special token embeddings [B, 16, 2048]
  |
  v
L1RegressionActionHead (MLPResNet, 2 residual blocks)
  |
  v
Predicted actions [B, 16, 10]   (16-step action horizon, 10D per step)

Loss: L1 between predicted and target actions
```

- State: 10D absolute EEF pose (pos + rot6d + gripper)
- Action: 10D relative EEF pose (pre-computed relative, treated as "abs" mode in code)
- Camera: 1x wrist (256x256), h264 video, loaded via torchvision_av backend
