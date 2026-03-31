# 0330 What Done: v40/v42 OFT Finetune Scripts

## Goal

Create finetune versions of v40/v42 training pipelines, switching from **QwenPI + 16D EE-space + Qwen2.5-VL-3B** to **QwenOFT + 14D joint-space + Qwen3-VL-4B**, finetuning from `Qwen3-VL-OFT-Robotwin2` checkpoint.

## Key Differences from EE Scripts

| | EE (existing) | Joint-space finetune (new) |
|---|---|---|
| Framework | QwenPI (flow-matching DiT) | QwenOFT (L1 MLP regression) |
| VLM | Qwen2.5-VL-3B-Instruct-Action | Qwen3-VL-4B-Instruct |
| action_dim / state_dim | 16 | 14 |
| Data dir | playground/Datasets/CustomEE | playground/Datasets/Custom |
| robot_type | robotwin_ee | robotwin |
| Pretrained checkpoint | none | Qwen3-VL-OFT-Robotwin2/steps_40000 |
| QwenPI params (DiT, noise, etc.) | yes | removed |

## Files Changed

### Modified
- **`starVLA/dataloader/gr00t_lerobot/mixtures.py`**
  - Added `custom_v0320_v40`: 4 tasks, 14D, `"robotwin"`
  - Added `custom_v0320_v42`: 3 tasks, 14D, `"robotwin"`

### Created
- **`scripts/split_custom_v0320_v40.sh`** — splits `custom_v0320_v40_repo` → `playground/Datasets/Custom/`, 4 tasks, no modality file, verifies 14D
- **`scripts/split_custom_v0320_v42.sh`** — same but 3 tasks (750 episodes)
- **`r-preference/0320-v40-training-finetune-version1.sh`** — QwenOFT finetune, 4 variants, run_id `v0320_v40_qwenOFT_finetune_v1`
- **`r-preference/0320-v42-training-finetune-version1.sh`** — QwenOFT finetune, 3 variants, run_id `v0320_v42_qwenOFT_finetune_v1`

## Prerequisites (in ar-research-kempner)

Need to generate joint-space LeRobot repos from raw HDF5:
- `$LAB_ROOT/.cache/huggingface/lerobot/custom_v0320_v40_repo/`
- `$LAB_ROOT/.cache/huggingface/lerobot/custom_v0320_v42_repo/`

## Verification Checklist

1. `playground/Pretrained_models/Qwen3-VL-4B-Instruct` exists
2. `checkpoints/Qwen3-VL-OFT-Robotwin2/checkpoints/steps_40000_pytorch_model.pt` exists
3. Run split scripts after data is ready, check `playground/Datasets/Custom/` has 14D dirs
4. Sanity check: `--trainer.max_train_steps 20` to confirm no crash
