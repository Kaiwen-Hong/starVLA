# QwenPI Training for FastUMI PickAndPlace

## What is QwenPI?

QwenPI is a diffusion-based VLA framework in starVLA that pairs **Qwen2.5-VL-3B** as the vision-language backbone with a **Flow-Matching (FM) action expert** for continuous action prediction. It follows the pi_0 paradigm: the VLM encodes visual observations and language instructions into rich token representations, which are then consumed by a DiT-based diffusion model that denoises action trajectories.

### QwenPI vs QwenOFT

| Aspect | QwenOFT | QwenPI |
|--------|---------|--------|
| VLM backbone | Qwen3-VL-4B | Qwen2.5-VL-3B |
| Action head | MLP / linear projection | DiT-B flow-matching (LayerwiseFlowmatchingActionHead) |
| Action prediction | Single-step regression | Iterative denoising (diffusion) |
| Inference steps | 1 | Configurable (`num_inference_timesteps`, default 4) |
| Multi-modality handling | Continuous actions via regression | Continuous actions via flow matching |

## Architecture Overview

```
 Image + Language Instruction
          |
  [Qwen2.5-VL-3B-Instruct-Action]
          |
  VLM hidden states (2048-dim)
          |
  [LayerwiseFlowmatchingActionHead]
     - DiT-B Transformer (16 layers)
     - Cross-attention to VLM features
     - Flow-matching denoising loop
          |
  Action chunk (16 steps x 10-dim)
```

- **VLM**: `Qwen2.5-VL-3B-Instruct-Action` produces 2048-dim hidden states.
- **Action expert**: A DiT-B transformer with 16 layers, 1024 hidden dim, and cross-attention (dim 2048) to the VLM features. Uses ada_norm normalization and dropout 0.2.
- **Flow matching**: Beta-distribution noise schedule (alpha=1.5, beta=1.0, s=0.999), 1000 timestep buckets during training, 4 inference steps.

## Parameter Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `action_dim` | 10 | FastUMI: 3 pos + 6 rot6d + 1 gripper |
| `state_dim` | 10 | Same as action_dim for FastUMI |
| `future_action_window_size` | 15 | Chunk size 16 (15 future + 1 current), same as QwenOFT |
| `past_action_window_size` | 0 | No action history conditioning |
| `action_model_type` | DiT-B | Base-size DiT, 16 layers |
| `action_hidden_dim` / `hidden_size` | 1024 | DiT-B default |
| `repeated_diffusion_steps` | 8 | Training diffusion steps per sample |
| `num_inference_timesteps` | 4 | Fast inference with few denoising steps |
| `num_target_vision_tokens` | 32 | Compressed vision token count for action head |
| `noise_beta_alpha/beta` | 1.5 / 1.0 | Beta distribution schedule for flow matching |

## Usage

### Training

```bash
# Default: 4 GPUs
bash realworld/0311-train-pickandplace-qwenpi.sh

# Custom GPU count
bash realworld/0311-train-pickandplace-qwenpi.sh 2
```

The script will auto-download `StarVLA/Qwen2.5-VL-3B-Instruct-Action` from HuggingFace on first run.

### Output

- Checkpoints: `./results/Checkpoints/fastumi_pickandplace_qwenPI/`
- Logs: `./results/Checkpoints/fastumi_pickandplace_qwenPI/logs/`

### Effective batch size

The script maintains an effective batch size of 64 by adjusting gradient accumulation based on GPU count:

| GPUs | per_device_batch | grad_accum | effective |
|------|-----------------|------------|-----------|
| 1 | 8 | 8 | 64 |
| 2 | 8 | 4 | 64 |
| 4 | 8 | 2 | 64 |
| 8 | 8 | 1 | 64 |

## File References

- **Training script**: `realworld/0311-train-pickandplace-qwenpi.sh`
- **QwenOFT baseline**: `realworld/0306-train-pickandplace.sh`
- **Config YAML template**: `examples/calvin/train_files/starvla_train_calvin.yaml`
- **QwenPI framework**: `starVLA/model/framework/QwenPI.py`
- **FM action head**: `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- **Data mixtures**: `starVLA/dataloader/gr00t_lerobot/mixtures.py`
