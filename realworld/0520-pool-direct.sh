#!/usr/bin/env bash
############# 0520 pool-pocket-wall DIRECT #############
# Mirrors 0520-dynamic-air-hockey-static-openpi.sh — same QwenPI recipe, swapped to the
# pool "direct" task (single wrist cam, FastUMI, 10D EEF + rot6d, action chunk 16).
#   task: "Strike the white ball into the black ball to pocket it"  (50 demos, 20fps)
#
# Default = all 8 H100. If GPUs 0,1 are busy, run on the 6 free ones instead:
#   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NUM_GPUS=6 bash realworld/0520-pool-direct.sh
NUM_GPUS=${NUM_GPUS:-8}

# Run after `conda activate starVLA`.
# DeepSpeed needs a CUDA toolkit or it raises MissingCUDAException; the starVLA env ships nvcc 12.4.
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
# base VLM is local — avoid HF network calls / update checks
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --framework.action_model.action_dim 10 \
  --framework.action_model.state_dim 10 \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.past_action_window_size 0 \
  --framework.action_model.action_hidden_dim 1024 \
  --framework.action_model.hidden_size 1024 \
  --framework.action_model.action_model_type DiT-B \
  --framework.action_model.add_pos_embed True \
  --framework.action_model.max_seq_len 1024 \
  --framework.action_model.noise_beta_alpha 1.5 \
  --framework.action_model.noise_beta_beta 1.0 \
  --framework.action_model.noise_s 0.999 \
  --framework.action_model.num_timestep_buckets 1000 \
  --framework.action_model.num_inference_timesteps 4 \
  --framework.action_model.num_target_vision_tokens 32 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix pool_pocket_wall_direct \
  --datasets.vla_data.include_state false \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 10000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_pool_direct_qwenPI_0520 \
  --wandb_project starVLA_pool_direct_0520 \
  --wandb_entity kaiwenh-17-uiuc
