#!/usr/bin/env bash
############# QwenPI training-time RTC finetune (both pick_to_moved + pick_from_moved) #############
#
# Each block:
#   - loads weights from the existing QwenPI 30k-step checkpoint via
#     --trainer.pretrained_checkpoint (NOT --trainer.is_resume's auto-pick)
#   - writes to a fresh run_id "<original>_rtc_ft" so the original
#     checkpoint directory is left untouched
#   - sets --framework.action_model.simulated_delay 4 to activate the
#     training-time RTC loss (see chenserver-files/ref_codes/IMPLEMENTATION.md)
#   - keeps --trainer.is_resume true so if the run is interrupted, the
#     re-launch picks up from the new _rtc_ft dir instead of restarting
#     from the source checkpoint (the trainer prefers existing checkpoints
#     in output_dir over pretrained_checkpoint when both are present).
#
# Skip broken GPU 3 on tams02.

set -e
export CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7
echo "[rtc-ft] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

CKPT_ROOT=/scratch/wangpc/starVLA/results/Checkpoints

############# Block 1: pick_to_moved #############
PRETRAINED_PT_0=$CKPT_ROOT/fastumi_pickandplace_qwenPI_0403_0_pick_to_moved/checkpoints/steps_30000_pytorch_model.pt

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 7 \
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
  --framework.action_model.simulated_delay 4 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix dynamic-0403-0-pick_to_moved \
  --datasets.vla_data.include_state false \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.pretrained_checkpoint $PRETRAINED_PT_0 \
  --trainer.is_resume true \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_pickandplace_qwenPI_0403_0_pick_to_moved_rtc_ft \
  --wandb_project starVLA_FastUMI_dynamic_0403_0_pick_to_moved \
  --wandb_entity 2200011093-peking-university \
  --hf_repo_id outsider86/DiscreteRTC


############# Block 2: pick_from_moved #############
PRETRAINED_PT_1=$CKPT_ROOT/fastumi_pickandplace_qwenPI_0403_1_pick_from_moved/checkpoints/steps_30000_pytorch_model.pt

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 7 \
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
  --framework.action_model.simulated_delay 4 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix dynamic-0403-1-pick_from_moved \
  --datasets.vla_data.include_state false \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.pretrained_checkpoint $PRETRAINED_PT_1 \
  --trainer.is_resume true \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_pickandplace_qwenPI_0403_1_pick_from_moved_rtc_ft \
  --wandb_project starVLA_FastUMI_dynamic_0403_1_pick_from_moved \
  --wandb_entity 2200011093-peking-university \
  --hf_repo_id outsider86/DiscreteRTC
