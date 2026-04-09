# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025]. 

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Qwen2.5 vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states


Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
  or /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md （adpat a little code)
  
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# Image token ID for Qwen2.5/Qwen3 VL
_IMAGE_TOKEN_ID = 151655

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.training.trainer_utils.trainer_tools import resize_images

@FRAMEWORK_REGISTRY.register("QwenOFT")
class Qwenvl_OFT(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        config.framework.action_model.action_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        # self.hidden_dim = config.framework.action_model.action_hidden_dim
        
        self.action_token = "🔍" # TODO also can add spacail token to Qwen, but too complex
        self.action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]

        # L1 损失
        self.l1_loss = nn.L1Loss()

        # Attention capture (controlled by env var)
        self._save_attention = os.environ.get("STARVLA_SAVE_ATTENTION", "0") == "1"
        if self._save_attention:
            self._attn_output_dir = os.environ.get(
                "STARVLA_ATTENTION_DIR",
                os.path.join("results", "attention_maps", "default"),
            )
            self._attn_num_layers = int(os.environ.get("STARVLA_ATTENTION_LAYERS", "4"))
            self._attn_save_interval = int(os.environ.get("STARVLA_ATTENTION_INTERVAL", "1"))
            self._attn_step_counter = 0
            os.makedirs(self._attn_output_dir, exist_ok=True)
            # sdpa does not support output_attentions; switch to eager
            self.qwen_vl_interface.model.set_attn_implementation("eager")
            logger.info(
                f"Attention capture enabled → {self._attn_output_dir} "
                f"(interval={self._attn_save_interval}, attn_impl → eager)"
            )

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        训练前向：直接回归未来动作（无扩散）。

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        # step 0: add special action token to instruction
        action_tokens = self.action_token* self.chunk_len #can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 提取动作 token embedding 作为动作预测查询
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

            # 标签对齐：取最后 chunk_len 段
            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            # 计算 L1 损失
            action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # step 0: add special action token to instruction
        action_tokens = self.action_token* self.chunk_len #can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        # Only request attention tensors on steps we will actually save
        capture_attn = (
            self._save_attention
            and self._attn_step_counter % self._attn_save_interval == 0
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=capture_attn,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 提取动作 token embedding 作为动作预测查询
            input_ids = qwen_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)

        # --- Attention capture ---
        if capture_attn and qwenvl_outputs.attentions is not None:
            try:
                self._process_and_save_attention(
                    attentions=qwenvl_outputs.attentions,
                    input_ids=input_ids,
                    image_grid_thw=qwen_inputs.get("image_grid_thw", None),
                    raw_images=batch_images,
                )
            except Exception as e:
                logger.warning(f"Attention capture failed at step {self._attn_step_counter}: {e}")
        # Always increment counter (even when not saving) so interval tracking works
        if self._save_attention:
            self._attn_step_counter += 1

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def _process_and_save_attention(
        self,
        attentions: Tuple[torch.Tensor, ...],
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        raw_images: List,
    ) -> None:
        """
        Extract, summarize, and save attention data from the last N layers.

        Saves per-step .npz with:
          - aggregate: per-layer attention proportions from action tokens → {image, text, action}
          - spatial: per-image spatial attention map (reshaped to grid)
          - images: raw observation images (for overlay visualization)
        """
        num_layers = self._attn_num_layers
        # attentions: tuple of (B, num_heads, seq_len, seq_len), one per layer
        total_layers = len(attentions)
        layer_indices = list(range(max(0, total_layers - num_layers), total_layers))

        ids = input_ids[0]  # first sample in batch
        seq_len = ids.shape[0]

        # --- Segment token types ---
        image_mask = (ids == _IMAGE_TOKEN_ID)
        action_mask = (ids == self.action_token_id)
        text_mask = ~image_mask & ~action_mask

        image_positions = image_mask.nonzero(as_tuple=False).squeeze(-1)
        action_positions = action_mask.nonzero(as_tuple=False).squeeze(-1)

        if action_positions.numel() == 0:
            logger.warning("No action tokens found — skipping attention capture")
            return

        # --- Aggregate: action→{image, text, action} per layer ---
        aggregate = {}  # layer_idx -> {"image": float, "text": float, "action": float}
        per_act_aggregate = {}  # layer_idx -> {"image": (num_act,), ...}
        for li in layer_indices:
            # attn shape: (B, num_heads, seq_len, seq_len)
            attn = attentions[li][0]  # (num_heads, seq_len, seq_len)
            # Average over heads
            attn_avg = attn.float().mean(dim=0)  # (seq_len, seq_len)
            # Rows = action token positions, cols = what they attend to
            action_attn = attn_avg[action_positions]  # (num_action_tokens, seq_len)

            # Per-action-token proportions
            img_per_act = action_attn[:, image_mask].sum(dim=1)  # (num_action_tokens,)
            txt_per_act = action_attn[:, text_mask].sum(dim=1)
            act_per_act = action_attn[:, action_mask].sum(dim=1)
            total_per_act = img_per_act + txt_per_act + act_per_act + 1e-12
            per_act_aggregate[f"layer_{li}"] = {
                "image": (img_per_act / total_per_act).cpu().numpy(),
                "text": (txt_per_act / total_per_act).cpu().numpy(),
                "action": (act_per_act / total_per_act).cpu().numpy(),
            }

            # Global aggregate (backward compatible)
            img_attn = img_per_act.sum().item()
            txt_attn = txt_per_act.sum().item()
            act_attn = act_per_act.sum().item()
            total = img_attn + txt_attn + act_attn + 1e-12
            aggregate[f"layer_{li}"] = {
                "image": img_attn / total,
                "text": txt_attn / total,
                "action": act_attn / total,
            }

        # --- Spatial: per-image attention heatmap ---
        spatial_maps = []
        per_act_spatial_maps = []
        if image_grid_thw is not None and image_positions.numel() > 0:
            # Use attention from the last layer, averaged over heads
            last_attn = attentions[layer_indices[-1]][0].float().mean(dim=0)  # (seq_len, seq_len)
            # action tokens attending to image tokens
            action_to_image = last_attn[action_positions][:, image_positions]  # (num_act, num_img_tokens)
            num_act = action_to_image.shape[0]
            # Average across action tokens (backward compatible)
            avg_img_attn = action_to_image.mean(dim=0).cpu().numpy()  # (num_img_tokens,)
            # Keep per-action-token attention for fine-grained visualization
            per_act_img_attn = action_to_image.cpu().numpy()  # (num_act, num_img_tokens)

            # Split by image using grid_thw
            offset = 0
            for img_idx in range(image_grid_thw.shape[0]):
                t, h, w = image_grid_thw[img_idx].tolist()
                t, h, w = int(t), int(h), int(w)
                # After spatial merge (factor=2), the LLM token grid is:
                llm_h = h // 2
                llm_w = w // 2
                num_tokens = int(t) * llm_h * llm_w
                if offset + num_tokens > len(avg_img_attn):
                    break

                # Averaged spatial map (backward compatible)
                img_attn_slice = avg_img_attn[offset:offset + num_tokens]
                spatial_map = img_attn_slice.reshape(int(t), llm_h, llm_w)
                if int(t) == 1:
                    spatial_map = spatial_map[0]  # (llm_h, llm_w)
                spatial_maps.append(spatial_map)

                # Per-action-token spatial maps
                per_act_slice = per_act_img_attn[:, offset:offset + num_tokens]
                per_act_spatial = per_act_slice.reshape(num_act, int(t), llm_h, llm_w)
                if int(t) == 1:
                    per_act_spatial = per_act_spatial[:, 0, :, :]  # (num_act, llm_h, llm_w)
                per_act_spatial_maps.append(per_act_spatial)

                offset += num_tokens

        # --- Per-text-token attention (for word-level analysis) ---
        text_token_attention = None
        text_token_words = None
        text_positions = text_mask.nonzero(as_tuple=False).squeeze(-1)
        if text_positions.numel() > 0:
            last_attn = attentions[layer_indices[-1]][0].float().mean(dim=0)
            # action tokens → each text token: (num_act, num_text_tokens)
            action_to_text = last_attn[action_positions][:, text_positions]
            # Average across action tokens → (num_text_tokens,)
            text_token_attention = action_to_text.mean(dim=0).cpu().numpy()
            # Decode each text token to its word/subword
            text_ids = ids[text_positions].cpu().tolist()
            tokenizer = self.qwen_vl_interface.processor.tokenizer
            text_token_words = np.array(
                [tokenizer.decode([tid]) for tid in text_ids],
                dtype=object,
            )

        # --- Save raw images for overlay ---
        saved_images = []
        if raw_images and len(raw_images) > 0:
            for img in raw_images[0]:  # first sample in batch
                if isinstance(img, Image.Image):
                    saved_images.append(np.array(img))
                elif isinstance(img, np.ndarray):
                    saved_images.append(img)

        # --- Log one-line summary from last layer ---
        last_layer_key = f"layer_{layer_indices[-1]}"
        agg = aggregate[last_layer_key]
        logger.info(
            f"[Attn step {self._attn_step_counter}] "
            f"image={agg['image']:.1%}, text={agg['text']:.1%}, action={agg['action']:.1%}"
        )
        # Also print to stdout for visibility
        print(
            f"Attn step {self._attn_step_counter}: "
            f"image={agg['image']:.1%}, text={agg['text']:.1%}, action={agg['action']:.1%}"
        )

        # --- Save to disk ---
        save_dict = {
            "step": self._attn_step_counter,
        }
        # Aggregate data
        for key, vals in aggregate.items():
            save_dict[f"agg_{key}_image"] = vals["image"]
            save_dict[f"agg_{key}_text"] = vals["text"]
            save_dict[f"agg_{key}_action"] = vals["action"]
        # Spatial maps
        for i, smap in enumerate(spatial_maps):
            save_dict[f"spatial_img{i}"] = smap
        # Per-action-token aggregate
        for key, vals in per_act_aggregate.items():
            save_dict[f"per_act_agg_{key}_image"] = vals["image"]
            save_dict[f"per_act_agg_{key}_text"] = vals["text"]
            save_dict[f"per_act_agg_{key}_action"] = vals["action"]
        # Per-action-token spatial maps
        for i, smap in enumerate(per_act_spatial_maps):
            save_dict[f"per_act_spatial_img{i}"] = smap
        # Raw images
        for i, img_arr in enumerate(saved_images):
            save_dict[f"raw_img{i}"] = img_arr
        # Per-text-token attention
        if text_token_attention is not None:
            save_dict["text_token_attention"] = text_token_attention
            save_dict["text_token_words"] = text_token_words
        # Grid info
        if image_grid_thw is not None:
            save_dict["image_grid_thw"] = image_grid_thw.cpu().numpy()

        save_path = os.path.join(
            self._attn_output_dir,
            f"step_{self._attn_step_counter:05d}.npz",
        )
        np.savez_compressed(save_path, **save_dict)

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,   # [B, L, H]
        input_ids: torch.Tensor,     # [B, L]
        action_token_id=None,        # 可为 int 或 List[int]
    ) -> torch.Tensor:
        """
        向量化批量提取动作 token embedding:
          - 不再逐样本 for 循环
          - 取每个样本里最靠后的 chunk_len 个动作占位 token
        Args:
            last_hidden: [B, L, H]
            input_ids:   [B, L]
            action_token_id: int 或 List[int]
        Returns:
            action_queries: [B, chunk_len, H]
        """
        if action_token_id is None:
            raise ValueError("action_token_id 不能为空")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # 支持多 id（如多个变体）
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            # torch.isin 需要 PyTorch >=1.10
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"以下样本动作 token 数量不足 {self.chunk_len}: {insufficient} | counts={counts.tolist()}"
            )

        # 位置索引
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))   # 非动作位置置 -1

        # 取最后 chunk_len 个（索引大的在序列靠后）
        # 注意: 已确保数量足够，不会出现 -1 被错误选中的问题
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values     # [B, chunk_len] 未排序
        # 时间顺序排序
        selected_pos = topk_pos.sort(dim=-1).values                     # [B, chunk_len]

        # Gather
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)   # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.action_model.action_hidden_dim = 2048

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"
    

    # try get model
    model = Qwenvl_OFT(cfg)
    print(model)

    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # two views
        "lang": "This is a fake instruction for testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # two views
        "lang": "For testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")


    # try forward model
    # can be fake sample， but here get from dataloader for simpler
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # zhe
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        batch
        break

    # try get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model(batch)
    pass
    action = model.predict_action(batch)
