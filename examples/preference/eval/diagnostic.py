"""
Augmented diagnostic for VQA gate failures.

Adds on top of stage_a_gate.py:
  - Per-sample raw logit dump (low_logit, high_logit, gap)
  - Per-sample two-token softmax (the model's reported 'confidence')
  - Per-sample full-vocab marginal: softmax over ALL vocab, then sum of
    just {low,high} ids = probability mass the model assigns to either
    answer (sanity for "is the model even thinking about answering with
    one of these two?")
  - Unconditional bias check: 8-frame BLACK clip and 8-frame NOISE clip
    -> if model still strongly predicts low/high, the bias lives in the
    LM head's class prior, not in visual perception.
  - Class confusion matrix per-task (FP / FN / TP / TN breakdown).

CLI:
  python -m examples.preference.eval.diagnostic \
      --category contact \
      --vqa_yaml  examples/preference/train_files/starvla_pref_stage_a_vqa_contact.yaml \
      --vqa_ckpt  /mnt/localssd/.../steps_35000_pytorch_model.pt \
      --taskB_data_root /mnt/localssd/kaiwenh/pref/data/contact/taskB \
      --taskA_n_eps_per_task 5 \
      --out r-preference/eval/diagnostic_contact_35k.json
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from examples.preference.dataset.prompt import PREF_CATEGORIES
from examples.preference.dataset.pref_hdf5_dataset import PrefHDF5Dataset
from examples.preference.dataset.vqa_sample import (
    VQA_CATEGORIES,
    CLIP_STRATEGY_NAMES,
    load_clip_by_strategy,
)
from examples.preference.eval.stage_a_gate import (
    build_framework,
    load_ckpt_into_model,
    load_clip_from_h5,  # kept for back-compat; new path uses load_clip_by_strategy
    list_taskB_episodes,
    subsample_taskA_episodes,
    FrameSample,
)


@torch.inference_mode()
def predict_with_logits(model, clip: List[Image.Image]) -> dict:
    """Like Qwen_PI_VQA.predict_preference but exposes raw logits + marginals."""
    processor = model.qwen_vl_interface.processor
    device = model.qwen_vl_interface.model.device
    msg = [{
        "role": "user",
        "content": [{"type": "image", "image": img} for img in clip] +
                   [{"type": "text", "text": model._vqa_question}],
    }]
    inputs = processor.apply_chat_template(
        [msg], tokenize=True, padding=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.qwen_vl_interface(**inputs, return_dict=True)
    ans_logits = out.logits[0, -1].float()  # (V,)

    A_logit = float(ans_logits[model._vqa_id_A].item())
    B_logit = float(ans_logits[model._vqa_id_B].item())
    pair = torch.tensor([A_logit, B_logit])
    pair_probs = F.softmax(pair, dim=0)
    p_A_pair = float(pair_probs[0].item())
    p_B_pair = float(pair_probs[1].item())

    # Full-vocab softmax (then marginal over our two tokens — what's the
    # probability mass the model 'really' assigns to either answer vs
    # something else like a continuation token or punctuation?)
    full = F.softmax(ans_logits, dim=0)
    p_A_full = float(full[model._vqa_id_A].item())
    p_B_full = float(full[model._vqa_id_B].item())
    p_AB_mass = p_A_full + p_B_full

    # Top-5 raw logits (for sanity — what was the model actually about to say?)
    top5 = torch.topk(ans_logits, 5)
    top5_ids = top5.indices.cpu().tolist()
    top5_logits = top5.values.cpu().tolist()
    tok = processor.tokenizer
    top5_decoded = [tok.decode([tid]) for tid in top5_ids]

    winner_pk = model._vqa_pk_A if A_logit > B_logit else model._vqa_pk_B
    return {
        "pred_pref_key": winner_pk,
        "A_logit": A_logit, "B_logit": B_logit,
        "logit_gap": A_logit - B_logit,
        "p_A_pair": p_A_pair, "p_B_pair": p_B_pair,
        "p_A_full": p_A_full, "p_B_full": p_B_full,
        "p_AB_mass_full": p_AB_mass,
        "top5": [(tok_id, decoded, logit) for tok_id, decoded, logit
                 in zip(top5_ids, top5_decoded, top5_logits)],
    }


def make_blank_clip(n: int = 8, size: int = 224) -> List[Image.Image]:
    return [Image.new("RGB", (size, size), (0, 0, 0)) for _ in range(n)]


def make_noise_clip(n: int = 8, size: int = 224, seed: int = 0) -> List[Image.Image]:
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
            for _ in range(n)]


def confusion(records: List[dict], pk_A: str, pk_B: str) -> dict:
    """Return per-task confusion stats."""
    by_task: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"TP_A": 0, "FP_A": 0, "TP_B": 0, "FP_B": 0,
                 "n_gt_A": 0, "n_gt_B": 0})
    for r in records:
        tg = r["task_group"]; gt = r["gt_pref_key"]; pr = r["pred_pref_key"]
        d = by_task[tg]
        if gt == pk_A:
            d["n_gt_A"] += 1
            if pr == pk_A: d["TP_A"] += 1
            else:          d["FP_B"] += 1  # predicted B when GT was A
        else:
            d["n_gt_B"] += 1
            if pr == pk_B: d["TP_B"] += 1
            else:          d["FP_A"] += 1
    return dict(by_task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=list(PREF_CATEGORIES))
    ap.add_argument("--vqa_yaml", required=True)
    ap.add_argument("--vqa_ckpt", required=True)
    ap.add_argument("--taskA_n_eps_per_task", type=int, default=5)
    ap.add_argument("--taskB_data_root", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip_strategy", default="uniform_8", choices=list(CLIP_STRATEGY_NAMES),
                    help="Clip selection strategy for VQA forward. Default uniform_8 "
                         "(back-compat with original gate eval). For Stage B labeling "
                         "use mid_8 or gripper_anchored (validated).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cat = PREF_CATEGORIES[args.category]
    vqa_cat = VQA_CATEGORIES[args.category]
    pk_A, pk_B = vqa_cat.pref_keys
    print(f"[diag] category={args.category} pk_A={pk_A} pk_B={pk_B} clip_strategy={args.clip_strategy}")

    # Build taskA val ds
    cfg = OmegaConf.load(args.vqa_yaml)
    val_ds = PrefHDF5Dataset(
        data_root_dir=cfg.datasets.vla_data.data_root_dir,
        split="val",
        chunk_size=int(cfg.datasets.vla_data.future_action_window_size) + 1,
        past_window=int(cfg.datasets.vla_data.past_action_window_size),
        include_state=bool(cfg.datasets.vla_data.include_state),
        cameras=tuple(cfg.datasets.vla_data.cameras),
        image_size=tuple(cfg.datasets.vla_data.image_size),
        stats_json_path=cfg.datasets.vla_data.stats_json_path,
        category=args.category,
    )
    taskA_eps = subsample_taskA_episodes(val_ds, args.taskA_n_eps_per_task, rng)
    print(f"[diag] taskA picked: {len(taskA_eps)} episodes")

    taskB_eps = []
    if args.taskB_data_root:
        taskB_eps = list_taskB_episodes(Path(args.taskB_data_root), cat.pref_keys)
        print(f"[diag] taskB enumerated: {len(taskB_eps)} episodes")

    # Load VQA model
    t0 = time.time()
    model, _ = build_framework(args.vqa_yaml, "QwenPI_VQA")
    load_ckpt_into_model(model, args.vqa_ckpt)
    model = model.cuda().eval()
    print(f"[diag] VQA model loaded in {time.time()-t0:.1f}s")

    results = {
        "category": args.category, "vqa_ckpt": args.vqa_ckpt,
        "pk_A": pk_A, "pk_B": pk_B,
        "clip_strategy": args.clip_strategy,
    }

    # === 1. TaskB per-sample diagnostic ===
    if taskB_eps:
        print(f"\n  === TaskB per-sample diagnostic ({len(taskB_eps)} ep, strategy={args.clip_strategy}) ===")
        t0 = time.time()
        records = []
        for i, fs in enumerate(taskB_eps):
            h5p = Path(args.taskB_data_root) / fs.task_dir / "data" / f"episode{fs.ep_id}.hdf5"
            clip, info = load_clip_by_strategy(h5p, strategy=args.clip_strategy)
            r = predict_with_logits(model, clip)
            r["task_dir"] = fs.task_dir
            r["task_group"] = fs.task_group
            r["gt_pref_key"] = fs.pref_key
            r["ep_id"] = fs.ep_id
            r["correct"] = int(r["pred_pref_key"] == fs.pref_key)
            r["clip_info"] = info  # strategy, indices, T, grasp_t
            records.append(r)
            if (i+1) % 20 == 0:
                print(f"    {i+1}/{len(taskB_eps)} ({time.time()-t0:.1f}s)")
        results["taskB_records"] = records

        # Aggregate stats
        acc = float(np.mean([r["correct"] for r in records]))
        gaps = np.array([r["logit_gap"] for r in records])
        p_AB = np.array([r["p_AB_mass_full"] for r in records])
        results["taskB_summary"] = {
            "n": len(records), "acc": acc,
            "logit_gap_mean": float(gaps.mean()),
            "logit_gap_std":  float(gaps.std()),
            "logit_gap_p10":  float(np.percentile(gaps, 10)),
            "logit_gap_p90":  float(np.percentile(gaps, 90)),
            "p_AB_mass_mean": float(p_AB.mean()),  # how much prob mass on the two tokens (vs other)
            "p_AB_mass_min":  float(p_AB.min()),
            "confusion": confusion(records, pk_A, pk_B),
        }
        # Breakdown: per-GT-class
        for gt_pk in (pk_A, pk_B):
            sub = [r for r in records if r["gt_pref_key"] == gt_pk]
            sub_acc = float(np.mean([r["correct"] for r in sub])) if sub else 0.0
            results["taskB_summary"][f"acc_when_gt={gt_pk}"] = {
                "n": len(sub), "acc": sub_acc,
                "n_pred_A": sum(1 for r in sub if r["pred_pref_key"] == pk_A),
                "n_pred_B": sum(1 for r in sub if r["pred_pref_key"] == pk_B),
            }
        print(f"  taskB acc={acc:.3f}, logit_gap mean={gaps.mean():.2f} (A>B if pred=A)")

    # === 2. TaskA per-sample diagnostic ===
    if taskA_eps:
        print(f"\n  === TaskA per-sample diagnostic ({len(taskA_eps)} ep, strategy={args.clip_strategy}) ===")
        t0 = time.time()
        records = []
        for i, fs in enumerate(taskA_eps):
            h5p = val_ds.data_root / fs.task_dir / "data" / f"episode{fs.ep_id}.hdf5"
            clip, info = load_clip_by_strategy(h5p, strategy=args.clip_strategy)
            r = predict_with_logits(model, clip)
            r.update({"task_dir": fs.task_dir, "task_group": fs.task_group,
                      "gt_pref_key": fs.pref_key, "ep_id": fs.ep_id,
                      "correct": int(r["pred_pref_key"] == fs.pref_key),
                      "clip_info": info})
            records.append(r)
            if (i+1) % 20 == 0:
                print(f"    {i+1}/{len(taskA_eps)} ({time.time()-t0:.1f}s)")
        results["taskA_records"] = records

        # Per-task aggregate
        by_tg: Dict[str, List[dict]] = defaultdict(list)
        for r in records:
            by_tg[r["task_group"]].append(r)
        per_task = {}
        for tg, rs in by_tg.items():
            gaps = np.array([r["logit_gap"] for r in rs])
            per_task[tg] = {
                "n": len(rs), "acc": float(np.mean([r["correct"] for r in rs])),
                "logit_gap_mean": float(gaps.mean()),
                "logit_gap_std":  float(gaps.std()),
                "pred_dist": dict(defaultdict(int, **{r["pred_pref_key"]: 0 for r in rs})),
            }
            for r in rs:
                per_task[tg]["pred_dist"][r["pred_pref_key"]] = per_task[tg]["pred_dist"].get(r["pred_pref_key"], 0) + 1
        results["taskA_per_task"] = per_task

    # === 3. Unconditional bias check ===
    print(f"\n  === Unconditional bias check ===")
    for label, clip in [("blank_black", make_blank_clip()),
                         ("noise_seed0", make_noise_clip(seed=0)),
                         ("noise_seed1", make_noise_clip(seed=1)),
                         ("noise_seed2", make_noise_clip(seed=2))]:
        r = predict_with_logits(model, clip)
        results.setdefault("unconditional", {})[label] = r
        print(f"    {label:15s} pred={r['pred_pref_key']!r} gap={r['logit_gap']:+.2f} "
              f"top5={[(d,f'{l:.1f}') for _,d,l in r['top5']]}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[diag] saved {out_path}")


if __name__ == "__main__":
    main()
