"""
Critical comparison: does VQA cotrain ADD pseudo-labeling capability,
or is the baseline (action-only) ckpt's LM head equally good at answering
"low or high contact at grasp?"

If baseline acc ≈ 0.99 on taskB with mid_8 → VQA training was redundant.
If baseline acc ≈ 0.5 random → VQA training was load-bearing.
Anything in between → VQA helped partially.

Loads BOTH ckpts sequentially. Manually injects the VQA-specific attrs
(question, answer tokens) onto baseline so predict_with_logits works
identically — testing apples to apples.

CLI:
  python -m examples.preference.eval.baseline_vqa_compare \
      --baseline_yaml ... --baseline_ckpt ... \
      --vqa_yaml ... --vqa_ckpt ... \
      --taskB_data_root ... --taskA_n_eps_per_task 5 \
      --strategies mid_8,gripper_anchored,uniform_8 \
      --out r-preference/eval/baseline_vqa_compare_35k.json
"""
from __future__ import annotations
import argparse, io, json, time
from collections import defaultdict
from pathlib import Path
from typing import List

import h5py, numpy as np, torch
from omegaconf import OmegaConf
from PIL import Image

from examples.preference.dataset.prompt import PREF_CATEGORIES
from examples.preference.dataset.pref_hdf5_dataset import PrefHDF5Dataset
from examples.preference.dataset.vqa_sample import VQA_CATEGORIES
from examples.preference.eval.stage_a_gate import (
    build_framework, load_ckpt_into_model, list_taskB_episodes,
    subsample_taskA_episodes,
)
from examples.preference.eval.diagnostic import predict_with_logits
from examples.preference.eval.frame_window_test import load_clip_strategy
from examples.preference.eval.gripper_anchored_test import make_gripper_anchored_clip


def inject_vqa_attrs(model, category: str):
    """Inject Qwen_PI_VQA's required attrs onto a vanilla Qwen_PI so
    predict_with_logits / predict_preference work uniformly."""
    vqa_cat = VQA_CATEGORIES[category]
    model._vqa_question = vqa_cat.question
    model._vqa_answer_text = dict(vqa_cat.answer_text)
    model._vqa_answer_token_ids = dict(vqa_cat.answer_token_ids)
    model._vqa_pk_A, model._vqa_pk_B = vqa_cat.pref_keys
    model._vqa_id_A = vqa_cat.answer_token_ids[model._vqa_pk_A]
    model._vqa_id_B = vqa_cat.answer_token_ids[model._vqa_pk_B]
    model._vqa_text_A = model._vqa_answer_text[model._vqa_pk_A]
    model._vqa_text_B = model._vqa_answer_text[model._vqa_pk_B]
    model._vqa_text_by_id = {model._vqa_id_A: model._vqa_text_A,
                              model._vqa_id_B: model._vqa_text_B}


def eval_model_on_episodes(model, episodes, source_root, val_ds, strategy: str, category: str):
    cat = PREF_CATEGORIES[category]
    records = []
    t0 = time.time()
    for i, fs in enumerate(episodes):
        if source_root is None:
            h5p = val_ds.data_root / fs.task_dir / "data" / f"episode{fs.ep_id}.hdf5"
        else:
            h5p = source_root / fs.task_dir / "data" / f"episode{fs.ep_id}.hdf5"
        if strategy == "gripper_anchored":
            clip, _info = make_gripper_anchored_clip(h5p)
        else:
            clip = load_clip_strategy(h5p, strategy)
        r = predict_with_logits(model, clip)
        r.update({"gt_pref_key": fs.pref_key, "ep_id": fs.ep_id,
                  "task_group": fs.task_group,
                  "correct": int(r["pred_pref_key"] == fs.pref_key)})
        records.append(r)
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{len(episodes)} ({time.time()-t0:.1f}s)")
    return records


def summarize(records, cat):
    if not records: return {}
    acc = float(np.mean([r["correct"] for r in records]))
    gt_low = [r for r in records if r["gt_pref_key"] == cat.pref_keys[0]]
    gt_high = [r for r in records if r["gt_pref_key"] == cat.pref_keys[1]]
    return {
        "n": len(records), "acc": acc,
        "acc_gt_low":  float(np.mean([r["correct"] for r in gt_low])) if gt_low else 0.0,
        "acc_gt_high": float(np.mean([r["correct"] for r in gt_high])) if gt_high else 0.0,
        "n_pred_low":  sum(1 for r in records if r["pred_pref_key"] == cat.pref_keys[0]),
        "n_pred_high": sum(1 for r in records if r["pred_pref_key"] == cat.pref_keys[1]),
        "gap_mean_overall": float(np.mean([r["logit_gap"] for r in records])),
        "gap_diff_signal": (float(np.mean([r["logit_gap"] for r in gt_low])) -
                            float(np.mean([r["logit_gap"] for r in gt_high])))
                            if (gt_low and gt_high) else 0.0,
    }


def summarize_per_task(records, cat):
    per = defaultdict(list)
    for r in records: per[r["task_group"]].append(r)
    return {tg: summarize(rs, cat) for tg, rs in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="contact")
    ap.add_argument("--baseline_yaml", required=True)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--vqa_yaml", required=True)
    ap.add_argument("--vqa_ckpt", required=True)
    ap.add_argument("--taskB_data_root", required=True)
    ap.add_argument("--taskA_n_eps_per_task", type=int, default=5)
    ap.add_argument("--strategies", default="mid_8,gripper_anchored,uniform_8")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import random as _random
    rng = _random.Random(args.seed)
    cat = PREF_CATEGORIES[args.category]

    # Build taskA val ds + taskB enumeration
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
    taskB_eps = list_taskB_episodes(Path(args.taskB_data_root), cat.pref_keys)
    print(f"[bvq] taskA={len(taskA_eps)} taskB={len(taskB_eps)}")

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    out = {"category": args.category,
           "baseline_ckpt": args.baseline_ckpt,
           "vqa_ckpt": args.vqa_ckpt,
           "strategies": strategies,
           "results": {}}

    for tag, yaml_path, ckpt_path, framework_name in [
        ("baseline", args.baseline_yaml, args.baseline_ckpt, "QwenPI"),
        ("vqa", args.vqa_yaml, args.vqa_ckpt, "QwenPI_VQA"),
    ]:
        print(f"\n========== PASS: {tag} ckpt ({framework_name}) ==========")
        t0 = time.time()
        model, _ = build_framework(yaml_path, framework_name)
        load_ckpt_into_model(model, ckpt_path)
        if framework_name == "QwenPI":
            inject_vqa_attrs(model, args.category)
            print(f"  injected VQA attrs onto baseline framework")
        model = model.cuda().eval()
        print(f"  loaded in {time.time()-t0:.1f}s")

        out["results"][tag] = {}
        for strategy in strategies:
            print(f"\n  --- strategy={strategy} ---")
            recs_B = eval_model_on_episodes(model, taskB_eps, Path(args.taskB_data_root),
                                              None, strategy, args.category)
            recs_A = eval_model_on_episodes(model, taskA_eps, None, val_ds,
                                              strategy, args.category)
            sB = summarize(recs_B, cat)
            sA = summarize(recs_A, cat)
            per_A = summarize_per_task(recs_A, cat)
            out["results"][tag][strategy] = {
                "taskB": sB,
                "taskA_overall": sA,
                "taskA_per_task": per_A,
            }
            print(f"    taskB:  acc={sB['acc']:.3f}  pred_low/high={sB['n_pred_low']}/{sB['n_pred_high']}  signal={sB['gap_diff_signal']:+.2f}")
            print(f"    taskA:  acc={sA['acc']:.3f}  pred_low/high={sA['n_pred_low']}/{sA['n_pred_high']}  signal={sA['gap_diff_signal']:+.2f}")
            for tg, st in per_A.items():
                print(f"      {tg:30s}  acc={st['acc']:.2f}  pred_low/high={st['n_pred_low']}/{st['n_pred_high']}")

        del model
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[bvq] saved {args.out}")

    # Print key comparison table
    print("\n========== KEY COMPARISON ==========")
    print(f"{'strategy':<20} {'baseline_taskB':<15} {'vqa_taskB':<15} {'baseline_taskA':<15} {'vqa_taskA':<15}")
    for s in strategies:
        r = out["results"]
        bB = r["baseline"][s]["taskB"]["acc"]
        vB = r["vqa"][s]["taskB"]["acc"]
        bA = r["baseline"][s]["taskA_overall"]["acc"]
        vA = r["vqa"][s]["taskA_overall"]["acc"]
        print(f"{s:<20} {bB:<15.3f} {vB:<15.3f} {bA:<15.3f} {vA:<15.3f}")


if __name__ == "__main__":
    main()
