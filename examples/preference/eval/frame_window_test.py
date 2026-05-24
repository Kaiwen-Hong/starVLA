"""
Test recommendation #1 from analysis: does a denser mid-window clip help
the VQA head perceive grasp pose on taskB?

Compares 4 frame-selection strategies on the same VQA ckpt + same taskB
episodes:
  - uniform_8     : baseline (current strategy, np.linspace(0, T-1, 8))
  - mid_8         : 8 frames densely packed around grasp moment, fractions
                    [0.25, 0.32, 0.40, 0.45, 0.50, 0.55, 0.62, 0.70]
  - dense_16      : 16 uniform frames (more visual context)
  - mid_dense_16  : 16 frames mid-clustered, fractions linspace(0.20, 0.70, 16)

If acc improves with a denser/mid-clustered window, frame selection is the
bottleneck. Otherwise the issue is feature-learning/scene-shortcut.

CLI: python -m examples.preference.eval.frame_window_test \
        --vqa_yaml ... --vqa_ckpt ... --taskB_data_root ... --out ...
"""
from __future__ import annotations
import argparse, io, json, time
from collections import defaultdict
from pathlib import Path
from typing import List

import h5py, numpy as np, torch
from PIL import Image

from examples.preference.dataset.prompt import PREF_CATEGORIES
from examples.preference.dataset.vqa_sample import VQA_CATEGORIES
from examples.preference.eval.stage_a_gate import (
    build_framework, load_ckpt_into_model, list_taskB_episodes,
    subsample_taskA_episodes,
)
from examples.preference.eval.diagnostic import predict_with_logits
from examples.preference.dataset.pref_hdf5_dataset import PrefHDF5Dataset
from omegaconf import OmegaConf


def _decode(h5, cam, idx, hw=(224,224)):
    H, W = hw
    out = []
    for t in idx:
        raw = h5[f"observation/{cam}/rgb"][int(t)]
        data = raw.tobytes() if hasattr(raw, "tobytes") else raw
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.size != (W, H): img = img.resize((W, H))
        out.append(img)
    return out


def load_clip_strategy(h5_path, strategy: str, cam: str = "head_camera") -> List[Image.Image]:
    with h5py.File(h5_path, "r") as h5:
        T = h5[f"observation/{cam}/rgb"].shape[0]
        if strategy == "uniform_8":
            idx = np.linspace(0, T-1, 8).round().astype(int)
        elif strategy == "mid_8":
            fracs = np.array([0.25, 0.32, 0.40, 0.45, 0.50, 0.55, 0.62, 0.70])
            idx = (fracs * (T-1)).round().astype(int)
        elif strategy == "dense_16":
            idx = np.linspace(0, T-1, 16).round().astype(int)
        elif strategy == "mid_dense_16":
            idx = np.linspace(0.20 * (T-1), 0.70 * (T-1), 16).round().astype(int)
        else:
            raise ValueError(strategy)
        return _decode(h5, cam, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="contact")
    ap.add_argument("--vqa_yaml", required=True)
    ap.add_argument("--vqa_ckpt", required=True)
    ap.add_argument("--taskB_data_root", required=True)
    ap.add_argument("--taskA_n_eps_per_task", type=int, default=0,
                    help="If >0, also test frame strategies on taskA val (this many eps per task_dir)")
    ap.add_argument("--strategies", default="uniform_8,mid_8,dense_16,mid_dense_16",
                    help="Comma-separated list of strategies to test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import random as _random
    rng = _random.Random(args.seed)

    cat = PREF_CATEGORIES[args.category]
    taskB_eps = list_taskB_episodes(Path(args.taskB_data_root), cat.pref_keys)
    print(f"[fw] taskB: {len(taskB_eps)} episodes")

    # Optional taskA val
    val_ds = None
    taskA_eps = []
    if args.taskA_n_eps_per_task > 0:
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
        print(f"[fw] taskA val: {len(taskA_eps)} episodes")

    t0 = time.time()
    model, _ = build_framework(args.vqa_yaml, "QwenPI_VQA")
    load_ckpt_into_model(model, args.vqa_ckpt)
    model = model.cuda().eval()
    print(f"[fw] model loaded in {time.time()-t0:.1f}s")

    out = {"category": args.category, "vqa_ckpt": args.vqa_ckpt, "taskB": {}, "taskA": {}}
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    def _eval_episodes(strategy, episodes, source_root, tag):
        records = []
        t0 = time.time()
        for i, fs in enumerate(episodes):
            if source_root is None:
                # taskA via val_ds
                h5p = val_ds.data_root / fs.task_dir / "data" / f"episode{fs.ep_id}.hdf5"
            else:
                h5p = source_root / fs.task_dir / "data" / f"episode{fs.ep_id}.hdf5"
            clip = load_clip_strategy(h5p, strategy)
            r = predict_with_logits(model, clip)
            r.update({"gt_pref_key": fs.pref_key, "ep_id": fs.ep_id,
                      "task_group": fs.task_group,
                      "correct": int(r["pred_pref_key"] == fs.pref_key)})
            records.append(r)
            if (i+1) % 50 == 0:
                print(f"    {tag} {i+1}/{len(episodes)} ({time.time()-t0:.1f}s)")
        return records

    def _summarize(records):
        if not records: return {}
        acc = float(np.mean([r["correct"] for r in records]))
        gt_low = [r for r in records if r["gt_pref_key"] == cat.pref_keys[0]]
        gt_high = [r for r in records if r["gt_pref_key"] == cat.pref_keys[1]]
        return {
            "n": len(records), "acc": acc,
            "acc_gt_low":  float(np.mean([r["correct"] for r in gt_low]))  if gt_low else 0.0,
            "acc_gt_high": float(np.mean([r["correct"] for r in gt_high])) if gt_high else 0.0,
            "n_pred_low":  sum(1 for r in records if r["pred_pref_key"] == cat.pref_keys[0]),
            "n_pred_high": sum(1 for r in records if r["pred_pref_key"] == cat.pref_keys[1]),
            "gap_mean_overall": float(np.mean([r["logit_gap"] for r in records])),
            "gap_diff_signal": (float(np.mean([r["logit_gap"] for r in gt_low])) -
                                float(np.mean([r["logit_gap"] for r in gt_high])))
                                if (gt_low and gt_high) else 0.0,
        }

    def _summarize_per_task(records):
        per = defaultdict(list)
        for r in records: per[r["task_group"]].append(r)
        return {tg: _summarize(rs) for tg, rs in per.items()}

    for strategy in strategies:
        print(f"\n  === strategy={strategy} ===")
        if taskB_eps:
            recs_B = _eval_episodes(strategy, taskB_eps, Path(args.taskB_data_root), "taskB")
            s = _summarize(recs_B)
            out["taskB"][strategy] = s
            print(f"  taskB: acc={s['acc']:.3f}  pred={s['n_pred_low']}/{s['n_pred_high']}  signal={s['gap_diff_signal']:+.2f}")
        if taskA_eps:
            recs_A = _eval_episodes(strategy, taskA_eps, None, "taskA")
            s = _summarize(recs_A)
            per_tg = _summarize_per_task(recs_A)
            out["taskA"][strategy] = {"summary": s, "per_task": per_tg}
            print(f"  taskA overall: acc={s['acc']:.3f}  pred={s['n_pred_low']}/{s['n_pred_high']}  signal={s['gap_diff_signal']:+.2f}")
            for tg, st in per_tg.items():
                print(f"    {tg:30s}  acc={st['acc']:.2f}  pred_low/high={st['n_pred_low']}/{st['n_pred_high']}  signal={st['gap_diff_signal']:+.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[fw] saved {args.out}")


if __name__ == "__main__":
    main()
