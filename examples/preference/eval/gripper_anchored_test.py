"""
Gripper-anchored clip selection — universal-by-design replacement for the
hard-coded uniform_linspace clip strategy.

Heuristic: for each episode, find the FIRST frame where the gripper state
crosses below a closure threshold (= grasp moment). Sample 8 frames
centered on that moment, half-step apart.

If gripper never closes (rare in this dataset), fall back to uniform_8.

This is independent of episode length and physics, so it should be the
universally-correct fix for the frame-selection mismatch found in the
gate analysis.

CLI:
  python -m examples.preference.eval.gripper_anchored_test \
      --category contact \
      --vqa_yaml ... --vqa_ckpt ... \
      --taskB_data_root ... --taskA_n_eps_per_task 10 \
      --out r-preference/eval/gripper_anchored_50k.json
"""
from __future__ import annotations
import argparse, io, json, time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import h5py, numpy as np, torch
from omegaconf import OmegaConf
from PIL import Image

from examples.preference.dataset.prompt import PREF_CATEGORIES
from examples.preference.dataset.pref_hdf5_dataset import PrefHDF5Dataset
from examples.preference.eval.stage_a_gate import (
    build_framework, load_ckpt_into_model, list_taskB_episodes,
    subsample_taskA_episodes,
)
from examples.preference.eval.diagnostic import predict_with_logits


def find_grasp_frame(h5: h5py.File, gripper_close_thresh: float = 0.5) -> Optional[int]:
    """Find first frame where EITHER gripper crosses BELOW closure threshold.

    Gripper values in this dataset: ~0.0 (closed) to ~1.0 (open). Grasp =
    transition open->closed = value drops below 0.5.

    Returns frame index of first such transition, or None if no transition.
    """
    L = np.asarray(h5["endpose/left_gripper"][:])
    R = np.asarray(h5["endpose/right_gripper"][:])
    # First frame where left OR right falls below threshold (transitions
    # are easier to detect than absolute thresholding if start state varies)
    for t in range(1, len(L)):
        if (L[t] < gripper_close_thresh and L[t-1] >= gripper_close_thresh) or \
           (R[t] < gripper_close_thresh and R[t-1] >= gripper_close_thresh):
            return t
    # Fallback: just check where it's below thresh
    where = np.where((L < gripper_close_thresh) | (R < gripper_close_thresh))[0]
    return int(where[0]) if len(where) > 0 else None


def make_gripper_anchored_clip(h5_path: Path, n: int = 8, half_window: int = 4,
                                 cam: str = "head_camera",
                                 hw=(224, 224)) -> Tuple[List[Image.Image], dict]:
    """Sample n frames centered on the grasp moment. half_window = how many
    frames before/after to span (so total span = 2*half_window frames, but
    we down-sample to n)."""
    H, W = hw
    with h5py.File(h5_path, "r") as h5:
        T = h5[f"observation/{cam}/rgb"].shape[0]
        grasp_t = find_grasp_frame(h5)
        if grasp_t is None:
            # uniform fallback
            idx = np.linspace(0, T - 1, n).round().astype(int)
            mode = "uniform_fallback"
        else:
            # Sample n frames spanning grasp_t - half_window*n//2 to grasp_t + half_window*n//2
            # but clipped to [0, T-1]
            span = max(half_window * 2, 1)
            t0 = max(0, grasp_t - span * n // 2)
            t1 = min(T - 1, grasp_t + span * n // 2)
            # If span hit boundary, expand the other side
            if t0 == 0:
                t1 = min(T - 1, t0 + span * (n - 1))
            elif t1 == T - 1:
                t0 = max(0, t1 - span * (n - 1))
            idx = np.linspace(t0, t1, n).round().astype(int)
            mode = "grasp_anchored"

        frames = []
        for t in idx:
            raw = h5[f"observation/{cam}/rgb"][int(t)]
            data = raw.tobytes() if hasattr(raw, "tobytes") else raw
            img = Image.open(io.BytesIO(data)).convert("RGB")
            if img.size != (W, H): img = img.resize((W, H))
            frames.append(img)

    return frames, {"grasp_t": grasp_t, "T": T, "frame_idx": idx.tolist(), "mode": mode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="contact")
    ap.add_argument("--vqa_yaml", required=True)
    ap.add_argument("--vqa_ckpt", required=True)
    ap.add_argument("--taskB_data_root", required=True)
    ap.add_argument("--taskA_n_eps_per_task", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import random as _random
    rng = _random.Random(args.seed)

    cat = PREF_CATEGORIES[args.category]
    taskB_eps = list_taskB_episodes(Path(args.taskB_data_root), cat.pref_keys)
    print(f"[ga] taskB: {len(taskB_eps)} episodes")

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
    print(f"[ga] taskA val: {len(taskA_eps)} episodes")

    t0 = time.time()
    model, _ = build_framework(args.vqa_yaml, "QwenPI_VQA")
    load_ckpt_into_model(model, args.vqa_ckpt)
    model = model.cuda().eval()
    print(f"[ga] model loaded in {time.time()-t0:.1f}s")

    out = {"category": args.category, "vqa_ckpt": args.vqa_ckpt,
           "taskB_records": [], "taskA_records": []}

    print(f"\n  === taskB ({len(taskB_eps)} ep) gripper-anchored ===")
    grasp_ts = []
    Ts = []
    t0 = time.time()
    for i, fs in enumerate(taskB_eps):
        h5p = Path(args.taskB_data_root) / fs.task_dir / "data" / f"episode{fs.ep_id}.hdf5"
        clip, info = make_gripper_anchored_clip(h5p)
        r = predict_with_logits(model, clip)
        r.update({"gt_pref_key": fs.pref_key, "ep_id": fs.ep_id,
                  "task_group": fs.task_group,
                  "correct": int(r["pred_pref_key"] == fs.pref_key),
                  **info})
        out["taskB_records"].append(r)
        if info["grasp_t"] is not None:
            grasp_ts.append(info["grasp_t"]); Ts.append(info["T"])
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{len(taskB_eps)} ({time.time()-t0:.1f}s)")
    acc = float(np.mean([r["correct"] for r in out["taskB_records"]]))
    print(f"  taskB acc={acc:.3f}")
    if grasp_ts:
        ratios = [g/T for g,T in zip(grasp_ts, Ts)]
        print(f"  grasp_t/T: mean={np.mean(ratios):.3f} std={np.std(ratios):.3f} min={min(ratios):.3f} max={max(ratios):.3f}")
        print(f"  grasp_t (absolute): mean={np.mean(grasp_ts):.1f} std={np.std(grasp_ts):.1f}")

    print(f"\n  === taskA val ({len(taskA_eps)} ep) gripper-anchored ===")
    grasp_ts = []
    Ts = []
    t0 = time.time()
    per_task = defaultdict(list)
    for i, fs in enumerate(taskA_eps):
        h5p = val_ds.data_root / fs.task_dir / "data" / f"episode{fs.ep_id}.hdf5"
        clip, info = make_gripper_anchored_clip(h5p)
        r = predict_with_logits(model, clip)
        r.update({"gt_pref_key": fs.pref_key, "ep_id": fs.ep_id,
                  "task_group": fs.task_group,
                  "correct": int(r["pred_pref_key"] == fs.pref_key),
                  **info})
        out["taskA_records"].append(r)
        per_task[fs.task_group].append(r)
        if info["grasp_t"] is not None:
            grasp_ts.append(info["grasp_t"]); Ts.append(info["T"])
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{len(taskA_eps)} ({time.time()-t0:.1f}s)")
    acc = float(np.mean([r["correct"] for r in out["taskA_records"]]))
    print(f"  taskA overall acc={acc:.3f}")
    if grasp_ts:
        ratios = [g/T for g,T in zip(grasp_ts, Ts)]
        print(f"  grasp_t/T: mean={np.mean(ratios):.3f} std={np.std(ratios):.3f}")
    print("  per-task:")
    for tg, recs in per_task.items():
        a = float(np.mean([r['correct'] for r in recs]))
        nl = sum(1 for r in recs if r['pred_pref_key'] == cat.pref_keys[0])
        nh = sum(1 for r in recs if r['pred_pref_key'] == cat.pref_keys[1])
        print(f"    {tg:30s}  acc={a:.2f}  pred_low/high={nl}/{nh}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[ga] saved {args.out}")


if __name__ == "__main__":
    main()
