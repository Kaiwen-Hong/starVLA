#!/usr/bin/env python3
"""Gate-eval a trained (OFT_)VQA ckpt: VQA accuracy on taskA-val + taskB.

Uses the redesigned per-cat VQA config (VQA_CATEGORIES[cat]: question, single-token
answers, head-only cameras, 3-frame place3/orient3) via predict_preference — exactly
the clips the model was cotrained on. taskA = held-out val episodes (seed-42 split);
taskB = the unseen-object leaves under <data_root>/taskB/.
"""
import argparse, json, random
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf

from starVLA.model.framework import build_framework
from examples.preference.dataset.vqa_sample import VQA_CATEGORIES, load_clip_and_state
from examples.preference.dataset.pref_hdf5_dataset import _task_dirs_for_groups, _split_episodes
from examples.preference.dataset.prompt import PREF_CATEGORIES


def _predict(model, vc, h5_path):
    frames, state, _ = load_clip_and_state(
        h5_path, strategy=vc.clip_strategy, n=vc.n_frames, cameras=vc.cameras, jitter=0)
    return model.predict_preference(frames, state=state if vc.state_in_vqa else None)


def eval_leaves(model, vc, leaf_dirs, n_per_leaf, rng):
    """leaf_dirs: list of (dir_path, gt_pref_key). Returns rows."""
    rows = []
    for d, gt in leaf_dirs:
        eps = sorted(int(p.stem[7:]) for p in (d / "data").glob("episode*.hdf5"))
        if not eps:
            continue
        rng.shuffle(eps)
        for ep in eps[:n_per_leaf]:
            r = _predict(model, vc, d / "data" / f"episode{ep}.hdf5")
            rows.append((d.name, gt, r["pref_key"], int(r["pref_key"] == gt)))
    return rows


def agg(rows, keys):
    if not rows:
        return {"n": 0}
    out = {"n": len(rows), "acc": round(np.mean([r[3] for r in rows]), 3)}
    per = []
    for k in keys:
        sub = [r for r in rows if r[1] == k]
        a = round(np.mean([r[3] for r in sub]), 3) if sub else None
        out[f"acc_gt_{k}"] = a
        if a is not None:
            per.append(a)
    out["balanced_acc"] = round(float(np.mean(per)), 3) if per else None
    out["pred_dist"] = {k: sum(1 for r in rows if r[2] == k) for k in keys}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)  # .../0526/<cat>
    ap.add_argument("--n_taskA", type=int, default=5)
    ap.add_argument("--n_taskB", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.yaml)
    cfg.framework.qwenvl.base_vlm = "Qwen/Qwen3-VL-4B-Instruct"
    cfg.trainer.pretrained_checkpoint = None  # we load weights manually below
    model = build_framework(cfg).to("cuda").eval()
    sd = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded ckpt: {len(missing)} missing, {len(unexpected)} unexpected keys")

    vc = VQA_CATEGORIES[args.cat]
    pc = PREF_CATEGORIES[args.cat]
    keys = list(vc.pref_keys)
    root = Path(args.data_root)
    rng = random.Random(42)

    # taskA: held-out val episodes per leaf
    split = _split_episodes(root, pc.task_groups, pc.pref_keys, val_fraction=0.2, seed=42)
    dirs = _task_dirs_for_groups(root, pc.task_groups, pc.pref_keys)
    # restrict to val episodes by temporarily filtering in eval_leaves via a val set
    taskA_rows = []
    for task_dir, tg, pk in dirs:
        d = root / task_dir
        val_eps = split[task_dir]["val"]
        rng.shuffle(val_eps)
        for ep in val_eps[:args.n_taskA]:
            r = _predict(model, vc, d / "data" / f"episode{ep}.hdf5")
            taskA_rows.append((task_dir, pk, r["pref_key"], int(r["pref_key"] == pk)))

    # taskB: all leaves under taskB/
    taskB_dirs = []
    tb = root / "taskB"
    if tb.is_dir():
        for d in sorted(tb.iterdir()):
            if d.is_dir() and (d / "data").is_dir():
                gt = d.name.rsplit("_", 1)[-1]
                if gt in keys:
                    taskB_dirs.append((d, gt))
    taskB_rows = eval_leaves(model, vc, taskB_dirs, args.n_taskB, rng)

    res = {"cat": args.cat, "ckpt": args.ckpt,
           "question": vc.question, "answers": dict(vc.answer_text),
           "taskA": agg(taskA_rows, keys), "taskB": agg(taskB_rows, keys)}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
