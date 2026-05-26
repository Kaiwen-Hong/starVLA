"""
Stage B offline pseudo-labeling — generate the pref-pseudo-label cache
that PrefHDF5StageBDataset consumes.

Per Stage B doc §3 (`r-preference/doc/0524-stageB-contact-plan.md`):
  - Load frozen VQA-Stage-A 35k ckpt (this is the labeler — independent
    of the trainable policy in §4).
  - For each taskB episode:
      * Run predict_preference on mid_8 clip (PRIMARY)
      * Run predict_preference on gripper_anchored clip (SECOND OPINION)
  - Apply default filter: keep iff mid_8 |logit_gap| >= 10.0
      (Q-A validated; bimodal empty-band center; margin ~10 to nearest wrong)
  - Write cache JSON with rich per-episode records.
  - Launch-gate verification: compare pseudo-labels vs dir-derived GT
    (ONLY here — the dataset reads cache, never GT) and abort if
    pseudo-label acc < 0.95.

CLI:
  python -m examples.preference.stage_b.pseudo_label_offline \
      --category contact \
      --vqa_yaml  examples/preference/train_files/starvla_pref_stage_a_vqa_contact.yaml \
      --vqa_ckpt  /mnt/.../pref_main_stage_a_v1_VQA_contact/checkpoints/steps_35000_pytorch_model.pt \
      --taskB_data_root /mnt/localssd/kaiwenh/pref/data/contact/taskB \
      --task_groups put_boxdrink3_plate \
      --filter_threshold 10.0 \
      --out r-preference/eval/pref_pseudo_labels_contact_B.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from examples.preference.dataset.prompt import PREF_CATEGORIES
from examples.preference.dataset.vqa_sample import (
    VQA_CATEGORIES,
    CLIP_STRATEGY_NAMES,
    load_clip_by_strategy,
)
from examples.preference.eval.stage_a_gate import build_framework, load_ckpt_into_model
from examples.preference.eval.diagnostic import predict_with_logits


def enumerate_taskB_episodes(
    taskB_root: Path,
    task_groups: List[str],
    pref_keys: Tuple[str, str],
) -> List[Tuple[str, str, str, int]]:
    """Enumerate (task_dir, task_group, pref_key, ep_id) for taskB.

    NB: pref_key here is the DIR-PARSED key (used ONLY by this script for
    the §3.2 launch-gate verification, NEVER by the dataset). The cache
    JSON we write will store it as `gt_pref_key`; PrefHDF5StageBDataset's
    cache loader whitelists it OUT.
    """
    out = []
    for tg in task_groups:
        for pk in pref_keys:
            name = f"{tg}_{pk}"
            d = taskB_root / name / "data"
            if not d.is_dir():
                print(f"  [skip] no data dir at {d}")
                continue
            for h5p in sorted(d.glob("episode*.hdf5")):
                ep_id = int(h5p.stem.replace("episode", ""))
                out.append((name, tg, pk, ep_id))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, choices=list(PREF_CATEGORIES))
    ap.add_argument("--vqa_yaml", required=True)
    ap.add_argument("--vqa_ckpt", required=True)
    ap.add_argument("--taskB_data_root", required=True)
    ap.add_argument("--task_groups", nargs="+", required=True,
                    help="Task groups under taskB (e.g. put_boxdrink3_plate)")
    ap.add_argument("--filter_threshold", type=float, default=10.0,
                    help="mid_8 |logit_gap| threshold for keep decision. "
                         "Default 10.0 = Q-A validated bimodal empty-band center.")
    ap.add_argument("--launch_gate_min_acc", type=float, default=0.95,
                    help="If pseudo-labels (post-filter) match GT < this fraction, abort.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cat = PREF_CATEGORIES[args.category]
    vqa_cat = VQA_CATEGORIES[args.category]
    pk_A, pk_B = vqa_cat.pref_keys
    pref_labels = cat.pref_labels  # pk → action prompt label e.g. "low contact"

    print(f"[label] category={args.category}  pk_A={pk_A}({pref_labels[pk_A]!r}) "
          f"pk_B={pk_B}({pref_labels[pk_B]!r})")
    print(f"[label] filter_threshold (|mid_8 logit_gap|) = {args.filter_threshold}")
    print(f"[label] launch_gate_min_acc = {args.launch_gate_min_acc}")

    # === Enumerate taskB ===
    episodes = enumerate_taskB_episodes(
        Path(args.taskB_data_root), args.task_groups, cat.pref_keys,
    )
    print(f"[label] enumerated {len(episodes)} taskB episodes "
          f"across {len(args.task_groups)} task_group(s)")
    # Per-class GT count (for §3.2 verification reporting later)
    gt_counts = defaultdict(int)
    for _, tg, gt_pk, _ in episodes:
        gt_counts[(tg, gt_pk)] += 1
    print(f"[label] per (task_group, GT pref_key) count: {dict(gt_counts)}")

    # === Load frozen VQA-Stage-A ===
    print(f"\n[label] loading frozen VQA-Stage-A from {args.vqa_ckpt}")
    t0 = time.time()
    model, _ = build_framework(args.vqa_yaml, "QwenPI_VQA")
    load_ckpt_into_model(model, args.vqa_ckpt)
    model = model.cuda().eval()
    print(f"[label] model loaded in {time.time()-t0:.1f}s\n")

    # === Run both strategies, build cache ===
    cache: Dict[str, dict] = {}
    t0 = time.time()
    for i, (task_dir, tg, gt_pk, ep_id) in enumerate(episodes):
        h5p = Path(args.taskB_data_root) / task_dir / "data" / f"episode{ep_id}.hdf5"

        # mid_8 — pass per-cat cameras so multi-cam-trained VQA (e.g. place=
        # head+active_wrist) sees the same spatial input at label-time as at train-time.
        clip_mid, info_mid = load_clip_by_strategy(h5p, strategy="mid_8", cameras=vqa_cat.cameras)
        out_mid = predict_with_logits(model, clip_mid)
        # gripper_anchored — second opinion, same cameras
        clip_grp, info_grp = load_clip_by_strategy(h5p, strategy="gripper_anchored", cameras=vqa_cat.cameras)
        out_grp = predict_with_logits(model, clip_grp)

        # Filter decision (default: mid_8 |gap| >= threshold)
        mid_gap_abs = abs(out_mid["logit_gap"])
        decision = "keep" if mid_gap_abs >= args.filter_threshold else f"reject:low_gap_{mid_gap_abs:.2f}"

        # Pseudo-label = mid_8's prediction (primary strategy)
        pseudo_pk = out_mid["pred_pref_key"]
        action_prompt_label = pref_labels[pseudo_pk]

        # Build cache entry (rich schema — but PrefHDF5StageBDataset
        # whitelist-loads only action_prompt_label + decision + pref_key)
        ep_key = f"{task_dir}/episode{ep_id}"
        cache[ep_key] = {
            # === DATASET-READABLE (whitelist) ===
            "action_prompt_label": action_prompt_label,
            "pref_key": pseudo_pk,
            "decision": decision,
            # === DATASET-FORBIDDEN (filtered out by cache loader whitelist) ===
            # ONLY for §3.2 launch-gate verification + future §7 deferred eval
            "gt_pref_key": gt_pk,
            "gt_match": (pseudo_pk == gt_pk),
            # === DEBUG / ANALYSIS ===
            "mid_8": {
                "pred": out_mid["pred_pref_key"],
                "logit_gap": out_mid["logit_gap"],
                "A_logit": out_mid["A_logit"],
                "B_logit": out_mid["B_logit"],
                "p_A_pair": out_mid["p_A_pair"],
                "p_B_pair": out_mid["p_B_pair"],
                "indices": info_mid["indices"],
                "T": info_mid["T"],
            },
            "gripper_anchored": {
                "pred": out_grp["pred_pref_key"],
                "logit_gap": out_grp["logit_gap"],
                "A_logit": out_grp["A_logit"],
                "B_logit": out_grp["B_logit"],
                "p_A_pair": out_grp["p_A_pair"],
                "p_B_pair": out_grp["p_B_pair"],
                "indices": info_grp["indices"],
                "T": info_grp["T"],
                "grasp_t": info_grp["grasp_t"],
            },
        }

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(episodes)} ({time.time()-t0:.1f}s)")

    print(f"\n[label] done all {len(episodes)} eps in {time.time()-t0:.1f}s")

    # === Filter + balance + launch-gate stats ===
    kept = [(k, v) for k, v in cache.items() if v["decision"] == "keep"]
    rejected = [(k, v) for k, v in cache.items() if v["decision"] != "keep"]
    kept_correct = sum(1 for k, v in kept if v["gt_match"])
    overall_correct = sum(1 for k, v in cache.items() if v["gt_match"])

    # Per-class (GT-derived, for verification reporting)
    kept_per_gt = defaultdict(int)
    for k, v in kept:
        kept_per_gt[v["gt_pref_key"]] += 1
    # Per-class (pseudo, what dataset will see)
    kept_per_pseudo = defaultdict(int)
    for k, v in kept:
        kept_per_pseudo[v["pref_key"]] += 1

    print("\n" + "=" * 80)
    print(f"FILTER + LAUNCH-GATE SUMMARY")
    print("=" * 80)
    print(f"  Total episodes:          {len(cache)}")
    print(f"  Kept (passed filter):    {len(kept)}  ({len(kept)/max(1,len(cache)):.1%})")
    print(f"  Rejected:                {len(rejected)}")
    for k, v in rejected:
        print(f"    - {k}  GT={v['gt_pref_key']}  pseudo={v['pref_key']}  "
              f"mid_8_gap={v['mid_8']['logit_gap']:+.2f}  reason={v['decision']}")
    print(f"")
    print(f"  Overall (unfiltered) pseudo-vs-GT acc: {overall_correct}/{len(cache)} = "
          f"{overall_correct/max(1,len(cache)):.3f}")
    print(f"  POST-FILTER (kept) pseudo-vs-GT acc:   {kept_correct}/{len(kept)} = "
          f"{kept_correct/max(1,len(kept)):.3f}  ← § 3.2 launch gate target")
    print(f"")
    print(f"  Per-class kept count (by GT):    {dict(kept_per_gt)}")
    print(f"  Per-class kept count (by pseudo): {dict(kept_per_pseudo)}  ← what dataset sees")
    if len(kept_per_pseudo) == 2:
        cnts = list(kept_per_pseudo.values())
        delta = abs(cnts[0] - cnts[1]) / max(cnts)
        print(f"  Per-class imbalance Δ (pseudo): {delta:.1%}  "
              f"({'< 15% — no fallback needed' if delta < 0.15 else '>= 15% — TRIGGER FALLBACK per §4.5'})")

    # === Save cache ===
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cache, indent=2))
    print(f"\n[label] saved cache to {out_path}  ({len(cache)} entries)")

    # === Launch gate ===
    kept_acc = kept_correct / max(1, len(kept))
    if kept_acc < args.launch_gate_min_acc:
        print(f"\n!!! LAUNCH GATE FAILED !!!")
        print(f"  post-filter acc {kept_acc:.3f} < required {args.launch_gate_min_acc}")
        print(f"  STOP. Do NOT proceed to Stage B training.")
        print(f"  Diagnose: (1) check mid_8 |gap| dist via diagnostic.py")
        print(f"            (2) consider raising filter_threshold")
        print(f"            (3) check if labeler ckpt is correct + step-matched")
        import sys
        sys.exit(2)
    else:
        print(f"\n  ✓ LAUNCH GATE PASSED  (acc {kept_acc:.3f} >= {args.launch_gate_min_acc})")
        print(f"  Stage B training can proceed.")


if __name__ == "__main__":
    main()
