#!/usr/bin/env python3
"""Vanilla-VLM pseudo-labeler for the Stage-B (VQA=no, cond=yes) cell.

Labels taskB episodes with the UNTRAINED base VLM (no token-VQA, no
geometric tricks) via image-only forced-choice, and writes a pseudo-label
cache in the exact format the Stage-B dataset reads. This is the honest
"off-the-shelf VLM gives the labels" arm of the place 2x2 ablation.

Reuses the scoring from zeroshot_vqa_probe.py so the label rule is byte-for-byte
the same as the probe (calibrated forced-choice argmax by default).

  python pseudo_label_vanilla_vlm.py --cat place --model <path> \
      --root <0526 root> --out <cache.json>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeroshot_vqa_probe import CATS, load_frames, score_answer, MODEL_ID, gt_key

# action_prompt_label must match the main-mode cache so the " Preference: ..."
# suffix string is identical; only the label SOURCE differs (vanilla VLM here).
PROMPT_LABEL = {
    "place":  {"center": "center placement", "corner": "corner placement"},
    "orient": {"0": "side grasp", "90": "top grasp"},
    "height": {"high": "high drop", "low": "low drop"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="place")
    ap.add_argument("--model", default=MODEL_ID, help="HF id or local path to base VLM")
    ap.add_argument("--root", default="/mnt/localssd/kaiwenh/pref/data/0526")
    ap.add_argument("--n", type=int, default=100000, help="max episodes per taskB leaf")
    ap.add_argument("--no_calib", action="store_true", help="use raw (uncalibrated) preds")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except Exception:
        from transformers import AutoModelForImageTextToText as ModelCls
    print(f"loading {args.model} ...", flush=True)
    model = ModelCls.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                     attn_implementation="sdpa").eval().to("cuda:0")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model)

    cfg = CATS[args.cat]
    cand = list(cfg["cand"].keys())
    from PIL import Image
    black = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in cfg["fracs"]]
    prior = {k: score_answer(model, processor, black, cfg["question"], cfg["cand"][k]) for k in cand}
    plab = PROMPT_LABEL[args.cat]
    print(f"black-img prior: { {k: round(prior[k],3) for k in cand} }  calib={not args.no_calib}")

    root = Path(args.root) / args.cat
    cache, n_match, n_tot, dist = {}, 0, 0, {k: 0 for k in cand}
    for leaf in cfg["taskB"]:
        gk = gt_key(leaf)
        ddir = root / leaf / "data"
        if not ddir.is_dir():
            print(f"  skip {leaf}: no data dir"); continue
        eps = sorted(int(p.stem[7:]) for p in ddir.glob("episode*.hdf5"))[:args.n]
        leaf_base = leaf.split("/")[-1]
        for ep in eps:
            imgs = load_frames(ddir / f"episode{ep}.hdf5", cfg["fracs"])
            sc = {k: score_answer(model, processor, imgs, cfg["question"], cfg["cand"][k]) for k in cand}
            if not args.no_calib:
                sc = {k: sc[k] - prior[k] for k in cand}
            pred = max(sc, key=sc.get)
            cache[f"{leaf_base}/episode{ep}"] = {
                "action_prompt_label": plab[pred],
                "pref_key": pred,
                "decision": "keep",
                "gt_pref_key": gk,                 # IGNORED by dataset (whitelist); kept for audit
                "gt_match": bool(pred == gk),
                "labeler": "vanilla_vlm" + ("" if args.no_calib else "_calib"),
            }
            n_match += int(pred == gk); n_tot += 1; dist[pred] += 1
    Path(args.out).write_text(json.dumps(cache, indent=2))
    print(f"\nwrote {args.out}\n  n={n_tot}  acc-vs-GT={n_match / max(n_tot,1):.3f}  pred_dist={dist}")


if __name__ == "__main__":
    main()
