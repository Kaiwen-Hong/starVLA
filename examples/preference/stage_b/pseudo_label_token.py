"""Stage B offline pseudo-labeling with the TOKEN-VQA labeler (OFT backbone).

Replaces pseudo_label_offline.py (which used QwenPI_VQA + head-cam clip strategies).
Here the frozen Stage-A token-VQA reads the active-arm EE STATE token (state_mode=token)
to label each taskB episode's preference. Much more accurate than the old head-cam VQA
(gate-eval 0.98-1.00). Output cache schema is identical to what PrefHDF5StageBDataset
consumes (whitelist: action_prompt_label / decision / pref_key).

CLI:
  python -m examples.preference.stage_b.pseudo_label_token \
    --cat height \
    --yaml examples/preference/train_files/starvla_pref_stage_a_oftvqa_token_height.yaml \
    --ckpt <token Stage-A ckpt steps_10000_pytorch_model.pt> \
    --taskB_root /mnt/localssd/kaiwenh/pref/data/0526/height/taskB \
    --out r-preference/eval/pref_pseudo_labels_height_B.json
"""
from __future__ import annotations
import argparse, json, math, time
from collections import defaultdict
from pathlib import Path
import torch
from omegaconf import OmegaConf

from starVLA.model.framework import build_framework
from examples.preference.dataset.prompt import PREF_CATEGORIES
from examples.preference.dataset.vqa_sample import VQA_CATEGORIES, load_clip_and_state

VLM_LOCAL = "/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--taskB_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_conf", type=float, default=0.55,
                    help="keep iff max(p_A,p_B) >= this (token-VQA is very confident; "
                         "this only drops true coin-flips). Launch gate enforces acc.")
    ap.add_argument("--launch_gate_min_acc", type=float, default=0.95)
    args = ap.parse_args()

    pc = PREF_CATEGORIES[args.cat]
    vc = VQA_CATEGORIES[args.cat]
    pref_labels = pc.pref_labels                      # pk -> action prompt label
    print(f"[label] cat={args.cat} pref_keys={pc.pref_keys} labels={pref_labels}")

    # --- build frozen token-VQA labeler ---
    cfg = OmegaConf.load(args.yaml)
    cfg.framework.qwenvl.base_vlm = VLM_LOCAL
    cfg.trainer.pretrained_checkpoint = None
    t0 = time.time()
    model = build_framework(cfg).to("cuda").eval()
    sd = torch.load(args.ckpt, map_location="cpu")
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[label] loaded ckpt in {time.time()-t0:.1f}s ({len(miss)} missing, {len(unexp)} unexpected)")
    assert any("vqa_state_proj" in k for k in sd), "ckpt has no vqa_state_proj — not a token-VQA ckpt!"

    taskB = Path(args.taskB_root)
    cache = {}
    t0 = time.time()
    for pk in pc.pref_keys:
        for d in sorted(taskB.glob(f"*_{pk}")):
            if not (d / "data").is_dir() or d.name.endswith(".zip"):
                continue
            tg = d.name.rsplit("_", 1)[0]
            for h5p in sorted((d / "data").glob("episode*.hdf5")):
                frames, state, _ = load_clip_and_state(
                    h5p, strategy=vc.clip_strategy, n=vc.n_frames, cameras=vc.cameras, jitter=0)
                r = model.predict_preference(frames, state=state if vc.state_in_vqa else None)
                pA, pB = r["p_A"], r["p_B"]
                conf = max(pA, pB)
                gap = math.log(max(pA, 1e-9)) - math.log(max(pB, 1e-9))
                pred_pk = r["pref_key"]
                decision = "keep" if conf >= args.min_conf else f"reject:low_conf_{conf:.3f}"
                cache[f"{d.name}/{h5p.stem}"] = {
                    "action_prompt_label": pref_labels[pred_pk],
                    "pref_key": pred_pk,
                    "decision": decision,
                    "gt_pref_key": pk,
                    "gt_match": (pred_pk == pk),
                    "conf": round(conf, 4), "logit_gap": round(gap, 3),
                }
    n = len(cache)
    print(f"[label] {n} eps in {time.time()-t0:.1f}s")

    kept = [v for v in cache.values() if v["decision"] == "keep"]
    kept_correct = sum(v["gt_match"] for v in kept)
    overall_correct = sum(v["gt_match"] for v in cache.values())
    per_pseudo = defaultdict(int)
    for v in kept:
        per_pseudo[v["pref_key"]] += 1
    print("=" * 70)
    print(f"  total={n}  kept={len(kept)}  rejected={n-len(kept)}")
    print(f"  overall pseudo-vs-GT acc: {overall_correct}/{n} = {overall_correct/max(1,n):.3f}")
    kept_acc = kept_correct / max(1, len(kept))
    print(f"  POST-FILTER acc:          {kept_correct}/{len(kept)} = {kept_acc:.3f}  (gate >= {args.launch_gate_min_acc})")
    print(f"  kept per pseudo-class:    {dict(per_pseudo)}")
    for k, v in cache.items():
        if v["decision"] != "keep" or not v["gt_match"]:
            print(f"    flag {k}: pred={v['pref_key']} gt={v['gt_pref_key']} conf={v['conf']} dec={v['decision']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cache, indent=2))
    print(f"[label] wrote {args.out}")
    if kept_acc < args.launch_gate_min_acc:
        print(f"!!! LAUNCH GATE FAILED ({kept_acc:.3f} < {args.launch_gate_min_acc}) — do not train.")
        raise SystemExit(2)
    print(f"  ✓ LAUNCH GATE PASSED")


if __name__ == "__main__":
    main()
