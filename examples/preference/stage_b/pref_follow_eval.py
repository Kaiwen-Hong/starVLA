"""Stage-B controllability eval — FK-free, action-space-agnostic.

For each taskB episode (GT pref known, eval-only), at a pref-discriminative frame:
  a_hi = predict(base + " Preference: <high-label>")
  a_lo = predict(base + " Preference: <low-label>")
  demo = the episode's actual (normalized) action chunk at that frame
Metrics:
  - follows-correctly: mean|a_GTpref - demo| < mean|a_flip - demo|  (direction-aware)
  - effect size: mean|a_hi - a_lo|  (does the policy respond to the pref at all)
A controllable main policy -> follow-acc >> 0.5 AND effect >> 0.
B0 control (trained without suffix) -> effect ~ 0.

CLI:
  python -m examples.preference.stage_b.pref_follow_eval --cat height \
    --yaml examples/preference/train_files/starvla_pref_stageb_main_height.yaml \
    --ckpt results/Checkpoints/pref_stageb_main_height/checkpoints/steps_1500_pytorch_model.pt \
    --taskB_root /mnt/localssd/kaiwenh/pref/data/0526/height/taskB --tag main_height
"""
from __future__ import annotations
import argparse, io, json
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image
from omegaconf import OmegaConf

from starVLA.model.framework import build_framework
from examples.preference.dataset.prompt import PREF_CATEGORIES

VLM_LOCAL = "/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct"
# pref-discriminative episode fraction per cat (where the chunk should be centered)
DISC_FRAC = {"height": 0.85, "place": 0.85, "orient": 0.45, "contact": 0.45, "hvlv": 0.6}
TASKB_GROUP = {"height": "place_playingcards1_box", "orient": "move_can5_away",
               "contact": "put_boxdrink3_plate", "place": "place_soap2_stand", "hvlv": "stamp_seal6"}
CAMS = ("head_camera", "left_camera", "right_camera")


def load_imgs(h5, t):
    out = []
    for cam in CAMS:
        raw = h5[f"observation/{cam}/rgb"][t]
        data = raw.tobytes() if hasattr(raw, "tobytes") else raw
        out.append(Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--taskB_root", required=True)
    ap.add_argument("--n_per_class", type=int, default=12)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pc = PREF_CATEGORIES[args.cat]
    pkA, pkB = pc.pref_keys
    labels = pc.pref_labels
    base = pc.clean_templates[TASKB_GROUP[args.cat]]
    cfg = OmegaConf.load(args.yaml)
    cfg.framework.qwenvl.base_vlm = VLM_LOCAL
    cfg.trainer.pretrained_checkpoint = None
    chunk = int(cfg.framework.action_model.action_horizon)
    stats = json.loads(Path(cfg.datasets.vla_data.stats_json_path).read_text())["action"]
    q01 = np.asarray(stats["q01"], np.float32); q99 = np.asarray(stats["q99"], np.float32)
    span = np.where((q99 - q01) != 0, q99 - q01, 1.0)

    model = build_framework(cfg).to("cuda").eval()
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    pl, ph = f"{base} Preference: {labels[pkA]}", f"{base} Preference: {labels[pkB]}"
    print(f"[ctrl] cat={args.cat} chunk={chunk}\n  A({pkA}): {pl!r}\n  B({pkB}): {ph!r}")

    rows = []
    root = Path(args.taskB_root)
    for gt in (pkA, pkB):
        d = next(root.glob(f"*_{gt}"))
        eps = sorted((d / "data").glob("episode*.hdf5"))[:args.n_per_class]
        for h5p in eps:
            with h5py.File(h5p, "r") as h5:
                T = h5["observation/head_camera/rgb"].shape[0]
                fi = int(np.clip(int(DISC_FRAC[args.cat] * T) - chunk // 2, 0, max(0, T - 2)))
                imgs = load_imgs(h5, fi)
                act = np.asarray(h5["joint_action/vector"][fi:fi + chunk], np.float32)
            demo = np.clip(2 * (act - q01) / span - 1, -1, 1)
            L = demo.shape[0]
            aA = model.predict_action([{"image": imgs, "lang": pl}])["normalized_actions"][0][:L]
            aB = model.predict_action([{"image": imgs, "lang": ph}])["normalized_actions"][0][:L]
            a_correct, a_flip = (aA, aB) if gt == pkA else (aB, aA)
            d_corr = float(np.mean(np.abs(a_correct - demo)))
            d_flip = float(np.mean(np.abs(a_flip - demo)))
            rows.append({"gt": gt, "ep": h5p.stem, "frame": fi,
                         "follows": int(d_corr < d_flip),
                         "L1_correct": round(d_corr, 4), "L1_flip": round(d_flip, 4),
                         "effect": round(float(np.mean(np.abs(aA - aB))), 4)})

    acc = float(np.mean([r["follows"] for r in rows]))
    eff = float(np.mean([r["effect"] for r in rows]))
    per = {gt: round(float(np.mean([r["follows"] for r in rows if r["gt"] == gt])), 3) for gt in (pkA, pkB)}
    print("=" * 64)
    print(f"[{args.tag or args.cat}]  follow-acc={acc:.3f}  effect(|a_hi-a_lo|)={eff:.4f}  per_gt={per}  n={len(rows)}")
    print(f"  (follow-acc>>0.5 + effect>>0 = controllable; effect~0 = ignores pref)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"cat": args.cat, "tag": args.tag, "acc": acc,
                                              "effect": eff, "per_gt": per, "rows": rows}, indent=2))
        print(f"[ctrl] wrote {args.out}")


if __name__ == "__main__":
    main()
