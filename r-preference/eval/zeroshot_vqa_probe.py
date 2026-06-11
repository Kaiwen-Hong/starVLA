#!/usr/bin/env python3
"""Zero-shot base-VLM forced-choice preference accuracy probe.

Tests whether the UNTRAINED Qwen3-VL-4B-Instruct can answer the (redesigned,
task-context) preference question from a small head-camera clip, on taskA
(training-distribution leaves) + taskB (unseen-object leaves), for the 0526 data.

Per category: question carries task context; two candidate answers are scored by
length-normalized teacher-forced log-likelihood; argmax = predicted pref. Reports
overall / per-split / per-GT-class accuracy + prediction distribution (to catch
a model that just always picks one class).

Image-only (no robot state) — the honest "frames+question" test.
"""
import argparse, io, json, time
from pathlib import Path
import numpy as np, h5py, torch
from PIL import Image
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

CATS = {
    "orient": dict(
        fracs=[0.40, 0.60, 0.80],
        question="We are picking up an object. Are we grasping it from the side (horizontal grasp) or from the top (vertical grasp)?",
        cand={"0": "From the side.", "90": "From the top."},
        taskA=[f"place_{o}_{w}_{p}" for o in ["bottle", "callbell", "can", "chipstub"]
               for w in ["box", "left"] for p in ["0", "90"]],
        taskB=["taskB/move_can5_away_0", "taskB/move_can5_away_90"],
    ),
    "place": dict(
        fracs=[0.60, 0.80, 1.00],
        question="We are placing an object onto a target. Are we placing it in the middle or on the corner?",
        cand={"center": "In the middle.", "corner": "On the corner."},
        taskA=[f"move_{o}_pad_{p}" for o in ["mouse", "pillbottle", "playingcards", "soap"] for p in ["center", "corner"]]
              + [f"place_{o}_tray_{p}" for o in ["mouse", "pillbottle", "playingcards", "soap"] for p in ["center", "corner"]],
        taskB=["taskB/place_soap2_stand_center", "taskB/place_soap2_stand_corner"],
    ),
    "height": dict(
        fracs=[0.60, 0.80, 1.00],
        question="We are placing an object onto a target. Are we dropping it from a high position or from a low position?",
        cand={"high": "From a high position.", "low": "From a low position."},
        taskA=[f"move_{o}_pad_{p}" for o in ["mouse", "pillbottle", "playingcards", "soap"] for p in ["high", "low"]]
              + [f"place_{o}_stand_{p}" for o in ["mouse", "pillbottle", "playingcards", "soap"] for p in ["high", "low"]],
        taskB=["taskB/place_playingcards1_box_high", "taskB/place_playingcards1_box_low"],
    ),
}


def gt_key(leaf):  # pref key = last underscore field of the leaf basename
    return leaf.split("/")[-1].rsplit("_", 1)[-1]


def load_frames(h5_path, fracs, size=(224, 224)):
    with h5py.File(h5_path, "r") as h5:
        ds = h5["observation/head_camera/rgb"]
        T = ds.shape[0]
        idx = [int(round(f * (T - 1))) for f in fracs]
        out = []
        for t in idx:
            raw = ds[t]
            data = raw.tobytes() if hasattr(raw, "tobytes") else raw
            im = Image.open(io.BytesIO(data)).convert("RGB").resize((size[1], size[0]))
            out.append(im)
    return out


@torch.inference_mode()
def score_answer(model, processor, images, question, answer):
    """length-normalized log-prob of `answer` given images+question (teacher forced)."""
    device = model.device
    user = {"role": "user", "content": [{"type": "image", "image": im} for im in images]
            + [{"type": "text", "text": question}]}
    asst = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    prompt = processor.apply_chat_template([[user]], tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt")
    full = processor.apply_chat_template([[user, asst]], tokenize=True, add_generation_prompt=False,
                                         return_dict=True, return_tensors="pt")
    P = int(prompt["attention_mask"].sum().item())
    Lf = int(full["attention_mask"].sum().item())
    ans_lo, ans_hi = P, Lf - 2  # drop trailing im_end + \n
    if ans_hi <= ans_lo:
        ans_hi = Lf  # fallback (very short)
    full = {k: v.to(device) for k, v in full.items()}
    out = model(**full)
    logits = out.logits[0].float()  # (S, V)
    ids = full["input_ids"][0]
    lps = []
    for j in range(ans_lo, ans_hi):
        lp = torch.log_softmax(logits[j - 1], dim=-1)[ids[j]]
        lps.append(lp.item())
    return float(np.mean(lps)) if lps else -1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="orient,place,height")
    ap.add_argument("--root", default="/mnt/localssd/kaiwenh/pref/data/0526")
    ap.add_argument("--n_taskA", type=int, default=3, help="episodes per taskA leaf")
    ap.add_argument("--n_taskB", type=int, default=25, help="episodes per taskB leaf")
    ap.add_argument("--out", default="/home/kaiwenh/starVLA/r-preference/eval/zeroshot_vqa_probe.json")
    ap.add_argument("--model", default=MODEL_ID, help="HF id or local path to the base VLM")
    ap.add_argument("--smoke", type=int, default=0, help="if >0, only this many eps per leaf, 1 cat")
    args = ap.parse_args()

    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except Exception:
        from transformers import AutoModelForImageTextToText as ModelCls
    print(f"loading {args.model} ...", flush=True)
    model = ModelCls.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                     attn_implementation="sdpa").eval().to("cuda:0")
    processor = AutoProcessor.from_pretrained(args.model)

    black = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(3)]

    def agg_for(rows, predfield, cand_keys):
        def acc(sub):
            return round(float(np.mean([r["correct" if predfield == "pred" else f"correct_{predfield}"] for r in sub])), 3) if sub else None
        out = {}
        for sp in ("taskA", "taskB"):
            ss = [r for r in rows if r["split"] == sp]
            out[f"{sp}_acc"] = acc(ss)
            out[f"{sp}_n"] = len(ss)
            per_class = []
            for k in cand_keys:
                gg = [r for r in ss if r["gt"] == k]
                a = acc(gg)
                out[f"{sp}_acc_gt_{k}"] = a
                if a is not None:
                    per_class.append(a)
            out[f"{sp}_balanced_acc"] = round(float(np.mean(per_class)), 3) if per_class else None
            preds = [r[predfield] for r in ss]
            out[f"{sp}_pred_dist"] = {k: preds.count(k) for k in cand_keys}
        return out

    results = {}
    for cat in args.cats.split(","):
        cfg = CATS[cat]
        root = Path(args.root) / cat
        cand_keys = list(cfg["cand"].keys())
        # answer-phrase prior (contextual calibration): score each candidate on a
        # content-free (black) image so we can subtract the model's answer bias.
        prior = {k: score_answer(model, processor, black, cfg["question"], cfg["cand"][k])
                 for k in cand_keys}
        rows = []
        for split, leaves, n in [("taskA", cfg["taskA"], args.n_taskA),
                                 ("taskB", cfg["taskB"], args.n_taskB)]:
            if args.smoke:
                n = args.smoke
            for leaf in leaves:
                gk = gt_key(leaf)
                ddir = root / leaf / "data"
                if not ddir.is_dir():
                    continue
                eps = sorted(int(p.stem[7:]) for p in ddir.glob("episode*.hdf5"))[:n]
                for ep in eps:
                    imgs = load_frames(ddir / f"episode{ep}.hdf5", cfg["fracs"])
                    scores = {k: score_answer(model, processor, imgs, cfg["question"], cfg["cand"][k])
                              for k in cand_keys}
                    pred = max(scores, key=scores.get)
                    cal = {k: scores[k] - prior[k] for k in cand_keys}
                    pred_cal = max(cal, key=cal.get)
                    rows.append(dict(split=split, leaf=leaf.split("/")[-1], gt=gk,
                                     pred=pred, correct=int(pred == gk),
                                     pred_cal=pred_cal, correct_pred_cal=int(pred_cal == gk),
                                     scores={k: round(scores[k], 3) for k in cand_keys}))
        raw = agg_for(rows, "pred", cand_keys)
        calagg = agg_for(rows, "pred_cal", cand_keys)
        results[cat] = {"raw": raw, "calibrated": calagg, "prior": {k: round(prior[k], 3) for k in cand_keys},
                        "question": cfg["question"], "cand": cfg["cand"], "rows": rows}
        print(f"\n=== {cat} ===  prior(black-img logprob): {results[cat]['prior']}")
        print("  RAW       :", {k: raw[k] for k in ("taskA_balanced_acc","taskB_balanced_acc","taskA_pred_dist","taskB_pred_dist")})
        print("  CALIBRATED:", {k: calagg[k] for k in ("taskA_balanced_acc","taskB_balanced_acc","taskA_pred_dist","taskB_pred_dist")})

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
