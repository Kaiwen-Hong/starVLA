"""PRIVILEGE-FREE EE-only pseudo-labeler for the contact axis (grasp height on object).

Replaces the sim-privileged `height_fraction` (scene_info) feature of
pseudo_label_geom.py with a feature a REAL robot owns outright:

    feature = EE z of the first-closing arm at the first gripper-close frame
              (endpose/{left,right}_endpose + endpose/{left,right}_gripper ONLY)

Why it transfers: within a target scene (single object type standing on a fixed
table) grasp-25% vs grasp-75% differ by ~half the object height in absolute EE z
(taskB measured: 25-class z 0.820+-0.015 vs 75-class 0.904+-0.011, min-gap +1.5cm).

Labeling rule on the UNLABELED target pool (no GT anywhere):
  1. DIRECTION is learned on taskA (source, fully labeled - privilege there is
     legitimate): higher z = pref "75". Verified: taskA pooled mean z(25)=0.844 <
     z(75)=0.877, and 1.000 per-group split acc on every upright-object group.
  2. SPLIT on the target pool is UNSUPERVISED (default exact 1-D 2-means;
     --split otsu|median variants). Transductive: uses only the pool's own
     feature histogram, never a target label.
  3. REJECT band: episodes within --margin (default 8mm, ~0.75x the taskA
     within-class sigma) of the split threshold get decision:"reject" so the
     Stage-B firewall drops them (post-filter accuracy >> coverage).

Known limitation (measured on taskA, documented in 0612-privilege-free-labelers.md):
the z-feature carries no signal for objects LYING FLAT (fork/screwdriver: grasp-z
identical for 25/75 because the contact fraction runs along a horizontal axis).
Upright objects (boxdrink/callbell + the taskB boxdrink3) are the valid regime.

FIREWALL: on the target (taskB) this labeler reads ONLY hdf5 endpose/gripper
streams. scene_info is NEVER opened for taskB. The dir-name pref suffix is read
ONLY into gt_* audit fields for SCORING the finished predictions (same protocol
as pseudo_label_geom.py); it never influences feature, split, or direction.

CLI:
  python -m examples.preference.stage_b.pseudo_label_ee --cat contact \
      --out r-preference/eval/pref_pseudo_labels_contact_B_ee.json
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import Counter

import h5py
import numpy as np

from examples.preference.dataset.prompt import PREF_CATEGORIES

ROOT = "/mnt/localssd/kaiwenh/pref/data/0526"
GRIP_THRESH = 0.5  # contact gripper convention: 1.0=open, 0.0=closed (vqa_sample.py)


# ------------------------------------------------------------------
# Feature (privilege-free: proprioception only)
# ------------------------------------------------------------------

def feat_grasp_z(ep_path: str):
    """EE z of the arm whose gripper closes FIRST (open->closed crossing 0.5)."""
    with h5py.File(ep_path, "r") as h:
        L = np.asarray(h["endpose/left_gripper"][:])
        R = np.asarray(h["endpose/right_gripper"][:])
        T = len(L)
        gt, arm = None, None
        for t in range(1, T):
            if L[t] < GRIP_THRESH and L[t - 1] >= GRIP_THRESH:
                gt, arm = t, "left"
                break
            if R[t] < GRIP_THRESH and R[t - 1] >= GRIP_THRESH:
                gt, arm = t, "right"
                break
        if gt is None:  # started closed: first below-threshold frame
            w = np.where((L < GRIP_THRESH) | (R < GRIP_THRESH))[0]
            if len(w) == 0:
                return None
            gt = int(w[0])
            arm = "left" if L[gt] < GRIP_THRESH else "right"
        return float(h[f"endpose/{arm}_endpose"][gt][2])


# ------------------------------------------------------------------
# Unsupervised 1-D 2-splits
# ------------------------------------------------------------------

def split_2means(x: np.ndarray) -> float:
    """Exact 1-D 2-means: scan sorted boundaries, min within-class sum of squares."""
    xs = np.sort(x)
    n = len(xs)
    csum = np.cumsum(xs)
    tot = csum[-1]
    best, bi = None, 1
    for i in range(1, n):
        m1 = csum[i - 1] / i
        m2 = (tot - csum[i - 1]) / (n - i)
        ss = np.sum((xs[:i] - m1) ** 2) + np.sum((xs[i:] - m2) ** 2)
        if best is None or ss < best:
            best, bi = ss, i
    return 0.5 * (xs[bi - 1] + xs[bi])


def split_otsu(x: np.ndarray) -> float:
    xs = np.sort(x)
    n = len(xs)
    best, thr = -1.0, float(np.median(xs))
    for i in range(1, n):
        v = (i / n) * (1 - i / n) * (xs[:i].mean() - xs[i:].mean()) ** 2
        if v > best:
            best, thr = v, 0.5 * (xs[i - 1] + xs[i])
    return thr


SPLITS = {"2means": split_2means, "otsu": split_otsu,
          "median": lambda x: float(np.median(x))}


# ------------------------------------------------------------------
# Leaf enumeration (same convention as pseudo_label_geom.py)
# ------------------------------------------------------------------

def leaves(cat: str, split: str):
    pc = PREF_CATEGORIES[cat]
    root = os.path.join(ROOT, cat)
    out = []
    if split == "taskA":
        for tg in pc.task_groups:
            for pk in pc.pref_keys:
                d = os.path.join(root, f"{tg}_{pk}")
                if os.path.isdir(d):
                    out.append((pk, d))
    else:
        for d in sorted(glob.glob(os.path.join(root, "taskB", "*"))):
            if os.path.isdir(d) and not d.endswith(".zip"):
                # dir-name suffix = GT, used ONLY for scoring (audit fields)
                out.append((os.path.basename(d).rsplit("_", 1)[-1], d))
    return out


def gather(cat: str, split: str, n=None):
    rows = []  # (leafname, pref_key_or_gt, ep_idx, feature)
    for pk, d in leaves(cat, split):
        eps = sorted(glob.glob(os.path.join(d, "data", "episode*.hdf5")))
        if n:
            eps = eps[:n]
        for ep in eps:
            i = int(os.path.basename(ep)[7:-5])
            try:
                f = feat_grasp_z(ep)
            except Exception:
                f = None
            if f is not None:
                rows.append((os.path.basename(d), pk, i, f))
    return rows


# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="contact", choices=["contact"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="2means", choices=sorted(SPLITS))
    ap.add_argument("--margin", type=float, default=0.008,
                    help="reject band half-width (m) around the target split threshold")
    ap.add_argument("--n-fit", type=int, default=40, help="taskA eps per leaf for validation")
    args = ap.parse_args()

    pc = PREF_CATEGORIES[args.cat]
    labels = pc.pref_labels

    # ---------- taskA: learn DIRECTION + validate the unsupervised split ----------
    tA = gather(args.cat, "taskA", n=args.n_fit)
    mA = {pk: np.mean([f for _, p, _, f in tA if p == pk]) for pk in pc.pref_keys}
    high_pk = max(mA, key=mA.get)   # taskA-learned direction: higher z -> this pref
    low_pk = min(mA, key=mA.get)
    print(f"[ee {args.cat}] taskA direction: mean z({low_pk})={mA[low_pk]:.3f} < "
          f"z({high_pk})={mA[high_pk]:.3f}  -> higher z = '{high_pk}'")

    groups = sorted({l.rsplit("_", 1)[0] for l, _, _, _ in tA})
    accs = {}
    for g in groups:
        rows = [r for r in tA if r[0].rsplit("_", 1)[0] == g]
        z = np.array([r[3] for r in rows])
        thr = SPLITS[args.split](z)
        acc = np.mean([(high_pk if f >= thr else low_pk) == p for _, p, _, f in rows])
        accs[g] = float(acc)
        print(f"[ee {args.cat}] taskA group {g:26s} n={len(rows):3d} unsup-{args.split} acc={acc:.3f}")
    upright = [g for g, a in accs.items() if a >= 0.95]
    print(f"[ee {args.cat}] taskA per-group acc mean={np.mean(list(accs.values())):.3f} | "
          f"valid-regime groups (>=0.95): {upright}")
    zA = np.array([r[3] for r in tA])
    thrA = SPLITS[args.split](zA)
    accA_pooled = np.mean([(high_pk if f >= thrA else low_pk) == p for _, p, _, f in tA])
    print(f"[ee {args.cat}] taskA POOLED (cross-object, expected weaker) acc={accA_pooled:.3f}")

    # ---------- taskB: unsupervised split on the unlabeled pool ----------
    tB = gather(args.cat, "taskB")
    zB = np.array([r[3] for r in tB])
    thrB = SPLITS[args.split](zB)
    print(f"[ee {args.cat}] taskB n={len(tB)} split={args.split} thr={thrB:.4f} "
          f"margin=±{args.margin:.3f}")

    cache = {}
    kept_ok = kept = correct = 0
    for leaf, gt_pk, i, f in tB:
        pred = high_pk if f >= thrB else low_pk
        decision = "keep" if abs(f - thrB) >= args.margin else "reject"
        match = (pred == gt_pk)
        correct += int(match)
        if decision == "keep":
            kept += 1
            kept_ok += int(match)
        cache[f"{leaf}/episode{i}"] = {
            "action_prompt_label": labels[pred],
            "pref_key": pred,
            "decision": decision,
            "feature": round(f, 4),
            "gt_pref_key": gt_pk,          # audit/scoring only (firewall-dropped at load)
            "gt_match": match,
        }
    accB = correct / max(1, len(tB))
    accB_kept = kept_ok / max(1, kept)
    pred_dist = Counter(v["pref_key"] for v in cache.values() if v["decision"] == "keep")
    print(f"[ee {args.cat}] taskB overall acc={accB:.3f} ({correct}/{len(tB)}) | "
          f"post-filter acc={accB_kept:.3f} ({kept_ok}/{kept}), coverage={kept}/{len(tB)} | "
          f"kept pred_dist={dict(pred_dist)}")
    bad = [k for k, v in cache.items() if v["decision"] == "keep" and not v["gt_match"]]
    if bad:
        print(f"[ee {args.cat}] kept-but-wrong: {bad}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(cache, open(args.out, "w"), indent=2)
    gate = "GATE PASS" if accB_kept >= 0.95 else "below 0.95"
    print(f"[ee {args.cat}] wrote {args.out}  ({gate} on post-filter acc)")


if __name__ == "__main__":
    main()
