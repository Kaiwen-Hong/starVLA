"""Geometric/trajectory pseudo-labeler for the RELATIONAL pref cats (place/contact/hvlv),
where token-VQA (absolute EE pose only) failed because the pref is relative to an
object/target/obstacle. Each cat uses a relational feature that transfers to taskB:
  place  : ||EE_release_xy - receptacle_xy||   (receptacle pose from scene_info)  small=center, large=corner
  contact: height_fraction                     (object-relative grasp height, scene_info)  small=25(low), large=75(high)
  hvlv   : max perpendicular detour of the active-arm xy path from its transport chord
           (PURE EE trajectory, NO obstacle pose)  small=lv, large=hv
  orient : wrist-rotation R[2,2] (tool z-axis world-vertical component) of the active arm
           at grasp frac 0.40 (PURE EE pose, no privilege)  small(more negative)=0(side), large=90(top)
           Probe 2026-06-11 (/tmp/orient_geom_probe.py): taskA 1.000 / taskB 1.000 (n=100),
           S/N 5-11 on several redundant rotation features; the 5 eps the token-VQA
           mislabeled (move_can5_away_0/ep{7,27,31,33,38}) are all correctly 0 here.

Threshold is FIT on taskA (which has pref labels) as the midpoint of the two class
means, then applied to the unlabeled taskB. Writes the same cache schema
PrefHDF5StageBDataset consumes. NOTE: place/contact read scene_info (sim-privileged
receptacle/object geometry; we never read the GT `preference`/`grasp_region` field);
hvlv uses only the EE trajectory.

CLI: python -m examples.preference.stage_b.pseudo_label_geom --cat place --out r-preference/eval/pref_pseudo_labels_place_B.json
"""
from __future__ import annotations
import argparse, json, glob, os
import numpy as np, h5py
from examples.preference.dataset.prompt import PREF_CATEGORIES

ROOT = "/mnt/localssd/kaiwenh/pref/data/0526"
REL_FRAC = 0.90


def _receptacle_xy(info):
    for k in ("stand_xy", "tray_xy", "pad_xy"):
        if k in info:
            return np.asarray(info[k], float)
    return None


def feat_place(ep, info):
    rxy = _receptacle_xy(info); arm = info.get("{a}", "right")
    if rxy is None:
        return None
    with h5py.File(ep, "r") as h:
        ee = h[f"endpose/{arm}_endpose"]; t = int(round(REL_FRAC * (ee.shape[0] - 1)))
        exy = np.asarray(ee[t][:2], float)
    return float(np.linalg.norm(exy - rxy))


def feat_contact(ep, info):
    hf = info.get("height_fraction")
    return float(hf) if hf is not None else None


def feat_hvlv(ep, info):
    arm = info.get("{a}", "left")
    with h5py.File(ep, "r") as h:
        xy = np.asarray(h[f"endpose/{arm}_endpose"][:, :2], float)
    T = len(xy); seg = xy[int(0.25 * T):int(0.85 * T) + 1]
    if len(seg) < 3:
        return None
    p0, p1 = seg[0], seg[-1]; v = p1 - p0; vn = v / (np.linalg.norm(v) + 1e-9)
    d = seg - p0; perp = np.abs(d[:, 0] * (-vn[1]) + d[:, 1] * vn[0])
    return float(perp.max())


def _quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def feat_orient(ep, info):
    # wrist rotation at the grasp window: R[2,2] (tool z-axis vertical comp) at frac 0.40.
    # Pure EE pose (active arm by xyz range), no scene_info needed.
    with h5py.File(ep, "r") as h:
        L = h["endpose/left_endpose"][:]; R = h["endpose/right_endpose"][:]
        arm = L if np.linalg.norm(L[:, :3].ptp(0)) > np.linalg.norm(R[:, :3].ptp(0)) else R
        t = int(round(0.40 * (len(arm) - 1)))
        return float(_quat_to_R(arm[t, 3:7])[2, 2])


FEAT = {"place": feat_place, "contact": feat_contact, "hvlv": feat_hvlv, "orient": feat_orient}
# pref_key on the SMALL-feature side (the other key is large side)
SMALL = {"place": "center", "contact": "25", "hvlv": "lv", "orient": "0"}


def leaves(cat, split):
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
                out.append((os.path.basename(d).rsplit("_", 1)[-1], d))
    return out


def gather(cat, split, n=None):
    rows = []  # (leafname, pref_key, ep_id, feature)
    fn = FEAT[cat]
    for pk, d in leaves(cat, split):
        sj = os.path.join(d, "scene_info.json")
        sinfo = json.load(open(sj)) if os.path.exists(sj) else {}
        eps = sorted(glob.glob(os.path.join(d, "data", "episode*.hdf5")))
        if n:
            eps = eps[:n]
        for ep in eps:
            i = int(os.path.basename(ep)[7:-5])
            info = sinfo.get(f"episode_{i}", {}).get("info", {})
            try:
                f = fn(ep, info)
            except Exception:
                f = None
            if f is not None:
                rows.append((os.path.basename(d), pk, i, f))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True, choices=["place", "contact", "hvlv", "orient"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pc = PREF_CATEGORIES[args.cat]
    small_pk = SMALL[args.cat]
    large_pk = [k for k in pc.pref_keys if k != small_pk][0]
    labels = pc.pref_labels

    # fit threshold on taskA
    tA = gather(args.cat, "taskA", n=40)
    s = [f for _, pk, _, f in tA if pk == small_pk]
    l = [f for _, pk, _, f in tA if pk == large_pk]
    thr = 0.5 * (np.mean(s) + np.mean(l))
    accA = (np.mean(np.array(s) < thr) + np.mean(np.array(l) >= thr)) / 2
    print(f"[geom {args.cat}] feature thr (fit taskA) = {thr:.4f}  | small={small_pk}(<thr) mean={np.mean(s):.3f}  large={large_pk} mean={np.mean(l):.3f}  taskA-acc={accA:.3f}")

    # label taskB
    tB = gather(args.cat, "taskB")
    cache = {}
    correct = 0
    for leaf, gt_pk, i, f in tB:
        pred = small_pk if f < thr else large_pk
        correct += int(pred == gt_pk)
        cache[f"{leaf}/episode{i}"] = {
            "action_prompt_label": labels[pred], "pref_key": pred, "decision": "keep",
            "gt_pref_key": gt_pk, "gt_match": (pred == gt_pk), "feature": round(f, 4),
        }
    accB = correct / max(1, len(tB))
    from collections import Counter
    pred_dist = Counter(v["pref_key"] for v in cache.values())
    print(f"[geom {args.cat}] taskB: n={len(tB)} pseudo-vs-GT acc={accB:.3f}  pred_dist={dict(pred_dist)}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(cache, open(args.out, "w"), indent=2)
    print(f"[geom {args.cat}] wrote {args.out}  ({'GATE PASS' if accB >= 0.95 else 'below 0.95 (borderline)'})")


if __name__ == "__main__":
    main()
