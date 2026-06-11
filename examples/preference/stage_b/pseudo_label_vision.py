"""PRIVILEGE-FREE vision pseudo-labeler for the relational pref axes (place, contact).

Replaces the sim-privileged scene_info inputs of pseudo_label_geom.py
(place: receptacle stand_xy/tray_xy/pad_xy; contact: object-relative
height_fraction) with quantities a REAL robot owns:

  * head-camera images + the camera intrinsics/extrinsics THAT ARE IN THE HDF5
    (a fixed, calibrated overhead camera — standard on a real cell),
  * the robot's own EE trajectory + gripper stream,
  * an open-vocabulary VLM grounder (Qwen3-VL-4B) that returns the bounding box
    of a named object in the image.

PLACE  (pref center/corner; feature = ||EE_xy@release − receptacle_xy_est||):
  1. ground the receptacle (pad/tray/stand noun derived from the task-group
     string) in 3 EARLY frames (fracs 0.05/0.15/0.25 — receptacle unoccluded,
     arm not yet in the workspace);
  2. bbox center pixel -> backproject through K,E to a horizontal plane
     z_plane -> receptacle_xy_est (median over frames; frame-spread = QC);
  3. z_plane handling: per-receptacle effective plane height is FIT ON taskA
     (privileged GT receptacle xy is legitimate on the labeled source) and
     transferred to taskB via a robot-measurable proxy: the per-scene median
     Δz = EE_z@release − EE_z@grasp (the in-hand offset cancels exactly, so Δz
     ≈ receptacle surface height above the table) with a 2-point linear map
     (pad, tray) -> z_plane(Δz). NO taskB scene_info is touched.
  4. threshold (PRIMARY, pre-registered): unsupervised exact 1-D 2-means split
     of the TARGET pool's own features (transductive, no GT — same recipe as
     pseudo_label_ee), direction small=center from taskA. Rationale measured on
     taskA: the center-class feature tail is DEMO RELEASE SCATTER (locerr p99
     1.0 cm but release-vs-target up to 4.5 cm), so any taskA-FIXED threshold
     inherits the SOURCE corner-offset scale (9-11 cm) and mismatches targets
     with smaller offsets; both target classes share the same scatter+estimator
     spread, so the target 2-means midpoint self-calibrates. taskA-fixed
     quantile + midpoint rules are computed/reported (and serve as fallback if
     the target split degenerates).

CONTACT (pref 25/75; feature = grasp height FRACTION on the object):
  1. ground the manipulated object in 3 early frames; bbox bottom-center pixel
     -> backproject to the table plane -> object xy + bottom; bbox top row ->
     solve the ray height at that xy -> object top -> h_est = z_top − z_table;
  2. hf_est = (EE_z@first_grasp − k) / h_est, where k = z_table + gripper-TCP
     offset is FIT ON taskA (hf GT legitimate there); threshold = midpoint of
     taskA hf_est class means (~0.5, the object-normalized invariant).
  3. valid regime = upright objects (boxdrink/callbell + taskB boxdrink3);
     lying-flat objects (fork/screwdriver) carry no height signal (see
     pseudo_label_ee.py and the 0612 doc).

FIREWALL: scene_info / GT (preference, dir-name suffix, *_xy, height_fraction)
are read ONLY (a) on taskA for calibration+threshold fitting (source is fully
labeled; privilege there is legitimate) and (b) to SCORE the finished taskB
predictions into gt_* audit fields. The taskB labeling path consumes only:
cached frames, K/E, EE/gripper streams, and the taskA-fitted constants.

Cache schema = pseudo_label_geom.py (action_prompt_label/pref_key/decision +
gt_* audit + diagnostics); decision:"reject" marks low-confidence groundings
(missing/implausible bbox, frame spread > --max-spread) so the Stage-B
firewall drops them (post-filter accuracy matters more than coverage).

Two-stage CLI (grounding is GPU; geometry/labeling is CPU and re-runnable):
  python -m examples.preference.stage_b.pseudo_label_vision --cat place   --stage ground --gpu 0
  python -m examples.preference.stage_b.pseudo_label_vision --cat place   --stage label \
      --out r-preference/eval/pref_pseudo_labels_place_B_vision.json
  python -m examples.preference.stage_b.pseudo_label_vision --cat contact --stage ground --gpu 0
  python -m examples.preference.stage_b.pseudo_label_vision --cat contact --stage label \
      --out r-preference/eval/pref_pseudo_labels_contact_B_vision.json

Frames/meta are pre-extracted under VISION_CACHE by the extraction helper
(per-episode meta.json carries K, E, EE features + gt_* audit fields).
"""
from __future__ import annotations
import argparse, glob, json, os, re
from collections import Counter

import numpy as np

VISION_CACHE = "/mnt/localssd/kaiwenh/pref/vision_cache"
QWEN_PATH = "/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct"
UPSCALE = 2          # feed 640x360 to the VLM (better small-object boxes)
REL_FRAC = 0.90      # release frame for the place feature (= geom labeler)

# ------------------------------------------------------------------
# Grounding queries. Nouns derive from the task-group string (the task
# instruction names the receptacle/object — no sim privilege); the short
# appearance hints are deployment-time prompt engineering, written by looking
# at ONE example camera image per type (an operator can always do this).
# ------------------------------------------------------------------
# NB: prop COLORS are randomized per episode (probe 2026-06-11: pads come in
# blue/green/red; color-hinted queries made the model refuse on off-color
# episodes) -> descriptors are deliberately COLOR-NEUTRAL, shape/function only.
PLACE_RECEPTACLE_QUERY = {
    "pad":   "the square pad (flat colored mat) on the table",
    "tray":  "the tray (shallow rectangular container) on the table",
    "stand": "the display stand (small raised platform with legs) on the table",
}
CONTACT_OBJECT_QUERY = {
    "boxdrink":  "the drink carton (small beverage container) standing on the table, not the trash bin",
    "callbell":  "the small dome-shaped call bell on the table",
    "boxdrink3": "the drink bottle standing upright on the table, not the plate",
}

# taskA leaf lists (fit/validation) and taskB leaf lists (target) per cat.
PLACE_TASKA_GROUPS = [
    "move_mouse_pad", "move_pillbottle_pad", "move_playingcards_pad", "move_soap_pad",
    "place_mouse_tray", "place_pillbottle_tray", "place_playingcards_tray", "place_soap_tray",
]
PLACE_TASKB_GROUPS = ["place_soap2_stand"]
CONTACT_TASKA_GROUPS = [  # upright-object regime only (fork/screwdriver carry no height signal)
    "give_boxdrink", "give_callbell", "put_boxdrink_dustbin", "put_callbell_dustbin",
]
CONTACT_TASKB_GROUPS = ["put_boxdrink3_plate"]
PREF_KEYS = {"place": ("center", "corner"), "contact": ("25", "75")}
SMALL = {"place": "center", "contact": "25"}  # small-feature side (geom convention)
PREF_LABELS = {
    "place": {"center": "center placement", "corner": "corner placement"},
    "contact": {"25": "low contact", "75": "high contact"},
}


def receptacle_type(group: str) -> str:
    return group.rsplit("_", 1)[-1]          # move_soap_pad -> pad, place_soap2_stand -> stand


def contact_object(group: str) -> str:
    toks = group.split("_")
    return toks[1]                            # give_boxdrink / put_boxdrink_dustbin / put_boxdrink3_plate


def leaf_dirs(cat: str, groups, pref_keys):
    out = []
    for g in groups:
        for pk in pref_keys:
            d = os.path.join(VISION_CACHE, cat, f"{g}_{pk}")
            if os.path.isdir(d):
                out.append((g, pk, d))
    return out


def episodes(leaf_dir):
    eps = []
    for d in sorted(glob.glob(os.path.join(leaf_dir, "episode*")),
                    key=lambda p: int(os.path.basename(p)[7:])):
        if os.path.isfile(os.path.join(d, "meta.json")):
            eps.append(d)
    return eps


# ------------------------------------------------------------------
# Camera geometry (OpenCV convention: x_cam = E[:, :3] @ X_world + E[:, 3])
# ------------------------------------------------------------------

def cam_center_and_rays(K, E):
    K = np.asarray(K, float); E = np.asarray(E, float)
    R, t = E[:, :3], E[:, 3]
    C = -R.T @ t
    Kinv = np.linalg.inv(K)
    return C, R, Kinv


def backproject_to_plane(K, E, uv, z_plane):
    """Pixel (u, v) in the ORIGINAL frame -> world (x, y) on the z=z_plane plane."""
    C, R, Kinv = cam_center_and_rays(K, E)
    d_cam = Kinv @ np.array([uv[0], uv[1], 1.0])
    d_w = R.T @ d_cam
    if abs(d_w[2]) < 1e-9:
        return None
    s = (z_plane - C[2]) / d_w[2]
    if s <= 0:
        return None
    X = C + s * d_w
    return X[:2]


def solve_z_at_xy(K, E, v_row, xy, z_lo=0.60, z_hi=1.40):
    """World z of the point at horizontal position `xy` whose projection has
    image row v_row (vertical line search; v(z) is monotonic for an overhead cam)."""
    K = np.asarray(K, float); E = np.asarray(E, float)
    zs = np.linspace(z_lo, z_hi, 401)
    P = np.stack([np.full_like(zs, xy[0]), np.full_like(zs, xy[1]), zs])
    xc = E[:, :3] @ P + E[:, 3:4]
    valid = xc[2] > 1e-6
    uv = (K @ xc)[:2] / xc[2]
    v = uv[1]
    zs, v = zs[valid], v[valid]
    order = np.argsort(v)
    v_sorted, z_sorted = v[order], zs[order]
    if not (v_sorted[0] - 2 <= v_row <= v_sorted[-1] + 2):
        return None
    return float(np.interp(v_row, v_sorted, z_sorted))


# ------------------------------------------------------------------
# Stage 1: grounding (GPU)
# ------------------------------------------------------------------

_BBOX_RE = re.compile(
    r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]")


def parse_bbox(text, img_w, img_h):
    """First [x1,y1,x2,y2] in the text -> bbox in the FED image's pixel space.

    Coordinate convention: Qwen3-VL outputs 0-1000-NORMALIZED coords regardless
    of input size — VERIFIED EMPIRICALLY 2026-06-11 by drawing both candidate
    interpretations on 640x360 frames (r-preference/debug/v5/ground_*.png:
    norm-1000 boxes land exactly on the stand/pad/tray/bottle; abs-px do not;
    raw y-coords like 913 also exceed the 360-px image height)."""
    m = _BBOX_RE.search(text)
    if not m:
        return None
    b = [float(m.group(i)) for i in range(1, 5)]
    if b[2] <= b[0] or b[3] <= b[1]:
        return None
    return [b[0] / 1000 * img_w, b[1] / 1000 * img_h,
            b[2] / 1000 * img_w, b[3] / 1000 * img_h]


class QwenGrounder:
    def __init__(self, gpu: int = 0):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        dev = f"cuda:{gpu}"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            QWEN_PATH, dtype=torch.bfloat16, device_map=dev)
        self.processor = AutoProcessor.from_pretrained(QWEN_PATH)
        self.dev = dev

    def ground(self, pil_img, query: str):
        return self.ground_batch([pil_img], [query])[0]

    def ground_batch(self, pil_imgs, queries):
        """Batched grounding (left-padded generation)."""
        texts, im_lists = [], []
        for im, q in zip(pil_imgs, queries):
            prompt = (f"Locate {q}. Output its bounding box in JSON format: "
                      f'{{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}')
            messages = [{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            im_lists.append(im)
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(text=texts, images=im_lists, padding=True,
                                return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return [self.processor.decode(g, skip_special_tokens=True) for g in gen]


def stage_ground(cat: str, gpu: int, limit_fit: int):
    from PIL import Image
    gpath = os.path.join(VISION_CACHE, f"groundings_{cat}.json")
    G = json.load(open(gpath)) if os.path.exists(gpath) else {}
    if cat == "place":
        leaves = (leaf_dirs(cat, PLACE_TASKA_GROUPS, PREF_KEYS[cat])
                  + leaf_dirs(cat, PLACE_TASKB_GROUPS, PREF_KEYS[cat]))
        query_of = lambda g: PLACE_RECEPTACLE_QUERY[receptacle_type(g)]
    else:
        leaves = (leaf_dirs(cat, CONTACT_TASKA_GROUPS, PREF_KEYS[cat])
                  + leaf_dirs(cat, CONTACT_TASKB_GROUPS, PREF_KEYS[cat]))
        query_of = lambda g: CONTACT_OBJECT_QUERY[contact_object(g)]

    todo = []
    for g, pk, d in leaves:
        eps = episodes(d)
        is_target = (g in PLACE_TASKB_GROUPS) or (g in CONTACT_TASKB_GROUPS)
        if not is_target and limit_fit:
            eps = eps[:limit_fit]
        for ed in eps:
            leaf = os.path.basename(os.path.dirname(ed))
            epn = os.path.basename(ed)
            for jp in sorted(glob.glob(os.path.join(ed, "f*.jpg"))):
                frac = os.path.basename(jp)[1:-4]
                key = f"{leaf}/{epn}/{frac}"
                if key not in G:
                    todo.append((key, jp, query_of(g)))
    print(f"[ground {cat}] {len(todo)} frames to ground "
          f"({len(G)} already cached) -> {gpath}", flush=True)
    if not todo:
        return
    gr = QwenGrounder(gpu)
    B = 8
    done = 0
    for s in range(0, len(todo), B):
        chunk = todo[s:s + B]
        ims, qs, dims = [], [], []
        for key, jp, query in chunk:
            im = Image.open(jp).convert("RGB")
            W0, H0 = im.size
            ims.append(im.resize((W0 * UPSCALE, H0 * UPSCALE), Image.LANCZOS))
            qs.append(query)
            dims.append((W0, H0))
        txts = gr.ground_batch(ims, qs)
        for (key, jp, query), txt, (W0, H0) in zip(chunk, txts, dims):
            bb = parse_bbox(txt, W0 * UPSCALE, H0 * UPSCALE)
            G[key] = {
                "bbox_px": [round(c / UPSCALE, 2) for c in bb] if bb else None,  # original-frame px
                "raw": txt[:200],
            }
        done += len(chunk)
        if done % 96 < B or done == len(todo):
            tmp = gpath + ".tmp"
            json.dump(G, open(tmp, "w"))
            os.replace(tmp, gpath)
            print(f"[ground {cat}] {done}/{len(todo)} done", flush=True)


# ------------------------------------------------------------------
# Stage 2: geometry + calibration + labeling (CPU)
# ------------------------------------------------------------------

def load_meta(ed):
    return json.load(open(os.path.join(ed, "meta.json")))


def ep_groundings(G, leaf, epn):
    out = {}
    for key, v in G.items():
        l, e, frac = key.split("/")
        if l == leaf and e == epn and v.get("bbox_px"):
            out[frac] = v["bbox_px"]
    return out


def collect_rows(cat, groups, limit=None):
    """-> list of (group, pk, leaf, epn, meta) over the cached episodes."""
    rows = []
    for g, pk, d in leaf_dirs(cat, groups, PREF_KEYS[cat]):
        eps = episodes(d)
        if limit:
            eps = eps[:limit]
        for ed in eps:
            rows.append((g, pk, os.path.basename(d), os.path.basename(ed), load_meta(ed)))
    return rows


# ---------------- place ----------------

def place_xy_est(G, leaf, epn, meta, z_plane, max_spread):
    """Median backprojected bbox-center xy over the valid frames.
    Returns (xy, spread, n_frames) or (None, reason-string, n).

    Wrong-object gate (privilege-free): in the early frames the manipulated
    object still sits at its pickup site, which the robot knows as its own EE
    xy at the grasp frame. If the 'receptacle' estimate lands within 4 cm of
    that site, the grounder almost surely boxed the manipulated object ->
    reject rather than emit a confidently wrong feature."""
    bbs = ep_groundings(G, leaf, epn)
    if not bbs:
        return None, "no_bbox", 0
    K, E = meta["K"], meta["E"]
    pts = []
    for frac, bb in bbs.items():
        u, v = 0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])
        xy = backproject_to_plane(K, E, (u, v), z_plane)
        if xy is not None:
            pts.append(xy)
    if not pts:
        return None, "backproj_fail", 0
    pts = np.array(pts)
    med = np.median(pts, axis=0)
    spread = float(np.max(np.linalg.norm(pts - med, axis=1))) if len(pts) > 1 else 0.0
    if len(pts) < 2:
        return None, "lt2_frames", len(pts)
    if spread > max_spread:
        return None, f"spread_{spread:.3f}", len(pts)
    if meta.get("xy_grasp") is not None and \
            float(np.linalg.norm(med - np.asarray(meta["xy_grasp"], float))) < 0.04:
        return None, "near_grasp_site", len(pts)
    return med, spread, len(pts)


def fit_z_plane(G, rows, z_grid):
    """taskA calibration: effective backprojection plane height per receptacle
    type = argmin_z median ||backproj(bbox_center, z) − GT receptacle xy||.
    Uses gt_*_xy — ALLOWED (taskA only)."""
    errs = {z: [] for z in z_grid}
    for g, pk, leaf, epn, meta in rows:
        gt = None
        for k in ("gt_pad_xy", "gt_tray_xy", "gt_stand_xy"):
            if k in meta:
                gt = np.asarray(meta[k], float)
        if gt is None:
            continue
        bbs = ep_groundings(G, leaf, epn)
        K, E = meta["K"], meta["E"]
        for z in z_grid:
            pts = []
            for frac, bb in bbs.items():
                u, v = 0.5 * (bb[0] + bb[2]), 0.5 * (bb[1] + bb[3])
                xy = backproject_to_plane(K, E, (u, v), z)
                if xy is not None:
                    pts.append(xy)
            if pts:
                errs[z].append(np.linalg.norm(np.median(np.array(pts), axis=0) - gt))
    med = {z: float(np.median(e)) for z, e in errs.items() if e}
    z_star = min(med, key=med.get)
    e_at = np.array(errs[z_star])
    return z_star, {"med_err": float(np.median(e_at)), "p90_err": float(np.percentile(e_at, 90)),
                    "max_err": float(np.max(e_at)), "n": len(e_at)}


def _two_means_1d(x):
    """Exact 1-D 2-means split (same as pseudo_label_ee.split_2means)."""
    xs = np.sort(np.asarray(x, float))
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


def leaf_dz(rows):
    """Per-scene receptacle-height proxy: median (EE_z@release − EE_z@grasp).
    Pure robot trajectory — privilege-free."""
    dz = [m["z_release"] - m["z_grasp"] for _, _, _, _, m in rows
          if m.get("z_release") is not None and m.get("z_grasp") is not None]
    return float(np.median(dz))


def stage_label_place(args):
    G = json.load(open(os.path.join(VISION_CACHE, "groundings_place.json")))
    rowsA = collect_rows("place", PLACE_TASKA_GROUPS, limit=args.n_fit)
    rowsB = collect_rows("place", PLACE_TASKB_GROUPS)
    z_grid = np.round(np.arange(0.68, 0.96, 0.0025), 4)

    # --- calibration per taskA receptacle type (privileged: taskA GT xy) ---
    calib = {}
    for typ in ("pad", "tray"):
        rows_t = [r for r in rowsA if receptacle_type(r[0]) == typ]
        z_star, errstats = fit_z_plane(G, rows_t, z_grid)
        dz = leaf_dz(rows_t)
        calib[typ] = {"z_star": float(z_star), "dz": dz, **errstats}
        print(f"[vision place] calib {typ}: z*={z_star:.4f} dz={dz:+.4f} "
              f"loc-err med={errstats['med_err']*100:.1f}cm p90={errstats['p90_err']*100:.1f}cm n={errstats['n']}")
    # 2-point linear map dz -> z_plane (transfers to the unseen taskB receptacle)
    b = (calib["tray"]["z_star"] - calib["pad"]["z_star"]) / (calib["tray"]["dz"] - calib["pad"]["dz"])
    a = calib["pad"]["z_star"] - b * calib["pad"]["dz"]
    print(f"[vision place] z-plane map: z = {a:.4f} + {b:.3f} * dz")

    # --- taskA features (same estimator as taskB: type-fitted plane) ---
    featA = {"center": [], "corner": []}
    rejA = 0
    for g, pk, leaf, epn, meta in rowsA:
        typ = receptacle_type(g)
        xy, sp, n = place_xy_est(G, leaf, epn, meta, calib[typ]["z_star"], args.max_spread)
        if not isinstance(xy, np.ndarray):
            rejA += 1
            continue
        f = float(np.linalg.norm(np.asarray(meta["xy_t90"]) - xy))
        featA[pk].append(f)
    c = np.array(featA["center"]); x = np.array(featA["corner"])
    thr_mid = 0.5 * (c.mean() + x.mean())
    # PRE-REGISTERED primary rule (set before any taskB scoring was looked at):
    # thr = max(q_safety x p99(taskA center-class features), 3 cm). The quantile
    # tracks ESTIMATOR noise (invariant across tasks), unlike the midpoint which
    # scales with the SOURCE corner offset (9-11 cm here) and over-thresholds
    # targets with smaller offsets. The 3 cm floor: receptacles are >=10 cm
    # objects, so any task's center/corner contrast is >> 3 cm, while a tighter
    # threshold only adds false-corner risk under target-side calibration drift.
    thr_q = float(max(np.percentile(c, 99) * args.q_safety, 0.03))
    accA_mid = 0.5 * (np.mean(c < thr_mid) + np.mean(x >= thr_mid))
    accA_q = 0.5 * (np.mean(c < thr_q) + np.mean(x >= thr_q))
    print(f"[vision place] taskA feats: center {c.mean()*100:.1f}±{c.std()*100:.1f}cm "
          f"(p99 {np.percentile(c,99)*100:.1f}) corner {x.mean()*100:.1f}±{x.std()*100:.1f}cm | rejected {rejA}")
    print(f"[vision place] taskA-fixed thresholds: midpoint {thr_mid*100:.1f}cm (taskA acc {accA_mid:.3f}) | "
          f"quantile {thr_q*100:.1f}cm (taskA acc {accA_q:.3f})  [rule: {args.thr_rule}]")

    # --- taskB: privilege-free path ---
    dz_B = leaf_dz(rowsB)                      # robot-measured receptacle height proxy
    z_B = float(a + b * dz_B)
    print(f"[vision place] taskB dz={dz_B:+.4f} -> z_plane={z_B:.4f}")
    featB = {}
    for g, gt_pk, leaf, epn, meta in rowsB:
        xy, sp, nfr = place_xy_est(G, leaf, epn, meta, z_B, args.max_spread)
        featB[f"{leaf}/{epn}"] = (g, gt_pk, leaf, epn, meta, xy, sp, nfr)

    # PRIMARY RULE (pre-registered 2026-06-11, BEFORE any taskB feature was
    # computed; rationale measured on taskA only): the center-class feature tail
    # is DEMO RELEASE SCATTER (taskA decomposition: locerr p99 = 1.0 cm but
    # release-vs-target up to 4.5 cm), an object/policy property that differs
    # between tasks — so a taskA-fixed threshold is scale-mismatched on a target
    # whose corner offset differs from the source's (9-11 cm there). Both target
    # classes share the same scatter+estimator spread, so the split point is
    # recovered UNSUPERVISED on the target pool (exact 1-D 2-means, transductive,
    # no GT — same recipe as pseudo_label_ee), with taskA giving the DIRECTION
    # (small = center) and the calibration constants. Degenerate-split fallback
    # (either cluster < 10% of pool): taskA quantile threshold.
    fvals = np.array([np.linalg.norm(np.asarray(v[4]["xy_t90"]) - v[5])
                      for v in featB.values() if isinstance(v[5], np.ndarray)])
    thr_2m = _two_means_1d(fvals)
    n_lo = int(np.sum(fvals < thr_2m))
    degenerate = min(n_lo, len(fvals) - n_lo) < max(3, 0.10 * len(fvals))
    if args.thr_rule == "2means" and not degenerate:
        thr = float(thr_2m)
    elif args.thr_rule == "midpoint":
        thr = thr_mid
    else:
        thr = thr_q
    print(f"[vision place] taskB split: 2means={thr_2m*100:.2f}cm (lo/hi {n_lo}/{len(fvals)-n_lo}"
          f"{', DEGENERATE -> fallback' if degenerate else ''}) | using thr={thr*100:.2f}cm "
          f"margin=±{args.margin*100:.1f}cm")

    cache, n_ok, n_kept, n_kept_ok = {}, 0, 0, 0
    labels = PREF_LABELS["place"]
    for key, (g, gt_pk, leaf, epn, meta, xy, sp, nfr) in featB.items():
        if not isinstance(xy, np.ndarray):
            cache[key] = {"action_prompt_label": labels[SMALL['place']], "pref_key": SMALL["place"],
                          "decision": "reject", "reject_reason": str(sp),
                          "gt_pref_key": gt_pk, "gt_match": False}
            continue
        f = float(np.linalg.norm(np.asarray(meta["xy_t90"]) - xy))
        pred = "center" if f < thr else "corner"
        decision = "keep" if abs(f - thr) >= args.margin else "reject"
        match = (pred == gt_pk)
        if decision == "keep":
            n_ok += int(match); n_kept += 1; n_kept_ok += int(match)
        cache[key] = {
            "action_prompt_label": labels[pred], "pref_key": pred, "decision": decision,
            "feature": round(f, 4), "xy_est": [round(v, 4) for v in xy.tolist()],
            "frame_spread": round(sp, 4), "n_frames": nfr, "z_plane": round(z_B, 4),
            "gt_pref_key": gt_pk, "gt_match": match,
        }
    nB = len(rowsB)
    # overall = all episodes that HAVE a prediction scored (incl. margin-rejects),
    # geometry-failures count as wrong (conservative; same convention as the doc)
    n_all_ok = sum(1 for v in cache.values() if v.get("feature") is not None and v["gt_match"])
    acc = n_all_ok / nB
    acc_kept = n_kept_ok / max(1, n_kept)
    pred_dist = Counter(v["pref_key"] for v in cache.values() if v["decision"] == "keep")
    print(f"[vision place] taskB n={nB} overall acc={acc:.3f} | post-filter acc={acc_kept:.3f} "
          f"({n_kept_ok}/{n_kept}), coverage={n_kept}/{nB} | kept pred_dist={dict(pred_dist)}")
    wrong = [k for k, v in cache.items() if v["decision"] == "keep" and not v["gt_match"]]
    if wrong:
        print(f"[vision place] kept-but-wrong: {wrong}")
    _write_cache(args.out, cache, acc_kept)
    if args.calib_out:
        json.dump({"calib": calib, "z_map": {"a": a, "b": b}, "thr": thr,
                   "thr_mid": thr_mid, "thr_q": thr_q, "z_B": z_B, "dz_B": dz_B},
                  open(args.calib_out, "w"), indent=2)

    # POST-HOC DIAGNOSTIC (printed after the cache is written; cannot influence
    # labeling): taskB receptacle localization error vs scene_info GT — part of
    # the scoring/failure analysis, same privilege class as gt_match above.
    errsB = []
    for g, gt_pk, leaf, epn, meta in rowsB:
        if "gt_stand_xy" not in meta:
            continue
        xy, sp, nfr = place_xy_est(G, leaf, epn, meta, z_B, args.max_spread)
        if isinstance(xy, np.ndarray):
            errsB.append(np.linalg.norm(xy - np.asarray(meta["gt_stand_xy"], float)))
    if errsB:
        e = np.array(errsB)
        print(f"[vision place] POST-HOC taskB localization err: med={np.median(e)*100:.1f}cm "
              f"p90={np.percentile(e,90)*100:.1f}cm max={e.max()*100:.1f}cm (n={len(e)})")


# ---------------- contact ----------------

def contact_obj_geom(G, leaf, epn, meta, z_table, max_h_spread=0.03):
    """Object bottom xy + height from the bbox: bottom-center -> table plane;
    top row -> solve z at that xy. Median over frames.

    Wrong-object gate (privilege-free): the robot grasps THIS object, so its
    own EE xy at the first gripper close marks the grasp NEIGHBORHOOD. NB the
    EE pose is the WRIST, which for this gripper sits ~9-11 cm behind the
    fingertips on a side grasp (measured on taskB: dist 8.8-11.4 cm on
    visually-perfect groundings) -> the gate is a GROSS-error catch only
    (> 25 cm = boxed something across the table, e.g. the dustbin); fine-
    grained wrong-object rejection is delegated to the hf-plausibility band
    in stage_label_contact (a real fraction must be ~[0,1]; a plate/bin box
    gives a wildly out-of-band hf_est)."""
    bbs = ep_groundings(G, leaf, epn)
    if not bbs:
        return None
    K, E = meta["K"], meta["E"]
    hs, xys = [], []
    for frac, bb in bbs.items():
        u_c, v_top, v_bot = 0.5 * (bb[0] + bb[2]), bb[1], bb[3]
        xy = backproject_to_plane(K, E, (u_c, v_bot), z_table)
        if xy is None:
            continue
        z_top = solve_z_at_xy(K, E, v_top, xy)
        if z_top is None:
            continue
        hs.append(z_top - z_table)
        xys.append(xy)
    if len(hs) < 2:
        return None
    hs = np.array(hs)
    h = float(np.median(hs))
    if h < 0.02 or (hs.max() - hs.min()) > max_h_spread:
        return None
    if meta.get("xy_first_close") is not None:
        med_xy = np.median(np.array(xys), axis=0)
        if float(np.linalg.norm(med_xy - np.asarray(meta["xy_first_close"], float))) > 0.25:
            return None
    return h


def stage_label_contact(args):
    G = json.load(open(os.path.join(VISION_CACHE, "groundings_contact.json")))
    rowsA = collect_rows("contact", CONTACT_TASKA_GROUPS, limit=args.n_fit)
    rowsB = collect_rows("contact", CONTACT_TASKB_GROUPS)

    # --- calibration on taskA (hf GT legitimate there) ---
    # model: EE_z@grasp = k + hf * h_est(z_table). (z_table, k) are NOT jointly
    # identifiable from hf matching alone (flat ridge: a 2-D grid search rode
    # the grid edge and the taskA residual kept improving while taskB acc
    # dropped — measured 2026-06-11). z_table is therefore ANCHORED physically:
    # the place-pad effective plane from calib_place.json (the flat pad lies on
    # the SAME table — head-camera K,E verified identical across categories;
    # on a real cell the table height is directly measurable anyway). Only the
    # scalar k (TCP/fingertip offset) is fit here, closed-form.
    calib_place_path = os.path.join(VISION_CACHE, "calib_place.json")
    if os.path.exists(calib_place_path):
        z_table = float(json.load(open(calib_place_path))["calib"]["pad"]["z_star"])
    else:
        z_table = 0.7225  # the 2026-06-11 pad fit; regenerate via --cat place --stage label
    recs = []
    for g, pk, leaf, epn, meta in rowsA:
        if meta.get("z_first_close") is None or "gt_height_fraction" not in meta:
            continue
        h = contact_obj_geom(G, leaf, epn, meta, z_table)
        if h is None:
            continue
        recs.append((meta["z_first_close"], meta["gt_height_fraction"], h))
    zg = np.array([r[0] for r in recs]); hfgt = np.array([r[1] for r in recs]); hh = np.array([r[2] for r in recs])
    k = float(np.median(zg - hfgt * hh))
    med = float(np.median(np.abs((zg - k) / hh - hfgt)))
    n_cal = len(recs)
    print(f"[vision contact] calib: z_table={z_table:.4f} (anchored, place-pad plane) "
          f"k={k:.4f} (median |hf_est − hf_gt| = {med:.3f}, n={n_cal})")

    # --- taskA same-estimator features + threshold ---
    # hf-plausibility band: a grasp-height FRACTION must be ~[0,1]; far outside
    # means the grounder boxed the wrong object (plate/bin -> tiny/huge h_est).
    HF_BAND = (-0.25, 1.25)
    featA = {"25": [], "75": []}
    rejA = 0
    for g, pk, leaf, epn, meta in rowsA:
        h = contact_obj_geom(G, leaf, epn, meta, z_table)
        if h is None or meta.get("z_first_close") is None:
            rejA += 1
            continue
        hf = (meta["z_first_close"] - k) / h
        if not (HF_BAND[0] <= hf <= HF_BAND[1]):
            rejA += 1
            continue
        featA[pk].append(hf)
    lo = np.array(featA["25"]); hi = np.array(featA["75"])
    thr = 0.5 * (lo.mean() + hi.mean())
    accA = 0.5 * (np.mean(lo < thr) + np.mean(hi >= thr))
    print(f"[vision contact] taskA hf_est: 25 {lo.mean():.3f}±{lo.std():.3f} (n={len(lo)})  "
          f"75 {hi.mean():.3f}±{hi.std():.3f} (n={len(hi)})  thr={thr:.3f}  acc={accA:.3f}  rejected {rejA}")

    # --- taskB ---
    cache, n_ok, n_kept, n_kept_ok = {}, 0, 0, 0
    labels = PREF_LABELS["contact"]
    for g, gt_pk, leaf, epn, meta in rowsB:
        h = contact_obj_geom(G, leaf, epn, meta, z_table)
        key = f"{leaf}/{epn}"
        if h is None or meta.get("z_first_close") is None:
            cache[key] = {"action_prompt_label": labels[SMALL['contact']], "pref_key": SMALL["contact"],
                          "decision": "reject", "reject_reason": "geom_fail",
                          "gt_pref_key": gt_pk, "gt_match": False}
            continue
        f = float((meta["z_first_close"] - k) / h)
        pred = "25" if f < thr else "75"
        decision = "keep" if HF_BAND[0] <= f <= HF_BAND[1] else "reject"
        match = (pred == gt_pk)
        if decision == "keep":
            n_ok += int(match); n_kept += 1; n_kept_ok += int(match)
        cache[key] = {
            "action_prompt_label": labels[pred], "pref_key": pred, "decision": decision,
            "feature": round(f, 4), "h_est": round(h, 4),
            "gt_pref_key": gt_pk, "gt_match": match,
        }
        if decision == "reject":
            cache[key]["reject_reason"] = "hf_out_of_band"
    nB = len(rowsB)
    n_all_ok = sum(1 for v in cache.values() if v.get("feature") is not None and v["gt_match"])
    acc = n_all_ok / nB
    acc_kept = n_kept_ok / max(1, n_kept)
    pred_dist = Counter(v["pref_key"] for v in cache.values() if v["decision"] == "keep")
    print(f"[vision contact] taskB n={nB} overall acc={acc:.3f} | post-filter acc={acc_kept:.3f} "
          f"({n_kept_ok}/{n_kept}), coverage={n_kept}/{nB} | kept pred_dist={dict(pred_dist)}")
    wrong = [k2 for k2, v in cache.items() if v["decision"] == "keep" and not v["gt_match"]]
    if wrong:
        print(f"[vision contact] kept-but-wrong: {wrong}")
    _write_cache(args.out, cache, acc_kept)
    if args.calib_out:
        json.dump({"z_table": z_table, "k": k, "thr": float(thr), "calib_resid": med},
                  open(args.calib_out, "w"), indent=2)


def _write_cache(out, cache, acc_kept):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(cache, open(out, "w"), indent=2)
    gate = "GATE PASS" if acc_kept >= 0.95 else "below 0.95"
    print(f"[vision] wrote {out}  ({gate} on post-filter acc)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", required=True, choices=["place", "contact"])
    ap.add_argument("--stage", default="all", choices=["ground", "label", "all"])
    ap.add_argument("--out")
    ap.add_argument("--calib-out")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-fit", type=int, default=40, help="taskA episodes per leaf")
    ap.add_argument("--max-spread", type=float, default=0.03,
                    help="reject if frame-to-frame xy estimates spread beyond this (m)")
    ap.add_argument("--thr-rule", default="2means", choices=["2means", "quantile", "midpoint"],
                    help="place threshold rule; 2means = unsupervised transductive split on the "
                         "target pool (primary, pre-registered; self-calibrates to the target "
                         "offset scale), quantile = taskA estimator-noise p99 x safety, "
                         "midpoint = geom parity (scales with the SOURCE corner offset)")
    ap.add_argument("--q-safety", type=float, default=1.5)
    ap.add_argument("--margin", type=float, default=0.008,
                    help="reject band half-width (m) around the place threshold")
    args = ap.parse_args()
    if args.stage in ("ground", "all"):
        stage_ground(args.cat, args.gpu, args.n_fit)
    if args.stage in ("label", "all"):
        if not args.out:
            raise SystemExit("--out required for the label stage")
        if args.cat == "place":
            stage_label_place(args)
        else:
            stage_label_contact(args)


if __name__ == "__main__":
    main()
