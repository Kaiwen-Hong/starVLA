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
  4. threshold: primary rule = estimator-noise quantile (p99 of taskA CENTER-
     class features × safety) — invariant to the source corner offset, which
     differs from the target's; the geom-parity midpoint rule is also reported.

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
PLACE_RECEPTACLE_QUERY = {
    "pad":   "the blue square pad lying flat on the table",
    "tray":  "the light blue tray on the table",
    "stand": "the black display stand (small raised platform) on the table",
}
CONTACT_OBJECT_QUERY = {
    "boxdrink":  "the grey drink carton standing upright on the table",
    "callbell":  "the small blue dome-shaped call bell on the table",
    "boxdrink3": "the blue drink bottle standing upright on the table",
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
        prompt = (f"Locate {query}. Output its bounding box in JSON format: "
                  f'{{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}')
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True)


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
    for n, (key, jp, query) in enumerate(todo):
        im = Image.open(jp).convert("RGB")
        W0, H0 = im.size
        im2 = im.resize((W0 * UPSCALE, H0 * UPSCALE), Image.LANCZOS)
        txt = gr.ground(im2, query)
        bb = parse_bbox(txt, W0 * UPSCALE, H0 * UPSCALE)
        G[key] = {
            "bbox_px": [round(c / UPSCALE, 2) for c in bb] if bb else None,  # original-frame px
            "raw": txt[:200],
        }
        if (n + 1) % 100 == 0 or n + 1 == len(todo):
            tmp = gpath + ".tmp"
            json.dump(G, open(tmp, "w"))
            os.replace(tmp, gpath)
            print(f"[ground {cat}] {n + 1}/{len(todo)} done", flush=True)


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
    Returns (xy, spread, n_frames) or (None, reason-string, n)."""
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

    # cross-type transfer check (fit-on-one, locate-the-other) — taskA diagnostics
    for src, dst in (("pad", "tray"), ("tray", "pad")):
        z_pred = a + b * calib[dst]["dz"]  # using the full map (trivially exact at the 2 anchors)
        rows_t = [r for r in rowsA if receptacle_type(r[0]) == dst]
        errs = []
        for g, pk, leaf, epn, meta in rows_t:
            gt = next((np.asarray(meta[k], float) for k in ("gt_pad_xy", "gt_tray_xy") if k in meta), None)
            xy, sp, n = place_xy_est(G, leaf, epn, meta, calib[dst]["z_star"], args.max_spread)
            if gt is not None and isinstance(xy, np.ndarray):
                errs.append(np.linalg.norm(xy - gt))

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
    thr_q = float(np.percentile(c, 99) * args.q_safety)
    accA_mid = 0.5 * (np.mean(c < thr_mid) + np.mean(x >= thr_mid))
    accA_q = 0.5 * (np.mean(c < thr_q) + np.mean(x >= thr_q))
    print(f"[vision place] taskA feats: center {c.mean()*100:.1f}±{c.std()*100:.1f}cm "
          f"(p99 {np.percentile(c,99)*100:.1f}) corner {x.mean()*100:.1f}±{x.std()*100:.1f}cm | rejected {rejA}")
    print(f"[vision place] thresholds: midpoint {thr_mid*100:.1f}cm (taskA acc {accA_mid:.3f}) | "
          f"quantile {thr_q*100:.1f}cm (taskA acc {accA_q:.3f})  [primary: {args.thr_rule}]")
    thr = thr_q if args.thr_rule == "quantile" else thr_mid

    # --- taskB: privilege-free path ---
    dz_B = leaf_dz(rowsB)                      # robot-measured receptacle height proxy
    z_B = float(a + b * dz_B)
    print(f"[vision place] taskB dz={dz_B:+.4f} -> z_plane={z_B:.4f}")
    cache, n_ok, n_kept, n_kept_ok = {}, 0, 0, 0
    labels = PREF_LABELS["place"]
    for g, gt_pk, leaf, epn, meta in rowsB:
        xy, sp, nfr = place_xy_est(G, leaf, epn, meta, z_B, args.max_spread)
        key = f"{leaf}/{epn}"
        if not isinstance(xy, np.ndarray):
            cache[key] = {"action_prompt_label": labels[SMALL['place']], "pref_key": SMALL["place"],
                          "decision": "reject", "reject_reason": str(sp),
                          "gt_pref_key": gt_pk, "gt_match": False}
            continue
        f = float(np.linalg.norm(np.asarray(meta["xy_t90"]) - xy))
        pred = "center" if f < thr else "corner"
        match = (pred == gt_pk)
        n_ok += int(match); n_kept += 1; n_kept_ok += int(match)
        cache[key] = {
            "action_prompt_label": labels[pred], "pref_key": pred, "decision": "keep",
            "feature": round(f, 4), "xy_est": [round(v, 4) for v in xy.tolist()],
            "frame_spread": round(sp, 4), "n_frames": nfr, "z_plane": round(z_B, 4),
            "gt_pref_key": gt_pk, "gt_match": match,
        }
    nB = len(rowsB)
    acc = n_ok / nB
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


# ---------------- contact ----------------

def contact_obj_geom(G, leaf, epn, meta, z_table, max_h_spread=0.03):
    """Object bottom xy + height from the bbox: bottom-center -> table plane;
    top row -> solve z at that xy. Median over frames."""
    bbs = ep_groundings(G, leaf, epn)
    if not bbs:
        return None
    K, E = meta["K"], meta["E"]
    hs = []
    for frac, bb in bbs.items():
        u_c, v_top, v_bot = 0.5 * (bb[0] + bb[2]), bb[1], bb[3]
        xy = backproject_to_plane(K, E, (u_c, v_bot), z_table)
        if xy is None:
            continue
        z_top = solve_z_at_xy(K, E, v_top, xy)
        if z_top is None:
            continue
        hs.append(z_top - z_table)
    if len(hs) < 2:
        return None
    hs = np.array(hs)
    h = float(np.median(hs))
    if h < 0.02 or (hs.max() - hs.min()) > max_h_spread:
        return None
    return h


def stage_label_contact(args):
    G = json.load(open(os.path.join(VISION_CACHE, "groundings_contact.json")))
    rowsA = collect_rows("contact", CONTACT_TASKA_GROUPS, limit=args.n_fit)
    rowsB = collect_rows("contact", CONTACT_TASKB_GROUPS)

    # --- joint calibration of (z_table, k) on taskA (hf GT legitimate there) ---
    # model: EE_z@grasp = k + hf * h_est(z_table); k absorbs table z + TCP offset.
    best = None
    for z_table in np.round(np.arange(0.70, 0.80, 0.0025), 4):
        recs = []
        for g, pk, leaf, epn, meta in rowsA:
            if meta.get("z_first_close") is None or "gt_height_fraction" not in meta:
                continue
            h = contact_obj_geom(G, leaf, epn, meta, z_table)
            if h is None:
                continue
            recs.append((meta["z_first_close"], meta["gt_height_fraction"], h))
        if len(recs) < 20:
            continue
        zg = np.array([r[0] for r in recs]); hf = np.array([r[1] for r in recs]); h = np.array([r[2] for r in recs])
        k = float(np.median(zg - hf * h))
        resid = np.abs((zg - k) / h - hf)
        med = float(np.median(resid))
        if best is None or med < best[0]:
            best = (med, z_table, k, len(recs))
    med, z_table, k, n_cal = best
    print(f"[vision contact] calib: z_table={z_table:.4f} k={k:.4f} "
          f"(median |hf_est − hf_gt| = {med:.3f}, n={n_cal})")

    # --- taskA same-estimator features + threshold ---
    featA = {"25": [], "75": []}
    rejA = 0
    for g, pk, leaf, epn, meta in rowsA:
        h = contact_obj_geom(G, leaf, epn, meta, z_table)
        if h is None or meta.get("z_first_close") is None:
            rejA += 1
            continue
        featA[pk].append((meta["z_first_close"] - k) / h)
    lo = np.array(featA["25"]); hi = np.array(featA["75"])
    thr = 0.5 * (lo.mean() + hi.mean())
    accA = 0.5 * (np.mean(lo < thr) + np.mean(hi >= thr))
    print(f"[vision contact] taskA hf_est: 25 {lo.mean():.3f}±{lo.std():.3f}  "
          f"75 {hi.mean():.3f}±{hi.std():.3f}  thr={thr:.3f}  acc={accA:.3f}  rejected {rejA}")

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
        match = (pred == gt_pk)
        n_ok += int(match); n_kept += 1; n_kept_ok += int(match)
        cache[key] = {
            "action_prompt_label": labels[pred], "pref_key": pred, "decision": "keep",
            "feature": round(f, 4), "h_est": round(h, 4),
            "gt_pref_key": gt_pk, "gt_match": match,
        }
    nB = len(rowsB)
    acc = n_ok / nB
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
    ap.add_argument("--thr-rule", default="quantile", choices=["quantile", "midpoint"],
                    help="place threshold rule; quantile = estimator-noise p99 x safety "
                         "(invariant to the source corner offset), midpoint = geom parity")
    ap.add_argument("--q-safety", type=float, default=1.5)
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
