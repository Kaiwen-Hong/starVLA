#!/usr/bin/env python3
"""Aggregate the 0529 paired-controllability eval into summary.json + summary.md.

Reads per_episode/<ckpt_run>/<env>__prompt_<pk>/*.json (written by pref_metric via the
joint bridge) and reports, per category, the two-prompt means + separation + per-episode
follow-rate (fraction on the correct side of the class midpoint). Task success rates are
taken from the eval logs (recipe-bound check_success) and recorded alongside for context.
"""
import json, glob, os, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PE = os.path.join(HERE, "per_episode")
PE2 = os.path.join(HERE, "..", "0611_ctrl")

# axis -> (lower-value pref, higher-value pref)
AXIS_ORDER = {
    "drop_height":      ("low", "high"),
    "grasp_height":     ("25", "75"),
    "grasp_tilt_deg":   ("90", "0"),
    "place_offset_m":   ("center", "corner"),
    "obstacle_clear_m": ("lv", "hv"),
}
# scale of the eval per ckpt (height/orient = 50-ep; geom = 5-ep sample)
SCALE = {"pref_stageb_main_height": "50ep", "pref_stageb_main_orient": "50ep",
         "pref_stageb_b0_height": "10ep-0611", "pref_stageb_b0_orient": "10ep-0611", "pref_stageb_main_orient_geom": "10ep-0611",
         "pref_stageb_main_place_geom": "5ep-sample", "pref_stageb_main_contact_geom": "5ep-sample",
         "pref_stageb_main_hvlv_geom": "5ep-sample"}
# task success (recipe-bound check_success) from eval logs, per (ckpt, pref): succ/total
SUCCESS = {
    "pref_stageb_b0_height":  {"high": "11/11", "low": "11/11"},
    "pref_stageb_b0_orient":  {"0": "10/11", "90": "10/11"},
    "pref_stageb_main_orient_geom": {"0": "11/11", "90": "9/11"},
    "pref_stageb_main_height":      {"high": "50/50", "low": "41/50"},
    "pref_stageb_main_orient":      {"0": "48/50", "90": "43/50"},
    "pref_stageb_main_place_geom":  {"center": "4/5", "corner": "0/5"},
    "pref_stageb_main_contact_geom":{"25": "4/5", "75": "4/5"},
    "pref_stageb_main_hvlv_geom":   {"hv": "0/5", "lv": "0/5"},
}
METRIC_NAME = {"drop_height": "drop_height (m)", "grasp_tilt_deg": "grasp_tilt ee_x (deg)",
               "place_offset_m": "EE@release->receptacle (m)", "grasp_height": "EE.z-obj.z @grasp (m)",
               "obstacle_clear_m": "max EE detour (m)"}

# fixed source(taskA)-statistics thresholds (official follow convention, 2026-06-11)
SRC_THR = {}
_thr_path = os.path.join(HERE, "..", "source_follow_thresholds.json")
if os.path.exists(_thr_path):
    SRC_THR = {ax: e["threshold"] for ax, e in json.load(open(_thr_path)).items()}

summary = {}
for run_dir in sorted(glob.glob(os.path.join(PE, "*")) + glob.glob(os.path.join(PE2, "pref_stageb_*"))):
    run = os.path.basename(run_dir)
    by = defaultdict(list)
    axis = None
    for f in glob.glob(os.path.join(run_dir, "*", "*.json")):
        d = json.load(open(f))
        if d.get("value") is None:
            continue
        axis = d["axis"]
        by[str(d["preference"])].append(float(d["value"]))
    if axis is None:
        continue
    import statistics as st
    means = {p: sum(v) / len(v) for p, v in by.items()}
    lo, hi = AXIS_ORDER.get(axis, (None, None))
    entry = {"axis": axis, "metric": METRIC_NAME.get(axis, axis), "scale": SCALE.get(run, "?"),
             "prompts": {}}
    for p, v in sorted(by.items()):
        entry["prompts"][p] = {"n": len(v), "mean": round(means[p], 4),
                               "std": round(st.pstdev(v), 4) if len(v) > 1 else 0.0,
                               "success": SUCCESS.get(run, {}).get(p),
                               "vals": [round(x, 3) for x in v]}
    if lo in means and hi in means:
        thr = (means[lo] + means[hi]) / 2.0
        hi_ok = sum(1 for x in by[hi] if x > thr) / len(by[hi])
        lo_ok = sum(1 for x in by[lo] if x < thr) / len(by[lo])
        entry["separation"] = round(means[hi] - means[lo], 4)
        entry["midpoint"] = round(thr, 4)
        entry["follow"] = {hi: round(hi_ok, 2), lo: round(lo_ok, 2), "overall": round((hi_ok + lo_ok) / 2, 2)}
        entry["direction"] = f"{hi} > {lo}"
        # OFFICIAL convention (2026-06-11, user-approved): FIXED source(taskA)-statistics
        # threshold per axis (../source_follow_thresholds.json). The per-run midpoint
        # above flatters weak separations (it adapts to each run); keep it for reference.
        if axis in SRC_THR:
            ft = SRC_THR[axis]
            # contact guard: src thr is a height_fraction; old-box records fell back to
            # EE.z-obj.z meters (all < 0.25) -> units mismatch, fixed column is n/a.
            if axis == "grasp_height" and max(max(v) for v in by.values()) < 0.25:
                entry["follow_fixed_note"] = "n/a: records use fallback meters, src thr is a fraction"
                ft = None
            if ft is not None:
                hi_f = sum(1 for x in by[hi] if x > ft) / len(by[hi])
                lo_f = sum(1 for x in by[lo] if x < ft) / len(by[lo])
                entry["src_threshold"] = round(ft, 4)
                entry["follow_fixed"] = {hi: round(hi_f, 2), lo: round(lo_f, 2),
                                         "overall": round((hi_f + lo_f) / 2, 2)}
    summary[run] = entry

json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=2)

# markdown table
rows = []
order = ["pref_stageb_main_height", "pref_stageb_b0_height", "pref_stageb_main_orient", "pref_stageb_main_orient_geom", "pref_stageb_b0_orient", "pref_stageb_main_place_geom",
         "pref_stageb_main_contact_geom", "pref_stageb_main_hvlv_geom"]
lines = ["| ckpt (cat) | scale | metric | prompt-A (mean,n,succ) | prompt-B (mean,n,succ) | separation | follow (midpoint) | follow (FIXED src-thr) |",
         "|---|---|---|---|---|---|---|---|"]
for run in order:
    e = summary.get(run)
    if not e:
        continue
    ps = list(e["prompts"].items())
    def cell(p, x):
        return f"{p}: {x['mean']} (n={x['n']}, {x['success']})"
    a, b = ps[0], ps[1]
    sep = e.get("separation", "—")
    fo = e.get("follow", {}).get("overall", "—")
    ff = e.get("follow_fixed", {}).get("overall", "—")
    lines.append(f"| {run} | {e['scale']} | {e['metric']} | {cell(*a)} | {cell(*b)} | {sep} | {fo} | {ff} |")
open(os.path.join(HERE, "summary.md"), "w").write("\n".join(lines) + "\n")
print("wrote summary.json + summary.md")
print("\n".join(lines))
