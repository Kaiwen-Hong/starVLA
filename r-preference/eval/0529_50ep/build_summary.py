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
         "pref_stageb_main_place_geom": "5ep-sample", "pref_stageb_main_contact_geom": "5ep-sample",
         "pref_stageb_main_hvlv_geom": "5ep-sample"}
# task success (recipe-bound check_success) from eval logs, per (ckpt, pref): succ/total
SUCCESS = {
    "pref_stageb_main_height":      {"high": "50/50", "low": "41/50"},
    "pref_stageb_main_orient":      {"0": "48/50", "90": "43/50"},
    "pref_stageb_main_place_geom":  {"center": "4/5", "corner": "0/5"},
    "pref_stageb_main_contact_geom":{"25": "4/5", "75": "4/5"},
    "pref_stageb_main_hvlv_geom":   {"hv": "0/5", "lv": "0/5"},
}
METRIC_NAME = {"drop_height": "drop_height (m)", "grasp_tilt_deg": "grasp_tilt ee_x (deg)",
               "place_offset_m": "EE@release->receptacle (m)", "grasp_height": "EE.z-obj.z @grasp (m)",
               "obstacle_clear_m": "max EE detour (m)"}

summary = {}
for run_dir in sorted(glob.glob(os.path.join(PE, "*"))):
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
    summary[run] = entry

json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=2)

# markdown table
rows = []
order = ["pref_stageb_main_height", "pref_stageb_main_orient", "pref_stageb_main_place_geom",
         "pref_stageb_main_contact_geom", "pref_stageb_main_hvlv_geom"]
lines = ["| ckpt (cat) | scale | metric | prompt-A (mean,n,succ) | prompt-B (mean,n,succ) | separation | follow overall |",
         "|---|---|---|---|---|---|---|"]
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
    lines.append(f"| {run} | {e['scale']} | {e['metric']} | {cell(*a)} | {cell(*b)} | {sep} | {fo} |")
open(os.path.join(HERE, "summary.md"), "w").write("\n".join(lines) + "\n")
print("wrote summary.json + summary.md")
print("\n".join(lines))
