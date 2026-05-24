"""
Analyze sanity_pref_flip dump JSONs across multiple ckpts.

Reads r-preference/eval/stageb_diag/*.json (each contains per-episode
a_low/a_high chunks under "records") and aggregates:
  - per-axis sign-acc (which action dim carries pref signal?)
  - per-chunk-step sign-acc (when in the action chunk does pref signal peak?)
  - magnitude per axis (mean abs Δ on axis)
  - cross-ckpt trajectory (Phase A: is sign-acc improving with steps?)

Usage:
  python -m examples.preference.stage_b.analyze_pref_flip r-preference/eval/stageb_diag/
"""

import json
import sys
from pathlib import Path

import numpy as np

# Action layout (from 0522-doc §5.1):
ACTION_DIMS = [
    "L_x", "L_y", "L_z",
    "L_6d_0", "L_6d_1", "L_6d_2", "L_6d_3", "L_6d_4", "L_6d_5",
    "L_grip",
    "R_x", "R_y", "R_z",
    "R_6d_0", "R_6d_1", "R_6d_2", "R_6d_3", "R_6d_4", "R_6d_5",
    "R_grip",
]


def load_run(path: Path) -> dict:
    d = json.loads(path.read_text())
    return d


def per_axis_sign_acc(records, axis, expected_sign=+1):
    """For each episode, take Δa[k_max_on_this_axis, axis] sign. Return acc."""
    correct = 0
    abs_max_sum = 0.0
    for r in records:
        a_low = np.asarray(r["a_low"])   # (T, 20)
        a_high = np.asarray(r["a_high"])
        delta = a_high[:, axis] - a_low[:, axis]  # (T,)
        k = int(np.argmax(np.abs(delta)))
        d = float(delta[k])
        if np.sign(d) == expected_sign:
            correct += 1
        abs_max_sum += abs(d)
    return correct, abs_max_sum / max(1, len(records))


def per_step_sign_acc(records, axis, expected_sign=+1):
    """For each chunk step k, count sign-acc across all episodes."""
    T = len(records[0]["a_low"])
    out = []
    for k in range(T):
        cor = 0
        for r in records:
            a_low = np.asarray(r["a_low"])
            a_high = np.asarray(r["a_high"])
            delta = a_high[k, axis] - a_low[k, axis]
            if np.sign(delta) == expected_sign:
                cor += 1
        out.append(cor / len(records))
    return out


def summarize_run(d):
    """Return dict of axis-wise summary."""
    recs = d["records"]
    if not recs:
        return None
    # Determine expected sign per axis (default +1 for L_z; for others unknown)
    out = {"tag": Path(d.get("policy_ckpt", "?")).name, "n_ep": len(recs)}
    out["per_axis"] = {}
    for axis_idx, axis_name in enumerate(ACTION_DIMS):
        cor, mag = per_axis_sign_acc(recs, axis_idx, expected_sign=+1)
        # Also try -1 expected sign and report whichever is higher
        cor_neg, _ = per_axis_sign_acc(recs, axis_idx, expected_sign=-1)
        best_cor = max(cor, cor_neg)
        best_sign = +1 if cor >= cor_neg else -1
        out["per_axis"][axis_name] = {
            "sign_acc_pos1": cor / len(recs),  # expected +1
            "sign_acc_neg1": cor_neg / len(recs),
            "best_sign": best_sign,
            "best_acc": best_cor / len(recs),
            "mean_abs_max_delta": mag,
        }
    # Also report sign-acc per chunk step for L_z (axis 2)
    out["per_step_Lz_acc"] = per_step_sign_acc(recs, 2, expected_sign=+1)
    return out


def print_main_table(summaries):
    """Cross-ckpt comparison on the user-chosen axis (L_z, +1)."""
    print("\n" + "=" * 90)
    print(f"L_z (axis 2) sign-acc with expected_sign=+1 across all runs:")
    print("=" * 90)
    print(f"  {'tag':<25}  {'N':>3}  {'L_z sign-acc':>12}  {'mean |Δ| L_z':>13}  {'best axis':<10}  {'best acc':>8}")
    for s in summaries:
        ax = s["per_axis"]["L_z"]
        # Find best axis across all 20
        best = max(s["per_axis"].items(), key=lambda kv: kv[1]["best_acc"])
        print(f"  {s['tag'][:25]:<25}  {s['n_ep']:>3}  "
              f"{ax['sign_acc_pos1']:>12.2f}  {ax['mean_abs_max_delta']:>13.4f}  "
              f"{best[0]:<10}  {best[1]['best_acc']:>8.2f}")


def print_per_axis_table(summaries, top_n=8):
    """For each run, show top-N axes by sign-acc."""
    print("\n" + "=" * 90)
    print(f"Per-axis sign-acc (top-{top_n} per run, best_sign included)")
    print("=" * 90)
    for s in summaries:
        print(f"\n  {s['tag']}  (N={s['n_ep']})")
        # Sort axes by best_acc desc
        items = sorted(s["per_axis"].items(), key=lambda kv: -kv[1]["best_acc"])
        print(f"    {'axis':<10}  {'best_sign':>9}  {'best_acc':>8}  {'mean |Δ|':>10}")
        for name, ax in items[:top_n]:
            print(f"    {name:<10}  {ax['best_sign']:>+9d}  {ax['best_acc']:>8.2f}  {ax['mean_abs_max_delta']:>10.4f}")


def print_per_step_table(summaries):
    """Per-chunk-step sign-acc on L_z (when does pref signal peak in chunk?)"""
    print("\n" + "=" * 90)
    print(f"Per-chunk-step sign-acc (L_z axis, expected +1)")
    print("=" * 90)
    print(f"  {'tag':<25}  " + "  ".join([f"k={k:>2}" for k in range(16)]))
    for s in summaries:
        per_step = s["per_step_Lz_acc"]
        row = "  ".join([f"{v:>4.1f}" for v in per_step])
        print(f"  {s['tag'][:25]:<25}  {row}")


def main():
    diag_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("r-preference/eval/stageb_diag")
    paths = sorted(diag_dir.glob("*.json"))
    print(f"Loaded {len(paths)} runs from {diag_dir}")

    # Sort: stageA_baseline, stageA_VQA, main_step500-3000, b0
    def sort_key(p):
        n = p.stem
        if "stageA_baseline" in n: return (0, 0)
        if "stageA_VQA" in n: return (0, 1)
        if n.startswith("main_step"):
            try: return (1, int(n.replace("main_step", "")))
            except: return (1, 9999)
        if n.startswith("b0"): return (2, 0)
        return (3, 0)
    paths = sorted(paths, key=sort_key)

    summaries = []
    for p in paths:
        d = load_run(p)
        s = summarize_run(d)
        if s:
            s["tag"] = p.stem
            summaries.append(s)
            print(f"  {p.stem}: n_ep={s['n_ep']}")

    if not summaries:
        print("no records found")
        return

    print_main_table(summaries)
    print_per_axis_table(summaries, top_n=8)
    print_per_step_table(summaries)


if __name__ == "__main__":
    main()
