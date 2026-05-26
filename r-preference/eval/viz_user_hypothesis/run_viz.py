"""Visual evidence for pref signal in hvlv and height datasets.

Gripper convention: 0 = closed, 1 = open.
Transport segment: [grasp_t, release_t] where
  grasp_t   = first index where gripper goes 1 -> 0 (open -> close)
  release_t = last index where gripper goes 0 -> 1 (close -> open)
Active arm per scene = scene_info['info']['{a}'] ('left' or 'right').
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("/home/kaiwenh/starVLA/r-preference/eval/viz_user_hypothesis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_EPS = 10  # episodes per pref for trajectory overlays
HVLV_BASE = Path("/mnt/localssd/kaiwenh/pref/data/hvlv")
HEIGHT_BASE = Path("/mnt/localssd/kaiwenh/pref/data/height")


# ---------- helpers ----------
def find_grasp_release(gripper: np.ndarray, thr: float = 0.5):
    """Return (grasp_t, release_t) using thresholded transitions.

    grasp_t = first 1->0 (open->close).
    release_t = last 0->1 (close->open).
    Returns (None, None) if either is not found.
    """
    binary = (gripper > thr).astype(np.int8)  # 1 = open, 0 = closed
    diff = np.diff(binary)
    close_idxs = np.where(diff == -1)[0]  # transitions to closed
    open_idxs = np.where(diff == 1)[0]    # transitions to open
    if len(close_idxs) == 0 or len(open_idxs) == 0:
        return None, None
    grasp_t = int(close_idxs[0]) + 1
    release_t = int(open_idxs[-1]) + 1
    if release_t <= grasp_t:
        return None, None
    return grasp_t, release_t


def pick_active_arm(left_xyz: np.ndarray, right_xyz: np.ndarray) -> str:
    """Active arm = arm with larger xyz range across episode."""
    def total_range(xyz):
        return float((xyz.max(0) - xyz.min(0)).sum())
    return "left" if total_range(left_xyz) >= total_range(right_xyz) else "right"


def load_episode(hdf5_path: Path):
    with h5py.File(hdf5_path, "r") as f:
        left_ep = f["endpose/left_endpose"][:]
        right_ep = f["endpose/right_endpose"][:]
        left_g = f["endpose/left_gripper"][:]
        right_g = f["endpose/right_gripper"][:]
    return left_ep[:, :3], right_ep[:, :3], left_g, right_g


def load_scene_info(task_dir: Path) -> dict:
    with open(task_dir / "scene_info.json") as f:
        return json.load(f)


def collect_episodes(task_dir: Path, pref_arm_field: str = "{a}", n: int = N_EPS,
                     use_scene_arm: bool = True):
    """Collect first n episodes, returning list of dicts with active-arm info."""
    sinfo = load_scene_info(task_dir)
    out = []
    for i in range(n):
        ep_path = task_dir / "data" / f"episode{i}.hdf5"
        if not ep_path.exists():
            continue
        l_xyz, r_xyz, l_g, r_g = load_episode(ep_path)
        scene_arm = sinfo.get(f"episode_{i}", {}).get("info", {}).get(pref_arm_field)
        if use_scene_arm and scene_arm in ("left", "right"):
            arm = scene_arm
        else:
            arm = pick_active_arm(l_xyz, r_xyz)
        if arm == "left":
            xyz, grip = l_xyz, l_g
        else:
            xyz, grip = r_xyz, r_g
        grasp_t, release_t = find_grasp_release(grip)
        out.append({
            "ep": i,
            "arm": arm,
            "xyz": xyz,
            "grip": grip,
            "grasp_t": grasp_t,
            "release_t": release_t,
            "scene_info": sinfo.get(f"episode_{i}", {}).get("info", {}),
        })
    return out


# ---------- 1) hvlv xy trajectory overlays ----------
def hvlv_plot(task_name: str, ax):
    hv_dir = HVLV_BASE / f"{task_name}_hv"
    lv_dir = HVLV_BASE / f"{task_name}_lv"
    hv_eps = collect_episodes(hv_dir)
    lv_eps = collect_episodes(lv_dir)

    def plot_set(eps, color, label):
        plotted = 0
        for e in eps:
            if e["grasp_t"] is None:
                continue
            seg = e["xyz"][e["grasp_t"] : e["release_t"] + 1]
            ax.plot(seg[:, 0], seg[:, 1], color=color, alpha=0.6, linewidth=1.4,
                    label=label if plotted == 0 else None)
            ax.scatter(seg[0, 0], seg[0, 1], color=color, s=18, marker="o",
                       edgecolor="black", linewidth=0.4, zorder=3)
            ax.scatter(seg[-1, 0], seg[-1, 1], color=color, s=24, marker="X",
                       edgecolor="black", linewidth=0.4, zorder=3)
            plotted += 1
        return plotted

    n_hv = plot_set(hv_eps, "red", "hv (high-volume detour)")
    n_lv = plot_set(lv_eps, "blue", "lv (low-volume close)")
    ax.set_title(f"{task_name}\nhv={n_hv}  lv={n_lv} (transport seg, active arm xy)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    # Compute summary metrics: path length in xy & max |x| deviation (proxy for detour width)
    def metric(eps):
        plens, maxabsx = [], []
        for e in eps:
            if e["grasp_t"] is None:
                continue
            seg = e["xyz"][e["grasp_t"] : e["release_t"] + 1]
            d = np.linalg.norm(np.diff(seg[:, :2], axis=0), axis=1).sum()
            plens.append(float(d))
            maxabsx.append(float(np.max(np.abs(seg[:, 0] - seg[0, 0]))))
        return np.array(plens), np.array(maxabsx)

    hv_pl, hv_mx = metric(hv_eps)
    lv_pl, lv_mx = metric(lv_eps)
    return {
        "task": task_name,
        "hv_pathlen_m": (float(hv_pl.mean()), float(hv_pl.std())),
        "lv_pathlen_m": (float(lv_pl.mean()), float(lv_pl.std())),
        "hv_max_x_dev_m": (float(hv_mx.mean()), float(hv_mx.std())),
        "lv_max_x_dev_m": (float(lv_mx.mean()), float(lv_mx.std())),
        "n_hv": n_hv, "n_lv": n_lv,
    }


def fig_hvlv():
    tasks = ["place_apple_plate", "place_cup_plate"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 6))
    if len(tasks) == 1:
        axes = [axes]
    metrics = []
    for ax, t in zip(axes, tasks):
        metrics.append(hvlv_plot(t, ax))
    fig.suptitle("hvlv: hv (red) vs lv (blue) active-arm transport xy", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "1_hvlv_xy_trajectories.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out, metrics


# ---------- 2) height: z_grasp, z_release, z(t) ----------
def height_collect(task_dir: Path, n: int = None):
    """Collect all episodes (or first n)."""
    sinfo = load_scene_info(task_dir)
    eps = sorted(int(k.split("_")[1]) for k in sinfo.keys() if k.startswith("episode_"))
    if n is not None:
        eps = eps[:n]
    out = []
    for i in eps:
        ep_path = task_dir / "data" / f"episode{i}.hdf5"
        if not ep_path.exists():
            continue
        l_xyz, r_xyz, l_g, r_g = load_episode(ep_path)
        scene = sinfo.get(f"episode_{i}", {}).get("info", {})
        arm = scene.get("{a}")
        if arm not in ("left", "right"):
            arm = pick_active_arm(l_xyz, r_xyz)
        xyz = l_xyz if arm == "left" else r_xyz
        grip = l_g if arm == "left" else r_g
        grasp_t, release_t = find_grasp_release(grip)
        if grasp_t is None:
            continue
        out.append({
            "ep": i,
            "arm": arm,
            "xyz": xyz,
            "grip": grip,
            "grasp_t": grasp_t,
            "release_t": release_t,
            "z_grasp": float(xyz[grasp_t, 2]),
            "z_release": float(xyz[release_t, 2]),
            "z_max_in_transport": float(xyz[grasp_t:release_t + 1, 2].max()),
            "scene_info": scene,
        })
    return out


def height_plots(task_pair: str = "move_mouse_pad"):
    hi = height_collect(HEIGHT_BASE / f"{task_pair}_high")
    lo = height_collect(HEIGHT_BASE / f"{task_pair}_low")

    z_grasp_hi = np.array([e["z_grasp"] for e in hi])
    z_grasp_lo = np.array([e["z_grasp"] for e in lo])
    z_rel_hi = np.array([e["z_release"] for e in hi])
    z_rel_lo = np.array([e["z_release"] for e in lo])
    z_max_hi = np.array([e["z_max_in_transport"] for e in hi])
    z_max_lo = np.array([e["z_max_in_transport"] for e in lo])

    def sn(a, b):
        d = abs(a.mean() - b.mean())
        s = np.sqrt(0.5 * (a.var() + b.var()))
        return d / s if s > 0 else float("inf")

    stats = {
        "task_pair": task_pair,
        "n_hi": len(hi), "n_lo": len(lo),
        "z_grasp_hi_mean_mm": float(z_grasp_hi.mean() * 1000),
        "z_grasp_lo_mean_mm": float(z_grasp_lo.mean() * 1000),
        "z_grasp_delta_mm": float((z_grasp_hi.mean() - z_grasp_lo.mean()) * 1000),
        "z_grasp_SN": float(sn(z_grasp_hi, z_grasp_lo)),
        "z_release_hi_mean_mm": float(z_rel_hi.mean() * 1000),
        "z_release_lo_mean_mm": float(z_rel_lo.mean() * 1000),
        "z_release_delta_mm": float((z_rel_hi.mean() - z_rel_lo.mean()) * 1000),
        "z_release_SN": float(sn(z_rel_hi, z_rel_lo)),
        "z_max_hi_mean_mm": float(z_max_hi.mean() * 1000),
        "z_max_lo_mean_mm": float(z_max_lo.mean() * 1000),
        "z_max_delta_mm": float((z_max_hi.mean() - z_max_lo.mean()) * 1000),
        "z_max_SN": float(sn(z_max_hi, z_max_lo)),
    }

    # Figure: (a) histogram z_grasp, (b) hist z_release, (c) z(t) overlays
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3)

    ax_a = fig.add_subplot(gs[0, 0])
    bins = 25
    ax_a.hist(z_grasp_hi * 1000, bins=bins, alpha=0.6, color="red", label=f"high (n={len(hi)})")
    ax_a.hist(z_grasp_lo * 1000, bins=bins, alpha=0.6, color="blue", label=f"low (n={len(lo)})")
    ax_a.axvline(z_grasp_hi.mean() * 1000, color="red", ls="--", lw=1)
    ax_a.axvline(z_grasp_lo.mean() * 1000, color="blue", ls="--", lw=1)
    ax_a.set_title(f"(a) z at grasp moment\nΔ={stats['z_grasp_delta_mm']:.1f}mm  S/N={stats['z_grasp_SN']:.2f}")
    ax_a.set_xlabel("z [mm]"); ax_a.legend()

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.hist(z_rel_hi * 1000, bins=bins, alpha=0.6, color="red", label=f"high (n={len(hi)})")
    ax_b.hist(z_rel_lo * 1000, bins=bins, alpha=0.6, color="blue", label=f"low (n={len(lo)})")
    ax_b.axvline(z_rel_hi.mean() * 1000, color="red", ls="--", lw=1)
    ax_b.axvline(z_rel_lo.mean() * 1000, color="blue", ls="--", lw=1)
    ax_b.set_title(f"(b) z at release moment\nΔ={stats['z_release_delta_mm']:.1f}mm  S/N={stats['z_release_SN']:.2f}")
    ax_b.set_xlabel("z [mm]"); ax_b.legend()

    ax_d = fig.add_subplot(gs[0, 2])
    ax_d.hist(z_max_hi * 1000, bins=bins, alpha=0.6, color="red", label=f"high (n={len(hi)})")
    ax_d.hist(z_max_lo * 1000, bins=bins, alpha=0.6, color="blue", label=f"low (n={len(lo)})")
    ax_d.axvline(z_max_hi.mean() * 1000, color="red", ls="--", lw=1)
    ax_d.axvline(z_max_lo.mean() * 1000, color="blue", ls="--", lw=1)
    ax_d.set_title(f"max z in transport\nΔ={stats['z_max_delta_mm']:.1f}mm  S/N={stats['z_max_SN']:.2f}")
    ax_d.set_xlabel("z [mm]"); ax_d.legend()

    # (c) z(t) overlays - 10 each, time-aligned to grasp_t
    ax_c = fig.add_subplot(gs[1, :])
    for grp, color, lbl in [(hi[:10], "red", "high"), (lo[:10], "blue", "low")]:
        first = True
        for e in grp:
            t0 = e["grasp_t"]; t1 = e["release_t"]
            seg = e["xyz"][t0:t1 + 1, 2] * 1000  # mm
            ax_c.plot(np.arange(seg.shape[0]), seg, color=color, alpha=0.6,
                      lw=1.4, label=(lbl if first else None))
            first = False
    ax_c.set_title("(c) z(t) over transport segment (10 eps each, t=0 at grasp)")
    ax_c.set_xlabel("frame since grasp"); ax_c.set_ylabel("z [mm]")
    ax_c.legend(); ax_c.grid(True, alpha=0.3)

    fig.suptitle(f"height: {task_pair} — high (red) vs low (blue)", fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / f"2_height_{task_pair}_z_analysis.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out, stats, hi, lo


# ---------- 3) scene_info pref field investigation ----------
def scene_pref_audit(task_pair: str = "move_mouse_pad"):
    hi = load_scene_info(HEIGHT_BASE / f"{task_pair}_high")
    lo = load_scene_info(HEIGHT_BASE / f"{task_pair}_low")
    keys_hi = set(hi["episode_0"]["info"].keys())
    keys_lo = set(lo["episode_0"]["info"].keys())
    # Union of all info keys across all episodes
    all_keys_hi = set()
    for k, v in hi.items():
        all_keys_hi.update(v.get("info", {}).keys())
    all_keys_lo = set()
    for k, v in lo.items():
        all_keys_lo.update(v.get("info", {}).keys())

    def field_stats(d, field):
        vals = []
        for k, v in d.items():
            val = v.get("info", {}).get(field)
            if val is None:
                continue
            vals.append(val)
        return vals

    fields = sorted(all_keys_hi | all_keys_lo)
    # Check numeric fields for separability
    pref_candidates = {}
    for f in fields:
        vh = field_stats(hi, f)
        vl = field_stats(lo, f)
        # Only numeric (scalar)
        try:
            vh_arr = np.array([float(x) for x in vh if not isinstance(x, list)])
            vl_arr = np.array([float(x) for x in vl if not isinstance(x, list)])
            if len(vh_arr) < 5 or len(vl_arr) < 5:
                continue
            d = abs(vh_arr.mean() - vl_arr.mean())
            s = np.sqrt(0.5 * (vh_arr.var() + vl_arr.var()))
            sn = d / s if s > 0 else float("inf")
            pref_candidates[f] = {
                "hi_mean": float(vh_arr.mean()),
                "lo_mean": float(vl_arr.mean()),
                "hi_std": float(vh_arr.std()),
                "lo_std": float(vl_arr.std()),
                "delta": float(vh_arr.mean() - vl_arr.mean()),
                "SN": float(sn),
                "n_hi": int(len(vh_arr)),
                "n_lo": int(len(vl_arr)),
            }
        except Exception:
            pass
    return {
        "task_pair": task_pair,
        "info_keys_hi": sorted(all_keys_hi),
        "info_keys_lo": sorted(all_keys_lo),
        "numeric_field_separability": pref_candidates,
    }


def main():
    print("=== 1) hvlv xy trajectories ===")
    p1, m1 = fig_hvlv()
    print("wrote", p1)
    for d in m1:
        print(" ", d)

    print("\n=== 2) height z analysis: move_mouse_pad ===")
    p2a, s2a, hi_a, lo_a = height_plots("move_mouse_pad")
    print("wrote", p2a)
    print(" stats:", json.dumps(s2a, indent=2))

    print("\n=== 2) height z analysis: place_mouse_stand ===")
    p2b, s2b, hi_b, lo_b = height_plots("place_mouse_stand")
    print("wrote", p2b)
    print(" stats:", json.dumps(s2b, indent=2))

    print("\n=== 3) scene_info pref field audit: move_mouse_pad ===")
    audit_a = scene_pref_audit("move_mouse_pad")
    print(json.dumps(audit_a, indent=2))

    print("\n=== 3) scene_info pref field audit: place_mouse_stand ===")
    audit_b = scene_pref_audit("place_mouse_stand")
    print(json.dumps(audit_b, indent=2))

    # Save consolidated summary
    summary = {
        "hvlv": m1,
        "height_move_mouse_pad": s2a,
        "height_place_mouse_stand": s2b,
        "scene_info_audit_move_mouse_pad": audit_a,
        "scene_info_audit_place_mouse_stand": audit_b,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
