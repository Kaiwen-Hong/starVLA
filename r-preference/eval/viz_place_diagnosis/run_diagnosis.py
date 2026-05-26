"""
Visual diagnosis for place taskB VQA transfer failure.

Hypotheses under test:
(a) taskB scene is too OOD from place taskA training distribution
(b) center vs corner is not visually obvious on the soap stand target
"""
import io
import os
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path("/mnt/localssd/kaiwenh/pref/data/place")
TASKB_ROOT = ROOT / "taskB"
OUT = Path("/home/kaiwenh/starVLA/r-preference/eval/viz_place_diagnosis")
OUT.mkdir(parents=True, exist_ok=True)

# 8 taskA task groups (object_target pairs). For each there's _center and _corner.
TASKA_GROUPS = [
    "move_mouse_pad",
    "move_pillbottle_pad",
    "move_playingcards_pad",
    "move_soap_pad",
    "place_mouse_tray",
    "place_pillbottle_tray",
    "place_playingcards_tray",
    "place_soap_tray",
]
TASKB_GROUPS = ["place_soap2_stand"]
PREFS = ["center", "corner"]


def load_head_frame(hdf5_path: Path, frac: float):
    """Decode JPEG bytes for head_camera frame at fractional index."""
    with h5py.File(hdf5_path, "r") as f:
        rgb = f["observation/head_camera/rgb"]
        T = rgb.shape[0]
        idx = int(round(frac * (T - 1)))
        idx = max(0, min(T - 1, idx))
        raw = rgb[idx]
        if isinstance(raw, bytes):
            data = raw
        else:
            data = bytes(raw)
    img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    return img, idx, T


def get_dir(group: str, pref: str, taskB: bool = False) -> Path:
    base = TASKB_ROOT if taskB else ROOT
    return base / f"{group}_{pref}"


def list_episodes(d: Path):
    # Episodes live under d/data/episode*.hdf5
    data_dir = d / "data"
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("episode*.hdf5"), key=lambda p: int(p.stem.replace("episode", "")))


# ============================================================
# PART 1: Placement-moment grid (frac=0.91)
# ============================================================
def part1_placement_grid():
    """2 rows (center, corner) x 9 cols (8 taskA + taskB)."""
    frac = 0.91
    n_cols = len(TASKA_GROUPS) + len(TASKB_GROUPS)  # 9
    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 5.6))
    fig.suptitle(f"Place taskA vs taskB - head_camera at frac={frac} (release moment)", fontsize=12)
    for row, pref in enumerate(PREFS):
        for col, group in enumerate(TASKA_GROUPS + TASKB_GROUPS):
            taskB = group in TASKB_GROUPS
            d = get_dir(group, pref, taskB=taskB)
            eps = list_episodes(d)
            if not eps:
                axes[row, col].axis("off")
                axes[row, col].set_title(f"{group}\n(missing)", fontsize=7)
                continue
            ep = eps[0]
            img, idx, T = load_head_frame(ep, frac)
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")
            label = group.replace("place_", "p_").replace("move_", "m_")
            if taskB:
                label = f"[B] {label}"
            ax.set_title(f"{label}\n{pref} (T={T},i={idx})", fontsize=7)
    plt.tight_layout()
    out = OUT / "placement_grid.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[part1] saved {out}")


# ============================================================
# PART 2: TaskB close-up sequence (6 fracs, 2 prefs)
# ============================================================
def part2_taskB_sequence():
    fracs = [0.85, 0.88, 0.91, 0.94, 0.97, 1.0]
    n_cols = len(fracs)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.7 * n_cols, 5.6))
    fig.suptitle("TaskB place_soap2_stand - head_camera placement sequence (episode0)", fontsize=12)
    for row, pref in enumerate(PREFS):
        d = TASKB_ROOT / f"place_soap2_stand_{pref}"
        eps = list_episodes(d)
        ep = eps[0]
        for col, frac in enumerate(fracs):
            img, idx, T = load_head_frame(ep, frac)
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{pref} frac={frac} (i={idx}/{T-1})", fontsize=8)
    plt.tight_layout()
    out = OUT / "placement_taskB_sequence.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[part2] saved {out}")


# ============================================================
# PART 3: Pixel RGB mean / std distance
# ============================================================
def part3_rgb_distance():
    frac = 0.91

    def collect(group, pref, taskB, n_max):
        d = get_dir(group, pref, taskB=taskB)
        eps = list_episodes(d)[:n_max]
        means = []
        stds = []
        for ep in eps:
            try:
                img, _, _ = load_head_frame(ep, frac)
            except Exception as e:
                print(f"  failed {ep}: {e}")
                continue
            arr = img.astype(np.float32) / 255.0
            means.append(arr.reshape(-1, 3).mean(axis=0))
            stds.append(arr.reshape(-1, 3).std(axis=0))
        return np.stack(means), np.stack(stds)

    table_rows = []
    print("\n=== Per-task RGB statistics (head_camera at frac=0.91) ===")
    print(f"{'task':<40s} {'pref':<7s} {'n':>3s}  mean(R,G,B)            std(R,G,B)")
    cache = {}  # (group, pref) -> means stack
    for group in TASKA_GROUPS:
        for pref in PREFS:
            means, stds = collect(group, pref, taskB=False, n_max=5)
            cache[(group, pref)] = means
            mn = means.mean(axis=0)
            sd = stds.mean(axis=0)
            print(f"{group:<40s} {pref:<7s} {means.shape[0]:>3d}  "
                  f"({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f})  "
                  f"({sd[0]:.3f},{sd[1]:.3f},{sd[2]:.3f})")
            table_rows.append({
                "task": group, "pref": pref, "taskB": False, "n": int(means.shape[0]),
                "mean": mn.tolist(), "std": sd.tolist(),
            })
    for group in TASKB_GROUPS:
        for pref in PREFS:
            means, stds = collect(group, pref, taskB=True, n_max=50)
            cache[(group, pref)] = means
            mn = means.mean(axis=0)
            sd = stds.mean(axis=0)
            print(f"[B]{group:<37s} {pref:<7s} {means.shape[0]:>3d}  "
                  f"({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f})  "
                  f"({sd[0]:.3f},{sd[1]:.3f},{sd[2]:.3f})")
            table_rows.append({
                "task": group, "pref": pref, "taskB": True, "n": int(means.shape[0]),
                "mean": mn.tolist(), "std": sd.tolist(),
            })

    # L2 distance: taskB center mean vs each taskA task mean (averaged center+corner)
    taskb_center_mean = cache[("place_soap2_stand", "center")].mean(axis=0)
    taskb_corner_mean = cache[("place_soap2_stand", "corner")].mean(axis=0)
    print("\n=== L2(RGB-mean) distance from taskB center to each taskA task mean ===")
    print(f"{'taskA group':<30s}  d(taskB_c, A_c)  d(taskB_c, A_cor)  d(taskB_cor, A_c)  d(taskB_cor, A_cor)")
    for group in TASKA_GROUPS:
        ac = cache[(group, "center")].mean(axis=0)
        aco = cache[(group, "corner")].mean(axis=0)
        d_bc_ac = np.linalg.norm(taskb_center_mean - ac)
        d_bc_aco = np.linalg.norm(taskb_center_mean - aco)
        d_bcor_ac = np.linalg.norm(taskb_corner_mean - ac)
        d_bcor_aco = np.linalg.norm(taskb_corner_mean - aco)
        print(f"{group:<30s}  {d_bc_ac:>13.4f}  {d_bc_aco:>17.4f}  {d_bcor_ac:>17.4f}  {d_bcor_aco:>19.4f}")

    # Key: distance between taskB center vs corner within taskB
    d_within_taskB = np.linalg.norm(taskb_center_mean - taskb_corner_mean)
    print(f"\n*** Within-taskB:  L2(center_mean - corner_mean) = {d_within_taskB:.4f}")
    # Compare to within-taskA: averaged across the 8 groups
    within_taskA_ds = []
    for group in TASKA_GROUPS:
        ac = cache[(group, "center")].mean(axis=0)
        aco = cache[(group, "corner")].mean(axis=0)
        within_taskA_ds.append(np.linalg.norm(ac - aco))
    print(f"*** Within-taskA:  mean L2(center_mean - corner_mean) over 8 groups = {np.mean(within_taskA_ds):.4f}")
    print(f"    per-group: {[f'{x:.4f}' for x in within_taskA_ds]}")

    # Cross-task baseline: median pairwise distance among all 16 taskA task means
    taskA_means = np.stack([cache[(g, p)].mean(axis=0) for g in TASKA_GROUPS for p in PREFS])
    n = taskA_means.shape[0]
    pair_ds = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_ds.append(np.linalg.norm(taskA_means[i] - taskA_means[j]))
    print(f"\n*** TaskA cross-task pairwise L2 distance: median={np.median(pair_ds):.4f}, "
          f"p25={np.percentile(pair_ds,25):.4f}, p75={np.percentile(pair_ds,75):.4f}")
    print(f"*** Best-matched taskA group to taskB center: ",
          end="")
    best_idx = int(np.argmin([np.linalg.norm(taskb_center_mean - taskA_means[i]) for i in range(n)]))
    labels = [f"{g}_{p}" for g in TASKA_GROUPS for p in PREFS]
    print(f"{labels[best_idx]}  d={np.linalg.norm(taskb_center_mean - taskA_means[best_idx]):.4f}")

    # JSON dump for reference
    with open(OUT / "rgb_stats.json", "w") as f:
        json.dump({
            "rows": table_rows,
            "within_taskB_center_vs_corner": float(d_within_taskB),
            "within_taskA_center_vs_corner_per_group": {
                g: float(within_taskA_ds[i]) for i, g in enumerate(TASKA_GROUPS)
            },
            "taskA_pairwise_p25_med_p75": [
                float(np.percentile(pair_ds, 25)),
                float(np.median(pair_ds)),
                float(np.percentile(pair_ds, 75)),
            ],
        }, f, indent=2)
    print(f"[part3] saved {OUT/'rgb_stats.json'}")


# ============================================================
# PART 4: Stand vs pad/tray close-up comparison
# ============================================================
def part4_target_geometry():
    frac = 0.91
    n_eps = 3
    # 4 row groups: taskA center, taskA corner, taskB center, taskB corner
    # Columns: 3 eps each
    rows = [
        ("place_soap_tray_center",   ROOT / "place_soap_tray_center"),
        ("place_soap_tray_corner",   ROOT / "place_soap_tray_corner"),
        ("place_soap2_stand_center", TASKB_ROOT / "place_soap2_stand_center"),
        ("place_soap2_stand_corner", TASKB_ROOT / "place_soap2_stand_corner"),
    ]
    fig, axes = plt.subplots(len(rows), n_eps, figsize=(2.8 * n_eps, 2.6 * len(rows)))
    fig.suptitle(f"Stand (taskB) vs tray (taskA) target geometry - head_camera frac={frac}", fontsize=12)
    for r, (label, d) in enumerate(rows):
        eps = list_episodes(d)[:n_eps]
        for c, ep in enumerate(eps):
            img, idx, T = load_head_frame(ep, frac)
            ax = axes[r, c]
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"{label} ep{ep.stem.replace('episode','')}  i={idx}/{T-1}", fontsize=8)
    plt.tight_layout()
    out = OUT / "target_geometry_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[part4] saved {out}")


if __name__ == "__main__":
    part1_placement_grid()
    part2_taskB_sequence()
    part3_rgb_distance()
    part4_target_geometry()
    print("\nDone. Outputs in", OUT)
