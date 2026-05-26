#!/usr/bin/env python3
"""
Pool open-loop evaluation by replaying frames from pool_strike_combined
(2-task LeRobot dataset). Robot/camera NOT used.

Interactive REPL:
  Input "ep_idx,frame_idx"        → load mp4 frame + parquet state
                                  → run inference (auto-selects instruction
                                     from the episode's task_index)
                                  → compare predicted chunk_len-step trajectory
                                     against ground-truth actions[frame_idx :
                                     frame_idx + chunk_len].

Model is QwenPI flow-matching, action_dim=10 (Δpos_world + ΔR_world rot6d +
gripper_target), state_dim=10 (pos + rot6d + gripper). Cue is bolt-gripped
so gripper is constant 1.0 throughout; the dataset uses sentinel 0.0 that
V3 binarizes to 1.0.

Usage:
    # Terminal 1 (once the model exists):
    python ur5n/pool-preference/fm/inference_server.py --checkpoint <path>

    # Terminal 2:
    python ur5n/pool-preference/fm/replay_openloop_eval.py
"""

import sys
import os
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "checkpoints/discreteRTC/fastumi_pool_qwenPI_0522_DiT-S/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
DEFAULT_DATASET_DIR = (
    "/home/kaiwen/Desktop/research/fastumipro-collection/"
    "0srarvla-lerobo/starvla/datasets/pool_strike_combined"
)

# Pool has 2 tasks (per docs/pool-static.md). Instruction lookup mirrors
# the strings in TASK_MAP from assign_pool_tasks.py.
TASK_INSTRUCTIONS = {
    0: "Strike the white ball into the red ball to pocket it",
    1: "Strike the white ball so that the red ball bounces off the walls into the goal",
}

FPS = 20


# ═══════════════════════════════════════════════════════════════════
#  Model loading (only used with --no_use_server)
# ═══════════════════════════════════════════════════════════════════

def _detect_attn():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        print("[INFO] flash_attn not available, falling back to sdpa")
        return "sdpa"


def load_model(checkpoint_path):
    print(f"Loading model: {checkpoint_path}")
    t0 = time.time()
    checkpoint_pt = Path(checkpoint_path)
    model_config, norm_stats = read_mode_config(checkpoint_pt)
    config = dict_to_namespace(model_config)
    config.trainer.pretrained_checkpoint = None
    config.framework.qwenvl.attn_implementation = _detect_attn()

    model = build_framework(cfg=config)
    model.norm_stats = norm_stats
    state_dict = torch.load(checkpoint_pt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda").eval()

    if not getattr(config.datasets.vla_data, "image_size", None):
        config.datasets.vla_data.image_size = [256, 256]
        print("[FIX] Set image_size=[256,256] (match pool native)")

    print(f"Loaded in {time.time() - t0:.1f}s  chunk_len={model.chunk_len}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  10-D rotation helpers — match V3-convert exactly (SO(3) composition).
#
# Training action is computed by V3 as:
#     delta_pos    = pos_next - pos_curr
#     R_delta      = R_next @ R_curr.T          # world-frame rotation delta
#     delta_rot6d  = mat_to_rot6d(R_delta)
#     gripper      = target_state[9]            # absolute target, NOT delta
#
# Inverse (forward roll-out used here):
#     pos_new      = pos_curr + delta_pos
#     R_new        = R_delta @ R_curr           # SO(3) composition
#     gripper_new  = action_gripper             # absolute target
#
# IMPORTANT: rotvec/euler addition is NOT a valid substitute for SO(3)
# composition. The earlier version of this script accumulated rotvecs
# additively, which silently introduced per-step errors of ~8e-4 in
# rot6d that compounded across the chunk and made the viz "loss" look
# much worse than the model actually was. The functions below operate
# strictly on rotation matrices via Gram-Schmidt-decoded rot6d.
# ═══════════════════════════════════════════════════════════════════

def mat_to_rot6d(mat):
    """3×3 rotation matrix -> rot6d (first two rows flattened, row-major)."""
    return mat[:2, :].flatten().astype(np.float32)


def rot6d_to_mat(d6):
    """rot6d -> 3×3 rotation matrix via Gram-Schmidt (matches utils/pose_utils)."""
    a1 = np.asarray(d6[:3], dtype=np.float64)
    a2 = np.asarray(d6[3:], dtype=np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def rot6d_to_euler(d6):
    """rot6d -> (roll, pitch, yaw) in radians, scipy 'xyz' intrinsic."""
    R = rot6d_to_mat(d6)
    return Rot.from_matrix(R).as_euler('xyz')


def states_10d_to_pos_euler(states_10d):
    """(T, 10) -> (T, 3) positions, (T, 3) euler (roll/pitch/yaw)."""
    T = states_10d.shape[0]
    pos = states_10d[:, :3].astype(np.float64)
    euler = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
        euler[t] = rot6d_to_euler(states_10d[t, 3:9])
    return pos, euler


def accumulate_actions_to_states(start_state_10d, actions_10d):
    """Apply (delta_pos, delta_R_rot6d, gripper_target) actions to a 10-D
    state. Returns (T, 10) absolute world-frame states.

    Inverse of V3's compute_world_relative_action_10d."""
    pos = np.asarray(start_state_10d[:3], dtype=np.float64).copy()
    R = rot6d_to_mat(start_state_10d[3:9])
    T = actions_10d.shape[0]
    out = np.zeros((T, 10), dtype=np.float64)
    for t in range(T):
        pos = pos + actions_10d[t, :3]
        R_delta = rot6d_to_mat(actions_10d[t, 3:9])
        R = R_delta @ R                      # SO(3) composition
        out[t, :3] = pos
        out[t, 3:9] = mat_to_rot6d(R)
        out[t, 9] = actions_10d[t, 9]        # gripper is absolute target
    return out


# ═══════════════════════════════════════════════════════════════════
#  Episode loading (parquet + mp4)
# ═══════════════════════════════════════════════════════════════════

def _read_mp4_frames(mp4_path):
    """Read every frame from an mp4 into a uint8 array (T, H, W, 3) RGB.
    Tries torchvision.io first, falls back to cv2.VideoCapture."""
    try:
        import torchvision.io as tvio
        vid, _, _ = tvio.read_video(str(mp4_path), pts_unit='sec')
        return vid.numpy().astype(np.uint8)
    except Exception as e:
        print(f"[WARN] torchvision read_video failed ({e}); falling back to cv2")

    import cv2
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to decode any frame from {mp4_path}")
    return np.stack(frames).astype(np.uint8)


def load_episode(dataset_dir, ep_idx):
    parquet = Path(dataset_dir) / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
    mp4 = (Path(dataset_dir) /
           f"videos/chunk-000/observation.images.wrist/episode_{ep_idx:06d}.mp4")
    if not parquet.exists():
        raise FileNotFoundError(parquet)
    if not mp4.exists():
        raise FileNotFoundError(mp4)
    df = pd.read_parquet(parquet)
    frames = _read_mp4_frames(mp4)
    if len(frames) != len(df):
        print(f"[WARN] frame count mismatch: mp4={len(frames)} parquet={len(df)}")
    return df, frames


# ═══════════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════════

def predict_and_compare(model, action_stats, df, frames, frame_idx,
                        instruction, chunk_len, infer_kwargs):
    state_10d = np.array(df['observation.state'].iloc[frame_idx],
                         dtype=np.float64)            # (10,)
    state_pos = state_10d[:3]
    state_euler = rot6d_to_euler(state_10d[3:9])      # (3,) roll/pitch/yaw

    frame_np = frames[frame_idx]                       # (H, W, 3) uint8
    pil_img = Image.fromarray(frame_np)

    # === GT trajectory: pull absolute states directly from parquet ===
    # Action[t] applied to state[t] yields state[t+1] (V3 convention),
    # so the chunk_len-step trajectory under GT actions is
    # state[frame_idx + 1 : frame_idx + 1 + chunk_len]. Avoids
    # re-accumulating and any rotation-composition mistakes.
    T_full = len(df)
    gt_end = min(frame_idx + 1 + chunk_len, T_full)
    gt_states_10d = np.stack(
        df['observation.state'].iloc[frame_idx + 1:gt_end].values
    ).astype(np.float64)                                # (n_gt, 10)

    # GT actions: keep the same length as gt_states (action[t] yields
    # state[t+1], so at the episode tail the last few actions point at
    # states that aren't in the parquet — clip them off).
    n_gt = gt_states_10d.shape[0]
    gt_actions_10d = np.stack(
        df['action'].iloc[frame_idx:frame_idx + n_gt].values
    ).astype(np.float32)                                # (n_gt, 10)

    example = {"image": [pil_img], "lang": instruction}
    t0 = time.time()
    out = model.predict_action(examples=[example], **infer_kwargs)
    infer_ms = (time.time() - t0) * 1000

    pred_norm = out["normalized_actions"][0].astype(np.float32)
    pred_actions_10d = baseframework.unnormalize_actions(
        pred_norm, action_stats)                                       # (L, 10)

    # === Pred trajectory: PROPER SO(3) composition ===
    pred_states_10d = accumulate_actions_to_states(
        state_10d, pred_actions_10d)                                   # (L, 10)

    # Decompose for plotting.
    pred_pos, pred_euler = states_10d_to_pos_euler(pred_states_10d)
    gt_pos, gt_euler = states_10d_to_pos_euler(gt_states_10d)
    full_states = np.stack(df['observation.state'].values).astype(np.float64)
    full_pos, full_euler = states_10d_to_pos_euler(full_states)

    return {
        'state_10d': state_10d,
        'state_pos': state_pos,
        'state_euler': state_euler,
        'pred_actions_10d': pred_actions_10d,
        'gt_actions_10d': gt_actions_10d,
        'pred_states_10d': pred_states_10d,
        'gt_states_10d': gt_states_10d,
        'pred_pos': pred_pos,
        'pred_euler': pred_euler,
        'gt_pos': gt_pos,
        'gt_euler': gt_euler,
        'full_pos': full_pos,
        'full_euler': full_euler,
        'pil_img': pil_img,
        'infer_ms': infer_ms,
        'frame_idx': frame_idx,
    }


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize(result, ep_idx, frame_idx, task_idx, save_path, instruction):
    state_pos = result['state_pos']
    state_euler = result['state_euler']
    pred_pos = result['pred_pos']
    pred_euler = result['pred_euler']
    gt_pos = result['gt_pos']
    gt_euler = result['gt_euler']
    full_pos = result['full_pos']
    full_euler = result['full_euler']
    pred_a = result['pred_actions_10d']
    gt_a = result['gt_actions_10d']
    T_pred = pred_pos.shape[0]
    T_gt = gt_pos.shape[0]
    ts_pred = np.arange(T_pred) / FPS
    ts_gt = np.arange(T_gt) / FPS

    fig = plt.figure(figsize=(26, 16))
    gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.32,
                  width_ratios=[1, 1.2, 1.2])

    # Camera frame (top-left)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(result['pil_img'])
    ax_img.set_title(f"Wrist (fisheye)  ep{ep_idx} f{frame_idx}",
                     fontsize=11, fontweight="bold")
    ax_img.axis("off")

    # Top-down XY (bottom-left, spans 4 rows)
    ax_xy = fig.add_subplot(gs[1:, 0])
    ax_xy.plot(full_pos[:, 0], full_pos[:, 1], "-", color="lightgray",
               lw=1.5, alpha=0.7, label=f"full ep ({len(full_pos)} frames)")
    ax_xy.plot(gt_pos[:, 0], gt_pos[:, 1], "o-", color="#377eb8", ms=4,
               lw=1.8, label=f"GT [{frame_idx+1}:{frame_idx+1+T_gt}]")
    ax_xy.plot(pred_pos[:, 0], pred_pos[:, 1], "s-", color="#e41a1c", ms=4,
               lw=1.8, label=f"Pred ({T_pred})")
    ax_xy.plot(state_pos[0], state_pos[1], "^", color="green", ms=12,
               label="now")
    ax_xy.set_xlabel("x (world m)")
    ax_xy.set_ylabel("y (world m)")
    ax_xy.set_title("Top-down XY (world)", fontweight="bold")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=8)

    # Middle column: absolute pose vs time (x, y, z, yaw, gripper)
    #   roll/pitch are essentially constant in the pool dataset
    #   (range ~5e-4 rad), so we collapse them into "yaw" only.
    traj_rows = [
        ('x (m)',       gt_pos[:, 0],   pred_pos[:, 0],   state_pos[0]),
        ('y (m)',       gt_pos[:, 1],   pred_pos[:, 1],   state_pos[1]),
        ('z (m)',       gt_pos[:, 2],   pred_pos[:, 2],   state_pos[2]),
        ('yaw (deg)',   np.rad2deg(gt_euler[:, 2]),
                        np.rad2deg(pred_euler[:, 2]),
                        np.rad2deg(state_euler[2])),
        ('gripper',     result['gt_states_10d'][:, 9],
                        result['pred_states_10d'][:, 9],   None),
    ]
    for row, (name, gtv, prv, nowv) in enumerate(traj_rows):
        ax = fig.add_subplot(gs[row, 1])
        ax.plot(ts_gt, gtv, "o-", color="#377eb8", ms=3, lw=1.5, label="GT")
        ax.plot(ts_pred, prv, "s-", color="#e41a1c", ms=3, lw=1.5,
                label="Pred")
        if nowv is not None:
            ax.axhline(nowv, color="gray", ls=":", lw=1,
                       label=f"now={nowv:.4f}")
        ax.set_ylabel(name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title("Absolute pose vs time", fontweight="bold")
            ax.legend(fontsize=8)
        if row < len(traj_rows) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel(f"t (s) — {FPS}Hz")

    # Right column: per-step action deltas (dx, dy, dz, d_yaw_deg/step, gripper_target)
    # d_yaw per step ≈ arcsin(action[3:9][4])  (since R_delta[0,1] ≈ -sin(d_yaw)
    # for tiny roll/pitch; equivalent to euler-from-mat on R_delta).
    def deltas_to_dyaw_deg(actions_10d):
        n = actions_10d.shape[0]
        out = np.zeros(n)
        for i in range(n):
            R_delta = rot6d_to_mat(actions_10d[i, 3:9])
            out[i] = np.rad2deg(Rot.from_matrix(R_delta).as_euler('xyz')[2])
        return out
    delta_rows = [
        ('dx (mm/step)',    gt_a[:, 0]*1000,   pred_a[:, 0]*1000),
        ('dy (mm/step)',    gt_a[:, 1]*1000,   pred_a[:, 1]*1000),
        ('dz (mm/step)',    gt_a[:, 2]*1000,   pred_a[:, 2]*1000),
        ('d_yaw (deg/step)', deltas_to_dyaw_deg(gt_a),
                              deltas_to_dyaw_deg(pred_a)),
        ('gripper_target',  gt_a[:, 9],         pred_a[:, 9]),
    ]
    for row, (name, gtv, prv) in enumerate(delta_rows):
        ax = fig.add_subplot(gs[row, 2])
        ax.plot(ts_gt, gtv, "o-", color="#377eb8", ms=3, lw=1.5, label="GT")
        ax.plot(ts_pred, prv, "s-", color="#e41a1c", ms=3, lw=1.5,
                label="Pred")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel(name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title("Per-step action deltas", fontweight="bold")
            ax.legend(fontsize=8)
        if row < len(delta_rows) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel(f"t (s) — {FPS}Hz")

    # ── Metrics: position MAE in mm, yaw MAE in degrees ─────────────
    n_overlap = min(T_pred, T_gt)
    pos_mae_mm = float(np.mean(np.linalg.norm(
        pred_pos[:n_overlap] - gt_pos[:n_overlap], axis=1))) * 1000
    yaw_mae_deg = float(np.mean(np.abs(
        np.rad2deg(pred_euler[:n_overlap, 2] - gt_euler[:n_overlap, 2]))))
    pos_final_err_mm = float(np.linalg.norm(
        pred_pos[n_overlap - 1] - gt_pos[n_overlap - 1])) * 1000
    yaw_final_err_deg = float(abs(
        np.rad2deg(pred_euler[n_overlap - 1, 2] -
                    gt_euler[n_overlap - 1, 2])))

    fig.suptitle(
        f'Pool replay — ep {ep_idx} (task {task_idx}), '
        f'frame {frame_idx}/{len(full_pos) - 1}\n'
        f'"{instruction}"   |   infer={result["infer_ms"]:.0f}ms   |   '
        f'overlap={n_overlap}\n'
        f'mean pos MAE = {pos_mae_mm:.2f} mm   '
        f'mean yaw MAE = {yaw_mae_deg:.3f}°   '
        f'final pos err = {pos_final_err_mm:.2f} mm   '
        f'final yaw err = {yaw_final_err_deg:.3f}°',
        fontsize=11, fontweight="bold")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main REPL
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pool replay open-loop eval (QwenPI flow-matching, 10-D)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override instruction. Default: auto-pick from "
                             "the episode's task_index (0=pocket, 1=wall_bounce)")
    parser.add_argument("--use_server", action="store_true", default=True,
                        help="Connect to inference_server.py")
    parser.add_argument("--no_use_server", dest="use_server",
                        action="store_false")
    parser.add_argument("--server_socket", type=str,
                        default="/tmp/starvla_infer_pool.sock")
    args = parser.parse_args()

    if not Path(args.dataset_dir).exists():
        print(f"[ERR] Dataset dir not found: {args.dataset_dir}")
        print(f"      Build it via 0srarvla-lerobo/{{prep_pool_for_v3.py, "
              f"V3-...delta-world-frame.py, assign_pool_tasks.py}} first.")
        sys.exit(1)

    if args.use_server:
        from remote_model import RemoteModel
        model = RemoteModel(args.server_socket)
        args.checkpoint = model.checkpoint_path
    else:
        model = load_model(args.checkpoint)
    chunk_len = model.chunk_len
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', "
          f"modes={action_stats.get('norm_modes', 'legacy')}")
    print(f"chunk_len = {chunk_len}")

    session_ts = time.strftime("%Y%m%d_%H%M%S")
    viz_dir = Path("ur5n/pool-preference/fm/viz_replay") / f"session_{session_ts}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Viz output: {viz_dir}")

    cache = {}

    def get_ep(ep_idx):
        if ep_idx not in cache:
            print(f"  Loading ep {ep_idx} (decoding mp4)...", flush=True)
            t0 = time.time()
            df, frames = load_episode(args.dataset_dir, ep_idx)
            task_idx = int(df['task_index'].iloc[0])
            cache[ep_idx] = (df, frames, task_idx)
            print(f"  Loaded ep {ep_idx}: {len(df)} frames, "
                  f"task_idx={task_idx}, video={frames.shape}  "
                  f"({time.time() - t0:.1f}s)", flush=True)
        return cache[ep_idx]

    infer_kwargs = dict()

    print("=" * 60)
    print("  Pool Replay Open-Loop Eval (QwenPI, 10-D)")
    print(f"  Dataset:      {args.dataset_dir}")
    print(f"  Tasks:        0={TASK_INSTRUCTIONS[0][:50]}...")
    print(f"                1={TASK_INSTRUCTIONS[1][:50]}...")
    print(f"  fps={FPS}  chunk_len={chunk_len}  action_dim=10  state_dim=10")
    print("  Enter 'ep_idx,frame_idx' (e.g. '0,10'), 'q' to quit")
    print("=" * 60)

    run_idx = 0
    try:
        while True:
            line = input(f"\n[Run {run_idx}] ep,frame: ").strip()
            if not line or line.lower() == 'q':
                break
            try:
                ep_s, fr_s = line.split(",")
                ep_idx = int(ep_s.strip())
                frame_idx = int(fr_s.strip())
            except ValueError:
                print("Bad input. Format: <ep_idx>,<frame_idx>  (e.g. 12,8)")
                continue

            try:
                df, frames, task_idx = get_ep(ep_idx)
            except FileNotFoundError as e:
                print(f"[ERR] {e}")
                continue

            if frame_idx < 0 or frame_idx >= len(df):
                print(f"[ERR] frame_idx out of range [0, {len(df) - 1}]")
                continue

            if args.instruction is not None:
                instruction = args.instruction
            else:
                instruction = TASK_INSTRUCTIONS.get(
                    task_idx, f"[unknown task_idx={task_idx}]")
            print(f"  task_idx={task_idx}  instruction=\"{instruction[:80]}\"")
            print(f"  Running inference...", flush=True)
            result = predict_and_compare(
                model, action_stats, df, frames, frame_idx,
                instruction, chunk_len, infer_kwargs)

            n_overlap = min(result['pred_pos'].shape[0],
                            result['gt_pos'].shape[0])
            pos_mae_mm = float(np.mean(np.linalg.norm(
                result['pred_pos'][:n_overlap] -
                result['gt_pos'][:n_overlap], axis=1))) * 1000
            yaw_mae_deg = float(np.mean(np.abs(np.rad2deg(
                result['pred_euler'][:n_overlap, 2] -
                result['gt_euler'][:n_overlap, 2]))))
            print(f"  state pos=[{result['state_pos'][0]:.3f}, "
                  f"{result['state_pos'][1]:.3f}, "
                  f"{result['state_pos'][2]:.3f}]  "
                  f"yaw={np.rad2deg(result['state_euler'][2]):+.2f}°  "
                  f"infer={result['infer_ms']:.0f}ms")
            print(f"  MAE (overlap={n_overlap}):  "
                  f"pos={pos_mae_mm:.2f} mm  yaw={yaw_mae_deg:.3f}°")
            print(f"  Pred final: pos=[{result['pred_pos'][-1, 0]:.3f}, "
                  f"{result['pred_pos'][-1, 1]:.3f}, "
                  f"{result['pred_pos'][-1, 2]:.3f}]  "
                  f"yaw={np.rad2deg(result['pred_euler'][-1, 2]):+.2f}°")
            if n_overlap > 0:
                print(f"  GT   final: pos=[{result['gt_pos'][-1, 0]:.3f}, "
                      f"{result['gt_pos'][-1, 1]:.3f}, "
                      f"{result['gt_pos'][-1, 2]:.3f}]  "
                      f"yaw={np.rad2deg(result['gt_euler'][-1, 2]):+.2f}°")

            save_path = (viz_dir /
                         f"run_{run_idx:03d}_ep{ep_idx:03d}_f{frame_idx:04d}.png")
            visualize(result, ep_idx, frame_idx, task_idx, str(save_path),
                      instruction)
            print(f"  Saved: {save_path}")
            run_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")

    if hasattr(model, 'close'):
        try:
            model.close()
        except Exception:
            pass
    print("Done.")


if __name__ == "__main__":
    main()
