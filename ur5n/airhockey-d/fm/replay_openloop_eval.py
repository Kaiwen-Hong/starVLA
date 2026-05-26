#!/usr/bin/env python3
"""
Air-hockey open-loop evaluation by replaying frames from the
3airhockey_dynamic_bounce dataset. Robot/camera NOT used.

Interactive REPL:
  Input "ep_idx,frame_idx" → load mp4 frame + parquet state → run inference
  → compare predicted 32-step world-XY trajectory against ground-truth
  actions[frame_idx : frame_idx + 32]

Model: QwenPI / DiT-S, action_dim=2 (dx, dy world), chunk_len=32, fps=50.
State is NOT fed to the model (training used include_state=false).

Usage:
    python ur5n/airhockey-d/fm/replay_openloop_eval.py
    python ur5n/airhockey-d/fm/replay_openloop_eval.py --checkpoint <path>
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

REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework

DEFAULT_CHECKPOINT = (
    "checkpoints/discreteRTC/fastumi_dynamic_qwenpi/"
    "checkpoints/steps_10000_pytorch_model.pt"
)
DEFAULT_DATASET_DIR = (
    "/home/kaiwen/Desktop/research/fastumipro-collection/"
    "0srarvla-lerobo/starvla/datasets/3airhockey_dynamic_bounce"
)
DEFAULT_INSTRUCTION = "strike the red puck back when it comes"

FPS = 50
CHUNK_LEN = 32


# ═══════════════════════════════════════════════════════════════════
#  Model loading
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
        print("[FIX] Set image_size=[256,256] (match training native)")

    print(f"Model loaded in {time.time() - t0:.1f}s  chunk_len={model.chunk_len}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  Episode loading (parquet + mp4)
# ═══════════════════════════════════════════════════════════════════

def _read_mp4_frames(mp4_path):
    """Read every frame from an mp4 into a uint8 array (T, H, W, 3) RGB.

    Tries torchvision.io first (training pipeline uses torchvision_av),
    falls back to cv2.VideoCapture sequential read.
    """
    try:
        import torchvision.io as tvio
        vid, _, _ = tvio.read_video(str(mp4_path), pts_unit='sec')
        # vid: (T, H, W, C) uint8 by default
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
                        instruction, infer_kwargs):
    state_xy = np.array(df['observation.state'].iloc[frame_idx],
                        dtype=np.float64)  # (2,)
    frame_np = frames[frame_idx]            # (H, W, 3) uint8
    pil_img = Image.fromarray(frame_np)

    T_full = len(df)
    end = min(frame_idx + CHUNK_LEN, T_full)
    gt_actions = np.stack(df['action'].iloc[frame_idx:end].values).astype(
        np.float32)                          # (n_gt, 2)

    example = {"image": [pil_img], "lang": instruction}
    t0 = time.time()
    out = model.predict_action(examples=[example], **infer_kwargs)
    infer_ms = (time.time() - t0) * 1000

    pred_norm = out["normalized_actions"][0].astype(np.float32)  # (32, 2)
    pred_delta = baseframework.unnormalize_actions(
        pred_norm, action_stats)                                  # (32, 2)

    pred_traj = state_xy + np.cumsum(pred_delta, axis=0)          # (32, 2)
    gt_traj = state_xy + np.cumsum(gt_actions, axis=0)            # (n_gt, 2)

    full_states = np.stack(df['observation.state'].values).astype(np.float64)
    return {
        'state_xy': state_xy,
        'pred_delta': pred_delta,
        'pred_traj': pred_traj,
        'gt_actions': gt_actions,
        'gt_traj': gt_traj,
        'full_states': full_states,
        'pil_img': pil_img,
        'infer_ms': infer_ms,
        'frame_idx': frame_idx,
    }


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize(result, ep_idx, frame_idx, save_path, instruction):
    state = result['state_xy']
    pred = result['pred_traj']
    gt = result['gt_traj']
    full = result['full_states']
    pred_delta = result['pred_delta']
    gt_delta = result['gt_actions']
    T_pred = pred.shape[0]
    T_gt = gt.shape[0]
    ts_pred = np.arange(T_pred) / FPS
    ts_gt = np.arange(T_gt) / FPS

    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.30,
                  width_ratios=[1, 1.2, 1.2])

    # Camera frame (top-left)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(result['pil_img'])
    ax_img.set_title(f"Wrist frame  ep{ep_idx} f{frame_idx}",
                     fontsize=11, fontweight="bold")
    ax_img.axis("off")

    # Top-down XY (bottom-left, spans 2 rows)
    ax_xy = fig.add_subplot(gs[1:, 0])
    ax_xy.plot(full[:, 0], full[:, 1], "-", color="lightgray", lw=1.5,
               alpha=0.7, label=f"full ep ({len(full)} frames)")
    ax_xy.plot(gt[:, 0], gt[:, 1], "o-", color="#377eb8", ms=4, lw=1.8,
               label=f"GT [{frame_idx}:{frame_idx + T_gt}]")
    ax_xy.plot(pred[:, 0], pred[:, 1], "s-", color="#e41a1c", ms=4, lw=1.8,
               label=f"Pred (32)")
    ax_xy.plot(state[0], state[1], "^", color="green", ms=12, label="now")
    ax_xy.set_xlabel("x (world m)")
    ax_xy.set_ylabel("y (world m)")
    ax_xy.set_title("Top-down XY (world)", fontweight="bold")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=8)

    # x(t), y(t) (middle column)
    ax_xt = fig.add_subplot(gs[0:2, 1])
    ax_xt.plot(ts_gt, gt[:, 0], "o-", color="#377eb8", ms=3, lw=1.5, label="GT")
    ax_xt.plot(ts_pred, pred[:, 0], "s-", color="#e41a1c", ms=3, lw=1.5,
               label="Pred")
    ax_xt.axhline(state[0], color="gray", ls=":", lw=1,
                  label=f"now={state[0]:.4f}")
    ax_xt.set_ylabel("x (world m)", fontweight="bold")
    ax_xt.set_title("Absolute x, y vs time", fontweight="bold")
    ax_xt.legend(fontsize=8)
    ax_xt.grid(True, alpha=0.3)
    plt.setp(ax_xt.get_xticklabels(), visible=False)

    ax_yt = fig.add_subplot(gs[2, 1], sharex=ax_xt)
    ax_yt.plot(ts_gt, gt[:, 1], "o-", color="#377eb8", ms=3, lw=1.5)
    ax_yt.plot(ts_pred, pred[:, 1], "s-", color="#e41a1c", ms=3, lw=1.5)
    ax_yt.axhline(state[1], color="gray", ls=":", lw=1)
    ax_yt.set_xlabel(f"t (s) — {FPS}Hz")
    ax_yt.set_ylabel("y (world m)", fontweight="bold")
    ax_yt.grid(True, alpha=0.3)

    # dx(t), dy(t) (right column)
    ax_dx = fig.add_subplot(gs[0:2, 2])
    ax_dx.plot(ts_gt, gt_delta[:, 0], "o-", color="#377eb8", ms=3, lw=1.5,
               label="GT")
    ax_dx.plot(ts_pred, pred_delta[:, 0], "s-", color="#e41a1c", ms=3, lw=1.5,
               label="Pred")
    ax_dx.axhline(0, color="black", lw=0.5)
    ax_dx.set_ylabel("dx (m / step)", fontweight="bold")
    ax_dx.set_title("Per-step delta dx, dy", fontweight="bold")
    ax_dx.legend(fontsize=8)
    ax_dx.grid(True, alpha=0.3)
    plt.setp(ax_dx.get_xticklabels(), visible=False)

    ax_dy = fig.add_subplot(gs[2, 2], sharex=ax_dx)
    ax_dy.plot(ts_gt, gt_delta[:, 1], "o-", color="#377eb8", ms=3, lw=1.5)
    ax_dy.plot(ts_pred, pred_delta[:, 1], "s-", color="#e41a1c", ms=3, lw=1.5)
    ax_dy.axhline(0, color="black", lw=0.5)
    ax_dy.set_xlabel(f"t (s) — {FPS}Hz")
    ax_dy.set_ylabel("dy (m / step)", fontweight="bold")
    ax_dy.grid(True, alpha=0.3)

    n_overlap = min(T_pred, T_gt)
    dxy_mae = float(np.mean(np.abs(pred_delta[:n_overlap] - gt_delta[:n_overlap])))
    fig.suptitle(
        f'Air-hockey replay  —  ep {ep_idx}, frame {frame_idx}/{len(full) - 1}\n'
        f'"{instruction}"   |   infer={result["infer_ms"]:.0f}ms   |   '
        f'delta MAE (overlap={n_overlap}) = {dxy_mae:.5f} m',
        fontsize=12, fontweight="bold")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main REPL
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Air-hockey replay open-loop eval (QwenPI, 2-D world XY)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--use_server", action="store_true", default=True,
                        help="Connect to inference_server.py over a Unix socket "
                             "instead of loading the model in-process. "
                             "(default: True — start the server first)")
    parser.add_argument("--no_use_server", dest="use_server",
                        action="store_false",
                        help="Load the model in-process (slower startup)")
    parser.add_argument("--server_socket", type=str,
                        default="/tmp/starvla_infer_airhockey.sock")
    args = parser.parse_args()

    if args.use_server:
        from remote_model import RemoteModel
        model = RemoteModel(args.server_socket)
        args.checkpoint = model.checkpoint_path
    else:
        model = load_model(args.checkpoint)
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', "
          f"modes={action_stats.get('norm_modes', 'legacy')}")
    assert model.chunk_len == CHUNK_LEN, (
        f"chunk_len mismatch: model={model.chunk_len} expected={CHUNK_LEN}")

    session_ts = time.strftime("%Y%m%d_%H%M%S")
    viz_dir = Path("ur5n/airhockey-d/fm/viz_replay") / f"session_{session_ts}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Viz output: {viz_dir}")

    # Per-episode cache (parquet + decoded frames)
    cache = {}

    def get_ep(ep_idx):
        if ep_idx not in cache:
            print(f"  Loading ep {ep_idx} (decoding mp4)...", flush=True)
            t0 = time.time()
            df, frames = load_episode(args.dataset_dir, ep_idx)
            cache[ep_idx] = (df, frames)
            print(f"  Loaded ep {ep_idx}: {len(df)} frames, "
                  f"video={frames.shape}  ({time.time() - t0:.1f}s)",
                  flush=True)
        return cache[ep_idx]

    infer_kwargs = dict()  # QwenPI ignores extras

    print("=" * 60)
    print("  Air-hockey Replay Open-Loop Eval (QwenPI)")
    print(f"  Instruction:  \"{args.instruction}\"")
    print(f"  Dataset:      {args.dataset_dir}")
    print(f"  fps={FPS}  chunk_len={CHUNK_LEN}  action_dim=2 (world dx, dy)")
    print("  Enter 'ep_idx,frame_idx' (e.g. '0,20'), 'q' to quit")
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
                print("Bad input. Format: <ep_idx>,<frame_idx>  (e.g. 12,30)")
                continue

            try:
                df, frames = get_ep(ep_idx)
            except FileNotFoundError as e:
                print(f"[ERR] {e}")
                continue

            if frame_idx < 0 or frame_idx >= len(df):
                print(f"[ERR] frame_idx out of range [0, {len(df) - 1}]")
                continue

            print(f"  Running inference...", flush=True)
            result = predict_and_compare(
                model, action_stats, df, frames, frame_idx,
                args.instruction, infer_kwargs)

            n_overlap = min(result['pred_delta'].shape[0],
                            result['gt_actions'].shape[0])
            mae_dx = float(np.mean(np.abs(
                result['pred_delta'][:n_overlap, 0] -
                result['gt_actions'][:n_overlap, 0])))
            mae_dy = float(np.mean(np.abs(
                result['pred_delta'][:n_overlap, 1] -
                result['gt_actions'][:n_overlap, 1])))
            print(f"  state_xy=[{result['state_xy'][0]:.4f}, "
                  f"{result['state_xy'][1]:.4f}]  "
                  f"infer={result['infer_ms']:.0f}ms")
            print(f"  delta MAE (overlap={n_overlap}):  "
                  f"dx={mae_dx:.5f}  dy={mae_dy:.5f}  "
                  f"sum={(mae_dx + mae_dy) / 2:.5f}")
            print(f"  Pred final XY: "
                  f"[{result['pred_traj'][-1, 0]:.4f}, "
                  f"{result['pred_traj'][-1, 1]:.4f}]")
            if n_overlap > 0:
                print(f"  GT   final XY: "
                      f"[{result['gt_traj'][-1, 0]:.4f}, "
                      f"{result['gt_traj'][-1, 1]:.4f}]")

            save_path = (viz_dir /
                         f"run_{run_idx:03d}_ep{ep_idx:03d}_f{frame_idx:04d}.png")
            visualize(result, ep_idx, frame_idx, str(save_path), args.instruction)
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
