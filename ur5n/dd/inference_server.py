#!/usr/bin/env python3
"""
StarVLA inference server (dd/ — pick-from-static, place-on-turntable variant).

Loads the model once into GPU memory and serves predict_action /
predict_action_realtime requests over a Unix-domain socket. Pair with
remote_model.RemoteModel on the client side (used by closedloop_rtc_v6.py
when --use_server is enabled, which is the default).

The point is to avoid the ~30s model-load cost on every iteration of
client-side code: keep the server running in one terminal, edit and re-run
the closed-loop script in another. The model stays warm in GPU memory
across many client runs.

Default socket is /tmp/starvla_infer_dd.sock so this server can run side
by side with the 2dd/ server (which uses /tmp/starvla_infer.sock).

Usage:
    # Terminal 1 — start the server (load once, leave running)
    python ur5n/dd/inference_server.py
    # optional: --checkpoint <path>  --socket /tmp/starvla_infer_dd.sock

    # Terminal 2 — run the closed-loop script (defaults to --use_server)
    python ur5n/dd/closedloop_rtc_v6.py

Wire format (pickle over multiprocessing.connection):
    request:  {'cmd': 'info'}
              {'cmd': 'predict_action',          'examples': [...],
               'infer_kwargs': {...}}
              {'cmd': 'predict_action_realtime', 'examples': [...],
               'prev_action_chunk_normalized': np.ndarray,
               'inference_delay': int,
               'infer_kwargs': {...}}
    reply:    {'ok': True,  'normalized_actions': np.ndarray}        # predict
              {'ok': True,  'chunk_len': int,
               'norm_stats': dict, 'checkpoint_path': str}            # info
              {'ok': False, 'error': str}                             # any
"""

import sys
import os
import time
import argparse
import traceback
from pathlib import Path
from multiprocessing.connection import Listener

import numpy as np
import torch

# ── Repo setup (mirrors closedloop_rtc_v6.py) ──────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "ur5"))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework

DEFAULT_CHECKPOINT = (
    "checkpoints/discreteRTC/fastumi_pickandplace_qwenDiscreteDiffusion_0409_0_pick_to_moved_filtered/"
    "checkpoints/steps_30000_pytorch_model.pt"
)
DEFAULT_SOCKET = "/tmp/starvla_infer_dd.sock"
AUTHKEY = b'starvla'


def _detect_attn():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_model(checkpoint_path):
    """Duplicated from closedloop_rtc_v6.load_model so the server has zero
    import-time coupling to the client script."""
    print(f"[server] Loading model from: {checkpoint_path}")
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
        config.datasets.vla_data.image_size = [224, 224]
        print("[server] [FIX] Set image_size=[224,224]")

    print(f"[server] Model loaded in {time.time() - t0:.1f}s  "
          f"chunk_len={model.chunk_len}")
    return model


def serve(socket_path, checkpoint_path, injected_delay_ms=50):
    model = load_model(checkpoint_path)

    # Clean up stale socket file from a previous (crashed) server run.
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    listener = Listener(socket_path, family='AF_UNIX', authkey=AUTHKEY)
    print(f"[server] Listening on {socket_path}")
    print(f"[server] Injected post-inference delay: {injected_delay_ms} ms")
    print(f"[server] Ready. Ctrl+C to stop.")

    req_count = 0
    try:
        while True:
            try:
                conn = listener.accept()
            except (KeyboardInterrupt, EOFError):
                break
            print(f"[server] Client connected")

            try:
                while True:
                    try:
                        msg = conn.recv()
                    except (EOFError, ConnectionResetError):
                        print("[server] Client disconnected")
                        break

                    cmd = msg.get('cmd') if isinstance(msg, dict) else None
                    t0 = time.monotonic()
                    try:
                        if cmd == 'info':
                            reply = {
                                'ok': True,
                                'chunk_len': model.chunk_len,
                                'norm_stats': model.norm_stats,
                                'checkpoint_path': str(checkpoint_path),
                            }
                        elif cmd == 'predict_action':
                            out = model.predict_action(
                                examples=msg['examples'],
                                **msg.get('infer_kwargs', {}),
                            )
                            reply = {
                                'ok': True,
                                'normalized_actions': out['normalized_actions'],
                            }
                        elif cmd == 'predict_action_realtime':
                            out = model.predict_action_realtime(
                                examples=msg['examples'],
                                prev_action_chunk_normalized=msg['prev_action_chunk_normalized'],
                                inference_delay=msg['inference_delay'],
                                **msg.get('infer_kwargs', {}),
                            )
                            reply = {
                                'ok': True,
                                'normalized_actions': out['normalized_actions'],
                            }
                        else:
                            reply = {'ok': False, 'error': f"unknown cmd: {cmd!r}"}
                    except Exception as e:
                        traceback.print_exc()
                        reply = {'ok': False, 'error': f"{type(e).__name__}: {e}"}

                    elapsed_ms = (time.monotonic() - t0) * 1000
                    req_count += 1
                    if cmd in ('predict_action', 'predict_action_realtime'):
                        if reply.get('ok'):
                            shape = tuple(reply['normalized_actions'].shape)
                        else:
                            shape = 'err'
                        print(f"[server] #{req_count} {cmd} → {shape} "
                              f"in {elapsed_ms:.1f}ms")
                    else:
                        print(f"[server] #{req_count} {cmd} "
                              f"in {elapsed_ms:.1f}ms")

                    if injected_delay_ms > 0 and cmd in ('predict_action', 'predict_action_realtime'):
                        time.sleep(injected_delay_ms / 1000.0)

                    try:
                        conn.send(reply)
                    except (BrokenPipeError, ConnectionResetError):
                        print("[server] Client gone before reply could be sent")
                        break
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[server] Shutting down.")
        try:
            listener.close()
        except Exception:
            pass
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        print("[server] Bye.")


def main():
    parser = argparse.ArgumentParser(
        description="StarVLA inference server — dd/ pick-to-turntable variant "
                    "(load model once, serve over IPC)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--socket", type=str, default=DEFAULT_SOCKET,
                        help=f"Unix socket path (default: {DEFAULT_SOCKET})")
    parser.add_argument("--injected_delay", type=int, default=30,
                        help="Artificial delay (ms) added after each inference "
                             "call before sending the reply. Simulates slower "
                             "models for RTC timing tests. Default: 50")
    args = parser.parse_args()
    serve(args.socket, args.checkpoint, args.injected_delay)


if __name__ == "__main__":
    main()
