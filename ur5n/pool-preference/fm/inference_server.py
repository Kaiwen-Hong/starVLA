#!/usr/bin/env python3
"""
StarVLA inference server — pool (QwenPI flow-matching, 10-D world-frame
delta action).

Loads the model once into GPU memory and serves predict_action /
predict_action_realtime requests over a Unix-domain socket. Pair with
remote_model.RemoteModel on the client side.

Default socket /tmp/starvla_infer_pool.sock so this server can run side
by side with the airhockey + dd servers (which use other sockets).

Usage:
    # Terminal 1 — start the server (load once, leave running)
    python ur5n/pool-preference/fm/inference_server.py \
      --checkpoint <path-to-pool-checkpoint>

    # Terminal 2 — run an eval script with --use_server
    python ur5n/pool-preference/fm/replay_openloop_eval.py --use_server
    python ur5n/pool-preference/fm/live_openloop_eval.py   --use_server
"""

import sys
import os
import time
import argparse
import traceback
from pathlib import Path
from multiprocessing.connection import Listener

import torch  # noqa: F401 — required side-effect of CUDA init

# ── Repo setup ─────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework

DEFAULT_CHECKPOINT = (
    "checkpoints/discreteRTC/fastumi_pool_qwenPI_0522_DiT-S/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
DEFAULT_SOCKET = "/tmp/starvla_infer_pool.sock"
AUTHKEY = b'starvla'


def _detect_attn():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_model(checkpoint_path):
    print(f"[server] Loading model from: {checkpoint_path}")
    t0 = time.time()
    checkpoint_pt = Path(checkpoint_path)
    if not checkpoint_pt.exists():
        raise FileNotFoundError(
            f"Pool checkpoint not found: {checkpoint_pt}\n"
            f"  Pass --checkpoint <path> once the pool model is trained.")
    model_config, norm_stats = read_mode_config(checkpoint_pt)
    config = dict_to_namespace(model_config)
    config.trainer.pretrained_checkpoint = None
    config.framework.qwenvl.attn_implementation = _detect_attn()

    model = build_framework(cfg=config)
    model.norm_stats = norm_stats
    state_dict = torch.load(checkpoint_pt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda").eval()

    # Pool dataset native resolution is 256x256 (V3 center-crop). Mirror
    # airhockey: if image_size missing in config, fall back to [256, 256]
    # so eval-time resize matches training-time native size.
    if not getattr(config.datasets.vla_data, "image_size", None):
        config.datasets.vla_data.image_size = [256, 256]
        print("[server] [FIX] Set image_size=[256,256] (match pool native)")

    print(f"[server] Model loaded in {time.time() - t0:.1f}s  "
          f"chunk_len={model.chunk_len}  "
          f"action_dim={getattr(config.framework.action_model, 'action_dim', 'N/A')}")
    return model


def serve(socket_path, checkpoint_path, injected_delay_ms=0):
    model = load_model(checkpoint_path)

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
                            reply = {'ok': False,
                                     'error': f"unknown cmd: {cmd!r}"}
                    except Exception as e:
                        traceback.print_exc()
                        reply = {'ok': False,
                                 'error': f"{type(e).__name__}: {e}"}

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

                    if injected_delay_ms > 0 and cmd in (
                            'predict_action', 'predict_action_realtime'):
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
        description="StarVLA inference server — pool QwenPI variant")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to pool checkpoint (default is 'TODO' — "
                             "must override until the pool model is trained)")
    parser.add_argument("--socket", type=str, default=DEFAULT_SOCKET,
                        help=f"Unix socket path (default: {DEFAULT_SOCKET})")
    parser.add_argument("--injected_delay", type=int, default=0,
                        help="Artificial delay (ms) added after each inference "
                             "call before sending the reply. Default: 0")
    args = parser.parse_args()
    serve(args.socket, args.checkpoint, args.injected_delay)


if __name__ == "__main__":
    main()
