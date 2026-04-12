#!/usr/bin/env python3
"""
RemoteModel — thin client that mimics the StarVLA model API but forwards
predict_action / predict_action_realtime calls to inference_server.py over
a Unix-domain socket.

The whole point: avoid the ~30s model-load cost when iterating on
closedloop_rtc_v6.py. Run inference_server.py once in another terminal;
this client connects, fetches metadata (chunk_len, norm_stats, checkpoint
path), and proxies inference calls.

Used by ur5n/dd/closedloop_rtc_v6.py when --use_server is enabled (default).
The dd-side server defaults to /tmp/starvla_infer_dd.sock so it can run
side by side with the 2dd-side server (/tmp/starvla_infer.sock).
"""

import threading
from multiprocessing.connection import Client


AUTHKEY = b'starvla'


class RemoteModel:
    """Drop-in replacement for the local model object on the client side.

    Exposes only the surface area that closedloop_rtc_v6.py touches:
        - chunk_len             (attr)
        - norm_stats            (attr — needed by client-side
                                 baseframework.unnormalize_actions)
        - checkpoint_path       (attr — for rollout config logging)
        - predict_action(examples, **kwargs)
        - predict_action_realtime(examples, prev_action_chunk_normalized,
                                  inference_delay, **kwargs)

    Thread-safe: a single lock serializes send/recv on the socket so the
    Inferencer thread and the main thread can share one connection. The
    underlying inference is GPU-bound and serial on the server side
    anyway, so this lock costs nothing in throughput.
    """

    def __init__(self, socket_path):
        self._socket_path = socket_path
        self._lock = threading.Lock()
        print(f"[RemoteModel] Connecting to {socket_path}...")
        try:
            self._conn = Client(socket_path, family='AF_UNIX', authkey=AUTHKEY)
        except (FileNotFoundError, ConnectionRefusedError) as e:
            raise RuntimeError(
                f"Could not connect to inference server at {socket_path}: {e}\n"
                f"  Start it first:  python ur5n/dd/inference_server.py"
            ) from e

        # Fetch metadata so the rest of the script doesn't need the model
        # in-process to read chunk_len / norm_stats.
        reply = self._call({'cmd': 'info'})
        self.chunk_len = reply['chunk_len']
        self.norm_stats = reply['norm_stats']
        self.checkpoint_path = reply['checkpoint_path']
        print(f"[RemoteModel] Connected. chunk_len={self.chunk_len}  "
              f"checkpoint={self.checkpoint_path}")

    def _call(self, msg):
        with self._lock:
            self._conn.send(msg)
            reply = self._conn.recv()
        if not isinstance(reply, dict) or not reply.get('ok'):
            err = reply.get('error') if isinstance(reply, dict) else f"bad reply: {reply!r}"
            raise RuntimeError(f"remote call failed: {err}")
        return reply

    def predict_action(self, examples, **kwargs):
        reply = self._call({
            'cmd': 'predict_action',
            'examples': examples,
            'infer_kwargs': kwargs,
        })
        return {'normalized_actions': reply['normalized_actions']}

    def predict_action_realtime(self, examples, prev_action_chunk_normalized,
                                inference_delay, **kwargs):
        reply = self._call({
            'cmd': 'predict_action_realtime',
            'examples': examples,
            'prev_action_chunk_normalized': prev_action_chunk_normalized,
            'inference_delay': inference_delay,
            'infer_kwargs': kwargs,
        })
        return {'normalized_actions': reply['normalized_actions']}

    def close(self):
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            pass
