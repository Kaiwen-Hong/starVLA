"""Logging utilities: web stopwatch (SSE), timing tracker, visualization, rollout saving."""

import json
import time
import queue
import threading
import http.server
import webbrowser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ═════════════════════════════════════════════════════════════════════
#  Timing tracker
# ═════════════════════════════════════════════════════════════════════

class TimingTracker:
    """Accumulates per-step timing and prints a summary."""

    def __init__(self):
        self.steps = 0
        self.obs_ms = 0.0
        self.infer_ms = 0.0
        self.send_ms = 0.0
        self.wall_ms = 0.0

    def record(self, obs_ms, infer_ms, send_ms, wall_ms):
        self.steps += 1
        self.obs_ms += obs_ms
        self.infer_ms += infer_ms
        self.send_ms += send_ms
        self.wall_ms += wall_ms

    def summary(self):
        if self.steps == 0:
            return
        n = self.steps
        avg_wall = self.wall_ms / n
        avg_hz = 1000.0 / avg_wall if avg_wall > 0 else 0
        print(f"\n{'=' * 60}")
        print(f"  Timing Summary ({n} steps)")
        print(f"{'=' * 60}")
        print(f"  Avg wall-clock per step: {avg_wall:6.1f}ms  ({avg_hz:.1f}Hz)")
        print(f"  ├─ Observation:          {self.obs_ms / n:6.1f}ms")
        print(f"  ├─ Inference:            {self.infer_ms / n:6.1f}ms")
        if self.send_ms > 0:
            print(f"  ├─ Execution:            {self.send_ms / n:6.1f}ms")
        print(f"  └─ Sum:                  {(self.obs_ms + self.infer_ms + self.send_ms) / n:6.1f}ms")
        print(f"{'=' * 60}")


# ═════════════════════════════════════════════════════════════════════
#  Visualization
# ═════════════════════════════════════════════════════════════════════

_VIZ_DIMS = [
    (0, "x (world, m)", "#e41a1c"),
    (1, "y (world, m)", "#377eb8"),
    (2, "z (world, m)", "#4daf4a"),
    (6, "gripper",      "#ff7f00"),
]


def visualize_step(camera_image, current_ee, n_exec, step_idx,
                   save_path, instruction, mode,
                   pred_poses=None, exec_poses=None, new_poses=None,
                   inference_delay=0):
    """Visualize predicted trajectory for one step.

    For sync mode: only pred_poses is needed.
    For rtc mode: exec_poses (just-executed chunk) + new_poses (new RTC prediction).
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.15, wspace=0.30,
                  width_ratios=[1, 1.3])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(camera_image)
    ax_img.set_title("Camera (center-cropped)", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    axes = []
    for row, (dim_idx, label, color) in enumerate(_VIZ_DIMS):
        share = axes[0] if axes else None
        ax = fig.add_subplot(gs[row, 1], sharex=share)
        axes.append(ax)

        start_val = current_ee[dim_idx] if dim_idx < 3 else None

        if mode == "sync" and pred_poses is not None:
            _plot_sync(ax, pred_poses, n_exec, color, start_val, dim_idx, row)
        elif mode in ("async", "rtc") and exec_poses is not None and new_poses is not None:
            _plot_rtc(ax, exec_poses, new_poses, n_exec, inference_delay,
                      color, start_val, dim_idx, row)

        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        if row == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=3)
        if row < len(_VIZ_DIMS) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (s) — 20Hz", fontsize=10)

    title = f'Step {step_idx} ({mode}): exec={n_exec}'
    if mode == "rtc":
        title += f', delay={inference_delay}'
    fig.suptitle(f'{title}\n"{instruction}"',
                 fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_sync(ax, pred_poses, n_exec, color, start_val, dim_idx, row):
    T = pred_poses.shape[0]
    ts = np.arange(T) / 20.0
    vals = pred_poses[:, dim_idx]
    ax.plot(ts[:n_exec], vals[:n_exec], "o-", markersize=5, linewidth=2.0,
            color=color, label="executed" if row == 0 else None)
    if n_exec < T:
        ax.plot(ts[n_exec - 1:], vals[n_exec - 1:], "o:", markersize=3,
                linewidth=1.2, color=color, alpha=0.3,
                label="tail" if row == 0 else None)
    if start_val is not None:
        ax.axhline(start_val, color=color, linewidth=1.0, linestyle="--", alpha=0.4)
    ax.axvline(ts[n_exec - 1], color="black", linewidth=1.0, linestyle=":", alpha=0.6,
               label="exec boundary" if row == 0 else None)


def _plot_rtc(ax, exec_poses, new_poses, n_exec, inference_delay,
              color, start_val, dim_idx, row):
    T_exec = exec_poses.shape[0]
    T_new = new_poses.shape[0]
    ts_exec = np.arange(T_exec) / 20.0
    ts_new = np.arange(T_new) / 20.0 + n_exec / 20.0

    exec_vals = exec_poses[:, dim_idx]
    ax.plot(ts_exec[:n_exec], exec_vals[:n_exec], "o-", markersize=5,
            linewidth=2.0, color=color, label="executed" if row == 0 else None)
    if n_exec < T_exec:
        ax.plot(ts_exec[n_exec - 1:], exec_vals[n_exec - 1:], "o:",
                markersize=3, linewidth=1.2, color=color, alpha=0.3,
                label="prev tail" if row == 0 else None)

    new_vals = new_poses[:, dim_idx]
    if inference_delay > 0 and inference_delay <= T_new:
        ax.plot(ts_new[:inference_delay], new_vals[:inference_delay], "s--",
                markersize=4, linewidth=1.5, color=color, alpha=0.6,
                label="RTC prefix" if row == 0 else None)
    free_start = max(0, inference_delay)
    if free_start < T_new:
        connect = max(0, free_start - 1)
        ax.plot(ts_new[connect:], new_vals[connect:], "D--", markersize=3,
                linewidth=1.5, color=color, alpha=0.5,
                label="RTC new" if row == 0 else None)

    if start_val is not None:
        ax.axhline(start_val, color=color, linewidth=1.0, linestyle="--", alpha=0.4)
    ax.axvline(ts_exec[n_exec - 1], color="black", linewidth=1.0,
               linestyle=":", alpha=0.6, label="exec boundary" if row == 0 else None)


# ═════════════════════════════════════════════════════════════════════
#  Rollout saving
# ═════════════════════════════════════════════════════════════════════

class RolloutSaver:
    """Manages rollout directory, config, per-step data, and async visualization."""

    def __init__(self, rollout_dir: Path, run_config: dict):
        self.dir = rollout_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "images").mkdir(exist_ok=True)
        self.log = []
        self._viz_pool = ThreadPoolExecutor(max_workers=1)

        with open(self.dir / "config.json", "w") as f:
            json.dump(run_config, f, indent=2)
        print(f"  Rollout saving: {self.dir}")

    def save_step(self, step_data: dict):
        self.log.append(step_data)

    def submit_viz(self, fn, *args, **kwargs):
        self._viz_pool.submit(fn, *args, **kwargs)

    def finalize(self):
        print("Waiting for pending viz saves...")
        self._viz_pool.shutdown(wait=True)
        if self.log:
            with open(self.dir / "rollout.json", "w") as f:
                json.dump(self.log, f, indent=2)
            print(f"Rollout saved: {self.dir}  ({len(self.log)} steps)")


# ═════════════════════════════════════════════════════════════════════
#  Stopwatch web server (SSE-based, daemon thread)
# ═════════════════════════════════════════════════════════════════════

STOPWATCH_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Policy Stopwatch</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: #0a0a0f; font-family: 'JetBrains Mono', monospace;
    user-select: none; overflow: hidden;
  }
  .ring {
    position: absolute; width: 420px; height: 420px; border-radius: 50%;
    background: conic-gradient(from 0deg, #6366f1, #a855f7, #ec4899, #6366f1);
    opacity: 0.15; animation: spin 6s linear infinite;
    filter: blur(30px); transition: opacity 0.5s;
  }
  .ring.active { opacity: 0.35; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .container {
    position: relative; z-index: 1; display: flex;
    flex-direction: column; align-items: center; gap: 28px;
  }
  #time {
    font-size: 6rem; font-weight: 700; letter-spacing: 4px; color: #e2e8f0;
    text-shadow: 0 0 40px rgba(99,102,241,0.4); transition: text-shadow 0.3s;
  }
  #time.active { text-shadow: 0 0 60px rgba(168,85,247,0.6), 0 0 120px rgba(99,102,241,0.3); }
  #ms { font-size: 3rem; font-weight: 300; color: #94a3b8; margin-left: 4px; }
  #ms.active { color: #c4b5fd; }
  #status {
    font-size: 1rem; font-weight: 400; letter-spacing: 6px;
    text-transform: uppercase; color: #475569; transition: color 0.3s;
  }
  #status.active { color: #a78bfa; }
  .hint { position: fixed; bottom: 32px; font-size: 0.8rem; color: #334155; letter-spacing: 2px; }
  #laps {
    margin-top: 12px; max-height: 180px; overflow-y: auto; width: 360px;
    scrollbar-width: thin; scrollbar-color: #1e1e2e transparent;
  }
  .lap {
    display: flex; justify-content: space-between; padding: 6px 16px;
    font-size: 0.85rem; color: #64748b; border-bottom: 1px solid #1e1e2e;
  }
  .lap:first-child { color: #a78bfa; }
  .splash {
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    background: #0a0a0f; z-index: 100; opacity: 0;
    pointer-events: none; transition: opacity 0.8s ease;
  }
  .splash.active { opacity: 1; pointer-events: auto; }
  .splash-title {
    font-size: 2.6rem; font-weight: 700; color: #e2e8f0; text-align: center;
    line-height: 1.4; letter-spacing: 3px; padding: 0 40px;
    animation: splash-glow 2.5s ease-in-out infinite;
  }
  @keyframes splash-glow {
    0%, 100% { text-shadow: 0 0 30px rgba(99,102,241,0.3), 0 0 60px rgba(168,85,247,0.15); }
    50%      { text-shadow: 0 0 60px rgba(99,102,241,0.6), 0 0 120px rgba(168,85,247,0.4), 0 0 180px rgba(236,72,153,0.15); }
  }
  .splash-sub {
    margin-top: 20px; font-size: 1rem; font-weight: 300;
    letter-spacing: 6px; text-transform: uppercase; color: #64748b;
  }
  .splash-bar-track {
    margin-top: 48px; width: 320px; height: 3px;
    background: #1e1e2e; border-radius: 2px; overflow: hidden;
  }
  .splash-bar {
    height: 100%; width: 0%;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899); border-radius: 2px;
  }
</style>
</head>
<body>
<div class="splash" id="splash">
  <div class="splash-title" id="splashTitle"></div>
  <div class="splash-sub" id="splashSub">starting in 5s</div>
  <div class="splash-bar-track"><div class="splash-bar" id="splashBar"></div></div>
</div>
<div class="ring" id="ring"></div>
<div class="container">
  <div id="status">Waiting for policy</div>
  <div><span id="time">00:00</span><span id="ms">.000</span></div>
  <div id="laps"></div>
</div>
<div class="hint">auto-controlled by robot policy via SSE</div>
<script>
  const timeEl=document.getElementById('time'), msEl=document.getElementById('ms'),
        statusEl=document.getElementById('status'), ringEl=document.getElementById('ring'),
        lapsEl=document.getElementById('laps'), splashEl=document.getElementById('splash'),
        splashTitle=document.getElementById('splashTitle'),
        splashSub=document.getElementById('splashSub'),
        splashBar=document.getElementById('splashBar');
  let running=false, startTs=0, elapsed=0, rafId=null, lapCount=0, splashRaf=null;

  function fmt(t){
    const m=Math.floor(t/60000), s=Math.floor((t%60000)/1000), ms=Math.floor(t%1000);
    return {main:String(m).padStart(2,'0')+':'+String(s).padStart(2,'0'),
            sub:'.'+String(ms).padStart(3,'0')};
  }
  function render(t){const{main,sub}=fmt(t);timeEl.textContent=main;msEl.textContent=sub;}
  function tick(){render(elapsed+(performance.now()-startTs));rafId=requestAnimationFrame(tick);}
  function showSplash(title,dur){
    splashTitle.textContent=title; splashEl.classList.add('active');
    const t0=performance.now();
    (function a(){
      const p=Math.min((performance.now()-t0)/dur,1);
      splashBar.style.width=(p*100)+'%';
      const r=Math.ceil((dur-(performance.now()-t0))/1000);
      splashSub.textContent=r>0?'starting in '+r+'s':'go';
      if(p<1)splashRaf=requestAnimationFrame(a);
    })();
  }
  function hideSplash(){if(splashRaf)cancelAnimationFrame(splashRaf);splashEl.classList.remove('active');}
  function start(){
    if(running)return; hideSplash(); running=true; elapsed=0; startTs=performance.now();
    statusEl.textContent='Running';
    [statusEl,timeEl,msEl,ringEl].forEach(e=>e.classList.add('active')); tick();
  }
  function stop(){
    if(!running)return; running=false; elapsed+=performance.now()-startTs;
    cancelAnimationFrame(rafId); statusEl.textContent='Finished';
    [statusEl,timeEl,msEl,ringEl].forEach(e=>e.classList.remove('active'));
    lapCount++; const d=document.createElement('div'); d.className='lap';
    const{main,sub}=fmt(elapsed);
    d.innerHTML='<span>#'+lapCount+'</span><span>'+main+sub+'</span>';
    lapsEl.prepend(d);
  }
  const es=new EventSource('/events');
  es.addEventListener('splash',e=>{const[t,d]=e.data.split('|');showSplash(t,parseInt(d)||5000);});
  es.addEventListener('timer',e=>{if(e.data==='start')start();else if(e.data==='stop')stop();});
  es.onerror=()=>{statusEl.textContent='Disconnected';};
  document.body.addEventListener('click',()=>{
    if(splashEl.classList.contains('active')){hideSplash();start();return;}
    if(running)stop();else start();
  });
  document.body.addEventListener('dblclick',e=>{
    e.preventDefault(); hideSplash(); running=false; elapsed=0; lapCount=0;
    cancelAnimationFrame(rafId); render(0); statusEl.textContent='Waiting for policy';
    [statusEl,timeEl,msEl,ringEl].forEach(el=>el.classList.remove('active'));
    lapsEl.innerHTML='';
  });
</script>
</body>
</html>"""


class StopwatchServer:
    """Embedded HTTP + SSE server for the browser stopwatch. Runs in a daemon thread."""

    def __init__(self, port=8765):
        self._port = port
        self._sse_queues = []
        self._lock = threading.Lock()
        self._server = None

    def _broadcast(self, event: str, data: str):
        with self._lock:
            dead = []
            for q in self._sse_queues:
                try:
                    q.put_nowait((event, data))
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._sse_queues.remove(q)

    def start_timer(self, splash_title=None, splash_duration=5):
        if splash_title:
            self._broadcast("splash", f"{splash_title}|{splash_duration * 1000}")
            time.sleep(splash_duration)
        self._broadcast("timer", "start")

    def stop_timer(self):
        self._broadcast("timer", "stop")

    def serve(self, open_browser=True):
        parent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/events":
                    self._handle_sse()
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(STOPWATCH_HTML.encode())

            def _handle_sse(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                q = queue.Queue(maxsize=64)
                with parent._lock:
                    parent._sse_queues.append(q)
                try:
                    while True:
                        try:
                            event, data = q.get(timeout=15)
                            self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode())
                            self.wfile.flush()
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with parent._lock:
                        if q in parent._sse_queues:
                            parent._sse_queues.remove(q)

            def log_message(self, *args):
                pass

        for port in range(self._port, self._port + 100):
            try:
                self._server = http.server.HTTPServer(("", port), Handler)
                break
            except OSError:
                continue
        else:
            raise RuntimeError(f"No free port in {self._port}-{self._port + 99}")

        self._port = port
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        url = f"http://localhost:{self._port}"
        print(f"Stopwatch server running at {url}")
        if open_browser:
            threading.Timer(0.3, lambda: webbrowser.open(url)).start()
        return url

    def shutdown(self):
        if self._server:
            self._server.shutdown()
