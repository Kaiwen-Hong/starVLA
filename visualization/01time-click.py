"""
Stopwatch Timer — click to start, click to stop.
Displays minutes : seconds . milliseconds
Run: python 01time-click.py
Opens http://localhost:8765
"""

import http.server
import webbrowser
import threading

PORT = 8765

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stopwatch</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #0a0a0f;
    font-family: 'JetBrains Mono', monospace;
    cursor: pointer;
    user-select: none;
    overflow: hidden;
  }

  /* subtle animated gradient ring */
  .ring {
    position: absolute;
    width: 420px; height: 420px;
    border-radius: 50%;
    background: conic-gradient(from 0deg, #6366f1, #a855f7, #ec4899, #6366f1);
    opacity: 0.15;
    animation: spin 6s linear infinite;
    filter: blur(30px);
    transition: opacity 0.5s;
  }
  .ring.active { opacity: 0.35; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .container {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 28px;
  }

  #time {
    font-size: 6rem;
    font-weight: 700;
    letter-spacing: 4px;
    color: #e2e8f0;
    text-shadow: 0 0 40px rgba(99,102,241,0.4);
    transition: text-shadow 0.3s;
  }
  #time.active {
    text-shadow: 0 0 60px rgba(168,85,247,0.6), 0 0 120px rgba(99,102,241,0.3);
  }

  #ms {
    font-size: 3rem;
    font-weight: 300;
    color: #94a3b8;
    margin-left: 4px;
  }
  #ms.active { color: #c4b5fd; }

  #status {
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 6px;
    text-transform: uppercase;
    color: #475569;
    transition: color 0.3s;
  }
  #status.active { color: #a78bfa; }

  .hint {
    position: fixed;
    bottom: 32px;
    font-size: 0.8rem;
    color: #334155;
    letter-spacing: 2px;
  }

  /* lap list */
  #laps {
    margin-top: 12px;
    max-height: 180px;
    overflow-y: auto;
    width: 360px;
    scrollbar-width: thin;
    scrollbar-color: #1e1e2e transparent;
  }
  .lap {
    display: flex;
    justify-content: space-between;
    padding: 6px 16px;
    font-size: 0.85rem;
    color: #64748b;
    border-bottom: 1px solid #1e1e2e;
  }
  .lap:first-child { color: #a78bfa; }
</style>
</head>
<body>

<div class="ring" id="ring"></div>

<div class="container">
  <div id="status">Ready</div>
  <div>
    <span id="time">00:00</span><span id="ms">.000</span>
  </div>
  <div id="laps"></div>
</div>

<div class="hint">click anywhere to start / stop &nbsp;&middot;&nbsp; double-click to reset</div>

<script>
  const timeEl  = document.getElementById('time');
  const msEl    = document.getElementById('ms');
  const statusEl= document.getElementById('status');
  const ringEl  = document.getElementById('ring');
  const lapsEl  = document.getElementById('laps');

  let running   = false;
  let startTs   = 0;
  let elapsed   = 0;   // ms accumulated before current run
  let rafId     = null;
  let lapCount  = 0;

  function fmt(totalMs) {
    const mins = Math.floor(totalMs / 60000);
    const secs = Math.floor((totalMs % 60000) / 1000);
    const ms   = Math.floor(totalMs % 1000);
    return {
      main: String(mins).padStart(2,'0') + ':' + String(secs).padStart(2,'0'),
      sub:  '.' + String(ms).padStart(3,'0')
    };
  }

  function render(totalMs) {
    const {main, sub} = fmt(totalMs);
    timeEl.textContent = main;
    msEl.textContent   = sub;
  }

  function tick() {
    const now = performance.now();
    render(elapsed + (now - startTs));
    rafId = requestAnimationFrame(tick);
  }

  function start() {
    running = true;
    startTs = performance.now();
    statusEl.textContent = 'Running';
    statusEl.classList.add('active');
    timeEl.classList.add('active');
    msEl.classList.add('active');
    ringEl.classList.add('active');
    tick();
  }

  function stop() {
    running = false;
    elapsed += performance.now() - startTs;
    cancelAnimationFrame(rafId);
    statusEl.textContent = 'Stopped';
    statusEl.classList.remove('active');
    timeEl.classList.remove('active');
    msEl.classList.remove('active');
    ringEl.classList.remove('active');

    // record lap
    lapCount++;
    const div = document.createElement('div');
    div.className = 'lap';
    const {main, sub} = fmt(elapsed);
    div.innerHTML = '<span>#' + lapCount + '</span><span>' + main + sub + '</span>';
    lapsEl.prepend(div);
  }

  function reset() {
    running = false;
    elapsed = 0;
    lapCount = 0;
    cancelAnimationFrame(rafId);
    render(0);
    statusEl.textContent = 'Ready';
    statusEl.classList.remove('active');
    timeEl.classList.remove('active');
    msEl.classList.remove('active');
    ringEl.classList.remove('active');
    lapsEl.innerHTML = '';
  }

  document.body.addEventListener('click', () => {
    if (running) stop(); else start();
  });

  document.body.addEventListener('dblclick', (e) => {
    e.preventDefault();
    reset();
  });
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode())

    def log_message(self, *args):
        pass  # silence logs


def main():
    server = http.server.HTTPServer(("", PORT), Handler)
    print(f"Stopwatch running at  http://localhost:{PORT}")
    print("Press Ctrl+C to quit")
    threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
