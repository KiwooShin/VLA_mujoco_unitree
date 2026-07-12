"""Flask MJPEG + REST web UI for the REPL demo (code/demo.py, RF-1 split).

Owns the single `_start_web_ui` entry point that boots the Flask app (HTML
shell + `/execute`, `/scene_info`, `/new_scene`, `/events`, `/stream` routes)
in a background daemon thread.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator

import numpy as np

from code.apps.repl.constants import WEB_PORT
from code.apps.repl.executor import EventBus, Executor
from code.apps.repl.planner import Planner, SceneManager


def _start_web_ui(bus: EventBus, executor: Executor, planner: Planner,
                   scene_manager: SceneManager, port: int = WEB_PORT) -> threading.Thread | None:
    """Start Flask web server in background thread.

    Returns:
        The background server thread, or None if Flask is not installed.
    """
    try:
        from flask import Flask, Response, request, jsonify, render_template_string
    except ImportError:
        print("[demo] Flask not installed. Falling back to terminal UI.", flush=True)
        return None

    app = Flask(__name__)

    HTML = """<!DOCTYPE html>
<html>
<head>
  <title>G1Nav Interactive Demo</title>
  <style>
    body { font-family: monospace; background: #1a1a1a; color: #e0e0e0; margin: 0; }
    .container { display: flex; height: 100vh; }
    .video-pane { flex: 2; display: flex; align-items: center; justify-content: center; background: #000; }
    .video-pane img { max-width: 100%; max-height: 80vh; }
    .side-pane { flex: 1; padding: 16px; overflow-y: auto; background: #222; border-left: 2px solid #444; }
    h2 { color: #61dafb; }
    .plan-item { padding: 4px 8px; margin: 4px 0; border-radius: 4px; }
    .plan-item.pending  { background: #333; }
    .plan-item.running  { background: #1a4a1a; border-left: 3px solid #4caf50; }
    .plan-item.done     { background: #1a2a4a; border-left: 3px solid #2196f3; }
    .plan-item.failed   { background: #4a1a1a; border-left: 3px solid #f44336; }
    .progress-bar { height: 8px; background: #333; border-radius: 4px; margin: 4px 0; }
    .progress-fill { height: 100%; background: #4caf50; border-radius: 4px; transition: width 0.3s; }
    textarea { width: 100%; background: #111; color: #e0e0e0; border: 1px solid #444; padding: 8px;
               font-family: monospace; font-size: 14px; border-radius: 4px; resize: vertical; }
    button { background: #61dafb; color: #000; border: none; padding: 8px 16px;
             border-radius: 4px; cursor: pointer; font-weight: bold; margin: 4px; }
    button:hover { background: #21b4cb; }
    .chat-msg { padding: 4px 0; border-bottom: 1px solid #333; font-size: 13px; }
    .chat-msg.assistant { color: #61dafb; }
    .chat-msg.user { color: #fff; }
    .chat-msg.system { color: #888; }
    #scene-desc { font-size: 12px; color: #aaa; white-space: pre; }
    #status-badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
                    font-size: 12px; font-weight: bold; }
    .status-idle    { background: #555; color: #ccc; }
    .status-running { background: #1a4a1a; color: #4caf50; }
    .status-done    { background: #1a2a4a; color: #2196f3; }
  </style>
</head>
<body>
<div class="container">
  <div class="video-pane">
    <div>
      <h2 style="text-align:center; margin:8px">G1Nav Live View</h2>
      <img id="live-view" src="/stream" onerror="this.alt='No stream (idle)'" alt="Waiting..."/>
      <p style="text-align:center; color:#888; font-size:12px">ego | third-person</p>
    </div>
  </div>
  <div class="side-pane">
    <h2>G1Nav Demo REPL
      <span id="status-badge" class="status-idle">IDLE</span>
    </h2>

    <div id="scene-desc">(loading scene...)</div>
    <hr/>

    <h3>Plan</h3>
    <div id="plan-list">(no plan yet)</div>
    <hr/>

    <h3>Instruction</h3>
    <textarea id="instruction" rows="3" placeholder="Type instruction here..."></textarea>
    <br/>
    <button onclick="sendInstruction()">Execute</button>
    <button onclick="newScene()">New Scene</button>
    <hr/>

    <h3>Chat / Progress</h3>
    <div id="chat-log" style="max-height: 300px; overflow-y: auto;"></div>
  </div>
</div>

<script>
let lastTs = 0;
let polling = false;

function addChat(text, role) {
  const div = document.getElementById('chat-log');
  const msg = document.createElement('div');
  msg.className = 'chat-msg ' + role;
  msg.textContent = '[' + new Date().toLocaleTimeString() + '] ' + text;
  div.appendChild(msg);
  div.scrollTop = div.scrollHeight;
}

function sendInstruction() {
  const instr = document.getElementById('instruction').value.trim();
  if (!instr) return;
  addChat('You: ' + instr, 'user');
  fetch('/execute', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({instruction: instr})
  }).then(r => r.json()).then(data => {
    if (data.clarify) {
      addChat('Bot: ' + data.clarify, 'assistant');
    } else {
      addChat('Bot: Executing plan: ' + data.plan.join(', '), 'assistant');
      updatePlan(data.plan, data.statuses);
    }
  });
}

function newScene() {
  fetch('/new_scene', {method: 'POST'}).then(r => r.json()).then(data => {
    document.getElementById('scene-desc').textContent = data.scene_desc;
    addChat('System: New scene generated', 'system');
    document.getElementById('plan-list').innerHTML = '(no plan yet)';
  });
}

function updatePlan(goals, statuses) {
  const div = document.getElementById('plan-list');
  div.innerHTML = goals.map((g, i) => {
    const s = (statuses || [])[i] || 'pending';
    return '<div class="plan-item ' + s + '">' + (i+1) + '. ' + g + ' <em>(' + s + ')</em></div>';
  }).join('');
}

function pollEvents() {
  fetch('/events?since=' + lastTs)
    .then(r => r.json())
    .then(data => {
      data.events.forEach(ev => {
        lastTs = Math.max(lastTs, ev._ts || 0);
        if (ev.type === 'goal_start') {
          addChat('Running: ' + ev.goal, 'system');
          document.getElementById('status-badge').className = 'status-badge status-running';
          document.getElementById('status-badge').textContent = 'RUNNING';
        } else if (ev.type === 'goal_done') {
          const status = ev.success ? '✓' : '✗';
          addChat(status + ' ' + ev.goal + ' → ' + ev.failure_tag, ev.success ? 'assistant' : 'system');
        } else if (ev.type === 'episode_done') {
          document.getElementById('status-badge').className = 'status-badge status-done';
          document.getElementById('status-badge').textContent = 'DONE';
          addChat('Episode done: ' + ev.n_success + '/' + ev.n_total + ' succeeded', 'assistant');
        } else if (ev.type === 'clarify') {
          addChat('Bot: ' + ev.message, 'assistant');
        } else if (ev.type === 'search_stub') {
          addChat('[STUB] ' + ev.message, 'system');
        } else if (ev.type === 'goto_progress') {
          // Update progress bar silently
        } else if (ev.type === 'plan_updated') {
          updatePlan(ev.goals, ev.statuses);
        }
      });
    }).catch(() => {}).finally(() => {
      setTimeout(pollEvents, 500);
    });
}

// Load initial scene
fetch('/scene_info').then(r => r.json()).then(data => {
  document.getElementById('scene-desc').textContent = data.scene_desc;
});

pollEvents();
</script>
</body>
</html>"""

    # State shared between threads
    _current_goals   = []
    _exec_lock       = threading.Lock()
    _exec_thread     = None
    _stream_frame    = [None]  # latest JPEG frame for MJPEG stream
    _stream_lock     = threading.Lock()

    def _run_execute(instruction: str) -> None:
        nonlocal _current_goals
        # Parse
        goals, clarify = planner.parse(instruction)
        if clarify:
            bus.emit({"type": "clarify", "message": clarify})
            return

        _current_goals = goals
        bus.emit({
            "type": "plan_updated",
            "goals": [str(g) for g in goals],
            "statuses": [g.status for g in goals],
        })

        # Execute
        results = executor.execute(goals)
        bus.emit({
            "type": "plan_updated",
            "goals": [str(g) for g in goals],
            "statuses": [g.status for g in goals],
        })

    @app.route("/")
    def index() -> str:
        return render_template_string(HTML)

    @app.route("/scene_info")
    def scene_info() -> Response:
        return jsonify({"scene_desc": scene_manager.describe_scene()})

    @app.route("/new_scene", methods=["POST"])
    def new_scene() -> Response:
        scene_manager.new_scene()
        return jsonify({"scene_desc": scene_manager.describe_scene()})

    @app.route("/execute", methods=["POST"])
    def execute() -> Response | tuple[Response, int]:
        nonlocal _exec_thread
        data        = request.get_json() or {}
        instruction = data.get("instruction", "").strip()
        if not instruction:
            return jsonify({"error": "empty instruction"}), 400

        goals, clarify = planner.parse(instruction)
        if clarify:
            bus.emit({"type": "clarify", "message": clarify})
            return jsonify({
                "clarify": clarify,
                "plan": [],
                "statuses": [],
            })

        plan_strs = [str(g) for g in goals]

        # Launch in background thread
        if _exec_thread and _exec_thread.is_alive():
            return jsonify({"error": "execution in progress", "plan": plan_strs}), 429

        _exec_thread = threading.Thread(
            target=_run_execute, args=(instruction,), daemon=True
        )
        _exec_thread.start()

        return jsonify({
            "clarify": None,
            "plan": plan_strs,
            "statuses": ["pending"] * len(goals),
        })

    @app.route("/events")
    def events() -> Response:
        since = float(request.args.get("since", 0))
        evts  = bus.get_events(since_ts=since)
        return jsonify({"events": evts})

    @app.route("/stream")
    def stream() -> Response:
        """MJPEG stream (static placeholder — real video saved as MP4)."""
        def gen() -> Iterator[bytes]:
            while True:
                with _stream_lock:
                    frame = _stream_frame[0]
                if frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    # Return placeholder frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + _placeholder_jpeg() + b'\r\n')
                time.sleep(0.1)
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _placeholder_jpeg() -> bytes:
        try:
            import cv2
            img = np.zeros((240, 640, 3), dtype=np.uint8)
            cv2.putText(img, "G1Nav Demo — waiting for rollout",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 2)
            _, buf = cv2.imencode('.jpg', img)
            return buf.tobytes()
        except Exception:
            return b''

    def _run_flask() -> None:
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=_run_flask, daemon=True)
    t.start()
    print(f"[demo] Web UI started at http://localhost:{port}", flush=True)
    return t
