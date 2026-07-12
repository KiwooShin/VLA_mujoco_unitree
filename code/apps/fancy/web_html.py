"""Flask web-UI HTML/CSS/JS template for the fancy demo (code/fancy_demo.py,
RF-1 split). Pure template data, no logic — kept in its own module so
code/apps/fancy/web.py (the Flask route logic) stays comfortably under the
package's line-count budget.
"""

from __future__ import annotations


_HTML_FANCY = """<!DOCTYPE html>
<html>
<head>
  <title>G1Nav Fancy Demo</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Courier New', monospace; background: #0d0d14; color: #e0e0e0; }
    .container { display: flex; height: 100vh; }
    .video-pane {
      flex: 3; display: flex; flex-direction: column; align-items: center;
      justify-content: center; background: #000; padding: 8px;
    }
    .video-pane img { max-width: 100%; max-height: 75vh; border: 1px solid #333; }
    .video-label { color: #555; font-size: 11px; margin-top: 4px; }
    .side-pane {
      flex: 1; min-width: 300px; padding: 14px; overflow-y: auto;
      background: #13131f; border-left: 2px solid #2a2a40;
    }
    h1 { color: #a0f0d0; font-size: 16px; margin-bottom: 10px; letter-spacing: 1px; }
    h3 { color: #7080c0; font-size: 13px; margin: 10px 0 4px 0; }
    .state-badge {
      display: inline-block; padding: 3px 10px; border-radius: 12px;
      font-size: 13px; font-weight: bold; margin-bottom: 8px;
    }
    .badge-idle     { background:#333; color:#aaa; }
    .badge-searching{ background:#004070; color:#00cfff; }
    .badge-located  { background:#003020; color:#40ff80; }
    .badge-moving   { background:#403000; color:#ffa020; }
    .badge-reached  { background:#300010; color:#ff6080; }
    .badge-failed   { background:#301010; color:#888; }
    .dist-badge {
      float: right; color: #ffd060; font-size: 13px; font-weight: bold;
    }
    .prompt-box {
      background: #1a1a2e; border: 1px solid #3a3a5a; padding: 8px;
      border-radius: 4px; font-size: 12px; color: #d0d8f0; margin: 6px 0; word-break: break-word;
    }
    .scene-box {
      background: #0f0f1e; border: 1px solid #222; padding: 6px;
      font-size: 11px; color: #8898b8; white-space: pre; max-height: 120px;
      overflow-y: auto; border-radius: 4px;
    }
    textarea {
      width: 100%; background: #0a0a14; color: #e0e8ff; border: 1px solid #3a3a5a;
      padding: 8px; font-family: monospace; font-size: 13px; border-radius: 4px;
      resize: vertical;
    }
    button {
      background: #2040a0; color: #e0f0ff; border: none; padding: 7px 14px;
      border-radius: 4px; cursor: pointer; font-weight: bold; margin: 3px 2px;
      font-size: 12px; letter-spacing: 0.5px;
    }
    button:hover { background: #3060d0; }
    button.danger { background: #602020; }
    button.danger:hover { background: #903030; }
    .log-entry { font-size: 11px; border-bottom: 1px solid #1a1a2e; padding: 3px 0; }
    .log-sys  { color: #556677; }
    .log-user { color: #c0d8f0; }
    .log-bot  { color: #50c0a0; }
    .log-ok   { color: #40d060; }
    .log-fail { color: #d04040; }
    #log-panel { max-height: 200px; overflow-y: auto; background: #090912; padding: 6px;
                 border-radius: 4px; border: 1px solid #1a1a2e; }
    .result-box { background: #1a1a2a; border: 1px solid #2a3a5a; padding: 8px;
                  border-radius: 4px; font-size: 12px; margin-top: 6px; }
    .result-ok   { border-color: #40d060; }
    .result-fail { border-color: #d04040; }
    hr { border: none; border-top: 1px solid #202030; margin: 10px 0; }
    .tip { color: #445; font-size: 10px; margin-top: 4px; }
  </style>
</head>
<body>
<div class="container">
  <div class="video-pane">
    <h1 style="margin-bottom:6px;">G1Nav Fancy Demo</h1>
    <img id="live-view" src="/stream" onerror="this.alt='No stream'" alt="Loading..."/>
    <div class="video-label">ACTIVE CAM (HEAD far / PROXIMITY near, CAM-2 handoff) &nbsp;|&nbsp;
      BEV FOLLOW-CAM (45° elevation, diagonal)
      &nbsp;·&nbsp; overlays: path trail · target ring · FOV cone · status banner
    </div>
  </div>
  <div class="side-pane">
    <h1>G1Nav Fancy Demo
      <span id="state-badge" class="state-badge badge-idle">IDLE</span>
      <span id="dist-badge" class="dist-badge" style="display:none"></span>
    </h1>

    <h3>Scene</h3>
    <div id="scene-box" class="scene-box">(loading...)</div>
    <h3>Active Prompt</h3>
    <div id="prompt-box" class="prompt-box">(none)</div>
    <hr/>

    <h3>Send Instruction</h3>
    <textarea id="instruction" rows="2"
      placeholder="e.g. 'find the red ball' / 'go to the orange cube'"></textarea>
    <button onclick="sendInstr()">Execute</button>
    <button onclick="newScene()">New Scene</button>
    <p class="tip">Name the object you want, e.g. 'find the red ball' -- the robot
      pursues exactly that object. Ambiguous instructions (e.g. 'the ball' with two
      balls) get a one-line clarification; unmatched ones list the scene's objects.
      Chain goals: 'find the red ball then find the yellow cube'.</p>
    <hr/>

    <h3>Last Result</h3>
    <div id="result-box" class="result-box">(no results yet)</div>
    <hr/>

    <h3>Log</h3>
    <div id="log-panel"></div>
  </div>
</div>

<script>
let pollTs = 0;
let executing = false;

function addLog(text, cls) {
  const panel = document.getElementById('log-panel');
  const d = document.createElement('div');
  d.className = 'log-entry ' + (cls || 'log-sys');
  d.textContent = new Date().toLocaleTimeString() + ' ' + text;
  panel.insertBefore(d, panel.firstChild);
  if (panel.children.length > 80) panel.removeChild(panel.lastChild);
}

function updateStateBadge(state) {
  const b = document.getElementById('state-badge');
  const clsMap = {
    'IDLE': 'badge-idle',
    'SEARCHING': 'badge-searching',
    'LOCATED': 'badge-located',
    'MOVING': 'badge-moving',
    'REACHED': 'badge-reached',
    'FAILED': 'badge-failed',
  };
  b.textContent = state;
  b.className = 'state-badge ' + (clsMap[state] || 'badge-idle');
}

function sendInstr() {
  const txt = document.getElementById('instruction').value.trim();
  if (!txt) return;
  if (executing) { addLog('Execution in progress — please wait', 'log-sys'); return; }
  addLog('> ' + txt, 'log-user');
  document.getElementById('prompt-box').textContent = txt;
  executing = true;
  fetch('/execute', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({instruction: txt})
  }).then(r => r.json()).then(d => {
    if (d.error) { addLog('Error: ' + d.error, 'log-fail'); executing = false; }
    else if (d.clarify) { addLog('Bot: ' + d.clarify, 'log-bot'); executing = false; }
    else {
      const tgt = (d.targets && d.targets.length) ? (' -> ' + d.targets.join(' then ')) : '';
      addLog('Launched: ' + txt + tgt, 'log-bot');
    }
  }).catch(() => { executing = false; });
}

function newScene() {
  if (executing) { addLog('Execution in progress', 'log-sys'); return; }
  fetch('/new_scene', {method: 'POST'}).then(r => r.json()).then(d => {
    document.getElementById('scene-box').textContent = d.scene_desc;
    document.getElementById('prompt-box').textContent = '(none)';
    document.getElementById('result-box').className = 'result-box';
    document.getElementById('result-box').textContent = '(no results yet)';
    addLog('New scene generated', 'log-sys');
  });
}

function poll() {
  fetch('/status').then(r => r.json()).then(d => {
    updateStateBadge(d.state || 'IDLE');

    const db = document.getElementById('dist-badge');
    if (d.dist != null) {
      db.style.display = '';
      db.textContent = d.dist.toFixed(2) + 'm';
    } else {
      db.style.display = 'none';
    }

    if (d.result) {
      const ok = d.result.success;
      const rb = document.getElementById('result-box');
      rb.className = 'result-box ' + (ok ? 'result-ok' : 'result-fail');
      const ft = d.result.failure_tag || '?';
      const steps = d.result.steps || 0;
      const dist = d.result.final_dist != null ? d.result.final_dist.toFixed(3) + 'm' : '?';
      rb.textContent = (ok ? '✓ SUCCESS' : '✗ ' + ft.toUpperCase()) +
        '  steps=' + steps + '  final_dist=' + dist;
      if (d.result.video_path) {
        rb.textContent += '  video: ' + d.result.video_path;
      }
      if (executing && (ft === 'success' || ft.startsWith('fall') || ft.startsWith('didnt') || ft === 'scan_timeout')) {
        executing = false;
        if (ok) addLog('SUCCESS! dist=' + dist + ' steps=' + steps, 'log-ok');
        else addLog('FAILED: ' + ft + ' dist=' + dist, 'log-fail');
      }
    }

    if (d.scene_desc && d.scene_desc !== document.getElementById('scene-box').textContent) {
      document.getElementById('scene-box').textContent = d.scene_desc;
    }

  }).catch(() => {}).finally(() => setTimeout(poll, 400));
}

// Initial scene load
fetch('/scene_info').then(r => r.json()).then(d => {
  document.getElementById('scene-box').textContent = d.scene_desc;
});

poll();
</script>
</body>
</html>"""
