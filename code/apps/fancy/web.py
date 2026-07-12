"""Flask MJPEG + REST web UI for the fancy demo (code/fancy_demo.py, RF-1
split).

Owns the shared MJPEG stream-frame state, the status-panel state dict, and
the single `_start_fancy_web_ui` entry point that boots the Flask app in a
background daemon thread. HTML/CSS/JS template lives in
code/apps/fancy/web_html.py.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Iterator, Optional

import numpy as np

from code.apps.fancy.constants import BEV_H, STATE_FAILED, STATE_IDLE, STATE_REACHED, STATE_SEARCHING, STREAM_W, WEB_PORT, MAXSTEPS_FANCY
from code.apps.fancy.live import FancySceneManager, resolve_live_instruction
from code.apps.fancy.multi_goal import run_fancy_rollout_multi
from code.apps.fancy.rollout import run_fancy_rollout
from code.apps.fancy.web_html import _HTML_FANCY


_stream_lock: threading.Lock = threading.Lock()
_stream_frame: list[Optional[bytes]] = [None]    # bytes: latest MJPEG JPEG frame
_status_lock: threading.Lock = threading.Lock()
_status_state: dict[str, Any] = {
    "state": STATE_IDLE,
    "prompt": "",
    "dist": None,
    "step": 0,
    "scene_desc": "",
    "result": None,
}


def _set_stream_frame(bgr_frame: np.ndarray) -> None:
    """Encode BGR numpy frame to JPEG bytes and push to stream."""
    try:
        import cv2
        _, buf = cv2.imencode('.jpg', bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _stream_lock:
            _stream_frame[0] = buf.tobytes()
    except Exception:
        pass


def _placeholder_frame(state: str = STATE_IDLE, prompt: str = "") -> bytes:
    """Render a placeholder JPEG frame shown before the first rollout frame arrives."""
    try:
        import cv2
        img = np.zeros((BEV_H, STREAM_W + 3, 3), dtype=np.uint8)
        cv2.putText(img, f"G1Nav Fancy Demo  [{state}]", (20, BEV_H // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 100), 2)
        if prompt:
            cv2.putText(img, f"Prompt: {prompt[:60]}", (20, BEV_H // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(img, "Waiting for rollout...", (20, BEV_H // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        _, buf = cv2.imencode('.jpg', img)
        return buf.tobytes()
    except Exception:
        return b''




def _start_fancy_web_ui(
    inf: "Inferencer",
    scene_manager: "FancySceneManager",
    out_dir: str,
    port: int = WEB_PORT,
    maxsteps: int = MAXSTEPS_FANCY,
    render_video: bool = True,
    scenario_title: str = "G1Nav Autonomous Fetch",
) -> Optional[threading.Thread]:
    """Start Flask web UI for fancy demo in a background thread.

    Args:
        inf: Inferencer instance (goal_source='classical').
        scene_manager: FancySceneManager instance holding the current scene.
        out_dir: Output directory for per-rollout MP4s.
        port: TCP port to serve the Flask app on.
        maxsteps: Hard step cap forwarded to each rollout.
        render_video: Whether to render ego|BEV SBS frames at all.
        scenario_title: Scenario name shown on the VF-1 title card.

    Returns:
        The daemon thread running the Flask app, or None if Flask isn't
        installed.
    """
    try:
        from flask import Flask, Response, request, jsonify, render_template_string
    except ImportError:
        print("[fancy_demo] Flask not installed. Web UI unavailable.", flush=True)
        return None

    app = Flask(__name__)

    _exec_lock   = threading.Lock()
    _exec_thread = [None]

    def _scene_desc() -> str:
        # NX-15: no more "<TARGET" marker -- the sampler's target_index is only a
        # fallback default for scripted/headless callers; in live mode the real
        # target is whatever object the typed instruction resolves to, so marking
        # one object as THE target here would be misleading again.
        if scene_manager._scene_cfg is None:
            return "(no scene)"
        objs = scene_manager._scene_cfg["objects"]
        lines = []
        for i, o in enumerate(objs):
            lines.append(f"  [{i}] {o['color_name']:7s} {o['shape_name']:8s}  "
                         f"dist={o['dist_from_robot']:.2f}m")
        return "\n".join(lines)

    def _do_rollout(instruction: str, parsed: dict) -> None:
        """Run the rollout for an already-parsed+resolved instruction (see the
        /execute route below, which does the NX-15 parsing/resolution
        synchronously before launching this thread)."""
        scene_cfg = scene_manager._scene_cfg
        if scene_cfg is None:
            with _status_lock:
                _status_state['state'] = STATE_FAILED
                _status_state['result'] = {'success': False, 'failure_tag': 'no_scene', 'steps': 0}
            return

        prompt = instruction

        with _status_lock:
            _status_state['state']  = STATE_SEARCHING
            _status_state['prompt'] = prompt
            _status_state['result'] = None

        def _cb(frame_bgr: np.ndarray, state: str, dist: Optional[float], step: int) -> None:
            with _status_lock:
                _status_state['state'] = state
                _status_state['dist']  = dist
                _status_state['step']  = step
            _set_stream_frame(frame_bgr)

        os.makedirs(out_dir, exist_ok=True)
        vid_path = os.path.join(out_dir, f"fancy_ep_{int(time.time())}.mp4")

        try:
            if parsed["mode"] == "multi":
                result = run_fancy_rollout_multi(
                    inf=inf,
                    goals=parsed["goals"],
                    scene_cfg=scene_cfg,
                    maxsteps=maxsteps,
                    render_video=render_video,
                    video_path=vid_path,
                    frame_callback=_cb,
                    scenario_title=scenario_title,
                )
            else:
                # NX-15: target comes from the resolved instruction, not the
                # sampler's default scene_cfg['target_index']. scene_cfg itself
                # is left untouched (a copy carries the override) so any other
                # reader of scene_manager._scene_cfg still sees the sampler default.
                resolved_scene = dict(scene_cfg)
                resolved_scene["target_index"] = parsed["target_indices"][0]
                result = run_fancy_rollout(
                    inf=inf,
                    scene_cfg=resolved_scene,
                    prompt=prompt,
                    maxsteps=maxsteps,
                    render_video=render_video,
                    video_path=vid_path,
                    frame_callback=_cb,
                    scenario_title=scenario_title,
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {'success': False, 'failure_tag': 'error', 'steps': 0, 'final_dist': 999.0}

        with _status_lock:
            _status_state['state']  = STATE_REACHED if result.get('success') else STATE_FAILED
            # keep only JSON-serializable scalars: the raw result dict can hold
            # np.ndarrays, which make /status throw until the auto-reset below
            _status_state['result'] = {
                k: (v.item() if hasattr(v, 'item') and getattr(v, 'ndim', 1) == 0 else v)
                for k, v in result.items()
                if isinstance(v, (bool, int, float, str, type(None)))
                or (hasattr(v, 'item') and getattr(v, 'ndim', 1) == 0)
            }
            _status_state['dist']   = result.get('final_dist')

        print(f"[fancy_web] rollout done: {result.get('failure_tag', result.get('success'))}  "
              f"video={result.get('video_path')}", flush=True)

        # Auto new scene after brief pause
        time.sleep(3.0)
        scene_manager.new_scene()
        with _status_lock:
            _status_state['state']  = STATE_IDLE
            _status_state['prompt'] = ''
            _status_state['dist']   = None
            _status_state['result'] = None
            _status_state['scene_desc'] = _scene_desc()

    @app.route("/")
    def index() -> str:
        return render_template_string(_HTML_FANCY)

    @app.route("/scene_info")
    def scene_info() -> Response:
        return jsonify({"scene_desc": _scene_desc()})

    @app.route("/new_scene", methods=["POST"])
    def new_scene() -> Response:
        scene_manager.new_scene()
        return jsonify({"scene_desc": _scene_desc()})

    @app.route("/execute", methods=["POST"])
    def execute() -> Any:
        if _exec_thread[0] and _exec_thread[0].is_alive():
            return jsonify({"error": "Execution in progress"}), 429
        data        = request.get_json() or {}
        instruction = data.get("instruction", "").strip()
        if not instruction:
            return jsonify({"error": "empty instruction"}), 400

        scene_cfg = scene_manager._scene_cfg
        if scene_cfg is None:
            return jsonify({"error": "no scene loaded"}), 400

        # NX-15: parse + resolve the instruction against the CURRENT scene
        # synchronously, before launching the rollout thread, so ambiguous/
        # no-match/no-parse instructions get an immediate response over the
        # same /execute channel the UI already reads (see sendInstr() JS above)
        # instead of silently driving the wrong (or a pre-picked) object.
        parsed = resolve_live_instruction(instruction, scene_cfg)
        if parsed["mode"] == "clarify":
            with _status_lock:
                _status_state['prompt'] = instruction
            return jsonify({"launched": False, "clarify": parsed["message"]})
        if parsed["mode"] in ("no_parse", "no_match"):
            with _status_lock:
                _status_state['prompt'] = instruction
            return jsonify({"launched": False, "error": parsed["message"]})

        t = threading.Thread(target=_do_rollout, args=(instruction, parsed), daemon=True)
        _exec_thread[0] = t
        t.start()
        targets = [f"{g['color']} {g['shape']}" for g in parsed["goals"]]
        return jsonify({"launched": True, "instruction": instruction,
                         "mode": parsed["mode"], "targets": targets})

    @app.route("/status")
    def status() -> Response:
        with _status_lock:
            st = dict(_status_state)
        st['scene_desc'] = _scene_desc()
        return jsonify(st)

    @app.route("/stream")
    def stream() -> Response:
        def gen() -> "Iterator[bytes]":
            while True:
                with _stream_lock:
                    frame = _stream_frame[0]
                if frame is not None:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    with _status_lock:
                        st = _status_state.get('state', STATE_IDLE)
                        pt = _status_state.get('prompt', '')
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + _placeholder_frame(st, pt) + b'\r\n')
                time.sleep(0.08)  # ~12 fps stream cap
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def _run_flask() -> None:
        import logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=_run_flask, daemon=True)
    t.start()
    print(f"[fancy_demo] Web UI: http://localhost:{port}", flush=True)
    return t

