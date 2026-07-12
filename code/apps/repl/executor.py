"""Sub-goal executor + progress event bus for the REPL demo (code/demo.py,
RF-1 split).

Owns:
  - EventBus: thread-safe event bus used by both the terminal REPL and the
    Flask web UI to observe rollout progress.
  - Executor: runs a list of Planner SubGoals one by one using the trained
    goto/maneuver/search policies, emitting EventBus progress events.
"""

from __future__ import annotations

import collections
import math
import os
import threading
import time
from typing import Any

from code.apps.repl.constants import (
    GOTO_CKPT, MANEUVER_CKPT, MAXSTEPS_GOTO, MAXSTEPS_MANEUVER, DEMO_OUT_DIR,
    _get_lang_emb,
)
from code.apps.repl.maneuver_inferencer import ManeuverInferencer
from code.apps.repl.planner import SubGoal


# ---------------------------------------------------------------------------
# Progress event bus
# ---------------------------------------------------------------------------
class EventBus:
    """Simple thread-safe event bus for UI updates."""

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._events = collections.deque(maxlen=200)
        self._state  = {}

    def emit(self, event: dict) -> None:
        with self._lock:
            event['_ts'] = time.time()
            self._events.append(event)
            self._state.update(event)

    def get_events(self, since_ts: float = 0.0) -> list[dict]:
        with self._lock:
            return [e for e in self._events if e.get('_ts', 0) > since_ts]

    def get_state(self) -> dict:
        with self._lock:
            return dict(self._state)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
class Executor:
    """
    Runs sub-goals one by one, using the trained policies.

    Emits progress events to EventBus.
    """

    def __init__(
        self,
        scene_manager: "SceneManager",
        bus: EventBus,
        device: str = "cpu",
        render_video: bool = True,
        out_dir: str = str(DEMO_OUT_DIR),
        maxsteps_goto: int = MAXSTEPS_GOTO,
        maxsteps_maneuver: int = MAXSTEPS_MANEUVER,
    ) -> None:
        self.scene      = scene_manager
        self.bus        = bus
        self.device     = device
        self.render_video = render_video
        self.out_dir    = out_dir
        self.maxsteps_goto     = maxsteps_goto
        self.maxsteps_maneuver = maxsteps_maneuver
        self._goto_inferencer = None
        self._maneuver_inferencer = None
        self._ep_count = 0

    def _get_goto_inferencer(self) -> "Inferencer":
        """Lazily construct (and cache) the goto skill's Inferencer."""
        if self._goto_inferencer is None:
            from code.inferencer import Inferencer
            self._goto_inferencer = Inferencer(
                checkpoint_path=GOTO_CKPT,
                arch='A',
                device=self.device,
                goal_source='classical',
                verbose=False,
            )
            print(f"[executor] goto inferencer loaded: {GOTO_CKPT}", flush=True)
        return self._goto_inferencer

    def _get_maneuver_inferencer(self) -> ManeuverInferencer:
        """Lazily construct (and cache) the maneuver skill's inferencer."""
        if self._maneuver_inferencer is None:
            self._maneuver_inferencer = ManeuverInferencer(
                checkpoint_path=str(MANEUVER_CKPT),
                device=self.device,
            )
        return self._maneuver_inferencer

    def execute(self, goals: list[SubGoal]) -> list[dict[str, Any]]:
        """Execute all sub-goals; returns list of result dicts."""
        results = []
        self._ep_count += 1
        ep_id = self._ep_count

        self.bus.emit({
            "type": "episode_start",
            "ep": ep_id,
            "n_goals": len(goals),
            "goals": [str(g) for g in goals],
        })

        for gi, goal in enumerate(goals):
            goal.status = "running"
            self.bus.emit({
                "type": "goal_start",
                "goal_idx": gi,
                "goal": str(goal),
                "skill": goal.skill,
            })

            if goal.skill == "goto":
                result = self._run_goto(goal, ep_id, gi)
            elif goal.skill == "maneuver":
                result = self._run_maneuver(goal, ep_id, gi)
            elif goal.skill == "search":
                result = self._run_search_stub(goal, ep_id, gi)
            else:
                result = {"success": False, "failure_tag": "unknown_skill"}

            goal.result = result
            goal.status = "done" if result.get("success") else "failed"

            self.bus.emit({
                "type": "goal_done",
                "goal_idx": gi,
                "goal": str(goal),
                "skill": goal.skill,
                "success": result.get("success", False),
                "failure_tag": result.get("failure_tag", "unknown"),
                "steps": result.get("steps", 0),
                "video_path": result.get("video_path"),
            })

            results.append(result)

            # Early abort if goal failed critically
            if not result.get("success") and goal.skill != "search":
                # Continue anyway (demo resilience)
                pass

        self.bus.emit({
            "type": "episode_done",
            "ep": ep_id,
            "n_success": sum(1 for r in results if r.get("success")),
            "n_total": len(results),
        })

        return results

    def _run_goto(self, goal: SubGoal, ep_id: int, gi: int) -> dict[str, Any]:
        """Run goto skill using Inferencer."""
        scene_cfg    = self.scene.scene_cfg
        if scene_cfg is None:
            return {"success": False, "failure_tag": "no_scene"}

        # Build a scene_cfg with the correct target
        objects = scene_cfg["objects"]
        # Find target index matching goal color + shape
        tgt_idx = None
        for i, o in enumerate(objects):
            if o["color_name"] == goal.color and o["shape_name"] == goal.shape:
                tgt_idx = i
                break
        if tgt_idx is None:
            return {"success": False, "failure_tag": "target_not_in_scene"}

        sc = dict(scene_cfg)
        sc["target_index"] = tgt_idx
        sc["instruction"]  = goal.description

        video_path = None
        if self.render_video:
            os.makedirs(self.out_dir, exist_ok=True)
            video_path = os.path.join(
                self.out_dir,
                f"ep{ep_id:03d}_goal{gi:02d}_goto_{goal.color}_{goal.shape}.mp4"
            )

        def _progress_cb(info: dict) -> None:
            self.bus.emit({
                "type": "goto_progress",
                "goal_idx": gi,
                "step": info["step"],
                "pct": info["pct"],
                "dist": info.get("dist", 0.0),
            })

        inf = self._get_goto_inferencer()

        # Get lang embedding (from cache or zeros)
        lang_emb = _get_lang_emb(goal.description)

        t0 = time.time()
        try:
            result = inf.rollout(
                scene_cfg=sc,
                instruction=goal.description,
                lang_emb=lang_emb,
                maxsteps=self.maxsteps_goto,
                render_video=self.render_video,
                video_path=video_path,
                render_tp=True,   # ego+third-person SBS video
            )
        except Exception as e:
            print(f"[executor] goto rollout failed: {e}", flush=True)
            return {"success": False, "failure_tag": "error", "steps": 0, "video_path": None}
        dt = time.time() - t0

        return {
            "success":       result.success,
            "failure_tag":   result.failure_tag,
            "steps":         result.steps,
            "final_dist":    result.final_dist,
            "forward_disp":  result.forward_disp,
            "wall_time_s":   dt,
            "video_path":    result.video_path,
        }

    def _run_maneuver(self, goal: SubGoal, ep_id: int, gi: int) -> dict[str, Any]:
        """Run maneuver skill."""
        from code.maneuver_scene import sample_maneuver_scene, derive_rng

        # Check if current scene_cfg is already a maneuver scene
        current_sc = self.scene.scene_cfg
        if current_sc is not None and current_sc.get('task') == 'maneuver':
            # Already a maneuver scene — use it directly
            sc = current_sc
            # Override direction if needed
            if goal.direction and sc.get('turn_direction') != goal.direction:
                sc = dict(sc)
                sc['turn_direction'] = goal.direction
                sc['target_heading'] = math.pi / 2 if goal.direction == 'left' else -math.pi / 2
        else:
            # Sample a fresh maneuver scene matching the goal landmark
            rng   = derive_rng(999 + ep_id, gi)
            sc    = sample_maneuver_scene(rng)

            # Override landmark color/shape to match goal
            lm_idx = sc.get('landmark_index', 0)
            if lm_idx < len(sc['objects']):
                found = False
                for i, o in enumerate(sc['objects']):
                    if o['color_name'] == goal.color and o['shape_name'] == goal.shape:
                        sc['landmark_index'] = i
                        sc['landmark_xy']    = (o['x'], o['y'])
                        found = True
                        break
                if not found:
                    sc['objects'][lm_idx]['color_name'] = goal.color
                    sc['objects'][lm_idx]['shape_name'] = goal.shape

            # Set turn direction
            sc['turn_direction'] = goal.direction
            sc['target_heading'] = math.pi / 2 if goal.direction == 'left' else -math.pi / 2

        video_path = None
        if self.render_video:
            os.makedirs(self.out_dir, exist_ok=True)
            video_path = os.path.join(
                self.out_dir,
                f"ep{ep_id:03d}_goal{gi:02d}_maneuver_{goal.direction}_{goal.color}_{goal.shape}.mp4"
            )

        def _progress_cb(info: dict) -> None:
            self.bus.emit({
                "type": "maneuver_progress",
                "goal_idx": gi,
                "step": info["step"],
                "pct": info["pct"],
                "phase": info.get("phase", ""),
                "heading_err": info.get("heading_err_deg", 0.0),
            })

        t0 = time.time()
        maneuver_inf = self._get_maneuver_inferencer()
        result = maneuver_inf.rollout(
            scene_cfg=sc,
            instruction=goal.description,
            maxsteps=self.maxsteps_maneuver,
            render_video=self.render_video,
            video_path=video_path,
            progress_cb=_progress_cb,
        )
        dt = time.time() - t0

        result['wall_time_s'] = dt
        return result

    def _run_search_stub(self, goal: SubGoal, ep_id: int, gi: int) -> dict[str, Any]:
        """
        search_then_goto: student-driven CCW scan to find the target (out-of-FOV),
        then GOTO once it enters the FOV.

        Mechanism (H3 student-driven scan, WBC-free):
          1. Robot starts with target outside initial FOV.
          2. Classical grounding runs every GROUNDING_PERIOD steps.
          3. Student injects wz>0 (CCW) into the action head — no WBC ONNX.
          4. When grounding detects the target AND bearing < 40°, scan exits → GOTO.
          5. Classical HSV grounding guides approach to within STOP_R.

        This is DISTINCT from goto (which may also scan, but for targets in-FOV;
        search scenes guarantee the target is OUTSIDE initial FOV).
        """
        from code.eval_search import _run_search_rollout, STOP_R_SEARCH, MAXSTEPS_SEARCH

        scene_cfg = self.scene.scene_cfg
        if scene_cfg is None:
            return {"success": False, "failure_tag": "no_scene", "steps": 0}

        # Find target object in scene
        objects = scene_cfg["objects"]
        tgt_idx = None
        for i, o in enumerate(objects):
            if o["color_name"] == goal.color and o["shape_name"] == goal.shape:
                tgt_idx = i
                break
        if tgt_idx is None:
            self.bus.emit({
                "type": "search_info",
                "goal_idx": gi,
                "message": f"[search] {goal.color} {goal.shape} not in scene — cannot search",
            })
            return {"success": False, "failure_tag": "target_not_in_scene", "steps": 0}

        sc = dict(scene_cfg)
        sc["target_index"] = tgt_idx
        sc["instruction"]  = goal.description
        sc["stop_r"]       = STOP_R_SEARCH

        self.bus.emit({
            "type": "search_start",
            "goal_idx": gi,
            "message": (
                f"[search] Searching for {goal.color} {goal.shape} — "
                "student-driven CCW scan (WBC-free) → GOTO on detect"
            ),
        })

        video_path = None
        if self.render_video:
            os.makedirs(self.out_dir, exist_ok=True)
            video_path = os.path.join(
                self.out_dir,
                f"ep{ep_id:03d}_goal{gi:02d}_search_{goal.color}_{goal.shape}.mp4"
            )

        inf = self._get_goto_inferencer()
        t0  = time.time()

        try:
            raw = _run_search_rollout(
                inf=inf,
                scene_cfg=sc,
                instruction=goal.description,
                maxsteps=MAXSTEPS_SEARCH,
                render_video=self.render_video,
                video_path=video_path,
            )
        except Exception as e:
            import traceback
            print(f"[executor] search rollout failed: {e}", flush=True)
            traceback.print_exc()
            raw = dict(success=False, spotted=False, scan_steps=0, failure_tag='error',
                       steps=0, final_dist=999.0, fell=False, ms_per_step=0.0, video_path=None)

        dt = time.time() - t0

        self.bus.emit({
            "type": "search_done",
            "goal_idx": gi,
            "spotted":  raw.get("spotted", False),
            "success":  raw.get("success", False),
            "scan_steps": raw.get("scan_steps", 0),
            "steps":    raw.get("steps", 0),
        })

        return {
            "success":      raw["success"],
            "failure_tag":  raw["failure_tag"],
            "steps":        raw["steps"],
            "final_dist":   raw.get("final_dist", 0.0),
            "spotted":      raw.get("spotted", False),
            "scan_steps":   raw.get("scan_steps", 0),
            "wall_time_s":  dt,
            "video_path":   raw.get("video_path"),
        }
