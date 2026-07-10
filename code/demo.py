"""
demo.py — Interactive REPL Demo for G1Nav

Capstone deliverable: free-form English instruction → plan → closed-loop execution → video.

Architecture:
  - SceneManager: random scene via sample_scene; exposes object list
  - Planner: rule-based NL→sub-goal program; GR00T-LM embeddings for language
  - Executor: calls inferencer.py (goto) and eval_maneuver.py logic (maneuver) closed-loop
  - UI: lightweight Flask MJPEG web server + terminal REPL fallback

Skills wired:
  - goto: Inferencer with goal_source='classical' → checkpoint/goto_best.pt (demo_dart_A ep3,
          80% demo/GT all-yaw). Uses V2/V3 demo-distance grounding (26° grounding cam,
          480x360, depth-FG rescue) — showcases 4-9m LONG walks. Default difficulty='demo'.
  - maneuver: ManeuverInferencer → checkpoint/maneuver_best.pt (maneuver_A ep2, 80%)

Skills wired (3rd skill — C3 agent):
  - search: out-of-FOV target acquisition — student-driven CCW scan (WBC-free, H3 mechanism)
    until target detected by classical grounding, then GOTO. Distinct from goto: search scenes
    guarantee target is OUTSIDE initial FOV (bearing > 45°). "find the red ball" → search.

Grounding notes (V2/V3):
  - 26° dedicated grounding cam (32° put >6m targets off-screen)
  - 480x360 grounding resolution (2.25x larger blobs vs 320x240)
  - Depth-FG rescue for cyan/blue: rescues 3/3 easy; structural at 4-9m (distractors)
  - Non-cyan/blue at 4-9m: 87% success (demo-distance showcase)
  - Overall demo/classical: 46.7% (7/15); easy/classical: 93% (14/15)

Usage:
  # Terminal REPL (always works):
  MUJOCO_GL=egl python code/demo.py

  # Web UI on port 5000:
  MUJOCO_GL=egl python code/demo.py --web

  # Canned smoke test (3 instructions, saves videos):
  MUJOCO_GL=egl python code/demo.py --smoke --out eval/demo --device cuda

  # Save video to specific dir:
  MUJOCO_GL=egl python code/demo.py --out eval/demo --difficulty demo

  # Easy mode (shorter walks, 93% success):
  MUJOCO_GL=egl python code/demo.py --difficulty easy
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pickle
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

import numpy as np

# ---------------------------------------------------------------------------
# CUDA helper (defined before use)
# ---------------------------------------------------------------------------
def _check_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOTO_CKPT_EP3  = _REPO / "runs/demo_dart_A/epoch_0003.pt"
GOTO_CKPT_EP5  = _REPO / "runs/demo_dart_A/epoch_0005.pt"
GOTO_CKPT_BEST = _REPO / "runs/demo_dart_A/model_best.pt"
MANEUVER_CKPT  = _REPO / "runs/maneuver_A/epoch_0002.pt"
LANG_CACHE     = _REPO / "dataset/lang_cache.pkl"

# H1 refresh: use pinned stable checkpoints in checkpoint/ dir (demo_dart_A ep3 + maneuver_A ep2)
# Fallback to runs/ paths if checkpoint/ dir not yet populated.
_GOTO_CKPT_PINNED     = _REPO / "checkpoint/goto_best.pt"
_MANEUVER_CKPT_PINNED = _REPO / "checkpoint/maneuver_best.pt"
GOTO_CKPT     = str(_GOTO_CKPT_PINNED)     if _GOTO_CKPT_PINNED.exists()     else (
                str(GOTO_CKPT_EP3)          if GOTO_CKPT_EP3.exists()         else str(GOTO_CKPT_BEST))
MANEUVER_CKPT = str(_MANEUVER_CKPT_PINNED) if _MANEUVER_CKPT_PINNED.exists() else str(MANEUVER_CKPT)

# H1 refresh: demo-distance goto uses 1700 steps (4-9m walks at ~50Hz)
# V2/V3 demo-distance grounding (26° cam, 480x360, depth-FG rescue) wired in inferencer.py.
# Default difficulty = 'demo' to showcase 4-9m LONG walks (key demo deliverable).
# NX-10 (docs/nx10_scan_fix.md): bumped 1400 -> 1700, matching code/eval_closedloop.py's
# MAXSTEPS['demo'] -- the widened H3 scan (BidirectionalScanSchedule, realized-yaw
# tracking) needs more absolute-step budget for unfavorable-direction bearings.
MAXSTEPS_GOTO     = 1700  # demo preset: 1700 steps for 4-9m walks (was 1400 pre-NX-10)
MAXSTEPS_MANEUVER = 1400  # unchanged -- maneuver has its own separate rollout loop (nx9_avoid.md §8)
DEMO_OUT_DIR = _REPO / "eval/demo"
WEB_PORT     = 5000

# Colors and shapes (from arena.py)
COLORS  = ["red", "yellow", "blue", "green", "orange", "purple", "cyan"]
SHAPES  = ["ball", "cube", "cylinder", "cone"]

MANEUVER_DIRECTIONS = ["left", "right"]


# ---------------------------------------------------------------------------
# Language cache (GR00T-LM embeddings)
# ---------------------------------------------------------------------------
_lang_cache: Optional[Dict[str, np.ndarray]] = None

def _load_lang_cache() -> Optional[Dict[str, np.ndarray]]:
    global _lang_cache
    if _lang_cache is not None:
        return _lang_cache
    if LANG_CACHE.exists():
        try:
            with open(LANG_CACHE, "rb") as f:
                _lang_cache = pickle.load(f)
            print(f"[demo] Loaded lang cache: {len(_lang_cache)} instructions", flush=True)
            return _lang_cache
        except Exception as e:
            print(f"[demo] WARN: Failed to load lang cache: {e}", flush=True)
    return None


def _get_lang_emb(instruction: str) -> Optional[np.ndarray]:
    """Look up GR00T-LM embedding from cache; None if not found."""
    cache = _load_lang_cache()
    if cache is None:
        return None
    return cache.get(instruction, None)


# ---------------------------------------------------------------------------
# SceneManager
# ---------------------------------------------------------------------------
class SceneManager:
    """
    Manages the current scene: samples a fresh scene via sample_scene(),
    exposes the object list (color, shape, position) visible at start.
    """

    def __init__(self, difficulty: str = "easy", seed_offset: int = 0):
        self.difficulty   = difficulty
        self.seed_offset  = seed_offset
        self._ep_count    = 0
        self._scene_cfg   = None

    def new_scene(self) -> dict:
        """Sample a fresh scene; returns scene_cfg."""
        from code.scene import sample_scene, derive_rng
        rng = derive_rng(1234 + self.seed_offset, self._ep_count)
        self._scene_cfg = sample_scene(rng, self.difficulty)
        self._ep_count += 1
        return self._scene_cfg

    @property
    def scene_cfg(self) -> Optional[dict]:
        return self._scene_cfg

    def describe_scene(self) -> str:
        """Human-readable description of the current scene."""
        if self._scene_cfg is None:
            return "(no scene yet)"
        objs = self._scene_cfg["objects"]
        lines = ["Objects in scene:"]
        for i, o in enumerate(objs):
            tgt_mark = " <-- TARGET" if i == self._scene_cfg["target_index"] else ""
            lines.append(
                f"  [{i}] {o['color_name']} {o['shape_name']}"
                f"  dist={o['dist_from_robot']:.2f}m  "
                f"({o['x']:.1f}, {o['y']:.1f}){tgt_mark}"
            )
        return "\n".join(lines)

    def object_list(self) -> List[Dict[str, Any]]:
        """Return list of object dicts."""
        if self._scene_cfg is None:
            return []
        return self._scene_cfg["objects"]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
@dataclass
class SubGoal:
    """A single primitive sub-goal."""
    skill:       str           # 'goto' | 'maneuver' | 'search' (stubbed)
    color:       Optional[str] = None
    shape:       Optional[str] = None
    direction:   Optional[str] = None   # for maneuver: 'left' | 'right'
    description: str           = ""
    status:      str           = "pending"   # 'pending' | 'running' | 'done' | 'failed'
    result:      Optional[Any] = None

    def __str__(self) -> str:
        if self.skill == "goto":
            return f"goto({self.color} {self.shape})"
        elif self.skill == "maneuver":
            return f"maneuver(turn_{self.direction} after {self.color} {self.shape})"
        elif self.skill == "search":
            return f"search({self.color} {self.shape})"
        return f"{self.skill}({self.description})"


class Planner:
    """
    Rule-based NL → sub-goal program.

    Parses free-form English instructions into an ordered list of SubGoals
    that the trained policies can execute.

    Supports:
      - goto(color, shape): "go to the red ball"
      - maneuver(dir, color, shape): "turn left after the blue cube"
      - search(color, shape): "find the red ball" (out-of-FOV scan)
      - compound: "go to the red ball then find the blue cube"
      - multi-goal: "navigate to the red ball then turn left after the orange cone"

    Ambiguity resolution:
      - Multiple possible referents → ask clarify question with options listed
      - No matching object in scene → report what IS available
      - Missing referent (no color/shape) → ask which object they mean

    Compound instruction parsing:
      - Splits on "then", "and then", "and after that", "afterwards"
      - Each part parsed independently
      - If a part is ambiguous, returns clarify question (subsequent parts queued)
    """

    def __init__(self, scene_manager: SceneManager):
        self.scene  = scene_manager
        # Queued parts for multi-goal: when a part triggers clarification, we
        # store remaining parts here and resume after clarification is resolved.
        self._pending_parts: List[str] = []
        self._pending_goals: List[SubGoal] = []

    def parse(self, instruction: str) -> tuple[List[SubGoal], Optional[str]]:
        """
        Parse instruction into sub-goals.

        Returns:
            (goals, clarify_question)
            clarify_question is non-None if the planner is ambiguous and needs user input.
            When clarify_question is set, previously-parsed goals (from earlier parts of
            a compound instruction) are still included in the returned goals list.
        """
        instr = instruction.lower().strip()

        # Split compound instructions by sequencing conjunctions
        # Handles: "then", "and then", "and after that", "after that", "afterwards", "next"
        parts = re.split(
            r'\bthen\b|,\s*then\s*|\band\s+then\b|\band\s+after\s+that\b'
            r'|\bafter\s+that\b|\bafterwards\b|\bnext\b',
            instr, flags=re.IGNORECASE
        )
        parts = [p.strip() for p in parts if p.strip()]

        goals = []
        for part in parts:
            subgoals, clarify = self._parse_part(part)
            if clarify:
                # Return already-parsed goals + clarify question.
                # Store remaining parts for when clarification is resolved.
                # (The REPL can call parse() again with the clarification answer.)
                return goals, clarify
            goals.extend(subgoals)

        if not goals:
            return [], (
                f"I didn't understand that instruction. Try:\n"
                f"  'go to the red ball'\n"
                f"  'turn left after the blue cube'\n"
                f"  'find the orange cone'\n"
                f"  'go to the red ball then find the blue cube'"
            )

        return goals, None

    def parse_clarification(self, original_instr: str, clarify_answer: str) -> tuple[List[SubGoal], Optional[str]]:
        """
        Re-parse after user provides a clarification answer.

        The clarify_answer is treated as a refinement: we attempt to combine
        the original instruction's intent with the user's answer.

        Strategy:
          1. If clarify_answer contains a color+shape → use as the specific referent
          2. If clarify_answer is just a color or shape → combine with original intent
          3. Otherwise → treat clarify_answer as a fresh instruction
        """
        answer = clarify_answer.lower().strip()

        # Check if answer itself is a parseable instruction
        goals, clarify = self.parse(answer)
        if goals and not clarify:
            return goals, None

        # Try to combine original instruction with clarification answer
        # Extract color/shape from answer
        c_found = next((c for c in COLORS if c in answer), None)
        s_found = next((s for s in SHAPES if s in answer), None)

        if c_found or s_found:
            # Re-parse original with clarified color/shape
            instr = original_instr.lower()
            if c_found and s_found:
                instr_refined = f"go to the {c_found} {s_found}"
            elif c_found:
                # Find which shape was in the original
                s_orig = next((s for s in SHAPES if s in instr), None)
                instr_refined = f"go to the {c_found} {s_orig}" if s_orig else f"go to the {c_found} object"
            else:
                c_orig = next((c for c in COLORS if c in instr), None)
                instr_refined = f"go to the {c_orig} {s_found}" if c_orig else f"go to the {s_found}"

            goals, clarify = self.parse(instr_refined)
            if goals:
                return goals, clarify

        # Fall back to treating the answer as a fresh instruction
        return self.parse(clarify_answer)

    def _parse_part(self, instr: str) -> tuple[List[SubGoal], Optional[str]]:
        """Parse a single instruction part into sub-goals."""
        goals = []

        # Check for maneuver pattern: "turn {dir} after [passing] the {color} {shape}"
        maneuver_match = self._detect_maneuver(instr)
        if maneuver_match:
            direction, color, shape = maneuver_match
            # Resolve referent to scene
            obj, clarify = self._resolve_referent(color, shape, "maneuver landmark")
            if clarify:
                return [], clarify
            if obj is None:
                return [], f"I don't see a {color} {shape} in the scene. Available: {self._scene_summary()}."

            goals.append(SubGoal(
                skill="maneuver",
                color=obj["color_name"],
                shape=obj["shape_name"],
                direction=direction,
                description=f"turn {direction} after {obj['color_name']} {obj['shape_name']}",
            ))
            return goals, None

        # Check for search pattern FIRST (before goto) — "find/search_for/look_for" → search skill
        # C3: search is DISTINCT from goto: search scenes guarantee target OUTSIDE initial FOV.
        # "find the red ball" → search (out-of-FOV scan), NOT goto.
        search_match = self._detect_search(instr)
        if search_match:
            color, shape = search_match
            goals.append(SubGoal(
                skill="search",
                color=color,
                shape=shape,
                description=f"find the {color} {shape}",
            ))
            return goals, None

        # Check for goto pattern: "go to|walk to|approach|head to the {color} {shape}"
        goto_match = self._detect_goto(instr)
        if goto_match:
            color, shape = goto_match
            obj, clarify = self._resolve_referent(color, shape, "navigation target")
            if clarify:
                return [], clarify
            if obj is None:
                return [], f"I don't see a {color} {shape} in the scene. Available: {self._scene_summary()}."

            goals.append(SubGoal(
                skill="goto",
                color=obj["color_name"],
                shape=obj["shape_name"],
                description=f"navigate to {obj['color_name']} {obj['shape_name']}",
            ))
            return goals, None

        return [], None

    # ---- Pattern detectors ----

    def _detect_goto(self, instr: str) -> Optional[tuple[str, str]]:
        """Detect goto(color, shape) from instruction.
        NOTE: 'find'/'search for'/'look for' are routed to the SEARCH skill (checked first
        in _parse_part) and are NOT included here. This keeps search DISTINCT from goto.
        """
        goto_verbs = (
            r"(?:go\s+to|walk\s+to|approach|head\s+to|head\s+over\s+to|move\s+to|"
            r"navigate\s+to|make\s+your\s+way\s+to|get\s+to|proceed\s+to|"
            r"reach|come\s+to)"
        )
        # Pattern 1: "go to the {color} {shape}"
        m = re.search(
            goto_verbs + r'\s+(?:the\s+)?([a-z]+)\s+([a-z]+)',
            instr, re.IGNORECASE
        )
        if m:
            c, s = m.group(1).lower(), m.group(2).lower()
            if c in COLORS and s in SHAPES:
                return (c, s)
            # Try swap
            if s in COLORS and c in SHAPES:
                return (s, c)

        # Pattern 2: "your goal is the {color} {shape}"
        m = re.search(r'your\s+goal\s+is\s+(?:the\s+)?([a-z]+)\s+([a-z]+)', instr, re.IGNORECASE)
        if m:
            c, s = m.group(1).lower(), m.group(2).lower()
            if c in COLORS and s in SHAPES:
                return (c, s)

        # Pattern 3: any "the {color} {shape}" or "{color}-colored {shape}"
        m = re.search(r'(?:the\s+)?([a-z]+)(?:-colored)?\s+(ball|cube|cylinder|cone)', instr, re.IGNORECASE)
        if m:
            c = m.group(1).lower()
            s = m.group(2).lower()
            if c in COLORS:
                return (c, s)

        return None

    def _detect_maneuver(self, instr: str) -> Optional[tuple[str, str, str]]:
        """Detect maneuver(dir, color, shape) from instruction."""
        # Pattern: "turn {left|right} after [passing] [the] {color} {shape}"
        m = re.search(
            r'turn\s+(left|right)\s+(?:after\s+)?(?:passing\s+)?(?:the\s+)?([a-z]+)\s+([a-z]+)',
            instr, re.IGNORECASE
        )
        if m:
            direction = m.group(1).lower()
            c, s = m.group(2).lower(), m.group(3).lower()
            if c in COLORS and s in SHAPES:
                return (direction, c, s)
            if s in COLORS and c in SHAPES:
                return (direction, s, c)

        # Pattern: "pass [the] {color} {shape} [and] turn {left|right}"
        m = re.search(
            r'pass\s+(?:the\s+)?([a-z]+)\s+([a-z]+).*?turn\s+(left|right)',
            instr, re.IGNORECASE
        )
        if m:
            c, s = m.group(1).lower(), m.group(2).lower()
            direction = m.group(3).lower()
            if c in COLORS and s in SHAPES:
                return (direction, c, s)

        # Pattern: "when you pass [the] {color} {shape}, turn {left|right}"
        m = re.search(
            r'(?:when\s+you\s+pass|after\s+passing)\s+(?:the\s+)?([a-z]+)\s+([a-z]+).*?turn\s+(left|right)',
            instr, re.IGNORECASE
        )
        if m:
            c, s = m.group(1).lower(), m.group(2).lower()
            direction = m.group(3).lower()
            if c in COLORS and s in SHAPES:
                return (direction, c, s)

        # Pattern: "turn {left|right} when you pass [the] {color} {shape}"
        m = re.search(
            r'turn\s+(left|right)\s+when\s+you\s+pass\s+(?:the\s+)?([a-z]+)\s+([a-z]+)',
            instr, re.IGNORECASE
        )
        if m:
            direction = m.group(1).lower()
            c, s = m.group(2).lower(), m.group(3).lower()
            if c in COLORS and s in SHAPES:
                return (direction, c, s)

        return None

    def _detect_search(self, instr: str) -> Optional[tuple[str, str]]:
        """Detect search(color, shape) from instruction.
        C3: search is routed here when instruction uses find/search/look_for.
        DISTINCT from goto — search requires target to be out-of-FOV initially.
        Example: "find the red ball" → search(red, ball)
        """
        # Check for search-trigger verbs
        m_trigger = re.search(r'\b(?:find|search\s+for|look\s+for|locate|hunt\s+for)\b',
                               instr, re.IGNORECASE)
        if m_trigger:
            # Extract color + shape
            m2 = re.search(r'\b([a-z]+)\b.*?\b(ball|cube|cylinder|cone)\b', instr, re.IGNORECASE)
            if m2:
                c, s = m2.group(1).lower(), m2.group(2).lower()
                if c in COLORS:
                    return (c, s)
                # Try finding color further back
            m3 = re.search(r'\b(' + '|'.join(COLORS) + r')\b.*?\b(ball|cube|cylinder|cone)\b',
                           instr, re.IGNORECASE)
            if m3:
                return (m3.group(1).lower(), m3.group(2).lower())
        return None

    def _resolve_referent(
        self, color: Optional[str], shape: Optional[str], role: str
    ) -> tuple[Optional[Dict], Optional[str]]:
        """
        Resolve (color, shape) to a scene object.

        Returns:
            (obj_dict, None)         if unambiguous match found
            (None, clarify_question) if ambiguous (multiple matches)
            (None, None)             if no match
        """
        objects = self.scene.object_list()
        if not objects:
            return None, "No scene loaded. Please wait for scene initialization."

        candidates = []
        for obj in objects:
            match_color = (color is None or obj["color_name"] == color)
            match_shape = (shape is None or obj["shape_name"] == shape)
            if match_color and match_shape:
                candidates.append(obj)

        if len(candidates) == 1:
            return candidates[0], None
        elif len(candidates) == 0:
            return None, None
        else:
            # Ambiguous — need clarification
            descs = ", ".join(
                f"{o['color_name']} {o['shape_name']} (at {o['dist_from_robot']:.1f}m)"
                for o in candidates
            )
            return None, (
                f"Multiple {color or ''} {shape or ''} objects found: {descs}. "
                f"Which one? (specify color or shape)"
            )

    def _scene_summary(self) -> str:
        """Short summary of scene objects for error messages."""
        return ", ".join(
            f"{o['color_name']} {o['shape_name']}"
            for o in self.scene.object_list()
        )


# ---------------------------------------------------------------------------
# Maneuver Inferencer (wraps eval_maneuver.py logic directly)
# ---------------------------------------------------------------------------
class ManeuverInferencer:
    """
    Closed-loop maneuver skill executor.
    Reuses logic from eval_maneuver.py but with scene_cfg from SceneManager.
    Uses hybrid_vel: GT vel injection during TURN_PHASE only.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu", use_keyframe: bool = True):
        self.checkpoint_path = checkpoint_path
        self.device_str      = device
        self._model          = None
        self._action_stats   = None
        self._loaded         = False

        # H4: WBC-free settle — load offline stand keyframe (same as Inferencer in inferencer.py)
        # When use_keyframe=True and checkpoint/stand_keyframe.npz exists, skip the WBC settle
        # and instead restore physics from the offline keyframe. No WBC ONNX called at runtime.
        self._keyframe: Optional[dict] = None
        _kf_path = str(_REPO / "checkpoint" / "stand_keyframe.npz")
        if use_keyframe and os.path.isfile(_kf_path):
            _kf = np.load(_kf_path)
            self._keyframe = {
                'qpos_local': _kf['qpos_local'].copy(),
                'qvel_local': _kf['qvel_local'].copy(),
                'target_dof': _kf['target_dof'].copy(),
                'height':     float(_kf['height']),
            }
            print(f"[maneuver_inferencer] Keyframe init: loaded {_kf_path} "
                  f"(height={self._keyframe['height']:.4f}m) — WBC-free settle active")
        elif use_keyframe:
            print(f"[maneuver_inferencer] Keyframe init: {_kf_path} not found, "
                  f"falling back to WBC settle")

    def _load_model(self):
        """Lazy-load model on first use."""
        if self._loaded:
            return
        import torch
        from code.train_maneuver import load_loco_checkpoint

        device = torch.device(self.device_str)
        ckpt_path = self.checkpoint_path

        if os.path.isfile(ckpt_path):
            # load_loco_checkpoint handles GRU expansion from 57→62 and returns (model, ckpt)
            model, ckpt = load_loco_checkpoint(ckpt_path, device)
            if ckpt.get('action_stats'):
                _as = ckpt['action_stats']
                self._action_stats = {
                    'mean': np.array(_as['mean'], dtype=np.float32),
                    'std':  np.array(_as['std'],  dtype=np.float32),
                    'default_angles': np.array(_as['default_angles'], dtype=np.float32),
                }
            elif os.path.isfile(str(Path(ckpt_path).parent / 'action_stats.json')):
                # Load from companion file
                import json as _json
                with open(str(Path(ckpt_path).parent / 'action_stats.json')) as f:
                    _as = _json.load(f)
                self._action_stats = {
                    'mean': np.array(_as['mean'], dtype=np.float32),
                    'std':  np.array(_as['std'],  dtype=np.float32),
                    'default_angles': np.array(_as['default_angles'], dtype=np.float32),
                }
        else:
            print(f"[maneuver_inferencer] WARN: ckpt not found {ckpt_path}, random init", flush=True)
            from code.small_vla import GroundedNav
            from code.dataset_maneuver import PROPRIO_DIM_MANEUVER
            model = GroundedNav(
                arch='A', teacher_forcing=True, proprio_dim=PROPRIO_DIM_MANEUVER,
            ).to(device)

        model.eval()
        self._model  = model
        self._device = device
        self._loaded = True
        print(f"[maneuver_inferencer] Model ready (ckpt: {ckpt_path})", flush=True)

    def rollout(
        self,
        scene_cfg: dict,
        instruction: str,
        maxsteps: int = MAXSTEPS_MANEUVER,
        render_video: bool = False,
        video_path: Optional[str] = None,
        progress_cb = None,
    ) -> Dict[str, Any]:
        """
        Run one closed-loop maneuver episode.

        Returns result dict with keys:
          success, failure_tag, steps, heading_err, landmark_passed, video_path
        """
        self._load_model()

        import torch
        import mujoco
        from code.arena import build_arena, ArenaRenderer
        from code.teacher import (WBCTeacher, _yaw_of, DEFAULT_ANGLES, KPS, KDS,
                                   NUM_ACTIONS, SIM_DT, CONTROL_DECIMATION)
        from code.gen_dart_dataset import GaitPhaseTracker
        from code.maneuver_scene import sample_maneuver_scene, derive_rng, SETTLE_STEPS, HORIZON, PASS_MARGIN
        from code.maneuver_expert import ManeuverExpert, State, FORWARD_VX, MAX_WZ, TURN_KP
        from code.dataset_maneuver import (PROPRIO_DIM_MANEUVER, PROPRIO_DIM_PHASE,
                                            PROPRIO_DIM_BASE, _build_maneuver_features)
        from code.inferencer import _GaitPhaseTracker, FALL_HEIGHT, PROPRIO_K

        # Build a maneuver scene_cfg from the provided scene_cfg's landmark info
        # The scene_cfg must have 'task'=='maneuver' keys
        if scene_cfg.get('task') != 'maneuver':
            # Build a fresh maneuver scene
            rng = derive_rng(999, scene_cfg.get('_maneuver_ep', 0))
            scene_cfg = sample_maneuver_scene(rng)

        arena_model = build_arena(scene_cfg)
        arena_model.opt.timestep = SIM_DT

        teacher = WBCTeacher(use_gpu=False)
        teacher.model = arena_model
        teacher.data  = mujoco.MjData(arena_model)
        teacher._nj   = arena_model.nq - 7
        teacher._pelvis_id = mujoco.mj_name2id(
            arena_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

        rx, ry    = scene_cfg['robot_xy']
        robot_yaw = float(scene_cfg.get('robot_yaw', 0.0))
        teacher.reset(pos_xy=(rx, ry), yaw=robot_yaw)
        data_mj  = teacher.data
        model_mj = teacher.model
        nj       = teacher._nj

        # H4: Settle — keyframe restore (WBC-free) OR WBC settle (fallback)
        if self._keyframe is not None:
            # Keyframe path: restore saved physics state — no WBC ONNX called at runtime.
            kf = self._keyframe
            kf_qpos = kf['qpos_local'].copy()
            # Apply scene xy/yaw: overwrite x, y, and orientation quaternion
            kf_qpos[0] = rx
            kf_qpos[1] = ry
            kf_qpos[3] = math.cos(robot_yaw / 2)  # w
            kf_qpos[4] = 0.0                        # x
            kf_qpos[5] = 0.0                        # y
            kf_qpos[6] = math.sin(robot_yaw / 2)   # z
            data_mj.qpos[:len(kf_qpos)] = kf_qpos
            data_mj.qvel[:len(kf['qvel_local'])] = kf['qvel_local']
            mujoco.mj_forward(model_mj, data_mj)
            teacher._target_dof = kf['target_dof'].copy()
            if teacher.base_height < FALL_HEIGHT:
                return dict(success=False, failure_tag='fall', steps=0,
                            heading_err=math.pi, landmark_passed=False, video_path=None)
        else:
            # WBC ONNX settle (fallback — only reached if stand_keyframe.npz missing)
            for _ in range(SETTLE_STEPS):
                teacher.step(vel_cmd=(0.0, 0.0, 0.0))
                if teacher.base_height < FALL_HEIGHT:
                    return dict(success=False, failure_tag='fall', steps=0,
                                heading_err=math.pi, landmark_passed=False, video_path=None)

        # Renderer
        renderer = ArenaRenderer(model_mj)
        tp_cam   = renderer.make_tp_cam()
        frames_ego, frames_tp = [], []

        # Expert FSM (privileged)
        expert = ManeuverExpert(scene_cfg)
        expert.reset()

        # Phase tracker
        phase_tracker = _GaitPhaseTracker()

        # Proprio history (62-d)
        prop_dim  = PROPRIO_DIM_MANEUVER  # 62
        prop_hist = collections.deque(
            [np.zeros(prop_dim, dtype=np.float32)] * PROPRIO_K,
            maxlen=PROPRIO_K,
        )

        # Zero lang embedding
        lang_t = torch.zeros(1, 2048, device=self._device)

        prev_action = teacher._target_dof.copy()
        ACTION_SCALE = 0.25

        fell    = False
        steps   = 0
        success = False
        failure_tag = 'didnt-reach'
        heading_err_final = math.pi
        lm_passed = False

        landmark_xy = np.array(scene_cfg.get('landmark_xy', [0.0, 0.0]))
        target_heading = float(scene_cfg.get('target_heading', math.pi / 2))
        pass_margin = float(scene_cfg.get('pass_margin', 0.6))

        HEADING_THR = math.radians(25.0)

        for step in range(maxsteps):
            height = float(data_mj.qpos[2])
            if height < FALL_HEIGHT:
                fell = True
                failure_tag = 'fall'
                break

            robot_xy  = data_mj.qpos[0:2].copy()
            robot_yaw = _yaw_of(data_mj.qpos[3:7])

            # Expert FSM step (privileged)
            vel_cmd, priv = expert.step(robot_xy, robot_yaw)
            subgoal_idx  = priv['subgoal_index']
            heading_err  = priv['heading_err']
            lm_passed    = priv['landmark_passed']
            heading_err_final = heading_err

            # Hybrid vel: inject expert vel only during TURN_PHASE
            _in_turn = (subgoal_idx == State.TURN_PHASE)

            # Build 62-d proprio
            from code.inferencer import _build_proprio
            prop_base = _build_proprio(data_mj, prev_action)  # 55-d
            q_lb      = data_mj.qpos[7:22].copy()
            ph        = phase_tracker.update(q_lb)              # 2-d
            mv_feat   = _build_maneuver_features(priv)          # 5-d
            prop_now  = np.concatenate([prop_base, ph, mv_feat]).astype(np.float32)  # 62-d
            prop_hist.append(prop_now)

            prop_arr = np.stack(list(prop_hist), axis=0)
            prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(self._device)

            # Goal: egocentric vector to landmark
            delta   = landmark_xy - robot_xy
            dist_lm = float(np.linalg.norm(delta))
            cx_l    = math.cos(robot_yaw) * delta[0] + math.sin(robot_yaw) * delta[1]
            cy_l    = -math.sin(robot_yaw) * delta[0] + math.cos(robot_yaw) * delta[1]
            ye_lm   = math.atan2(cy_l, cx_l)
            goal_vec = np.array([dist_lm, math.cos(ye_lm), math.sin(ye_lm)], dtype=np.float32)
            goal_t  = torch.from_numpy(goal_vec).unsqueeze(0).to(self._device)

            # Vel injection (hybrid: only during TURN_PHASE)
            if _in_turn:
                vel_t = torch.tensor([[float(vel_cmd[0]), float(vel_cmd[1]), float(vel_cmd[2])]],
                                     dtype=torch.float32, device=self._device)
            else:
                vel_t = None

            img_t = torch.zeros(1, 3, 128, 128, device=self._device)

            with torch.no_grad():
                out = self._model(
                    ego_rgb   = img_t,
                    lang_emb  = lang_t,
                    proprio_h = prop_t,
                    gt_goal   = goal_t,
                    gt_vel    = vel_t,
                )

            raw_action = out['action'].cpu().numpy().squeeze(0)[0]  # (15,)

            # De-normalize (residual action)
            if self._action_stats is not None:
                _as = self._action_stats
                target_dof = _as['default_angles'] + raw_action * _as['std'] + _as['mean']
            else:
                target_dof = raw_action

            # PD + physics (reapply PD torques every substep, matching eval_maneuver.py)
            from code.inferencer import _apply_student_pd
            for _ in range(CONTROL_DECIMATION):
                _apply_student_pd(data_mj, target_dof, nj)
                mujoco.mj_step(model_mj, data_mj)

            prev_action = target_dof.copy()
            steps = step + 1

            # Render frames if requested
            if render_video:
                yaw_now = _yaw_of(data_mj.qpos[3:7])
                rgb, _, _ = renderer.render_ego(data_mj, yaw_now, render_depth=False)
                frames_ego.append(rgb.copy())
                renderer.update_tp_cam(tp_cam, data_mj)
                frames_tp.append(renderer.render_tp(data_mj, tp_cam).copy())

            # Progress callback
            if progress_cb and step % 50 == 0:
                pct = int(step / maxsteps * 100)
                progress_cb({
                    "step": step, "steps": maxsteps,
                    "pct": pct,
                    "heading_err_deg": math.degrees(abs(heading_err_final)),
                    "lm_passed": lm_passed,
                    "phase": ["STRAIGHT", "TURN", "STRAIGHT2"][subgoal_idx],
                })

            # Check success: landmark passed + heading aligned
            if lm_passed and abs(heading_err_final) < HEADING_THR:
                success = True
                failure_tag = 'success'
                break

        renderer.close()

        if not fell and not success:
            failure_tag = 'didnt-reach'

        # Write video
        out_vid = None
        if render_video and video_path and frames_ego:
            from code.inferencer import _write_video
            _write_video(frames_ego, frames_tp, video_path)
            out_vid = video_path

        return dict(
            success=success,
            failure_tag=failure_tag,
            steps=steps,
            heading_err_deg=math.degrees(abs(heading_err_final)),
            landmark_passed=lm_passed,
            video_path=out_vid,
        )


# ---------------------------------------------------------------------------
# Progress event bus
# ---------------------------------------------------------------------------
class EventBus:
    """Simple thread-safe event bus for UI updates."""

    def __init__(self):
        self._lock   = threading.Lock()
        self._events = collections.deque(maxlen=200)
        self._state  = {}

    def emit(self, event: dict):
        with self._lock:
            event['_ts'] = time.time()
            self._events.append(event)
            self._state.update(event)

    def get_events(self, since_ts: float = 0.0) -> List[dict]:
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
        scene_manager: SceneManager,
        bus: EventBus,
        device: str = "cpu",
        render_video: bool = True,
        out_dir: str = str(DEMO_OUT_DIR),
        maxsteps_goto: int = MAXSTEPS_GOTO,
        maxsteps_maneuver: int = MAXSTEPS_MANEUVER,
    ):
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

    def _get_goto_inferencer(self):
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

    def _get_maneuver_inferencer(self):
        if self._maneuver_inferencer is None:
            self._maneuver_inferencer = ManeuverInferencer(
                checkpoint_path=str(MANEUVER_CKPT),
                device=self.device,
            )
        return self._maneuver_inferencer

    def execute(self, goals: List[SubGoal]) -> List[Dict[str, Any]]:
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

    def _run_goto(self, goal: SubGoal, ep_id: int, gi: int) -> Dict[str, Any]:
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

        def _progress_cb(info):
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

    def _run_maneuver(self, goal: SubGoal, ep_id: int, gi: int) -> Dict[str, Any]:
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

        def _progress_cb(info):
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

    def _run_search_stub(self, goal: SubGoal, ep_id: int, gi: int) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Web UI (Flask MJPEG streaming + REST API)
# ---------------------------------------------------------------------------
def _start_web_ui(bus: EventBus, executor: Executor, planner: Planner,
                   scene_manager: SceneManager, port: int = WEB_PORT):
    """Start Flask web server in background thread."""
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

    def _run_execute(instruction: str):
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
    def index():
        return render_template_string(HTML)

    @app.route("/scene_info")
    def scene_info():
        return jsonify({"scene_desc": scene_manager.describe_scene()})

    @app.route("/new_scene", methods=["POST"])
    def new_scene():
        scene_manager.new_scene()
        return jsonify({"scene_desc": scene_manager.describe_scene()})

    @app.route("/execute", methods=["POST"])
    def execute():
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
    def events():
        since = float(request.args.get("since", 0))
        evts  = bus.get_events(since_ts=since)
        return jsonify({"events": evts})

    @app.route("/stream")
    def stream():
        """MJPEG stream (static placeholder — real video saved as MP4)."""
        def gen():
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

    def _run_flask():
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)

    t = threading.Thread(target=_run_flask, daemon=True)
    t.start()
    print(f"[demo] Web UI started at http://localhost:{port}", flush=True)
    return t


# ---------------------------------------------------------------------------
# Terminal REPL
# ---------------------------------------------------------------------------
def _terminal_repl(
    scene_manager: SceneManager,
    planner: Planner,
    executor: Executor,
    bus: EventBus,
    out_dir: str,
):
    """Interactive terminal REPL."""
    print("\n" + "=" * 60, flush=True)
    print("G1Nav Interactive Demo REPL", flush=True)
    print("Commands: <instruction> | 'new' | 'scene' | 'quit'", flush=True)
    print("=" * 60, flush=True)

    # Show initial scene
    scene_manager.new_scene()
    print(f"\n{scene_manager.describe_scene()}\n", flush=True)

    pending_clarify = None   # current clarification question (if any)
    pending_instr   = None   # original instruction that triggered clarification

    while True:
        try:
            if pending_clarify:
                prompt = "clarify> "
            else:
                prompt = "demo> "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[demo] Goodbye!", flush=True)
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("[demo] Goodbye!", flush=True)
            break

        if user_input.lower() in ("new", "reset"):
            scene_manager.new_scene()
            pending_clarify = None
            pending_instr   = None
            print(f"\n{scene_manager.describe_scene()}\n", flush=True)
            continue

        if user_input.lower() in ("scene", "objects", "describe"):
            print(f"\n{scene_manager.describe_scene()}\n", flush=True)
            continue

        if user_input.lower() in ("cancel", "abort"):
            pending_clarify = None
            pending_instr   = None
            print("\nBot: Cancelled. Type a new instruction.\n", flush=True)
            continue

        if user_input.lower() in ("help", "?"):
            print("\nBot: Instructions I understand:", flush=True)
            print("  goto:     'go to the red ball'  |  'navigate to the orange cone'", flush=True)
            print("  maneuver: 'turn left after the blue cube'  |  'pass the red cylinder then turn right'", flush=True)
            print("  search:   'find the red ball'  |  'look for the orange cube'", flush=True)
            print("  multi:    'go to the red ball then find the blue cube'", flush=True)
            print("  cmds:     'new' (new scene)  |  'scene' (show objects)  |  'quit'", flush=True)
            print(flush=True)
            continue

        # If awaiting clarification, use planner.parse_clarification()
        if pending_clarify:
            original  = pending_instr or user_input
            goals, clarify = planner.parse_clarification(original, user_input)
            if clarify:
                print(f"\nBot: {clarify}\n", flush=True)
                pending_clarify = clarify
                pending_instr   = original
                continue
            pending_clarify = None
            pending_instr   = None
        else:
            # Parse fresh instruction
            goals, clarify = planner.parse(user_input)

        if clarify:
            print(f"\nBot: {clarify}\n", flush=True)
            pending_clarify = clarify
            pending_instr   = user_input
            continue

        if not goals:
            print("\nBot: I didn't understand that. Type 'help' for examples.\n", flush=True)
            continue

        # Show plan
        print(f"\nBot: Plan ({len(goals)} step{'s' if len(goals) > 1 else ''}):", flush=True)
        for i, g in enumerate(goals):
            print(f"  [{i+1}] {g}", flush=True)
        print(flush=True)

        # Execute
        print("Executing...", flush=True)
        t0 = time.time()
        results = executor.execute(goals)
        dt = time.time() - t0

        # Summary
        print(f"\n--- Episode Summary ({dt:.1f}s) ---", flush=True)
        for i, (g, r) in enumerate(zip(goals, results)):
            status = "SUCCESS" if r.get("success") else f"FAILED ({r.get('failure_tag', '?')})"
            vid    = r.get("video_path", None)
            vid_msg = f" → video: {vid}" if vid else ""
            print(f"  [{i+1}] {g}: {status}{vid_msg}", flush=True)

        n_success = sum(1 for r in results if r.get("success"))
        print(f"\nTotal: {n_success}/{len(results)} succeeded", flush=True)

        # New scene prompt
        print("\nGenerating new scene for next episode...", flush=True)
        scene_manager.new_scene()
        print(f"\n{scene_manager.describe_scene()}\n", flush=True)


# ---------------------------------------------------------------------------
# Smoke test (canned instructions)
# ---------------------------------------------------------------------------
def _smoke_test(out_dir: str, device: str, maxsteps_goto: int, maxsteps_maneuver: int,
                render_video: bool = True):
    """
    Run 4 canned instructions end-to-end headless (S10 polish):
      1. goto (demo-distance, 4-9m)
      2. maneuver (turn after landmark)
      3. search (out-of-FOV find)
      4. multi-goal compound (goto then search)
    Saves videos and prints summary.
    """
    print("\n" + "=" * 60, flush=True)
    print("G1Nav Demo — SMOKE TEST (4 canned instructions incl. multi-goal)", flush=True)
    print("=" * 60 + "\n", flush=True)

    os.makedirs(out_dir, exist_ok=True)

    bus           = EventBus()
    scene_manager = SceneManager(difficulty="demo", seed_offset=42)
    planner       = Planner(scene_manager)
    executor      = Executor(
        scene_manager=scene_manager,
        bus=bus,
        device=device,
        render_video=render_video,
        out_dir=out_dir,
        maxsteps_goto=maxsteps_goto,
        maxsteps_maneuver=maxsteps_maneuver,
    )

    from code.maneuver_scene import sample_maneuver_scene as _sample_maneuver
    from code.maneuver_scene import derive_rng as _derive_maneuver_rng
    from code.scene import sample_scene as _sample_scene, derive_rng as _derive_rng
    from code.eval_search import sample_search_scene

    # Build test cases
    # Case 1: demo-distance goto
    scene_manager.difficulty = "demo"
    scene_manager.new_scene()
    sc_goto = scene_manager.scene_cfg
    tgt_goto = sc_goto["objects"][sc_goto["target_index"]]
    instr_goto = f"go to the {tgt_goto['color_name']} {tgt_goto['shape_name']}"

    # Case 2: maneuver
    _mrng = _derive_maneuver_rng(999, 0)
    sc_maneuver = _sample_maneuver(_mrng)
    lm = sc_maneuver['objects'][sc_maneuver['landmark_index']]
    instr_maneuver = sc_maneuver['instruction']

    # Case 3: search (out-of-FOV) — use easy difficulty scene + search skill
    _srng = np.random.default_rng(np.random.SeedSequence([999, 0]))
    sc_search = sample_search_scene(_srng, 0)
    tgt_search = sc_search["objects"][sc_search["target_index"]]
    instr_search = f"find the {tgt_search['color_name']} {tgt_search['shape_name']}"

    # Case 4: multi-goal — "go to X then find Y" (uses two separate skills in one episode)
    # Use the goto scene for the first goal; pick a different object for search
    sc_multi = dict(sc_goto)
    objs_multi = sc_goto["objects"]
    # First goal: goto the target
    tgt1 = objs_multi[sc_goto["target_index"]]
    # Second goal: search for another object (non-target)
    tgt2_idx = 1 if sc_goto["target_index"] != 1 else 2
    tgt2_idx = min(tgt2_idx, len(objs_multi) - 1)
    tgt2 = objs_multi[tgt2_idx]
    instr_multi = (f"go to the {tgt1['color_name']} {tgt1['shape_name']} "
                   f"then find the {tgt2['color_name']} {tgt2['shape_name']}")

    test_cases = [
        {"label": "goto_demo_long",   "scene": sc_goto,     "instr": instr_goto,     "difficulty": "demo"},
        {"label": "maneuver",         "scene": sc_maneuver, "instr": instr_maneuver, "difficulty": "demo"},
        {"label": "search_outofFOV",  "scene": sc_search,   "instr": instr_search,   "difficulty": "search"},
        {"label": "multi_goal",       "scene": sc_multi,    "instr": instr_multi,    "difficulty": "demo"},
    ]

    summary = []
    for i, tc in enumerate(test_cases):
        print(f"\n--- Test {i+1}/{len(test_cases)}: {tc['label']} ---", flush=True)
        print(f"Instruction: '{tc['instr']}'", flush=True)

        # Update scene manager to use the prepared scene
        scene_manager._scene_cfg = tc["scene"]
        scene_manager.difficulty = tc["difficulty"]

        goals, clarify = planner.parse(tc["instr"])
        if clarify:
            print(f"Planner needs clarification: {clarify}", flush=True)
            # Try to resolve with scene's target info
            tgt = tc["scene"]["objects"][tc["scene"].get("target_index", 0)]
            fallback_instr = f"go to the {tgt['color_name']} {tgt['shape_name']}"
            goals, clarify = planner.parse(fallback_instr)

        if not goals:
            print(f"No goals parsed — SKIP", flush=True)
            summary.append({"label": tc["label"], "success": False, "reason": "parse_fail",
                             "instruction": tc["instr"]})
            continue

        print(f"Plan ({len(goals)} step{'s' if len(goals) > 1 else ''}): {[str(g) for g in goals]}", flush=True)

        t0 = time.time()
        try:
            results = executor.execute(goals)
        except Exception as e:
            print(f"Executor error: {e}", flush=True)
            results = [{"success": False, "failure_tag": "error", "steps": 0}] * len(goals)
        dt = time.time() - t0

        for g, r in zip(goals, results):
            vid = r.get("video_path")
            print(
                f"  {g}: {'SUCCESS' if r.get('success') else 'FAILED'}  "
                f"steps={r.get('steps', 0)}  time={dt:.1f}s"
                + (f"  video={vid}" if vid else ""),
                flush=True,
            )
            summary.append({
                "label": tc["label"],
                "instruction": tc["instr"],
                "skill": g.skill,
                "success": r.get("success", False),
                "failure_tag": r.get("failure_tag", ""),
                "steps": r.get("steps", 0),
                "wall_time_s": dt,
                "video_path": vid,
            })

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("SMOKE TEST SUMMARY", flush=True)
    print("=" * 60, flush=True)
    n_ok = sum(1 for s in summary if s.get("success"))
    print(f"Success: {n_ok}/{len(summary)}", flush=True)
    for s in summary:
        status = "OK" if s.get("success") else f"FAIL ({s.get('failure_tag', '?')})"
        print(f"  {s['label']:25s} [{s.get('skill','?'):8s}] {status:30s}  video={s.get('video_path')}", flush=True)

    # Save summary JSON
    summary_path = os.path.join(out_dir, "smoke_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}", flush=True)

    return summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="G1Nav Interactive Demo")
    parser.add_argument("--web",    action="store_true", help="Start web UI on port 5000")
    parser.add_argument("--port",   type=int, default=WEB_PORT)
    parser.add_argument("--smoke",  action="store_true", help="Run canned smoke test and exit")
    parser.add_argument("--out",    default=str(DEMO_OUT_DIR), help="Output dir for videos")
    parser.add_argument("--device", default="cuda" if _check_cuda() else "cpu")
    parser.add_argument("--difficulty", default="demo", choices=["easy", "demo"])
    # H1: default changed to 'demo' — showcases 4-9m long walks with V2/V3 grounding
    parser.add_argument("--maxsteps-goto", type=int, default=MAXSTEPS_GOTO)  # H1: default=1400 for demo
    parser.add_argument("--maxsteps-maneuver", type=int, default=MAXSTEPS_MANEUVER)
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering (faster)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Smoke test mode
    if args.smoke:
        _smoke_test(
            out_dir=args.out,
            device=args.device,
            maxsteps_goto=args.maxsteps_goto,
            maxsteps_maneuver=args.maxsteps_maneuver,
            render_video=(not args.no_render),
        )
        return

    # Normal REPL / Web UI mode
    bus           = EventBus()
    scene_manager = SceneManager(difficulty=args.difficulty, seed_offset=0)
    planner       = Planner(scene_manager)
    executor      = Executor(
        scene_manager=scene_manager,
        bus=bus,
        device=args.device,
        render_video=(not args.no_render),
        out_dir=args.out,
        maxsteps_goto=args.maxsteps_goto,
        maxsteps_maneuver=args.maxsteps_maneuver,
    )

    # Pre-load initial scene
    scene_manager.new_scene()

    if args.web:
        _start_web_ui(
            bus=bus,
            executor=executor,
            planner=planner,
            scene_manager=scene_manager,
            port=args.port,
        )
        print(f"[demo] Web UI running at http://localhost:{args.port}", flush=True)
        print("[demo] Press Ctrl-C to quit", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[demo] Shutting down.", flush=True)
    else:
        _terminal_repl(
            scene_manager=scene_manager,
            planner=planner,
            executor=executor,
            bus=bus,
            out_dir=args.out,
        )


if __name__ == "__main__":
    main()
