"""
code/datagen/gen_det_common.py — Shared constants + tiny helpers for the NX-6 detector dataset.

Role: split out of gen_det_dataset.py (RF-1) — constants and one-liners
shared by det_labels.py, det_capture.py, det_scene.py, and gen_det_dataset.py.
"""

from __future__ import annotations

import os
import re

import mujoco

from code.arena import COLORS, SHAPES

# ---------------------------------------------------------------------------
# Class/color tables
# ---------------------------------------------------------------------------
COLOR_NAMES: list[str] = [c for c, _ in COLORS]                 # 7 colors
SHAPE_NAMES: list[str] = [s for s, _ in SHAPES]                 # 4 shapes: ball,cube,cylinder,cone
SIZE_M: dict[str, float] = dict(SHAPES)                         # nominal diameter/edge per shape
COLOR2I: dict[str, int] = {c: i for i, c in enumerate(COLOR_NAMES)}
SHAPE2I: dict[str, int] = {s: i for i, s in enumerate(SHAPE_NAMES)}

# ---------------------------------------------------------------------------
# Simulation / capture constants
# ---------------------------------------------------------------------------
FALL_HEIGHT: float      = 0.50
SETTLE_STEPS: int       = 80
MIN_PIXELS: int         = 6          # minimum mask pixels to keep a detection
GEOM_RE: re.Pattern[str] = re.compile(r"^obj_(\d+)(?:_tip)?$")

CAM_SWITCH_DIST_M: float = 1.8        # proximity below this true distance, else grounding
DUAL_RENDER_PROB: float  = 0.20      # occasionally render BOTH cams at the same pose

MAXSTEPS_TRAJ: dict[str, int] = {"easy": 260, "demo": 900, "search": 550}
N_TRAJ_TARGET: int  = 12             # aim for ~this many trajectory samples per scene
N_TELEPORT_FOCUS: int  = 10
N_TELEPORT_RANDOM: int = 6
MAX_GAIT_SNAPSHOTS: int = 8

DIFF_FOR_SEARCH: str = "search"     # pseudo-difficulty label for search-style scenes


def _env_note() -> None:
    """Prints the active MUJOCO_GL backend and mujoco version to stdout."""
    print(f"[gen_det_dataset] MUJOCO_GL={os.environ.get('MUJOCO_GL')}  "
          f"mujoco={mujoco.__version__}", flush=True)


def pick_cam(dist_m: float) -> str:
    """Returns "proximity" if `dist_m` is within switch range, else "grounding"."""
    return "proximity" if dist_m <= CAM_SWITCH_DIST_M else "grounding"
