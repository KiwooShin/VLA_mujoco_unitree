"""Shared constants + VF-1 render-toggle env-var flags for the fancy demo app
(code/fancy_demo.py, RF-1 split).

Owns:
  - sys.path / MUJOCO_GL bootstrap (module state coherence: runs exactly once,
    from this one place — every other fancy/* module imports from here).
  - Checkpoint/output-dir paths, step caps, camera resolutions, the BEV
    follow-cam pose, overlay colors, the reliable-color palette, and the
    STATE_* state-machine string constants.
  - VF-1's per-feature render toggles (FANCY_PLAIN / FANCY_<FEAT> env vars),
    read once at import time — see the toggle rationale comment below.

FIRST_SCENE_SEED (the curated first-launch scene seed) is owned by
code/apps/fancy/sampling.py instead of here, since it's fundamentally a
scene-sampling concern (see that module's docstring).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent.parent
sys.path.insert(0, str(_REPO))

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOTO_CKPT_DEFAULT   = str(_REPO / "checkpoint" / "goto_best.pt")
KEYFRAME_PATH       = str(_REPO / "checkpoint" / "stand_keyframe.npz")
FANCY_OUT_DIR       = str(_REPO / "eval" / "fancy_demo")
WEB_PORT            = 5001     # different port from demo.py (5000) to avoid conflict
MAXSTEPS_FANCY      = 2000     # hard cap per episode (NX-1: bumped from 1400 to match
                                # code/eval_search.py's MAXSTEPS_SEARCH -- the
                                # bidirectional bounded scan, code/scan_sched.py, can
                                # spend more total steps than the old fixed-CCW scan
                                # before spotting an unfavorable-side target; see
                                # docs/nx1_scan.md)
BEV_W, BEV_H       = 640, 480  # BEV camera resolution
EGO_W, EGO_H       = 320, 240  # ego camera resolution (native)
STREAM_W            = BEV_W + EGO_W  # side-by-side width

# BEV follow-camera parameters (diagonal azimuth, steep-diagonal elevation)
#
# VF-3 fix (docs/vf3_bev_fixes.md, user feedback #2 "move camera a little
# further ... currently only part of field is visible"): BEV_DISTANCE=6.0 /
# BEV_ELEVATION=-40.0 only ever showed a tight ~14x14m patch of ground
# straight-line-asymmetric around the robot (e.g. robot at arena origin in
# an ARENA_HALF_LONG=8.0 arena: visible ground x/y in [-11.0, 3.4] -- the
# opposite walls, and often the target itself 4-9m away, were simply never
# in frame). Re-derived analytically (ground-plane ray intersection of the
# camera frustum's corners, `framing_calc.py`-style) and re-verified with
# real renders: BEV_DISTANCE=28.0 / BEV_ELEVATION=-67.0 kept the FULL arena
# floor (all 4 walls) in frame -- but a near-top-down view at this distance
# reduced every object to a few-pixel dot (VF-5 user feedback #1: "move
# camera a little bit closer to the ground. I cannot distinguish objects in
# current setting well").
#
# VF-5 fix (docs/vf5_cam_objects.md): pulled the camera in and shallowed the
# elevation so objects render as recognizable side-profile shapes (cone vs.
# cylinder vs. cube vs. ball all legible, incl. at 480px gallery scale)
# while keeping the SAME analytic-footprint methodology as VF-3 to bound
# the crop risk. Swept (distance, elevation) in {(14,-38), (16,-42),
# (18,-45), (20,-50)} plus fine interpolation, checking BOTH (a) rendered
# object legibility and (b) world_to_bev_pixel()'s ground-footprint bounds
# for every (angle, distance) an object can actually spawn at in this
# file's samplers (distractors up to 5m anywhere, targets up to 7m outside
# the robot's initial FOV). (14,-38) is the sharpest but crops real target
# spawns (bearing ~45-90°, dist 6-7m lands 5-63px below the frame -- a
# ~5% slice of the target-spawn distribution, confirmed both analytically
# and by rendering). (16,-42) shrinks that to a 3-point/120 sliver right at
# the 7.0m/30-60° corner (up to 18px over). (18,-45)/(20,-50) have zero
# out-of-frame points anywhere in the reachable envelope but are
# noticeably less zoomed. BEV_DISTANCE=17.0 / BEV_ELEVATION=-43.5 (almost
# exactly between the (16,-42) and (18,-45) candidates) is the sweet spot:
# only a single, mathematically-excluded probe point (the target sampler
# requires bearing strictly > the FOV half-angle, never exactly ==) is
# ever out of frame, by 0.1px -- see docs/vf5_cam_objects.md for the full
# sweep table and comparison renders. Azimuth (diagonal viewpoint) unchanged.
BEV_DISTANCE   = 17.0    # metres from robot
BEV_ELEVATION  = -43.5   # degrees (negative = looking down)
BEV_AZIMUTH    = 225.0   # degrees diagonal (SW view → robot in frame, facing right)
BEV_LOOKAT_Z   = 0.3     # lookat height (ground-level scene)

# Overlay colors (BGR for cv2)
COLOR_PATH_TRAIL: tuple[int, int, int]   = (0,   220, 100)   # green path line
COLOR_TARGET_RING: tuple[int, int, int]  = (0,   80,  255)   # bright orange ring
COLOR_FOV_CONE: tuple[int, int, int]     = (255, 255,  80)   # yellow FOV wedge
COLOR_BANNER_BG: tuple[int, int, int]    = (30,   30,  30)
COLOR_STATE_SEARCH: tuple[int, int, int] = (0,   200, 255)   # cyan text — SEARCHING
COLOR_STATE_LOCATE: tuple[int, int, int] = (50,  255,  50)   # green — LOCATED
COLOR_STATE_MOVE: tuple[int, int, int]   = (255, 165,   0)   # orange — MOVING
COLOR_STATE_REACH: tuple[int, int, int]  = (255,  80,  80)   # red-pink — REACHED

# Reliable color palette (avoid cyan/blue — wall HSV collision)
# See docs/grounding_dist.md: red/orange/yellow/purple = 87-100% detection at demo distances
RELIABLE_COLORS: list[str] = ["red", "orange", "yellow", "purple"]
RELIABLE_SHAPES: list[str] = ["ball", "cube", "cylinder", "cone"]

# State machine states
STATE_IDLE      = "IDLE"
STATE_SEARCHING = "SEARCHING"
STATE_LOCATED   = "LOCATED"
STATE_MOVING    = "MOVING"
STATE_REACHED   = "REACHED"
STATE_FAILED    = "FAILED"

# FD2: Long-distance scene parameters (arena must be large enough)
ARENA_HALF_LONG  = 8.0   # 8m half-size → room for 7m targets
DIST_MIN_LONG    = 4.0   # minimum target distance (impressive walk)
DIST_MAX_LONG    = 7.0   # maximum target distance (reliable detection ceiling)


# ---------------------------------------------------------------------------
# VF-1 (docs/vf1_showpiece.md): render-side-only visual upgrade toggles.
#
# Every overlay gated below reads state that ALREADY EXISTS in run_fancy_rollout
# (cached_goal_vec, _avoid_bias_wz, path_trail, current_state, ...) or a pure-read
# cache populated alongside a computation the code already does (e.g.
# code.grounding's GROUND_NET confidence-heatmap cache, populated inside the same
# forward pass ground()/_ground_net() already runs every grounding cycle -- zero
# extra inference). None of it writes back into anything a control-flow decision
# reads (goal vectors, avoid bias, scan schedule, physics, RNG) -- these are
# strictly VISUAL/OVERLAY additions on top of a behavior-frozen system.
#
# Individually toggleable (one env var per feature, default ON) so a recorder
# can drop a single overlay that misbehaves without losing the rest.
# FANCY_PLAIN=1 hard-disables ALL of them at once and restores the pre-VF1
# rendering exactly (same functions / same code paths / same 960-ish x 480
# canvas) — the single "something broke, fall back" switch.
# ---------------------------------------------------------------------------
def _fancy_env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip() == "1"


FANCY_PLAIN = _fancy_env_flag("FANCY_PLAIN", "0")


def _fancy_feat(name: str) -> bool:
    """One VF-1 overlay toggle: default ON, forced OFF (regardless of its own
    env var) whenever FANCY_PLAIN=1."""
    if FANCY_PLAIN:
        return False
    return _fancy_env_flag(f"FANCY_{name}", "1")


FEAT_HEATMAP    = _fancy_feat("HEATMAP")        # item 1: detector confidence overlay
FEAT_AVOID_VIZ  = _fancy_feat("AVOID_VIZ")      # item 2: avoidance vector + corridor tint
FEAT_HUD        = _fancy_feat("HUD")            # item 3: bottom HUD bar
FEAT_TRAIL      = _fancy_feat("TRAIL_GRADIENT") # item 4: gradient trail + dashed goal line
FEAT_TITLECARD  = _fancy_feat("TITLECARD")      # item 5: title card + outro stats card
FEAT_HIRES      = _fancy_feat("HIRES")          # item 6: ~1600x600 canvas

# Item 6 resolution: both panels displayed at 800x600. Native MuJoCo render sizes
# (BEV_W/H, EGO_W/H below) are COMPLETELY UNCHANGED -- this is a cheap cv2.resize
# on the already-rendered frame, never a higher-resolution render, so per-step
# wall time is unaffected by this toggle (measured, see docs/vf1_showpiece.md).
PANEL_DISPLAY_W = 800
PANEL_DISPLAY_H = 600
HUD_BAR_H       = 46    # extra strip appended below the two panels when FEAT_HUD

# Detector heatmap overlay blend strength (gate check: keep in ~0.35-0.45 so the
# scene stays legible underneath the color map).
HEATMAP_ALPHA = 0.40

# 5-stage skill breadcrumb shown in the HUD bar (item 3).
SKILL_STAGES: list[str] = ["SCAN", "LOCK", "WALK", "HANDOFF", "REACH"]
