"""
fancy_demo.py — FANCY Interactive Demo for G1Nav  (FD2 — enhanced)

Polished visualization + interaction layer on top of demo.py's planner/executor/inferencer.
Reuses: Planner, Executor, SceneManager, Inferencer, _run_search_rollout from demo.py / eval_search.py.

View: ego camera (RGB robot view) | ELEVATED 3D DIAGONAL BEV FOLLOW-CAM side-by-side.

BEV overlays (drawn with cv2 by projecting world→BEV camera via MuJoCo camera matrices):
  (a) Robot PATH TRAIL — polyline of past base positions (continues across sub-goals)
  (b) TARGET HIGHLIGHT — ring/marker on CURRENT sub-goal target object
  (c) Robot ego-camera FOV cone/wedge on the ground
  (d) STATUS BANNER — prompt text + state + sub-goal progress "goal N/M" + live distance

Interaction:
  - Flask WEB UI: type prompt in browser → watch ego|BEV MJPEG stream + status panel
  - Headless smoke: 5-6 episodes saved as ego|BEV MP4s (long-distance + multi-goal)

Scenes (FD2 — long-distance bias):
  - Target OUTSIDE initial ego FOV (search always demonstrated)
  - Distance 4–7m (medium-long search→walk — impressive)
  - Reliable colors only: red/orange/yellow/purple (avoids HSV wall collision at 4-9m)
  - Arena 8m half-size (room for 7m targets)
  - Multi-goal support: "go to the red ball then find the yellow cube" → sequential sub-goals

Usage:
    # Headless smoke test — 5-6 long episodes + multi-goal:
    MUJOCO_GL=egl python code/fancy_demo.py --smoke --out eval/fancy_demo --device cuda

    # Showcase reel (6 episodes, auto concat):
    MUJOCO_GL=egl python code/fancy_demo.py --smoke --out eval/fancy_demo --device cuda --n-smoke 6

    # Web UI mode:
    MUJOCO_GL=egl python code/fancy_demo.py --web --out eval/fancy_demo --device cuda

    # Quick smoke (no render, validation only):
    MUJOCO_GL=egl python code/fancy_demo.py --smoke --out eval/fancy_demo --no-render

Anti-hang:
  - Single ArenaRenderer reused per episode (EGL context exhaustion prevention)
  - Hard MAXSTEPS caps (1400 per sub-goal)
  - Flush prints throughout
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

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

# FS-1 (2026-07-10): curated deterministic FIRST scene for --web/terminal launch.
# The old always-[1234,0] first draw reproducibly landed on `red cone,
# dist=4.35m, bearing=77.6°` -- the documented walking-instability residual
# (robot SPOTTED the target at step 20 then walked steadily *away*,
# dist 4.24m->9.4m; docs/vr1_rehearsal.md friction #3), i.e. a recruiter's
# very first impression was a known-bad draw. Subsequent new_scene() calls
# were never the problem (they auto-resample after every rollout), so only
# this one fixed draw needed curating.
#
# FS-2 (2026-07-11, docs/vf5_cam_objects.md's flag / docs/fs2_first_scene_
# resample.md): VF-5 rewrote sample_fancy_scene_long's placement loop to
# guarantee >=7 objects per scene (was 3). Same FIRST_SCENE_SEED=1259 draw
# under the NEW sampler produces a DIFFERENT 7-object scene (yellow cube,
# dist=4.95m, bearing=54.6° -- still fine geometrically, but never
# re-verified end-to-end against the new sampler/camera), so the seed was
# re-picked from scratch under the actual current sampler rather than
# trusting the old pick to still apply.
# Picked by: geometry pre-filter (color in RELIABLE_COLORS, dist 4-7m,
# bearing in [60,110] AND positive-signed -- matches BidirectionalScan-
# Schedule's _LEG_SIGNS=(+1,-1,-1,+1) "positive leg0 first" so no leg0->leg1
# reversal is needed, sidestepping the rotation-order-instability class
# documented in docs/gen1_multiseed.md §3.1 / docs/nx12_turn_dwell.md; no
# same-color/SAME-shape distractor -- docs/gen1_multiseed.md §3.3's
# false-lock risk (now also guaranteed by construction for every VF-5
# scene, via _select_fancy_distractor_combos' pairwise-distinct combos);
# no distractor within 0.5m of the straight robot->target path; target
# shape != cone -- docs/nx16_cone_stall.md's cone-specific confidence-decay
# risk), then verified by actually running the full rollout headlessly 2x:
#   seed=3461 -> target=yellow cube, dist=4.97m, bearing=84.9° (out-of-FOV),
#   7 objects incl. a same-color/diff-shape yellow ball + yellow cylinder,
#   nearest distractor 2.15m off the straight path.
#   Both runs: success=True, fell=False, steps=714/711, final_dist=0.468m/
#   0.468m, wall~308-309s each (small step-count delta is the documented
#   EGL/physics jitter, docs/gen1_multiseed.md -- not a concern). See
#   docs/vf5_cam_objects.md (first-scene re-curation notes).
FIRST_SCENE_SEED = 3461


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


# ---------------------------------------------------------------------------
# World→BEV projection helper
# ---------------------------------------------------------------------------

def world_to_bev_pixel(
    world_pts: np.ndarray,   # (N, 3) world XYZ
    bev_cam: "mujoco.MjvCamera",
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    w: int = BEV_W,
    h: int = BEV_H,
    fovy_deg: float = 45.0,
) -> np.ndarray:
    """Project world XYZ points into BEV pixel coordinates.

    Uses MuJoCo's camera view matrix + a pinhole projection. Clips
    out-of-frame points but does not filter them.

    Args:
        world_pts: (N, 3) world XYZ points to project (or (3,) for a single
            point, promoted to (1, 3)).
        bev_cam: MuJoCo free camera (lookat/azimuth/elevation/distance) used
            for the BEV follow-cam view.
        model: MuJoCo model (unused in this function; kept for interface
            parity with the other render helpers).
        data: MuJoCo data (unused in this function; kept for interface
            parity with the other render helpers).
        w: Output image width in pixels.
        h: Output image height in pixels.
        fovy_deg: Vertical field of view in degrees.

    Returns:
        (N, 2) float32 array of (u, v) pixel coordinates.
    """
    import mujoco

    # Build camera view matrix from lookat / azimuth / elevation / distance.
    #
    # VF-3 fix (docs/vf3_bev_fixes.md): the formula below was previously
    # cam_fwd = (-sin(az)*cos(el), cos(az)*cos(el), sin(el)) -- i.e. the
    # world-space (cos(az), sin(az)) direction rotated +90 deg. This does NOT
    # match MuJoCo's real mjCAMERA_FREE convention, so every BEV overlay
    # (FOV cone + path trail + AVOID viz) that goes through this function was
    # silently drawn ~90 deg rotated from what the ACTUAL rendered BEV image
    # (produced by renderer.render_tp() -> real MuJoCo camera math) shows.
    #
    # Ground truth for MuJoCo's real convention comes from code/arena.py's
    # `_set_ego_cam` (empirically verified pitch-independent by CAM-P0,
    # docs/cam_p0.md, via cam.distance=1.0): it sets cam.azimuth=degrees(yaw),
    # cam.elevation=-pitch_deg, and its OWN forward vector (used to place
    # `cam.lookat`) is (cos(pitch)*cos(yaw), cos(pitch)*sin(yaw), -sin(pitch))
    # == (cos(el)*cos(az), cos(el)*sin(az), sin(el)) with el=-pitch, az=yaw.
    # Verified empirically here too: rendering known-position colored markers
    # via the real render_tp() and comparing their true pixel centroid against
    # this function's projection dropped the error from 300-560px (old buggy
    # formula) to 1-8px (this formula) across yaw=0/90/offset-position cases.
    az  = math.radians(bev_cam.azimuth)
    el  = math.radians(bev_cam.elevation)  # negative = below horizon
    dist = bev_cam.distance

    cosel = math.cos(el)
    sinel = math.sin(el)
    cosaz = math.cos(az)
    sinaz = math.sin(az)

    # Camera forward (from cam toward lookat) -- MuJoCo's real convention.
    cam_fwd = np.array([cosaz * cosel, sinaz * cosel, sinel], dtype=np.float64)

    lookat = np.array(bev_cam.lookat, dtype=np.float64)
    cam_pos = lookat - dist * cam_fwd

    # Camera right = cross(fwd, up) normalized; up is approximately world +Z
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cam_right = np.cross(cam_fwd, world_up)
    norm_r = np.linalg.norm(cam_right)
    if norm_r < 1e-8:
        cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        cam_right = cam_right / norm_r
    cam_up = np.cross(cam_right, cam_fwd)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)

    # Pinhole projection
    fovy_rad = math.radians(fovy_deg)
    fy = (h / 2.0) / math.tan(fovy_rad / 2.0)
    fovx_rad = 2.0 * math.atan(math.tan(fovy_rad / 2.0) * w / h)
    fx = (w / 2.0) / math.tan(fovx_rad / 2.0)
    cx, cy = w / 2.0 - 0.5, h / 2.0 - 0.5

    pts = np.asarray(world_pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]

    # Transform to camera frame
    delta = pts - cam_pos[np.newaxis, :]  # (N, 3)
    z_cam =  np.dot(delta, cam_fwd)     # forward  (N,)
    x_cam =  np.dot(delta, cam_right)   # right    (N,)
    y_cam = -np.dot(delta, cam_up)      # down (screen Y = down)

    # Perspective divide
    z_cam_safe = np.where(z_cam > 0.01, z_cam, 0.01)
    u = fx * x_cam / z_cam_safe + cx
    v = fy * y_cam / z_cam_safe + cy

    return np.stack([u, v], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# VF-1 small drawing helpers (pure cv2 pixel-pushing, no state reads beyond
# their arguments)
# ---------------------------------------------------------------------------
def _dashed_line(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 9,
    gap_len: int = 7,
) -> None:
    """Draw a dashed line segment from p0 to p1 (both (x,y) int tuples)."""
    import cv2
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return
    n_dashes = max(1, int(length / (dash_len + gap_len)))
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        seg_end = min(pos + dash_len, length)
        sx, sy = x0 + ux * pos, y0 + uy * pos
        ex, ey = x0 + ux * seg_end, y0 + uy * seg_end
        cv2.line(img, (int(round(sx)), int(round(sy))),
                  (int(round(ex)), int(round(ey))), color, thickness, cv2.LINE_AA)
        pos += dash_len + gap_len


def _lerp_color_bgr(
    c_cool: tuple[int, int, int], c_warm: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    """Linear-interpolate two BGR color tuples, t in [0,1] (0=cool, 1=warm)."""
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c_cool, c_warm))


# Path-trail gradient endpoints (BGR): cool blue (old) -> warm orange/red (recent).
TRAIL_COOL_BGR: tuple[int, int, int] = (230, 120, 40)   # blue-ish
TRAIL_WARM_BGR: tuple[int, int, int] = (30,  90, 255)   # warm orange-red


def draw_avoid_overlay(bev_img: np.ndarray, robot_xy: np.ndarray, robot_yaw: float,
                       bev_cam: "mujoco.MjvCamera", model: "mujoco.MjModel",
                       data: "mujoco.MjData", avoid_bias_wz: float,
                       avoid_info: Optional[dict], fovy_deg: float = 45.0) -> np.ndarray:
    """VF-1 item 2: visualize NX-9 AVOID's obstacle-repulsion bias on the BEV panel.

    Pure render-side read of the ALREADY-COMPUTED `avoid_bias_wz` / `avoid_info`
    (code/avoid.py's compute_obstacle_bias() return values, cached by the caller
    each grounding cycle) -- this function never calls compute_obstacle_bias
    itself and never influences the value fed back into steer.py's control law.

    No-op (returns bev_img unchanged) when the bias is within the deadband.

    Args:
        bev_img: (H, W, 3) uint8 BGR BEV frame to draw on (mutated in place).
        robot_xy: (2,) current robot world position.
        robot_yaw: Current robot yaw, in radians.
        bev_cam: MuJoCo free camera used for the BEV follow-cam view.
        model: MuJoCo model, forwarded to world_to_bev_pixel().
        data: MuJoCo data, forwarded to world_to_bev_pixel().
        avoid_bias_wz: Already-computed AVOID yaw-rate bias (positive = steer
            left, negative = steer right; same sign convention as steer.py).
        avoid_info: compute_obstacle_bias()'s own debug dict (`left`/`right`
            severities), or None when AVOID has no active bias this cycle.
        fovy_deg: Vertical field of view in degrees, forwarded to
            world_to_bev_pixel().

    Returns:
        The BGR frame with the AVOID overlay drawn (same array as `bev_img`
        when a bias was drawn; `bev_img` unchanged when there is nothing to
        draw).
    """
    import cv2
    from code import avoid as _avoid

    if avoid_info is None or abs(avoid_bias_wz) < 1e-6:
        return bev_img

    img = bev_img
    H, W = img.shape[:2]

    def w2p(xy: np.ndarray) -> tuple[int, int]:
        """World (x, y) to BEV pixel (u, v), rounded to the nearest int."""
        pix = world_to_bev_pixel(np.array([[xy[0], xy[1], 0.0]]), bev_cam, model, data, W, H, fovy_deg)
        return (int(round(pix[0, 0])), int(round(pix[0, 1])))

    # --- Obstacle-corridor wedge tint (same corridor geometry AVOID reads from
    # the depth frame: +/-AVOID_CORRIDOR_HALF_DEG about the robot heading, out to
    # AVOID_NEAR_M). Left/right halves shaded independently by L/R severity so the
    # tint visually matches which side the obstacle mass is actually on. ---
    half_rad = math.radians(_avoid.AVOID_CORRIDOR_HALF_DEG)
    rng_m    = _avoid.AVOID_NEAR_M
    n_pts    = 14
    L = float(avoid_info.get('left', 0.0))
    R = float(avoid_info.get('right', 0.0))

    for side, sev in (("left", L), ("right", R)):
        if sev <= 1e-4:
            continue
        if side == "left":
            ang_lo, ang_hi = robot_yaw, robot_yaw + half_rad
        else:
            ang_lo, ang_hi = robot_yaw - half_rad, robot_yaw
        poly = [[robot_xy[0], robot_xy[1]]]
        for i in range(n_pts + 1):
            ang = ang_lo + i * (ang_hi - ang_lo) / n_pts
            poly.append([robot_xy[0] + rng_m * math.cos(ang),
                         robot_xy[1] + rng_m * math.sin(ang)])
        poly_pix = np.array([w2p(p) for p in poly], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly_pix], (20, 50, 200), cv2.LINE_AA)  # warm red-orange, BGR
        a = 0.12 + 0.35 * min(1.0, sev)
        cv2.addWeighted(overlay, a, img, 1.0 - a, 0, img)

    # --- Repulsion arrow: bias_wz > 0 = steer LEFT, < 0 = steer RIGHT (same sign
    # convention as steer.py/avoid.py). Drawn as a curved-looking short arrow
    # tangential to the heading, scaled by |bias| / cap. ---
    sev_overall = min(1.0, abs(avoid_bias_wz) / max(_avoid.AVOID_MAX_WZ_BIAS, 1e-6))
    turn_sign   = 1.0 if avoid_bias_wz > 0 else -1.0
    arrow_ang   = robot_yaw + turn_sign * (math.pi / 2.0) * (0.5 + 0.5 * sev_overall)
    arrow_len_m = 0.5 + 1.2 * sev_overall
    base_pt = w2p(robot_xy)
    tip_pt  = w2p([robot_xy[0] + arrow_len_m * math.cos(arrow_ang),
                   robot_xy[1] + arrow_len_m * math.sin(arrow_ang)])
    cv2.arrowedLine(img, base_pt, tip_pt, (0, 210, 255), 3, cv2.LINE_AA, tipLength=0.35)

    # --- "AVOID" indicator chip (top-left of BEV panel, only lit while active) ---
    chip_txt = "AVOID"
    cv2.rectangle(img, (6, 6), (76, 28), (0, 140, 255), -1)
    cv2.putText(img, chip_txt, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, chip_txt, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# BEV Overlay drawing
# ---------------------------------------------------------------------------

def draw_bev_overlays(
    bev_img: np.ndarray,         # (H, W, 3) uint8 BGR
    path_trail: List[np.ndarray],# list of (x,y) world positions
    target_xy: Optional[np.ndarray],  # (2,) world position of target, or None
    robot_xy: np.ndarray,        # (2,) current robot world position
    robot_yaw: float,            # current robot yaw (radians)
    bev_cam: "mujoco.MjvCamera",
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    state: str = STATE_IDLE,
    prompt: str = "",
    dist_to_target: Optional[float] = None,
    fovy_deg: float = 45.0,
    # FD2: multi-goal progress
    goal_idx: int = 0,
    n_goals: int = 1,
    completed_targets: Optional[List[np.ndarray]] = None,  # already-reached targets
    # VF-1 item 4: dashed robot->target goal line drawn in the target's own color.
    target_color_bgr: Optional[tuple] = None,
    # VF-1 item 2: AVOID visualization (pure read of already-computed bias/info).
    avoid_bias_wz: float = 0.0,
    avoid_info: Optional[dict] = None,
) -> np.ndarray:
    """Draw all BEV overlays on bev_img (in-place + return).

    Overlays:
      (a) PATH TRAIL — green polyline (VF-1: gradient cool->warm when FEAT_TRAIL)
      (b) TARGET HIGHLIGHT — orange ring + cross
      (c) FOV CONE — yellow wedge on ground
      (d) STATUS BANNER — bottom banner with state + distance
      (e) VF-1: AVOID repulsion viz (item 2), dashed goal line in target color (item 4)

    Args:
        bev_img: (H, W, 3) uint8 BGR BEV frame (a copy is drawn on, not
            mutated in place -- the annotated copy is returned).
        path_trail: List of (x, y) world positions, oldest first.
        target_xy: (2,) world position of the current sub-goal target, or
            None if there is no target to highlight.
        robot_xy: (2,) current robot world position.
        robot_yaw: Current robot yaw, in radians.
        bev_cam: MuJoCo free camera used for the BEV follow-cam view.
        model: MuJoCo model, forwarded to world_to_bev_pixel().
        data: MuJoCo data, forwarded to world_to_bev_pixel().
        state: Current state-machine state (one of the STATE_* constants),
            used for the status banner and for gating the goal line.
        prompt: Typed instruction text shown in the status banner.
        dist_to_target: Current distance to target in meters, or None.
        fovy_deg: Vertical field of view in degrees, forwarded to
            world_to_bev_pixel().
        goal_idx: Zero-based index of the current sub-goal (multi-goal runs).
        n_goals: Total number of sub-goals in this episode.
        completed_targets: World (x, y) positions of already-reached targets,
            drawn as green check marks.
        target_color_bgr: The current target's own BGR color, used for the
            dashed goal line (VF-1 item 4); None falls back to the plain
            magenta line.
        avoid_bias_wz: Already-computed AVOID yaw-rate bias, forwarded to
            draw_avoid_overlay().
        avoid_info: compute_obstacle_bias()'s own debug dict, forwarded to
            draw_avoid_overlay().

    Returns:
        A new (H, W, 3) uint8 BGR frame with all overlays drawn.
    """
    import cv2

    img = bev_img.copy()
    H, W = img.shape[:2]

    def w2p(world_xyz: np.ndarray) -> tuple[int, int]:
        """World XYZ (or XY with Z=0) to pixel (u,v) as int tuple."""
        if len(world_xyz) == 2:
            world_xyz = np.array([world_xyz[0], world_xyz[1], 0.0])
        pix = world_to_bev_pixel(np.array([world_xyz]), bev_cam, model, data, W, H, fovy_deg)
        return (int(round(pix[0, 0])), int(round(pix[0, 1])))

    # (a) PATH TRAIL — polyline of past base positions
    # VF-1 item 4: gradient color by recency (cool blue -> warm orange/red),
    # thicker line. FEAT_TRAIL=0 (or FANCY_PLAIN) keeps the exact original
    # single-color fade-only trail.
    if len(path_trail) >= 2:
        pts_pix = []
        for xy in path_trail:
            p = w2p(xy)
            pts_pix.append(p)
        n_pts = len(pts_pix)
        for i in range(1, n_pts):
            p0 = pts_pix[i - 1]
            p1 = pts_pix[i]
            if not ((0 <= p0[0] < W and 0 <= p0[1] < H) or (0 <= p1[0] < W and 0 <= p1[1] < H)):
                continue
            if FEAT_TRAIL:
                t = i / max(1, n_pts - 1)          # 0 (oldest) -> 1 (most recent)
                c = _lerp_color_bgr(TRAIL_COOL_BGR, TRAIL_WARM_BGR, t)
                thickness = 2 + (2 if t > 0.7 else 0)
                cv2.line(img, p0, p1, c, thickness, cv2.LINE_AA)
            else:
                alpha = i / n_pts  # fade from dim to bright (ORIGINAL behavior)
                c = tuple(int(x * alpha + x * 0.3) for x in COLOR_PATH_TRAIL)
                cv2.line(img, p0, p1, COLOR_PATH_TRAIL, 2, cv2.LINE_AA)

    # Robot position dot (white)
    robot_pix = w2p([robot_xy[0], robot_xy[1], 0.0])
    if 0 <= robot_pix[0] < W and 0 <= robot_pix[1] < H:
        cv2.circle(img, robot_pix, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, robot_pix, 7, (0, 0, 0), 1, cv2.LINE_AA)

    # (c) FOV CONE — the robot's ACTIVE detection camera's real horizontal FOV
    # wedge on the ground, anchored at robot_yaw (== the camera's azimuth;
    # arena._set_ego_cam sets cam.azimuth=degrees(yaw) exactly, no independent
    # pan, for the ego/GROUNDING/PROXIMITY cameras alike).
    #
    # VF-3 fix (docs/vf3_bev_fixes.md): this used to be a hardcoded ±45 deg,
    # from a comment that conflated the camera's VERTICAL FOVY with a
    # horizontal half-angle ("90° FOVY -> ±45°"). Two separate errors:
    #   1. The ACTUAL rendered FOVY is 45° (MuJoCo's model.vis.global_.fovy
    #      default — see code/grounding.py's EGO_FOVY_RENDERED / "E6 fix v4"
    #      comment), not the stale arena.EGO_FOVY=90 constant the old ±45 was
    #      (incorrectly) derived from.
    #   2. Even the correct FOVY needs the aspect-ratio pinhole conversion —
    #      the same atan(tan(fovy/2)*w/h) formula this file already uses in
    #      world_to_bev_pixel()/arena.get_ego_intrinsics() — to get the
    #      HORIZONTAL half-angle, not a naive fovy/2.
    # Real horizontal half-FOV for FOVY=45° at the GROUNDING/PROXIMITY cameras'
    # shared 4:3 aspect (480x360 / 320x240 — both equal 4:3, so this is the
    # same value regardless of which one is currently active):
    # atan(tan(22.5°)*4/3) = 28.87°.
    _FOV_CONE_FOVY_DEG = 45.0        # actual rendered FOVY (all non-widefov cams)
    _FOV_CONE_ASPECT   = 4.0 / 3.0   # GROUNDING_W/H == PROXIMITY_W/H aspect ratio
    fov_half_rad = math.atan(math.tan(math.radians(_FOV_CONE_FOVY_DEG) / 2.0) * _FOV_CONE_ASPECT)
    fov_range    = 3.0                  # metres shown
    N_CONE_PTS   = 10
    cone_pts = []
    cone_pts.append([robot_xy[0], robot_xy[1], 0.0])
    for i in range(N_CONE_PTS + 1):
        ang = robot_yaw - fov_half_rad + i * (2 * fov_half_rad / N_CONE_PTS)
        px = robot_xy[0] + fov_range * math.cos(ang)
        py = robot_xy[1] + fov_range * math.sin(ang)
        cone_pts.append([px, py, 0.0])
    cone_pts.append([robot_xy[0], robot_xy[1], 0.0])  # close the polygon

    cone_pix = []
    for pt in cone_pts:
        cone_pix.append(w2p(pt))
    cone_arr = np.array(cone_pix, dtype=np.int32)

    # Draw filled translucent cone
    overlay = img.copy()
    cv2.fillPoly(overlay, [cone_arr], (30, 80, 30), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    # Draw cone outline
    cv2.polylines(img, [cone_arr], isClosed=True, color=COLOR_FOV_CONE, thickness=1, lineType=cv2.LINE_AA)

    # (b) TARGET HIGHLIGHT — ring + cross at target position
    if target_xy is not None:
        tgt_pix = w2p([target_xy[0], target_xy[1], 0.0])
        if 0 <= tgt_pix[0] < W and 0 <= tgt_pix[1] < H:
            # VF-4 (user verification pass): the previous r=14 double ring +
            # center-filling crosshair completely covered the target mesh
            # (~5-9 px at whole-arena zoom), making it impossible to SEE that
            # the marker is actually on the object. Keep the center clear:
            # thin open ring + 4 corner ticks OUTSIDE the ring, nothing within
            # r-2 px of the object itself.
            r = 13
            cv2.circle(img, tgt_pix, r, COLOR_TARGET_RING, 1, cv2.LINE_AA)
            cv2.circle(img, tgt_pix, r + 5, COLOR_TARGET_RING, 1, cv2.LINE_AA)
            for sx, sy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p_in  = (tgt_pix[0] + sx * (r + 6), tgt_pix[1] + sy * (r + 6))
                p_out = (tgt_pix[0] + sx * (r + 12), tgt_pix[1] + sy * (r + 12))
                cv2.line(img, p_in, p_out, COLOR_TARGET_RING, 2, cv2.LINE_AA)
        else:
            # Target off-screen — draw arrow pointing toward it from robot
            # Direction arrow from robot
            if robot_pix[0] >= 0 and robot_pix[0] < W and robot_pix[1] >= 0 and robot_pix[1] < H:
                dx = tgt_pix[0] - robot_pix[0]
                dy = tgt_pix[1] - robot_pix[1]
                length = math.hypot(dx, dy)
                if length > 0:
                    nx, ny = dx / length, dy / length
                    tip = (int(robot_pix[0] + 40 * nx), int(robot_pix[1] + 40 * ny))
                    cv2.arrowedLine(img, robot_pix, tip, COLOR_TARGET_RING, 2, tipLength=0.4)

    # Draw completed targets (green circles — already reached)
    if completed_targets:
        for ct_xy in completed_targets:
            ct_pix = w2p([ct_xy[0], ct_xy[1], 0.0])
            if 0 <= ct_pix[0] < W and 0 <= ct_pix[1] < H:
                cv2.circle(img, ct_pix, 12, (60, 220, 60), 2, cv2.LINE_AA)
                cv2.putText(img, "✓", (ct_pix[0] - 6, ct_pix[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 220, 60), 1, cv2.LINE_AA)

    # Draw a line from robot to target if target visible in BEV.
    # VF-1 item 4: dashed, in the target's own color, when FEAT_TRAIL + a color
    # was supplied; otherwise the exact original thin solid magenta line.
    if target_xy is not None and dist_to_target is not None:
        tgt_pix2 = w2p([target_xy[0], target_xy[1], 0.0])
        if (0 <= tgt_pix2[0] < W and 0 <= tgt_pix2[1] < H and
                0 <= robot_pix[0] < W and 0 <= robot_pix[1] < H and
                state in (STATE_MOVING, STATE_LOCATED)):
            # VF-4: stop the goal line at the target ring's edge so the line
            # never crosses over (and hides) the target mesh itself.
            _gdx = tgt_pix2[0] - robot_pix[0]
            _gdy = tgt_pix2[1] - robot_pix[1]
            _glen = math.hypot(_gdx, _gdy)
            if _glen > 22.0:
                _gend = (int(tgt_pix2[0] - _gdx / _glen * 20), int(tgt_pix2[1] - _gdy / _glen * 20))
                if FEAT_TRAIL and target_color_bgr is not None:
                    _dashed_line(img, robot_pix, _gend, target_color_bgr, thickness=2)
                else:
                    cv2.line(img, robot_pix, _gend, (180, 80, 255), 1, cv2.LINE_AA)

    # VF-1 item 2: AVOID repulsion visualization (corridor tint + arrow + chip).
    # Pure read of already-computed avoid_bias_wz/avoid_info -- no-op internally
    # when the bias is within the deadband.
    if FEAT_AVOID_VIZ:
        img = draw_avoid_overlay(img, robot_xy, robot_yaw, bev_cam, model, data,
                                  avoid_bias_wz, avoid_info, fovy_deg=fovy_deg)

    # (d) STATUS BANNER — bottom strip (56px normal, 68px for multi-goal)
    banner_h = 68 if n_goals > 1 else 56
    banner = np.full((banner_h, W, 3), 20, dtype=np.uint8)

    # State color
    state_color_map = {
        STATE_SEARCHING: COLOR_STATE_SEARCH,
        STATE_LOCATED:   COLOR_STATE_LOCATE,
        STATE_MOVING:    COLOR_STATE_MOVE,
        STATE_REACHED:   COLOR_STATE_REACH,
        STATE_FAILED:    (100, 100, 100),
        STATE_IDLE:      (150, 150, 150),
    }
    sc = state_color_map.get(state, (200, 200, 200))

    # State badge
    badge_text = state
    cv2.rectangle(banner, (4, 4), (120, 30), sc, -1)
    cv2.putText(banner, badge_text, (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # FD2: Sub-goal progress badge (multi-goal) — e.g. "goal 1/2"
    if n_goals > 1:
        goal_text = f"goal {goal_idx + 1}/{n_goals}"
        # Draw progress bar (small dots)
        bar_x = 4
        bar_y = 38
        for gi in range(n_goals):
            dot_x = bar_x + gi * 18
            dot_clr = (80, 220, 80) if gi < goal_idx else ((255, 165, 0) if gi == goal_idx else (80, 80, 80))
            cv2.circle(banner, (dot_x, bar_y), 6, dot_clr, -1, cv2.LINE_AA)
        cv2.putText(banner, goal_text, (bar_x + n_goals * 18 + 6, bar_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        row2 = 58
    else:
        row2 = 50

    # Prompt text (row 1)
    prompt_disp = prompt[:55] + "..." if len(prompt) > 55 else prompt
    if prompt_disp:
        cv2.putText(banner, prompt_disp, (130, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    # Distance
    if dist_to_target is not None:
        dist_text = f"dist: {dist_to_target:.2f}m"
        cv2.putText(banner, dist_text, (W - 140, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 100), 1, cv2.LINE_AA)

    # "BEV CAM" label
    cv2.putText(banner, "BEV", (4, row2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    # Composite banner onto image (extend image if needed)
    if banner_h != img.shape[0]:
        # Resize image to accommodate taller banner
        pass  # banner replaces last banner_h rows
    img[-banner_h:, :] = banner

    return img


_STATE_COLOR_MAP: dict[str, tuple[int, int, int]] = {
    STATE_SEARCHING: COLOR_STATE_SEARCH,
    STATE_LOCATED:   COLOR_STATE_LOCATE,
    STATE_MOVING:    COLOR_STATE_MOVE,
    STATE_REACHED:   COLOR_STATE_REACH,
    STATE_FAILED:    (100, 100, 100),
    STATE_IDLE:      (150, 150, 150),
}


def draw_detector_heatmap_overlay(
    ego_bgr: np.ndarray,
    heatmap_cache: Optional[dict],
    target_color: str,
    target_shape: str,
    alpha: float = HEATMAP_ALPHA,
) -> tuple[np.ndarray, Optional[float]]:
    """VF-1 item 1: blend the NX-6 GROUND_NET detector's OWN confidence heatmap
    (cached by code/grounding.py's _ground_net() the same cycle it already ran
    the forward pass for detection -- ZERO extra inference here) onto the ego
    panel as a semi-transparent color map.

    Render-side only: reads a cache, writes only to the returned image copy.
    No-op (returns (ego_bgr, None) unchanged) when GROUND_NET was never
    invoked, the cache is empty/stale (wrong color+shape query -- i.e. the
    cache belongs to a different target than the one THIS episode is
    pursuing), or the cached cycle did not accept a detection.

    Args:
        ego_bgr: (H, W, 3) uint8 BGR ego panel frame to blend onto.
        heatmap_cache: get_ground_net_last_heatmap()'s cache dict (`prob`,
            `color`, `shape`, `accepted`, `confidence`), or None if GROUND_NET
            was never invoked.
        target_color: This episode's target color name, used to check the
            cache matches the currently-pursued target.
        target_shape: This episode's target shape name, used to check the
            cache matches the currently-pursued target.
        alpha: Maximum blend strength (per-pixel alpha is confidence-scaled,
            capped at this value).

    Returns:
        Tuple of (blended_bgr, confidence): `blended_bgr` is a new frame with
        the heatmap blended in, or `ego_bgr` unchanged when there is nothing
        to draw; `confidence` is the cached detection confidence, or None
        when nothing was drawn.
    """
    import cv2
    if heatmap_cache is None or heatmap_cache.get('prob') is None:
        return ego_bgr, None
    if (heatmap_cache.get('color') != target_color.lower().strip() or
            heatmap_cache.get('shape') != target_shape.lower().strip()):
        return ego_bgr, None
    if not heatmap_cache.get('accepted', False):
        return ego_bgr, None

    prob = heatmap_cache['prob']
    h, w = ego_bgr.shape[:2]
    prob_r = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    prob_u8 = np.clip(prob_r * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)

    # Per-pixel alpha proportional to confidence (capped at `alpha`) -- a smooth
    # glow that fades out with confidence rather than a hard-edged patch, and
    # is a provable no-op (alpha~0) wherever the map says "nothing here" (gate
    # check: heatmap must not obscure the scene). A small blur spreads the
    # (typically tight, few-pixel) detector peak into a visible glow radius --
    # purely a display nicety, the underlying confidence values are unchanged.
    alpha_map = np.clip(prob_r, 0.0, 1.0) * alpha
    blur_px = max(3, int(round(0.02 * w))) | 1   # odd kernel size, ~2% of panel width
    alpha_map = cv2.GaussianBlur(alpha_map, (blur_px, blur_px), 0)
    alpha_map = alpha_map[..., None]
    out = (heat_bgr.astype(np.float32) * alpha_map +
           ego_bgr.astype(np.float32) * (1.0 - alpha_map)).astype(np.uint8)
    return out, float(heatmap_cache.get('confidence', 0.0))


def compose_sbs_frame(
    ego_rgb: np.ndarray,   # (EGO_H, EGO_W, 3) uint8 RGB — CAM-2 ACTIVE camera feed
    bev_img: np.ndarray,   # (BEV_H, BEV_W, 3) uint8 BGR
    state: str = STATE_IDLE,
    prompt: str = "",
    dist_to_target: Optional[float] = None,
    goal_idx: int = 0,
    n_goals: int = 1,
    active_cam: str = "GROUNDING",   # CAM-2 (docs/cam_p1.md): 'GROUNDING' (head, far) | 'PROXIMITY' (near)
    # VF-1 item 1: detector heatmap cache + the query it should match.
    heatmap_cache: Optional[dict] = None,
    target_color: str = "",
    target_shape: str = "",
    # VF-1 item 3: HUD bar context dict (see draw_hud_bar) — None disables it
    # regardless of FEAT_HUD.
    hud_ctx: Optional[dict] = None,
) -> np.ndarray:
    """Compose side-by-side frame: ego (left, CAM-2 active-camera feed) | BEV (right)
    [+ VF-1 bottom HUD strip when FEAT_HUD and hud_ctx is given].

    When every VF-1 toggle is off (FANCY_PLAIN=1) this reproduces the pre-VF1
    frame byte-for-byte (same resize target, same badge layout, same divider).

    Args:
        ego_rgb: (EGO_H, EGO_W, 3) uint8 RGB CAM-2 active-camera feed.
        bev_img: (BEV_H, BEV_W, 3) uint8 BGR BEV frame (with its own overlays
            already drawn by draw_bev_overlays()).
        state: Current state-machine state (one of the STATE_* constants).
        prompt: Typed instruction text (unused directly here; forwarded to
            draw_bev_overlays() by the caller).
        dist_to_target: Current distance to target in meters, or None.
        goal_idx: Zero-based index of the current sub-goal (multi-goal runs).
        n_goals: Total number of sub-goals in this episode.
        active_cam: CAM-2 (docs/cam_p1.md) active camera name -- 'GROUNDING'
            (head, far) or 'PROXIMITY' (near).
        heatmap_cache: get_ground_net_last_heatmap()'s cache dict, forwarded
            to draw_detector_heatmap_overlay(); None disables the overlay.
        target_color: This episode's target color name, forwarded to
            draw_detector_heatmap_overlay().
        target_shape: This episode's target shape name, forwarded to
            draw_detector_heatmap_overlay().
        hud_ctx: HUD bar context dict (see draw_hud_bar()); None disables the
            HUD bar regardless of FEAT_HUD.

    Returns:
        (H, W, 3) uint8 BGR composited frame.
    """
    import cv2

    # Convert ego from RGB to BGR
    ego_bgr = cv2.cvtColor(ego_rgb, cv2.COLOR_RGB2BGR)

    # VF-1 item 6: display both panels at PANEL_DISPLAY_W x PANEL_DISPLAY_H
    # (upscaled from the UNCHANGED native render sizes via cv2.resize — no
    # extra MuJoCo render cost). Falls back to the exact original "scale ego
    # to BEV_H, keep BEV native" behavior when FEAT_HIRES is off.
    if FEAT_HIRES:
        target_h = PANEL_DISPLAY_H
        if (ego_bgr.shape[1], ego_bgr.shape[0]) != (PANEL_DISPLAY_W, PANEL_DISPLAY_H):
            ego_bgr = cv2.resize(ego_bgr, (PANEL_DISPLAY_W, PANEL_DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        if (bev_img.shape[1], bev_img.shape[0]) != (PANEL_DISPLAY_W, PANEL_DISPLAY_H):
            bev_img = cv2.resize(bev_img, (PANEL_DISPLAY_W, PANEL_DISPLAY_H), interpolation=cv2.INTER_LINEAR)
    else:
        target_h = BEV_H
        if ego_bgr.shape[0] != target_h:
            scale = target_h / ego_bgr.shape[0]
            ego_bgr = cv2.resize(ego_bgr, (int(ego_bgr.shape[1] * scale), target_h))

    # VF-1 item 1: blend the detector heatmap AFTER the resize (so both the
    # color blend and the text tag below are at final display resolution).
    heatmap_conf = None
    if FEAT_HEATMAP:
        ego_bgr, heatmap_conf = draw_detector_heatmap_overlay(ego_bgr, heatmap_cache,
                                                              target_color, target_shape)

    # Ego overlay: state badge + active-camera label. VF-1: larger badge/font
    # when FEAT_HIRES (kept at the original small size otherwise).
    badge_h = 70 if FEAT_HIRES else 36
    cv2.rectangle(ego_bgr, (0, 0), (ego_bgr.shape[1], badge_h), (20, 20, 20), -1)
    sc = _STATE_COLOR_MAP.get(state, (200, 200, 200))
    if FEAT_HIRES:
        cv2.rectangle(ego_bgr, (10, 8), (240, 58), sc, -1)
        cv2.putText(ego_bgr, state, (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 3, cv2.LINE_AA)
    else:
        cv2.rectangle(ego_bgr, (4, 4), (90, 30), sc, -1)
        cv2.putText(ego_bgr, state[:10], (7, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    # CAM-2 handoff label: "HEAD CAM" (GROUNDING, far) / "PROXIMITY CAM" (near) —
    # makes the camera handoff visible to viewers, distinct from the small
    # "CAM: GROUNDING|PROXIMITY d=X.XXm" overlay already baked into ego_rgb by
    # _label_active_cam() in the main rollout loop.
    cam_label = "PROXIMITY CAM" if active_cam == "PROXIMITY" else "HEAD CAM"
    cam_color = (60, 210, 255) if active_cam == "PROXIMITY" else (255, 200, 150)
    cam_font  = 0.9 if FEAT_HIRES else 0.4
    cam_thick = 2 if FEAT_HIRES else 1
    (tw, th_), _ = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, cam_font, cam_thick)
    cam_tx = ego_bgr.shape[1] - tw - (16 if FEAT_HIRES else 8)
    cam_ty = 44 if FEAT_HIRES else 23
    # VF-1 item 3: flash the camera chip's background for a few frames right
    # after a GROUNDING<->PROXIMITY handoff (hud_ctx['cam_flash'], a pure
    # render-side counter maintained by the caller — never read by control).
    if hud_ctx is not None and hud_ctx.get('cam_flash'):
        pad = 6
        cv2.rectangle(ego_bgr, (cam_tx - pad, cam_ty - th_ - pad),
                      (cam_tx + tw + pad, cam_ty + pad), (0, 255, 255), -1)
        cv2.putText(ego_bgr, cam_label, (cam_tx, cam_ty),
                    cv2.FONT_HERSHEY_SIMPLEX, cam_font, (0, 0, 0), cam_thick, cv2.LINE_AA)
    else:
        cv2.putText(ego_bgr, cam_label, (cam_tx, cam_ty),
                    cv2.FONT_HERSHEY_SIMPLEX, cam_font, cam_color, cam_thick, cv2.LINE_AA)

    # VF-1 item 1: "NEURAL DETECTOR" tag + live confidence, bottom-left of the
    # ego panel (drawn at final display resolution for a crisp font).
    if FEAT_HEATMAP and heatmap_conf is not None:
        tag = f"NEURAL DETECTOR  conf={heatmap_conf:.2f}"
        tag_font = 0.62 if FEAT_HIRES else 0.38
        ty = ego_bgr.shape[0] - (14 if FEAT_HIRES else 8)
        cv2.putText(ego_bgr, tag, (10, ty), cv2.FONT_HERSHEY_SIMPLEX, tag_font,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(ego_bgr, tag, (10, ty), cv2.FONT_HERSHEY_SIMPLEX, tag_font,
                    (60, 255, 210), 1, cv2.LINE_AA)

    # Divider line
    divider = np.full((ego_bgr.shape[0], 3, 3), 60, dtype=np.uint8)

    sbs = np.concatenate([ego_bgr, divider, bev_img], axis=1)

    # VF-1 item 3: bottom HUD bar (separate strip, full canvas width).
    if FEAT_HUD and hud_ctx is not None:
        hud_strip = draw_hud_bar(sbs.shape[1], hud_ctx)
        sbs = np.concatenate([sbs, hud_strip], axis=0)

    return sbs


def draw_hud_bar(width: int, ctx: dict) -> np.ndarray:
    """VF-1 item 3: bottom HUD strip spanning the full canvas width --
      - typed instruction, verbatim (left)
      - live distance + bearing, step counter, walk speed (right)
      - 5-stage skill breadcrumb SCAN > LOCK > WALK > HANDOFF > REACH, active
        stage highlighted (center)
      - camera-in-use chip (HEAD/PROXIMITY), flashes for a few frames right
        after a handoff

    Pure render-side function: every field in `ctx` is a read of state that
    already exists in run_fancy_rollout (see its call site in _render_sbs_frame).

    Args:
        width: Full canvas width in pixels (the HUD strip spans this width).
        ctx: Context dict with keys `prompt`, `stage_idx`, `dist`,
            `bearing_deg`, `step`, `walk_speed_mps`, `active_cam`,
            `cam_flash` (see _render_sbs_frame's hud_ctx construction).

    Returns:
        (HUD_BAR_H, width, 3) uint8 BGR HUD strip.
    """
    import cv2
    h = HUD_BAR_H
    img = np.full((h, width, 3), (18, 18, 24), dtype=np.uint8)
    cv2.line(img, (0, 0), (width, 0), (70, 70, 90), 1, cv2.LINE_AA)

    # --- Left: typed instruction, verbatim (truncated only if it can't fit) ---
    prompt = ctx.get('prompt') or ''
    max_chars = max(10, width // 12)
    prompt_disp = prompt if len(prompt) <= max_chars else prompt[:max_chars - 3] + "..."
    cv2.putText(img, f'"{prompt_disp}"', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (220, 220, 235), 1, cv2.LINE_AA)

    # --- Center: skill breadcrumb ---
    stage_idx = ctx.get('stage_idx', -1)
    # Pre-measure total width so the breadcrumb is truly centered.
    seg_font, sep = 0.44, "  >  "
    widths = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, seg_font, 2)[0][0] for s in SKILL_STAGES]
    sep_w  = cv2.getTextSize(sep, cv2.FONT_HERSHEY_SIMPLEX, seg_font, 1)[0][0]
    total_w = sum(widths) + sep_w * (len(SKILL_STAGES) - 1)
    bc_x = max(10, (width - total_w) // 2)
    bc_y = 28
    for i, lab in enumerate(SKILL_STAGES):
        active = (i == stage_idx)
        done = (i < stage_idx)
        color = (255, 255, 255) if active else ((100, 220, 130) if done else (95, 95, 105))
        thick = 2 if active else 1
        if active:
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, seg_font, thick)
            cv2.rectangle(img, (bc_x - 6, bc_y - th - 6), (bc_x + tw + 6, bc_y + 6), (150, 90, 20), -1)
        cv2.putText(img, lab, (bc_x, bc_y), cv2.FONT_HERSHEY_SIMPLEX, seg_font, color, thick, cv2.LINE_AA)
        bc_x += widths[i]
        if i < len(SKILL_STAGES) - 1:
            cv2.putText(img, sep, (bc_x, bc_y), cv2.FONT_HERSHEY_SIMPLEX, seg_font, (90, 90, 100), 1, cv2.LINE_AA)
            bc_x += sep_w

    # --- Right: distance / bearing / step / speed + camera chip ---
    dist        = ctx.get('dist')
    bearing_deg = ctx.get('bearing_deg')
    step        = ctx.get('step', 0)
    speed       = ctx.get('walk_speed_mps', 0.0)
    parts = []
    if dist is not None:
        parts.append(f"dist={dist:.2f}m")
    if bearing_deg is not None:
        parts.append(f"brg={bearing_deg:+.0f}deg")
    parts.append(f"step={step}")
    parts.append(f"v={speed:.2f}m/s")
    txt = "   ".join(parts)

    chip_w = 118
    (txt_w, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    chip_x0 = width - 12 - chip_w
    cv2.putText(img, txt, (chip_x0 - txt_w - 18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (150, 220, 255), 1, cv2.LINE_AA)

    active_cam = ctx.get('active_cam', 'GROUNDING')
    cam_flash  = bool(ctx.get('cam_flash'))
    cam_label  = "PROXIMITY" if active_cam == 'PROXIMITY' else "HEAD"
    cam_color  = (60, 210, 255) if active_cam == 'PROXIMITY' else (255, 200, 150)
    chip_bg    = (0, 230, 255) if cam_flash else (48, 48, 58)
    cv2.rectangle(img, (chip_x0, 8), (chip_x0 + chip_w, h - 8), chip_bg, -1)
    cv2.rectangle(img, (chip_x0, 8), (chip_x0 + chip_w, h - 8), (90, 90, 100), 1)
    cv2.putText(img, f"CAM: {cam_label}", (chip_x0 + 8, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 0) if cam_flash else cam_color, 1, cv2.LINE_AA)

    return img


def _final_canvas_dims() -> Tuple[int, int]:
    """Mirrors compose_sbs_frame's own size arithmetic WITHOUT rendering anything,
    so the title/outro card frames (built before/after the simulation loop, with
    no ego/bev frame at hand) match the exact (H, W) of the per-step SBS frames
    -- required since every frame appended to one video must share one shape.

    Returns:
        (height, width) in pixels, matching compose_sbs_frame()'s output shape
        for the current FEAT_HIRES / FEAT_HUD toggle state.
    """
    if FEAT_HIRES:
        w = PANEL_DISPLAY_W * 2 + 3
        h = PANEL_DISPLAY_H
    else:
        # ORIGINAL sizing: ego (always resized to EGO_W x EGO_H by
        # _label_active_cam) is rescaled to BEV_H tall in compose_sbs_frame,
        # i.e. width = EGO_W * (BEV_H / EGO_H); BEV stays native BEV_W x BEV_H.
        w = int(EGO_W * (BEV_H / EGO_H)) + 3 + BEV_W
        h = BEV_H
    if FEAT_HUD:
        h += HUD_BAR_H
    return h, w


def make_title_card(instruction: str, scenario_title: str, frame_idx: int, n_frames: int) -> np.ndarray:
    """VF-1 item 5: ~1.5s pre-roll title card -- scenario name (large) + the
    typed instruction, with a short fade-in over the first ~10 frames. Static
    content generated BEFORE the simulation loop starts (see its call site in
    run_fancy_rollout) -- purely additive frames, never interleaved with control.

    Args:
        instruction: Typed instruction text (or the combined multi-goal
            instruction) shown under the scenario title.
        scenario_title: Large scenario name shown at the top of the card.
        frame_idx: Zero-based index of this frame within the title-card
            sequence, used to compute the fade-in.
        n_frames: Total number of frames in the title-card sequence (fade-in
            completes at frame_idx >= 10, well before n_frames typically).

    Returns:
        (H, W, 3) uint8 BGR title-card frame, matching _final_canvas_dims().
    """
    import cv2
    h, w = _final_canvas_dims()
    img = np.full((h, w, 3), (24, 18, 14), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (60, 50, 40), 2)

    fade = min(1.0, frame_idx / 10.0)

    def _fade(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
        """Scale a BGR color tuple by the current fade-in level."""
        return tuple(int(c * fade) for c in bgr)

    title = scenario_title
    font_scale_title = min(1.8, w / 700.0)
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, 3)
    cv2.putText(img, title, ((w - tw) // 2, h // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_title, _fade((230, 230, 235)), 3, cv2.LINE_AA)

    instr = f'"{instruction}"'
    font_scale_instr = min(1.1, w / 900.0)
    (iw, ih), _ = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, font_scale_instr, 2)
    cv2.putText(img, instr, ((w - iw) // 2, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale_instr, _fade((120, 220, 255)), 2, cv2.LINE_AA)

    sub = "G1 HUMANOID  -  AUTONOMOUS VISUAL SEARCH & RETRIEVAL"
    (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img, sub, ((w - sw) // 2, h // 2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                _fade((150, 150, 160)), 1, cv2.LINE_AA)

    pipeline = "   ".join(SKILL_STAGES)
    (pw, ph), _ = cv2.getTextSize(pipeline, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, pipeline, ((w - pw) // 2, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                _fade((90, 200, 140)), 1, cv2.LINE_AA)

    return img


def make_outro_card(last_frame: np.ndarray, sim_time_s: float, dist_traveled_m: float,
                    final_dist_m: float, steps: int) -> np.ndarray:
    """VF-1 item 5: ~2s freeze-frame on REACHED with a stats card overlay (elapsed
    sim time, distance traveled, final distance to target, step count). Built
    from the ACTUAL last rendered SBS frame (scene/robot/target still visible)
    plus a semi-transparent stats panel -- never re-renders anything.

    Args:
        last_frame: The final rendered SBS frame of the episode (copied, not
            mutated), used as the freeze-frame background.
        sim_time_s: Elapsed simulated time in seconds.
        dist_traveled_m: Total odometry distance traveled in meters.
        final_dist_m: Final distance to target in meters.
        steps: Total number of control steps taken.

    Returns:
        (H, W, 3) uint8 BGR outro-card frame, same shape as `last_frame`.
    """
    import cv2
    img = last_frame.copy()
    h, w = img.shape[:2]

    panel_w = min(440, w - 40)
    panel_h = 190
    px0, py0 = (w - panel_w) // 2, (h - panel_h) // 2
    overlay = img.copy()
    cv2.rectangle(overlay, (px0, py0), (px0 + panel_w, py0 + panel_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)
    cv2.rectangle(img, (px0, py0), (px0 + panel_w, py0 + panel_h), (90, 220, 140), 2)

    cv2.putText(img, "REACHED", (px0 + 22, py0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                (90, 220, 140), 2, cv2.LINE_AA)
    lines = [
        f"time:       {sim_time_s:5.1f} s",
        f"traveled:   {dist_traveled_m:5.2f} m",
        f"final dist: {final_dist_m:5.3f} m",
        f"steps:      {steps}",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (px0 + 22, py0 + 74 + i * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Fancy rollout — search-then-goto with BEV follow-cam + overlays
# ---------------------------------------------------------------------------

def run_fancy_rollout(
    inf: "Inferencer",            # Inferencer instance (goal_source='classical')
    scene_cfg: dict,
    prompt: str,
    goto_ckpt_path: str = GOTO_CKPT_DEFAULT,
    maxsteps: int = MAXSTEPS_FANCY,
    render_video: bool = True,
    video_path: Optional[str] = None,
    # called each step with (sbs_bgr, state, dist, step)
    frame_callback: "Callable[..., None] | None" = None,
    # FD2: multi-goal context
    goal_idx: int = 0,
    n_goals: int = 1,
    path_trail_in: Optional[List[np.ndarray]] = None,  # carry trail from prior sub-goals
    completed_targets: Optional[List[np.ndarray]] = None,  # already-reached targets
    # VF-1 item 5: title card scenario name (CLI: --scenario-title). Only ever
    # rendered when goal_idx == 0 (once per overall episode, not per sub-goal).
    scenario_title: str = "G1Nav Autonomous Fetch",
    # VF-1 item 5: full instruction text for the title card, if different from
    # the per-sub-goal `prompt` (e.g. multi-goal's combined instruction).
    # Defaults to `prompt` when not given.
    title_instruction: Optional[str] = None,
    # VF-3 (docs/vf3_bev_fixes.md, user feedback #3): optional continuation
    # context, used ONLY by run_fancy_rollout_multi's sequential sub-goals.
    # When given, this call reuses the SAME MuJoCo model/data/teacher/renderer
    # and the policy's own carried-forward state (proprio history, gait-phase
    # tracker, last commanded action) instead of building a fresh arena and
    # resetting qpos/qvel back to scene_cfg['robot_xy']/['robot_yaw'] (the
    # ORIGINAL episode start) -- i.e. genuine continuous physics across
    # sub-goals, not a same-looking reset. Every existing single-goal caller
    # leaves this None and gets EXACTLY the prior build-fresh-arena-and-settle
    # behavior (see docs/vf3_bev_fixes.md's invariance check).
    resume_ctx: Optional[dict] = None,
    # When True, don't close the renderer / tear down the sim at the end of
    # this call -- instead return the live objects + carried policy state in
    # the result dict under 'live_ctx', for the caller to feed into the NEXT
    # sub-goal's `resume_ctx`. Only ever set by run_fancy_rollout_multi.
    keep_alive: bool = False,
) -> dict:
    """Search-then-goto rollout with ego|BEV side-by-side frames + 4 overlays.

    Always uses SEARCH behavior (student-driven bidirectional bounded scan,
    code/scan_sched.py, until target spotted — see docs/nx1_scan.md).

    Args:
        inf: Inferencer instance (goal_source='classical').
        scene_cfg: Scene config dict (objects, target_index, robot_xy/yaw,
            stop_r, etc.) as produced by the sample_fancy_scene* functions.
        prompt: Instruction text shown in the BEV status banner / HUD bar.
        goto_ckpt_path: Goto/search checkpoint path (currently unused inside
            this function -- the loaded policy comes from `inf`).
        maxsteps: Hard step cap for this sub-goal.
        render_video: Whether to render ego|BEV SBS frames at all.
        video_path: Output MP4 path for this rollout's own clip, or None to
            skip writing a per-call video (e.g. when the caller collects
            frames itself, as run_fancy_rollout_multi does).
        frame_callback: Optional callable invoked each rendered step with
            (sbs_bgr, state, dist, step).
        goal_idx: Zero-based index of this sub-goal (multi-goal runs).
        n_goals: Total number of sub-goals in this episode.
        path_trail_in: Path trail carried over from prior sub-goals, or None
            to start a fresh trail.
        completed_targets: World (x, y) positions of already-reached targets,
            carried over from prior sub-goals.
        scenario_title: Scenario name shown on the VF-1 title card (rendered
            only when goal_idx == 0).
        title_instruction: Full instruction text for the title card, if
            different from the per-sub-goal `prompt` (e.g. multi-goal's
            combined instruction); defaults to `prompt` when not given.
        resume_ctx: Optional continuation context (live MuJoCo model/data/
            teacher/renderer + carried policy state) from a prior sub-goal's
            `live_ctx`, used only by run_fancy_rollout_multi. None rebuilds a
            fresh arena and resets to scene_cfg's start state, as before VF-3.
        keep_alive: When True, don't close the renderer / tear down the sim
            at the end of this call -- instead return the live objects in the
            result dict under 'live_ctx' for the next sub-goal's `resume_ctx`.

    Returns:
        Dict with keys: success, spotted, scan_steps, failure_tag, steps,
        final_dist, fell, ms_per_step, video_path, frames_count,
        path_trail_out, frames_sbs, avoid_bias_active_frac, and (only when
        `keep_alive` and the robot didn't fall) live_ctx.
    """
    import cv2
    import mujoco
    import torch
    import math as _math
    from code.inferencer import (
        _build_proprio, _apply_student_pd, _GaitPhaseTracker, _label_active_cam,
        FALL_HEIGHT, GROUNDING_PERIOD, HOLD_STEPS_REQUIRED, ACTION_SCALE,
        PROPRIO_K, PROPRIO_DIM, PROPRIO_DIM_PHASE, IMG_SIZE,
    )
    from code.arena import build_arena, ArenaRenderer, GROUNDING_W, GROUNDING_H
    from code.teacher import (WBCTeacher, _yaw_of, DEFAULT_ANGLES, KPS, KDS,
                               NUM_ACTIONS, SIM_DT, CONTROL_DECIMATION)
    from code.grounding import (ground as classical_ground, get_ego_intrinsics_rendered,
                                get_ground_net_last_heatmap)
    from code.steer import steer as _steer_cmd
    from code.eval_search import STOP_R_SEARCH, SCAN_ALIGNED_THR_DEG
    from code.scan_sched import (BidirectionalScanSchedule, SCAN_LEG_DEG,
                                  SCAN_DWELL_STEPS, SCAN_TIMEOUT as _SCAN_TIMEOUT_DEFAULT)
    from code.lock_mgmt import ReacquisitionScan
    from code import avoid as _avoid
    from code.arena import CAM_HEAD_Z

    # --- Extract scene info ---
    objects      = scene_cfg['objects']
    target_idx   = scene_cfg['target_index']
    target_obj   = objects[target_idx]
    target_xy    = np.array([target_obj['x'], target_obj['y']], dtype=np.float64)
    target_color = target_obj['color_name']
    target_shape = target_obj['shape_name']
    stop_r       = float(scene_cfg.get('stop_r', STOP_R_SEARCH))

    if resume_ctx is None:
        # --- Build MuJoCo env ---
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

        # --- Single renderer (anti-EGL-exhaustion: reuse one renderer for all views) ---
        # ego: 320x240 @32°, grounding: 480x360 @26°, proximity: 320x240 @58°, BEV: 640x480 free cam
        # Use separate Renderer objects but all from the same model
        renderer    = ArenaRenderer(model_mj, tp_w=BEV_W, tp_h=BEV_H)
        # NOTE: intrinsics now come dynamically from whichever camera the CAM-2
        # Schmitt-trigger handoff selects each cycle (render_grounding()/render_proximity()
        # each return their own correct (dims, pitch_deg, is_proximity) intrinsics dict) —
        # mirrors code/inferencer.py's adopted CAM-2 champion (docs/cam_p1.md).

        # BEV follow-cam (elevated diagonal)
        bev_cam = mujoco.MjvCamera()
        bev_cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        bev_cam.distance  = BEV_DISTANCE
        bev_cam.azimuth   = BEV_AZIMUTH
        bev_cam.elevation = BEV_ELEVATION
        bev_cam.lookat[:] = [rx, ry, BEV_LOOKAT_Z]

        # --- Settle (keyframe or WBC fallback) ---
        kf = getattr(inf, '_keyframe', None)
        if kf is not None:
            kf_qpos = kf['qpos_local'].copy()
            kf_qpos[0] = rx
            kf_qpos[1] = ry
            kf_qpos[3] = _math.cos(robot_yaw / 2)
            kf_qpos[4] = 0.0
            kf_qpos[5] = 0.0
            kf_qpos[6] = _math.sin(robot_yaw / 2)
            data_mj.qpos[:len(kf_qpos)] = kf_qpos
            data_mj.qvel[:len(kf['qvel_local'])] = kf['qvel_local']
            mujoco.mj_forward(model_mj, data_mj)
            teacher._target_dof = kf['target_dof'].copy()
        else:
            for _ in range(80):
                teacher.step(vel_cmd=(0.0, 0.0, 0.0))

        if teacher.base_height < FALL_HEIGHT:
            renderer.close()
            return dict(success=False, spotted=False, scan_steps=0, failure_tag='fall',
                        steps=0, final_dist=float(np.linalg.norm(data_mj.qpos[0:2] - target_xy)),
                        fell=True, video_path=None)
        if os.environ.get("FANCY_MULTIGOAL_DEBUG"):
            print(f"    [multigoal_dbg] goal_idx={goal_idx} FRESH BUILD, robot_xy=({rx:.3f},{ry:.3f}) "
                  f"yaw={robot_yaw:.3f}rad (scene_cfg['robot_xy']={scene_cfg.get('robot_xy')})",
                  flush=True)
    else:
        # VF-3 (docs/vf3_bev_fixes.md, user feedback #3): continue the SAME
        # MuJoCo sim from the previous sub-goal's live end state -- no
        # build_arena(), no teacher.reset(), no keyframe re-settle. The
        # robot's actual qpos/qvel (position, heading, joint angles,
        # velocities) at the moment the previous sub-goal ended IS the
        # starting state of this one. Scene geometry is identical across
        # sub-goals anyway (run_fancy_rollout_multi's `sub_scene` only ever
        # changes `target_index`), so nothing here needs rebuilding.
        teacher  = resume_ctx['teacher']
        data_mj  = resume_ctx['data_mj']
        model_mj = resume_ctx['model_mj']
        nj       = resume_ctx['nj']
        renderer = resume_ctx['renderer']
        bev_cam  = resume_ctx['bev_cam']
        # `rx, ry` feed path_trail's/telemetry's start-point below -- use the
        # robot's CURRENT actual position (continuous from the prior
        # sub-goal), not scene_cfg['robot_xy'] (the original episode start —
        # exactly the bug being fixed here).
        rx, ry = float(data_mj.qpos[0]), float(data_mj.qpos[1])
        if os.environ.get("FANCY_MULTIGOAL_DEBUG"):
            print(f"    [multigoal_dbg] goal_idx={goal_idx} RESUMING sim at "
                  f"robot_xy=({rx:.3f},{ry:.3f}) yaw={_yaw_of(data_mj.qpos[3:7]):.3f}rad "
                  f"(continuing from prior sub-goal's live end state, "
                  f"scene_cfg start was {scene_cfg.get('robot_xy')})", flush=True)

    # --- Load action stats from inferencer ---
    _use_residual = (getattr(inf, '_action_stats', None) is not None)
    if _use_residual:
        _as       = inf._action_stats
        _da_mean  = _as['mean']
        _da_std   = _as['std']
        _da_deflt = _as['default_angles']

    _use_phase = getattr(inf, '_use_phase', False)
    _phase_tracker = (resume_ctx['phase_tracker'] if resume_ctx is not None
                      else (_GaitPhaseTracker() if _use_phase else None))
    _eff_pdim = PROPRIO_DIM_PHASE if _use_phase else PROPRIO_DIM

    # --- State ---
    if resume_ctx is not None:
        # Carry the policy's own recurrent-ish state forward too (last
        # commanded joint targets + the K-step proprio history window) so the
        # first few control cycles of the new sub-goal aren't fed a
        # discontinuous/zeroed history -- genuine continuity, not just a
        # matching (x,y,yaw).
        prev_action  = resume_ctx['prev_action']
        proprio_hist = resume_ctx['proprio_hist']
    else:
        prev_action  = teacher._target_dof.copy()
        proprio_hist = collections.deque(
            [np.zeros(_eff_pdim, dtype=np.float32)] * PROPRIO_K, maxlen=PROPRIO_K
        )
        prop_now = _build_proprio(data_mj, prev_action)
        if _use_phase:
            ph = _phase_tracker.update(data_mj.qpos[7:22].copy())
            prop_now = np.concatenate([prop_now, ph])
        for _ in range(PROPRIO_K):
            proprio_hist.append(prop_now.copy())

    lang_t = torch.zeros(1, 2048, device=inf.device)

    # Scan state — NX-1 bidirectional bounded-rotation sweep (same schedule as
    # code/eval_search.py, docs/nx1_scan.md); replaces the old fixed-CCW-only
    # scan (up to 600 continuous steps) that was the diagnosed root cause of
    # the search skill's falls (docs/fa1_failures.md #1 fix). See
    # code/scan_sched.py for the derivation.
    cached_goal_vec  = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    last_grounding_step = -999
    _scan_active     = True
    SCAN_TIMEOUT     = _SCAN_TIMEOUT_DEFAULT   # 900: safety-net cap
    SCAN_RATE        = 0.6
    SCAN_DT          = SIM_DT * CONTROL_DECIMATION
    SCAN_ALIGNED_THR = _math.radians(SCAN_ALIGNED_THR_DEG)
    _goal_ema        = None
    _GOAL_EMA_ALPHA  = 0.4
    _last_known_goal = None
    _frames_since_det = 0
    HOLD_GOAL_HORIZON = 100
    _scan_yaw_delta  = 0.0
    _scan_sched      = BidirectionalScanSchedule(scan_rate=SCAN_RATE,
                                                  leg_deg=SCAN_LEG_DEG,
                                                  dwell_steps=SCAN_DWELL_STEPS)

    # NX-16 (docs/nx16_cone_stall.md): sustained-loss-of-lock recovery.
    # code/inferencer.py / code/eval_search.py already have this exact
    # mechanism (lock_mgmt.py's M5 "coast-expiry -> drop lock + bounded
    # ReacquisitionScan"), but it is default-OFF there (LOCK_M5=0, REJECT-
    # verdicted on the gated eval protocols as a GLOBAL toggle, docs/nx2_iso.md)
    # and this file never imported lock_mgmt at all -- so a detection that is
    # lost for good (never fewer than HOLD_GOAL_HORIZON=100 frames since last
    # detection) left `cached_goal_vec` frozen FOREVER with no recovery path.
    # Root-caused (docs/nx16_cone_stall.md): the GROUND_NET detector's
    # confidence on a `cone` decays steadily as range closes under the
    # (shallow-pitch) GROUNDING camera -- cones are ~1.5-2.3x taller than the
    # other 3 shapes (code/arena.py's two-part cone mesh) and increasingly
    # clip out of frame -- and can drop under GROUND_NET_TAU right at the
    # GROUNDING/PROXIMITY handoff boundary (~1.6-1.8m, just above CAM_D_LO),
    # so the CAM-2 fallback probe never gets an EMA distance to re-trigger
    # the handoff on, and PROXIMITY itself also fails to (re)detect at that
    # range. The image-blind goto policy then keeps consuming the stale,
    # never-updated egocentric (dist,bearing), which is not re-grounded in
    # the robot's actual (moving) pose -- pure open-loop dead reckoning that
    # curves past the true target and settles into a stable orbit (exactly
    # DR-1's "approach to 0.6-0.9m, reverse, rock-stable plateau" signature).
    #
    # Fix, scoped ENTIRELY to this file's own local state (does not read or
    # write LOCK_M5 / lock_mgmt.LockGate, so code/inferencer.py's and
    # code/eval_search.py's default behavior -- and the M5 REJECT verdict --
    # are untouched): once a previously-SPOTTED lock has been missing for
    # more than HOLD_GOAL_HORIZON frames, drop it and re-enter scan mode via
    # a fresh ReacquisitionScan (own local step counter -- safe to start
    # mid-episode, unlike re-arming `_scan_sched`/SCAN_TIMEOUT which are keyed
    # off the episode's absolute step and would time out on the very next
    # cycle). Bounded: if the rescan itself times out without reacquiring,
    # falls back to the default forward-looking goal vector (same fallback
    # the original never-spotted scan timeout already used) rather than
    # freezing again.
    #
    # NX-16 mechanism-test finding: `ReacquisitionScan`'s own built-in bound
    # reuses the shared SCAN_TIMEOUT=1150 (the INITIAL blind-scan's budget,
    # sized for sweeping from a completely unknown bearing). A coast-expiry
    # rescan starts from a MUCH better prior (it was tracking the target right
    # up until the loss), and in gate testing reacquired within ~310-330
    # steps whenever the target was actually re-detectable -- but on one seed
    # where the target sat in a detector blind range no amount of turning
    # could escape (a cone specifically, docs/nx16_cone_stall.md), letting the
    # rescan run for a nearly-full ~1150-step continuous sweep before an
    # eventual (lucky, drift-induced) reacquisition produced an abrupt
    # scan-to-goto transition that ended in a fall on one of two repeated
    # runs (the other repeat instead simply timed out, no fall either way --
    # consistent with this being right at the edge of the policy's competence
    # envelope for an atypically long uninterrupted turn-in-place, not a
    # deterministic bug). Capping the LOCAL rescan budget well below the
    # shared 1150 (NX16_RESCAN_MAX_STEPS, ~2x the observed successful-
    # reacquisition time) keeps the common case (quick re-lock) unaffected
    # while preventing this file's own rescan from ever running long enough
    # to reach that observed instability regime -- falling back to the
    # default goal (same as the pre-existing scan-timeout fallback, proven
    # non-falling in DR-1's original 30+5-episode sweep) instead.
    NX16_RESCAN_MAX_STEPS = 600
    _using_rescan_sched = False
    _rescan_sched        = None
    _rescan_local_steps  = 0

    def _lock_drop_and_rescan() -> None:
        """NX-16: drop a coast-expired lock and re-enter scan via a fresh
        ReacquisitionScan (see the module comment above this closure)."""
        nonlocal _goal_ema, _last_known_goal, _frames_since_det
        nonlocal _scan_active, _using_rescan_sched, _rescan_sched, cached_goal_vec
        nonlocal _avoid_bias_wz, _rescan_local_steps
        _goal_ema               = None
        _last_known_goal        = None
        _frames_since_det       = 0
        _scan_active            = True
        _using_rescan_sched     = True
        _rescan_sched           = ReacquisitionScan(scan_rate=SCAN_RATE)
        _rescan_local_steps     = 0
        cached_goal_vec         = np.array([2.0, 1.0, 0.0], dtype=np.float32)
        _avoid_bias_wz          = 0.0

    # CAM-2 (docs/cam_p1.md, adopted champion): Schmitt-trigger handoff between the
    # GROUNDING camera (26° pitch, far/mid range) and the PROXIMITY camera (58° pitch,
    # ~0.22-1.81m), mirroring code/inferencer.py's main rollout loop exactly so the
    # ego panel shows exactly what's driving detection this cycle (the handoff visible
    # end-to-end, including through the final approach/stop).
    CAM_D_LO         = 1.2     # m — switch GROUNDING->PROXIMITY below this
    CAM_D_HI         = 1.6     # m — switch PROXIMITY->GROUNDING above this
    # CX-3 demo-generation finding: gating the fallback PROBE on CAM_D_HI (the
    # hysteresis threshold, tuned for the reverse PROXIMITY->GROUNDING switch) can
    # deadlock on some approach geometries — the EMA lags a fast monotonic approach
    # (it's a blend of past-higher and current-lower raw distances), so when GROUNDING
    # loses the target just above CAM_D_HI (observed: last EMA~1.70m at true ~1.2m
    # distance), the frozen last-known distance never re-updates (no further detection
    # occurs to refresh it) and the probe gate blocks PROXIMITY forever -> permanent
    # dead-reckoning for the rest of the approach (exactly the failure mode CAM-2 was
    # built to eliminate). Fix: gate the PROBE on the PROXIMITY camera's own physical
    # far limit (d_far~=1.81m, docs/cam_opt2_multicam.md/arena.py PROXIMITY_PITCH=58
    # geometry) instead of CAM_D_HI — still safely excludes genuinely-far detections
    # (e.g. the ep13 blue-ball-at-4.96m regression in docs/cam_p1.md, >>1.81m either
    # way) while covering the EMA-lag margin. Scoped to fancy_demo.py only (this file
    # is not used by the gated eval scripts) — code/inferencer.py's champion numbers
    # (easy 100/demo 66.7/search 80) are untouched.
    CAM_PROXIMITY_D_FAR = 1.81  # m — proximity camera's physical far limit (probe gate)
    _active_cam      = 'GROUNDING'   # default at episode start
    _cam_miss_count  = 0             # consecutive misses on the active camera
    _video_frame_cache = None        # last labeled active-cam frame (RGB, EGO_W x EGO_H)

    # NX-9 AVOID (docs/nx9_avoid.md): same per-episode state / carve-out
    # pattern as code/inferencer.py / code/eval_search.py -- see
    # code/inferencer.py's identical block for the full rationale. This file
    # has no lock_mgmt-driven rescan to reset on, so `_avoid_bias_wz` only
    # ever resets at episode start (its own decay/hysteresis handles the
    # rest, including the one scan->goto transition below).
    _avoid_bias_wz       = 0.0
    _avoid_is_maneuver   = (_avoid.AVOID and _avoid.is_maneuver_scene(scene_cfg))
    _avoid_cycles_total  = 0
    _avoid_cycles_active = 0
    # VF-1 item 2: last obstacle-bias debug dict (compute_obstacle_bias()'s own
    # `info` return) -- pure read target for draw_avoid_overlay(); never fed
    # back into the bias/control computation above.
    _last_avoid_dbg      = None

    spotted     = False
    scan_steps  = 0

    # Video + overlay state
    # FD2: carry path trail across sub-goals for visual continuity
    if path_trail_in is not None:
        path_trail = list(path_trail_in) + [np.array([rx, ry])]
    else:
        path_trail = [np.array([rx, ry])]   # list of (x,y) world pos
    _completed_targets = list(completed_targets) if completed_targets else []
    frames_sbs    = []                      # collected SBS frames for MP4
    step_times    = []
    hold_counter  = 0
    fell          = False
    steps_done    = 0
    current_state = STATE_SEARCHING
    current_dist  = float(np.linalg.norm(data_mj.qpos[0:2] - target_xy))

    TRAIL_SUBSAMPLE = 3    # record every N steps

    # VF-1 telemetry (pure-read/pure-accumulate, never fed back into control):
    #  - dist_traveled_m: running odometry total, for the outro stats card.
    #  - _hud_state: camera-handoff flash countdown for the HUD bar / cam chip
    #    (item 3) -- rendering-only mutable state, never read by any control path.
    # VF-3: carried forward across sub-goals too when resuming, so the outro
    # card's "distance traveled" reflects the WHOLE multi-goal journey, not
    # just the final leg (pure telemetry, same never-fed-back-into-control
    # guarantee as the rest of this accumulator).
    if resume_ctx is not None:
        dist_traveled_m = resume_ctx['dist_traveled_m']
        _prev_rxy_odom  = resume_ctx['prev_rxy_odom']
    else:
        dist_traveled_m = 0.0
        _prev_rxy_odom  = np.array([rx, ry], dtype=np.float64)
    _hud_state = {"prev_cam": None, "flash_frames_left": 0}
    CAM_FLASH_FRAMES = 10

    def _hud_cam_flash_update(active_cam_now: str) -> bool:
        """VF-1 item 3: returns True while a recent GROUNDING<->PROXIMITY handoff
        should still be flashing the camera chip. Render-side only."""
        if _hud_state["prev_cam"] is not None and _hud_state["prev_cam"] != active_cam_now:
            _hud_state["flash_frames_left"] = CAM_FLASH_FRAMES
        _hud_state["prev_cam"] = active_cam_now
        flashing = _hud_state["flash_frames_left"] > 0
        if flashing:
            _hud_state["flash_frames_left"] -= 1
        return flashing

    def _skill_stage_idx() -> int:
        """VF-1 item 3: map the existing state-machine variables to the 5-stage
        breadcrumb SCAN>LOCK>WALK>HANDOFF>REACH. Pure read of _scan_active /
        current_state / _active_cam -- computes a display index only."""
        if _scan_active:
            return 0  # SCAN
        if current_state == STATE_LOCATED:
            return 1  # LOCK
        if current_state == STATE_REACHED:
            return 4  # REACH
        return 3 if _active_cam == 'PROXIMITY' else 2  # HANDOFF vs WALK

    def _update_bev_cam() -> None:
        """Follow robot with BEV camera."""
        bxy = data_mj.qpos[0:2]
        bev_cam.lookat[:] = [bxy[0], bxy[1], BEV_LOOKAT_Z]
        bev_cam.distance  = BEV_DISTANCE
        bev_cam.azimuth   = BEV_AZIMUTH
        bev_cam.elevation = BEV_ELEVATION

    def _render_sbs_frame() -> tuple[np.ndarray, float]:
        """Render ACTIVE-camera ego feed + BEV + overlays → SBS frame."""
        yaw_now = _yaw_of(data_mj.qpos[3:7])
        rxy     = data_mj.qpos[0:2].copy()
        dist    = float(np.linalg.norm(rxy - target_xy))

        # Ego panel = the CAM-2 ACTIVE camera (GROUNDING far / PROXIMITY near) — same
        # camera that is actually driving detection this cycle, so the handoff (and the
        # target staying in-frame down to the stop) is visible in the recorded clip.
        # _video_frame_cache is refreshed on every grounding cycle below (already
        # labeled + resized to EGO_W x EGO_H by _label_active_cam); reused on
        # in-between steps so the video stays at full step-rate without extra renders.
        if _video_frame_cache is not None:
            ego_rgb = _video_frame_cache
        else:
            ego_rgb, _, _ = renderer.render_ego(data_mj, yaw_now, render_depth=False)

        # BEV RGB (640x480 from follow-cam = tp_rend)
        _update_bev_cam()
        bev_raw = renderer.render_tp(data_mj, bev_cam)   # (480, 640, 3) RGB
        bev_bgr = cv2.cvtColor(bev_raw, cv2.COLOR_RGB2BGR)

        # VF-1 item 4: dashed goal-line color = the target's own scene color (BGR).
        _tgt_rgb = target_obj.get('color_rgb')
        target_color_bgr = ((int(_tgt_rgb[2]), int(_tgt_rgb[1]), int(_tgt_rgb[0])
                              ) if _tgt_rgb is not None else None)

        # Draw overlays (FD2: pass goal progress + completed targets; VF-1: pass
        # AVOID bias/info -- pure read of the control loop's own already-computed
        # state -- and the target's color for the dashed goal line)
        bev_bgr = draw_bev_overlays(
            bev_img=bev_bgr,
            path_trail=path_trail,
            target_xy=target_xy,
            robot_xy=rxy,
            robot_yaw=yaw_now,
            bev_cam=bev_cam,
            model=model_mj,
            data=data_mj,
            state=current_state,
            prompt=prompt,
            dist_to_target=dist,
            goal_idx=goal_idx,
            n_goals=n_goals,
            completed_targets=_completed_targets,
            target_color_bgr=target_color_bgr,
            avoid_bias_wz=_avoid_bias_wz,
            avoid_info=_last_avoid_dbg,
        )

        # VF-1 item 1: last cached GROUND_NET heatmap (None when GROUND_NET is
        # off / never fired / query doesn't match this episode's target).
        heatmap_cache = get_ground_net_last_heatmap() if FEAT_HEATMAP else None

        # VF-1 item 3: HUD bar context -- pure reads of state that already
        # exists in this function (proprio-derived speed, geometry-derived
        # bearing, the loop's own step/prompt, the skill-stage mapping above).
        hud_ctx = None
        if FEAT_HUD:
            bearing_deg = _math.degrees(_math.atan2(target_xy[1] - rxy[1],
                                                     target_xy[0] - rxy[0]) - yaw_now)
            bearing_deg = _math.degrees(_math.atan2(_math.sin(_math.radians(bearing_deg)),
                                                     _math.cos(_math.radians(bearing_deg))))
            walk_speed = float(np.linalg.norm(data_mj.qvel[0:2]))
            cam_flash  = _hud_cam_flash_update(_active_cam)
            hud_ctx = dict(
                prompt=prompt, dist=dist, bearing_deg=bearing_deg, step=step,
                walk_speed_mps=walk_speed, stage_idx=_skill_stage_idx(),
                active_cam=_active_cam, cam_flash=cam_flash,
            )

        sbs = compose_sbs_frame(ego_rgb, bev_bgr, current_state, prompt, dist,
                                goal_idx=goal_idx, n_goals=n_goals, active_cam=_active_cam,
                                heatmap_cache=heatmap_cache, target_color=target_color,
                                target_shape=target_shape, hud_ctx=hud_ctx)
        return sbs, dist

    # ------------------------------------------------------------------
    # VF-1 item 5: title card pre-roll (~1.5s @ 25fps), once per overall
    # episode (goal_idx==0 only -- a multi-goal run's later sub-goals don't
    # repeat it). Static frames appended BEFORE the simulation loop below
    # starts -- never interleaved with control, purely additive to the video.
    # ------------------------------------------------------------------
    if render_video and FEAT_TITLECARD and goal_idx == 0:
        N_TITLE_FRAMES = 38   # ~1.5s @ 25fps
        _title_instr = title_instruction if title_instruction is not None else prompt
        for _fi in range(N_TITLE_FRAMES):
            _card = make_title_card(_title_instr, scenario_title, _fi, N_TITLE_FRAMES)
            frames_sbs.append(_card)
            if frame_callback:
                try:
                    frame_callback(_card, STATE_IDLE, None, -1)
                except Exception:
                    pass

    for step in range(maxsteps):
        t0 = time.perf_counter()

        height = float(data_mj.qpos[2])
        if height < FALL_HEIGHT:
            fell = True
            break

        yaw = _yaw_of(data_mj.qpos[3:7])
        rxy = data_mj.qpos[0:2].copy()

        # Update state machine
        dist_now = float(np.linalg.norm(rxy - target_xy))
        current_dist = dist_now
        if _scan_active:
            current_state = STATE_SEARCHING
        elif not spotted:
            current_state = STATE_MOVING  # scan timed out, fallback goto
        elif dist_now < stop_r * 2:
            current_state = STATE_REACHED if dist_now < stop_r else STATE_MOVING
        else:
            current_state = STATE_MOVING

        # Record path trail (subsampled)
        if step % TRAIL_SUBSAMPLE == 0:
            path_trail.append(rxy.copy())
            if len(path_trail) > 200:
                path_trail = path_trail[-200:]

        # Grounding cadence — CAM-2 Schmitt-trigger: render ONLY the currently-active
        # camera (GROUNDING far / PROXIMITY near), mirroring code/inferencer.py's
        # adopted CAM-2 champion (docs/cam_p1.md) exactly, so the ego panel always
        # shows what's actually driving detection this cycle.
        need_grounding = (step - last_grounding_step) >= GROUNDING_PERIOD
        need_render    = render_video or need_grounding

        rgb_ground, depth_ground, intr_active = None, None, None

        if need_render and need_grounding:
            if _active_cam == 'PROXIMITY':
                rgb_ground, depth_ground, intr_active = renderer.render_proximity(
                    data_mj, yaw, render_depth=True)
            else:
                rgb_ground, depth_ground, intr_active = renderer.render_grounding(
                    data_mj, yaw, render_depth=True)
            if render_video:
                _video_frame_cache = _label_active_cam(
                    rgb_ground, _active_cam, float(cached_goal_vec[0]),
                    resize_to=(EGO_W, EGO_H))

        # Classical grounding
        if need_grounding and rgb_ground is not None and depth_ground is not None:
            gr = classical_ground(rgb_ground, depth_ground, target_color, target_shape, intr_active)
            last_grounding_step = step
            if os.environ.get("FANCY_CAM_DEBUG"):
                print(f"    [camdbg] step={step} active={_active_cam} not_vis={gr.not_visible} "
                      f"miss={_cam_miss_count} last_known_d={(_last_known_goal[0] if _last_known_goal is not None else None)}",
                      flush=True)

            # CAM-2 bounded fallback probe (docs/cam_p1.md): after 2 consecutive misses
            # on the active camera, probe the OTHER camera once and adopt its result if
            # it detects. Plausibility-gated — only probe PROXIMITY when the last-known
            # EMA distance says the target could actually be inside its ~0.22-1.81m band
            # (prevents a far-range HSV false-positive from locking into PROXIMITY).
            if gr.not_visible:
                _cam_miss_count += 1
                if _cam_miss_count >= 2:
                    other_cam = 'GROUNDING' if _active_cam == 'PROXIMITY' else 'PROXIMITY'
                    _probe_ok = (other_cam == 'GROUNDING' or
                                 (_last_known_goal is not None and
                                  float(_last_known_goal[0]) <= CAM_PROXIMITY_D_FAR))
                    if _probe_ok:
                        if other_cam == 'PROXIMITY':
                            rgb2, depth2, intr2 = renderer.render_proximity(data_mj, yaw, render_depth=True)
                        else:
                            rgb2, depth2, intr2 = renderer.render_grounding(data_mj, yaw, render_depth=True)
                        gr2 = classical_ground(rgb2, depth2, target_color, target_shape, intr2)
                        if not gr2.not_visible:
                            gr = gr2
                            _active_cam = other_cam
                            _cam_miss_count = 0
                            if render_video:
                                _video_frame_cache = _label_active_cam(
                                    rgb2, _active_cam, float(gr2.goal_vec[0]),
                                    resize_to=(EGO_W, EGO_H))
            else:
                _cam_miss_count = 0

            if not gr.not_visible:
                _frames_since_det = 0
                raw_goal = gr.goal_vec.copy()
                if _goal_ema is None:
                    _goal_ema = raw_goal.copy()
                    _last_known_goal = raw_goal.copy()
                else:
                    _goal_ema = _GOAL_EMA_ALPHA * raw_goal + (1.0 - _GOAL_EMA_ALPHA) * _goal_ema
                    th = _math.atan2(_goal_ema[2], _goal_ema[1])
                    _goal_ema[1] = _math.cos(th)
                    _goal_ema[2] = _math.sin(th)
                    _last_known_goal = _goal_ema.copy()
                cached_goal_vec = _goal_ema.copy()

                # CAM-2 Schmitt-trigger handoff on the EMA'd distance (D_LO/D_HI
                # straddle the dual-visible band, so this flips at most once per
                # approach/retreat, not every cycle).
                _ema_dist = float(_goal_ema[0])
                if _active_cam == 'GROUNDING' and _ema_dist < CAM_D_LO:
                    _active_cam = 'PROXIMITY'
                elif _active_cam == 'PROXIMITY' and _ema_dist > CAM_D_HI:
                    _active_cam = 'GROUNDING'
                if os.environ.get("FANCY_CAM_DEBUG"):
                    print(f"    [camdbg] step={step} DETECTED ema_dist={_ema_dist:.3f} "
                          f"-> active={_active_cam}", flush=True)

                if _scan_active:
                    det_bearing = abs(_math.atan2(_goal_ema[2], _goal_ema[1]))
                    if det_bearing < SCAN_ALIGNED_THR:
                        _scan_active = False
                        spotted = True
                        current_state = STATE_LOCATED
                        print(f"  [fancy] SPOTTED at step={step}  bearing={_math.degrees(det_bearing):.1f}°",
                              flush=True)
            else:
                _frames_since_det += 1
                if _last_known_goal is not None and _frames_since_det <= HOLD_GOAL_HORIZON:
                    cached_goal_vec = _last_known_goal.copy()
                elif (not _scan_active) and _frames_since_det > HOLD_GOAL_HORIZON:
                    # NX-16: coast expired without ever re-detecting -- drop the
                    # stale lock and re-enter scan instead of freezing forever
                    # (see the module comment above `_lock_drop_and_rescan`).
                    print(f"  [fancy] NX-16 lock coast-expired at step={step} "
                          f"(frames_since_det={_frames_since_det}) -> drop+rescan",
                          flush=True)
                    _lock_drop_and_rescan()

            # NX-9 AVOID (docs/nx9_avoid.md): local obstacle avoidance --
            # same mechanism/carve-outs as code/inferencer.py's identical
            # block (shared helper, code/avoid.py), reusing this cycle's
            # already-rendered depth_ground/intr_active (zero extra renders).
            # Never while `_scan_active`; fresh bias only while the goal is
            # fresh (<= AVOID_STALE_MAX_MISSED_CYCLES missed cycles), decay
            # only on a longer stale coast -- see AVOID_STALE_MAX_MISSED_CYCLES'
            # comment in code/avoid.py for the ep14 fall trace behind this.
            if _avoid.AVOID and not _avoid_is_maneuver and not _scan_active:
                _avoid_cycles_total += 1
                if _frames_since_det > _avoid.AVOID_STALE_MAX_MISSED_CYCLES:
                    _avoid_bias_wz = _avoid.decay_bias(_avoid_bias_wz)
                else:
                    _avoid_goal_dist_now = float(cached_goal_vec[0])
                    _avoid_goal_bearing_now = _math.atan2(float(cached_goal_vec[2]),
                                                           float(cached_goal_vec[1]))
                    _avoid_carved = (_avoid_goal_dist_now < _avoid.AVOID_MIN_GOAL_DIST_M)
                    _avoid_cam_h = float(data_mj.qpos[2]) + CAM_HEAD_Z
                    _avoid_bias_wz, _avoid_dbg = _avoid.compute_obstacle_bias(
                        depth_ground, intr_active, cam_height_m=_avoid_cam_h,
                        goal_dist_m=_avoid_goal_dist_now,
                        goal_bearing_rad=_avoid_goal_bearing_now,
                        prev_bias_wz=_avoid_bias_wz, carved_out=_avoid_carved)
                    _last_avoid_dbg = _avoid_dbg   # VF-1: pure-read cache for the BEV viz
                if abs(_avoid_bias_wz) > 1e-9:
                    _avoid_cycles_active += 1

        # Scan mode
        if _scan_active:
            # NX-16: a coast-expiry drop+rescan (_lock_drop_and_rescan) uses a FRESH
            # ReacquisitionScan (its own LOCAL step counter) instead of the initial
            # `_scan_sched`/SCAN_TIMEOUT pair below, because SCAN_TIMEOUT is keyed on
            # the episode's absolute `step` -- re-arming it mid-episode would time out
            # on the very next cycle (step is already >> SCAN_TIMEOUT by then). This
            # branch is only ever taken after a coast-expiry trigger; otherwise
            # `_using_rescan_sched` stays False and the original H3-style scan below
            # (bounded by the absolute-step SCAN_TIMEOUT) runs exactly as before.
            if _using_rescan_sched:
                if _rescan_local_steps >= NX16_RESCAN_MAX_STEPS:
                    scan_wz = None   # NX-16 tighter local cap -- see comment above
                else:
                    scan_wz = _rescan_sched.step(yaw)
                if scan_wz is None:
                    _scan_active        = False
                    _using_rescan_sched = False
                    print(f"  [fancy] NX-16 RESCAN TIMEOUT step={step} "
                          f"(local_steps={_rescan_local_steps}), "
                          f"falling back to default goal", flush=True)
                else:
                    _rescan_local_steps += 1
            elif step >= SCAN_TIMEOUT:
                _scan_active = False
                scan_wz = None
                print(f"  [fancy] SCAN TIMEOUT step={step}", flush=True)
            else:
                scan_wz = _scan_sched.step(yaw)   # bounded CCW/CW schedule, dwells at 0.0

            if scan_wz is not None:
                scan_steps += 1
                _scan_yaw_delta += scan_wz * SCAN_DT

                prop_now = _build_proprio(data_mj, prev_action)
                if _use_phase:
                    ph = _phase_tracker.update(data_mj.qpos[7:22].copy())
                    prop_now = np.concatenate([prop_now, ph])
                proprio_hist.append(prop_now)
                prop_arr = np.stack(list(proprio_hist), axis=0)
                prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(inf.device)

                img_t_scan   = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32, device=inf.device)
                scan_goal_t  = torch.from_numpy(cached_goal_vec).unsqueeze(0).to(inf.device)
                scan_vel_t   = torch.tensor([[0.0, 0.0, scan_wz]], dtype=torch.float32, device=inf.device)

                with torch.no_grad():
                    out_scan = inf.model(
                        ego_rgb   = img_t_scan,
                        lang_emb  = lang_t,
                        proprio_h = prop_t,
                        gt_goal   = scan_goal_t,
                        gt_vel    = scan_vel_t,
                    )

                raw_scan = out_scan['action'].cpu().numpy().squeeze(0)[0]
                if _use_residual:
                    target_dof = _da_deflt + raw_scan * _da_std + _da_mean
                else:
                    target_dof = raw_scan

                for _ in range(CONTROL_DECIMATION):
                    _apply_student_pd(data_mj, target_dof, nj)
                    mujoco.mj_step(model_mj, data_mj)

                prev_action = target_dof.copy()
                steps_done = step + 1

                # VF-1 item 5: odometry accumulator for the outro stats card
                # (pure telemetry read -- data_mj.qpos already updated by the
                # mj_step() calls above; never influences any decision).
                _rxy_now_odom = data_mj.qpos[0:2].copy()
                dist_traveled_m += float(np.linalg.norm(_rxy_now_odom - _prev_rxy_odom))
                _prev_rxy_odom = _rxy_now_odom

                # Render SBS frame for video / stream
                if render_video and _video_frame_cache is not None:
                    try:
                        sbs, dist = _render_sbs_frame()
                        frames_sbs.append(sbs)
                        if frame_callback:
                            frame_callback(sbs, current_state, dist, step)
                    except Exception as e:
                        pass  # non-fatal

                t1 = time.perf_counter()
                step_times.append((t1 - t0) * 1000.0)
                continue

        # Normal GOTO step
        prop_now = _build_proprio(data_mj, prev_action)
        if _use_phase:
            ph = _phase_tracker.update(data_mj.qpos[7:22].copy())
            prop_now = np.concatenate([prop_now, ph])
        proprio_hist.append(prop_now)
        prop_arr = np.stack(list(proprio_hist), axis=0)
        prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(inf.device)

        img_t      = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32, device=inf.device)
        goal_inj_t = torch.from_numpy(cached_goal_vec).unsqueeze(0).to(inf.device)

        # NX-9 AVOID: replace the model's self-predicted velocity with
        # steer.py's own control law (from cached_goal_vec) plus the bounded
        # yaw bias, exactly matching code/inferencer.py's injection point --
        # only when a nonzero bias is active this cycle (provable no-op on
        # clear paths / when AVOID is off).
        gt_vel_t = None
        if _avoid.AVOID and not _avoid_is_maneuver and abs(_avoid_bias_wz) > 1e-9:
            vel_av = _avoid.biased_vel_cmd(
                float(cached_goal_vec[0]), float(cached_goal_vec[1]),
                float(cached_goal_vec[2]), _avoid_bias_wz, stop_r)
            gt_vel_t = torch.from_numpy(vel_av).unsqueeze(0).to(inf.device)

        with torch.no_grad():
            out = inf.model(
                ego_rgb   = img_t,
                lang_emb  = lang_t,
                proprio_h = prop_t,
                gt_goal   = goal_inj_t,
                gt_vel    = gt_vel_t,
            )

        raw_action = out['action'].cpu().numpy().squeeze(0)[0]
        if _use_residual:
            student_dof = _da_deflt + raw_action * _da_std + _da_mean
        else:
            student_dof = raw_action

        for _ in range(CONTROL_DECIMATION):
            _apply_student_pd(data_mj, student_dof, nj)
            mujoco.mj_step(model_mj, data_mj)

        prev_action = student_dof.copy()
        steps_done  = step + 1

        # VF-1 item 5: odometry accumulator for the outro stats card (same
        # pure-telemetry read as the scan branch above).
        _rxy_now_odom = data_mj.qpos[0:2].copy()
        dist_traveled_m += float(np.linalg.norm(_rxy_now_odom - _prev_rxy_odom))
        _prev_rxy_odom = _rxy_now_odom

        if render_video and _video_frame_cache is not None:
            try:
                sbs, dist = _render_sbs_frame()
                frames_sbs.append(sbs)
                if frame_callback:
                    frame_callback(sbs, current_state, dist, step)
            except Exception as e:
                pass

        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1000.0)

        dist_to_target = float(np.linalg.norm(data_mj.qpos[0:2] - target_xy))
        if dist_to_target < stop_r:
            hold_counter += 1
            if hold_counter >= HOLD_STEPS_REQUIRED:
                current_state = STATE_REACHED
                break
        else:
            hold_counter = 0

        if step % 100 == 0:
            print(f"  [fancy] step={step:4d}  dist={dist_to_target:.2f}m  "
                  f"scan={'ON' if _scan_active else 'OFF'}  spotted={spotted}  h={height:.3f}m",
                  flush=True)

    final_height = float(data_mj.qpos[2])
    upright      = final_height >= FALL_HEIGHT and not fell
    final_dist   = float(np.linalg.norm(data_mj.qpos[0:2] - target_xy))
    reached      = (final_dist < stop_r) and upright
    success      = spotted and reached

    if fell:
        failure_tag = 'fall'
    elif not spotted:
        failure_tag = 'scan_timeout'
    elif not reached:
        failure_tag = 'didnt-reach'
    else:
        failure_tag = 'success'

    ms_per_step = float(np.mean(step_times)) if step_times else 0.0
    print(f"  [fancy] DONE: {failure_tag}  final_dist={final_dist:.3f}m  "
          f"steps={steps_done}  ms/step={ms_per_step:.1f}", flush=True)

    # ------------------------------------------------------------------
    # VF-1 item 5: outro stats card (~2s freeze @ 25fps) on a successful
    # REACHED finish -- built from the actual LAST rendered SBS frame (so the
    # scene/robot/target are still visible) + a stats overlay (elapsed sim
    # time, distance traveled, final distance). Only appended once, at the
    # FINAL sub-goal (goal_idx == n_goals-1), so a multi-goal run gets one
    # outro at the very end rather than one per sub-goal.
    # ------------------------------------------------------------------
    if render_video and FEAT_TITLECARD and success and goal_idx == (n_goals - 1) and frames_sbs:
        N_OUTRO_FRAMES = 50   # ~2s @ 25fps
        sim_time_s = steps_done * SIM_DT * CONTROL_DECIMATION
        _outro = make_outro_card(frames_sbs[-1], sim_time_s, dist_traveled_m,
                                 final_dist, steps_done)
        for _ in range(N_OUTRO_FRAMES):
            frames_sbs.append(_outro)
            if frame_callback:
                try:
                    frame_callback(_outro, STATE_REACHED, final_dist, steps_done)
                except Exception:
                    pass

    # Save MP4 in background
    out_vid = None
    if render_video and video_path and frames_sbs:
        out_vid = _write_fancy_video(frames_sbs, video_path)

    # VF-3: only keep the renderer (and the rest of the live sim) open when a
    # caller (run_fancy_rollout_multi) both asked for it AND the robot is
    # still upright -- otherwise close it here exactly as before this fix.
    _continuing = keep_alive and not fell
    if os.environ.get("FANCY_MULTIGOAL_DEBUG"):
        _final_xy = data_mj.qpos[0:2]
        print(f"    [multigoal_dbg] goal_idx={goal_idx} ENDED at "
              f"robot_xy=({_final_xy[0]:.3f},{_final_xy[1]:.3f}) "
              f"yaw={_yaw_of(data_mj.qpos[3:7]):.3f}rad  fell={fell}  "
              f"keep_alive={keep_alive}  continuing={_continuing}", flush=True)
    if not _continuing:
        renderer.close()

    result = dict(
        success=success,
        spotted=spotted,
        scan_steps=scan_steps,
        failure_tag=failure_tag,
        steps=steps_done,
        final_dist=final_dist,
        fell=fell,
        ms_per_step=ms_per_step,
        video_path=out_vid,
        frames_count=len(frames_sbs),
        # FD2: carry path trail forward across sub-goals
        path_trail_out=list(path_trail),
        frames_sbs=frames_sbs,    # returned for multi-goal video concat
        avoid_bias_active_frac=(_avoid_cycles_active / _avoid_cycles_total
                                 if _avoid_cycles_total > 0 else 0.0),
    )
    if _continuing:
        # VF-3 (docs/vf3_bev_fixes.md): hand the LIVE sim + carried policy
        # state back to run_fancy_rollout_multi so the NEXT sub-goal's
        # resume_ctx can continue the SAME simulation (no rebuild, no reset).
        # Withheld if the robot fell -- there is no physically sensible way
        # to "continue" a fallen robot's simulation into the next sub-goal;
        # run_fancy_rollout_multi treats a missing 'live_ctx' as "stop here".
        result['live_ctx'] = dict(
            teacher=teacher, data_mj=data_mj, model_mj=model_mj, nj=nj,
            renderer=renderer, bev_cam=bev_cam,
            prev_action=prev_action, proprio_hist=proprio_hist,
            phase_tracker=_phase_tracker,
            dist_traveled_m=dist_traveled_m, prev_rxy_odom=_prev_rxy_odom,
        )
    return result


# ---------------------------------------------------------------------------
# FD2: Multi-goal rollout — sequential sub-goals with shared BEV view
# ---------------------------------------------------------------------------

# NX-15: Live instruction parsing + scene resolution — the ONE shared function
# used by BOTH _terminal_loop() and the Flask /execute handler (_do_rollout()).
# Fixes docs/dr1_demo_reliability.md's headline finding: previously neither live
# path parsed the typed instruction at all (target always came from
# scene_cfg['target_index']), and this file's only parser, _parse_multi_goal_fancy()
# (below), was dead code never called from any live entry point.
#
# _parse_multi_goal_fancy()'s clause-splitting regex (then / and-then / after-that /
# next) is reused verbatim and kept under its original name/signature for backward
# compat. Its "does this word belong to the known COLORS/SHAPES set" philosophy is
# generalized from an adjacent-pair regex to a whole-clause word scan
# (_extract_goal_hint) so word order ("the ball that is red"), inserted adjectives
# ("the reddish ball" — doesn't false-match "red" thanks to \b), and "-colored"
# phrasing all resolve instead of silently returning []. Ambiguity handling
# (_resolve_goal_to_index) mirrors demo.py's Planner._resolve_referent(): unique
# (color, shape) match -> go; multiple candidates -> score by how many of the
# OTHER words in the clause match each candidate's attributes, tie -> one-line
# clarification question; zero candidates -> "no <X> in this scene" + inventory.

_ALL_COLORS: list[str] = ["red", "yellow", "blue", "green", "orange", "purple", "cyan"]
_ALL_SHAPES: list[str] = RELIABLE_SHAPES  # ["ball", "cube", "cylinder", "cone"] -- full shape set


def _split_multi_goal_parts(instruction: str) -> List[str]:
    """Split a compound instruction on then/and-then/after-that/next conjunctions.
    Same regex as demo.py's Planner.parse() / the original _parse_multi_goal_fancy()."""
    parts = re.split(
        r'\bthen\b|,\s*then\s*|\band\s+then\b|\band\s+after\s+that\b'
        r'|\bafter\s+that\b|\bafterwards\b|\bnext\b',
        instruction, flags=re.IGNORECASE
    )
    return [p.strip() for p in parts if p.strip()]


def _extract_goal_hint(part: str) -> dict:
    """Extract a best-effort (color, shape) hint from one instruction clause.

    Scans the whole clause for known color/shape words (order-independent --
    handles "red ball", "the ball that is red", "red-colored ball", etc.) rather
    than requiring the two words to be adjacent. `color`/`shape` are set only
    when exactly one candidate word of that kind is present in the clause;
    `colors_mentioned`/`shapes_mentioned` keep the full sets for ambiguity
    scoring (see _resolve_goal_to_index).

    Args:
        part: One instruction clause (already split on then/and-then/etc.).

    Returns:
        Dict with keys `color` (str or None), `shape` (str or None),
        `colors_mentioned` (set[str]), `shapes_mentioned` (set[str]), and
        `prompt_part` (the stripped input clause).
    """
    part_l = part.lower()
    colors_mentioned = {c for c in _ALL_COLORS if re.search(r'\b' + c + r'\b', part_l)}
    shapes_mentioned = {s for s in _ALL_SHAPES if re.search(r'\b' + s + r'\b', part_l)}
    color = next(iter(colors_mentioned)) if len(colors_mentioned) == 1 else None
    shape = next(iter(shapes_mentioned)) if len(shapes_mentioned) == 1 else None
    return {
        "color": color, "shape": shape,
        "colors_mentioned": colors_mentioned, "shapes_mentioned": shapes_mentioned,
        "prompt_part": part.strip(),
    }


def _parse_multi_goal_fancy(instruction: str) -> List[dict]:
    """Rule-based multi-goal parser for fancy_demo (kept under its original name and
    signature for backward compat). Splits on "then" conjunctions, extracts
    (color, shape) per part. Returns list of dicts: [{color, shape, prompt_part}, ...]

    NX-15: now implemented on top of _split_multi_goal_parts()/_extract_goal_hint()
    (the shared internals also used by resolve_live_instruction() below) instead of
    its own standalone regex -- same public contract as before.

    Args:
        instruction: Raw typed instruction, possibly compound (e.g. "find the
            red ball then find the yellow cube").

    Returns:
        List of dicts [{color, shape, prompt_part}, ...], one per clause that
        yielded at least one recognized color/shape word.
    """
    goals = []
    for part in _split_multi_goal_parts(instruction):
        hint = _extract_goal_hint(part)
        if hint["color"] or hint["shape"]:
            goals.append({"color": hint["color"], "shape": hint["shape"],
                           "prompt_part": hint["prompt_part"]})
    return goals


def _resolve_goal_to_index(hint: dict, objects: List[dict]) -> tuple:
    """Resolve one (color, shape) hint against the current scene's object list.

    Args:
        hint: One _extract_goal_hint() result (`color`, `shape`,
            `colors_mentioned`, `shapes_mentioned`, `prompt_part`).
        objects: The current scene's object list (`color_name`/`shape_name`/
            `dist_from_robot` per object).

    Returns:
        Tuple (obj_idx, clarify_question):
          (idx, None)   -- unambiguous match (or unique best-attribute-match winner)
          (None, msg)   -- ambiguous, msg is a one-line clarification question
          (None, None)  -- no matching object in the scene
    """
    color, shape = hint["color"], hint["shape"]
    if color is None and shape is None:
        return None, None

    candidates = [
        i for i, o in enumerate(objects)
        if (color is None or o["color_name"] == color)
        and (shape is None or o["shape_name"] == shape)
    ]
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, None

    # Ambiguous (e.g. "the ball" with two balls in the scene): pick the
    # candidate matching more of the OTHER words mentioned in the clause;
    # only ask for clarification if that still leaves a tie.
    colors_m, shapes_m = hint["colors_mentioned"], hint["shapes_mentioned"]
    scored = [
        (int(objects[i]["color_name"] in colors_m) + int(objects[i]["shape_name"] in shapes_m), i)
        for i in candidates
    ]
    best = max(sc for sc, _ in scored)
    tied = [i for sc, i in scored if sc == best]
    if len(tied) == 1:
        return tied[0], None

    descs = ", ".join(
        f"{objects[i]['color_name']} {objects[i]['shape_name']} (at {objects[i]['dist_from_robot']:.1f}m)"
        for i in tied
    )
    return None, f"Multiple matching objects found: {descs}. Which one? (say the color and the shape)"


def resolve_live_instruction(instruction: str, scene_cfg: dict) -> dict:
    """NX-15: THE single shared instruction -> target resolver for both live entry
    points (_terminal_loop, Flask /execute). Never used by the scripted/headless
    entry points (run_smoke(), showcase/recording APIs), which continue to pass
    explicit scene_cfg['target_index'] values untouched -- that default remains
    ONLY the fallback for entry points that explicitly pass an index.

    Args:
        instruction: Raw typed instruction, possibly compound.
        scene_cfg: Current scene config dict (must contain `objects`).

    Returns:
        Dict with keys:
          mode:            "single" | "multi" | "clarify" | "no_match" | "no_parse"
          target_indices:  list[int]   resolved object indices, in goal order
          goals:           list[{"color","shape","prompt_part"}]  resolved goal specs
                           (color/shape are the ACTUAL matched object's attributes,
                           not just the raw parsed hint)
          message:         str or None  (clarify question / no-match / no-parse text)
    """
    objects = (scene_cfg or {}).get("objects", [])
    if not objects:
        return dict(mode="no_match", target_indices=[], goals=[],
                     message="No scene loaded yet.")

    parts = _split_multi_goal_parts(instruction)
    if not parts:
        return dict(mode="no_parse", target_indices=[], goals=[], message=(
            "I didn't understand that instruction. Try things like "
            "'find the red ball' or 'go to the orange cube'."
        ))

    hints = [_extract_goal_hint(p) for p in parts]

    for h in hints:
        if h["color"] is None and h["shape"] is None:
            return dict(mode="no_parse", target_indices=[], goals=[], message=(
                f"I didn't understand '{h['prompt_part']}'. Try things like "
                f"'find the red ball' or 'go to the orange cube'."
            ))

    resolved = [_resolve_goal_to_index(h, objects) for h in hints]

    for idx, clarify in resolved:
        if clarify:
            return dict(mode="clarify", target_indices=[], goals=[], message=clarify)

    for (idx, _clarify), h in zip(resolved, hints):
        if idx is None:
            inv = ", ".join(f"{o['color_name']} {o['shape_name']}" for o in objects)
            c   = h["color"] or "?"
            s   = h["shape"] or "object"
            return dict(mode="no_match", target_indices=[], goals=[], message=(
                f"No {c} {s} in this scene; scene has: {inv}"
            ))

    target_indices = [idx for idx, _ in resolved]
    goals = [
        {"color": objects[idx]["color_name"], "shape": objects[idx]["shape_name"],
         "prompt_part": h["prompt_part"]}
        for (idx, _), h in zip(resolved, hints)
    ]
    mode = "multi" if len(target_indices) > 1 else "single"
    return dict(mode=mode, target_indices=target_indices, goals=goals, message=None)


def run_fancy_rollout_multi(
    inf: "Inferencer",
    goals: List[dict],           # [{color, shape, prompt_part}, ...]
    scene_cfg: dict,
    maxsteps: int = MAXSTEPS_FANCY,
    render_video: bool = True,
    video_path: Optional[str] = None,
    frame_callback: "Callable[..., None] | None" = None,
    # VF-1 item 5: title card params, forwarded to the FIRST sub-goal's
    # run_fancy_rollout() call (which is the only one that renders a title card).
    scenario_title: str = "G1Nav Autonomous Fetch",
) -> dict:
    """Execute sequential sub-goals on the SAME scene.

    For each sub-goal:
      - Sets scene target_index to the matching object
      - Runs run_fancy_rollout() with path_trail carried over
      - BEV shows current target ring + completed target dots + goal N/M banner

    Args:
        inf: Inferencer instance (goal_source='classical').
        goals: List of dicts [{color, shape, prompt_part}, ...], in the order
            the sub-goals should be pursued.
        scene_cfg: Scene config dict shared by all sub-goals (only
            `target_index` is overridden per sub-goal).
        maxsteps: Hard step cap forwarded to each sub-goal's run_fancy_rollout().
        render_video: Whether to render ego|BEV SBS frames at all.
        video_path: Output MP4 path for the combined multi-goal video, or None
            to skip writing one.
        frame_callback: Optional callable forwarded to each sub-goal's
            run_fancy_rollout(), invoked each rendered step with
            (sbs_bgr, state, dist, step).
        scenario_title: Scenario name shown on the VF-1 title card, forwarded
            to the FIRST sub-goal's run_fancy_rollout() call.

    Returns:
        Dict with keys: success (overall), n_goals, goal_results (per-goal
        result dicts), total_steps, video_path, frames_count.
    """
    n_goals = len(goals)
    objects = scene_cfg["objects"]

    def _find_obj(color: str, shape: str) -> Optional[int]:
        """Find object index in scene by color+shape (first match)."""
        for i, o in enumerate(objects):
            if o["color_name"] == color and o["shape_name"] == shape:
                return i
        # Fuzzy: color only
        for i, o in enumerate(objects):
            if o["color_name"] == color:
                return i
        return None

    combined_frames = []  # all SBS frames across sub-goals
    path_trail = None
    completed_targets = []
    all_results = []
    overall_success = True
    # VF-3 (docs/vf3_bev_fixes.md, user feedback #3): carries the LIVE MuJoCo
    # sim (+ carried policy state) from one sub-goal's run_fancy_rollout()
    # call to the next, so the robot's actual physical state (position,
    # heading, joint angles/velocities) continues instead of being reset back
    # to the scene's ORIGINAL start for every sub-goal (the bug being fixed
    # here). None on the very first call (nothing to resume yet).
    live_ctx = None

    for gi, goal in enumerate(goals):
        color = goal["color"]
        shape = goal["shape"]
        prompt_part = goal.get("prompt_part", f"find the {color} {shape}")

        # Override scene target_index for this sub-goal
        obj_idx = _find_obj(color, shape)
        if obj_idx is None:
            print(f"  [multi] sub-goal {gi+1}/{n_goals}: '{color} {shape}' NOT in scene — SKIP", flush=True)
            all_results.append({"success": False, "failure_tag": "not_in_scene", "steps": 0})
            overall_success = False
            continue

        sub_scene = dict(scene_cfg)
        sub_scene["target_index"] = obj_idx
        tgt_obj = objects[obj_idx]
        tgt_xy = np.array([tgt_obj["x"], tgt_obj["y"]])

        print(f"\n  [multi] sub-goal {gi+1}/{n_goals}: '{color} {shape}' at "
              f"dist={tgt_obj['dist_from_robot']:.2f}m", flush=True)

        # Video path for this sub-goal clip (no write if part of multi)
        sub_vid_path = None  # we collect frames, write combined video later

        # VF-3: only the LAST sub-goal lets run_fancy_rollout tear down its
        # own sim/renderer as before -- every earlier sub-goal keeps it alive
        # so the NEXT one can resume from it (scene objects are untouched
        # across sub-goals -- `sub_scene` only ever changes `target_index` --
        # so goal-1's target stays exactly where it was; only the robot's
        # physical state and the goal query change).
        is_last_goal = (gi == n_goals - 1)
        result = run_fancy_rollout(
            inf=inf,
            scene_cfg=sub_scene,
            prompt=f"[{gi+1}/{n_goals}] {prompt_part}",
            maxsteps=maxsteps,
            render_video=render_video,
            video_path=None,   # don't save sub-clip yet
            frame_callback=frame_callback,
            goal_idx=gi,
            n_goals=n_goals,
            path_trail_in=path_trail,
            completed_targets=completed_targets,
            scenario_title=scenario_title,
            title_instruction=" then ".join(g.get("prompt_part", f"{g['color']} {g['shape']}") for g in goals),
            resume_ctx=live_ctx,
            keep_alive=(not is_last_goal),
        )

        # Carry trail forward
        path_trail = result.get("path_trail_out", path_trail)

        # Accumulate frames
        if render_video:
            combined_frames.extend(result.get("frames_sbs", []))

        # Mark completed
        if result.get("success"):
            completed_targets.append(tgt_xy.copy())
        else:
            overall_success = False

        all_results.append(result)
        print(f"  [multi] sub-goal {gi+1}/{n_goals} => {result.get('failure_tag')}  "
              f"dist={result.get('final_dist',0):.3f}m", flush=True)

        # VF-3: hand the live sim to the next sub-goal. A missing 'live_ctx'
        # on a non-last goal means run_fancy_rollout couldn't (or the robot
        # fell and there's nothing physically sensible to continue) -- stop
        # the sequence honestly rather than silently rebuilding a fresh scene
        # for the remaining goals (which would reintroduce the exact
        # teleport-back-to-start bug this fix addresses).
        if is_last_goal:
            live_ctx = None
        else:
            live_ctx = result.get("live_ctx")
            if live_ctx is None:
                print(f"  [multi] sub-goal {gi+1}/{n_goals} ended with no continuable "
                      f"sim state (failure_tag={result.get('failure_tag')}, "
                      f"fell={result.get('fell')}) — stopping multi-goal sequence",
                      flush=True)
                overall_success = False
                break

    # VF-3: defensive cleanup -- if the loop ended (break, or the true last
    # goal was skipped via `continue` above) while a live sim was still open,
    # close it here so its EGL renderer doesn't leak.
    if live_ctx is not None:
        try:
            live_ctx['renderer'].close()
        except Exception:
            pass

    # Write combined video
    out_vid = None
    if render_video and video_path and combined_frames:
        out_vid = _write_fancy_video(combined_frames, video_path)

    return dict(
        success=overall_success,
        n_goals=n_goals,
        goal_results=all_results,
        total_steps=sum(r.get("steps", 0) for r in all_results),
        video_path=out_vid,
        frames_count=len(combined_frames),
    )


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

def _write_fancy_video(frames: list[np.ndarray], path: str, fps: int = 25) -> str:
    """Write ego|BEV SBS frames to MP4. Returns path."""
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not frames:
        return path
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"  [fancy] Video saved: {path}  ({len(frames)} frames, {len(frames)/fps:.1f}s)", flush=True)
    return path


def _concat_reel(video_paths: list[str], reel_path: str) -> Optional[str]:
    """Concatenate multiple MP4s into a showcase reel."""
    import cv2
    valid = [p for p in video_paths if p and os.path.isfile(p)]
    if not valid:
        return None

    # Read first frame for size
    cap0 = cv2.VideoCapture(valid[0])
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap0.get(cv2.CAP_PROP_FPS)) or 25
    cap0.release()

    os.makedirs(os.path.dirname(os.path.abspath(reel_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(reel_path, fourcc, fps, (w, h))

    for vp in valid:
        cap = cv2.VideoCapture(vp)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h))
            out.write(frame)
        cap.release()

    out.release()
    print(f"  [fancy] Reel saved: {reel_path}  ({len(valid)} episodes)", flush=True)
    return reel_path


# ---------------------------------------------------------------------------
# Reliable-color search scene sampler
# ---------------------------------------------------------------------------

def sample_fancy_scene(rng: np.random.Generator, ep_idx: int) -> dict:
    """Sample a search scene with:
    - Target OUTSIDE initial FOV (bearing > 45° from robot_yaw=0)
    - RELIABLE colors biased: orange, red, yellow, purple (avoid cyan/blue HSV wall collision)
    - Distance 2-4m (easy enough to reach but shows search phase)
    - Multiple objects placed non-overlapping

    Args:
        rng: NumPy random generator used for all sampling.
        ep_idx: Episode index (unused directly here; kept for interface
            parity with the other sample_fancy_* functions).

    Returns:
        scene_cfg dict compatible with run_fancy_rollout.
    """
    from code.arena import COLORS, SHAPES
    from code.eval_search import SEARCH_FOV_HALF_DEG, SEARCH_DIST_MIN, SEARCH_DIST_MAX

    arena_half = 4.0
    margin     = 0.55

    rx = float(rng.uniform(-0.3, 0.3))
    ry = float(rng.uniform(-0.3, 0.3))
    robot_yaw  = 0.0
    fov_half_rad = math.radians(SEARCH_FOV_HALF_DEG)

    # Build color lookup
    color_name_to_rgb = {name: rgb for name, rgb in COLORS}

    # Bias target toward reliable colors
    reliable_idxs = [i for i, (cname, _) in enumerate(COLORS)
                     if cname in RELIABLE_COLORS]
    shape_idxs    = list(range(len(SHAPES)))

    # Choose 3 (color, shape) combos; first is target, rest are distractors
    # Target: reliable color
    tgt_ci = int(rng.choice(reliable_idxs))
    tgt_si = int(rng.choice(shape_idxs))

    # Distractors: any color/shape but avoid same as target
    combos = [(ci, si) for ci in range(len(COLORS)) for si in range(len(SHAPES))
              if (ci, si) != (tgt_ci, tgt_si)]
    d_indices = rng.choice(len(combos), size=2, replace=False)
    chosen_combos = [(tgt_ci, tgt_si)] + [combos[k] for k in d_indices]

    objects = []
    for local_i, (ci, si) in enumerate(chosen_combos):
        color_name, color_rgb = COLORS[ci]
        shape_name, size      = SHAPES[si]
        size_val = float(size)
        is_target = (local_i == 0)

        placed = False
        for _ in range(5000):
            if is_target:
                d     = float(rng.uniform(2.0, 4.0))
                side  = rng.integers(2)
                if side == 0:
                    angle = float(rng.uniform(robot_yaw + fov_half_rad, robot_yaw + math.pi))
                else:
                    angle = float(rng.uniform(robot_yaw - math.pi, robot_yaw - fov_half_rad))
            else:
                d     = float(rng.uniform(1.0, 3.5))
                angle = float(rng.uniform(-math.pi, math.pi))

            ox = rx + d * math.cos(angle)
            oy = ry + d * math.sin(angle)

            if abs(ox) + size_val / 2 + margin >= arena_half:
                continue
            if abs(oy) + size_val / 2 + margin >= arena_half:
                continue
            if any(math.hypot(ox - o["x"], oy - o["y"]) < 0.8 for o in objects):
                continue

            if is_target:
                dx, dy   = ox - rx, oy - ry
                obj_angle = math.atan2(dy, dx)
                err = math.atan2(math.sin(obj_angle - robot_yaw), math.cos(obj_angle - robot_yaw))
                if abs(err) <= fov_half_rad:
                    continue

            objects.append({
                "color_name": color_name,
                "color_rgb":  color_rgb,
                "shape_name": shape_name,
                "size":       size_val,
                "x":          float(ox),
                "y":          float(oy),
                "dist_from_robot": float(math.hypot(ox - rx, oy - ry)),
            })
            placed = True
            break

        if not placed:
            # Fallback
            for _ in range(10000):
                if is_target:
                    d = float(rng.uniform(2.0, 4.0))
                    side = rng.integers(2)
                    angle = (float(rng.uniform(robot_yaw + fov_half_rad, robot_yaw + math.pi))
                             if side == 0
                             else float(rng.uniform(robot_yaw - math.pi, robot_yaw - fov_half_rad)))
                    ox = rx + d * math.cos(angle)
                    oy = ry + d * math.sin(angle)
                else:
                    ox = float(rng.uniform(-(arena_half - margin), arena_half - margin))
                    oy = float(rng.uniform(-(arena_half - margin), arena_half - margin))

                if abs(ox) + 0.5 + margin < arena_half and abs(oy) + 0.5 + margin < arena_half:
                    if not any(math.hypot(ox - o["x"], oy - o["y"]) < 0.5 for o in objects):
                        objects.append({
                            "color_name": color_name,
                            "color_rgb":  color_rgb,
                            "shape_name": shape_name,
                            "size":       size_val,
                            "x":          float(ox),
                            "y":          float(oy),
                            "dist_from_robot": float(math.hypot(ox - rx, oy - ry)),
                        })
                        break

    tgt = objects[0]
    dx, dy = tgt["x"] - rx, tgt["y"] - ry
    init_bearing_deg = abs(math.degrees(
        math.atan2(math.sin(math.atan2(dy, dx) - robot_yaw),
                   math.cos(math.atan2(dy, dx) - robot_yaw))
    ))

    return {
        "arena_size":       arena_half,
        "robot_xy":         (rx, ry),
        "robot_yaw":        robot_yaw,
        "objects":          objects,
        "target_index":     0,
        "stop_r":           0.5,
        "horizon":          MAXSTEPS_FANCY,
        "lighting":         {"ambient": 0.45},
        "difficulty":       "search",
        "init_bearing_deg": init_bearing_deg,
    }


# ---------------------------------------------------------------------------
# VF-5 (docs/vf5_cam_objects.md, user feedback #2 "add more objects in the
# fields... at least 7 objects"): the fancy-demo scenes (single-goal AND
# multi-goal) now place >=7 objects total instead of the old 3 -- richer,
# more populated demo scenes. This section ONLY affects the fancy-demo
# samplers below (sample_fancy_scene_long, sample_fancy_multi_goal_scene) --
# code/scene.py (the FROZEN eval-benchmark sampler used by eval_closedloop/
# eval_search/eval_maneuver) is untouched.
#
# Placement rules applied to EVERY object (target/goals AND distractors):
#   - >=FANCY_OBJ_MIN_SEP_M between any two object centers
#   - >=FANCY_OBJ_WALL_MARGIN_M clearance from an object's edge to the wall
#   - >=FANCY_OBJ_MIN_ROBOT_M from the robot spawn point
#   - the target/each goal keeps its EXISTING distance-band + out-of-FOV logic
#     (unchanged -- this VF-5 pass only touches distractor count/placement and
#     tightens the shared wall/separation margins slightly)
# At least one same-color/different-shape pair is guaranteed in every sampled
# scene (nice demo content -- shows the detector discriminates by shape, not
# just color; grounding always queries a specific color+shape combo, so a
# same-color/different-shape distractor does not by itself create the
# same-color/SAME-shape "false lock" risk documented in
# docs/gen1_multiseed.md §3.3 -- that risk requires an exact combo match,
# which the guarantee below explicitly avoids by construction).
# Determinism is preserved: both helpers below are pure functions of the
# `rng` passed in (same np.random.Generator state -> same output), matching
# every other sampler in this file.
FANCY_MIN_OBJECTS       = 7      # target/goal(s) + distractors, total >= this
FANCY_OBJ_MIN_SEP_M     = 1.2    # min center-to-center distance between any two objects
FANCY_OBJ_WALL_MARGIN_M = 0.8    # min clearance from an object's edge to the wall
FANCY_OBJ_MIN_ROBOT_M   = 1.0    # min distance from any object to the robot spawn point


def _place_fancy_object_xy(
    rng: np.random.Generator,
    rx: float, ry: float, robot_yaw: float,
    arena_half: float,
    size_val: float,
    existing: List[dict],
    dist_bounds: Tuple[float, float],
    out_of_fov: bool = False,
    fov_half_rad: float = 0.0,
    min_sep: float = FANCY_OBJ_MIN_SEP_M,
    wall_margin: float = FANCY_OBJ_WALL_MARGIN_M,
    min_robot_dist: float = FANCY_OBJ_MIN_ROBOT_M,
    n_tries: int = 12000,
) -> Tuple[float, float]:
    """Sample one object's (x, y) honoring the VF-5 wall/separation/robot-
    distance rules, within `dist_bounds` of the robot spawn and (optionally)
    outside the robot's initial FOV.

    Three progressively-relaxed passes (mirrors the pre-existing fallback
    pattern in this file's samplers) guarantee placement always succeeds for
    the object counts/arena sizes this demo uses -- the strict rule is tried
    first (`n_tries` draws), then `min_sep`/`wall_margin` are loosened, so a
    late-placed object in a crowded scene never blocks the whole sampler.

    Args:
        rng: NumPy random generator used for all sampling.
        rx, ry: Robot spawn position.
        robot_yaw: Robot spawn yaw, in radians (used only when out_of_fov).
        arena_half: Arena half-size in meters.
        size_val: This object's placement size (diameter-ish), used for wall
            clearance.
        existing: Already-placed object dicts (with 'x'/'y' keys) to avoid.
        dist_bounds: (min, max) distance from the robot spawn point.
        out_of_fov: Whether this object must be outside the robot's initial
            FOV half-angle.
        fov_half_rad: FOV half-angle in radians (used only when out_of_fov).
        min_sep, wall_margin, min_robot_dist: VF-5 placement rules (see
            module-level constants above for the defaults).
        n_tries: Rejection-sampling attempts per relaxation stage.

    Returns:
        (x, y) world position for the new object.
    """
    d_lo, d_hi = dist_bounds
    ox, oy = rx, ry
    for relax in range(3):  # 0 = strict, 1 = relaxed separation, 2 = relaxed wall too
        cur_sep  = min_sep     if relax == 0 else (0.7 if relax == 1 else 0.5)
        cur_wall = wall_margin if relax < 2  else 0.5
        for _ in range(n_tries):
            d = float(rng.uniform(d_lo, d_hi))
            if out_of_fov:
                side = rng.integers(2)
                if side == 0:
                    angle = float(rng.uniform(robot_yaw + fov_half_rad, robot_yaw + math.pi))
                else:
                    angle = float(rng.uniform(robot_yaw - math.pi, robot_yaw - fov_half_rad))
            else:
                angle = float(rng.uniform(-math.pi, math.pi))
            ox = rx + d * math.cos(angle)
            oy = ry + d * math.sin(angle)

            if abs(ox) + size_val / 2 + cur_wall >= arena_half:
                continue
            if abs(oy) + size_val / 2 + cur_wall >= arena_half:
                continue
            if math.hypot(ox - rx, oy - ry) < min_robot_dist:
                continue
            if any(math.hypot(ox - o["x"], oy - o["y"]) < cur_sep for o in existing):
                continue
            return ox, oy
    # Should not happen for this demo's object counts/arena sizes, but never
    # crash the sampler -- fall through with the last candidate tried.
    return ox, oy


def _select_fancy_distractor_combos(
    rng: np.random.Generator,
    primary_combos: List[Tuple[int, int]],
    n_distractors: int,
    n_colors: int,
    n_shapes: int,
) -> List[Tuple[int, int]]:
    """Select `n_distractors` distinct (color_idx, shape_idx) combos, distinct
    from `primary_combos` and from each other, guaranteeing at least one
    selected combo shares its color with `primary_combos[0]` (the main
    target/first goal) while using a different shape -- VF-5's "at least one
    same-color/different-shape pair somewhere in the scene".

    Deterministic given `rng`'s state (same seed -> same combos).

    Args:
        rng: NumPy random generator used for all sampling.
        primary_combos: (color_idx, shape_idx) combos already used by the
            target/goal object(s), to be excluded from the distractor pool.
        n_distractors: Number of distractor combos to select.
        n_colors: Number of colors in the palette (len(COLORS)).
        n_shapes: Number of shapes in the palette (len(SHAPES)).

    Returns:
        List of `n_distractors` distinct (color_idx, shape_idx) combos.
    """
    used = set(primary_combos)
    all_combos = [(ci, si) for ci in range(n_colors) for si in range(n_shapes)]
    remaining = [c for c in all_combos if c not in used]

    chosen: List[Tuple[int, int]] = []
    if n_distractors > 0 and primary_combos:
        base_ci, base_si = primary_combos[0]
        partner_opts = [(ci, si) for (ci, si) in remaining if ci == base_ci and si != base_si]
        if partner_opts:
            k = int(rng.integers(len(partner_opts)))
            partner = partner_opts[k]
            chosen.append(partner)
            remaining.remove(partner)

    n_more = n_distractors - len(chosen)
    if n_more > 0:
        n_more = min(n_more, len(remaining))
        idxs = rng.choice(len(remaining), size=n_more, replace=False)
        chosen.extend(remaining[int(k)] for k in idxs)
    return chosen


# ---------------------------------------------------------------------------
# FD2: Long-distance scene sampler (4-7 m, reliable colors, large arena)
# VF-5: >=7 objects (target + >=6 distractors), see the placement-rules
# comment block above.
# ---------------------------------------------------------------------------

def sample_fancy_scene_long(rng: np.random.Generator, ep_idx: int,
                             dist_min: float = DIST_MIN_LONG,
                             dist_max: float = DIST_MAX_LONG) -> dict:
    """Sample a long-distance search scene:
    - Target OUTSIDE initial FOV (bearing > 45° from robot_yaw=0)
    - RELIABLE color for the target: red, orange, yellow, purple
      (grounding_dist.md: 78% success at 4-9m for non-cyan/blue)
    - Distance 4–7m (impressive walk; arena_size=8m to fit)
    - VF-5: >=7 objects total (target + >=6 distractors, mixed shapes/colors
      from the full palette), non-overlapping, >=1.2m apart, >=0.8m from
      walls, >=1.0m from the robot spawn -- see the placement-rules comment
      block above sample_fancy_scene_long. At least one same-color/
      different-shape pair is guaranteed (nice demo content).
    - Robot near origin (robot_xy ≈ 0)

    FD2: biases toward MEDIUM-LONG distances to make the reel impressive.

    Args:
        rng: NumPy random generator used for all sampling.
        ep_idx: Episode index (unused directly here; kept for interface
            parity with the other sample_fancy_* functions).
        dist_min: Minimum target distance in meters.
        dist_max: Maximum target distance in meters.

    Returns:
        scene_cfg dict compatible with run_fancy_rollout.
    """
    from code.arena import COLORS, SHAPES
    from code.eval_search import SEARCH_FOV_HALF_DEG

    arena_half   = ARENA_HALF_LONG  # 8m — room for 7m targets
    fov_half_rad = math.radians(SEARCH_FOV_HALF_DEG)

    # Robot slightly off-center
    rx = float(rng.uniform(-0.5, 0.5))
    ry = float(rng.uniform(-0.5, 0.5))
    robot_yaw = 0.0

    # Only reliable colors for the target (red, orange, yellow, purple)
    reliable_idxs = [i for i, (cname, _) in enumerate(COLORS)
                     if cname in RELIABLE_COLORS]
    shape_idxs = list(range(len(SHAPES)))

    tgt_ci = int(rng.choice(reliable_idxs))
    tgt_si = int(rng.choice(shape_idxs))

    # VF-5: >=6 distractors (any color/shape from the full palette), one of
    # which is guaranteed to share the target's color with a different shape.
    n_distractors = FANCY_MIN_OBJECTS - 1
    distractor_combos = _select_fancy_distractor_combos(
        rng, [(tgt_ci, tgt_si)], n_distractors, len(COLORS), len(SHAPES))
    chosen_combos = [(tgt_ci, tgt_si)] + distractor_combos

    objects: List[dict] = []
    for local_i, (ci, si) in enumerate(chosen_combos):
        color_name, color_rgb = COLORS[ci]
        shape_name, size      = SHAPES[si]
        size_val = float(size)
        is_target = (local_i == 0)

        if is_target:
            # Long distance: 4–7m, out-of-FOV
            ox, oy = _place_fancy_object_xy(
                rng, rx, ry, robot_yaw, arena_half, size_val, objects,
                dist_bounds=(dist_min, dist_max),
                out_of_fov=True, fov_half_rad=fov_half_rad)
        else:
            # Distractors: 2–5m from robot, anywhere
            ox, oy = _place_fancy_object_xy(
                rng, rx, ry, robot_yaw, arena_half, size_val, objects,
                dist_bounds=(2.0, 5.0))

        objects.append({
            "color_name": color_name,
            "color_rgb":  color_rgb,
            "shape_name": shape_name,
            "size":       size_val,
            "x":          float(ox),
            "y":          float(oy),
            "dist_from_robot": float(math.hypot(ox - rx, oy - ry)),
        })

    tgt = objects[0]
    dx, dy = tgt["x"] - rx, tgt["y"] - ry
    init_bearing_deg = abs(math.degrees(
        math.atan2(math.sin(math.atan2(dy, dx) - robot_yaw),
                   math.cos(math.atan2(dy, dx) - robot_yaw))
    ))

    return {
        "arena_size":       arena_half,
        "robot_xy":         (rx, ry),
        "robot_yaw":        robot_yaw,
        "objects":          objects,
        "target_index":     0,
        "stop_r":           0.5,
        "horizon":          MAXSTEPS_FANCY,
        "lighting":         {"ambient": 0.45},
        "difficulty":       "search_long",
        "init_bearing_deg": init_bearing_deg,
    }


def sample_fancy_multi_goal_scene(rng: np.random.Generator, n_goals: int = 2) -> dict:
    """Sample a scene with n_goals sub-goal objects at varied distances
    (2–6.5m), each a distinct reliable color+shape, PLUS (VF-5) >=5
    additional distractor objects (mixed shapes/colors from the full
    palette) so every multi-goal scene has >=7 objects total. Robot at
    origin, yaw=0.

    All objects (goals AND distractors) honor the VF-5 placement rules
    (>=1.2m apart, >=0.8m from walls, >=1.0m from the robot spawn -- see the
    comment block above sample_fancy_scene_long); at least one same-color/
    different-shape pair is guaranteed.

    Args:
        rng: NumPy random generator used for all sampling.
        n_goals: Number of distinct (color, shape) sub-goal objects to place.

    Returns:
        scene_cfg dict; target_index=0 (first object is 1st sub-goal).
        Multi-goal rollout iterates target_index across 0..n_goals-1.
    """
    from code.arena import COLORS, SHAPES
    from code.eval_search import SEARCH_FOV_HALF_DEG

    arena_half   = ARENA_HALF_LONG
    fov_half_rad = math.radians(SEARCH_FOV_HALF_DEG)

    rx, ry    = 0.0, 0.0
    robot_yaw = 0.0

    reliable_idxs = [i for i, (cname, _) in enumerate(COLORS)
                     if cname in RELIABLE_COLORS]
    shape_idxs = list(range(len(SHAPES)))

    # Pick n_goals distinct (color, shape) combos from reliable colors
    chosen = []
    all_reliable = [(ci, si) for ci in reliable_idxs for si in shape_idxs]
    rng.shuffle(all_reliable)
    for combo in all_reliable:
        if len(chosen) >= n_goals:
            break
        if combo not in chosen:
            chosen.append(combo)
    # Fill with extras if needed
    while len(chosen) < n_goals:
        chosen.append(all_reliable[len(chosen) % len(all_reliable)])

    # VF-5: >=5 additional distractors beyond the n_goals sub-goals (any
    # color/shape from the full palette), one guaranteed to share the first
    # sub-goal's color with a different shape.
    n_others = 5
    other_combos = _select_fancy_distractor_combos(
        rng, chosen, n_others, len(COLORS), len(SHAPES))
    all_combos = chosen + other_combos   # first n_goals entries ARE the sub-goals

    objects: List[dict] = []
    for local_i, (ci, si) in enumerate(all_combos):
        color_name, color_rgb = COLORS[ci]
        shape_name, size      = SHAPES[si]
        size_val = float(size)
        is_first_goal = (local_i == 0)
        is_goal = local_i < n_goals

        # Vary distance: first sub-goal further, later sub-goals medium,
        # extra distractors spread across the same medium band.
        if is_first_goal:
            d_lo, d_hi = 4.5, 6.5   # first goal: long
        else:
            d_lo, d_hi = 2.5, 5.0   # subsequent sub-goals + distractors: medium

        ox, oy = _place_fancy_object_xy(
            rng, rx, ry, robot_yaw, arena_half, size_val, objects,
            dist_bounds=(d_lo, d_hi),
            out_of_fov=is_first_goal, fov_half_rad=fov_half_rad)

        objects.append({
            "color_name": color_name,
            "color_rgb":  color_rgb,
            "shape_name": shape_name,
            "size":       size_val,
            "x":          float(ox),
            "y":          float(oy),
            "dist_from_robot": float(math.hypot(ox - rx, oy - ry)),
            "is_goal":    is_goal,
        })

    tgt = objects[0]
    dx, dy = tgt["x"] - rx, tgt["y"] - ry
    init_bearing_deg = abs(math.degrees(
        math.atan2(math.sin(math.atan2(dy, dx) - robot_yaw),
                   math.cos(math.atan2(dy, dx) - robot_yaw))
    ))

    return {
        "arena_size":       arena_half,
        "robot_xy":         (rx, ry),
        "robot_yaw":        robot_yaw,
        "objects":          objects,
        "target_index":     0,   # multi-goal: executor overrides this per sub-goal
        "stop_r":           0.5,
        "horizon":          MAXSTEPS_FANCY,
        "lighting":         {"ambient": 0.45},
        "difficulty":       "multi_goal",
        "init_bearing_deg": init_bearing_deg,
        "n_goals":          n_goals,
    }


# ---------------------------------------------------------------------------
# Shared stream state (for Flask MJPEG)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Flask Web UI
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Simple SceneManager for fancy demo
# ---------------------------------------------------------------------------
class FancySceneManager:
    """Manages the current fancy search scene."""

    def __init__(self, seed_offset: int = 0) -> None:
        self.seed_offset = seed_offset
        self._ep_count   = 0
        self._scene_cfg  = None

    def new_scene(self, long_dist: bool = True) -> dict:
        """Sample a new scene. FD2: long_dist=True (4-7m) by default.

        FS-1: the very first scene (self._ep_count == 0) draws from the
        curated FIRST_SCENE_SEED instead of the plain [1234+seed_offset, 0]
        sequence, so a fresh --web/terminal launch always opens on a
        verified-good scene. Every later call (manual "New Scene" button,
        the post-rollout auto-resample, terminal 'new') is untouched and
        keeps drawing from the original random sequence -- only this one
        fixed first draw needed curating.

        Args:
            long_dist: Whether to sample from the long-distance (4-7m)
                scene distribution (sample_fancy_scene_long) instead of the
                shorter-range one (sample_fancy_scene).

        Returns:
            The newly sampled scene_cfg dict (also stored on `self._scene_cfg`).
        """
        if self._ep_count == 0:
            seed_seq = np.random.SeedSequence([FIRST_SCENE_SEED, 0])
        else:
            seed_seq = np.random.SeedSequence([1234 + self.seed_offset, self._ep_count])
        rng = np.random.default_rng(seed_seq)
        if long_dist:
            self._scene_cfg = sample_fancy_scene_long(rng, self._ep_count)
        else:
            self._scene_cfg = sample_fancy_scene(rng, self._ep_count)
        self._ep_count  += 1
        tgt = self._scene_cfg['objects'][self._scene_cfg['target_index']]
        print(f"[fancy] New scene ep={self._ep_count-1}: "
              f"target={tgt['color_name']} {tgt['shape_name']}  "
              f"dist={tgt['dist_from_robot']:.2f}m  "
              f"bearing={self._scene_cfg['init_bearing_deg']:.1f}° (out-of-FOV)",
              flush=True)
        return self._scene_cfg

    @property
    def _scene_cfg(self) -> Optional[dict]: return self.__scene_cfg
    @_scene_cfg.setter
    def _scene_cfg(self, v: Optional[dict]) -> None: self.__scene_cfg = v


# ---------------------------------------------------------------------------
# Smoke test (FD2 — long-distance + multi-goal)
# ---------------------------------------------------------------------------
def run_smoke(
    out_dir: str,
    ckpt_path: str,
    device: str = "cpu",
    maxsteps: int = MAXSTEPS_FANCY,
    render_video: bool = True,
    n_episodes: int = 6,
    scenario_title: str = "G1Nav Autonomous Fetch",
) -> tuple[list[dict], Optional[str]]:
    """FD2 Headless smoke: LONG-DISTANCE search episodes + multi-goal, saved as MP4s.

    Episode plan (default n=6):
      ep0: long single-goal search (4-7m)
      ep1: long single-goal search (4-7m)
      ep2: long single-goal search (4-7m) — smoke verify 1st episode
      ep3: long single-goal search (4-7m)
      ep4: long single-goal search (4-7m)
      ep5: MULTI-GOAL (2 sub-goals, different reliable colors)

    Only SUCCESS episodes go into the showcase reel (fail-filtered).

    Args:
        out_dir: Output directory for per-episode MP4s + the showcase reel.
        ckpt_path: Goto/search checkpoint path passed to Inferencer.
        device: Torch device string ("cpu" or "cuda").
        maxsteps: Hard step cap forwarded to each episode's rollout.
        render_video: Whether to render ego|BEV SBS frames at all.
        n_episodes: Number of smoke episodes to run (last one is multi-goal
            when n_episodes >= 2).
        scenario_title: Scenario name shown on each episode's VF-1 title card.

    Returns:
        Tuple (summary, reel_path): `summary` is the list of per-episode
        result dicts; `reel_path` is the showcase reel's MP4 path, or None
        if no episode produced a video.
    """
    from code.inferencer import Inferencer

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}", flush=True)
    print(f"G1Nav Fancy Demo — FD2 SMOKE TEST ({n_episodes} episodes)", flush=True)
    print(f"  ckpt:      {ckpt_path}", flush=True)
    print(f"  device:    {device}", flush=True)
    print(f"  maxsteps:  {maxsteps}", flush=True)
    print(f"  render:    {render_video}", flush=True)
    print(f"  dist bias: {DIST_MIN_LONG}–{DIST_MAX_LONG}m (long-range)", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Load inferencer ONCE — anti-EGL exhaustion: don't recreate per episode
    print("[smoke] Loading inferencer...", flush=True)
    inf = Inferencer(
        checkpoint_path=ckpt_path,
        arch='A',
        device=device,
        goal_source='classical',
        verbose=False,
    )
    print("[smoke] Inferencer ready", flush=True)

    summary   = []
    vid_paths = []   # all episode clips
    ok_vids   = []   # SUCCESS only → for reel

    # Determine which episodes are multi-goal
    # Last episode is multi-goal if n_episodes >= 2
    multi_goal_ep = n_episodes - 1 if n_episodes >= 2 else -1

    rng_master = np.random.default_rng(np.random.SeedSequence([42, 2026]))

    for ep_i in range(n_episodes):
        ep_seed = int(rng_master.integers(0, 2**31))
        rng     = np.random.default_rng(ep_seed)

        is_multi = (ep_i == multi_goal_ep)
        print(f"\n{'='*50}", flush=True)
        print(f"--- FD2 Episode {ep_i+1}/{n_episodes}"
              f"  ({'MULTI-GOAL' if is_multi else 'SINGLE long-dist'}) ---", flush=True)

        if is_multi:
            # ── Multi-goal episode ──
            scene_cfg = sample_fancy_multi_goal_scene(rng, n_goals=2)
            objs = scene_cfg["objects"]
            # Sub-goals: first 2 objects (both reliable color+shape)
            goals = []
            for gi in range(min(2, len(objs))):
                o = objs[gi]
                goals.append({
                    "color":       o["color_name"],
                    "shape":       o["shape_name"],
                    "prompt_part": f"find the {o['color_name']} {o['shape_name']}",
                })
            prompt = " then ".join(g["prompt_part"] for g in goals)
            print(f"  Multi-goal: {prompt}", flush=True)
            for g in goals:
                oi = next((i for i, o in enumerate(objs)
                           if o["color_name"] == g["color"] and o["shape_name"] == g["shape"]), None)
                if oi is not None:
                    print(f"    sub-goal: {g['color']} {g['shape']}  dist={objs[oi]['dist_from_robot']:.2f}m", flush=True)

            vid_path = None
            if render_video:
                vid_path = os.path.join(out_dir, f"ep{ep_i:02d}_multi_goal.mp4")

            t0 = time.time()
            try:
                result = run_fancy_rollout_multi(
                    inf=inf,
                    goals=goals,
                    scene_cfg=scene_cfg,
                    maxsteps=maxsteps,
                    render_video=render_video,
                    video_path=vid_path,
                    scenario_title=scenario_title,
                )
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}", flush=True)
                traceback.print_exc()
                result = {"success": False, "n_goals": 2, "goal_results": [],
                          "total_steps": 0, "video_path": None}

            dt = time.time() - t0
            ok_tag = "SUCCESS" if result.get("success") else "FAILED"
            print(f"  {ok_tag}  total_steps={result.get('total_steps',0)}  wall={dt:.1f}s", flush=True)
            vid_out = result.get("video_path")
            if vid_out:
                print(f"  Video: {vid_out}", flush=True)
                vid_paths.append(vid_out)
                if result.get("success"):
                    ok_vids.append(vid_out)

            # Per-subgoal info for summary
            sub_info = []
            for gi2, sr in enumerate(result.get("goal_results", [])):
                sub_info.append({
                    "goal_idx": gi2,
                    "color":    goals[gi2]["color"] if gi2 < len(goals) else "?",
                    "shape":    goals[gi2]["shape"] if gi2 < len(goals) else "?",
                    "success":  sr.get("success", False),
                    "steps":    sr.get("steps", 0),
                    "final_dist": sr.get("final_dist", 0.0),
                    "spotted":  sr.get("spotted", False),
                })
            summary.append({
                "ep": ep_i,
                "type": "multi_goal",
                "n_goals": result.get("n_goals", 2),
                "prompt": prompt,
                "success": result.get("success", False),
                "total_steps": result.get("total_steps", 0),
                "sub_goals": sub_info,
                "wall_time_s": dt,
                "video_path": vid_out,
            })

        else:
            # ── Single long-distance episode ──
            scene_cfg = sample_fancy_scene_long(rng, ep_i)
            tgt       = scene_cfg["objects"][scene_cfg["target_index"]]
            dist_m    = tgt["dist_from_robot"]
            prompt    = f"find the {tgt['color_name']} {tgt['shape_name']}"
            bearing   = scene_cfg["init_bearing_deg"]
            print(f"  Target: {tgt['color_name']} {tgt['shape_name']}  "
                  f"dist={dist_m:.2f}m  bearing={bearing:.1f}° (out-of-FOV)", flush=True)
            print(f"  Prompt: '{prompt}'", flush=True)

            vid_path = None
            if render_video:
                vid_path = os.path.join(
                    out_dir,
                    f"ep{ep_i:02d}_{tgt['color_name']}_{tgt['shape_name']}_{dist_m:.1f}m.mp4"
                )

            t0 = time.time()
            try:
                result = run_fancy_rollout(
                    inf=inf,
                    scene_cfg=scene_cfg,
                    prompt=prompt,
                    maxsteps=maxsteps,
                    render_video=render_video,
                    video_path=vid_path,
                    scenario_title=scenario_title,
                )
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}", flush=True)
                traceback.print_exc()
                result = {"success": False, "failure_tag": "error",
                          "steps": 0, "final_dist": 999.0,
                          "spotted": False, "scan_steps": 0}

            dt = time.time() - t0
            ok_tag = "SUCCESS" if result.get("success") else f"FAILED({result.get('failure_tag','?')})"
            print(f"  {ok_tag}  steps={result.get('steps',0)}  "
                  f"dist={result.get('final_dist',0):.3f}m  wall={dt:.1f}s  "
                  f"spotted={result.get('spotted',False)}  scan_steps={result.get('scan_steps',0)}", flush=True)
            vid_out = result.get("video_path")
            if vid_out:
                print(f"  Video: {vid_out}", flush=True)
                vid_paths.append(vid_out)
                if result.get("success"):
                    ok_vids.append(vid_out)

            summary.append({
                "ep": ep_i,
                "type": "single_long",
                "prompt": prompt,
                "color": tgt["color_name"],
                "shape": tgt["shape_name"],
                "target_dist_m": dist_m,
                "init_bearing_deg": bearing,
                "success": result.get("success", False),
                "failure_tag": result.get("failure_tag", "?"),
                "steps": result.get("steps", 0),
                "final_dist": result.get("final_dist", 0.0),
                "spotted": result.get("spotted", False),
                "scan_steps": result.get("scan_steps", 0),
                "wall_time_s": dt,
                "video_path": vid_out,
            })

    # ── Showcase reel — SUCCESS episodes only ──
    reel_path = None
    reel_src  = ok_vids if ok_vids else vid_paths   # fall back to all if none succeeded
    if reel_src:
        reel_path = os.path.join(out_dir, "fancy_showcase_reel.mp4")
        reel_path = _concat_reel(reel_src, reel_path)
        print(f"\n[FD2] Showcase reel ({len(reel_src)} clips): {reel_path}", flush=True)

    # ── Print summary table ──
    print(f"\n{'='*60}", flush=True)
    print("FD2 FANCY SMOKE SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    n_ok = sum(1 for s in summary if s["success"])
    print(f"  Success: {n_ok}/{len(summary)}", flush=True)
    for s in summary:
        if s["type"] == "multi_goal":
            ok_str = "OK" if s["success"] else "FAIL"
            print(f"  ep{s['ep']} [MULTI-{s['n_goals']}]: {ok_str:4s}  "
                  f"steps={s['total_steps']:5d}  video={s.get('video_path','none')}", flush=True)
            for sg in s.get("sub_goals", []):
                sg_ok = "OK" if sg["success"] else "FAIL"
                print(f"    sub-goal {sg['goal_idx']+1}: {sg['color']} {sg['shape']}  "
                      f"{sg_ok}  spotted={sg['spotted']}  dist={sg['final_dist']:.3f}m", flush=True)
        else:
            ok_str = "OK" if s["success"] else f"FAIL({s.get('failure_tag','?')})"
            print(f"  ep{s['ep']} [SINGLE  {s.get('target_dist_m',0):.1f}m]: "
                  f"{s['color']:7s} {s['shape']:8s}  {ok_str:20s}  "
                  f"steps={s.get('steps',0):5d}  spotted={s.get('spotted','?')}  "
                  f"video={s.get('video_path','none')}", flush=True)
    if reel_path:
        print(f"\n  Showcase reel: {reel_path}", flush=True)

    # Save summary JSON
    summary_path = os.path.join(out_dir, "fancy_showcase_summary_fd2.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON: {summary_path}", flush=True)

    return summary, reel_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI entry point: dispatches to the headless smoke test, the Flask web
    UI, or the interactive terminal loop, based on the parsed arguments."""
    parser = argparse.ArgumentParser(description="G1Nav Fancy Demo")
    parser.add_argument("--smoke",     action="store_true", help="Headless smoke test")
    parser.add_argument("--web",       action="store_true", help="Flask web UI")
    parser.add_argument("--out",       default=FANCY_OUT_DIR, help="Output dir")
    parser.add_argument("--device",    default="cuda" if _has_cuda() else "cpu")
    parser.add_argument("--ckpt",      default=GOTO_CKPT_DEFAULT,
                        help="Goto/search checkpoint path (default: checkpoint/goto_best.pt)")
    parser.add_argument("--port",      type=int, default=WEB_PORT)
    parser.add_argument("--maxsteps",  type=int, default=MAXSTEPS_FANCY)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--n-smoke",   type=int, default=6,
                        help="Number of smoke episodes (FD2: last ep is multi-goal)")
    parser.add_argument("--scenario-title", default="G1Nav Autonomous Fetch",
                        help="VF-1: scenario name shown on the pre-roll title card")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.smoke:
        run_smoke(
            out_dir=args.out,
            ckpt_path=args.ckpt,
            device=args.device,
            maxsteps=args.maxsteps,
            render_video=not args.no_render,
            n_episodes=args.n_smoke,
            scenario_title=args.scenario_title,
        )
        return

    # Web UI / interactive mode
    from code.inferencer import Inferencer

    print("[fancy_demo] Loading inferencer...", flush=True)
    inf = Inferencer(
        checkpoint_path=args.ckpt,
        arch='A',
        device=args.device,
        goal_source='classical',
        verbose=False,
    )
    print("[fancy_demo] Inferencer ready", flush=True)

    scene_mgr = FancySceneManager(seed_offset=0)
    scene_mgr.new_scene()

    if args.web:
        _start_fancy_web_ui(
            inf=inf,
            scene_manager=scene_mgr,
            out_dir=args.out,
            port=args.port,
            maxsteps=args.maxsteps,
            render_video=not args.no_render,
            scenario_title=args.scenario_title,
        )
        print(f"[fancy_demo] Web UI running at http://localhost:{args.port}", flush=True)
        print("[fancy_demo] Open browser → type 'find the red ball' → watch ego|BEV stream", flush=True)
        print("[fancy_demo] Press Ctrl-C to quit", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[fancy_demo] Shutting down.", flush=True)
    else:
        # Interactive terminal fallback
        _terminal_loop(inf, scene_mgr, args.out, args.maxsteps, not args.no_render,
                       scenario_title=args.scenario_title)


def _terminal_loop(
    inf: "Inferencer",
    scene_mgr: "FancySceneManager",
    out_dir: str,
    maxsteps: int,
    render_video: bool,
    scenario_title: str = "G1Nav Autonomous Fetch",
) -> None:
    """Simple terminal loop."""
    print("\n" + "=" * 60, flush=True)
    print("G1Nav Fancy Demo — Terminal Mode", flush=True)
    print("Name the object you want, e.g. 'find the red ball'.", flush=True)
    print("Multi-goal: 'find the red ball then find the yellow cube'.", flush=True)
    print("Type 'new' / 'quit'", flush=True)
    print("=" * 60 + "\n", flush=True)

    ep_num = 0
    vid_paths = []

    while True:
        scene_cfg = scene_mgr._scene_cfg
        # NX-15: no "<TARGET" marker -- which object gets pursued is now decided
        # by what the user types, not by the sampler's default target_index.
        print(f"Scene objects:", flush=True)
        for i, o in enumerate(scene_cfg['objects']):
            print(f"  [{i}] {o['color_name']} {o['shape_name']}  "
                  f"dist={o['dist_from_robot']:.2f}m", flush=True)

        try:
            user = input("\nfancy> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user:
            continue
        if user.lower() in ('quit', 'exit', 'q'):
            break
        if user.lower() in ('new', 'reset'):
            scene_mgr.new_scene()
            continue

        # NX-15: parse instruction -> resolve against the CURRENT scene's objects
        parsed = resolve_live_instruction(user, scene_cfg)
        if parsed["mode"] in ("no_parse", "no_match", "clarify"):
            print(f"\nBot: {parsed['message']}\n", flush=True)
            continue

        ep_num += 1
        vid_path = None
        if render_video:
            os.makedirs(out_dir, exist_ok=True)
            vid_path = os.path.join(out_dir, f"fancy_ep{ep_num:03d}.mp4")

        tgt_desc = ' then '.join(f"{g['color']} {g['shape']}" for g in parsed["goals"])
        print(f"\nExecuting: '{user}' -> target: {tgt_desc}", flush=True)
        t0 = time.time()
        try:
            if parsed["mode"] == "multi":
                result = run_fancy_rollout_multi(
                    inf=inf,
                    goals=parsed["goals"],
                    scene_cfg=scene_cfg,
                    maxsteps=maxsteps,
                    render_video=render_video,
                    video_path=vid_path,
                    scenario_title=scenario_title,
                )
            else:
                # NX-15: target comes from the resolved instruction; scene_cfg
                # itself is left untouched (a copy carries the override) so the
                # scene manager's own state (incl. its default target_index,
                # unused here) is unaffected.
                resolved_scene = dict(scene_cfg)
                resolved_scene["target_index"] = parsed["target_indices"][0]
                result = run_fancy_rollout(
                    inf=inf,
                    scene_cfg=resolved_scene,
                    prompt=user,
                    maxsteps=maxsteps,
                    render_video=render_video,
                    scenario_title=scenario_title,
                    video_path=vid_path,
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {'success': False, 'failure_tag': 'error'}
        dt = time.time() - t0

        if parsed["mode"] == "multi":
            status = "SUCCESS" if result.get('success') else "FAILED"
            print(f"\nResult: {status}  total_steps={result.get('total_steps',0)}  wall={dt:.1f}s", flush=True)
            for gi, sr in enumerate(result.get("goal_results", [])):
                g_ok = "OK" if sr.get("success") else f"FAIL({sr.get('failure_tag','?')})"
                g = parsed["goals"][gi] if gi < len(parsed["goals"]) else {"color": "?", "shape": "?"}
                print(f"  sub-goal {gi+1}: {g['color']} {g['shape']}  "
                      f"{g_ok}  dist={sr.get('final_dist',0):.3f}m", flush=True)
        else:
            status = "SUCCESS" if result.get('success') else f"FAILED ({result.get('failure_tag')})"
            print(f"\nResult: {status}  steps={result.get('steps',0)}  "
                  f"dist={result.get('final_dist',0):.3f}m  wall={dt:.1f}s", flush=True)

        if result.get('video_path'):
            print(f"Video: {result['video_path']}", flush=True)
            vid_paths.append(result['video_path'])

        scene_mgr.new_scene()

    if len(vid_paths) > 1:
        reel = os.path.join(out_dir, "fancy_reel.mp4")
        _concat_reel(vid_paths, reel)


def _has_cuda() -> bool:
    """Return True if a CUDA device is available to torch, else False."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
