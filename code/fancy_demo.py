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

# BEV follow-camera parameters (45° elevation, diagonal azimuth)
BEV_DISTANCE   = 6.0     # metres from robot
BEV_ELEVATION  = -40.0   # degrees (negative = looking down)
BEV_AZIMUTH    = 225.0   # degrees diagonal (SW view → robot in frame, facing right)
BEV_LOOKAT_Z   = 0.3     # lookat height (ground-level scene)

# Overlay colors (BGR for cv2)
COLOR_PATH_TRAIL   = (0,   220, 100)   # green path line
COLOR_TARGET_RING  = (0,   80,  255)   # bright orange ring
COLOR_FOV_CONE     = (255, 255,  80)   # yellow FOV wedge
COLOR_BANNER_BG    = (30,   30,  30)
COLOR_STATE_SEARCH = (0,   200, 255)   # cyan text — SEARCHING
COLOR_STATE_LOCATE = (50,  255,  50)   # green — LOCATED
COLOR_STATE_MOVE   = (255, 165,   0)   # orange — MOVING
COLOR_STATE_REACH  = (255,  80,  80)   # red-pink — REACHED

# Reliable color palette (avoid cyan/blue — wall HSV collision)
# See docs/grounding_dist.md: red/orange/yellow/purple = 87-100% detection at demo distances
RELIABLE_COLORS = ["red", "orange", "yellow", "purple"]
RELIABLE_SHAPES = ["ball", "cube", "cylinder", "cone"]

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
# Picked by: geometry pre-filter (color in RELIABLE_COLORS, dist 4-7m,
# |bearing| in [60,110] AND positive-signed -- matches BidirectionalScan-
# Schedule's _LEG_SIGNS=(+1,-1,-1,+1) "positive leg0 first" so no leg0->leg1
# reversal is needed, sidestepping the rotation-order-instability class
# documented in docs/gen1_multiseed.md §3.1 / docs/nx12_turn_dwell.md; no
# same-color distractor -- docs/gen1_multiseed.md §3.3's false-lock risk; no
# distractor within 0.5m of the straight robot->target path; target shape
# != cone -- docs/nx16_cone_stall.md's cone-specific confidence-decay risk),
# then verified by actually running the full rollout headlessly 2x:
#   seed=1259 -> target=yellow cube, dist=4.31m, bearing=85.2° (out-of-FOV).
#   Both runs: success=True, fell=False, steps=637, final_dist=0.472m,
#   byte-identical trajectories, wall~315s each. See docs/fs1_first_scene.md.
FIRST_SCENE_SEED = 1259


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
    """
    Project world XYZ points into BEV pixel coordinates.

    Uses MuJoCo's camera view matrix + a pinhole projection.
    Returns (N, 2) float array of (u, v) pixel coords.
    Clips out-of-frame points but does not filter them.
    """
    import mujoco

    # Build camera view matrix from lookat / azimuth / elevation / distance
    # MuJoCo uses azimuth (from +X, counter-clockwise), elevation (from XY plane)
    az  = math.radians(bev_cam.azimuth)
    el  = math.radians(bev_cam.elevation)  # negative = below horizon
    dist = bev_cam.distance

    # Camera position in world frame
    # MuJoCo free camera: position = lookat + distance * (-sinaz*cosel, -cosaz*cosel, -sinel)
    # Actually: MuJoCo azimuth measures from -Y (south) axis, counter-clockwise when viewed from top
    # Let's compute cam position directly:
    cosel = math.cos(el)
    sinel = math.sin(el)
    cosaz = math.cos(az)
    sinaz = math.sin(az)

    # Camera Z-axis (forward = from cam toward lookat)
    cam_fwd = np.array([-sinaz * cosel, cosaz * cosel, sinel], dtype=np.float64)

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
) -> np.ndarray:
    """
    Draw all BEV overlays on bev_img (in-place + return).

    Overlays:
      (a) PATH TRAIL — green polyline
      (b) TARGET HIGHLIGHT — orange ring + cross
      (c) FOV CONE — yellow wedge on ground
      (d) STATUS BANNER — bottom banner with state + distance
    """
    import cv2

    img = bev_img.copy()
    H, W = img.shape[:2]

    def w2p(world_xyz):
        """World XYZ (or XY with Z=0) to pixel (u,v) as int tuple."""
        if len(world_xyz) == 2:
            world_xyz = np.array([world_xyz[0], world_xyz[1], 0.0])
        pix = world_to_bev_pixel(np.array([world_xyz]), bev_cam, model, data, W, H, fovy_deg)
        return (int(round(pix[0, 0])), int(round(pix[0, 1])))

    # (a) PATH TRAIL — polyline of past base positions
    if len(path_trail) >= 2:
        pts_pix = []
        for xy in path_trail:
            p = w2p(xy)
            pts_pix.append(p)
        for i in range(1, len(pts_pix)):
            alpha = i / len(pts_pix)  # fade from dim to bright
            c = tuple(int(x * alpha + x * 0.3) for x in COLOR_PATH_TRAIL)
            # Clamp to image bounds before drawing
            p0 = pts_pix[i - 1]
            p1 = pts_pix[i]
            if (0 <= p0[0] < W and 0 <= p0[1] < H) or (0 <= p1[0] < W and 0 <= p1[1] < H):
                cv2.line(img, p0, p1, COLOR_PATH_TRAIL, 2, cv2.LINE_AA)

    # Robot position dot (white)
    robot_pix = w2p([robot_xy[0], robot_xy[1], 0.0])
    if 0 <= robot_pix[0] < W and 0 <= robot_pix[1] < H:
        cv2.circle(img, robot_pix, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, robot_pix, 7, (0, 0, 0), 1, cv2.LINE_AA)

    # (c) FOV CONE — robot ego camera wedge on the ground (±45° from robot_yaw, 3m range)
    fov_half_rad = math.radians(45.0)  # ego cam FOV half-angle (90° FOVY → ±45°)
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
            # Pulsing effect based on time
            pulse = 1.0  # static for video (no time.time() drift in video)
            r = 14
            cv2.circle(img, tgt_pix, r, COLOR_TARGET_RING, 2, cv2.LINE_AA)
            cv2.circle(img, tgt_pix, r + 5, COLOR_TARGET_RING, 1, cv2.LINE_AA)
            # Cross hair
            sz = 8
            cv2.line(img, (tgt_pix[0] - sz, tgt_pix[1]), (tgt_pix[0] + sz, tgt_pix[1]),
                     COLOR_TARGET_RING, 2, cv2.LINE_AA)
            cv2.line(img, (tgt_pix[0], tgt_pix[1] - sz), (tgt_pix[0], tgt_pix[1] + sz),
                     COLOR_TARGET_RING, 2, cv2.LINE_AA)
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

    # Draw a line from robot to target if target visible in BEV
    if target_xy is not None and dist_to_target is not None:
        tgt_pix2 = w2p([target_xy[0], target_xy[1], 0.0])
        if (0 <= tgt_pix2[0] < W and 0 <= tgt_pix2[1] < H and
                0 <= robot_pix[0] < W and 0 <= robot_pix[1] < H and
                state in (STATE_MOVING, STATE_LOCATED)):
            cv2.line(img, robot_pix, tgt_pix2, (180, 80, 255), 1, cv2.LINE_AA)

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


def compose_sbs_frame(
    ego_rgb: np.ndarray,   # (EGO_H, EGO_W, 3) uint8 RGB — CAM-2 ACTIVE camera feed
    bev_img: np.ndarray,   # (BEV_H, BEV_W, 3) uint8 BGR
    state: str = STATE_IDLE,
    prompt: str = "",
    dist_to_target: Optional[float] = None,
    goal_idx: int = 0,
    n_goals: int = 1,
    active_cam: str = "GROUNDING",   # CAM-2 (docs/cam_p1.md): 'GROUNDING' (head, far) | 'PROXIMITY' (near)
) -> np.ndarray:
    """
    Compose side-by-side frame: ego (left, CAM-2 active-camera feed) | BEV (right).

    Returns (max_h, total_w, 3) uint8 BGR frame.
    """
    import cv2

    # Convert ego from RGB to BGR
    ego_bgr = cv2.cvtColor(ego_rgb, cv2.COLOR_RGB2BGR)

    # Scale ego to match BEV height
    target_h = BEV_H
    if ego_bgr.shape[0] != target_h:
        scale = target_h / ego_bgr.shape[0]
        ego_bgr = cv2.resize(ego_bgr, (int(ego_bgr.shape[1] * scale), target_h))

    # Ego overlay: state badge + active-camera label
    badge_h = 36
    cv2.rectangle(ego_bgr, (0, 0), (ego_bgr.shape[1], badge_h), (20, 20, 20), -1)
    state_color_map = {
        STATE_SEARCHING: COLOR_STATE_SEARCH,
        STATE_LOCATED:   COLOR_STATE_LOCATE,
        STATE_MOVING:    COLOR_STATE_MOVE,
        STATE_REACHED:   COLOR_STATE_REACH,
        STATE_FAILED:    (100, 100, 100),
        STATE_IDLE:      (150, 150, 150),
    }
    sc = state_color_map.get(state, (200, 200, 200))
    cv2.rectangle(ego_bgr, (4, 4), (90, 30), sc, -1)
    cv2.putText(ego_bgr, state[:10], (7, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    # CAM-2 handoff label: "HEAD CAM" (GROUNDING, far) / "PROXIMITY CAM" (near) —
    # makes the camera handoff visible to viewers, distinct from the small
    # "CAM: GROUNDING|PROXIMITY d=X.XXm" overlay already baked into ego_rgb by
    # _label_active_cam() in the main rollout loop.
    cam_label = "PROXIMITY CAM" if active_cam == "PROXIMITY" else "HEAD CAM"
    cam_color = (60, 210, 255) if active_cam == "PROXIMITY" else (255, 200, 150)
    (tw, _), _ = cv2.getTextSize(cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(ego_bgr, cam_label, (ego_bgr.shape[1] - tw - 8, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, cam_color, 1, cv2.LINE_AA)

    # Divider line
    divider = np.full((target_h, 3, 3), 60, dtype=np.uint8)

    sbs = np.concatenate([ego_bgr, divider, bev_img], axis=1)
    return sbs


# ---------------------------------------------------------------------------
# Fancy rollout — search-then-goto with BEV follow-cam + overlays
# ---------------------------------------------------------------------------

def run_fancy_rollout(
    inf,                          # Inferencer instance (goal_source='classical')
    scene_cfg: dict,
    prompt: str,
    goto_ckpt_path: str = GOTO_CKPT_DEFAULT,
    maxsteps: int = MAXSTEPS_FANCY,
    render_video: bool = True,
    video_path: Optional[str] = None,
    frame_callback=None,          # called each step with (sbs_bgr, state, dist, step)
    # FD2: multi-goal context
    goal_idx: int = 0,
    n_goals: int = 1,
    path_trail_in: Optional[List[np.ndarray]] = None,  # carry trail from prior sub-goals
    completed_targets: Optional[List[np.ndarray]] = None,  # already-reached targets
) -> dict:
    """
    Search-then-goto rollout with ego|BEV side-by-side frames + 4 overlays.

    Always uses SEARCH behavior (student-driven bidirectional bounded scan,
    code/scan_sched.py, until target spotted — see docs/nx1_scan.md).
    Returns dict: success, spotted, scan_steps, steps, final_dist, fell, video_path
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
    from code.grounding import ground as classical_ground, get_ego_intrinsics_rendered
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

    # --- Load action stats from inferencer ---
    _use_residual = (getattr(inf, '_action_stats', None) is not None)
    if _use_residual:
        _as       = inf._action_stats
        _da_mean  = _as['mean']
        _da_std   = _as['std']
        _da_deflt = _as['default_angles']

    _use_phase = getattr(inf, '_use_phase', False)
    _phase_tracker = _GaitPhaseTracker() if _use_phase else None
    _eff_pdim = PROPRIO_DIM_PHASE if _use_phase else PROPRIO_DIM

    # --- State ---
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

    def _lock_drop_and_rescan():
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

    def _update_bev_cam():
        """Follow robot with BEV camera."""
        bxy = data_mj.qpos[0:2]
        bev_cam.lookat[:] = [bxy[0], bxy[1], BEV_LOOKAT_Z]
        bev_cam.distance  = BEV_DISTANCE
        bev_cam.azimuth   = BEV_AZIMUTH
        bev_cam.elevation = BEV_ELEVATION

    def _render_sbs_frame():
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

        # Draw overlays (FD2: pass goal progress + completed targets)
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
        )

        sbs = compose_sbs_frame(ego_rgb, bev_bgr, current_state, prompt, dist,
                                goal_idx=goal_idx, n_goals=n_goals, active_cam=_active_cam)
        return sbs, dist

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

    renderer.close()

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

    # Save MP4 in background
    out_vid = None
    if render_video and video_path and frames_sbs:
        out_vid = _write_fancy_video(frames_sbs, video_path)

    return dict(
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

_ALL_COLORS = ["red", "yellow", "blue", "green", "orange", "purple", "cyan"]
_ALL_SHAPES = RELIABLE_SHAPES  # ["ball", "cube", "cylinder", "cone"] -- full shape set


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
    """
    Extract a best-effort (color, shape) hint from one instruction clause.

    Scans the whole clause for known color/shape words (order-independent --
    handles "red ball", "the ball that is red", "red-colored ball", etc.) rather
    than requiring the two words to be adjacent. `color`/`shape` are set only
    when exactly one candidate word of that kind is present in the clause;
    `colors_mentioned`/`shapes_mentioned` keep the full sets for ambiguity
    scoring (see _resolve_goal_to_index).
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
    """
    Rule-based multi-goal parser for fancy_demo (kept under its original name and
    signature for backward compat). Splits on "then" conjunctions, extracts
    (color, shape) per part. Returns list of dicts: [{color, shape, prompt_part}, ...]

    NX-15: now implemented on top of _split_multi_goal_parts()/_extract_goal_hint()
    (the shared internals also used by resolve_live_instruction() below) instead of
    its own standalone regex -- same public contract as before.
    """
    goals = []
    for part in _split_multi_goal_parts(instruction):
        hint = _extract_goal_hint(part)
        if hint["color"] or hint["shape"]:
            goals.append({"color": hint["color"], "shape": hint["shape"],
                           "prompt_part": hint["prompt_part"]})
    return goals


def _resolve_goal_to_index(hint: dict, objects: List[dict]) -> tuple:
    """
    Resolve one (color, shape) hint against the current scene's object list.

    Returns (obj_idx, clarify_question):
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
    """
    NX-15: THE single shared instruction -> target resolver for both live entry
    points (_terminal_loop, Flask /execute). Never used by the scripted/headless
    entry points (run_smoke(), showcase/recording APIs), which continue to pass
    explicit scene_cfg['target_index'] values untouched -- that default remains
    ONLY the fallback for entry points that explicitly pass an index.

    Returns a dict:
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
    inf,
    goals: List[dict],           # [{color, shape, prompt_part}, ...]
    scene_cfg: dict,
    maxsteps: int = MAXSTEPS_FANCY,
    render_video: bool = True,
    video_path: Optional[str] = None,
    frame_callback=None,
) -> dict:
    """
    Execute sequential sub-goals on the SAME scene.
    For each sub-goal:
      - Sets scene target_index to the matching object
      - Runs run_fancy_rollout() with path_trail carried over
      - BEV shows current target ring + completed target dots + goal N/M banner

    Returns combined dict with per-goal results + overall success.
    """
    n_goals = len(goals)
    objects = scene_cfg["objects"]

    def _find_obj(color, shape):
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

def _write_fancy_video(frames: list, path: str, fps: int = 25) -> str:
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


def _concat_reel(video_paths: list, reel_path: str) -> Optional[str]:
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
    """
    Sample a search scene with:
    - Target OUTSIDE initial FOV (bearing > 45° from robot_yaw=0)
    - RELIABLE colors biased: orange, red, yellow, purple (avoid cyan/blue HSV wall collision)
    - Distance 2-4m (easy enough to reach but shows search phase)
    - Multiple objects placed non-overlapping

    Returns scene_cfg dict compatible with run_fancy_rollout.
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
# FD2: Long-distance scene sampler (4-7 m, reliable colors, large arena)
# ---------------------------------------------------------------------------

def sample_fancy_scene_long(rng: np.random.Generator, ep_idx: int,
                             dist_min: float = DIST_MIN_LONG,
                             dist_max: float = DIST_MAX_LONG) -> dict:
    """
    Sample a long-distance search scene:
    - Target OUTSIDE initial FOV (bearing > 45° from robot_yaw=0)
    - RELIABLE colors ONLY: red, orange, yellow, purple
      (grounding_dist.md: 78% success at 4-9m for non-cyan/blue)
    - Distance 4–7m (impressive walk; arena_size=8m to fit)
    - 3 objects total, non-overlapping
    - Robot near origin (robot_xy ≈ 0)

    FD2: biases toward MEDIUM-LONG distances to make the reel impressive.
    """
    from code.arena import COLORS, SHAPES
    from code.eval_search import SEARCH_FOV_HALF_DEG

    arena_half   = ARENA_HALF_LONG  # 8m — room for 7m targets
    margin       = 0.6
    fov_half_rad = math.radians(SEARCH_FOV_HALF_DEG)

    # Robot slightly off-center
    rx = float(rng.uniform(-0.5, 0.5))
    ry = float(rng.uniform(-0.5, 0.5))
    robot_yaw = 0.0

    # Only reliable colors for target (red, orange, yellow, purple)
    reliable_idxs = [i for i, (cname, _) in enumerate(COLORS)
                     if cname in RELIABLE_COLORS]
    shape_idxs = list(range(len(SHAPES)))

    tgt_ci = int(rng.choice(reliable_idxs))
    tgt_si = int(rng.choice(shape_idxs))

    # Distractors: also prefer reliable colors but can differ
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
        for _ in range(8000):
            if is_target:
                # Long distance: 4–7m, out-of-FOV
                d    = float(rng.uniform(dist_min, dist_max))
                side = rng.integers(2)
                if side == 0:
                    angle = float(rng.uniform(robot_yaw + fov_half_rad, robot_yaw + math.pi))
                else:
                    angle = float(rng.uniform(robot_yaw - math.pi, robot_yaw - fov_half_rad))
            else:
                # Distractors: 2–5m from robot, anywhere
                d     = float(rng.uniform(2.0, 5.0))
                angle = float(rng.uniform(-math.pi, math.pi))

            ox = rx + d * math.cos(angle)
            oy = ry + d * math.sin(angle)

            if abs(ox) + size_val / 2 + margin >= arena_half:
                continue
            if abs(oy) + size_val / 2 + margin >= arena_half:
                continue
            if any(math.hypot(ox - o["x"], oy - o["y"]) < 1.0 for o in objects):
                continue

            if is_target:
                dx, dy    = ox - rx, oy - ry
                obj_angle = math.atan2(dy, dx)
                err = math.atan2(math.sin(obj_angle - robot_yaw), math.cos(obj_angle - robot_yaw))
                if abs(err) <= fov_half_rad:
                    continue  # target must be out-of-FOV

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
            # Fallback: relax constraints
            for _ in range(20000):
                if is_target:
                    d     = float(rng.uniform(dist_min, dist_max))
                    side  = rng.integers(2)
                    angle = (float(rng.uniform(robot_yaw + fov_half_rad, robot_yaw + math.pi))
                             if side == 0
                             else float(rng.uniform(robot_yaw - math.pi, robot_yaw - fov_half_rad)))
                    ox = rx + d * math.cos(angle)
                    oy = ry + d * math.sin(angle)
                else:
                    ox = float(rng.uniform(-(arena_half - margin), arena_half - margin))
                    oy = float(rng.uniform(-(arena_half - margin), arena_half - margin))
                if (abs(ox) + 0.5 + margin < arena_half and
                        abs(oy) + 0.5 + margin < arena_half and
                        not any(math.hypot(ox - o["x"], oy - o["y"]) < 0.6 for o in objects)):
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
        "difficulty":       "search_long",
        "init_bearing_deg": init_bearing_deg,
    }


def sample_fancy_multi_goal_scene(rng: np.random.Generator, n_goals: int = 2) -> dict:
    """
    Sample a scene with n_goals objects at varied distances (2–6m), each
    a distinct reliable color+shape. Robot at origin, yaw=0.

    Returns scene_cfg; target_index=0 (first object is 1st sub-goal).
    Multi-goal rollout iterates target_index across 0..n_goals-1.
    """
    from code.arena import COLORS, SHAPES
    from code.eval_search import SEARCH_FOV_HALF_DEG

    arena_half   = ARENA_HALF_LONG
    margin       = 0.6
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

    objects = []
    for local_i, (ci, si) in enumerate(chosen):
        color_name, color_rgb = COLORS[ci]
        shape_name, size      = SHAPES[si]
        size_val = float(size)
        is_target = (local_i == 0)

        placed = False
        # Vary distance: first target further, subsequent closer
        if local_i == 0:
            d_lo, d_hi = 4.5, 6.5   # first goal: long
        else:
            d_lo, d_hi = 2.5, 5.0   # subsequent: medium

        for _ in range(8000):
            d    = float(rng.uniform(d_lo, d_hi))
            if is_target:
                # First sub-goal: out-of-FOV
                side  = rng.integers(2)
                angle = (float(rng.uniform(robot_yaw + fov_half_rad, robot_yaw + math.pi))
                         if side == 0
                         else float(rng.uniform(robot_yaw - math.pi, robot_yaw - fov_half_rad)))
            else:
                angle = float(rng.uniform(-math.pi, math.pi))

            ox = rx + d * math.cos(angle)
            oy = ry + d * math.sin(angle)

            if abs(ox) + size_val / 2 + margin >= arena_half:
                continue
            if abs(oy) + size_val / 2 + margin >= arena_half:
                continue
            if any(math.hypot(ox - o["x"], oy - o["y"]) < 1.2 for o in objects):
                continue
            if is_target:
                dx, dy    = ox - rx, oy - ry
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
            for _ in range(20000):
                d     = float(rng.uniform(d_lo, d_hi))
                angle = float(rng.uniform(-math.pi, math.pi))
                ox    = rx + d * math.cos(angle)
                oy    = ry + d * math.sin(angle)
                if (abs(ox) + 0.5 + margin < arena_half and
                        abs(oy) + 0.5 + margin < arena_half and
                        not any(math.hypot(ox - o["x"], oy - o["y"]) < 0.8 for o in objects)):
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
_stream_lock  = threading.Lock()
_stream_frame = [None]    # bytes: latest MJPEG JPEG frame
_status_lock  = threading.Lock()
_status_state = {
    "state": STATE_IDLE,
    "prompt": "",
    "dist": None,
    "step": 0,
    "scene_desc": "",
    "result": None,
}


def _set_stream_frame(bgr_frame):
    """Encode BGR numpy frame to JPEG bytes and push to stream."""
    try:
        import cv2
        _, buf = cv2.imencode('.jpg', bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with _stream_lock:
            _stream_frame[0] = buf.tobytes()
    except Exception:
        pass


def _placeholder_frame(state: str = STATE_IDLE, prompt: str = "") -> bytes:
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
    inf,
    scene_manager,
    out_dir: str,
    port: int = WEB_PORT,
    maxsteps: int = MAXSTEPS_FANCY,
    render_video: bool = True,
):
    """Start Flask web UI for fancy demo in a background thread."""
    try:
        from flask import Flask, Response, request, jsonify, render_template_string
    except ImportError:
        print("[fancy_demo] Flask not installed. Web UI unavailable.", flush=True)
        return None

    app = Flask(__name__)

    _exec_lock   = threading.Lock()
    _exec_thread = [None]

    def _scene_desc():
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

    def _do_rollout(instruction: str, parsed: dict):
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

        def _cb(frame_bgr, state, dist, step):
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
    def index():
        return render_template_string(_HTML_FANCY)

    @app.route("/scene_info")
    def scene_info():
        return jsonify({"scene_desc": _scene_desc()})

    @app.route("/new_scene", methods=["POST"])
    def new_scene():
        scene_manager.new_scene()
        return jsonify({"scene_desc": _scene_desc()})

    @app.route("/execute", methods=["POST"])
    def execute():
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
    def status():
        with _status_lock:
            st = dict(_status_state)
        st['scene_desc'] = _scene_desc()
        return jsonify(st)

    @app.route("/stream")
    def stream():
        def gen():
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

    def _run_flask():
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

    def __init__(self, seed_offset: int = 0):
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
    def _scene_cfg(self): return self.__scene_cfg
    @_scene_cfg.setter
    def _scene_cfg(self, v): self.__scene_cfg = v


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
):
    """
    FD2 Headless smoke: LONG-DISTANCE search episodes + multi-goal, saved as MP4s.

    Episode plan (default n=6):
      ep0: long single-goal search (4-7m)
      ep1: long single-goal search (4-7m)
      ep2: long single-goal search (4-7m) — smoke verify 1st episode
      ep3: long single-goal search (4-7m)
      ep4: long single-goal search (4-7m)
      ep5: MULTI-GOAL (2 sub-goals, different reliable colors)

    Only SUCCESS episodes go into the showcase reel (fail-filtered).
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
def main():
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
        _terminal_loop(inf, scene_mgr, args.out, args.maxsteps, not args.no_render)


def _terminal_loop(inf, scene_mgr, out_dir, maxsteps, render_video):
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


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
