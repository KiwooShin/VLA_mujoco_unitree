"""
eval_search.py — Closed-loop evaluation for the SEARCH skill.

The search skill = student-driven bidirectional bounded scan (WBC-free) until
the target enters the FOV (classical grounding detects it), then GOTO.
NX-1 (docs/nx1_scan.md, docs/fa1_failures.md #1 fix): the scan alternates
CCW/CW in ~150°-bounded legs with stand dwells between them (code/scan_sched.py)
instead of a fixed single CCW direction — caps every continuous rotation
segment well inside the in-distribution range, eliminating the 3 falls caused
by "wrong-side" targets needing a near-full continuous rotation under the
old fixed-CCW scan.

This is assembled PURELY behaviorally — the existing Inferencer with goal_source='classical'
already implements a (differently-bounded, ±90°) scan-and-acquire mechanism for the demo
skill's own in-rollout scan (H3); this file's standalone rollout is search-specific:
  1. target starts OUT of initial FOV  → grounding.not_visible=True → scan_active=True
  2. student-driven bounded bidirectional scan (inject wz into action head, WBC-free)
     while checking grounding every cycle
  3. when target detected AND bearing < 40°  → scan_active=False → GOTO begins
  4. classical HSV grounding guides approach → stop within STOP_R

ANTI-HANG:
  - Smoke 1 scene first (fast, MAXSTEPS=200)
  - Hard MAXSTEPS cap: 1400 steps
  - Background process + poll every 10s
  - Flush prints throughout

Eval protocol:
  - Seed: 999 (held-out)
  - n=15 scenes, target OUTSIDE initial FOV (bearing > 45°)
  - Success = spotted (target entered FOV during scan) + reached (final_dist < STOP_R) + upright
  - Reports SPOT-rate + REACH-rate separately
  - Renders 2-3 success videos (ego|third-person: rotate → spot → approach)

Usage:
    MUJOCO_GL=egl python code/eval_search.py --smoke
    MUJOCO_GL=egl python code/eval_search.py --n 15 --out eval/search --device cuda
    MUJOCO_GL=egl python code/eval_search.py --n 15 --out eval/search --no-video
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_SEED       = 999
# NX-1: bumped from 1400 (the old "same as demo preset" value) -- the bidirectional
# bounded scan (code/scan_sched.py) caps every continuous rotation segment safely,
# but can spend more TOTAL steps finding an unfavorable-side target than the old
# fixed-CCW scan did in its common case (it now always visits up to ~3*SCAN_LEG_DEG
# of yaw before giving up, vs. sometimes finding a favorable-side target almost
# immediately). SCAN_TIMEOUT=1150 (scan_sched.py) alone left too little of the
# 1400 budget for the approach phase on several previously-passing episodes
# (e.g. ep0: spotted at step 890, only 510 left, final_dist=0.51 -- one hair
# outside STOP_R_SEARCH). 2000 gives ~850 steps of approach headroom even in the
# worst observed case. See docs/nx1_scan.md.
MAXSTEPS_SEARCH = 2000           # hard cap (was 1400 pre-NX-1)
STOP_R_SEARCH   = 0.5            # slightly lenient stop radius
N_RENDER        = 3              # max videos to render
GOTO_CKPT       = str(_REPO / "checkpoint" / "goto_best.pt")

# FOV constraint for search scenes: target MUST start outside this cone
SEARCH_FOV_HALF_DEG = 45.0       # target angle from robot heading must exceed this
SEARCH_DIST_MIN     = 2.0        # target distance (easy case, no obstacles)
SEARCH_DIST_MAX     = 4.5        # keep it reachable after scan

# Scan threshold from inferencer.py (target_bearing < this to exit scan mode)
SCAN_ALIGNED_THR_DEG = 40.0

# ---------------------------------------------------------------------------
# Out-of-FOV scene sampler
# ---------------------------------------------------------------------------

def sample_search_scene(rng: np.random.Generator, episode_idx: int) -> dict:
    """
    Sample a search scene where the target is OUTSIDE the initial FOV.

    Strategy:
      - Robot near centre, facing +X (yaw=0)
      - Target placed at bearing > SEARCH_FOV_HALF_DEG from robot heading
      - Easy case: no obstacles, small arena (4m half), target at 2-4.5m
      - 3 objects total (1 target + 2 distractors)

    The FOV constraint ensures scan is REQUIRED to find the target.
    """
    from code.arena import COLORS, SHAPES
    from code.scene import _make_instruction

    arena_half = 4.0
    margin = 0.55

    # Robot: near centre, fixed yaw=0 (face +X)
    rx = float(rng.uniform(-0.3, 0.3))
    ry = float(rng.uniform(-0.3, 0.3))
    robot_yaw = 0.0

    # Choose 3 unique (color, shape) combos
    all_combos = [(ci, si) for ci in range(len(COLORS)) for si in range(len(SHAPES))]
    chosen_indices = rng.choice(len(all_combos), size=3, replace=False)
    chosen_combos  = [all_combos[k] for k in chosen_indices]
    target_local   = 0   # first combo is always the target

    objects = []
    fov_half_rad = math.radians(SEARCH_FOV_HALF_DEG)

    for local_i, (ci, si) in enumerate(chosen_combos):
        color_name, color_rgb = COLORS[ci]
        shape_name, size      = SHAPES[si]
        size_val = float(size)
        is_target = (local_i == target_local)

        placed = False
        for _ in range(5000):
            if is_target:
                # Force OUT of FOV: bearing must be > SEARCH_FOV_HALF_DEG from robot yaw
                d = float(rng.uniform(SEARCH_DIST_MIN, SEARCH_DIST_MAX))
                # Sample bearing in the "behind" arc: outside ±fov_half_rad from robot_yaw
                # Use two side arcs: [robot_yaw+fov_half_rad, robot_yaw+pi] and
                # [robot_yaw-pi, robot_yaw-fov_half_rad]
                side = rng.integers(2)
                if side == 0:
                    angle = float(rng.uniform(robot_yaw + fov_half_rad,
                                              robot_yaw + math.pi))
                else:
                    angle = float(rng.uniform(robot_yaw - math.pi,
                                              robot_yaw - fov_half_rad))
            else:
                # Distractors: anywhere, at least 0.8m from robot
                d = float(rng.uniform(0.8, 3.5))
                angle = float(rng.uniform(-math.pi, math.pi))

            ox = rx + d * math.cos(angle)
            oy = ry + d * math.sin(angle)

            # Bounds check
            if abs(ox) + size_val / 2 + margin >= arena_half:
                continue
            if abs(oy) + size_val / 2 + margin >= arena_half:
                continue

            # No overlap with already-placed objects (min 0.8m)
            if any(math.hypot(ox - o["x"], oy - o["y"]) < 0.8 for o in objects):
                continue

            # Verify target bearing constraint
            if is_target:
                dx, dy = ox - rx, oy - ry
                obj_angle = math.atan2(dy, dx)
                err = math.atan2(math.sin(obj_angle - robot_yaw),
                                 math.cos(obj_angle - robot_yaw))
                if abs(err) <= fov_half_rad:
                    continue   # accidentally inside FOV, resample

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
            # Fallback: place anywhere in arena (relaxed constraints)
            for _ in range(10000):
                if is_target:
                    # Must still be out of FOV
                    side = rng.integers(2)
                    d    = float(rng.uniform(SEARCH_DIST_MIN, SEARCH_DIST_MAX))
                    if side == 0:
                        angle = float(rng.uniform(robot_yaw + fov_half_rad,
                                                   robot_yaw + math.pi))
                    else:
                        angle = float(rng.uniform(robot_yaw - math.pi,
                                                   robot_yaw - fov_half_rad))
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

    tgt = objects[target_local]
    instruction = _make_instruction(rng, tgt["color_name"], tgt["shape_name"])

    # Compute initial bearing for verification
    dx, dy = tgt["x"] - rx, tgt["y"] - ry
    init_angle = math.atan2(dy, dx)
    init_bearing_deg = abs(math.degrees(
        math.atan2(math.sin(init_angle - robot_yaw), math.cos(init_angle - robot_yaw))
    ))

    return {
        "arena_size":       arena_half,
        "robot_xy":         (rx, ry),
        "robot_yaw":        robot_yaw,
        "objects":          objects,
        "target_index":     target_local,
        "instruction":      instruction,
        "stop_r":           STOP_R_SEARCH,
        "horizon":          MAXSTEPS_SEARCH,
        "lighting":         {"ambient": 0.4},
        "difficulty":       "search",
        "init_bearing_deg": init_bearing_deg,   # diagnostic: how far out of FOV
    }


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    ep_idx:          int
    instruction:     str
    target_color:    str
    target_shape:    str
    target_dist:     float
    init_bearing_deg: float       # initial bearing to target (must be > SEARCH_FOV_HALF_DEG)
    spotted:         bool          # target entered FOV during scan (scan_active became False)
    reached:         bool          # final_dist < stop_r AND upright
    success:         bool          # spotted AND reached
    failure_tag:     str
    steps:           int
    scan_steps:      int           # steps spent in scan mode (until spotted or timeout)
    final_dist:      float
    fell:            bool
    ms_per_step:     float
    video_path:      Optional[str] = None


# ---------------------------------------------------------------------------
# Instrumented rollout (wraps Inferencer to track spot event)
# ---------------------------------------------------------------------------

def _run_search_rollout(
    inf,                            # Inferencer instance (goal_source='classical')
    scene_cfg:    dict,
    instruction:  str,
    maxsteps:     int = MAXSTEPS_SEARCH,
    render_video: bool = False,
    video_path:   Optional[str] = None,
) -> dict:
    """
    Run one search rollout using the existing Inferencer.
    Tracks when the target is first spotted (scan_active → False).

    Returns dict with: success, spotted, scan_steps, failure_tag, steps,
                        final_dist, fell, ms_per_step, video_path
    """
    # We run the standard rollout, which already has the H3 scan-and-acquire logic.
    # To track "spotted", we monkey-patch the verbose flag and capture scan exit.
    # Simpler: just check scan_active via a wrapper class.

    import mujoco
    import torch
    import collections
    import math as _math
    from code.inferencer import (
        _build_proprio, _apply_student_pd, _rgb_to_tensor, _write_video,
        _GaitPhaseTracker, _compute_gt_goal,
        FALL_HEIGHT, GROUNDING_PERIOD, HOLD_STEPS_REQUIRED, ACTION_SCALE,
        PROPRIO_K, PROPRIO_DIM, PROPRIO_DIM_PHASE, IMG_SIZE, SIM_DT
    )
    from code.arena import build_arena, ArenaRenderer, GROUNDING_W, GROUNDING_H, CAMERA_MODE
    from code.teacher import (WBCTeacher, _yaw_of, DEFAULT_ANGLES, KPS, KDS,
                               NUM_ACTIONS, SIM_DT as _SIM_DT, CONTROL_DECIMATION)
    from code.grounding import ground as classical_ground, get_ego_intrinsics_rendered
    from code.steer import steer as _steer_cmd
    from code.scan_sched import (BidirectionalScanSchedule, SCAN_LEG_DEG,
                                  SCAN_DWELL_STEPS, SCAN_TIMEOUT as _SCAN_TIMEOUT_DEFAULT)
    from code.lock_mgmt import LockGate, ReacquisitionScan

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

    # --- Renderer ---
    renderer = ArenaRenderer(model_mj)
    # CAM-1 (Phase 2, toggle): this loop is a hand-duplicated copy of
    # Inferencer.rollout()'s render/grounding logic (predates the toggle) and always
    # called render_grounding() with these fixed (45deg-FOVY-assuming) intrinsics --
    # harmless for cam2 (the default; matches build_arena()'s untouched 45deg render),
    # but WRONG in widefov mode: build_arena() sets model.vis.global_.fovy model-wide,
    # so render_grounding() would actually render at WIDEFOV_FOVY while this precomputed
    # `intr` still assumed 45deg, corrupting every backprojected (dist,bearing). Only
    # used now as the cam2-mode fallback; widefov mode gets fresh per-cycle intrinsics
    # from renderer.render_widefov() below (see intr_active in the main loop).
    intr     = get_ego_intrinsics_rendered(GROUNDING_W, GROUNDING_H)
    tp_cam   = renderer.make_tp_cam()
    frames_ego, frames_tp = [], []

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
                    fell=True, ms_per_step=0.0, video_path=None)

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

    # Lang embedding (zeros — same as inferencer default)
    lang_t = torch.zeros(1, 2048, device=inf.device)

    # Scan state — NX-1 bidirectional bounded-rotation sweep (docs/nx1_scan.md,
    # docs/fa1_failures.md #1 fix). Replaces the old fixed-CCW-only scan (up to
    # 600 continuous steps / ~413°), which forced "wrong-side" targets through
    # a near-full continuous rotation — exactly the 3 falls in
    # eval/p4_gate_search_rerun (ep5/7/8, 550-600 continuous steps, OOD per
    # docs/rot_dart.md). The new schedule caps every continuous same-direction
    # rotation at SCAN_LEG_DEG=150° (~218 steps, well inside the ~470-step/
    # ~323° in-distribution ceiling) with a stand-still dwell between legs, and
    # alternates CCW/CW so a target on either side is found by ITS favorable
    # direction instead of requiring the long way around. See code/scan_sched.py
    # for the full derivation (150° per leg is the minimum needed for 360°
    # bearing coverage given SCAN_ALIGNED_THR_DEG=40°).
    cached_goal_vec = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    last_grounding_step = -999
    _scan_active    = True
    _scan_yaw_delta = 0.0
    SCAN_TIMEOUT    = _SCAN_TIMEOUT_DEFAULT   # 900: safety-net cap; nominal full
                                               # bidirectional coverage pass completes
                                               # in ~727 steps (see scan_sched.py)
    SCAN_RATE       = 0.6        # rad/s — same as H3 goto scan (trained, stable)
    SCAN_DT         = SIM_DT * CONTROL_DECIMATION
    SCAN_ALIGNED_THR = _math.radians(SCAN_ALIGNED_THR_DEG)
    _scan_sched     = BidirectionalScanSchedule(scan_rate=SCAN_RATE,
                                                 leg_deg=SCAN_LEG_DEG,
                                                 dwell_steps=SCAN_DWELL_STEPS)
    _goal_ema       = None
    _GOAL_EMA_ALPHA = 0.4
    _last_known_goal = None
    _frames_since_det = 0
    HOLD_GOAL_HORIZON = 100

    # NX-2 (docs/rs1_lock_mgmt.md): shared lock-management gate (LOCK_M1..M5,
    # independently toggled via env var; M1/M3 default ON (opt-out), M2/M4/M5
    # default OFF (opt-in) per docs/nx2_final.md -- see code/lock_mgmt.py).
    # search has no CAM-2 Schmitt handoff / fallback probe (single grounding
    # camera only), so unlike inferencer.py this call site never calls
    # `mark_discontinuity()` -- M3/M4 always apply their full gate here.
    _lock_gate          = LockGate()
    _using_rescan_sched = False   # True only while a M4/M5-triggered bounded
                                   # rescan (ReacquisitionScan) is driving _scan_active,
                                   # as opposed to the initial BidirectionalScanSchedule
                                   # sweep (_scan_sched) above.
    _rescan_sched        = None

    def _lock_drop_and_rescan():
        """M4 (divergence) / M5 (coast-expiry) shared action: drop the lock,
        clear EMA/last-known-goal, and re-enter scan via a FRESH
        ReacquisitionScan. Not the same `_scan_sched` instance used for the
        initial scan above: that scan's own outer SCAN_TIMEOUT check is keyed
        on the episode's ABSOLUTE step, so re-arming it mid-episode would
        immediately time out (see code/lock_mgmt.py's ReacquisitionScan
        docstring)."""
        nonlocal _goal_ema, _last_known_goal, _frames_since_det
        nonlocal _scan_active, _using_rescan_sched, _rescan_sched, cached_goal_vec
        _lock_gate.force_drop()
        _goal_ema           = None
        _last_known_goal    = None
        _frames_since_det   = 0
        _scan_active        = True
        _using_rescan_sched = True
        _rescan_sched       = ReacquisitionScan(scan_rate=SCAN_RATE)
        cached_goal_vec      = np.array([2.0, 1.0, 0.0], dtype=np.float32)

    # Search-specific tracking
    spotted     = False    # set True when scan_active first becomes False
    scan_steps  = 0        # incremented while _scan_active is True

    step_times   = []
    hold_counter = 0
    fell         = False
    steps_done   = 0
    _all_target_dofs = []

    for step in range(maxsteps):
        t0 = time.perf_counter()

        # Fall check
        height = float(data_mj.qpos[2])
        if height < FALL_HEIGHT:
            fell = True
            break

        yaw = _yaw_of(data_mj.qpos[3:7])

        # Grounding cadence
        need_grounding = (step - last_grounding_step) >= GROUNDING_PERIOD
        need_render    = render_video or need_grounding

        intr_active = intr   # default (cam2): the loop-invariant 45deg-FOVY intrinsics
        if need_render:
            if need_grounding:
                if CAMERA_MODE == 'widefov':
                    # CAM-1 (Phase 2, toggle): single wide-FOV camera — use its own
                    # per-call intrinsics (correct FOVY/pitch), not the cam2 `intr`.
                    rgb, depth, intr_active = renderer.render_widefov(
                        data_mj, yaw, render_depth=True)
                else:
                    rgb, depth, _ = renderer.render_grounding(data_mj, yaw, render_depth=True)
                if render_video:
                    rgb_video, _, _ = renderer.render_ego(data_mj, yaw, render_depth=False)
                else:
                    rgb_video = None
            else:
                rgb, depth = None, None
                if render_video:
                    rgb_video, _, _ = renderer.render_ego(data_mj, yaw, render_depth=False)
                else:
                    rgb_video = None
        else:
            rgb, depth, rgb_video = None, None, None

        # Classical grounding
        if need_grounding and rgb is not None and depth is not None:
            gr = classical_ground(rgb, depth, target_color, target_shape, intr_active)
            last_grounding_step = step

            if not gr.not_visible:
                raw_goal = gr.goal_vec.copy()
                # NX-2 (LOCK_M1/M2/M3, docs/rs1_lock_mgmt.md): gate the raw detection
                # before it's allowed to feed the EMA/last-known-goal. Provable
                # pass-through (always True) with all three toggles off.
                _accept_hit = _lock_gate.gate_detection(
                    float(raw_goal[0]), _math.atan2(raw_goal[2], raw_goal[1]), gr.best_area)
                if _accept_hit:
                    _frames_since_det = 0
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

                    # Exit scan when aligned
                    if _scan_active:
                        det_bearing = abs(_math.atan2(_goal_ema[2], _goal_ema[1]))
                        if det_bearing < SCAN_ALIGNED_THR:
                            _scan_active = False
                            spotted = True
                            print(f"  [search] SPOTTED at step={step}  bearing={_math.degrees(det_bearing):.1f}°",
                                  flush=True)
                else:
                    # NX-2: gate rejected this detection -- treat this cycle like a miss.
                    _frames_since_det += 1
                    if _last_known_goal is not None and _frames_since_det <= HOLD_GOAL_HORIZON:
                        cached_goal_vec = _last_known_goal.copy()
                    elif _lock_gate.coast_expired(_frames_since_det, HOLD_GOAL_HORIZON):
                        print(f"  [lock] M5 coast expired (gate-rejected) -> "
                              f"drop+rescan at step={step}", flush=True)
                        _lock_drop_and_rescan()
            else:
                _frames_since_det += 1
                if _last_known_goal is not None and _frames_since_det <= HOLD_GOAL_HORIZON:
                    cached_goal_vec = _last_known_goal.copy()
                elif _lock_gate.coast_expired(_frames_since_det, HOLD_GOAL_HORIZON):
                    # NX-2 (LOCK_M5): bounded coast -> reroute to rescan instead of an
                    # unbounded silent freeze.
                    print(f"  [lock] M5 coast expired -> drop+rescan at step={step}", flush=True)
                    _lock_drop_and_rescan()

            # NX-2 (LOCK_M4): divergence watchdog -- runs once per grounding cycle
            # regardless of hit/miss/gate outcome above. Provable no-op when off.
            _walking_toward_goal = (not _scan_active) and (float(cached_goal_vec[0]) > stop_r)
            if _lock_gate.end_of_cycle(float(cached_goal_vec[0]), _walking_toward_goal):
                print(f"  [lock] M4 divergence -> drop+rescan at step={step}", flush=True)
                _lock_drop_and_rescan()

        # Scan mode: NX-1 bidirectional bounded-rotation sweep (see setup above).
        # Observable (memoryless per-frame visibility check), WBC-free.
        # SCAN_TIMEOUT=900 is a safety-net cap — the schedule itself normally
        # exits (spotted) well before that.
        if _scan_active:
            if _using_rescan_sched:
                # NX-2 (LOCK_M4/M5): a lock-drop-triggered rescan uses a FRESH
                # ReacquisitionScan (local step counter) rather than re-arming
                # `_scan_sched`/`SCAN_TIMEOUT` above, which is keyed on the episode's
                # absolute step and would immediately time out mid-episode.
                scan_wz = _rescan_sched.step(yaw)
                if scan_wz is None:
                    _scan_active        = False
                    _using_rescan_sched = False
                    print(f"  [lock][rescan] TIMEOUT at step={step}, no target spotted", flush=True)
                else:
                    scan_steps += 1
                    prop_now = _build_proprio(data_mj, prev_action)
                    if _use_phase:
                        ph = _phase_tracker.update(data_mj.qpos[7:22].copy())
                        prop_now = np.concatenate([prop_now, ph])
                    proprio_hist.append(prop_now)
                    prop_arr = np.stack(list(proprio_hist), axis=0)
                    prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(inf.device)

                    img_t_scan = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32,
                                             device=inf.device)
                    scan_goal_t = torch.from_numpy(cached_goal_vec).unsqueeze(0).to(inf.device)
                    scan_vel_t  = torch.tensor([[0.0, 0.0, scan_wz]], dtype=torch.float32,
                                               device=inf.device)

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
                    _all_target_dofs.append(prev_action.copy())
                    steps_done = step + 1

                    if render_video and rgb_video is not None:
                        frames_ego.append(rgb_video.copy())
                        renderer.update_tp_cam(tp_cam, data_mj)
                        frames_tp.append(renderer.render_tp(data_mj, tp_cam).copy())

                    t1 = time.perf_counter()
                    step_times.append((t1 - t0) * 1000.0)
                    continue   # skip normal student step
            elif step >= SCAN_TIMEOUT:
                _scan_active = False   # timeout — fallback to default goal, not spotted
                print(f"  [search] SCAN TIMEOUT at step={step}, no target spotted", flush=True)
            else:
                scan_steps += 1
                scan_wz = _scan_sched.step(yaw)   # bounded CCW/CW schedule, dwells at 0.0
                _scan_yaw_delta += scan_wz * SCAN_DT

                # Student forward pass with injected wz
                prop_now = _build_proprio(data_mj, prev_action)
                if _use_phase:
                    ph = _phase_tracker.update(data_mj.qpos[7:22].copy())
                    prop_now = np.concatenate([prop_now, ph])
                proprio_hist.append(prop_now)
                prop_arr = np.stack(list(proprio_hist), axis=0)
                prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(inf.device)

                img_t_scan = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32,
                                         device=inf.device)
                scan_goal_t = torch.from_numpy(cached_goal_vec).unsqueeze(0).to(inf.device)
                scan_vel_t  = torch.tensor([[0.0, 0.0, scan_wz]], dtype=torch.float32,
                                           device=inf.device)

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
                _all_target_dofs.append(prev_action.copy())
                steps_done = step + 1

                if render_video and rgb_video is not None:
                    frames_ego.append(rgb_video.copy())
                    renderer.update_tp_cam(tp_cam, data_mj)
                    frames_tp.append(renderer.render_tp(data_mj, tp_cam).copy())

                t1 = time.perf_counter()
                step_times.append((t1 - t0) * 1000.0)
                continue   # skip normal student step

        # Normal GOTO student step (after spotted or scan timeout)
        prop_now = _build_proprio(data_mj, prev_action)
        if _use_phase:
            ph = _phase_tracker.update(data_mj.qpos[7:22].copy())
            prop_now = np.concatenate([prop_now, ph])
        proprio_hist.append(prop_now)
        prop_arr = np.stack(list(proprio_hist), axis=0)
        prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(inf.device)

        img_t      = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32, device=inf.device)
        goal_inj_t = torch.from_numpy(cached_goal_vec).unsqueeze(0).to(inf.device)

        with torch.no_grad():
            out = inf.model(
                ego_rgb   = img_t,
                lang_emb  = lang_t,
                proprio_h = prop_t,
                gt_goal   = goal_inj_t,
                gt_vel    = None,
            )

        raw_action = out['action'].cpu().numpy().squeeze(0)[0]
        if _use_residual:
            student_target_dof = _da_deflt + raw_action * _da_std + _da_mean
        else:
            student_target_dof = raw_action

        _all_target_dofs.append(student_target_dof.copy())

        for _ in range(CONTROL_DECIMATION):
            _apply_student_pd(data_mj, student_target_dof, nj)
            mujoco.mj_step(model_mj, data_mj)

        prev_action = student_target_dof.copy()
        steps_done  = step + 1

        if render_video and rgb_video is not None:
            frames_ego.append(rgb_video.copy())
            renderer.update_tp_cam(tp_cam, data_mj)
            frames_tp.append(renderer.render_tp(data_mj, tp_cam).copy())

        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1000.0)

        dist_to_target = float(np.linalg.norm(data_mj.qpos[0:2] - target_xy))
        if dist_to_target < stop_r:
            hold_counter += 1
            if hold_counter >= HOLD_STEPS_REQUIRED:
                break
        else:
            hold_counter = 0

        if step % 100 == 0:
            dist_to_target = float(np.linalg.norm(data_mj.qpos[0:2] - target_xy))
            print(f"  step={step:4d}  dist={dist_to_target:.2f}m  scan={'ON' if _scan_active else 'OFF'}  "
                  f"spotted={spotted}  h={height:.3f}m", flush=True)

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

    # Write video
    out_vid = None
    if render_video and video_path and frames_ego:
        _write_video(frames_ego, frames_tp, video_path)
        out_vid = video_path

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
    )


# Import _write_video from inferencer
from code.inferencer import _write_video


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate_search(
    checkpoint_path: Optional[str] = None,
    n_scenes:    int   = 15,
    device:      str   = 'cpu',
    out_dir:     str   = 'eval/search',
    render_video: bool = True,
    smoke:       bool  = False,
    seed:        int   = 999,
) -> dict:
    """
    Run search evaluation: target starts outside initial FOV.
    Returns summary dict with spot_rate, reach_rate, success_rate.
    """
    from code.inferencer import Inferencer

    os.makedirs(out_dir, exist_ok=True)

    ckpt = checkpoint_path or GOTO_CKPT
    print(f"[search_eval] Loading inferencer: {ckpt}", flush=True)
    inf = Inferencer(
        checkpoint_path=ckpt,
        arch='A',
        device=device,
        goal_source='classical',
        verbose=False,
    )
    print(f"[search_eval] Inferencer ready", flush=True)

    if smoke:
        n_scenes = 1
        print(f"[search_eval] SMOKE MODE: 1 scene, MAXSTEPS=200", flush=True)

    results: List[SearchResult] = []
    ep_results = []

    for ep_i in range(n_scenes):
        rng = np.random.default_rng(np.random.SeedSequence([seed, ep_i]))
        scene_cfg = sample_search_scene(rng, ep_i)

        tgt    = scene_cfg['objects'][scene_cfg['target_index']]
        init_b = float(scene_cfg.get('init_bearing_deg', 0.0))

        print(f"\n[search_eval] ep={ep_i:02d}  {tgt['color_name']} {tgt['shape_name']}  "
              f"dist={tgt['dist_from_robot']:.2f}m  init_bearing={init_b:.1f}°", flush=True)

        # Render only first N_RENDER episodes (success videos)
        do_render  = render_video and (ep_i < N_RENDER)
        video_path = None
        if do_render:
            video_path = os.path.join(
                out_dir,
                f"search_ep{ep_i:02d}_{tgt['color_name']}_{tgt['shape_name']}.mp4"
            )

        maxsteps = 200 if smoke else MAXSTEPS_SEARCH
        t0 = time.time()

        try:
            raw = _run_search_rollout(
                inf=inf,
                scene_cfg=scene_cfg,
                instruction=scene_cfg['instruction'],
                maxsteps=maxsteps,
                render_video=do_render,
                video_path=video_path,
            )
        except Exception as e:
            import traceback
            print(f"[search_eval] ep={ep_i} EXCEPTION: {e}", flush=True)
            traceback.print_exc()
            raw = dict(success=False, spotted=False, scan_steps=0, failure_tag='error',
                       steps=0, final_dist=999.0, fell=False, ms_per_step=0.0, video_path=None)

        dt = time.time() - t0
        sr = SearchResult(
            ep_idx           = ep_i,
            instruction      = scene_cfg['instruction'],
            target_color     = tgt['color_name'],
            target_shape     = tgt['shape_name'],
            target_dist      = tgt['dist_from_robot'],
            init_bearing_deg = init_b,
            spotted          = raw['spotted'],
            reached          = (raw['final_dist'] < STOP_R_SEARCH) and not raw['fell'],
            success          = raw['success'],
            failure_tag      = raw['failure_tag'],
            steps            = raw['steps'],
            scan_steps       = raw['scan_steps'],
            final_dist       = raw['final_dist'],
            fell             = raw['fell'],
            ms_per_step      = raw['ms_per_step'],
            video_path       = raw.get('video_path'),
        )
        results.append(sr)
        ep_results.append(asdict(sr))

        print(f"  → spotted={sr.spotted}  reached={sr.reached}  success={sr.success}  "
              f"tag={sr.failure_tag}  steps={sr.steps}  fd={sr.final_dist:.2f}m  "
              f"wall={dt:.1f}s", flush=True)

        # EGL non-determinism fix: force GC between episodes to free EGL renderer objects.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- Summary ----
    n    = len(results)
    spot  = sum(1 for r in results if r.spotted)
    reach = sum(1 for r in results if r.reached)
    succ  = sum(1 for r in results if r.success)
    falls = sum(1 for r in results if r.fell)

    spot_rate  = spot  / n if n else 0.0
    reach_rate = reach / n if n else 0.0
    succ_rate  = succ  / n if n else 0.0

    print(f"\n{'='*60}", flush=True)
    print(f"[search_eval] RESULTS  (n={n}, seed={seed})", flush=True)
    print(f"  SPOT-rate:  {spot}/{n} = {spot_rate:.1%}", flush=True)
    print(f"  REACH-rate: {reach}/{n} = {reach_rate:.1%}", flush=True)
    print(f"  SUCCESS:    {succ}/{n} = {succ_rate:.1%}  (spotted+reached+upright)", flush=True)
    print(f"  Falls:      {falls}/{n}", flush=True)
    print(f"{'='*60}", flush=True)

    summary = {
        "n_scenes":    n,
        "eval_seed":   seed,
        "checkpoint":  ckpt,
        "spot_rate":   spot_rate,
        "reach_rate":  reach_rate,
        "success_rate": succ_rate,
        "n_spot":      spot,
        "n_reach":     reach,
        "n_success":   succ,
        "n_falls":     falls,
        "episodes":    ep_results,
    }

    # Save
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[search_eval] Summary saved → {summary_path}", flush=True)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Search skill evaluator")
    ap.add_argument("--checkpoint", default=None, help="Path to goto checkpoint")
    ap.add_argument("--n",          type=int, default=15, help="Number of scenes")
    ap.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out",        default="eval/search", help="Output directory")
    ap.add_argument("--smoke",      action="store_true", help="1-scene smoke test")
    ap.add_argument("--no-video",   action="store_true", help="Disable video rendering")
    ap.add_argument("--seed",       type=int, default=999,
                    help="Eval seed (default=999 held-out; use other values for robustness)")
    args = ap.parse_args()

    evaluate_search(
        checkpoint_path=args.checkpoint,
        n_scenes=args.n,
        device=args.device,
        out_dir=args.out,
        render_video=not args.no_video,
        smoke=args.smoke,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
