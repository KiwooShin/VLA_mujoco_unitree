"""code.eval.search_rollout_state — env/settle setup + mutable state for the
search-skill rollout.

Split out of the original ``eval_search.py`` (RF-1): everything
``_run_search_rollout`` (``code.eval.search_rollout``) does ONCE, before its
per-step loop starts — build the MuJoCo arena, settle the robot (keyframe or
WBC fallback), and initialize the scan/lock/avoid bookkeeping the loop then
mutates step by step. Returned as one ``_RolloutSetup`` bundle so the loop
function can unpack it into local variables and proceed exactly as the
original single-function rollout did (the lock-drop-and-rescan closure still
needs to live in the loop function's own frame — see search_rollout.py).

This is a mechanical extraction: no control flow or numeric logic changed,
only *where* the code lives.
"""

from __future__ import annotations

import collections
import math as _math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from code.inferencer import _build_proprio, _GaitPhaseTracker
from code.arena import build_arena, ArenaRenderer, GROUNDING_W, GROUNDING_H
from code.teacher import WBCTeacher, SIM_DT, CONTROL_DECIMATION
from code.grounding import get_ego_intrinsics_rendered
from code.scan_sched import (BidirectionalScanSchedule, SCAN_LEG_DEG,
                              SCAN_DWELL_STEPS, SCAN_TIMEOUT as _SCAN_TIMEOUT_DEFAULT)
from code.lock_mgmt import LockGate
from code import avoid as _avoid
from code.inferencer import FALL_HEIGHT, PROPRIO_K, PROPRIO_DIM, PROPRIO_DIM_PHASE

from code.eval.search_types import STOP_R_SEARCH, SCAN_ALIGNED_THR_DEG


@dataclass
class _RolloutSetup:
    """Bundle of env handles + initialized mutable state for one search episode.

    Field names mirror the local variables the original (unsplit)
    ``_run_search_rollout`` used, including the leading-underscore
    "private" ones, so the caller can unpack this 1:1 and run the
    per-step loop exactly as before.

    If ``early_result`` is not None, the episode ended during settle
    (a fall) and the caller must return it immediately without entering
    the step loop.
    """
    early_result: dict | None = None

    # Scene / env handles
    objects: list = field(default_factory=list)
    target_idx: int = 0
    target_obj: dict | None = None
    target_xy: np.ndarray | None = None
    target_color: str = ""
    target_shape: str = ""
    stop_r: float = STOP_R_SEARCH

    arena_model: Any = None
    teacher: Any = None
    data_mj: Any = None
    model_mj: Any = None
    nj: int = 0

    renderer: Any = None
    intr: Any = None
    tp_cam: Any = None
    frames_ego: list = field(default_factory=list)
    frames_tp: list = field(default_factory=list)

    # Residual action de-normalization
    _use_residual: bool = False
    _da_mean: np.ndarray | None = None
    _da_std: np.ndarray | None = None
    _da_deflt: np.ndarray | None = None

    # Phase / proprio
    _use_phase: bool = False
    _phase_tracker: Any = None
    _eff_pdim: int = PROPRIO_DIM

    prev_action: np.ndarray | None = None
    proprio_hist: Any = None   # collections.deque
    lang_t: Any = None

    # Scan state
    cached_goal_vec: np.ndarray | None = None
    last_grounding_step: int = -999
    _scan_active: bool = True
    _scan_yaw_delta: float = 0.0
    SCAN_TIMEOUT: int = _SCAN_TIMEOUT_DEFAULT
    SCAN_RATE: float = 0.6
    SCAN_DT: float = 0.0
    SCAN_ALIGNED_THR: float = 0.0
    _scan_sched: Any = None

    _goal_ema: np.ndarray | None = None
    _GOAL_EMA_ALPHA: float = 0.4
    _last_known_goal: np.ndarray | None = None
    _frames_since_det: int = 0
    HOLD_GOAL_HORIZON: int = 100

    # Lock management (NX-2/NX-5)
    _lock_gate: Any = None
    _using_rescan_sched: bool = False
    _rescan_sched: Any = None
    _m7_prev_xy: np.ndarray | None = None

    # Avoid (NX-9)
    _avoid_bias_wz: float = 0.0
    _avoid_is_maneuver: bool = False
    _avoid_cycles_total: int = 0
    _avoid_cycles_active: int = 0

    # Search-specific tracking
    spotted: bool = False
    scan_steps: int = 0
    step_times: list = field(default_factory=list)
    hold_counter: int = 0
    fell: bool = False
    steps_done: int = 0
    _all_target_dofs: list = field(default_factory=list)


def _setup_search_rollout(inf, scene_cfg: dict) -> _RolloutSetup:
    """Builds the MuJoCo env, settles the robot, and initializes rollout state.

    Mirrors the setup preamble of the original (pre-RF-1) monolithic
    ``_run_search_rollout`` verbatim. See ``_RolloutSetup`` for the returned
    field set; if ``.early_result`` is set, the caller must return it
    immediately (the robot fell during settle).

    Args:
        inf: Inferencer instance (goal_source='classical') providing the
            model and any cached keyframe/action-stats state.
        scene_cfg: Scene configuration from sample_search_scene (robot pose,
            objects, target_index, stop_r, etc.).

    Returns:
        A populated ``_RolloutSetup``.
    """
    s = _RolloutSetup()

    # --- Extract scene info ---
    s.objects      = scene_cfg['objects']
    s.target_idx   = scene_cfg['target_index']
    s.target_obj   = s.objects[s.target_idx]
    s.target_xy    = np.array([s.target_obj['x'], s.target_obj['y']], dtype=np.float64)
    s.target_color = s.target_obj['color_name']
    s.target_shape = s.target_obj['shape_name']
    s.stop_r       = float(scene_cfg.get('stop_r', STOP_R_SEARCH))

    # --- Build MuJoCo env ---
    arena_model = build_arena(scene_cfg)
    arena_model.opt.timestep = SIM_DT
    s.arena_model = arena_model

    teacher = WBCTeacher(use_gpu=False)
    teacher.model = arena_model
    teacher.data  = mujoco.MjData(arena_model)
    teacher._nj   = arena_model.nq - 7
    teacher._pelvis_id = mujoco.mj_name2id(
        arena_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )
    s.teacher = teacher

    rx, ry    = scene_cfg['robot_xy']
    robot_yaw = float(scene_cfg.get('robot_yaw', 0.0))
    teacher.reset(pos_xy=(rx, ry), yaw=robot_yaw)

    s.data_mj  = teacher.data
    s.model_mj = teacher.model
    s.nj       = teacher._nj

    # --- Renderer ---
    renderer = ArenaRenderer(s.model_mj)
    s.renderer = renderer
    # CAM-1 (Phase 2, toggle): this loop is a hand-duplicated copy of
    # Inferencer.rollout()'s render/grounding logic (predates the toggle) and always
    # called render_grounding() with these fixed (45deg-FOVY-assuming) intrinsics --
    # harmless for cam2 (the default; matches build_arena()'s untouched 45deg render),
    # but WRONG in widefov mode: build_arena() sets model.vis.global_.fovy model-wide,
    # so render_grounding() would actually render at WIDEFOV_FOVY while this precomputed
    # `intr` still assumed 45deg, corrupting every backprojected (dist,bearing). Only
    # used now as the cam2-mode fallback; widefov mode gets fresh per-cycle intrinsics
    # from renderer.render_widefov() below (see intr_active in the main loop).
    s.intr     = get_ego_intrinsics_rendered(GROUNDING_W, GROUNDING_H)
    s.tp_cam   = renderer.make_tp_cam()
    s.frames_ego, s.frames_tp = [], []

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
        s.data_mj.qpos[:len(kf_qpos)] = kf_qpos
        s.data_mj.qvel[:len(kf['qvel_local'])] = kf['qvel_local']
        mujoco.mj_forward(s.model_mj, s.data_mj)
        teacher._target_dof = kf['target_dof'].copy()
    else:
        for _ in range(80):
            teacher.step(vel_cmd=(0.0, 0.0, 0.0))

    if teacher.base_height < FALL_HEIGHT:
        renderer.close()
        s.early_result = dict(
            success=False, spotted=False, scan_steps=0, failure_tag='fall',
            steps=0, final_dist=float(np.linalg.norm(s.data_mj.qpos[0:2] - s.target_xy)),
            fell=True, ms_per_step=0.0, video_path=None,
        )
        return s

    # --- Load action stats from inferencer ---
    s._use_residual = (getattr(inf, '_action_stats', None) is not None)
    if s._use_residual:
        _as       = inf._action_stats
        s._da_mean  = _as['mean']
        s._da_std   = _as['std']
        s._da_deflt = _as['default_angles']

    s._use_phase = getattr(inf, '_use_phase', False)
    s._phase_tracker = _GaitPhaseTracker() if s._use_phase else None
    s._eff_pdim = PROPRIO_DIM_PHASE if s._use_phase else PROPRIO_DIM

    # --- State ---
    s.prev_action  = teacher._target_dof.copy()
    s.proprio_hist = collections.deque(
        [np.zeros(s._eff_pdim, dtype=np.float32)] * PROPRIO_K, maxlen=PROPRIO_K
    )
    prop_now = _build_proprio(s.data_mj, s.prev_action)
    if s._use_phase:
        ph = s._phase_tracker.update(s.data_mj.qpos[7:22].copy())
        prop_now = np.concatenate([prop_now, ph])
    for _ in range(PROPRIO_K):
        s.proprio_hist.append(prop_now.copy())

    # Lang embedding (zeros — same as inferencer default)
    s.lang_t = torch.zeros(1, 2048, device=inf.device)

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
    s.cached_goal_vec = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    s.last_grounding_step = -999
    s._scan_active    = True
    s._scan_yaw_delta = 0.0
    s.SCAN_TIMEOUT    = _SCAN_TIMEOUT_DEFAULT   # 900: safety-net cap; nominal full
                                                 # bidirectional coverage pass completes
                                                 # in ~727 steps (see scan_sched.py)
    s.SCAN_RATE       = 0.6        # rad/s — same as H3 goto scan (trained, stable)
    s.SCAN_DT         = SIM_DT * CONTROL_DECIMATION
    s.SCAN_ALIGNED_THR = _math.radians(SCAN_ALIGNED_THR_DEG)
    s._scan_sched     = BidirectionalScanSchedule(scan_rate=s.SCAN_RATE,
                                                   leg_deg=SCAN_LEG_DEG,
                                                   dwell_steps=SCAN_DWELL_STEPS)
    s._goal_ema       = None
    s._GOAL_EMA_ALPHA = 0.4
    s._last_known_goal = None
    s._frames_since_det = 0
    s.HOLD_GOAL_HORIZON = 100

    # NX-2/NX-5 (docs/rs1_lock_mgmt.md, docs/nx5_coherence.md): shared
    # lock-management gate (LOCK_M1..M5, LOCK_M7, independently toggled via
    # env var; M1/M3 default ON (opt-out), M2/M4/M5/M7 default OFF (opt-in)
    # per docs/nx2_final.md / docs/nx5_coherence.md -- see code/lock_mgmt.py).
    # search has no CAM-2 Schmitt handoff / fallback probe (single grounding
    # camera only), so unlike inferencer.py this call site never calls
    # `mark_discontinuity()` -- M3/M4/M7 always apply their full gate here.
    s._lock_gate          = LockGate()
    s._using_rescan_sched = False   # True only while a M4/M5/M7-triggered bounded
                                     # rescan (ReacquisitionScan) is driving _scan_active,
                                     # as opposed to the initial BidirectionalScanSchedule
                                     # sweep (_scan_sched) above.
    s._rescan_sched        = None
    # M7 odometry-coherence watchdog: robot's own world-frame XY at the
    # previous grounding cycle (see code/inferencer.py's identical block for
    # the full rationale). Always maintained; no-op cost when LOCK_M7 is off.
    s._m7_prev_xy          = None

    # NX-9 AVOID (docs/nx9_avoid.md): same per-episode state / carve-out
    # pattern as code/inferencer.py -- see that file's identical block for
    # the full rationale.
    s._avoid_bias_wz       = 0.0
    s._avoid_is_maneuver   = (_avoid.AVOID and _avoid.is_maneuver_scene(scene_cfg))
    s._avoid_cycles_total  = 0
    s._avoid_cycles_active = 0

    # Search-specific tracking
    s.spotted     = False    # set True when scan_active first becomes False
    s.scan_steps  = 0        # incremented while _scan_active is True

    s.step_times   = []
    s.hold_counter = 0
    s.fell         = False
    s.steps_done   = 0
    s._all_target_dofs = []

    return s
