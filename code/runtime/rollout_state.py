"""
code.runtime.rollout_state — env/settle setup + mutable per-episode state for
`Inferencer.rollout()`.

RF-1 split of code/inferencer.py (docs/refactor_plan.md): everything the
original monolithic `rollout()` did ONCE, before its per-step loop starts —
resolve the language embedding, build the MuJoCo arena, settle the robot
(keyframe or WBC fallback), and initialize the proprio/STALL_BREAK/temporal-
ensembling bookkeeping the loop then mutates step by step. The
grounding/EMA/scan/lock/AVOID state is a separate `GoalPipeline` instance
(code.runtime.goal_pipeline) held as one field on this bundle.

This is a mechanical extraction: no control flow or numeric logic changed,
only *where* the code lives (mirrors code/eval/search_rollout_state.py's
precedent for the same kind of split).
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import torch

from code.sim.arena import build_arena, ArenaRenderer
from code.sim.teacher import WBCTeacher, SIM_DT
from code.control import avoid as _avoid
from code.runtime.constants import (FALL_HEIGHT, PROPRIO_K, PROPRIO_DIM, PROPRIO_DIM_PHASE,
                                    STALL_BREAK, STALL_WINDOW_STEPS, AVOID)
from code.runtime.gait_phase import _GaitPhaseTracker
from code.runtime.helpers import _build_proprio
from code.runtime.io import RolloutResult
from code.runtime.goal_pipeline import GoalPipeline

_COLORS_ORDERED = ["red", "yellow", "blue", "green", "orange", "purple", "cyan"]
_SHAPES_ORDERED = ["ball", "cube", "cylinder", "cone"]


@dataclass
class RolloutState:
    """Bundle of env handles + initialized mutable state for one episode.

    If `early_result` is not None, the episode ended during settle (a fall)
    and the caller must return it immediately without entering the step loop.
    """
    early_result: RolloutResult | None = None

    # Scene / target info
    target_xy:    np.ndarray | None = None
    target_color: str = ""
    target_shape: str = ""
    stop_r:       float = 0.6
    lang_t:       Any = None

    # MuJoCo env handles
    arena_model: Any = None
    teacher:     Any = None
    data_mj:     Any = None
    model_mj:    Any = None
    nj:          int = 0

    renderer:    Any = None
    tp_cam:      Any = None
    frames_ego:  list = field(default_factory=list)
    frames_tp:   list = field(default_factory=list)

    # Fix 4: gait phase
    use_phase:     bool = False
    phase_tracker: Any = None
    eff_proprio_dim: int = PROPRIO_DIM

    # Fix 1: residual action de-normalization
    use_residual: bool = False
    da_mean:  np.ndarray | None = None
    da_std:   np.ndarray | None = None
    da_deflt: np.ndarray | None = None

    # Fix 2: GT velocity injection
    inject_gt_vel: bool = False

    # Goal-source mode flags (precomputed once, reused every step — loop-invariant
    # across the whole episode, so hoisting out of the original per-step
    # recomputation changes nothing observable)
    inject_cached:    bool = False   # arch=='A' and (not use_learned_goal or (learned+need_learned_render))
    use_learned_goal: bool = False
    use_gt_goal:      bool = False
    need_learned_render: bool = False

    # Student loop state
    prev_action:  np.ndarray | None = None
    proprio_hist: Any = None   # collections.deque
    student_target_dof: np.ndarray | None = None
    te_buffer: list = field(default_factory=list)

    goal_pipeline: GoalPipeline | None = None

    # Oscillation / displacement tracking
    all_target_dofs: list = field(default_factory=list)
    start_xy: np.ndarray | None = None

    # Loop bookkeeping
    step_times:   list = field(default_factory=list)
    hold_counter: int = 0
    fell:         bool = False
    steps_done:   int = 0

    # NX-8 STALL_BREAK per-episode watchdog state
    stall_hist:               Any = None   # collections.deque
    stall_recovery_remaining: int = 0
    stall_cooldown_remaining: int = 0
    stall_trigger_count:      int = 0
    stall_is_maneuver:        bool = False
    cur_vx_cmd:               float = 0.0


def _resolve_lang_emb(inf, lang_emb: np.ndarray | None, target_color: str,
                       target_shape: str) -> np.ndarray:
    """Resolves the language embedding for this episode.

    For learned grounding (grounding_trained=True), builds a one-hot
    color+shape encoding in the first (N_COLORS + N_SHAPES) dims of the
    2048-d lang emb — matches the encoding used during grounding training
    (train_grounding.py). Otherwise zeros (unless caller supplied one).
    """
    if lang_emb is None:
        if getattr(inf, '_grounding_trained', False) and inf.goal_source == 'learned':
            lang_emb = np.zeros(2048, dtype=np.float32)
            c_idx = _COLORS_ORDERED.index(target_color) if target_color in _COLORS_ORDERED else 0
            s_idx = _SHAPES_ORDERED.index(target_shape) if target_shape in _SHAPES_ORDERED else 0
            lang_emb[c_idx] = 1.0
            lang_emb[len(_COLORS_ORDERED) + s_idx] = 1.0
        else:
            lang_emb = np.zeros(2048, dtype=np.float32)
    return lang_emb


def _setup_rollout(inf, scene_cfg: dict, lang_emb: np.ndarray | None,
                    stop_r: float | None) -> RolloutState:
    """Builds the MuJoCo env, settles the robot, and initializes rollout state.

    Mirrors the setup preamble of the original (pre-RF-1) monolithic
    `Inferencer.rollout()` verbatim. See `RolloutState` for the returned
    field set; if `.early_result` is set, the caller must return it
    immediately (the robot fell during settle).

    Args:
        inf: Inferencer instance providing the model / cached keyframe /
            action-stats / goal_source / vel_source / arch / verbose.
        scene_cfg: Scene configuration dict (objects, target_index, robot_xy,
            robot_yaw, stop_r, difficulty, ...).
        lang_emb: Optional (2048,) language embedding (see `_resolve_lang_emb`).
        stop_r: Success radius (m); defaults to `scene_cfg['stop_r']` (or 0.6).

    Returns:
        A populated `RolloutState`.
    """
    s = RolloutState()
    if stop_r is None:
        stop_r = float(scene_cfg.get('stop_r', 0.6))
    s.stop_r = stop_r

    objects    = scene_cfg['objects']
    target_idx = scene_cfg['target_index']
    target_obj = objects[target_idx]
    s.target_xy    = np.array([target_obj['x'], target_obj['y']], dtype=np.float64)
    s.target_color = target_obj['color_name']
    s.target_shape = target_obj['shape_name']

    lang_emb = _resolve_lang_emb(inf, lang_emb, s.target_color, s.target_shape)
    s.lang_t = torch.from_numpy(lang_emb.astype(np.float32)).unsqueeze(0).to(inf.device)

    # ---- Build arena (adds objects to G1 XML) ----
    arena_model = build_arena(scene_cfg)
    arena_model.opt.timestep = SIM_DT
    s.arena_model = arena_model

    # ---- Inject arena model into teacher (same pattern as gen_dataset.py) ----
    teacher = WBCTeacher(use_gpu=False)   # CPU is fine (0.32ms/step)
    teacher.model = arena_model
    teacher.data  = mujoco.MjData(arena_model)
    teacher._nj   = arena_model.nq - 7
    teacher._pelvis_id = mujoco.mj_name2id(
        arena_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )
    s.teacher = teacher

    # Reset to scene start pose
    rx, ry    = scene_cfg['robot_xy']
    robot_yaw = float(scene_cfg.get('robot_yaw', 0.0))
    teacher.reset(pos_xy=(rx, ry), yaw=robot_yaw)

    s.data_mj  = teacher.data
    s.model_mj = teacher.model
    s.nj       = teacher._nj

    # ---- Renderer ----
    renderer = ArenaRenderer(s.model_mj)
    s.renderer = renderer
    s.tp_cam   = renderer.make_tp_cam()

    # ---- Settle: either WBC ONNX (baseline) or keyframe restore (WBC-free) ----
    keyframe = getattr(inf, '_keyframe', None)
    if keyframe is not None:
        kf_qpos = keyframe['qpos_local'].copy()
        kf_qpos[0] = rx
        kf_qpos[1] = ry
        kf_qpos[3] = math.cos(robot_yaw / 2)   # w
        kf_qpos[4] = 0.0                         # x
        kf_qpos[5] = 0.0                         # y
        kf_qpos[6] = math.sin(robot_yaw / 2)    # z
        s.data_mj.qpos[:len(kf_qpos)] = kf_qpos
        s.data_mj.qvel[:len(keyframe['qvel_local'])] = keyframe['qvel_local']
        mujoco.mj_forward(s.model_mj, s.data_mj)
        teacher._target_dof = keyframe['target_dof'].copy()
        if teacher.base_height < FALL_HEIGHT:
            renderer.close()
            s.early_result = RolloutResult(
                success=False, failure_tag='fall', steps=0,
                final_dist=float(np.linalg.norm(s.data_mj.qpos[0:2] - s.target_xy)),
                fell=True, upright=False,
                ms_per_step=0.0, grounding_hz=0.0,
                scene_cfg=scene_cfg,
            )
            return s
    else:
        for _ in range(80):
            teacher.step(vel_cmd=(0.0, 0.0, 0.0))
            if teacher.base_height < FALL_HEIGHT:
                renderer.close()
                s.early_result = RolloutResult(
                    success=False, failure_tag='fall', steps=0,
                    final_dist=float(np.linalg.norm(s.data_mj.qpos[0:2] - s.target_xy)),
                    fell=True, upright=False,
                    ms_per_step=0.0, grounding_hz=0.0,
                    scene_cfg=scene_cfg,
                )
                return s

    # ---- Fix 4: gait phase tracker (if phase-conditioned checkpoint) ----
    s.use_phase = getattr(inf, '_use_phase', False)
    s.phase_tracker = _GaitPhaseTracker() if s.use_phase else None
    if s.use_phase:
        print(f"[inferencer] Fix-4 gait-phase tracking active: proprio_dim={PROPRIO_DIM_PHASE}")

    # ---- Fix 1: prepare de-normalization arrays if action_stats present ----
    action_stats = getattr(inf, '_action_stats', None)
    s.use_residual = (action_stats is not None)
    if s.use_residual:
        s.da_mean  = action_stats['mean']
        s.da_std   = action_stats['std']
        s.da_deflt = action_stats['default_angles']
        print(f"[inferencer] Fix-1 residual mode active: denorm = default + pred*std + mean")

    # ---- Fix 2: GT velocity injection ----
    s.inject_gt_vel = (inf.arch == 'A' and inf.vel_source == 'gt')
    if s.inject_gt_vel:
        print(f"[inferencer] Fix-2 GT velocity injection active (steer.py privileged cmd)")

    # ---- Goal-source mode flags ----
    s.need_learned_render = (inf.arch == 'A' and inf.goal_source == 'learned'
                              and getattr(inf, '_grounding_trained', False))
    need_classical_render = (inf.arch == 'A' and inf.goal_source == 'classical')
    s.use_gt_goal       = (inf.arch == 'A' and inf.goal_source == 'gt')
    s.use_learned_goal  = (inf.arch != 'A') or (inf.goal_source == 'learned')
    # Mirrors the original per-step `_inject_cached` computation (line ~1438 of the
    # pre-RF-1 file) — every operand here is loop-invariant, so hoisting this out
    # of the step loop into a single setup-time value is behavior-preserving.
    s.inject_cached = (inf.arch == 'A' and
                       (not s.use_learned_goal or
                        (inf.goal_source == 'learned' and s.need_learned_render)))

    # ---- State for student loop ----
    s.prev_action = teacher._target_dof.copy()  # last teacher target as initial
    s.eff_proprio_dim = PROPRIO_DIM_PHASE if s.use_phase else PROPRIO_DIM
    s.proprio_hist = collections.deque(
        [np.zeros(s.eff_proprio_dim, dtype=np.float32)] * PROPRIO_K,
        maxlen=PROPRIO_K,
    )
    prop_now = _build_proprio(s.data_mj, s.prev_action)
    if s.use_phase:
        q_lb_settle = s.data_mj.qpos[7:22].copy()
        ph_settle = s.phase_tracker.update(q_lb_settle)
        prop_now = np.concatenate([prop_now, ph_settle])   # (57,)
    for _ in range(PROPRIO_K):
        s.proprio_hist.append(prop_now.copy())

    # ---- Goal pipeline (grounding-cycle/goal/EMA/hold/handoff/scan state) ----
    # NX-9 AVOID (docs/nx9_avoid.md): per-episode obstacle-bias carve-out —
    # AVOID and _avoid.is_maneuver_scene(scene_cfg), exactly as the original
    # inline `_avoid_is_maneuver` computation.
    avoid_is_maneuver = (AVOID and _avoid.is_maneuver_scene(scene_cfg))
    s.goal_pipeline = GoalPipeline(
        need_classical_render=need_classical_render,
        need_learned_render=s.need_learned_render,
        avoid_is_maneuver=avoid_is_maneuver,
        verbose=inf.verbose,
    )

    s.student_target_dof = teacher._target_dof.copy()
    s.start_xy = s.data_mj.qpos[0:2].copy()

    # ---- NX-8 STALL_BREAK: per-episode watchdog state ----
    s.stall_hist = collections.deque(maxlen=STALL_WINDOW_STEPS)
    s.stall_is_maneuver = (STALL_BREAK and
                           str(scene_cfg.get('difficulty', '')).lower() == 'maneuver')

    return s
