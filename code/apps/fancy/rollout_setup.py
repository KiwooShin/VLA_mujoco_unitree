"""Per-call episode-state initialization + end-of-episode finalization for
run_fancy_rollout (code/apps/fancy/rollout.py, RF-2a split of the former
single-file rollout.py).

Companion to `rollout_state.py` (which holds the per-*cycle* CAM-2/AVOID/
NX-16 computations): this file holds the one-time-per-call bookends --
building the policy/scan/camera/telemetry state the loop starts from, and
assembling the result dict (+ outro card / video save / live_ctx handoff)
once the loop ends. Every `init_*` function returns a plain NamedTuple that
rollout.py immediately destructures, positionally, into the exact same bare
local names the pre-split single-file version used -- so the control loop's
own body (between init and finalize) is completely unchanged by this split.
"""

from __future__ import annotations

import os
from typing import NamedTuple, Optional

import numpy as np

from code.apps.fancy.cards import make_outro_card
from code.apps.fancy.constants import FEAT_TITLECARD, STATE_REACHED, STATE_SEARCHING
from code.apps.fancy.video import _write_fancy_video


class PolicyState(NamedTuple):
    use_residual: bool
    da_mean: object
    da_std: object
    da_deflt: object
    use_phase: bool
    phase_tracker: object
    eff_pdim: int
    prev_action: np.ndarray
    proprio_hist: object
    lang_t: object


def init_policy_state(inf, resume_ctx: Optional[dict], teacher, data_mj) -> PolicyState:
    """Residual action-stats mode, gait-phase tracker, and (fresh vs
    VF-3-resumed) prev_action/proprio history + the zero language-embedding
    tensor. Mirrors the original run_fancy_rollout's "Load action stats" /
    "State" setup exactly."""
    import collections

    import torch

    from code.inferencer import PROPRIO_DIM, PROPRIO_DIM_PHASE, PROPRIO_K, _build_proprio, _GaitPhaseTracker

    use_residual = (getattr(inf, '_action_stats', None) is not None)
    da_mean = da_std = da_deflt = None
    if use_residual:
        _as      = inf._action_stats
        da_mean  = _as['mean']
        da_std   = _as['std']
        da_deflt = _as['default_angles']

    use_phase = getattr(inf, '_use_phase', False)
    phase_tracker = (resume_ctx['phase_tracker'] if resume_ctx is not None
                      else (_GaitPhaseTracker() if use_phase else None))
    eff_pdim = PROPRIO_DIM_PHASE if use_phase else PROPRIO_DIM

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
            [np.zeros(eff_pdim, dtype=np.float32)] * PROPRIO_K, maxlen=PROPRIO_K
        )
        prop_now = _build_proprio(data_mj, prev_action)
        if use_phase:
            ph = phase_tracker.update(data_mj.qpos[7:22].copy())
            prop_now = np.concatenate([prop_now, ph])
        for _ in range(PROPRIO_K):
            proprio_hist.append(prop_now.copy())

    lang_t = torch.zeros(1, 2048, device=inf.device)

    return PolicyState(use_residual, da_mean, da_std, da_deflt, use_phase,
                        phase_tracker, eff_pdim, prev_action, proprio_hist, lang_t)


class ScanState(NamedTuple):
    cached_goal_vec: np.ndarray
    last_grounding_step: int
    scan_active: bool
    scan_timeout: int
    scan_rate: float
    scan_dt: float
    scan_aligned_thr: float
    goal_ema: Optional[np.ndarray]
    last_known_goal: Optional[np.ndarray]
    frames_since_det: int
    hold_goal_horizon: int
    scan_yaw_delta: float
    scan_sched: object
    using_rescan_sched: bool
    rescan_sched: object
    rescan_local_steps: int


def init_scan_state() -> ScanState:
    """NX-1 bidirectional bounded-rotation sweep scan state (same schedule as
    code/eval_search.py, docs/nx1_scan.md); replaces the old fixed-CCW-only
    scan (up to 600 continuous steps) that was the diagnosed root cause of
    the search skill's falls (docs/fa1_failures.md #1 fix). See
    code/scan_sched.py for the derivation. Always starts fresh, even on a
    VF-3-resumed sim -- a new sub-goal's target warrants its own scan, same
    as before this split."""
    import math as _math

    from code.apps.fancy.rollout_state import HOLD_GOAL_HORIZON
    from code.eval_search import SCAN_ALIGNED_THR_DEG
    from code.scan_sched import SCAN_DWELL_STEPS, SCAN_LEG_DEG
    from code.scan_sched import SCAN_TIMEOUT as _SCAN_TIMEOUT_DEFAULT
    from code.scan_sched import BidirectionalScanSchedule
    from code.teacher import CONTROL_DECIMATION, SIM_DT

    scan_rate = 0.6
    return ScanState(
        cached_goal_vec=np.array([2.0, 1.0, 0.0], dtype=np.float32),
        last_grounding_step=-999,
        scan_active=True,
        scan_timeout=_SCAN_TIMEOUT_DEFAULT,   # 900: safety-net cap
        scan_rate=scan_rate,
        scan_dt=SIM_DT * CONTROL_DECIMATION,
        scan_aligned_thr=_math.radians(SCAN_ALIGNED_THR_DEG),
        goal_ema=None,
        last_known_goal=None,
        frames_since_det=0,
        hold_goal_horizon=HOLD_GOAL_HORIZON,
        scan_yaw_delta=0.0,
        scan_sched=BidirectionalScanSchedule(scan_rate=scan_rate, leg_deg=SCAN_LEG_DEG,
                                              dwell_steps=SCAN_DWELL_STEPS),
        using_rescan_sched=False,
        rescan_sched=None,
        rescan_local_steps=0,
    )


class CamAvoidState(NamedTuple):
    active_cam: str
    cam_miss_count: int
    video_frame_cache: Optional[np.ndarray]
    avoid_bias_wz: float
    avoid_is_maneuver: bool
    avoid_cycles_total: int
    avoid_cycles_active: int
    last_avoid_dbg: object


def init_cam_avoid_state(scene_cfg: dict) -> CamAvoidState:
    """CAM-2 ego-camera-handoff state (see rollout_state.py's CAM_D_LO/
    CAM_D_HI/CAM_PROXIMITY_D_FAR + cam_handoff()/handle_camera_probe()) and
    NX-9 AVOID per-episode state (docs/nx9_avoid.md; see
    rollout_state.avoid_bias_step() for the per-cycle mechanism, shared
    helper code/avoid.py). AVOID's bias only ever resets here at episode
    start -- its own decay/hysteresis handles the rest, including the one
    scan->goto transition."""
    from code import avoid as _avoid

    return CamAvoidState(
        active_cam='GROUNDING',   # default at episode start
        cam_miss_count=0,         # consecutive misses on the active camera
        video_frame_cache=None,   # last labeled active-cam frame (RGB, EGO_W x EGO_H)
        avoid_bias_wz=0.0,
        avoid_is_maneuver=(_avoid.AVOID and _avoid.is_maneuver_scene(scene_cfg)),
        avoid_cycles_total=0,
        avoid_cycles_active=0,
        last_avoid_dbg=None,   # VF-1 item 2: pure-read cache for draw_avoid_overlay()
    )


class TelemetryState(NamedTuple):
    spotted: bool
    scan_steps: int
    path_trail: list
    completed_targets: list
    frames_sbs: list
    step_times: list
    hold_counter: int
    fell: bool
    steps_done: int
    current_state: object
    current_dist: float
    dist_traveled_m: float
    prev_rxy_odom: np.ndarray
    hud_state: dict
    cam_flash_frames: int


def init_telemetry_state(
    resume_ctx: Optional[dict],
    path_trail_in,
    completed_targets,
    rx: float,
    ry: float,
    data_mj,
    target_xy: np.ndarray,
) -> TelemetryState:
    """Path trail / video-frame collection / hold-counter / VF-1 telemetry
    (odometry + HUD camera-flash timer) -- all pure-read/pure-accumulate
    state, never fed back into control. `dist_traveled_m`/`prev_rxy_odom`
    carry forward across VF-3-resumed sub-goals so the outro card's
    "distance traveled" reflects the WHOLE multi-goal journey, not just the
    final leg."""
    # FD2: carry path trail across sub-goals for visual continuity
    if path_trail_in is not None:
        path_trail = list(path_trail_in) + [np.array([rx, ry])]
    else:
        path_trail = [np.array([rx, ry])]   # list of (x,y) world pos

    if resume_ctx is not None:
        dist_traveled_m = resume_ctx['dist_traveled_m']
        prev_rxy_odom   = resume_ctx['prev_rxy_odom']
    else:
        dist_traveled_m = 0.0
        prev_rxy_odom   = np.array([rx, ry], dtype=np.float64)

    return TelemetryState(
        spotted=False,
        scan_steps=0,
        path_trail=path_trail,
        completed_targets=(list(completed_targets) if completed_targets else []),
        frames_sbs=[],                      # collected SBS frames for MP4
        step_times=[],
        hold_counter=0,
        fell=False,
        steps_done=0,
        current_state=STATE_SEARCHING,
        current_dist=float(np.linalg.norm(data_mj.qpos[0:2] - target_xy)),
        dist_traveled_m=dist_traveled_m,
        prev_rxy_odom=prev_rxy_odom,
        hud_state={"prev_cam": None, "flash_frames_left": 0},
        cam_flash_frames=10,
    )


def finalize_result(
    data_mj,
    target_xy: np.ndarray,
    stop_r: float,
    fell: bool,
    spotted: bool,
    step_times: list,
    steps_done: int,
    render_video: bool,
    video_path,
    frames_sbs: list,
    goal_idx: int,
    n_goals: int,
    dist_traveled_m: float,
    frame_callback,
    keep_alive: bool,
    teacher,
    model_mj,
    nj: int,
    renderer,
    bev_cam,
    prev_action,
    proprio_hist,
    phase_tracker,
    prev_rxy_odom,
    avoid_cycles_active: int,
    avoid_cycles_total: int,
    path_trail: list,
    scan_steps: int,
) -> dict:
    """Compute final success/failure_tag/ms_per_step, append the VF-1 outro
    stats card (once, on the final sub-goal's success), save the combined
    MP4, decide whether to keep the live sim alive for the next sub-goal
    (VF-3), and assemble run_fancy_rollout's own result dict (+ live_ctx).
    Mirrors the original run_fancy_rollout's own tail exactly."""
    from code.inferencer import FALL_HEIGHT
    from code.teacher import CONTROL_DECIMATION, SIM_DT, _yaw_of

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
        avoid_bias_active_frac=(avoid_cycles_active / avoid_cycles_total
                                 if avoid_cycles_total > 0 else 0.0),
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
            phase_tracker=phase_tracker,
            dist_traveled_m=dist_traveled_m, prev_rxy_odom=prev_rxy_odom,
        )
    return result
