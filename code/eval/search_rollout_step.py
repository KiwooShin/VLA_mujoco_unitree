"""code.eval.search_rollout_step — per-step control logic for the search rollout.

Split out of the original ``eval_search.py`` (RF-1): the body of what used to
be a single iteration of ``_run_search_rollout``'s ``for step in
range(maxsteps):`` loop (grounding cycle, lock/avoid bookkeeping, the
bidirectional-scan branch, and the normal GOTO student step), now a standalone
function operating on a shared, mutable ``_RolloutSetup`` (see
``code.eval.search_rollout_state``) instead of function-local/``nonlocal``
variables. This is a mechanical extraction: the control flow and numeric
logic are unchanged from the pre-RF-1 monolithic function, only the state
lives on an object instead of in closure cells (needed to split the loop body
into its own <500-line file).
"""

from __future__ import annotations

import math as _math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from code.inferencer import (
    _build_proprio, _apply_student_pd,
    FALL_HEIGHT, GROUNDING_PERIOD, HOLD_STEPS_REQUIRED, IMG_SIZE,
)
from code.arena import CAMERA_MODE, CAM_HEAD_Z
from code.teacher import _yaw_of, CONTROL_DECIMATION
from code.grounding import ground as classical_ground
from code.lock_mgmt import ReacquisitionScan
from code import avoid as _avoid

from code.eval.search_rollout_state import _RolloutSetup


def _lock_drop_and_rescan(setup: _RolloutSetup) -> None:
    """M4 (divergence) / M5 (coast-expiry) shared action: drop the lock,
    clear EMA/last-known-goal, and re-enter scan via a FRESH
    ReacquisitionScan. Not the same ``setup._scan_sched`` instance used for
    the initial scan: that scan's own outer SCAN_TIMEOUT check is keyed on
    the episode's ABSOLUTE step, so re-arming it mid-episode would
    immediately time out (see code/lock_mgmt.py's ReacquisitionScan
    docstring).

    Args:
        setup: The rollout's mutable state, updated in place.
    """
    setup._lock_gate.force_drop()
    setup._goal_ema           = None
    setup._last_known_goal    = None
    setup._frames_since_det   = 0
    setup._scan_active        = True
    setup._using_rescan_sched = True
    setup._rescan_sched       = ReacquisitionScan(scan_rate=setup.SCAN_RATE)
    setup.cached_goal_vec      = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    setup._avoid_bias_wz       = 0.0   # NX-9: fresh depth read once normal mode resumes


def _search_step(setup: _RolloutSetup, inf, step: int, render_video: bool) -> bool:
    """Runs one control step of the search rollout, mutating ``setup`` in place.

    Args:
        setup: The rollout's mutable state (see ``_RolloutSetup``), updated
            in place.
        inf: Inferencer instance (goal_source='classical') providing the
            model.
        step: Current absolute step index (0-based).
        render_video: Whether to capture ego/third-person frames this step.

    Returns:
        True if the rollout should stop (fell, or held inside stop_r for
        HOLD_STEPS_REQUIRED consecutive steps); False to continue.
    """
    t0 = time.perf_counter()

    data_mj  = setup.data_mj
    model_mj = setup.model_mj

    # Fall check
    height = float(data_mj.qpos[2])
    if height < FALL_HEIGHT:
        setup.fell = True
        return True

    yaw = _yaw_of(data_mj.qpos[3:7])

    # Grounding cadence
    need_grounding = (step - setup.last_grounding_step) >= GROUNDING_PERIOD
    need_render    = render_video or need_grounding

    intr_active = setup.intr   # default (cam2): the loop-invariant 45deg-FOVY intrinsics
    if need_render:
        if need_grounding:
            if CAMERA_MODE == 'widefov':
                # CAM-1 (Phase 2, toggle): single wide-FOV camera — use its own
                # per-call intrinsics (correct FOVY/pitch), not the cam2 `intr`.
                rgb, depth, intr_active = setup.renderer.render_widefov(
                    data_mj, yaw, render_depth=True)
            else:
                rgb, depth, _ = setup.renderer.render_grounding(data_mj, yaw, render_depth=True)
            if render_video:
                rgb_video, _, _ = setup.renderer.render_ego(data_mj, yaw, render_depth=False)
            else:
                rgb_video = None
        else:
            rgb, depth = None, None
            if render_video:
                rgb_video, _, _ = setup.renderer.render_ego(data_mj, yaw, render_depth=False)
            else:
                rgb_video = None
    else:
        rgb, depth, rgb_video = None, None, None

    # Classical grounding
    if need_grounding and rgb is not None and depth is not None:
        gr = classical_ground(rgb, depth, setup.target_color, setup.target_shape, intr_active)
        setup.last_grounding_step = step

        if not gr.not_visible:
            raw_goal = gr.goal_vec.copy()
            # NX-2 (LOCK_M1/M2/M3, docs/rs1_lock_mgmt.md): gate the raw detection
            # before it's allowed to feed the EMA/last-known-goal. Provable
            # pass-through (always True) with all three toggles off.
            _accept_hit = setup._lock_gate.gate_detection(
                float(raw_goal[0]), _math.atan2(raw_goal[2], raw_goal[1]), gr.best_area)
            if _accept_hit:
                setup._frames_since_det = 0
                if setup._goal_ema is None:
                    setup._goal_ema = raw_goal.copy()
                    setup._last_known_goal = raw_goal.copy()
                else:
                    setup._goal_ema = (setup._GOAL_EMA_ALPHA * raw_goal
                                        + (1.0 - setup._GOAL_EMA_ALPHA) * setup._goal_ema)
                    th = _math.atan2(setup._goal_ema[2], setup._goal_ema[1])
                    setup._goal_ema[1] = _math.cos(th)
                    setup._goal_ema[2] = _math.sin(th)
                    setup._last_known_goal = setup._goal_ema.copy()
                setup.cached_goal_vec = setup._goal_ema.copy()

                # Exit scan when aligned
                if setup._scan_active:
                    det_bearing = abs(_math.atan2(setup._goal_ema[2], setup._goal_ema[1]))
                    if det_bearing < setup.SCAN_ALIGNED_THR:
                        setup._scan_active = False
                        setup.spotted = True
                        print(f"  [search] SPOTTED at step={step}  "
                              f"bearing={_math.degrees(det_bearing):.1f}°", flush=True)
            else:
                # NX-2: gate rejected this detection -- treat this cycle like a miss.
                setup._frames_since_det += 1
                if (setup._last_known_goal is not None
                        and setup._frames_since_det <= setup.HOLD_GOAL_HORIZON):
                    setup.cached_goal_vec = setup._last_known_goal.copy()
                elif setup._lock_gate.coast_expired(setup._frames_since_det, setup.HOLD_GOAL_HORIZON):
                    print(f"  [lock] M5 coast expired (gate-rejected) -> "
                          f"drop+rescan at step={step}", flush=True)
                    _lock_drop_and_rescan(setup)
        else:
            setup._frames_since_det += 1
            if (setup._last_known_goal is not None
                    and setup._frames_since_det <= setup.HOLD_GOAL_HORIZON):
                setup.cached_goal_vec = setup._last_known_goal.copy()
            elif setup._lock_gate.coast_expired(setup._frames_since_det, setup.HOLD_GOAL_HORIZON):
                # NX-2 (LOCK_M5): bounded coast -> reroute to rescan instead of an
                # unbounded silent freeze.
                print(f"  [lock] M5 coast expired -> drop+rescan at step={step}", flush=True)
                _lock_drop_and_rescan(setup)

        # NX-2 (LOCK_M4): divergence watchdog -- runs once per grounding cycle
        # regardless of hit/miss/gate outcome above. Provable no-op when off.
        # NX-5 (LOCK_M7, docs/nx5_coherence.md): odometry-coherence watchdog --
        # see code/inferencer.py's identical block for the projection derivation.
        _walking_toward_goal = (not setup._scan_active) and (float(setup.cached_goal_vec[0]) > setup.stop_r)
        _m7_proj_disp_m = 0.0
        _cur_xy = data_mj.qpos[0:2].copy()
        if setup._m7_prev_xy is not None:
            _dxw = float(_cur_xy[0] - setup._m7_prev_xy[0])
            _dyw = float(_cur_xy[1] - setup._m7_prev_xy[1])
            _cy, _sy = _math.cos(yaw), _math.sin(yaw)
            _d_body_x =  _dxw * _cy + _dyw * _sy
            _d_body_y = -_dxw * _sy + _dyw * _cy
            _m7_proj_disp_m = (_d_body_x * float(setup.cached_goal_vec[1])
                                + _d_body_y * float(setup.cached_goal_vec[2]))
        setup._m7_prev_xy = _cur_xy
        if setup._lock_gate.end_of_cycle(float(setup.cached_goal_vec[0]), _walking_toward_goal,
                                          _m7_proj_disp_m):
            reason = 'M4 divergence' if setup._lock_gate.last_trigger == 'M4' else 'M7 coherence'
            print(f"  [lock] {reason} -> drop+rescan at step={step}", flush=True)
            _lock_drop_and_rescan(setup)

        # NX-9 AVOID (docs/nx9_avoid.md): local obstacle avoidance --
        # same mechanism/carve-outs as code/inferencer.py's identical
        # block (shared helper, code/avoid.py), reusing this cycle's
        # already-rendered depth/intr_active (zero extra renders).
        # Never while `_scan_active`; fresh bias only while the goal is
        # fresh (<= AVOID_STALE_MAX_MISSED_CYCLES missed cycles), decay
        # only on a longer stale coast -- see AVOID_STALE_MAX_MISSED_CYCLES'
        # comment in code/avoid.py for the ep14 fall trace behind this.
        if _avoid.AVOID and not setup._avoid_is_maneuver and not setup._scan_active:
            setup._avoid_cycles_total += 1
            if setup._frames_since_det > _avoid.AVOID_STALE_MAX_MISSED_CYCLES:
                setup._avoid_bias_wz = _avoid.decay_bias(setup._avoid_bias_wz)
            else:
                _avoid_goal_dist_now = float(setup.cached_goal_vec[0])
                _avoid_goal_bearing_now = _math.atan2(float(setup.cached_goal_vec[2]),
                                                       float(setup.cached_goal_vec[1]))
                _avoid_carved = (_avoid_goal_dist_now < _avoid.AVOID_MIN_GOAL_DIST_M)
                _avoid_cam_h = float(data_mj.qpos[2]) + CAM_HEAD_Z
                setup._avoid_bias_wz, _avoid_dbg = _avoid.compute_obstacle_bias(
                    depth, intr_active, cam_height_m=_avoid_cam_h,
                    goal_dist_m=_avoid_goal_dist_now,
                    goal_bearing_rad=_avoid_goal_bearing_now,
                    prev_bias_wz=setup._avoid_bias_wz, carved_out=_avoid_carved)
            if abs(setup._avoid_bias_wz) > 1e-9:
                setup._avoid_cycles_active += 1

    # Scan mode: NX-1 bidirectional bounded-rotation sweep (see setup module).
    # Observable (memoryless per-frame visibility check), WBC-free.
    # SCAN_TIMEOUT=900 is a safety-net cap — the schedule itself normally
    # exits (spotted) well before that.
    if setup._scan_active:
        if setup._using_rescan_sched:
            # NX-2 (LOCK_M4/M5): a lock-drop-triggered rescan uses a FRESH
            # ReacquisitionScan (local step counter) rather than re-arming
            # `_scan_sched`/`SCAN_TIMEOUT`, which is keyed on the episode's
            # absolute step and would immediately time out mid-episode.
            scan_wz = setup._rescan_sched.step(yaw)
            if scan_wz is None:
                setup._scan_active        = False
                setup._using_rescan_sched = False
                print(f"  [lock][rescan] TIMEOUT at step={step}, no target spotted", flush=True)
            else:
                setup.scan_steps += 1
                prop_now = _build_proprio(data_mj, setup.prev_action)
                if setup._use_phase:
                    ph = setup._phase_tracker.update(data_mj.qpos[7:22].copy())
                    prop_now = np.concatenate([prop_now, ph])
                setup.proprio_hist.append(prop_now)
                prop_arr = np.stack(list(setup.proprio_hist), axis=0)
                prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(inf.device)

                img_t_scan = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32,
                                         device=inf.device)
                scan_goal_t = torch.from_numpy(setup.cached_goal_vec).unsqueeze(0).to(inf.device)
                scan_vel_t  = torch.tensor([[0.0, 0.0, scan_wz]], dtype=torch.float32,
                                           device=inf.device)

                with torch.no_grad():
                    out_scan = inf.model(
                        ego_rgb   = img_t_scan,
                        lang_emb  = setup.lang_t,
                        proprio_h = prop_t,
                        gt_goal   = scan_goal_t,
                        gt_vel    = scan_vel_t,
                    )

                raw_scan = out_scan['action'].cpu().numpy().squeeze(0)[0]
                if setup._use_residual:
                    target_dof = setup._da_deflt + raw_scan * setup._da_std + setup._da_mean
                else:
                    target_dof = raw_scan

                for _ in range(CONTROL_DECIMATION):
                    _apply_student_pd(data_mj, target_dof, setup.nj)
                    mujoco.mj_step(model_mj, data_mj)

                setup.prev_action = target_dof.copy()
                setup._all_target_dofs.append(setup.prev_action.copy())
                setup.steps_done = step + 1

                if render_video and rgb_video is not None:
                    setup.frames_ego.append(rgb_video.copy())
                    setup.renderer.update_tp_cam(setup.tp_cam, data_mj)
                    setup.frames_tp.append(setup.renderer.render_tp(data_mj, setup.tp_cam).copy())

                t1 = time.perf_counter()
                setup.step_times.append((t1 - t0) * 1000.0)
                return False   # skip normal student step
        elif step >= setup.SCAN_TIMEOUT:
            setup._scan_active = False   # timeout — fallback to default goal, not spotted
            print(f"  [search] SCAN TIMEOUT at step={step}, no target spotted", flush=True)
        else:
            setup.scan_steps += 1
            scan_wz = setup._scan_sched.step(yaw)   # bounded CCW/CW schedule, dwells at 0.0
            setup._scan_yaw_delta += scan_wz * setup.SCAN_DT

            # Student forward pass with injected wz
            prop_now = _build_proprio(data_mj, setup.prev_action)
            if setup._use_phase:
                ph = setup._phase_tracker.update(data_mj.qpos[7:22].copy())
                prop_now = np.concatenate([prop_now, ph])
            setup.proprio_hist.append(prop_now)
            prop_arr = np.stack(list(setup.proprio_hist), axis=0)
            prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(inf.device)

            img_t_scan = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32,
                                     device=inf.device)
            scan_goal_t = torch.from_numpy(setup.cached_goal_vec).unsqueeze(0).to(inf.device)
            scan_vel_t  = torch.tensor([[0.0, 0.0, scan_wz]], dtype=torch.float32,
                                       device=inf.device)

            with torch.no_grad():
                out_scan = inf.model(
                    ego_rgb   = img_t_scan,
                    lang_emb  = setup.lang_t,
                    proprio_h = prop_t,
                    gt_goal   = scan_goal_t,
                    gt_vel    = scan_vel_t,
                )

            raw_scan = out_scan['action'].cpu().numpy().squeeze(0)[0]
            if setup._use_residual:
                target_dof = setup._da_deflt + raw_scan * setup._da_std + setup._da_mean
            else:
                target_dof = raw_scan

            for _ in range(CONTROL_DECIMATION):
                _apply_student_pd(data_mj, target_dof, setup.nj)
                mujoco.mj_step(model_mj, data_mj)

            setup.prev_action = target_dof.copy()
            setup._all_target_dofs.append(setup.prev_action.copy())
            setup.steps_done = step + 1

            if render_video and rgb_video is not None:
                setup.frames_ego.append(rgb_video.copy())
                setup.renderer.update_tp_cam(setup.tp_cam, data_mj)
                setup.frames_tp.append(setup.renderer.render_tp(data_mj, setup.tp_cam).copy())

            t1 = time.perf_counter()
            setup.step_times.append((t1 - t0) * 1000.0)
            return False   # skip normal student step

    # Normal GOTO student step (after spotted or scan timeout)
    prop_now = _build_proprio(data_mj, setup.prev_action)
    if setup._use_phase:
        ph = setup._phase_tracker.update(data_mj.qpos[7:22].copy())
        prop_now = np.concatenate([prop_now, ph])
    setup.proprio_hist.append(prop_now)
    prop_arr = np.stack(list(setup.proprio_hist), axis=0)
    prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(inf.device)

    img_t      = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32, device=inf.device)
    goal_inj_t = torch.from_numpy(setup.cached_goal_vec).unsqueeze(0).to(inf.device)

    # NX-9 AVOID: replace the model's self-predicted velocity with
    # steer.py's own control law (from cached_goal_vec) plus the bounded
    # yaw bias, exactly matching code/inferencer.py's injection point --
    # only when a nonzero bias is active this cycle (provable no-op on
    # clear paths / when AVOID is off).
    gt_vel_t = None
    if _avoid.AVOID and not setup._avoid_is_maneuver and abs(setup._avoid_bias_wz) > 1e-9:
        vel_av = _avoid.biased_vel_cmd(
            float(setup.cached_goal_vec[0]), float(setup.cached_goal_vec[1]),
            float(setup.cached_goal_vec[2]), setup._avoid_bias_wz, setup.stop_r)
        gt_vel_t = torch.from_numpy(vel_av).unsqueeze(0).to(inf.device)

    with torch.no_grad():
        out = inf.model(
            ego_rgb   = img_t,
            lang_emb  = setup.lang_t,
            proprio_h = prop_t,
            gt_goal   = goal_inj_t,
            gt_vel    = gt_vel_t,
        )

    raw_action = out['action'].cpu().numpy().squeeze(0)[0]
    if setup._use_residual:
        student_target_dof = setup._da_deflt + raw_action * setup._da_std + setup._da_mean
    else:
        student_target_dof = raw_action

    setup._all_target_dofs.append(student_target_dof.copy())

    for _ in range(CONTROL_DECIMATION):
        _apply_student_pd(data_mj, student_target_dof, setup.nj)
        mujoco.mj_step(model_mj, data_mj)

    setup.prev_action = student_target_dof.copy()
    setup.steps_done  = step + 1

    if render_video and rgb_video is not None:
        setup.frames_ego.append(rgb_video.copy())
        setup.renderer.update_tp_cam(setup.tp_cam, data_mj)
        setup.frames_tp.append(setup.renderer.render_tp(data_mj, setup.tp_cam).copy())

    t1 = time.perf_counter()
    setup.step_times.append((t1 - t0) * 1000.0)

    dist_to_target = float(np.linalg.norm(data_mj.qpos[0:2] - setup.target_xy))
    if dist_to_target < setup.stop_r:
        setup.hold_counter += 1
        if setup.hold_counter >= HOLD_STEPS_REQUIRED:
            return True
    else:
        setup.hold_counter = 0

    if step % 100 == 0:
        dist_to_target = float(np.linalg.norm(data_mj.qpos[0:2] - setup.target_xy))
        print(f"  step={step:4d}  dist={dist_to_target:.2f}m  "
              f"scan={'ON' if setup._scan_active else 'OFF'}  "
              f"spotted={setup.spotted}  h={height:.3f}m", flush=True)

    return False
