"""
code.runtime.rollout_step — per-step control logic for `Inferencer.rollout()`.

RF-1 split of code/inferencer.py (docs/refactor_plan.md): the body of what
used to be a single iteration of `rollout()`'s `for step in
range(maxsteps):` loop, now a standalone function operating on a shared,
mutable `RolloutState` + `GoalPipeline` (code.runtime.rollout_state /
code.runtime.goal_pipeline) instead of function-local/`nonlocal` variables.
This is a mechanical extraction: the control flow and numeric logic are
unchanged from the pre-RF-1 monolithic function (mirrors
code/eval/search_rollout_step.py's precedent for the same kind of split).

`classical_ground` calls go through `inf._ground(...)` (defined in
code/runtime/inferencer.py) rather than a name imported directly into this
file — see that module's docstring for why (monkeypatch preservation for
`code.inferencer.classical_ground`).

One deliberate, zero-behavior-change tidy vs. the original: the original had
TWO textually near-identical "scan-mode student forward pass -> PD -> physics"
blocks (one under the M4/M5/M7-triggered ReacquisitionScan branch, one under
the original H3 branch), differing only in a single write-only `dist_to_target`
local (computed, never read, no side effects) present in the H3 copy only.
`GoalPipeline.try_scan_step` now resolves which branch supplied `scan_wz`
before returning, so this file has one such block, executed whenever
`try_scan_step` returns a value — this is provably behavior-identical (the
dropped line had zero observable effect) and is NOT the cross-file rollout-
loop duplication docs/refactor_plan.md invariant 4 protects (eval_search /
fancy_demo / demo REPL stay independently duplicated, untouched).
"""

from __future__ import annotations

import math
import time

import mujoco
import numpy as np
import torch

from code.sim.arena import CAMERA_MODE, EGO_W, EGO_H
from code.sim.teacher import _yaw_of, CONTROL_DECIMATION
from code.control.steer import steer as _steer_cmd, MAX_VX, MAX_WZ, YAW_KP, FACE_THR_RAD, DECEL_DIST
from code.control import avoid as _avoid
from code.runtime.constants import (
    FALL_HEIGHT, IMG_SIZE, HOLD_STEPS_REQUIRED, AVOID,
    STALL_BREAK, STALL_VX_THR_MPS, STALL_WINDOW_STEPS, STALL_DISP_THR_M,
    STALL_MIN_GOAL_DIST_M, STALL_RECOVERY_STEPS, STALL_COOLDOWN_STEPS,
)
from code.runtime.helpers import _build_proprio, _apply_student_pd, _rgb_to_tensor, _label_active_cam
from code.runtime.gt_goal import _compute_gt_goal
from code.runtime.rollout_state import RolloutState


def _rollout_step(inf, s: RolloutState, step: int, render_video: bool, render_tp: bool) -> bool:
    """Runs one control step of the rollout, mutating `s` (and `s.goal_pipeline`)
    in place.

    Args:
        inf: Inferencer instance providing the model, device, chunk_H,
            verbose flag, and the `_ground` classical-grounding call site.
        s: The rollout's mutable state (see `RolloutState`), updated in place.
        step: Current absolute step index (0-based).
        render_video: Whether to capture ego/third-person frames this step.
        render_tp: Whether to also capture third-person frames (only
            consulted when `render_video` is True).

    Returns:
        True if the rollout should stop (fell, or held inside `s.stop_r` for
        HOLD_STEPS_REQUIRED consecutive steps); False to continue.
    """
    gp = s.goal_pipeline
    t0 = time.perf_counter()

    # Height check
    height = float(s.data_mj.qpos[2])
    if height < FALL_HEIGHT:
        s.fell = True
        return True

    # Current yaw
    yaw = _yaw_of(s.data_mj.qpos[3:7])

    # Rendering: needed for classical/learned grounding or video recording
    need_classical_grounding = gp.due_for_classical_grounding(step)
    need_learned_grounding   = gp.due_for_learned_grounding(step)
    need_render = render_video or need_classical_grounding or need_learned_grounding

    intr_active = None   # intrinsics of whichever camera was actually rendered below
    if need_render:
        if need_classical_grounding:
            if CAMERA_MODE == 'widefov':
                rgb, depth, intr_active = s.renderer.render_widefov(
                    s.data_mj, yaw, render_depth=True)
            elif gp.active_cam == 'PROXIMITY':
                rgb, depth, intr_active = s.renderer.render_proximity(
                    s.data_mj, yaw, render_depth=True)
            else:
                rgb, depth, intr_active = s.renderer.render_grounding(
                    s.data_mj, yaw, render_depth=True)
            if render_video:
                cam_label = 'WIDEFOV' if CAMERA_MODE == 'widefov' else gp.active_cam
                gp.video_frame_cache = _label_active_cam(
                    rgb, cam_label, float(gp.cached_goal_vec[0]),
                    resize_to=(EGO_W, EGO_H))
                rgb_video = gp.video_frame_cache
            else:
                rgb_video = rgb   # unused (render_video=False)
        else:
            if render_video and gp.need_classical_render and gp.video_frame_cache is not None:
                rgb, depth = None, None
                rgb_video  = gp.video_frame_cache
            else:
                rgb, depth, _intr = s.renderer.render_ego(s.data_mj, yaw,
                                                          render_depth=s.need_learned_render)
                rgb_video = rgb
    else:
        rgb, depth = None, None
        rgb_video  = None

    # Grounding (Arch A, goal_source='classical', at ~5 Hz)
    if need_classical_grounding and rgb is not None and depth is not None:
        gr = inf._ground(rgb, depth, s.target_color, s.target_shape, intr_active)
        gp.mark_grounding_step(step)
        gp.register_detection_outcome(gr.not_visible)

        # CAM-2 (Phase 1): bounded fallback probe.
        if gr.not_visible:
            other_cam = gp.maybe_probe_camera()
            if other_cam is not None:
                if other_cam == 'PROXIMITY':
                    rgb2, depth2, intr2 = s.renderer.render_proximity(
                        s.data_mj, yaw, render_depth=True)
                else:
                    rgb2, depth2, intr2 = s.renderer.render_grounding(
                        s.data_mj, yaw, render_depth=True)
                gr2 = inf._ground(rgb2, depth2, s.target_color, s.target_shape, intr2)
                if not gr2.not_visible:
                    gr = gr2
                    gp.on_probe_adopted(other_cam)

        gp.process_classical_detection(gr, step)

        # NX-2 (LOCK_M4)/NX-5 (LOCK_M7): end-of-cycle watchdog.
        gp.end_of_cycle_lock_check(s.data_mj, yaw, s.stop_r, step)

        # NX-9 AVOID: per-cycle obstacle-bias update.
        gp.update_avoid_bias(depth, intr_active, s.data_mj, step)

    # Learned grounding (Arch A, goal_source='learned', trained grounding head)
    if need_learned_grounding and rgb is not None:
        gp.mark_grounding_step(step)
        img_t_gr = _rgb_to_tensor(rgb, inf.device)
        with torch.no_grad():
            out_gr = inf.model(
                ego_rgb   = img_t_gr,
                lang_emb  = s.lang_t,
                proprio_h = torch.zeros(1, 6, s.eff_proprio_dim, device=inf.device),
                gt_goal   = None,
                gt_vel    = None,
            )
        raw_gr = out_gr['goal'].cpu().numpy().squeeze(0)   # (3,)
        norm_gr = math.sqrt(raw_gr[1]**2 + raw_gr[2]**2 + 1e-6)
        raw_gr[1] /= norm_gr
        raw_gr[2] /= norm_gr
        gp.process_learned_detection(raw_gr)

    # GT goal (Arch A, goal_source='gt'): privileged sim-state goal, updated every step
    if s.use_gt_goal:
        gp.cached_goal_vec = _compute_gt_goal(s.data_mj, s.target_xy)

    # ---- H3: scan-and-acquire — STUDENT-DRIVEN (WBC-free) ----
    scan_wz = gp.try_scan_step(yaw, step)
    if scan_wz is not None:
        prop_now = _build_proprio(s.data_mj, s.prev_action)
        if s.use_phase:
            q_lb_now = s.data_mj.qpos[7:22].copy()
            ph_now   = s.phase_tracker.update(q_lb_now)
            prop_now = np.concatenate([prop_now, ph_now])
        s.proprio_hist.append(prop_now)
        prop_arr = np.stack(list(s.proprio_hist), axis=0)
        prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(inf.device)
        img_t_scan = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE,
                                 dtype=torch.float32, device=inf.device)
        scan_goal_t = torch.from_numpy(gp.cached_goal_vec).unsqueeze(0).to(inf.device)
        scan_vel_cmd = np.array([0.0, 0.0, scan_wz], dtype=np.float32)
        scan_vel_t   = torch.from_numpy(scan_vel_cmd).unsqueeze(0).to(inf.device)
        with torch.no_grad():
            out_scan = inf.model(
                ego_rgb   = img_t_scan,
                lang_emb  = s.lang_t,
                proprio_h = prop_t,
                gt_goal   = scan_goal_t,
                gt_vel    = scan_vel_t,
            )
        actions_scan = out_scan['action'].cpu().numpy().squeeze(0)
        raw_action_scan = actions_scan[0]
        if s.use_residual:
            scan_target_dof = s.da_deflt + raw_action_scan * s.da_std + s.da_mean
        else:
            scan_target_dof = raw_action_scan
        for _ in range(CONTROL_DECIMATION):
            _apply_student_pd(s.data_mj, scan_target_dof, s.nj)
            mujoco.mj_step(s.model_mj, s.data_mj)
        s.prev_action = scan_target_dof.copy()
        s.all_target_dofs.append(s.prev_action.copy())
        s.steps_done = step + 1
        t1 = time.perf_counter()
        s.step_times.append((t1 - t0) * 1000.0)
        if render_video and rgb_video is not None:
            s.frames_ego.append(rgb_video.copy())
            if render_tp:
                s.renderer.update_tp_cam(s.tp_cam, s.data_mj)
                s.frames_tp.append(s.renderer.render_tp(s.data_mj, s.tp_cam).copy())
        return False   # skip student forward pass (already done above)

    # ---- Student forward pass (normal mode) ----
    prop_now = _build_proprio(s.data_mj, s.prev_action)
    if s.use_phase:
        q_lb_now = s.data_mj.qpos[7:22].copy()
        ph_now   = s.phase_tracker.update(q_lb_now)
        prop_now = np.concatenate([prop_now, ph_now])
    s.proprio_hist.append(prop_now)
    prop_arr = np.stack(list(s.proprio_hist), axis=0)
    prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(inf.device)

    if rgb is not None:
        img_t = _rgb_to_tensor(rgb, inf.device)
    else:
        img_t = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32,
                            device=inf.device)

    if s.inject_cached:
        goal_inject_t = torch.from_numpy(gp.cached_goal_vec).unsqueeze(0).to(inf.device)
    else:
        goal_inject_t = None

    if s.need_learned_render and rgb is not None:
        img_t = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32,
                            device=inf.device)

    # Fix 2: GT velocity injection
    gt_vel_inject_t = None
    if s.inject_gt_vel:
        robot_xy  = s.data_mj.qpos[0:2].copy()
        robot_yaw = _yaw_of(s.data_mj.qpos[3:7])
        gt_vel_cmd, _, _ = _steer_cmd(robot_xy, robot_yaw, s.target_xy, s.stop_r)
        gt_vel_inject_t  = torch.from_numpy(gt_vel_cmd).unsqueeze(0).to(inf.device)

    # Learned-grounding velocity injection
    if s.need_learned_render and s.inject_cached and gt_vel_inject_t is None:
        d_gr  = float(gp.cached_goal_vec[0])
        cs_gr = float(gp.cached_goal_vec[1])
        sn_gr = float(gp.cached_goal_vec[2])
        ye_gr = math.atan2(sn_gr, cs_gr)  # yaw_err from grounding
        if d_gr < s.stop_r:
            vel_gr = np.zeros(3, dtype=np.float32)
        else:
            wz_gr = float(np.clip(YAW_KP * ye_gr, -MAX_WZ, MAX_WZ))
            if abs(ye_gr) > FACE_THR_RAD:
                vx_gr = 0.0
            else:
                decel_gr = min(1.0, max(0.0, (d_gr - s.stop_r) / max(DECEL_DIST - s.stop_r, 0.1)))
                vx_gr = float(np.clip(MAX_VX * max(0.0, cs_gr) * decel_gr, 0.0, MAX_VX))
            vel_gr = np.array([vx_gr, 0.0, wz_gr], dtype=np.float32)
        gt_vel_inject_t = torch.from_numpy(vel_gr).unsqueeze(0).to(inf.device)

    # NX-9 AVOID: integrate the obstacle-bias term into the velocity command.
    if AVOID and not gp.avoid_is_maneuver and gt_vel_inject_t is None and abs(gp.avoid_bias_wz) > 1e-9:
        vel_av = _avoid.biased_vel_cmd(
            float(gp.cached_goal_vec[0]), float(gp.cached_goal_vec[1]),
            float(gp.cached_goal_vec[2]), gp.avoid_bias_wz, s.stop_r)
        gt_vel_inject_t = torch.from_numpy(vel_av).unsqueeze(0).to(inf.device)

    # NX-8 STALL_BREAK: forced stop during an active recovery window.
    if STALL_BREAK and s.stall_recovery_remaining > 0:
        gt_vel_inject_t = torch.zeros(1, 3, dtype=torch.float32, device=inf.device)
        s.stall_recovery_remaining -= 1
        if s.stall_recovery_remaining == 0:
            s.stall_cooldown_remaining = STALL_COOLDOWN_STEPS
            s.stall_hist.clear()   # fresh window once normal control resumes

    # Student forward pass
    with torch.no_grad():
        out = inf.model(
            ego_rgb   = img_t,
            lang_emb  = s.lang_t,
            proprio_h = prop_t,
            gt_goal   = goal_inject_t,    # None → model predicts; tensor → injected
            gt_vel    = gt_vel_inject_t,   # Fix 2: None → vel head predicts; tensor → injected
        )

    # NX-8 STALL_BREAK: capture the commanded v_fwd this cycle.
    if STALL_BREAK and not s.stall_is_maneuver and 'vel' in out and out['vel'] is not None:
        s.cur_vx_cmd = float(out['vel'][0, 0].item())

    # Extract action chunk
    actions_raw = out['action'].cpu().numpy().squeeze(0)   # (H, 15)

    # Temporal ensembling (H > 1)
    if inf.chunk_H > 1:
        H = inf.chunk_H
        wt = np.exp(-0.1 * np.arange(H, dtype=np.float32))
        s.te_buffer.append((step, wt, actions_raw.copy()))
        s.te_buffer = [(st, w, a) for (st, w, a) in s.te_buffer if step - st < H]
        act_sum = np.zeros(15, dtype=np.float32)
        w_sum   = 0.0
        for (st, w, a) in s.te_buffer:
            k = step - st
            if 0 <= k < H:
                act_sum += w[k] * a[k]
                w_sum   += w[k]
        raw_action = (act_sum / w_sum) if w_sum > 1e-9 else actions_raw[0]
    else:
        raw_action = actions_raw[0]   # (15,)

    # Convert model output → absolute joint targets.
    if s.use_residual:
        s.student_target_dof = s.da_deflt + raw_action * s.da_std + s.da_mean
    else:
        s.student_target_dof = raw_action  # already absolute joint angles (old mode)

    # Track commanded targets for oscillation check
    s.all_target_dofs.append(s.student_target_dof.copy())

    # Apply PD + physics substeps (student drives physics, no teacher here)
    for _ in range(CONTROL_DECIMATION):
        _apply_student_pd(s.data_mj, s.student_target_dof, s.nj)
        mujoco.mj_step(s.model_mj, s.data_mj)

    s.prev_action = s.student_target_dof.copy()
    s.steps_done  = step + 1

    # Record frames (always use ego-resolution rgb_video to keep size consistent)
    if render_video and rgb_video is not None:
        s.frames_ego.append(rgb_video.copy())
        if render_tp:
            s.renderer.update_tp_cam(s.tp_cam, s.data_mj)
            s.frames_tp.append(s.renderer.render_tp(s.data_mj, s.tp_cam).copy())

    t1 = time.perf_counter()
    s.step_times.append((t1 - t0) * 1000.0)

    # Distance to target
    dist_to_target = float(np.linalg.norm(s.data_mj.qpos[0:2] - s.target_xy))

    # NX-8 STALL_BREAK: window update + trigger check.
    if STALL_BREAK and not s.stall_is_maneuver:
        if s.stall_cooldown_remaining > 0:
            s.stall_cooldown_remaining -= 1
        if s.stall_recovery_remaining > 0:
            pass   # still forcing the stop -- don't feed/check the window
        else:
            now_xy = s.data_mj.qpos[0:2]
            s.stall_hist.append((float(now_xy[0]), float(now_xy[1]), s.cur_vx_cmd))
            if (s.stall_cooldown_remaining == 0
                    and len(s.stall_hist) >= STALL_WINDOW_STEPS
                    and float(gp.cached_goal_vec[0]) > STALL_MIN_GOAL_DIST_M):
                x0, y0, _ = s.stall_hist[0]
                x1, y1, _ = s.stall_hist[-1]
                disp = math.hypot(x1 - x0, y1 - y0)
                sustained = all(abs(v) > STALL_VX_THR_MPS for (_, _, v) in s.stall_hist)
                if sustained and disp < STALL_DISP_THR_M:
                    s.stall_recovery_remaining = STALL_RECOVERY_STEPS
                    s.stall_trigger_count      += 1
                    s.stall_hist.clear()
                    if inf.verbose:
                        print(f"  [stall] STALL_BREAK #{s.stall_trigger_count} "
                              f"triggered at step={step} disp={disp:.3f}m "
                              f"goal_dist={float(gp.cached_goal_vec[0]):.2f}m -> "
                              f"forcing stop for {STALL_RECOVERY_STEPS} steps",
                              flush=True)

    # Success check
    if dist_to_target < s.stop_r:
        s.hold_counter += 1
        if s.hold_counter >= HOLD_STEPS_REQUIRED:
            return True
    else:
        s.hold_counter = 0

    if inf.verbose and step % 50 == 0:
        ms = (t1 - t0) * 1000.0
        print(f"  step={step:4d}  dist={dist_to_target:.2f}m  h={height:.3f}m  "
              f"ms={ms:.1f}  hold={s.hold_counter}", flush=True)

    return False
