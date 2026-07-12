"""Core search-then-goto rollout for the fancy demo (code/fancy_demo.py):
`run_fancy_rollout` -- BEV follow-cam + VF-1 overlays.

RF-2a split: this file used to hold `run_fancy_rollout` as a single
~960-line function built around ~15 nested closures sharing episode-scoped
mutable state (scan schedule, CAM-2 handoff, NX-9 AVOID bias, NX-16
lock/rescan, path trail, telemetry). It is now split three ways:

  - `rollout.py` (this file): the function signature/docstring and the
    step-by-step control loop itself -- state machine, scan-vs-goto
    dispatch, model inference + PD stepping, telemetry/print, and the
    pre-/post-loop title-card and outro-card/video-save bookkeeping. Owns
    ALL of the loop's persistent local state (nothing lives in a shared
    object); every helper below is a plain function call that takes its
    inputs explicitly and returns its outputs explicitly -- no
    nonlocal/closure capture crosses the file boundary.
  - `code/apps/fancy/rollout_state.py`: the one-time sim build/resume
    (fresh arena vs VF-3 `resume_ctx`), and the CAM-2 handoff / NX-9 AVOID /
    NX-16 lock-rescan per-cycle *computations* (each a pure-ish function of
    this cycle's inputs, called once per step from the loop below).
  - `code/apps/fancy/rollout_frames.py`: BEV camera follow, the HUD
    camera-flash timer, the skill-stage breadcrumb mapping, and the full
    ego|BEV side-by-side frame composition for one rendered step.

See those two modules' own docstrings for the mechanism-level rationale
that used to live inline here (NX-16 coast-expiry recovery, CAM-2 Schmitt
trigger, NX-9 AVOID carve-outs); the comments in THIS file now only cover
what's still local to the control loop itself.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, List, Optional

import numpy as np

from code.apps.fancy.constants import (
    EGO_W, EGO_H, FEAT_TITLECARD,
    GOTO_CKPT_DEFAULT, MAXSTEPS_FANCY,
    STATE_IDLE, STATE_LOCATED, STATE_MOVING, STATE_REACHED, STATE_SEARCHING,
)
from code.apps.fancy.cards import _final_canvas_dims, make_title_card
from code.apps.fancy import rollout_frames as _frames
from code.apps.fancy import rollout_setup
from code.apps.fancy import rollout_state


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
    import mujoco
    import torch
    import math as _math
    from code.inferencer import (
        _build_proprio, _apply_student_pd, _label_active_cam,
        FALL_HEIGHT, GROUNDING_PERIOD, HOLD_STEPS_REQUIRED, ACTION_SCALE, IMG_SIZE,
    )
    from code.arena import GROUNDING_W, GROUNDING_H
    from code.teacher import _yaw_of, DEFAULT_ANGLES, KPS, KDS, NUM_ACTIONS, CONTROL_DECIMATION
    from code.grounding import ground as classical_ground, get_ego_intrinsics_rendered
    from code.steer import steer as _steer_cmd
    from code.eval_search import STOP_R_SEARCH
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

    # --- Build a fresh MuJoCo env, or (VF-3) resume a live one -- see
    # rollout_state.build_or_resume_sim's own docstring for the full
    # fresh-build-vs-resume contract (unchanged from before the split).
    sim, _fall_result = rollout_state.build_or_resume_sim(inf, scene_cfg, resume_ctx, goal_idx, target_xy)
    if sim is None:
        return _fall_result
    teacher, data_mj, model_mj, nj, renderer, bev_cam, rx, ry = sim

    # --- Load action stats + gait-phase/proprio-history state from the
    # inferencer (rollout_setup.init_policy_state; fresh vs VF-3-resumed). ---
    (_use_residual, _da_mean, _da_std, _da_deflt, _use_phase, _phase_tracker, _eff_pdim,
     prev_action, proprio_hist, lang_t) = rollout_setup.init_policy_state(inf, resume_ctx, teacher, data_mj)

    # Scan state — NX-1 bidirectional bounded-rotation sweep + NX-16 coast-
    # expiry rescan bookkeeping (rollout_setup.init_scan_state; see that
    # function's docstring and rollout_state.reset_lock_and_rescan's
    # module-level comment for the full mechanism/rationale).
    (cached_goal_vec, last_grounding_step, _scan_active, SCAN_TIMEOUT, SCAN_RATE, SCAN_DT,
     SCAN_ALIGNED_THR, _goal_ema, _last_known_goal, _frames_since_det, HOLD_GOAL_HORIZON,
     _scan_yaw_delta, _scan_sched, _using_rescan_sched, _rescan_sched, _rescan_local_steps
     ) = rollout_setup.init_scan_state()

    # CAM-2 ego-camera-handoff + NX-9 AVOID per-episode state
    # (rollout_setup.init_cam_avoid_state; see rollout_state.py's
    # cam_handoff()/handle_camera_probe()/avoid_bias_step() for the per-cycle
    # mechanism these seed).
    (_active_cam, _cam_miss_count, _video_frame_cache, _avoid_bias_wz, _avoid_is_maneuver,
     _avoid_cycles_total, _avoid_cycles_active, _last_avoid_dbg
     ) = rollout_setup.init_cam_avoid_state(scene_cfg)

    # Path trail / collected-frames / hold-counter / VF-1 telemetry
    # (odometry + HUD camera-flash timer) state (rollout_setup.init_telemetry_state).
    (spotted, scan_steps, path_trail, _completed_targets, frames_sbs, step_times, hold_counter,
     fell, steps_done, current_state, current_dist, dist_traveled_m, _prev_rxy_odom, _hud_state,
     CAM_FLASH_FRAMES) = rollout_setup.init_telemetry_state(
        resume_ctx, path_trail_in, completed_targets, rx, ry, data_mj, target_xy)

    TRAIL_SUBSAMPLE = 3    # record every N steps

    def _render_sbs_frame() -> tuple[np.ndarray, float]:
        """Render ACTIVE-camera ego feed + BEV + overlays → SBS frame.
        Thin trampoline: forwards this call's (still function-local) episode
        state into rollout_frames.render_sbs_frame() -- see that module for
        the full implementation."""
        return _frames.render_sbs_frame(
            renderer=renderer, data_mj=data_mj, target_xy=target_xy, target_obj=target_obj,
            path_trail=path_trail, current_state=current_state, prompt=prompt,
            goal_idx=goal_idx, n_goals=n_goals, completed_targets=_completed_targets,
            bev_cam=bev_cam, model_mj=model_mj, video_frame_cache=_video_frame_cache,
            avoid_bias_wz=_avoid_bias_wz, avoid_info=_last_avoid_dbg,
            active_cam=_active_cam, step=step, hud_state=_hud_state,
            cam_flash_frames=CAM_FLASH_FRAMES,
            target_color=target_color, target_shape=target_shape,
            scan_active=_scan_active,
        )

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

            # CAM-2 bounded fallback probe (docs/cam_p1.md) -- see
            # rollout_state.handle_camera_probe() for the mechanism.
            if gr.not_visible:
                gr, _active_cam, _cam_miss_count, _probe_frame = rollout_state.handle_camera_probe(
                    gr, _active_cam, _cam_miss_count, _last_known_goal, renderer, data_mj, yaw,
                    target_color, target_shape, render_video, EGO_W, EGO_H)
                if _probe_frame is not None:
                    _video_frame_cache = _probe_frame
            else:
                _cam_miss_count = 0

            if not gr.not_visible:
                _frames_since_det = 0
                raw_goal = gr.goal_vec.copy()
                _goal_ema, _last_known_goal, cached_goal_vec = rollout_state.update_goal_ema(raw_goal, _goal_ema)

                # CAM-2 Schmitt-trigger handoff on the EMA'd distance (D_LO/D_HI
                # straddle the dual-visible band, so this flips at most once per
                # approach/retreat, not every cycle).
                _ema_dist = float(_goal_ema[0])
                _active_cam = rollout_state.cam_handoff(_active_cam, _ema_dist)
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
                    # stale lock and re-enter scan (rollout_state.reset_lock_and_rescan).
                    print(f"  [fancy] NX-16 lock coast-expired at step={step} "
                          f"(frames_since_det={_frames_since_det}) -> drop+rescan",
                          flush=True)
                    (_goal_ema, _last_known_goal, _frames_since_det, _scan_active,
                     _using_rescan_sched, _rescan_sched, _rescan_local_steps,
                     cached_goal_vec, _avoid_bias_wz) = rollout_state.reset_lock_and_rescan(SCAN_RATE)

            # NX-9 AVOID (docs/nx9_avoid.md) -- see rollout_state.avoid_bias_step()
            # for the full mechanism/carve-outs; reuses this cycle's
            # already-rendered depth_ground/intr_active (zero extra renders).
            _avoid_bias_wz, _last_avoid_dbg, _avoid_cycles_total, _avoid_cycles_active = (
                rollout_state.avoid_bias_step(
                    _avoid_is_maneuver, _scan_active, _frames_since_det, cached_goal_vec,
                    depth_ground, intr_active, data_mj, _avoid_bias_wz, _avoid_cycles_total,
                    _avoid_cycles_active, _last_avoid_dbg, CAM_HEAD_Z))

        # Scan mode
        if _scan_active:
            scan_wz, _scan_active, _using_rescan_sched, _rescan_local_steps = rollout_state.scan_or_rescan_step(
                step, yaw, _using_rescan_sched, _rescan_sched, _rescan_local_steps,
                _scan_active, _scan_sched, SCAN_TIMEOUT)

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

    # Final success/failure_tag + outro card + video save + live_ctx handoff
    # (rollout_setup.finalize_result) -- mirrors the original tail exactly.
    return rollout_setup.finalize_result(
        data_mj, target_xy, stop_r, fell, spotted, step_times, steps_done,
        render_video, video_path, frames_sbs, goal_idx, n_goals, dist_traveled_m,
        frame_callback, keep_alive, teacher, model_mj, nj, renderer, bev_cam,
        prev_action, proprio_hist, _phase_tracker, _prev_rxy_odom,
        _avoid_cycles_active, _avoid_cycles_total, path_trail, scan_steps,
    )
