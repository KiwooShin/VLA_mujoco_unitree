"""Episode-scope sim build/resume + CAM-2 handoff / NX-9 AVOID / NX-16 lock-
rescan state helpers for run_fancy_rollout (code/apps/fancy/rollout.py,
RF-2a split of the former single-file rollout.py).

Every function here is called from exactly one place in rollout.py's control
loop and returns its updated state as an explicit tuple (no nonlocal/closure
capture) so the loop keeps its own local variables as the single source of
truth -- this file only supplies the per-cycle *computation*, never holds
state across calls itself (ReacquisitionScan/BidirectionalScanSchedule
instances are still owned and carried by rollout.py's own locals). See each
function's docstring for the specific mechanism it implements; the module
docstring at the top of rollout.py explains the overall 3-way split
(rollout.py / rollout_frames.py / rollout_state.py).
"""

from __future__ import annotations

import os
from typing import NamedTuple, Optional, Tuple

import numpy as np

from code.apps.fancy.constants import BEV_AZIMUTH, BEV_DISTANCE, BEV_ELEVATION, BEV_LOOKAT_Z, BEV_W, BEV_H

# CAM-2 (docs/cam_p1.md, adopted champion): Schmitt-trigger handoff between the
# GROUNDING camera (26° pitch, far/mid range) and the PROXIMITY camera (58° pitch,
# ~0.22-1.81m), mirroring code/inferencer.py's main rollout loop exactly so the
# ego panel shows exactly what's driving detection this cycle (the handoff visible
# end-to-end, including through the final approach/stop).
CAM_D_LO = 1.2     # m — switch GROUNDING->PROXIMITY below this
CAM_D_HI = 1.6     # m — switch PROXIMITY->GROUNDING above this
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

GOAL_EMA_ALPHA = 0.4

# NX-16 (docs/nx16_cone_stall.md): sustained-loss-of-lock recovery.
# code/inferencer.py / code/eval_search.py already have this exact mechanism
# (lock_mgmt.py's M5 "coast-expiry -> drop lock + bounded ReacquisitionScan"),
# but it is default-OFF there (LOCK_M5=0, REJECT-verdicted on the gated eval
# protocols as a GLOBAL toggle, docs/nx2_iso.md) and this file never imported
# lock_mgmt at all -- so a detection that is lost for good (never fewer than
# HOLD_GOAL_HORIZON=100 frames since last detection) left `cached_goal_vec`
# frozen FOREVER with no recovery path. Root-caused (docs/nx16_cone_stall.md):
# the GROUND_NET detector's confidence on a `cone` decays steadily as range
# closes under the (shallow-pitch) GROUNDING camera -- cones are ~1.5-2.3x
# taller than the other 3 shapes (code/arena.py's two-part cone mesh) and
# increasingly clip out of frame -- and can drop under GROUND_NET_TAU right at
# the GROUNDING/PROXIMITY handoff boundary (~1.6-1.8m, just above CAM_D_LO), so
# the CAM-2 fallback probe never gets an EMA distance to re-trigger the
# handoff on, and PROXIMITY itself also fails to (re)detect at that range. The
# image-blind goto policy then keeps consuming the stale, never-updated
# egocentric (dist,bearing), which is not re-grounded in the robot's actual
# (moving) pose -- pure open-loop dead reckoning that curves past the true
# target and settles into a stable orbit (exactly DR-1's "approach to
# 0.6-0.9m, reverse, rock-stable plateau" signature).
#
# Fix, scoped ENTIRELY to fancy_demo's own local state (does not read or
# write LOCK_M5 / lock_mgmt.LockGate, so code/inferencer.py's and
# code/eval_search.py's default behavior -- and the M5 REJECT verdict -- are
# untouched): once a previously-SPOTTED lock has been missing for more than
# HOLD_GOAL_HORIZON frames, drop it and re-enter scan mode via a fresh
# ReacquisitionScan (own local step counter -- safe to start mid-episode,
# unlike re-arming `_scan_sched`/SCAN_TIMEOUT which are keyed off the
# episode's absolute step and would time out on the very next cycle). Bounded:
# if the rescan itself times out without reacquiring, falls back to the
# default forward-looking goal vector (same fallback the original
# never-spotted scan timeout already used) rather than freezing again.
#
# NX-16 mechanism-test finding: `ReacquisitionScan`'s own built-in bound reuses
# the shared SCAN_TIMEOUT=1150 (the INITIAL blind-scan's budget, sized for
# sweeping from a completely unknown bearing). A coast-expiry rescan starts
# from a MUCH better prior (it was tracking the target right up until the
# loss), and in gate testing reacquired within ~310-330 steps whenever the
# target was actually re-detectable -- but on one seed where the target sat in
# a detector blind range no amount of turning could escape (a cone
# specifically, docs/nx16_cone_stall.md), letting the rescan run for a
# nearly-full ~1150-step continuous sweep before an eventual (lucky,
# drift-induced) reacquisition produced an abrupt scan-to-goto transition that
# ended in a fall on one of two repeated runs (the other repeat instead simply
# timed out, no fall either way -- consistent with this being right at the
# edge of the policy's competence envelope for an atypically long
# uninterrupted turn-in-place, not a deterministic bug). Capping the LOCAL
# rescan budget well below the shared 1150 (NX16_RESCAN_MAX_STEPS, ~2x the
# observed successful-reacquisition time) keeps the common case (quick
# re-lock) unaffected while preventing this file's own rescan from ever
# running long enough to reach that observed instability regime -- falling
# back to the default goal (same as the pre-existing scan-timeout fallback,
# proven non-falling in DR-1's original 30+5-episode sweep) instead.
NX16_RESCAN_MAX_STEPS = 600

# Held-lock coast horizon: how many consecutive non-detections a previously
# SPOTTED lock survives on its last-known goal vector before NX-16's
# coast-expiry drop+rescan fires (see the block comment above).
HOLD_GOAL_HORIZON = 100


class SimHandle(NamedTuple):
    """Live MuJoCo sim handle returned by build_or_resume_sim()."""
    teacher: "WBCTeacher"
    data_mj: "mujoco.MjData"
    model_mj: "mujoco.MjModel"
    nj: int
    renderer: "ArenaRenderer"
    bev_cam: "mujoco.MjvCamera"
    rx: float
    ry: float


def build_or_resume_sim(
    inf: "Inferencer",
    scene_cfg: dict,
    resume_ctx: Optional[dict],
    goal_idx: int,
    target_xy: np.ndarray,
) -> Tuple[Optional[SimHandle], Optional[dict]]:
    """Build a fresh MuJoCo arena + renderer + BEV camera and settle the
    robot, OR (VF-3, docs/vf3_bev_fixes.md) resume the SAME live sim handed
    in via `resume_ctx` from a prior sub-goal's `live_ctx` -- mirrors
    run_fancy_rollout's own `resume_ctx`/`keep_alive` contract exactly.

    Returns:
        (sim, fall_result) -- exactly one is non-None. `sim` is a SimHandle
        on success. `fall_result` is run_fancy_rollout's own early-return
        result dict, only ever produced on the fresh-build path when the
        robot is already below FALL_HEIGHT right after settling (a resumed
        sim is never immediately fallen -- the caller only keeps a sim alive
        via keep_alive when the robot was upright at hand-off).
    """
    import math as _math

    import mujoco

    from code.arena import ArenaRenderer, build_arena
    from code.inferencer import FALL_HEIGHT
    from code.teacher import SIM_DT, WBCTeacher, _yaw_of

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
            return None, dict(success=False, spotted=False, scan_steps=0, failure_tag='fall',
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

    return SimHandle(teacher=teacher, data_mj=data_mj, model_mj=model_mj, nj=nj,
                      renderer=renderer, bev_cam=bev_cam, rx=rx, ry=ry), None


def reset_lock_and_rescan(scan_rate: float) -> tuple:
    """NX-16: drop a coast-expired lock and re-enter scan via a fresh
    ReacquisitionScan (see the NX-16 module comment above).

    Returns:
        Tuple (goal_ema, last_known_goal, frames_since_det, scan_active,
        using_rescan_sched, rescan_sched, rescan_local_steps, cached_goal_vec,
        avoid_bias_wz) -- assign directly onto rollout.py's own locals of the
        same names (mirrors the original closure's `nonlocal` writes 1:1).
    """
    from code.lock_mgmt import ReacquisitionScan

    goal_ema            = None
    last_known_goal      = None
    frames_since_det     = 0
    scan_active          = True
    using_rescan_sched   = True
    rescan_sched         = ReacquisitionScan(scan_rate=scan_rate)
    rescan_local_steps   = 0
    cached_goal_vec      = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    avoid_bias_wz        = 0.0
    return (goal_ema, last_known_goal, frames_since_det, scan_active,
            using_rescan_sched, rescan_sched, rescan_local_steps,
            cached_goal_vec, avoid_bias_wz)


def update_goal_ema(
    raw_goal: np.ndarray,
    goal_ema: Optional[np.ndarray],
    alpha: float = GOAL_EMA_ALPHA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend a freshly-detected goal vector into the running EMA (renormalizing
    the (cos,sin) bearing components after blending, since they must stay a
    unit vector for the atan2-based bearing reads downstream).

    Returns:
        (new_goal_ema, new_last_known_goal, new_cached_goal_vec) -- all three
        are `.copy()`d, matching the original closure's aliasing behavior.
    """
    import math as _math

    if goal_ema is None:
        goal_ema = raw_goal.copy()
    else:
        goal_ema = alpha * raw_goal + (1.0 - alpha) * goal_ema
        th = _math.atan2(goal_ema[2], goal_ema[1])
        goal_ema[1] = _math.cos(th)
        goal_ema[2] = _math.sin(th)
    last_known_goal = goal_ema.copy()
    cached_goal_vec  = goal_ema.copy()
    return goal_ema, last_known_goal, cached_goal_vec


def cam_handoff(
    active_cam: str,
    ema_dist: float,
    cam_d_lo: float = CAM_D_LO,
    cam_d_hi: float = CAM_D_HI,
) -> str:
    """CAM-2 Schmitt-trigger handoff on the EMA'd distance (D_LO/D_HI straddle
    the dual-visible band, so this flips at most once per approach/retreat,
    not every cycle)."""
    if active_cam == 'GROUNDING' and ema_dist < cam_d_lo:
        return 'PROXIMITY'
    if active_cam == 'PROXIMITY' and ema_dist > cam_d_hi:
        return 'GROUNDING'
    return active_cam


def handle_camera_probe(
    gr,
    active_cam: str,
    cam_miss_count: int,
    last_known_goal: Optional[np.ndarray],
    renderer,
    data_mj,
    yaw: float,
    target_color: str,
    target_shape: str,
    render_video: bool,
    ego_w: int,
    ego_h: int,
    cam_proximity_d_far: float = CAM_PROXIMITY_D_FAR,
) -> Tuple[object, str, int, Optional[np.ndarray]]:
    """CAM-2 bounded fallback probe (docs/cam_p1.md): after 2 consecutive
    misses on the active camera, probe the OTHER camera once and adopt its
    result if it detects. Plausibility-gated — only probe PROXIMITY when the
    last-known EMA distance says the target could actually be inside its
    ~0.22-1.81m band (prevents a far-range HSV false-positive from locking
    into PROXIMITY).

    Args:
        gr: This cycle's classical_ground() result on the active camera
            (already known to have `gr.not_visible == True` -- this function
            is only ever called from that branch).

    Returns:
        Tuple (gr, active_cam, cam_miss_count, video_frame_cache_update).
        `video_frame_cache_update` is the freshly labeled probe frame when the
        probe adopted a detection, else None (caller keeps its existing
        `_video_frame_cache` unchanged).
    """
    from code.grounding import ground as classical_ground
    from code.inferencer import _label_active_cam

    video_frame_cache_update = None
    cam_miss_count += 1
    if cam_miss_count >= 2:
        other_cam = 'GROUNDING' if active_cam == 'PROXIMITY' else 'PROXIMITY'
        probe_ok = (other_cam == 'GROUNDING' or
                    (last_known_goal is not None and
                     float(last_known_goal[0]) <= cam_proximity_d_far))
        if probe_ok:
            if other_cam == 'PROXIMITY':
                rgb2, depth2, intr2 = renderer.render_proximity(data_mj, yaw, render_depth=True)
            else:
                rgb2, depth2, intr2 = renderer.render_grounding(data_mj, yaw, render_depth=True)
            gr2 = classical_ground(rgb2, depth2, target_color, target_shape, intr2)
            if not gr2.not_visible:
                gr = gr2
                active_cam = other_cam
                cam_miss_count = 0
                if render_video:
                    video_frame_cache_update = _label_active_cam(
                        rgb2, active_cam, float(gr2.goal_vec[0]),
                        resize_to=(ego_w, ego_h))
    return gr, active_cam, cam_miss_count, video_frame_cache_update


def scan_or_rescan_step(
    step: int,
    yaw: float,
    using_rescan_sched: bool,
    rescan_sched,
    rescan_local_steps: int,
    scan_active: bool,
    scan_sched,
    scan_timeout: int,
    nx16_rescan_max_steps: int = NX16_RESCAN_MAX_STEPS,
) -> Tuple[Optional[float], bool, bool, int]:
    """Decide this cycle's scan angular rate (or None to fall through to a
    GOTO step), from either the initial bounded bidirectional sweep
    (`scan_sched`, bounded by the episode-absolute `scan_timeout`) or a NX-16
    coast-expiry `rescan_sched` (bounded by its own LOCAL step count, since
    re-arming the absolute-step timeout mid-episode would expire immediately
    -- see the NX-16 module comment above `reset_lock_and_rescan`).

    Returns:
        (scan_wz, scan_active, using_rescan_sched, rescan_local_steps).
    """
    if using_rescan_sched:
        if rescan_local_steps >= nx16_rescan_max_steps:
            scan_wz = None   # NX-16 tighter local cap -- see comment above
        else:
            scan_wz = rescan_sched.step(yaw)
        if scan_wz is None:
            scan_active        = False
            using_rescan_sched = False
            print(f"  [fancy] NX-16 RESCAN TIMEOUT step={step} "
                  f"(local_steps={rescan_local_steps}), "
                  f"falling back to default goal", flush=True)
        else:
            rescan_local_steps += 1
    elif step >= scan_timeout:
        scan_active = False
        scan_wz = None
        print(f"  [fancy] SCAN TIMEOUT step={step}", flush=True)
    else:
        scan_wz = scan_sched.step(yaw)   # bounded CCW/CW schedule, dwells at 0.0

    return scan_wz, scan_active, using_rescan_sched, rescan_local_steps


def avoid_bias_step(
    avoid_is_maneuver: bool,
    scan_active: bool,
    frames_since_det: int,
    cached_goal_vec: np.ndarray,
    depth_ground,
    intr_active,
    data_mj,
    avoid_bias_wz: float,
    avoid_cycles_total: int,
    avoid_cycles_active: int,
    last_avoid_dbg,
    cam_head_z: float,
) -> Tuple[float, object, int, int]:
    """NX-9 AVOID (docs/nx9_avoid.md): local obstacle avoidance -- same
    mechanism/carve-outs as code/inferencer.py's identical block (shared
    helper, code/avoid.py), reusing this cycle's already-rendered
    depth_ground/intr_active (zero extra renders). Never while `scan_active`;
    fresh bias only while the goal is fresh (<= AVOID_STALE_MAX_MISSED_CYCLES
    missed cycles), decay only on a longer stale coast -- see
    AVOID_STALE_MAX_MISSED_CYCLES' comment in code/avoid.py for the ep14 fall
    trace behind this. A no-op (unchanged pass-through) when AVOID is off, the
    scene is a maneuver scene, or a scan is active.

    Returns:
        (avoid_bias_wz, last_avoid_dbg, avoid_cycles_total, avoid_cycles_active).
    """
    import math as _math

    from code import avoid as _avoid

    if not (_avoid.AVOID and not avoid_is_maneuver and not scan_active):
        return avoid_bias_wz, last_avoid_dbg, avoid_cycles_total, avoid_cycles_active

    avoid_cycles_total += 1
    if frames_since_det > _avoid.AVOID_STALE_MAX_MISSED_CYCLES:
        avoid_bias_wz = _avoid.decay_bias(avoid_bias_wz)
    else:
        avoid_goal_dist_now    = float(cached_goal_vec[0])
        avoid_goal_bearing_now = _math.atan2(float(cached_goal_vec[2]), float(cached_goal_vec[1]))
        avoid_carved = (avoid_goal_dist_now < _avoid.AVOID_MIN_GOAL_DIST_M)
        avoid_cam_h  = float(data_mj.qpos[2]) + cam_head_z
        avoid_bias_wz, last_avoid_dbg = _avoid.compute_obstacle_bias(
            depth_ground, intr_active, cam_height_m=avoid_cam_h,
            goal_dist_m=avoid_goal_dist_now,
            goal_bearing_rad=avoid_goal_bearing_now,
            prev_bias_wz=avoid_bias_wz, carved_out=avoid_carved)
    if abs(avoid_bias_wz) > 1e-9:
        avoid_cycles_active += 1
    return avoid_bias_wz, last_avoid_dbg, avoid_cycles_total, avoid_cycles_active
