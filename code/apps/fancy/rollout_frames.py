"""Per-step frame composition / video plumbing for run_fancy_rollout
(code/apps/fancy/rollout.py, RF-2a split of the former single-file
rollout.py): BEV follow-cam tracking, the HUD camera-handoff flash timer, the
SCAN>LOCK>WALK>HANDOFF>REACH skill-stage mapping, and the full ego|BEV
side-by-side frame composition for one rendered step.

Every function takes its inputs explicitly (no nonlocal/closure capture over
rollout.py's loop state) and returns any updated state as its own return
value; rollout.py's own `_render_sbs_frame()` closure is a thin trampoline
that forwards its (still function-local) episode state into
`render_sbs_frame()` below on every call.
"""

from __future__ import annotations

import math as _math
from typing import Optional

import numpy as np

from code.apps.fancy.constants import (
    BEV_AZIMUTH, BEV_DISTANCE, BEV_ELEVATION, BEV_LOOKAT_Z,
    FEAT_HEATMAP, FEAT_HUD, STATE_LOCATED, STATE_REACHED,
)
from code.apps.fancy.overlays_bev import draw_bev_overlays
from code.apps.fancy.overlays_ego import compose_sbs_frame


def hud_cam_flash_update(hud_state: dict, active_cam_now: str, cam_flash_frames: int) -> bool:
    """VF-1 item 3: returns True while a recent GROUNDING<->PROXIMITY handoff
    should still be flashing the camera chip. Render-side only.

    Mutates `hud_state` in place (a dict, not a bare nonlocal, so this is a
    plain object mutation -- callable as a free function without closure
    capture); `hud_state` keeps keys "prev_cam" / "flash_frames_left" exactly
    as rollout.py's own `_hud_state` dict did before the split.
    """
    if hud_state["prev_cam"] is not None and hud_state["prev_cam"] != active_cam_now:
        hud_state["flash_frames_left"] = cam_flash_frames
    hud_state["prev_cam"] = active_cam_now
    flashing = hud_state["flash_frames_left"] > 0
    if flashing:
        hud_state["flash_frames_left"] -= 1
    return flashing


def skill_stage_idx(scan_active: bool, current_state, active_cam: str) -> int:
    """VF-1 item 3: map the existing state-machine variables to the 5-stage
    breadcrumb SCAN>LOCK>WALK>HANDOFF>REACH. Pure read of scan_active /
    current_state / active_cam -- computes a display index only."""
    if scan_active:
        return 0  # SCAN
    if current_state == STATE_LOCATED:
        return 1  # LOCK
    if current_state == STATE_REACHED:
        return 4  # REACH
    return 3 if active_cam == 'PROXIMITY' else 2  # HANDOFF vs WALK


def update_bev_cam(
    bev_cam,
    data_mj,
    bev_distance: float = BEV_DISTANCE,
    bev_azimuth: float = BEV_AZIMUTH,
    bev_elevation: float = BEV_ELEVATION,
    bev_lookat_z: float = BEV_LOOKAT_Z,
) -> None:
    """Follow robot with BEV camera (mutates `bev_cam`'s own attributes in
    place, matching the original closure's behavior exactly)."""
    bxy = data_mj.qpos[0:2]
    bev_cam.lookat[:] = [bxy[0], bxy[1], bev_lookat_z]
    bev_cam.distance  = bev_distance
    bev_cam.azimuth   = bev_azimuth
    bev_cam.elevation = bev_elevation


def render_sbs_frame(
    renderer,
    data_mj,
    target_xy: np.ndarray,
    target_obj: dict,
    path_trail: list,
    current_state,
    prompt: str,
    goal_idx: int,
    n_goals: int,
    completed_targets: list,
    bev_cam,
    model_mj,
    video_frame_cache: Optional[np.ndarray],
    avoid_bias_wz: float,
    avoid_info,
    active_cam: str,
    step: int,
    hud_state: dict,
    cam_flash_frames: int,
    target_color: str,
    target_shape: str,
    scan_active: bool,
):
    """Render ACTIVE-camera ego feed + BEV + overlays -> SBS frame.

    Returns:
        (sbs_bgr, dist_to_target) -- same contract as the original
        `_render_sbs_frame()` closure in rollout.py.
    """
    import cv2

    from code.grounding import get_ground_net_last_heatmap
    from code.teacher import _yaw_of

    yaw_now = _yaw_of(data_mj.qpos[3:7])
    rxy     = data_mj.qpos[0:2].copy()
    dist    = float(np.linalg.norm(rxy - target_xy))

    # Ego panel = the CAM-2 ACTIVE camera (GROUNDING far / PROXIMITY near) — same
    # camera that is actually driving detection this cycle, so the handoff (and the
    # target staying in-frame down to the stop) is visible in the recorded clip.
    # `video_frame_cache` is refreshed on every grounding cycle in rollout.py's own
    # loop (already labeled + resized to EGO_W x EGO_H by _label_active_cam); reused
    # on in-between steps so the video stays at full step-rate without extra renders.
    if video_frame_cache is not None:
        ego_rgb = video_frame_cache
    else:
        ego_rgb, _, _ = renderer.render_ego(data_mj, yaw_now, render_depth=False)

    # BEV RGB (640x480 from follow-cam = tp_rend)
    update_bev_cam(bev_cam, data_mj)
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
        completed_targets=completed_targets,
        target_color_bgr=target_color_bgr,
        avoid_bias_wz=avoid_bias_wz,
        avoid_info=avoid_info,
    )

    # VF-1 item 1: last cached GROUND_NET heatmap (None when GROUND_NET is
    # off / never fired / query doesn't match this episode's target).
    heatmap_cache = get_ground_net_last_heatmap() if FEAT_HEATMAP else None

    # VF-1 item 3: HUD bar context -- pure reads of state that already exists
    # in rollout.py's loop (proprio-derived speed, geometry-derived bearing,
    # the loop's own step/prompt, the skill-stage mapping above).
    hud_ctx = None
    if FEAT_HUD:
        bearing_deg = _math.degrees(_math.atan2(target_xy[1] - rxy[1],
                                                 target_xy[0] - rxy[0]) - yaw_now)
        bearing_deg = _math.degrees(_math.atan2(_math.sin(_math.radians(bearing_deg)),
                                                 _math.cos(_math.radians(bearing_deg))))
        walk_speed = float(np.linalg.norm(data_mj.qvel[0:2]))
        cam_flash  = hud_cam_flash_update(hud_state, active_cam, cam_flash_frames)
        hud_ctx = dict(
            prompt=prompt, dist=dist, bearing_deg=bearing_deg, step=step,
            walk_speed_mps=walk_speed, stage_idx=skill_stage_idx(scan_active, current_state, active_cam),
            active_cam=active_cam, cam_flash=cam_flash,
        )

    sbs = compose_sbs_frame(ego_rgb, bev_bgr, current_state, prompt, dist,
                            goal_idx=goal_idx, n_goals=n_goals, active_cam=active_cam,
                            heatmap_cache=heatmap_cache, target_color=target_color,
                            target_shape=target_shape, hud_ctx=hud_ctx)
    return sbs, dist
