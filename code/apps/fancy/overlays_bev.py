"""BEV-panel overlay drawing for the fancy demo (code/fancy_demo.py, RF-1
split): AVOID repulsion visualization + the full BEV overlay stack (path
trail, target ring, FOV cone, status banner).

Both consume code/apps/fancy/overlays_projection.py's `world_to_bev_pixel`.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from code.apps.fancy.constants import (
    COLOR_FOV_CONE, COLOR_PATH_TRAIL, COLOR_TARGET_RING,
    COLOR_STATE_SEARCH, COLOR_STATE_LOCATE, COLOR_STATE_MOVE, COLOR_STATE_REACH,
    FEAT_AVOID_VIZ, FEAT_TRAIL,
    STATE_IDLE, STATE_SEARCHING, STATE_LOCATED, STATE_MOVING, STATE_REACHED, STATE_FAILED,
)
from code.apps.fancy.overlays_projection import (
    TRAIL_COOL_BGR, TRAIL_WARM_BGR, _dashed_line, _lerp_color_bgr, world_to_bev_pixel,
)


def draw_avoid_overlay(bev_img: np.ndarray, robot_xy: np.ndarray, robot_yaw: float,
                       bev_cam: "mujoco.MjvCamera", model: "mujoco.MjModel",
                       data: "mujoco.MjData", avoid_bias_wz: float,
                       avoid_info: Optional[dict], fovy_deg: float = 45.0) -> np.ndarray:
    """VF-1 item 2: visualize NX-9 AVOID's obstacle-repulsion bias on the BEV panel.

    Pure render-side read of the ALREADY-COMPUTED `avoid_bias_wz` / `avoid_info`
    (code/avoid.py's compute_obstacle_bias() return values, cached by the caller
    each grounding cycle) -- this function never calls compute_obstacle_bias
    itself and never influences the value fed back into steer.py's control law.

    No-op (returns bev_img unchanged) when the bias is within the deadband.

    Args:
        bev_img: (H, W, 3) uint8 BGR BEV frame to draw on (mutated in place).
        robot_xy: (2,) current robot world position.
        robot_yaw: Current robot yaw, in radians.
        bev_cam: MuJoCo free camera used for the BEV follow-cam view.
        model: MuJoCo model, forwarded to world_to_bev_pixel().
        data: MuJoCo data, forwarded to world_to_bev_pixel().
        avoid_bias_wz: Already-computed AVOID yaw-rate bias (positive = steer
            left, negative = steer right; same sign convention as steer.py).
        avoid_info: compute_obstacle_bias()'s own debug dict (`left`/`right`
            severities), or None when AVOID has no active bias this cycle.
        fovy_deg: Vertical field of view in degrees, forwarded to
            world_to_bev_pixel().

    Returns:
        The BGR frame with the AVOID overlay drawn (same array as `bev_img`
        when a bias was drawn; `bev_img` unchanged when there is nothing to
        draw).
    """
    import cv2
    from code import avoid as _avoid

    if avoid_info is None or abs(avoid_bias_wz) < 1e-6:
        return bev_img

    img = bev_img
    H, W = img.shape[:2]

    def w2p(xy: np.ndarray) -> tuple[int, int]:
        """World (x, y) to BEV pixel (u, v), rounded to the nearest int."""
        pix = world_to_bev_pixel(np.array([[xy[0], xy[1], 0.0]]), bev_cam, model, data, W, H, fovy_deg)
        return (int(round(pix[0, 0])), int(round(pix[0, 1])))

    # --- Obstacle-corridor wedge tint (same corridor geometry AVOID reads from
    # the depth frame: +/-AVOID_CORRIDOR_HALF_DEG about the robot heading, out to
    # AVOID_NEAR_M). Left/right halves shaded independently by L/R severity so the
    # tint visually matches which side the obstacle mass is actually on. ---
    half_rad = math.radians(_avoid.AVOID_CORRIDOR_HALF_DEG)
    rng_m    = _avoid.AVOID_NEAR_M
    n_pts    = 14
    L = float(avoid_info.get('left', 0.0))
    R = float(avoid_info.get('right', 0.0))

    for side, sev in (("left", L), ("right", R)):
        if sev <= 1e-4:
            continue
        if side == "left":
            ang_lo, ang_hi = robot_yaw, robot_yaw + half_rad
        else:
            ang_lo, ang_hi = robot_yaw - half_rad, robot_yaw
        poly = [[robot_xy[0], robot_xy[1]]]
        for i in range(n_pts + 1):
            ang = ang_lo + i * (ang_hi - ang_lo) / n_pts
            poly.append([robot_xy[0] + rng_m * math.cos(ang),
                         robot_xy[1] + rng_m * math.sin(ang)])
        poly_pix = np.array([w2p(p) for p in poly], dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly_pix], (20, 50, 200), cv2.LINE_AA)  # warm red-orange, BGR
        a = 0.12 + 0.35 * min(1.0, sev)
        cv2.addWeighted(overlay, a, img, 1.0 - a, 0, img)

    # --- Repulsion arrow: bias_wz > 0 = steer LEFT, < 0 = steer RIGHT (same sign
    # convention as steer.py/avoid.py). Drawn as a curved-looking short arrow
    # tangential to the heading, scaled by |bias| / cap. ---
    sev_overall = min(1.0, abs(avoid_bias_wz) / max(_avoid.AVOID_MAX_WZ_BIAS, 1e-6))
    turn_sign   = 1.0 if avoid_bias_wz > 0 else -1.0
    arrow_ang   = robot_yaw + turn_sign * (math.pi / 2.0) * (0.5 + 0.5 * sev_overall)
    arrow_len_m = 0.5 + 1.2 * sev_overall
    base_pt = w2p(robot_xy)
    tip_pt  = w2p([robot_xy[0] + arrow_len_m * math.cos(arrow_ang),
                   robot_xy[1] + arrow_len_m * math.sin(arrow_ang)])
    cv2.arrowedLine(img, base_pt, tip_pt, (0, 210, 255), 3, cv2.LINE_AA, tipLength=0.35)

    # --- "AVOID" indicator chip (top-left of BEV panel, only lit while active) ---
    chip_txt = "AVOID"
    cv2.rectangle(img, (6, 6), (76, 28), (0, 140, 255), -1)
    cv2.putText(img, chip_txt, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, chip_txt, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    return img


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
    # VF-1 item 4: dashed robot->target goal line drawn in the target's own color.
    target_color_bgr: Optional[tuple] = None,
    # VF-1 item 2: AVOID visualization (pure read of already-computed bias/info).
    avoid_bias_wz: float = 0.0,
    avoid_info: Optional[dict] = None,
) -> np.ndarray:
    """Draw all BEV overlays on bev_img (in-place + return).

    Overlays:
      (a) PATH TRAIL — green polyline (VF-1: gradient cool->warm when FEAT_TRAIL)
      (b) TARGET HIGHLIGHT — orange ring + cross
      (c) FOV CONE — yellow wedge on ground
      (d) STATUS BANNER — bottom banner with state + distance
      (e) VF-1: AVOID repulsion viz (item 2), dashed goal line in target color (item 4)

    Args:
        bev_img: (H, W, 3) uint8 BGR BEV frame (a copy is drawn on, not
            mutated in place -- the annotated copy is returned).
        path_trail: List of (x, y) world positions, oldest first.
        target_xy: (2,) world position of the current sub-goal target, or
            None if there is no target to highlight.
        robot_xy: (2,) current robot world position.
        robot_yaw: Current robot yaw, in radians.
        bev_cam: MuJoCo free camera used for the BEV follow-cam view.
        model: MuJoCo model, forwarded to world_to_bev_pixel().
        data: MuJoCo data, forwarded to world_to_bev_pixel().
        state: Current state-machine state (one of the STATE_* constants),
            used for the status banner and for gating the goal line.
        prompt: Typed instruction text shown in the status banner.
        dist_to_target: Current distance to target in meters, or None.
        fovy_deg: Vertical field of view in degrees, forwarded to
            world_to_bev_pixel().
        goal_idx: Zero-based index of the current sub-goal (multi-goal runs).
        n_goals: Total number of sub-goals in this episode.
        completed_targets: World (x, y) positions of already-reached targets,
            drawn as green check marks.
        target_color_bgr: The current target's own BGR color, used for the
            dashed goal line (VF-1 item 4); None falls back to the plain
            magenta line.
        avoid_bias_wz: Already-computed AVOID yaw-rate bias, forwarded to
            draw_avoid_overlay().
        avoid_info: compute_obstacle_bias()'s own debug dict, forwarded to
            draw_avoid_overlay().

    Returns:
        A new (H, W, 3) uint8 BGR frame with all overlays drawn.
    """
    import cv2

    img = bev_img.copy()
    H, W = img.shape[:2]

    def w2p(world_xyz: np.ndarray) -> tuple[int, int]:
        """World XYZ (or XY with Z=0) to pixel (u,v) as int tuple."""
        if len(world_xyz) == 2:
            world_xyz = np.array([world_xyz[0], world_xyz[1], 0.0])
        pix = world_to_bev_pixel(np.array([world_xyz]), bev_cam, model, data, W, H, fovy_deg)
        return (int(round(pix[0, 0])), int(round(pix[0, 1])))

    # (a) PATH TRAIL — polyline of past base positions
    # VF-1 item 4: gradient color by recency (cool blue -> warm orange/red),
    # thicker line. FEAT_TRAIL=0 (or FANCY_PLAIN) keeps the exact original
    # single-color fade-only trail.
    if len(path_trail) >= 2:
        pts_pix = []
        for xy in path_trail:
            p = w2p(xy)
            pts_pix.append(p)
        n_pts = len(pts_pix)
        for i in range(1, n_pts):
            p0 = pts_pix[i - 1]
            p1 = pts_pix[i]
            if not ((0 <= p0[0] < W and 0 <= p0[1] < H) or (0 <= p1[0] < W and 0 <= p1[1] < H)):
                continue
            if FEAT_TRAIL:
                t = i / max(1, n_pts - 1)          # 0 (oldest) -> 1 (most recent)
                c = _lerp_color_bgr(TRAIL_COOL_BGR, TRAIL_WARM_BGR, t)
                thickness = 2 + (2 if t > 0.7 else 0)
                cv2.line(img, p0, p1, c, thickness, cv2.LINE_AA)
            else:
                alpha = i / n_pts  # fade from dim to bright (ORIGINAL behavior)
                c = tuple(int(x * alpha + x * 0.3) for x in COLOR_PATH_TRAIL)
                cv2.line(img, p0, p1, COLOR_PATH_TRAIL, 2, cv2.LINE_AA)

    # Robot position dot (white)
    robot_pix = w2p([robot_xy[0], robot_xy[1], 0.0])
    if 0 <= robot_pix[0] < W and 0 <= robot_pix[1] < H:
        cv2.circle(img, robot_pix, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, robot_pix, 7, (0, 0, 0), 1, cv2.LINE_AA)

    # (c) FOV CONE — the robot's ACTIVE detection camera's real horizontal FOV
    # wedge on the ground, anchored at robot_yaw (== the camera's azimuth;
    # arena._set_ego_cam sets cam.azimuth=degrees(yaw) exactly, no independent
    # pan, for the ego/GROUNDING/PROXIMITY cameras alike).
    #
    # VF-3 fix (docs/vf3_bev_fixes.md): this used to be a hardcoded ±45 deg,
    # from a comment that conflated the camera's VERTICAL FOVY with a
    # horizontal half-angle ("90° FOVY -> ±45°"). Two separate errors:
    #   1. The ACTUAL rendered FOVY is 45° (MuJoCo's model.vis.global_.fovy
    #      default — see code/grounding.py's EGO_FOVY_RENDERED / "E6 fix v4"
    #      comment), not the stale arena.EGO_FOVY=90 constant the old ±45 was
    #      (incorrectly) derived from.
    #   2. Even the correct FOVY needs the aspect-ratio pinhole conversion —
    #      the same atan(tan(fovy/2)*w/h) formula this file already uses in
    #      world_to_bev_pixel()/arena.get_ego_intrinsics() — to get the
    #      HORIZONTAL half-angle, not a naive fovy/2.
    # Real horizontal half-FOV for FOVY=45° at the GROUNDING/PROXIMITY cameras'
    # shared 4:3 aspect (480x360 / 320x240 — both equal 4:3, so this is the
    # same value regardless of which one is currently active):
    # atan(tan(22.5°)*4/3) = 28.87°.
    _FOV_CONE_FOVY_DEG = 45.0        # actual rendered FOVY (all non-widefov cams)
    _FOV_CONE_ASPECT   = 4.0 / 3.0   # GROUNDING_W/H == PROXIMITY_W/H aspect ratio
    fov_half_rad = math.atan(math.tan(math.radians(_FOV_CONE_FOVY_DEG) / 2.0) * _FOV_CONE_ASPECT)
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
            # VF-4 (user verification pass): the previous r=14 double ring +
            # center-filling crosshair completely covered the target mesh
            # (~5-9 px at whole-arena zoom), making it impossible to SEE that
            # the marker is actually on the object. Keep the center clear:
            # thin open ring + 4 corner ticks OUTSIDE the ring, nothing within
            # r-2 px of the object itself.
            r = 13
            cv2.circle(img, tgt_pix, r, COLOR_TARGET_RING, 1, cv2.LINE_AA)
            cv2.circle(img, tgt_pix, r + 5, COLOR_TARGET_RING, 1, cv2.LINE_AA)
            for sx, sy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p_in  = (tgt_pix[0] + sx * (r + 6), tgt_pix[1] + sy * (r + 6))
                p_out = (tgt_pix[0] + sx * (r + 12), tgt_pix[1] + sy * (r + 12))
                cv2.line(img, p_in, p_out, COLOR_TARGET_RING, 2, cv2.LINE_AA)
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

    # Draw a line from robot to target if target visible in BEV.
    # VF-1 item 4: dashed, in the target's own color, when FEAT_TRAIL + a color
    # was supplied; otherwise the exact original thin solid magenta line.
    if target_xy is not None and dist_to_target is not None:
        tgt_pix2 = w2p([target_xy[0], target_xy[1], 0.0])
        if (0 <= tgt_pix2[0] < W and 0 <= tgt_pix2[1] < H and
                0 <= robot_pix[0] < W and 0 <= robot_pix[1] < H and
                state in (STATE_MOVING, STATE_LOCATED)):
            # VF-4: stop the goal line at the target ring's edge so the line
            # never crosses over (and hides) the target mesh itself.
            _gdx = tgt_pix2[0] - robot_pix[0]
            _gdy = tgt_pix2[1] - robot_pix[1]
            _glen = math.hypot(_gdx, _gdy)
            if _glen > 22.0:
                _gend = (int(tgt_pix2[0] - _gdx / _glen * 20), int(tgt_pix2[1] - _gdy / _glen * 20))
                if FEAT_TRAIL and target_color_bgr is not None:
                    _dashed_line(img, robot_pix, _gend, target_color_bgr, thickness=2)
                else:
                    cv2.line(img, robot_pix, _gend, (180, 80, 255), 1, cv2.LINE_AA)

    # VF-1 item 2: AVOID repulsion visualization (corridor tint + arrow + chip).
    # Pure read of already-computed avoid_bias_wz/avoid_info -- no-op internally
    # when the bias is within the deadband.
    if FEAT_AVOID_VIZ:
        img = draw_avoid_overlay(img, robot_xy, robot_yaw, bev_cam, model, data,
                                  avoid_bias_wz, avoid_info, fovy_deg=fovy_deg)

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


_STATE_COLOR_MAP: dict[str, tuple[int, int, int]] = {
    STATE_SEARCHING: COLOR_STATE_SEARCH,
    STATE_LOCATED:   COLOR_STATE_LOCATE,
    STATE_MOVING:    COLOR_STATE_MOVE,
    STATE_REACHED:   COLOR_STATE_REACH,
    STATE_FAILED:    (100, 100, 100),
    STATE_IDLE:      (150, 150, 150),
}
