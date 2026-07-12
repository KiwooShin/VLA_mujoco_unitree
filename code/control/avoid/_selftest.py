"""
code.control.avoid._selftest — synthetic-depth-frame self-test for AVOID.

RF-1: ported verbatim (module paths updated only) from the
`if __name__ == "__main__":` block at the bottom of the old flat
code/avoid.py, so `python code/avoid.py` keeps printing the same PASS/FAIL
report it always has (see docs/nx9_avoid.md's citation of this self-test).
The same assertions are additionally exercised, case by case, as proper
`unittest` methods in tests/control/test_avoid.py.
"""

from __future__ import annotations

import math

import numpy as np

from code.control.avoid.core import (
    AVOID_MAX_WZ_BIAS,
    biased_vel_cmd,
    compute_obstacle_bias,
    is_maneuver_scene,
)
from code.control.avoid.geometry import backproject_frame
from code.control.steer import MAX_WZ
from code.grounding import CAM_ROBOT_FORWARD_OFFSET_M
from code.arena import GROUNDING_PITCH


def _get_intr(w: int, h: int) -> dict:
    """Build the synthetic-test camera intrinsics (grounding cam, rendered)."""
    from code.grounding import get_ego_intrinsics_rendered

    intr = get_ego_intrinsics_rendered(w, h)
    intr['pitch_deg'] = GROUNDING_PITCH   # 26.0, matches render_grounding()
    return intr


def _inverse_uncorrected(
    x_robot: float, y_vert: float, z_robot_raw: float, intr: dict
) -> tuple[float, float, float]:
    """Invert the UNCORRECTED-branch back-projection
    (z_robot_raw = sp*y_cam + cp*z_cam, y_vert = sp*z_cam + cp*y_cam —
    the grounding/ego pitch combination `backproject_frame` uses when
    `is_proximity` is not set) to build a synthetic depth pixel for a
    chosen robot-frame 3D point. Solved numerically (2x2 linear system)
    rather than by hand-algebra to avoid a repeat of the sign-error bug
    this self-test caught the first time around (see `y_vert`'s
    derivation comment in `backproject_frame`). Test-only helper —
    production code never needs the inverse.

    Returns:
        Tuple (u, v, z_cam): synthesized pixel coordinates and camera-
        frame depth for the given robot-frame point.
    """
    pitch_rad = math.radians(intr['pitch_deg'])
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    A = np.array([[sp, cp], [cp, sp]], dtype=np.float64)
    b = np.array([z_robot_raw, y_vert], dtype=np.float64)
    y_cam, z_cam = np.linalg.solve(A, b)
    x_cam = x_robot
    u = intr['cx'] + x_cam * intr['fx'] / z_cam
    v = intr['cy'] + y_cam * intr['fy'] / z_cam
    return u, v, z_cam


def _blank_floor_frame(intr: dict, w: int, h: int, cam_h: float,
                        near_m: float = 0.4, far_m: float = 6.0) -> np.ndarray:
    """A depth frame where every pixel is a genuine floor point (world
    height 0): for each image ROW, solve the ray/floor-plane intersection
    directly (depth is independent of column for a level floor + a
    rectilinear pinhole camera — un-pitched vertical position only
    depends on v). height_above_ground should read ~0 everywhere the
    ray actually hits the floor, so the floor cut must exclude ALL of it."""
    depth = np.full((h, w), far_m, dtype=np.float32)
    pitch_rad = math.radians(intr['pitch_deg'])
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    fy, cy = intr['fy'], intr['cy']
    for v in range(h):
        # y_vert(d) = sp*d + cp*(v-cy)*d/fy = d*(sp + cp*(v-cy)/fy); solve
        # y_vert(d) == cam_h (world height 0, matches backproject_frame's
        # y_vert = z_cam*sp + y_cam*cp).
        denom = sp + cp * (v - cy) / fy
        if abs(denom) < 1e-6:
            continue
        d = cam_h / denom
        if not math.isfinite(d) or d <= 0:
            continue
        depth[v, :] = float(np.clip(d, near_m, far_m))
    return depth


def _wall_frame(intr: dict, w: int, h: int, cam_h: float,
                 bearing_deg_lo: float, bearing_deg_hi: float, dist_m: float,
                 world_height_m: float = 1.1, background_m: float = 6.0) -> np.ndarray:
    """A frame that is background (far) everywhere except a wall/blob
    spanning [bearing_deg_lo, bearing_deg_hi] at world height
    `world_height_m` (i.e. NOT floor) and radial range `dist_m`. Both
    (u,v) are derived from the exact inverse projection (v is not
    assumed to be image-center — a pitched camera does not put
    eye-level objects at the center row)."""
    depth = np.full((h, w), background_m, dtype=np.float32)
    y_vert = cam_h - world_height_m
    for bearing_deg in np.linspace(bearing_deg_lo, bearing_deg_hi, 400):
        bearing_rad = math.radians(bearing_deg)
        x_robot = -dist_m * math.sin(bearing_rad)
        z_robot = dist_m * math.cos(bearing_rad)
        z_robot_raw = z_robot - CAM_ROBOT_FORWARD_OFFSET_M
        u, v, z_cam = _inverse_uncorrected(x_robot, y_vert, z_robot_raw, intr)
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and z_cam > 0:
            depth[max(0, vi - 15):min(h, vi + 15), ui] = max(z_cam, 0.05)
    return depth


def main() -> int:
    """Run the full synthetic-frame self-test and print a PASS/FAIL report.

    Returns:
        0 if every check passed, 1 otherwise (suitable as a process exit code).
    """
    W, H = 480, 360
    INTR = _get_intr(W, H)
    CAM_H = 1.34              # RESET_HEIGHT(0.79) + CAM_HEAD_Z(0.55), approx walking height

    n_pass = 0
    n_fail = 0

    def _check(name: str, cond: bool, extra: str = "") -> None:
        """Print a PASS/FAIL line for `name` and tally into n_pass/n_fail."""
        nonlocal n_pass, n_fail
        status = "PASS" if cond else "FAIL"
        if cond:
            n_pass += 1
        else:
            n_fail += 1
        print(f"  [{status}] {name}  {extra}")

    print("=" * 70)
    print("avoid.py synthetic-frame self-test")
    print("=" * 70)

    # 1. Floor-only frame -> zero bias (floor must be excluded regardless of range)
    floor_depth = _blank_floor_frame(INTR, W, H, CAM_H)
    bias, info = compute_obstacle_bias(floor_depth, INTR, CAM_H, goal_dist_m=5.0,
                                       goal_bearing_rad=0.0, prev_bias_wz=0.0)
    _check("floor-only -> zero bias", bias == 0.0 and info['n_obstacle_px'] == 0,
           f"bias={bias:.4f} n_obstacle_px={info['n_obstacle_px']}")

    # 2. Clear frame (all far background) -> zero bias
    clear_depth = np.full((H, W), 6.0, dtype=np.float32)
    bias, info = compute_obstacle_bias(clear_depth, INTR, CAM_H, goal_dist_m=5.0,
                                       goal_bearing_rad=0.0, prev_bias_wz=0.0)
    _check("clear/far frame -> zero bias", bias == 0.0, f"bias={bias:.4f}")

    # 3. Wall on the LEFT (positive bearing) -> bias should steer RIGHT (negative wz)
    wall_left = _wall_frame(INTR, W, H, CAM_H, 5.0, 22.0, dist_m=0.5)
    bias, info = compute_obstacle_bias(wall_left, INTR, CAM_H, goal_dist_m=5.0,
                                       goal_bearing_rad=0.0, prev_bias_wz=0.0)
    _check("wall-left -> negative (right-turn) bias", bias < -0.01,
           f"bias={bias:.4f} L={info['left']:.3f} R={info['right']:.3f}")

    # 4. Wall on the RIGHT (negative bearing) -> bias should steer LEFT (positive wz)
    wall_right = _wall_frame(INTR, W, H, CAM_H, -22.0, -5.0, dist_m=0.5)
    bias, info = compute_obstacle_bias(wall_right, INTR, CAM_H, goal_dist_m=5.0,
                                       goal_bearing_rad=0.0, prev_bias_wz=0.0)
    _check("wall-right -> positive (left-turn) bias", bias > 0.01,
           f"bias={bias:.4f} L={info['left']:.3f} R={info['right']:.3f}")

    # 5. Wall dead-center (symmetric) -> nonzero decisive bias (tie-break), bounded
    wall_center = _wall_frame(INTR, W, H, CAM_H, -20.0, 20.0, dist_m=0.5)
    bias, info = compute_obstacle_bias(wall_center, INTR, CAM_H, goal_dist_m=5.0,
                                       goal_bearing_rad=0.0, prev_bias_wz=0.0)
    _check("wall-center -> nonzero decisive bias, within cap",
           abs(bias) > 0.01 and abs(bias) <= AVOID_MAX_WZ_BIAS + 1e-6,
           f"bias={bias:.4f} cap={AVOID_MAX_WZ_BIAS}")

    # 6. Target-only: an obstacle-shaped blob exactly at the goal bearing, goal
    #    close (<2m) -> must be exempted (zero bias), since it's the target itself.
    target_blob = _wall_frame(INTR, W, H, CAM_H, -6.0, 6.0, dist_m=0.5)
    bias, info = compute_obstacle_bias(target_blob, INTR, CAM_H, goal_dist_m=1.0,
                                       goal_bearing_rad=0.0, prev_bias_wz=0.0)
    _check("target-window exempted when goal close -> zero bias",
           bias == 0.0 and info['n_obstacle_px'] == 0,
           f"bias={bias:.4f} n_obstacle_px={info['n_obstacle_px']}")

    # 6b. Same blob, but goal is FAR (>2m) -> NOT exempted, should produce bias
    #     (confirms the exemption is conditional on proximity, not unconditional).
    bias, info = compute_obstacle_bias(target_blob, INTR, CAM_H, goal_dist_m=5.0,
                                       goal_bearing_rad=0.0, prev_bias_wz=0.0)
    _check("same blob, goal far -> exemption does NOT apply, nonzero bias",
           abs(bias) > 0.01, f"bias={bias:.4f} n_obstacle_px={info['n_obstacle_px']}")

    # 7. carved_out=True hard-zeros regardless of obstacle content or prior bias
    bias, info = compute_obstacle_bias(wall_left, INTR, CAM_H, goal_dist_m=0.8,
                                       goal_bearing_rad=0.0, prev_bias_wz=0.25,
                                       carved_out=True)
    _check("carved_out=True -> hard zero", bias == 0.0 and info['carved_out'],
           f"bias={bias:.4f}")

    # 8. Hysteresis: obstacle disappears -> prior bias decays (not instant 0),
    #    and reaches (near-)zero within ~5 cycles (~1s @ 5Hz).
    b = 0.28
    decay_trace = [b]
    for _ in range(6):
        b, _ = compute_obstacle_bias(clear_depth, INTR, CAM_H, goal_dist_m=5.0,
                                     goal_bearing_rad=0.0, prev_bias_wz=b)
        decay_trace.append(b)
    _check("hysteresis: gradual decay, not instant zero",
           decay_trace[1] > 0.0 and decay_trace[1] < 0.28,
           f"trace={[round(x,4) for x in decay_trace]}")
    _check("hysteresis: reaches zero within ~5 cycles (~1s @5Hz)",
           decay_trace[5] == 0.0, f"trace={[round(x,4) for x in decay_trace]}")

    # 9. biased_vel_cmd: bias adds to wz, clipped to steer.py's MAX_WZ; vy always 0.
    vel = biased_vel_cmd(goal_dist=3.0, cos_th=1.0, sin_th=0.0, bias_wz=0.3, stop_r=0.6)
    _check("biased_vel_cmd: wz reflects bias, vy=0",
           abs(vel[2] - 0.3) < 1e-5 and vel[1] == 0.0, f"vel={vel}")
    vel_big = biased_vel_cmd(goal_dist=3.0, cos_th=1.0, sin_th=0.0, bias_wz=5.0, stop_r=0.6)
    _check("biased_vel_cmd: clipped to steer.py's MAX_WZ",
           abs(vel_big[2]) <= MAX_WZ + 1e-6, f"vel_big={vel_big} MAX_WZ={MAX_WZ}")
    vel_stop = biased_vel_cmd(goal_dist=0.3, cos_th=1.0, sin_th=0.0, bias_wz=5.0, stop_r=0.6)
    _check("biased_vel_cmd: within stop_r -> zeros regardless of bias",
           np.allclose(vel_stop, 0.0), f"vel_stop={vel_stop}")

    # 10. is_maneuver_scene helper
    _check("is_maneuver_scene detects difficulty='maneuver'",
           is_maneuver_scene({'difficulty': 'maneuver'}) is True)
    _check("is_maneuver_scene False for 'demo'",
           is_maneuver_scene({'difficulty': 'demo'}) is False)

    print("=" * 70)
    print(f"  {n_pass} passed, {n_fail} failed")
    print("=" * 70)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
