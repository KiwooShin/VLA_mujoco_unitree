"""
code.control.avoid.geometry — depth-frame back-projection helper for AVOID.

Mirrors code/grounding.py's `cam_to_egocentric` exactly (same pitch
un-rotation, same CAM_ROBOT_FORWARD_OFFSET_M, same bearing sign convention:
positive = LEFT) so bearings/distances computed here are directly comparable
to `cached_goal_vec`'s (dist, cos_th, sin_th).

RF-1: split out of code/avoid.py (see code/avoid.py, the old-path compat
alias, and docs/refactor_plan.md) — this is the "helpers" half of the
core/helpers split; code.control.avoid.core holds the bias-computation logic.
"""

from __future__ import annotations

import math

import numpy as np

from code.grounding import CAM_ROBOT_FORWARD_OFFSET_M
from code.arena import GROUNDING_PITCH


def backproject_frame(
    depth_m: np.ndarray, intr: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full-frame back-projection to robot-egocentric (dist, bearing_rad,
    height_above_cam_m) per pixel. No extra render — consumes the depth
    array the caller already has.

    Args:
        depth_m: (H,W) depth frame (metres) already rendered by the caller.
        intr: Camera intrinsics dict (fx,fy,cx,cy[,pitch_deg,is_proximity]).

    Returns:
        Tuple (dist, bearing, y_vert), each an (H,W) float32 array:
            dist: radial egocentric distance (m).
            bearing: signed bearing (rad), positive = LEFT.
            y_vert: vertical robot-frame coordinate (m, DOWN positive,
                i.e. "how far below the camera this point is" once
                leveled) — used for the floor cut: a point at world
                height ~0 is `cam_height_m` below the camera regardless
                of range.
    """
    h, w = depth_m.shape[:2]
    fx, fy = float(intr['fx']), float(intr['fy'])
    cx, cy = float(intr['cx']), float(intr['cy'])
    pitch_deg = float(intr.get('pitch_deg', GROUNDING_PITCH))
    use_corrected = bool(intr.get('is_proximity', False))

    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    d = depth_m.astype(np.float32)

    x_cam = (uu - cx) * d / fx
    y_cam = (vv - cy) * d / fy
    z_cam = d

    pitch_rad = math.radians(pitch_deg)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)

    if use_corrected:
        # CAM-2/Phase-1 corrected sign (docs/cam_p1.md), used for the
        # PROXIMITY camera's steep pitch — see code/grounding.py's
        # cam_to_egocentric docstring for the derivation.
        z_robot = z_cam * cp - y_cam * sp
    else:
        z_robot = y_cam * sp + z_cam * cp

    # Vertical term: the PROPER rotation-matrix pairing for the forward term
    # is `z_robot = z_cam*cp - y_cam*sp` (the "corrected" branch above) —
    # `[[cp,-sp],[sp,cp]]` applied to `(z_cam, y_cam)`. The "uncorrected"
    # z_robot branch used for grounding/ego cams is a known, accepted
    # approximation for the FORWARD term only (code/grounding.py's
    # cam_to_egocentric docstring); it does not correspond to a real
    # rotation, so it must not be used to derive the vertical term. The
    # vertical/height component therefore ALWAYS uses the proper paired
    # term `y_vert = z_cam*sp + y_cam*cp`, regardless of which z_robot
    # branch is selected — empirically verified against a real
    # ArenaRenderer.render_grounding() floor pixel (checkered-floor row,
    # known height 0): this formula reproduces height_above_ground≈0 to
    # <1cm; the naive `y_cam*cp - z_cam*sp` pairing used with the
    # uncorrected z_robot branch is off by ~1.4m (was mistaken for a
    # "matched" pair with the uncorrected branch — it is not).
    y_vert = z_cam * sp + y_cam * cp

    x_robot = x_cam
    z_robot = z_robot + CAM_ROBOT_FORWARD_OFFSET_M

    dist = np.hypot(x_robot, z_robot)
    bearing = np.arctan2(-x_robot, z_robot)   # positive = LEFT (matches grounding.py)

    return dist.astype(np.float32), bearing.astype(np.float32), y_vert.astype(np.float32)


# Old-attribute-compat: avoid.py's original name was private (`_backproject_frame`).
# core.py imports the public `backproject_frame` name; this alias keeps any
# incidental `from code.avoid import _backproject_frame`-style access working
# too (the old module exposed it at module scope, even though underscore-
# prefixed names are not part of the documented public API).
_backproject_frame = backproject_frame
