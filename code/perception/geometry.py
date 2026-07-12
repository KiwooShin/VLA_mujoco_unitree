"""
code/perception/geometry.py — camera intrinsics + camera-frame -> egocentric
transform (RF-1 split of code/grounding.py; docs/refactor_plan.md).

Owns:
  - `get_ego_intrinsics_rendered`: intrinsics for the ACTUALLY-rendered FOVY
    (see the E6 fix v4 note below -- arena.EGO_FOVY is wrong for the render).
  - `cam_to_egocentric`: camera-frame 3-D point -> robot-egocentric (dist, yaw_err).

Both are pure functions (no module-level mutable state) shared by the
classical HSV pipeline (code/perception/hsv_pipeline.py), the GROUND_NET
backend (code/perception/ground_net.py + code/perception/detector/model.py),
and every external caller that imports them via the `code.grounding` compat
alias.
"""
from __future__ import annotations

import math

from code.arena import EGO_H, EGO_W

# E6 fix v4: The MuJoCo model's actual rendered FOVY = model.vis.global_.fovy = 45 degrees.
# The arena.EGO_FOVY constant = 90 degrees is incorrect for the rendered image.
# Using the wrong FOVY inflates focal length error by 2x, causing lateral position errors.
# Override here for grounding only (training pipeline is unaffected).
EGO_FOVY_RENDERED = 45.0  # degrees — actual rendered FOVY (model.vis.global_.fovy)


def get_ego_intrinsics_rendered(w: int = EGO_W, h: int = EGO_H) -> dict:
    """Return intrinsics for the ACTUAL rendered image (FOVY=45 deg, not 90).

    Args:
        w: Rendered image width in pixels.
        h: Rendered image height in pixels.

    Returns:
        dict with keys fx, fy, cx, cy, width, height, fovy_deg.
    """
    fovy_rad = math.radians(EGO_FOVY_RENDERED)
    fy = (h / 2.0) / math.tan(fovy_rad / 2.0)
    fovx_rad = 2.0 * math.atan(math.tan(fovy_rad / 2.0) * w / h)
    fx = (w / 2.0) / math.tan(fovx_rad / 2.0)
    return dict(fx=fx, fy=fy, cx=w / 2.0 - 0.5, cy=h / 2.0 - 0.5,
                width=w, height=h, fovy_deg=EGO_FOVY_RENDERED)


# ---------------------------------------------------------------------------
# Camera-frame to robot-egocentric transform
# ---------------------------------------------------------------------------
# MuJoCo ego camera: x=right, y=down, z=forward (when tilt=0).
# With downward pitch, the forward direction has a z-component in world frame.
# We treat the camera as a pinhole with the optical axis pointing ≈ forward in
# the robot's horizontal plane for the purpose of bearing estimation.

CAM_PITCH_RAD = math.radians(32.0)   # matches arena.CAM_PITCH

# P0 fix (2026-07-08): arena._set_ego_cam's cam.distance was 0.001 with a lookat
# point 1.0m forward, which placed the TRUE rendered eye at
# `origin + (1-distance)*forward_dir` -- i.e. it DRIFTED with pitch. The old
# 0.947m constant below was empirically measured only at pitch=32 deg
# (0.10 + cos(32deg)*0.999 = 0.947) and was WRONG when reused, unchanged, for
# the grounding render's 26 deg pitch (should have been ~0.10+cos(26)=0.999m).
#
# Now that arena._set_ego_cam uses cam.distance=1.0 (see arena.py comment), the
# true eye sits EXACTLY at `origin_head = pelvis_xy + CAM_FWD*heading, pelvis_z+CAM_HEAD_Z`
# for ANY pitch -- the (1-distance) drift term is now exactly zero. So the
# camera-to-robot forward offset collapses to the constant forward mount offset
# CAM_FWD=0.10m, independent of pitch. This was verified empirically (see
# docs/cam_p0.md): rendering known-position targets through both the 26 deg
# grounding camera and the 32 deg ego camera and comparing ground()'s reported
# (dist, bearing) against the analytic ground-truth confirms the same 0.10m
# constant holds for both pitches (previously a single mismatched constant was
# reused across pitches with no such guarantee).
CAM_ROBOT_FORWARD_OFFSET_M = 0.10  # metres camera is forward of robot origin (CAM_FWD)


def cam_to_egocentric(x_cam: float, y_cam: float, z_cam: float,
                      pitch_deg: float = CAM_PITCH_RAD * 180.0 / math.pi,
                      use_corrected_unpitch: bool = False) -> tuple[float, float]:
    """
    Convert camera-frame 3-D point to robot-egocentric (dist, yaw_err).

    Camera frame: x=right, y=down, z=forward (before pitch).
    We un-pitch the z/y axes to get the horizontal-plane projection,
    then add the camera-to-robot forward offset so the returned values
    are relative to the robot's pelvis origin (matching training data convention).

    Args:
        x_cam: Camera-frame x coordinate (metres, + = image right).
        y_cam: Camera-frame y coordinate (metres, + = down).
        z_cam: Camera-frame z coordinate (metres, + = forward, before pitch).
        pitch_deg: camera downward tilt in degrees.
            Use 32° for standard ego render, 20° for grounding render (V2).
            The grounding render uses a shallower pitch so distant targets appear
            in frame; cam_to_egocentric must use the same pitch to un-rotate correctly.
        use_corrected_unpitch: CAM-2/Phase-1 finding (docs/cam_p1.md). The forward-distance
            term below (`z_robot = y_cam*sin + z_cam*cos`) has the WRONG SIGN on the
            y_cam term -- verified by a ground-truth distance sweep (known target
            positions vs. reported dist): at the existing shallow pitches (26°/32°)
            the resulting error is small enough that it hides inside the previously
            -documented "MuJoCo z-buffer underestimation" and the P0 gate still
            passed, but at the new PROXIMITY_PITCH=58° camera the SAME bug makes the
            reported distance *decrease* as the true distance *increases* (completely
            unusable). The geometrically-correct term is `z_cam*cos - y_cam*sin`
            (re-derived from the OpenGL look-at camera convention MuJoCo's free
            camera actually uses, and confirmed empirically: distance now increases
            monotonically with true distance and matches ground truth to within the
            object's own near-surface offset). This flag is threaded from
            `intrinsics['is_proximity']` in ground() below so it activates ONLY for
            the new proximity camera -- the existing, gated 26°/32° grounding/ego
            paths are left byte-for-byte unchanged (zero regression risk). Fixing
            this bug for the grounding camera too is flagged as follow-on work, out
            of scope for this Phase-1 (CAM-2) change.

    Returns:
        Tuple of (dist, yaw_err): distance in metres from the robot's pelvis
        origin, and yaw error in radians (positive when the target is to the
        LEFT, CCW).
    """
    # Un-pitch: rotate around camera x-axis by +pitch_rad
    #   x_robot = x_cam
    #   y_robot = y_cam * cos(pitch) - z_cam * sin(pitch)  (vertical)
    #   z_robot = y_cam * sin(pitch) + z_cam * cos(pitch)  (forward)  [pre-existing, see note above]
    pitch_rad = math.radians(pitch_deg)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    if use_corrected_unpitch:
        z_robot = z_cam * cp - y_cam * sp   # forward (+ = in front) -- CAM-2 corrected sign
    else:
        z_robot = y_cam * sp + z_cam * cp   # forward (+ = in front) -- pre-existing (unchanged)
    x_robot = x_cam                     # camera x (+ = image right)

    # Add camera-to-robot offset: camera is ~0.947m forward of robot origin.
    # Training goal_vec uses robot origin as reference; grounding uses camera position.
    # This correction aligns the grounding output with the training distribution.
    z_robot += CAM_ROBOT_FORWARD_OFFSET_M

    dist    = math.hypot(x_robot, z_robot)
    # E6 fix v4: MuJoCo's free camera has image x+ = world RIGHT from robot perspective,
    # but in the rendered image small-x (left side) = world-left = POSITIVE yaw_err
    # (matching steer.py's egocentric_goal convention where positive yaw_err = CCW = left).
    # The camera's image x-axis is mirrored relative to the robot's lateral axis.
    # Fix: negate x_robot so that image-left → positive yaw_err (turn left toward target).
    # (Verified empirically to hold at 58° pitch too -- lateral/bearing sign is unaffected
    # by the use_corrected_unpitch fix above, only the forward-distance term was wrong.)
    yaw_err = math.atan2(-x_robot, z_robot)  # positive when target is to the LEFT (CCW)
    return dist, yaw_err
