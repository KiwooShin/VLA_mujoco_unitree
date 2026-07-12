"""
arena_cameras.py — Pure camera-frame math for the G1Nav arena (RF-1 split).

Role: this file owns the geometry that turns robot pose + a pitch angle into
a positioned ``mujoco.MjvCamera``, the corresponding pinhole intrinsics, and
the inverse operation (pixel + depth -> camera-frame 3D point). It has no
MuJoCo renderer/EGL lifecycle (that is ``arena_render.py``) and no scene/model
construction (that is ``arena_build.py``).
"""

import math

import numpy as np
import mujoco

from code.sim.arena_build import CAM_FWD, CAM_HEAD_Z, CAM_PITCH, EGO_FOVY, EGO_H, EGO_W


# ---------------------------------------------------------------------------
# Ego-camera helpers
# ---------------------------------------------------------------------------
def _set_ego_cam(cam: mujoco.MjvCamera, qpos: np.ndarray, yaw: float,
                 pitch_deg: float = CAM_PITCH) -> None:
    """Position the free camera to simulate a robot-head ego camera.

    Args:
        cam: MjvCamera to position in-place.
        qpos: Robot qpos array; qpos[0:3] is the pelvis (x, y, z) position.
        yaw: Robot heading (rad).
        pitch_deg: Downward camera tilt in degrees (default=CAM_PITCH=32deg).
            For grounding renders, use GROUNDING_PITCH=20deg so distant targets
            (4-9m) appear in the image rather than falling below the bottom edge.
    """
    px, py, pz = qpos[0], qpos[1], qpos[2]
    # Camera origin: slightly forward + at head height
    cx = px + CAM_FWD * math.cos(yaw)
    cy = py + CAM_FWD * math.sin(yaw)
    cz = pz + CAM_HEAD_Z

    pitch_rad = math.radians(pitch_deg)
    # Forward direction of camera (tilted down)
    dx = math.cos(pitch_rad) * math.cos(yaw)
    dy = math.cos(pitch_rad) * math.sin(yaw)
    dz = -math.sin(pitch_rad)

    cam.lookat[:] = [cx + dx, cy + dy, cz + dz]
    # P0 fix (2026-07-08): cam.distance was 0.001 ("nearly pinhole"), but MuJoCo's
    # free camera places the eye at `lookat - distance*forward`. With lookat set to
    # `origin + 1.0*forward_dir` above, the true eye was `origin + (1-distance)*forward_dir`
    # -- i.e. it silently DRIFTED with pitch (drifted ~0.947m fwd / ~0.53m low at 32 deg).
    # This is exactly why grounding.py needed the empirical, pitch-specific
    # CAM_ROBOT_FORWARD_OFFSET_M=0.947 hack (valid only at 32 deg).
    # Setting distance=1.0 makes (1-distance)=0, so the eye sits EXACTLY at
    # `origin` (cx,cy,cz) regardless of pitch -- decoupling camera position from
    # tilt. This generalizes to any pitch (needed for multi-cam / dynamic-tilt
    # options) and lets CAM_ROBOT_FORWARD_OFFSET_M collapse to the constant
    # CAM_FWD (recalibrated in grounding.py).
    cam.distance   = 1.0
    cam.azimuth    = math.degrees(yaw)
    cam.elevation  = -pitch_deg


def get_ego_intrinsics(w: int = EGO_W, h: int = EGO_H,
                       fovy_deg: float = EGO_FOVY) -> dict:
    """Return pinhole camera intrinsics for the ego camera.

    Args:
        w: Image width in pixels.
        h: Image height in pixels.
        fovy_deg: Vertical field of view in degrees.

    Returns:
        dict with keys: fx, fy, cx, cy (all in pixels), width, height, fovy_deg.
    """
    fovy_rad = math.radians(fovy_deg)
    fy = (h / 2.0) / math.tan(fovy_rad / 2.0)
    fovx_rad = 2.0 * math.atan(math.tan(fovy_rad / 2.0) * w / h)
    fx = (w / 2.0) / math.tan(fovx_rad / 2.0)
    return dict(fx=fx, fy=fy, cx=w / 2.0 - 0.5, cy=h / 2.0 - 0.5,
                width=w, height=h, fovy_deg=fovy_deg)


# ---------------------------------------------------------------------------
# Back-projection helper (used by grounding.py)
# ---------------------------------------------------------------------------
def backproject_pixel(u: float, v: float, depth_m: float, intr: dict) -> np.ndarray:
    """Back-project image pixel (u,v) at given depth (metres) to camera-frame 3D point.

    Camera frame: x=right, y=down, z=forward (OpenCV convention).

    Args:
        u: Pixel column (x, right).
        v: Pixel row (y, down).
        depth_m: Depth at (u, v) in metres.
        intr: Camera intrinsics dict with keys fx, fy, cx, cy.

    Returns:
        np.ndarray shape (3,)  [x_cam, y_cam, z_cam]
    """
    x = (u - intr["cx"]) * depth_m / intr["fx"]
    y = (v - intr["cy"]) * depth_m / intr["fy"]
    z = depth_m
    return np.array([x, y, z], dtype=np.float32)
