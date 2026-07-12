"""World -> BEV-pixel projection + tiny cv2 pixel-pushing helpers for the
fancy demo's overlays (code/fancy_demo.py, RF-1 split).

`world_to_bev_pixel` is the shared projection primitive every BEV overlay in
code/apps/fancy/overlays_bev.py builds on; `_dashed_line`/`_lerp_color_bgr`
(+ the trail gradient endpoints) are pure drawing/color-math helpers with no
state reads beyond their arguments.
"""

from __future__ import annotations

import math

import numpy as np

from code.apps.fancy.constants import BEV_W, BEV_H


def world_to_bev_pixel(
    world_pts: np.ndarray,   # (N, 3) world XYZ
    bev_cam: "mujoco.MjvCamera",
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    w: int = BEV_W,
    h: int = BEV_H,
    fovy_deg: float = 45.0,
) -> np.ndarray:
    """Project world XYZ points into BEV pixel coordinates.

    Uses MuJoCo's camera view matrix + a pinhole projection. Clips
    out-of-frame points but does not filter them.

    Args:
        world_pts: (N, 3) world XYZ points to project (or (3,) for a single
            point, promoted to (1, 3)).
        bev_cam: MuJoCo free camera (lookat/azimuth/elevation/distance) used
            for the BEV follow-cam view.
        model: MuJoCo model (unused in this function; kept for interface
            parity with the other render helpers).
        data: MuJoCo data (unused in this function; kept for interface
            parity with the other render helpers).
        w: Output image width in pixels.
        h: Output image height in pixels.
        fovy_deg: Vertical field of view in degrees.

    Returns:
        (N, 2) float32 array of (u, v) pixel coordinates.
    """
    import mujoco

    # Build camera view matrix from lookat / azimuth / elevation / distance.
    #
    # VF-3 fix (docs/vf3_bev_fixes.md): the formula below was previously
    # cam_fwd = (-sin(az)*cos(el), cos(az)*cos(el), sin(el)) -- i.e. the
    # world-space (cos(az), sin(az)) direction rotated +90 deg. This does NOT
    # match MuJoCo's real mjCAMERA_FREE convention, so every BEV overlay
    # (FOV cone + path trail + AVOID viz) that goes through this function was
    # silently drawn ~90 deg rotated from what the ACTUAL rendered BEV image
    # (produced by renderer.render_tp() -> real MuJoCo camera math) shows.
    #
    # Ground truth for MuJoCo's real convention comes from code/arena.py's
    # `_set_ego_cam` (empirically verified pitch-independent by CAM-P0,
    # docs/cam_p0.md, via cam.distance=1.0): it sets cam.azimuth=degrees(yaw),
    # cam.elevation=-pitch_deg, and its OWN forward vector (used to place
    # `cam.lookat`) is (cos(pitch)*cos(yaw), cos(pitch)*sin(yaw), -sin(pitch))
    # == (cos(el)*cos(az), cos(el)*sin(az), sin(el)) with el=-pitch, az=yaw.
    # Verified empirically here too: rendering known-position colored markers
    # via the real render_tp() and comparing their true pixel centroid against
    # this function's projection dropped the error from 300-560px (old buggy
    # formula) to 1-8px (this formula) across yaw=0/90/offset-position cases.
    az  = math.radians(bev_cam.azimuth)
    el  = math.radians(bev_cam.elevation)  # negative = below horizon
    dist = bev_cam.distance

    cosel = math.cos(el)
    sinel = math.sin(el)
    cosaz = math.cos(az)
    sinaz = math.sin(az)

    # Camera forward (from cam toward lookat) -- MuJoCo's real convention.
    cam_fwd = np.array([cosaz * cosel, sinaz * cosel, sinel], dtype=np.float64)

    lookat = np.array(bev_cam.lookat, dtype=np.float64)
    cam_pos = lookat - dist * cam_fwd

    # Camera right = cross(fwd, up) normalized; up is approximately world +Z
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cam_right = np.cross(cam_fwd, world_up)
    norm_r = np.linalg.norm(cam_right)
    if norm_r < 1e-8:
        cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        cam_right = cam_right / norm_r
    cam_up = np.cross(cam_right, cam_fwd)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)

    # Pinhole projection
    fovy_rad = math.radians(fovy_deg)
    fy = (h / 2.0) / math.tan(fovy_rad / 2.0)
    fovx_rad = 2.0 * math.atan(math.tan(fovy_rad / 2.0) * w / h)
    fx = (w / 2.0) / math.tan(fovx_rad / 2.0)
    cx, cy = w / 2.0 - 0.5, h / 2.0 - 0.5

    pts = np.asarray(world_pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]

    # Transform to camera frame
    delta = pts - cam_pos[np.newaxis, :]  # (N, 3)
    z_cam =  np.dot(delta, cam_fwd)     # forward  (N,)
    x_cam =  np.dot(delta, cam_right)   # right    (N,)
    y_cam = -np.dot(delta, cam_up)      # down (screen Y = down)

    # Perspective divide
    z_cam_safe = np.where(z_cam > 0.01, z_cam, 0.01)
    u = fx * x_cam / z_cam_safe + cx
    v = fy * y_cam / z_cam_safe + cy

    return np.stack([u, v], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# VF-1 small drawing helpers (pure cv2 pixel-pushing, no state reads beyond
# their arguments)
# ---------------------------------------------------------------------------
def _dashed_line(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 9,
    gap_len: int = 7,
) -> None:
    """Draw a dashed line segment from p0 to p1 (both (x,y) int tuples)."""
    import cv2
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1e-6:
        return
    n_dashes = max(1, int(length / (dash_len + gap_len)))
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        seg_end = min(pos + dash_len, length)
        sx, sy = x0 + ux * pos, y0 + uy * pos
        ex, ey = x0 + ux * seg_end, y0 + uy * seg_end
        cv2.line(img, (int(round(sx)), int(round(sy))),
                  (int(round(ex)), int(round(ey))), color, thickness, cv2.LINE_AA)
        pos += dash_len + gap_len


def _lerp_color_bgr(
    c_cool: tuple[int, int, int], c_warm: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    """Linear-interpolate two BGR color tuples, t in [0,1] (0=cool, 1=warm)."""
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c_cool, c_warm))


# Path-trail gradient endpoints (BGR): cool blue (old) -> warm orange/red (recent).
TRAIL_COOL_BGR: tuple[int, int, int] = (230, 120, 40)   # blue-ish
TRAIL_WARM_BGR: tuple[int, int, int] = (30,  90, 255)   # warm orange-red
