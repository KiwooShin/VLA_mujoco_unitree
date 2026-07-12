"""
arena_render.py — ArenaRenderer: EGL renderer lifecycle for the G1Nav arena.

Role: this file owns the runtime rendering objects — one ``mujoco.Renderer``
per camera stream (ego, grounding, proximity, optional widefov, third-person)
plus their ``mujoco.MjvCamera`` handles — allocated once per arena instance
and reused every step (avoids EGL context exhaustion). Camera *positioning*
math lives in ``arena_cameras.py``; scene/model construction lives in
``arena_build.py``.
"""

import os

import numpy as np
import mujoco

from code.sim.arena_build import (
    CAMERA_MODE, EGO_FOVY, EGO_H, EGO_W,
    GROUNDING_H, GROUNDING_PITCH, GROUNDING_W,
    PROXIMITY_H, PROXIMITY_PITCH, PROXIMITY_W,
    TP_H, TP_W, WIDEFOV_FOVY, WIDEFOV_H, WIDEFOV_PITCH, WIDEFOV_W,
)
from code.sim.arena_cameras import get_ego_intrinsics, _set_ego_cam

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
_NVIDIA_EGL_ICD = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if os.path.exists(_NVIDIA_EGL_ICD):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NVIDIA_EGL_ICD)


class ArenaRenderer:
    """
    Manages ego + third-person renderers for an arena model.

    V2: Added separate high-resolution grounding renderer (480x360) for improved
    detection at 4-9m demo distances. The grounding renderer uses FOVY=45deg (same as
    ego camera) but at 1.5x resolution so distant targets occupy more pixels.

    Usage
    -----
    renderer = ArenaRenderer(model)
    rgb, depth, intr = renderer.render_ego(data, yaw)
    rgb_gr, depth_gr, intr_gr = renderer.render_grounding(data, yaw)
    tp_rgb            = renderer.render_tp(data, tp_cam)
    renderer.close()
    """

    def __init__(self, model: mujoco.MjModel,
                 ego_w: int = EGO_W, ego_h: int = EGO_H,
                 grounding_w: int = GROUNDING_W, grounding_h: int = GROUNDING_H,
                 proximity_w: int = PROXIMITY_W, proximity_h: int = PROXIMITY_H,
                 tp_w: int = TP_W, tp_h: int = TP_H,
                 widefov_w: int = WIDEFOV_W, widefov_h: int = WIDEFOV_H) -> None:
        """Allocate the ego/grounding/proximity/TP renderers and cameras.

        Args:
            model: Compiled arena MjModel to render.
            ego_w: Ego camera render width.
            ego_h: Ego camera render height.
            grounding_w: Grounding camera render width.
            grounding_h: Grounding camera render height.
            proximity_w: Proximity camera render width.
            proximity_h: Proximity camera render height.
            tp_w: Third-person camera render width.
            tp_h: Third-person camera render height.
            widefov_w: Wide-FOV camera render width (only used when
                CAMERA_MODE=='widefov').
            widefov_h: Wide-FOV camera render height (only used when
                CAMERA_MODE=='widefov').
        """
        self._model    = model
        self._ego_w    = ego_w
        self._ego_h    = ego_h
        self._grounding_w = grounding_w
        self._grounding_h = grounding_h
        self._proximity_w = proximity_w
        self._proximity_h = proximity_h
        self._ego_rend = mujoco.Renderer(model, ego_h, ego_w)
        # V2: dedicated grounding renderer at higher resolution
        # Reuse a single renderer (not created per-call) to avoid EGL context exhaustion
        self._gr_rend  = mujoco.Renderer(model, grounding_h, grounding_w)
        # CAM-2 (Phase 1): dedicated proximity renderer, same pre-allocate-once pattern
        # (no EGL context exhaustion — precedent from _gr_rend). Only ever rendered when
        # the Schmitt-trigger handoff in inferencer.py has selected the proximity camera,
        # so steady-state per-cycle cost is unchanged (one render per grounding cycle).
        self._prox_rend = mujoco.Renderer(model, proximity_h, proximity_w)
        self._tp_rend  = mujoco.Renderer(model, tp_h, tp_w)
        self._egc      = mujoco.MjvCamera()
        self._egc.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._gr_cam   = mujoco.MjvCamera()
        self._gr_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._prox_cam = mujoco.MjvCamera()
        self._prox_cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        # CAM-1 (Phase 2, toggle): additional wide-FOV renderer, built ONLY when
        # CAMERA_MODE=='widefov'. Purely additive — every line above is constructed
        # exactly as before regardless of mode, so cam2's renderer set (and EGL context
        # count) is completely unaffected when the toggle is at its 'cam2' default.
        self._widefov_w    = widefov_w
        self._widefov_h    = widefov_h
        self._widefov_rend = None
        self._widefov_cam  = None
        if CAMERA_MODE == 'widefov':
            self._widefov_rend = mujoco.Renderer(model, widefov_h, widefov_w)
            self._widefov_cam  = mujoco.MjvCamera()
            self._widefov_cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def render_ego(self, data: mujoco.MjData, yaw: float,
                   render_depth: bool = True) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Render the ego (head) camera at native 320x240 resolution.

        Args:
            data: MuJoCo simulation data (provides qpos for camera placement).
            yaw: Robot heading (rad) used to orient the ego camera.
            render_depth: If False, skip depth rendering (saves ~15ms/frame on EGL).

        Returns:
            rgb: np.ndarray  shape (H,W,3)  uint8
            depth: np.ndarray  shape (H,W)  float32  (metres) or None
            intr: dict  {fx,fy,cx,cy,width,height,fovy_deg}
        """
        _set_ego_cam(self._egc, data.qpos, yaw)

        self._ego_rend.update_scene(data, self._egc)
        rgb = self._ego_rend.render().copy()

        depth = None
        if render_depth:
            self._ego_rend.enable_depth_rendering()
            self._ego_rend.update_scene(data, self._egc)
            depth = self._ego_rend.render().copy().astype(np.float32)
            self._ego_rend.disable_depth_rendering()

        intr = get_ego_intrinsics(self._ego_w, self._ego_h, EGO_FOVY)
        return rgb, depth, intr

    def render_grounding(self, data: mujoco.MjData, yaw: float,
                         render_depth: bool = True) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Render at higher-resolution (480x360) with shallower pitch for demo-distance grounding.

        V2 key changes vs render_ego:
        1. Resolution: 480x360 (1.5x) -> targets are 2.25x larger in pixel area.
        2. Camera pitch: 20deg (vs 32deg for ego) -> targets at 4-9m stay in frame.
           At 32deg: target at 7m falls below image. At 20deg: target at 9m at row 279/360.
        3. Same EGL context (single renderer object) -> no context exhaustion.

        Bearing calculation: cam_to_egocentric() must use GROUNDING_PITCH (not CAM_PITCH)
        to correctly un-pitch the camera frame. This is handled in grounding.py.

        Args:
            data: MuJoCo simulation data (provides qpos for camera placement).
            yaw: Robot heading (rad) used to orient the grounding camera.
            render_depth: If False, skip depth rendering.

        Returns:
            rgb: np.ndarray  shape (360,480,3)  uint8
            depth: np.ndarray  shape (360,480)  float32  (metres) or None
            intr: dict  {fx,fy,cx,cy,width,height,fovy_deg,pitch_deg}
        """
        _set_ego_cam(self._gr_cam, data.qpos, yaw, pitch_deg=GROUNDING_PITCH)

        self._gr_rend.update_scene(data, self._gr_cam)
        rgb = self._gr_rend.render().copy()

        depth = None
        if render_depth:
            self._gr_rend.enable_depth_rendering()
            self._gr_rend.update_scene(data, self._gr_cam)
            depth = self._gr_rend.render().copy().astype(np.float32)
            self._gr_rend.disable_depth_rendering()

        # Intrinsics for grounding resolution (FOVY=45deg same as ego camera, scaled for 480x360)
        from code.grounding import get_ego_intrinsics_rendered
        intr = get_ego_intrinsics_rendered(self._grounding_w, self._grounding_h)
        # Include pitch so cam_to_egocentric can use the correct tilt
        intr['pitch_deg'] = GROUNDING_PITCH
        return rgb, depth, intr

    def render_proximity(self, data: mujoco.MjData, yaw: float,
                         render_depth: bool = True) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """CAM-2 (Phase 1): render the steep-pitch proximity camera, same head mount as
        render_ego/render_grounding (only PROXIMITY_PITCH differs — no XML change, no
        new offset calibration needed post-P0, see arena_build.py constants comment).

        Covers ~0.22-1.81m (docs/cam_opt2_multicam.md geometry table), i.e. the final
        close-range approach where the shallower ego/grounding cameras' target has
        already fallen below the frame. Selected by the Schmitt-trigger handoff in
        inferencer.py — NOT rendered every cycle, only when active_cam==PROXIMITY.

        Args:
            data: MuJoCo simulation data (provides qpos for camera placement).
            yaw: Robot heading (rad) used to orient the proximity camera.
            render_depth: If False, skip depth rendering.

        Returns:
            rgb: np.ndarray  shape (PROXIMITY_H,PROXIMITY_W,3)  uint8
            depth: np.ndarray  shape (PROXIMITY_H,PROXIMITY_W)  float32  (metres) or None
            intr: dict  {fx,fy,cx,cy,width,height,fovy_deg,pitch_deg,is_proximity}
        """
        _set_ego_cam(self._prox_cam, data.qpos, yaw, pitch_deg=PROXIMITY_PITCH)

        self._prox_rend.update_scene(data, self._prox_cam)
        rgb = self._prox_rend.render().copy()

        depth = None
        if render_depth:
            self._prox_rend.enable_depth_rendering()
            self._prox_rend.update_scene(data, self._prox_cam)
            depth = self._prox_rend.render().copy().astype(np.float32)
            self._prox_rend.disable_depth_rendering()

        from code.grounding import get_ego_intrinsics_rendered
        intr = get_ego_intrinsics_rendered(self._proximity_w, self._proximity_h)
        intr['pitch_deg']    = PROXIMITY_PITCH
        # Flag consumed by grounding.ground() to activate the stricter self-body-rejection
        # depth floor/geometry check (steeper pitch -> optical center closer to the robot's
        # own chest, see docs/cam_opt2_multicam.md "Real risk" + docs/cam_p0.md ep14 finding).
        intr['is_proximity'] = True
        return rgb, depth, intr

    def render_widefov(self, data: mujoco.MjData, yaw: float,
                       render_depth: bool = True) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """CAM-1 (Phase 2, toggle, docs/cam_opt1_widefov.md / docs/cam_p2.md): render the
        single wide-FOV camera, same head mount as render_ego/render_grounding
        (CAM_HEAD_Z, CAM_FWD unchanged), pitch=WIDEFOV_PITCH, FOVY=WIDEFOV_FOVY (the
        actual rendered FOVY, set at build time via spec.visual.global_.fovy in
        build_arena() — only when CAMERA_MODE=='widefov', so this method is only
        meaningful/called in that mode; the renderer is None otherwise, see __init__).

        No proximity camera, no handoff — this single render is the entire grounding
        camera for CAM-1.

        Args:
            data: MuJoCo simulation data (provides qpos for camera placement).
            yaw: Robot heading (rad) used to orient the wide-FOV camera.
            render_depth: If False, skip depth rendering.

        Returns:
            rgb: np.ndarray  shape (WIDEFOV_H,WIDEFOV_W,3)  uint8
            depth: np.ndarray  shape (WIDEFOV_H,WIDEFOV_W)  float32  (metres) or None
            intr: dict  {fx,fy,cx,cy,width,height,fovy_deg,pitch_deg,is_widefov}
        """
        _set_ego_cam(self._widefov_cam, data.qpos, yaw, pitch_deg=WIDEFOV_PITCH)

        self._widefov_rend.update_scene(data, self._widefov_cam)
        rgb = self._widefov_rend.render().copy()

        depth = None
        if render_depth:
            self._widefov_rend.enable_depth_rendering()
            self._widefov_rend.update_scene(data, self._widefov_cam)
            depth = self._widefov_rend.render().copy().astype(np.float32)
            self._widefov_rend.disable_depth_rendering()

        # Actual rendered FOVY (WIDEFOV_FOVY) fed straight into the standard pinhole
        # intrinsics formula — this is exactly cam_opt1_widefov.md Finding #1's fix,
        # scoped to the widefov path only (cam2's get_ego_intrinsics_rendered() hardcode
        # of 45 deg is untouched).
        intr = get_ego_intrinsics(self._widefov_w, self._widefov_h, WIDEFOV_FOVY)
        intr['pitch_deg']  = WIDEFOV_PITCH
        intr['is_widefov'] = True
        return rgb, depth, intr

    def render_tp(self, data: mujoco.MjData, tp_cam: mujoco.MjvCamera) -> np.ndarray:
        """Render third-person view (for video only)."""
        self._tp_rend.update_scene(data, tp_cam)
        return self._tp_rend.render().copy()

    def make_tp_cam(self) -> mujoco.MjvCamera:
        """Create a default third-person MjvCamera (not yet tracking any body)."""
        cam        = mujoco.MjvCamera()
        cam.type   = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance  = 5.0
        cam.azimuth   = 135.0
        cam.elevation = -20.0
        return cam

    def update_tp_cam(self, tp_cam: mujoco.MjvCamera, data: mujoco.MjData,
                      distance: float = 5.0) -> None:
        """Track the robot with the third-person camera."""
        bxy = data.qpos[0:2]
        tp_cam.lookat[:] = [bxy[0], bxy[1], 0.5]
        tp_cam.distance   = distance
        tp_cam.azimuth    = 135.0
        tp_cam.elevation  = -20.0

    def close(self) -> None:
        """Close all underlying EGL renderers."""
        self._ego_rend.close()
        self._gr_rend.close()
        self._prox_rend.close()
        self._tp_rend.close()
        if self._widefov_rend is not None:
            self._widefov_rend.close()
