"""
code/datagen/gen_det_capture.py — Persistent segmentation renderer + frame capture.

Role: split out of gen_det_dataset.py (RF-1) — the rendering-dependent
frame-capture layer (RGB + depth + segmentation -> labeled frame dict).
"""

from __future__ import annotations

import mujoco
import numpy as np

from code.arena import (
    ArenaRenderer, GROUNDING_H, GROUNDING_PITCH, GROUNDING_W, PROXIMITY_H,
    PROXIMITY_PITCH, PROXIMITY_W, _set_ego_cam,
)
from code.datagen.gen_det_labels import derive_object_labels, seg_to_objmap
from code.teacher import _yaw_of


# ---------------------------------------------------------------------------
# Persistent segmentation renderers (avoid EGL context exhaustion — same
# pre-allocate-once pattern as ArenaRenderer's own renderers)
# ---------------------------------------------------------------------------
class SegRenderer:
    """Persistent grounding/proximity segmentation renderers.

    Both MuJoCo renderers are pre-allocated once (the same
    pre-allocate-once pattern as `ArenaRenderer`'s own renderers) to avoid
    EGL context exhaustion from repeated renderer creation.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        """Allocates the grounding and proximity segmentation renderers.

        Args:
            model: Compiled MuJoCo model to render segmentation masks for.
        """
        self._gr_rend = mujoco.Renderer(model, GROUNDING_H, GROUNDING_W)
        self._pr_rend = mujoco.Renderer(model, PROXIMITY_H, PROXIMITY_W)
        self._gr_cam = mujoco.MjvCamera(); self._gr_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._pr_cam = mujoco.MjvCamera(); self._pr_cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def render(self, data: mujoco.MjData, yaw: float, cam_type: str) -> np.ndarray:
        """Renders a segmentation mask from the given robot pose.

        Args:
            data: MuJoCo data holding the current physics state.
            yaw: Robot yaw, in radians, used to place the egocentric camera.
            cam_type: Either "proximity" or anything else for "grounding".

        Returns:
            (H, W, 2) int32 segmentation array; channel 0 holds geom ids.
        """
        if cam_type == "proximity":
            rend, cam, pitch = self._pr_rend, self._pr_cam, PROXIMITY_PITCH
        else:
            rend, cam, pitch = self._gr_rend, self._gr_cam, GROUNDING_PITCH
        _set_ego_cam(cam, data.qpos, yaw, pitch_deg=pitch)
        rend.update_scene(data, cam)
        rend.enable_segmentation_rendering()
        seg = rend.render().copy()
        rend.disable_segmentation_rendering()
        return seg

    def close(self) -> None:
        """Closes both underlying MuJoCo renderers."""
        self._gr_rend.close()
        self._pr_rend.close()


# ---------------------------------------------------------------------------
# Frame capture: RGB + depth + segmentation -> per-object label rows
# ---------------------------------------------------------------------------
def capture_frame(renderer: ArenaRenderer, seg_rend: SegRenderer,
                  data_mj: mujoco.MjData, yaw: float, cam_type: str,
                  objects: list, id_to_obj: np.ndarray) -> dict:
    """Captures one frame: renders RGB/depth/segmentation and derives labels.

    Args:
        renderer: ArenaRenderer used to render RGB/depth for the camera.
        seg_rend: SegRenderer used to render the segmentation mask.
        data_mj: MuJoCo data holding the current physics state.
        yaw: Robot yaw, in radians, used to place the egocentric camera.
        cam_type: Either "proximity" or "grounding".
        objects: List of object dicts from the scene config.
        id_to_obj: Geom-id -> object-index lookup array, from
            `build_id_to_obj`.

    Returns:
        A dict with keys rgb, depth (float16), cam_type, robot_x, robot_y,
        robot_yaw, qpos, n_objects_visible, and labels (from
        `derive_object_labels`).
    """
    if cam_type == "proximity":
        rgb, depth, intr = renderer.render_proximity(data_mj, yaw, render_depth=True)
    else:
        rgb, depth, intr = renderer.render_grounding(data_mj, yaw, render_depth=True)
    seg = seg_rend.render(data_mj, yaw, cam_type)
    obj_map = seg_to_objmap(seg, id_to_obj)

    robot_xy = data_mj.qpos[0:2].copy()
    robot_yaw = _yaw_of(data_mj.qpos[3:7])
    labels = derive_object_labels(rgb, depth, obj_map, objects, robot_xy, robot_yaw, intr)

    return dict(
        rgb=rgb, depth=depth.astype(np.float16), cam_type=cam_type,
        robot_x=float(robot_xy[0]), robot_y=float(robot_xy[1]), robot_yaw=float(robot_yaw),
        qpos=data_mj.qpos.copy().astype(np.float32),
        n_objects_visible=len(labels), labels=labels,
    )
