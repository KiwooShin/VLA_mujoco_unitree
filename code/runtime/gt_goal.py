"""
code.runtime.gt_goal — privileged GT goal computation for the closed-loop
Inferencer.

RF-1 split of code/inferencer.py (docs/refactor_plan.md): `_compute_gt_goal`
moved verbatim, no logic changes. Kept import-visible at the old
`code.inferencer` path — 3 external importers rely on
`from code.inferencer import _compute_gt_goal`: code/check_ep0_objects.py,
code/check_goal2.py, code/check_goal_ep0.py (code/verify_settle.py imports it
too, as a 4th caller).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from code.sim.teacher import _yaw_of


def _compute_gt_goal(data_mj: mujoco.MjData, target_xy: np.ndarray) -> np.ndarray:
    """Computes the privileged GT goal (dist, cosθ, sinθ) from simulation state.

    The goal is egocentric: direction from robot to target in the robot's
    horizontal body frame (yaw-aligned).

    Args:
        data_mj: MuJoCo data holding the current physics state.
        target_xy: (2,) target position in world frame (m).

    Returns:
        np.float32[3]: (dist, cos(yaw_err), sin(yaw_err)).
    """
    robot_xy = data_mj.qpos[0:2].copy()
    delta = target_xy - robot_xy  # world-frame vector to target
    dist = float(np.linalg.norm(delta))
    robot_yaw = _yaw_of(data_mj.qpos[3:7])
    # Rotate delta into robot frame (yaw-only rotation)
    cos_y, sin_y = math.cos(robot_yaw), math.sin(robot_yaw)
    # world→robot: x_r = cos_y*dx + sin_y*dy, y_r = -sin_y*dx + cos_y*dy
    dx, dy = delta
    fwd  =  cos_y * dx + sin_y * dy   # forward in robot frame
    lat  = -sin_y * dx + cos_y * dy   # lateral in robot frame (right positive)
    yaw_err = math.atan2(lat, fwd)    # positive = target to the left
    return np.array([dist, math.cos(yaw_err), math.sin(yaw_err)], dtype=np.float32)
