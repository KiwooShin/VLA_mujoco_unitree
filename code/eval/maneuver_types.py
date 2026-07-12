"""code.eval.maneuver_types — maneuver result schema + rollout-support helpers.

Split out of the original ``eval_maneuver.py`` (RF-1): the per-episode result
schema (``ManeuverResult``), the constants shared by the rollout loop and the
evaluator, and small pure(ish) helpers (``_build_proprio_maneuver``,
``_apply_student_pd``) plus the video writer (``_write_video``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

from code.gen_dart_dataset import GaitPhaseTracker
from code.dataset_maneuver import PROPRIO_DIM_BASE
from code.teacher import KPS, KDS, NUM_ACTIONS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FALL_HEIGHT       = 0.50
PROPRIO_K         = 6
ACTION_SCALE      = 0.25
HEADING_SUCCESS_THR = math.radians(25.0)  # heading error must be < 25 deg to succeed
IMG_SIZE          = 128
HOLD_STEPS_REQUIRED = 5


@dataclass
class ManeuverResult:
    success:            bool
    failure_tag:        str     # 'success'|'fall'|'no_landmark'|'wrong_heading'
    steps:              int
    fell:               bool
    upright:            bool
    landmark_passed:    bool
    final_heading_err:  float   # rad
    final_state:        int     # FSM state at end
    ms_per_step:        float
    scene_cfg:          dict = field(default_factory=dict)
    video_path:         str | None = None


def _build_proprio_maneuver(data_mj: mujoco.MjData,
                             prev_action: np.ndarray,
                             phase_tracker: GaitPhaseTracker,
                             priv: dict) -> np.ndarray:
    """Build 62-d maneuver proprio.

    Layout:
      [0:55]  base proprio
      [55:57] gait phase [sin, cos]
      [57:62] maneuver features [subgoal_norm, cos_target, sin_target, heading_err_norm, lm_passed]

    Args:
        data_mj: MuJoCo sim data to read qpos/qvel from.
        prev_action: Previous step's target DOF vector, folded into the base
            proprio.
        phase_tracker: Gait phase tracker providing [sin, cos] phase features.
        priv: Privileged maneuver state dict (subgoal_index, cos_target,
            sin_target, heading_err, landmark_passed) from the FSM expert.

    Returns:
        Concatenated (62,) float32 proprio vector.
    """
    # Base 55-d
    p = np.empty(PROPRIO_DIM_BASE, dtype=np.float32)
    p[0:15]  = data_mj.qpos[7:22]
    p[15:30] = data_mj.qvel[6:21]
    p[30:34] = data_mj.qpos[3:7]
    p[34:37] = data_mj.qvel[3:6]
    p[37:40] = data_mj.qvel[0:3]
    p[40:55] = prev_action

    # Gait phase
    q_lb = data_mj.qpos[7:22].copy()
    ph_raw = phase_tracker.update(q_lb)
    ph = np.array(ph_raw, dtype=np.float32)   # ensure float32

    # Maneuver features
    man = np.array([
        float(priv["subgoal_index"]) / 2.0,
        float(priv["cos_target"]),
        float(priv["sin_target"]),
        float(priv["heading_err"]) / np.pi,
        float(priv["landmark_passed"]),
    ], dtype=np.float32)

    return np.concatenate([p, ph, man])   # (62,)


def _apply_student_pd(data_mj: mujoco.MjData, target_dof: np.ndarray, nj: int) -> None:
    """Apply PD control torques toward target_dof, writing into data_mj.ctrl."""
    leg_tau = (
        (target_dof - data_mj.qpos[7:7 + NUM_ACTIONS]) * KPS
        + (0.0 - data_mj.qvel[6:6 + NUM_ACTIONS]) * KDS
    )
    data_mj.ctrl[:NUM_ACTIONS] = leg_tau
    if nj > NUM_ACTIONS:
        arm_tau = (
            (0.0 - data_mj.qpos[7 + NUM_ACTIONS:7 + nj]) * 100.0
            + (0.0 - data_mj.qvel[6 + NUM_ACTIONS:6 + nj]) * 0.5
        )
        data_mj.ctrl[NUM_ACTIONS:nj] = arm_tau


def _write_video(
    frames_ego: list[np.ndarray],
    frames_tp: list[np.ndarray],
    out_path: str,
    fps: int = 50,
) -> None:
    """Write ego (optionally side-by-side with third-person) frames to an mp4."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    import cv2

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if frames_tp and len(frames_tp) == len(frames_ego):
        combo = []
        for ego, tp in zip(frames_ego, frames_tp):
            eh, ew = ego.shape[:2]
            th, tw = tp.shape[:2]
            if th != eh:
                tp = cv2.resize(tp, (int(tw * eh / th), eh))
            combo.append(np.concatenate([ego, tp], axis=1))
        frames_out = combo
    else:
        frames_out = frames_ego
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=1)
    for f in frames_out:
        writer.append_data(f.astype(np.uint8))
    writer.close()
    print(f"[eval_maneuver] Video: {out_path} ({len(frames_out)} frames)", flush=True)
