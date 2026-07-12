"""
code.runtime.helpers — pure per-step helpers for the closed-loop Inferencer
(proprio vector construction, student PD torques, image tensor conversion,
CAM-2 demo-viz labeling).

RF-1 split of code/inferencer.py (docs/refactor_plan.md): moved verbatim, no
logic changes. Kept import-visible at the old `code.inferencer` path (several
external callers do `from code.inferencer import _build_proprio,
_apply_student_pd, ...` — code/apps/repl/maneuver_inferencer.py,
code/eval/search_rollout_step.py, code/eval/search_rollout_state.py,
code/render_showcase_videos.py, code/bench_widefov_visibility.py,
code/apps/fancy/rollout.py, code/verify_settle.py).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from code.runtime.constants import PROPRIO_DIM, IMG_SIZE
from code.sim.teacher import NUM_ACTIONS, KPS, KDS


def _build_proprio(data: mujoco.MjData, prev_action: np.ndarray) -> np.ndarray:
    """Builds the 55-d proprio vector (exact layout from dataset.md / gen_dataset.py).

      [0:15]  lower-body joint positions
      [15:30] lower-body joint velocities
      [30:34] base IMU quaternion [w,x,y,z]
      [34:37] base angular velocity (rad/s)
      [37:40] base linear velocity (world frame)
      [40:55] prev_action (15 joint targets)

    Args:
        data: MuJoCo data holding the current physics state.
        prev_action: (15,) previous step's joint targets.

    Returns:
        (55,) float32 proprio vector.
    """
    p = np.empty(PROPRIO_DIM, dtype=np.float32)
    p[0:15]  = data.qpos[7:22]
    p[15:30] = data.qvel[6:21]
    p[30:34] = data.qpos[3:7]   # [w,x,y,z]
    p[34:37] = data.qvel[3:6]
    p[37:40] = data.qvel[0:3]
    p[40:55] = prev_action
    return p


def _apply_student_pd(data: mujoco.MjData, target_dof: np.ndarray, nj: int) -> None:
    """Applies PD torques from student joint targets (mirrored exactly from teacher.py).

    Args:
        data: MuJoCo data; `data.ctrl` is written in place.
        target_dof: (NUM_ACTIONS,) commanded lower-body joint targets.
        nj: Total number of actuated joints (lower-body + optional upper-body).
    """
    leg_tau = (
        (target_dof - data.qpos[7:7 + NUM_ACTIONS]) * KPS
        + (0.0 - data.qvel[6:6 + NUM_ACTIONS]) * KDS
    )
    data.ctrl[:NUM_ACTIONS] = leg_tau
    if nj > NUM_ACTIONS:
        n_upper = nj - NUM_ACTIONS
        arm_tau = (
            (0.0 - data.qpos[7 + NUM_ACTIONS:7 + nj]) * 100.0
            + (0.0 - data.qvel[6 + NUM_ACTIONS:6 + nj]) * 0.5
        )
        data.ctrl[NUM_ACTIONS:nj] = arm_tau


def _rgb_to_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Converts (H,W,3) uint8 RGB to a (1,3,128,128) float32 [0,1] tensor."""
    img = rgb.astype(np.float32) / 255.0
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        import cv2
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return img_t  # (1, 3, 128, 128)


def _label_active_cam(rgb: np.ndarray, active_cam: str, dist: float,
                      resize_to: tuple[int, int] | None = None) -> np.ndarray:
    """CAM-2 (Phase 1) demo visualization overlay.

    Overlays which grounding camera is currently active + the EMA'd distance
    onto a video frame, so the GROUNDING<->PROXIMITY handoff
    (docs/cam_opt2_multicam.md Schmitt trigger) is visible in rendered clips.
    Video-only — never called on the numeric eval path (render_video=False there).

    Args:
        rgb: (H,W,3) uint8 RGB frame to label.
        active_cam: Name of the currently active camera (e.g. 'GROUNDING',
            'PROXIMITY').
        dist: EMA'd distance to target (m), shown in the overlay text.
        resize_to: Optional (W,H) to resize BEFORE labeling (so the label stays
            a fixed font size regardless of which camera's native resolution fed
            it — the grounding cam is 480x360, the proximity cam 320x240).

    Returns:
        Labeled (H,W,3) uint8 RGB frame (resized if `resize_to` given).
    """
    import cv2
    out = rgb
    if resize_to is not None and (rgb.shape[1], rgb.shape[0]) != resize_to:
        out = cv2.resize(rgb, resize_to, interpolation=cv2.INTER_LINEAR)
    out = out.copy()
    color = (255, 210, 60) if active_cam == 'PROXIMITY' else (60, 210, 255)
    label = f"CAM: {active_cam}  d={dist:.2f}m"
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color, 1, cv2.LINE_AA)
    return out
