"""
inferencer.py — Closed-loop deploy rollout harness for GroundedNav.

ADR-001 / docs/architecture_decision.md: at deploy, the STUDENT outputs 15 joint
targets → PD → physics.  NO WBC teacher in the deploy loop.

Physics approach:
  - Load the G1 robot via WBCTeacher (it owns the MjModel/MjData and knows the
    exact XML + actuator setup).
  - Inject the arena objects into the same model using build_arena().
  - Use teacher.step() ONLY for settling (warmup, not logged), then switch to student.
  - After settle, student outputs raw_action → target_dof = raw_action*0.25+default_angles
    → PD → physics (teacher.step() is NOT called during the student rollout).

Three-rate design (per ADR-001):
  - Language: cached once per episode.
  - Grounding (Arch A): classical HSV+depth, runs every GROUNDING_PERIOD steps (~5 Hz).
  - Action head: 50 Hz (every control step).

Action chunking: if chunk_H>1, temporal ensembling (ACT-style).
MAXSTEPS hard cap: easy=600, demo=1400.

Goal source (Arch A only) — controls how the goal (dist, cosθ, sinθ) is sourced:
  - 'learned'   : grounding head's own predicted goal from vision+language (default deploy)
  - 'classical'  : classical HSV+depth grounding, replaces grounding head output (deployable)
  - 'gt'         : privileged GT goal computed from simulation state, bypasses grounding head
                   (upper-bound probe: answers "does goal→action navigation work at all?")

For 'gt' and 'learned' sources, ego rendering is skipped (zero ego_rgb fed to model)
to avoid render overhead and eliminate the untrained vision backbone as a confounder.
For 'classical', rendering runs at GROUNDING_PERIOD cadence (5 Hz).

Usage:
    from code.inferencer import Inferencer, RolloutResult
    inf = Inferencer(checkpoint_path=None, arch='A', device='cpu', goal_source='gt')
    result = inf.rollout(scene_cfg, instruction, lang_emb=None, maxsteps=600,
                         render_video=True, video_path='eval/ep0.mp4')
"""

from __future__ import annotations

import collections
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import mujoco

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from code.small_vla import GroundedNav, DEFAULTS
from code.arena import (build_arena, ArenaRenderer, EGO_W, EGO_H, EGO_FOVY, get_ego_intrinsics,
                        GROUNDING_W, GROUNDING_H, CAMERA_MODE)
from code.scene import DIFFICULTY_PRESETS
from code.grounding import ground as classical_ground, _parse_instruction, get_ego_intrinsics_rendered
from code.teacher import (WBCTeacher, _yaw_of, DEFAULT_ANGLES, KPS, KDS,
                           NUM_ACTIONS, SIM_DT, CONTROL_DECIMATION, RESET_HEIGHT)
from code.steer import steer as _steer_cmd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FALL_HEIGHT        = 0.50     # pelvis z below this → fall (matches gen_dataset.py)
GROUNDING_PERIOD   = 10       # run grounding every N steps → 50Hz/10 = 5 Hz
KEYFRAME_PATH      = str(_REPO / "checkpoint" / "stand_keyframe.npz")
PROPRIO_K          = 6        # proprio history length
PROPRIO_DIM        = 55
PROPRIO_DIM_PHASE  = 57       # proprio_dim when gait phase [sin,cos] appended (Fix 4)
IMG_SIZE           = 128
HOLD_STEPS_REQUIRED = 5       # consecutive in-range steps to succeed
ACTION_SCALE       = 0.25     # from teacher.py: target_dof = action * 0.25 + default_angles
SETTLE_STEPS       = 80       # warm-up steps using teacher (zero cmd)

# Default joint angles (imported from teacher.py, also available in action_stats from ckpt)
_DEFAULT_ANGLES_NP = DEFAULT_ANGLES.copy()  # shape (15,)

# Gait phase constants (Fix 4)
_LEFT_ANKLE_PITCH_IDX = 4     # in lower-body joint positions (qpos[7:22])
_LEFT_ANKLE_DEFAULT   = -0.2  # default angle (rad)


# ---------------------------------------------------------------------------
# Gait phase tracker (Fix 4)
# ---------------------------------------------------------------------------

class _GaitPhaseTracker:
    """
    Tracks gait phase phi in [0, 2pi] from left ankle pitch zero-crossings.
    Returns (sin_phi, cos_phi) as a 2-d gait-phase encoding.
    Same implementation as gen_dart_dataset.py.
    """

    def __init__(self, freq_hz: float = 1.8):
        self._phi: float = 0.0
        self._prev_q: float = 0.0
        self._initialized: bool = False
        self._freq_hz = freq_hz
        self._dt = SIM_DT * CONTROL_DECIMATION   # 0.02 s

    def update(self, q_lb: np.ndarray) -> np.ndarray:
        """q_lb: (15,) lower-body joint positions. Returns np.float32[2]."""
        q_ankle = float(q_lb[_LEFT_ANKLE_PITCH_IDX]) - _LEFT_ANKLE_DEFAULT
        if not self._initialized:
            self._prev_q = q_ankle
            self._initialized = True
            return np.array([0.0, 1.0], dtype=np.float32)

        self._phi += 2.0 * math.pi * self._freq_hz * self._dt
        if self._prev_q < 0.0 and q_ankle >= 0.0:
            self._phi = 0.0
        self._prev_q = q_ankle
        self._phi = self._phi % (2.0 * math.pi)
        return np.array([math.sin(self._phi), math.cos(self._phi)], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_proprio(data: mujoco.MjData, prev_action: np.ndarray) -> np.ndarray:
    """
    55-d proprio vector (exact layout from dataset.md / gen_dataset.py).
      [0:15]  lower-body joint positions
      [15:30] lower-body joint velocities
      [30:34] base IMU quaternion [w,x,y,z]
      [34:37] base angular velocity (rad/s)
      [37:40] base linear velocity (world frame)
      [40:55] prev_action (15 joint targets)
    """
    p = np.empty(PROPRIO_DIM, dtype=np.float32)
    p[0:15]  = data.qpos[7:22]
    p[15:30] = data.qvel[6:21]
    p[30:34] = data.qpos[3:7]   # [w,x,y,z]
    p[34:37] = data.qvel[3:6]
    p[37:40] = data.qvel[0:3]
    p[40:55] = prev_action
    return p


def _apply_student_pd(data: mujoco.MjData, target_dof: np.ndarray, nj: int):
    """Apply PD torques from student joint targets (mirrored exactly from teacher.py)."""
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
    """Convert (H,W,3) uint8 RGB → (1,3,128,128) float32 [0,1] tensor."""
    img = rgb.astype(np.float32) / 255.0
    if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
        import cv2
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return img_t  # (1, 3, 128, 128)


def _label_active_cam(rgb: np.ndarray, active_cam: str, dist: float,
                      resize_to: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    CAM-2 (Phase 1) demo visualization: overlay which grounding camera is currently
    active + the EMA'd distance onto a video frame, so the GROUNDING<->PROXIMITY
    handoff (docs/cam_opt2_multicam.md Schmitt trigger) is visible in rendered clips.
    Video-only — never called on the numeric eval path (render_video=False there).

    resize_to : optional (W,H) to resize BEFORE labeling (so the label stays a fixed
                font size regardless of which camera's native resolution fed it —
                the grounding cam is 480x360, the proximity cam 320x240).
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


# ---------------------------------------------------------------------------
# Rollout result
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    success:       bool
    failure_tag:   str    # 'success'|'fall'|'didnt-reach'|'lost-target'|'wrong-object'
    steps:         int
    final_dist:    float
    fell:          bool
    upright:       bool
    ms_per_step:   float
    grounding_hz:  float
    goal_source:   str = 'learned'   # 'learned'|'classical'|'gt'
    vel_source:    str = 'predicted'  # 'predicted'|'gt' — Fix 2 flag
    residual_action: bool = False     # True if checkpoint uses residual+standardized Fix 1
    action_osc_std: float = 0.0      # per-step std of commanded joint motion (gait oscillation)
    forward_disp:  float = 0.0       # forward displacement from start (m)
    scene_cfg:     dict = field(default_factory=dict)
    video_path:    Optional[str] = None


# ---------------------------------------------------------------------------
# Inferencer
# ---------------------------------------------------------------------------

def _compute_gt_goal(data_mj: mujoco.MjData, target_xy: np.ndarray) -> np.ndarray:
    """
    Compute privileged GT goal (dist, cosθ, sinθ) from simulation state.

    The goal is egocentric: direction from robot to target in the robot's
    horizontal body frame (yaw-aligned).

    Returns np.float32[3]: (dist, cos(yaw_err), sin(yaw_err))
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


class Inferencer:
    """
    Closed-loop rollout harness for GroundedNav student.

    Parameters
    ----------
    checkpoint_path : path to a GroundedNav .pt (None = random-init for harness test)
    arch            : 'A' or 'C'
    device          : 'cpu' | 'cuda'
    chunk_H         : action chunking horizon (1 = no chunking)
    goal_source     : 'learned' | 'classical' | 'gt' (Arch A only)
                      'learned'  — model's own grounding head (default)
                      'classical' — HSV+depth classical grounding
                      'gt'       — privileged goal from sim state (upper bound)
    verbose         : per-step print
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        arch:        str  = 'A',
        device:      str  = 'cpu',
        chunk_H:     int  = 1,
        goal_source: str  = 'classical',   # 'learned' | 'classical' | 'gt'
        vel_source:  str  = 'predicted',   # 'predicted' | 'gt' (Fix 2 upper bound)
        verbose:     bool = False,
        use_keyframe: bool = True,          # True → load stand_keyframe.npz (WBC-free settle)
    ):
        self.device      = torch.device(device)
        self.verbose     = verbose
        self.arch        = arch
        self.chunk_H     = chunk_H
        self.goal_source = goal_source if arch == 'A' else 'learned'  # C has no grounding
        if self.goal_source not in ('learned', 'classical', 'gt'):
            raise ValueError(f"goal_source must be 'learned', 'classical', or 'gt'; got {goal_source!r}")
        if vel_source not in ('predicted', 'gt'):
            raise ValueError(f"vel_source must be 'predicted' or 'gt'; got {vel_source!r}")
        self.vel_source = vel_source if arch == 'A' else 'predicted'  # C has no vel head

        # ---- Keyframe settle (WBC-free init) ----
        # When use_keyframe=True and checkpoint/stand_keyframe.npz exists, skip the 80-step
        # WBC ONNX settle at episode init and instead restore physics from the saved standing
        # keyframe. The keyframe was generated offline by running WBC settle once.
        # Legality: WBC used only offline to make the keyframe (like a physics config step),
        # not called during any episode rollout.
        self._keyframe: Optional[dict] = None
        if use_keyframe and os.path.isfile(KEYFRAME_PATH):
            _kf = np.load(KEYFRAME_PATH)
            self._keyframe = {
                'qpos_local':  _kf['qpos_local'].copy(),    # (nq,) robot-local frame, xy=0
                'qvel_local':  _kf['qvel_local'].copy(),    # (nv,) near-zero at settle end
                'target_dof':  _kf['target_dof'].copy(),    # (15,) last WBC joint targets
                'height':      float(_kf['height']),
            }
            print(f"[inferencer] Keyframe init: loaded {KEYFRAME_PATH} "
                  f"(height={self._keyframe['height']:.4f}m) — WBC-free settle active")
        elif use_keyframe:
            print(f"[inferencer] Keyframe init: {KEYFRAME_PATH} not found, "
                  f"falling back to WBC settle")

        # ---- Fix 1: action stats (residual + standardized) ----
        # Loaded from checkpoint if present (train_gaitfix.py embeds them).
        # If not present, falls back to raw absolute action (old behaviour).
        self._action_stats: Optional[dict] = None   # {mean, std, default_angles} as np arrays

        # ---- Fix 4: gait phase input ----
        # If checkpoint was trained with proprio_dim=57 (dart_phase flag), the inferencer
        # must append [sin(phi), cos(phi)] to proprio each step.
        self._use_phase: bool = False

        # ---- V6: vel_proprio flag ----
        # If checkpoint was trained with vel_proprio=True, the velocity head also takes
        # proprio_emb + phase as inputs. Detected from ckpt['vel_proprio'].
        self._vel_proprio: bool = False

        # ---- Grounding head trained flag ----
        # Set to True when loading a grounding checkpoint (grounding_trained=True in ckpt).
        # Enables learned grounding with actual vision rendering instead of zero RGB.
        self._grounding_trained: bool = False

        # ---- Load / random-init model ----
        model_state = None
        cfg = {}
        self._checkpoint_loaded = False

        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict):
                if 'arch'    in ckpt: arch    = ckpt['arch']
                if 'chunk_H' in ckpt: chunk_H = ckpt['chunk_H']
                cfg = ckpt.get('cfg', {})

                # Fix 1: load action stats if embedded (train_gaitfix.py checkpoints)
                if 'action_stats' in ckpt:
                    _as = ckpt['action_stats']
                    self._action_stats = {
                        'mean':           np.array(_as['mean'],           dtype=np.float32),
                        'std':            np.array(_as['std'],            dtype=np.float32),
                        'default_angles': np.array(_as['default_angles'], dtype=np.float32),
                    }
                    print(f"[inferencer] Fix-1 residual action mode: loaded action_stats "
                          f"(n_frames={_as.get('n_frames', '?')})")

                # Fix 4: detect gait-phase checkpoints (proprio_dim=57)
                ckpt_proprio_dim = ckpt.get('proprio_dim', PROPRIO_DIM)
                if ckpt_proprio_dim == PROPRIO_DIM_PHASE or ckpt.get('dart_phase', False):
                    self._use_phase = True
                    print(f"[inferencer] Fix-4 gait-phase mode active: proprio_dim=57")

                # Grounding-trained flag: set if checkpoint was saved with grounding_trained=True
                if ckpt.get('grounding_trained', False):
                    self._grounding_trained = True
                    print(f"[inferencer] Grounding head trained: vision rendering enabled for learned grounding")

                # V6: vel_proprio flag
                if ckpt.get('vel_proprio', False):
                    self._vel_proprio = True
                    print(f"[inferencer] V6 vel_proprio mode active: vel head takes proprio_emb+phase")

                # Try common key names for state dict
                for key in ('state_dict', 'model_state', 'model'):
                    if key in ckpt and isinstance(ckpt[key], dict):
                        # Verify it looks like a GroundedNav state dict
                        first_keys = list(ckpt[key].keys())[:3]
                        if any(k.startswith(('vision.', 'lang_proj.', 'proprio_enc.',
                                             'action_head.', 'grounding.')) for k in first_keys):
                            model_state = ckpt[key]
                            break
                if model_state is None:
                    # Maybe the checkpoint itself is a state dict
                    first_keys = list(ckpt.keys())[:3]
                    if any(k.startswith(('vision.', 'lang_proj.', 'proprio_enc.',
                                         'action_head.', 'grounding.')) for k in first_keys):
                        model_state = ckpt
            if model_state is not None:
                self._checkpoint_loaded = True
                print(f"[inferencer] Loaded checkpoint: {checkpoint_path}  arch={arch}  chunk_H={chunk_H}")
            else:
                print(f"[inferencer] WARN: unrecognized ckpt format in {checkpoint_path}; using random-init")
        elif checkpoint_path is not None:
            print(f"[inferencer] WARN: checkpoint not found: {checkpoint_path}; using random-init")

        # Build GroundedNav
        self.arch    = arch
        self.chunk_H = chunk_H
        # Use correct proprio_dim (57 for phase-conditioned checkpoints, 55 otherwise)
        _ckpt_proprio_dim = (PROPRIO_DIM_PHASE if self._use_phase else PROPRIO_DIM)
        model_cfg = {**DEFAULTS, **cfg, 'chunk_H': chunk_H,
                     'proprio_dim': _ckpt_proprio_dim,
                     'vel_proprio': self._vel_proprio}   # V6

        # teacher_forcing=True when we will inject an external goal OR an external velocity.
        # When True, forward() uses gt_goal (if not None) and gt_vel (if not None) in place
        # of the predicted values from the grounding/velocity heads.
        # For 'learned' goal and 'predicted' vel, keep teacher_forcing=False so both heads
        # run freely.
        _inject_goal = (arch == 'A' and
                        (self.goal_source in ('gt', 'classical') or
                         (self.goal_source == 'learned' and self._grounding_trained)))
        _inject_vel  = (arch == 'A' and self.vel_source == 'gt')
        _need_teacher_forcing = _inject_goal or _inject_vel
        self.model = GroundedNav(
            arch=arch,
            teacher_forcing=_need_teacher_forcing,   # True → gt injection active in forward()
            **{k: v for k, v in model_cfg.items() if k in DEFAULTS},
        ).to(self.device)
        self.model.eval()

        if model_state is not None:
            miss, unexp = self.model.load_state_dict(model_state, strict=False)
            if miss:  print(f"[inferencer]   {len(miss)} missing keys")
            if unexp: print(f"[inferencer]   {len(unexp)} unexpected keys")
        else:
            print(f"[inferencer] Random-init GroundedNav arch={arch} chunk_H={chunk_H}")

    # ------------------------------------------------------------------
    def rollout(
        self,
        scene_cfg:    dict,
        instruction:  str,
        lang_emb:     Optional[np.ndarray] = None,
        maxsteps:     int   = 600,
        render_video: bool  = False,
        video_path:   Optional[str] = None,
        render_tp:    bool  = True,
        stop_r:       Optional[float] = None,
    ) -> RolloutResult:
        """
        Run one closed-loop episode.

        The WBC teacher is used ONLY for the settle phase (SETTLE_STEPS with zero
        velocity command) to bring the G1 to a stable standing pose.
        After settle, the STUDENT drives the robot: student output → PD → physics.
        """
        if stop_r is None:
            stop_r = float(scene_cfg.get('stop_r', 0.6))

        # Target object info
        objects      = scene_cfg['objects']
        target_idx   = scene_cfg['target_index']
        target_obj   = objects[target_idx]
        target_xy    = np.array([target_obj['x'], target_obj['y']], dtype=np.float64)
        target_color = target_obj['color_name']
        target_shape = target_obj['shape_name']

        # Language embedding
        # For learned grounding (grounding_trained=True), we build a one-hot color+shape
        # encoding in the first (N_COLORS + N_SHAPES) dims of the 2048-d lang emb.
        # This matches the encoding used during grounding training (train_grounding.py).
        if lang_emb is None:
            if self._grounding_trained and self.goal_source == 'learned':
                # Build color+shape one-hot embedding for the grounding head
                _COLORS_ORDERED = ["red","yellow","blue","green","orange","purple","cyan"]
                _SHAPES_ORDERED = ["ball","cube","cylinder","cone"]
                lang_emb = np.zeros(2048, dtype=np.float32)
                c_idx = _COLORS_ORDERED.index(target_color) if target_color in _COLORS_ORDERED else 0
                s_idx = _SHAPES_ORDERED.index(target_shape) if target_shape in _SHAPES_ORDERED else 0
                lang_emb[c_idx] = 1.0
                lang_emb[len(_COLORS_ORDERED) + s_idx] = 1.0
            else:
                lang_emb = np.zeros(2048, dtype=np.float32)
        lang_t = torch.from_numpy(lang_emb.astype(np.float32)).unsqueeze(0).to(self.device)

        # ---- Build arena (adds objects to G1 XML) ----
        arena_model = build_arena(scene_cfg)
        arena_model.opt.timestep = SIM_DT

        # ---- Inject arena model into teacher (same pattern as gen_dataset.py) ----
        teacher = WBCTeacher(use_gpu=False)   # CPU is fine (0.32ms/step)
        teacher.model = arena_model
        teacher.data  = mujoco.MjData(arena_model)
        teacher._nj   = arena_model.nq - 7
        teacher._pelvis_id = mujoco.mj_name2id(
            arena_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

        # Reset to scene start pose
        rx, ry    = scene_cfg['robot_xy']
        robot_yaw = float(scene_cfg.get('robot_yaw', 0.0))
        teacher.reset(pos_xy=(rx, ry), yaw=robot_yaw)

        data_mj = teacher.data
        model_mj = teacher.model
        nj = teacher._nj

        # ---- Renderer ----
        # V2: ArenaRenderer now includes a dedicated grounding renderer at 480x360.
        # The grounding renderer uses the same EGL context (single renderer object)
        # to prevent context exhaustion (V1's silent-failure bug).
        renderer = ArenaRenderer(model_mj)
        tp_cam   = renderer.make_tp_cam()
        # CAM-2 (Phase 1): intrinsics now come dynamically from whichever camera the
        # Schmitt-trigger handoff selects each cycle (`intr_active`, set in the main
        # loop below) — render_grounding()/render_proximity() each return their own
        # correct (dims, pitch_deg, is_proximity) intrinsics dict.

        frames_ego: list = []
        frames_tp:  list = []

        # ---- Settle: either WBC ONNX (baseline) or keyframe restore (WBC-free) ----
        if self._keyframe is not None:
            # Keyframe path: restore saved physics state — no WBC ONNX called at runtime.
            # The keyframe was generated offline by running WBC settle once.
            kf = self._keyframe
            kf_qpos = kf['qpos_local'].copy()
            # Apply scene xy/yaw: overwrite x, y, and orientation quaternion
            kf_qpos[0] = rx
            kf_qpos[1] = ry
            # Overwrite orientation for scene yaw (keyframe was settled at yaw=0)
            kf_qpos[3] = math.cos(robot_yaw / 2)   # w
            kf_qpos[4] = 0.0                         # x
            kf_qpos[5] = 0.0                         # y
            kf_qpos[6] = math.sin(robot_yaw / 2)    # z
            data_mj.qpos[:len(kf_qpos)] = kf_qpos
            data_mj.qvel[:len(kf['qvel_local'])] = kf['qvel_local']
            mujoco.mj_forward(model_mj, data_mj)
            teacher._target_dof = kf['target_dof'].copy()
            # Sanity check
            if teacher.base_height < FALL_HEIGHT:
                renderer.close()
                return RolloutResult(
                    success=False, failure_tag='fall', steps=0,
                    final_dist=float(np.linalg.norm(data_mj.qpos[0:2] - target_xy)),
                    fell=True, upright=False,
                    ms_per_step=0.0, grounding_hz=0.0,
                    scene_cfg=scene_cfg,
                )
        else:
            # WBC ONNX settle (original baseline path)
            for _ in range(SETTLE_STEPS):
                teacher.step(vel_cmd=(0.0, 0.0, 0.0))
                if teacher.base_height < FALL_HEIGHT:
                    # fell during settle — very unusual, but cap it
                    renderer.close()
                    return RolloutResult(
                        success=False, failure_tag='fall', steps=0,
                        final_dist=float(np.linalg.norm(data_mj.qpos[0:2] - target_xy)),
                        fell=True, upright=False,
                        ms_per_step=0.0, grounding_hz=0.0,
                        scene_cfg=scene_cfg,
                    )

        # After settle, student takes over — PD applied directly to ctrl.
        # (WBC ONNX is NOT called during the student rollout in either path.)

        # ---- Fix 4: gait phase tracker (if phase-conditioned checkpoint) ----
        _use_phase = self._use_phase
        _phase_tracker = _GaitPhaseTracker() if _use_phase else None
        if _use_phase:
            print(f"[inferencer] Fix-4 gait-phase tracking active: proprio_dim={PROPRIO_DIM_PHASE}")

        # ---- Fix 1: prepare de-normalization arrays if action_stats present ----
        _use_residual = (self._action_stats is not None)
        if _use_residual:
            _as       = self._action_stats
            _da_mean  = _as['mean']           # (15,) delta mean
            _da_std   = _as['std']            # (15,) delta std
            _da_deflt = _as['default_angles'] # (15,) default angles
            print(f"[inferencer] Fix-1 residual mode active: denorm = default + pred*std + mean")

        # ---- Fix 2: prepare GT velocity injection ----
        _inject_gt_vel = (self.arch == 'A' and self.vel_source == 'gt')
        if _inject_gt_vel:
            print(f"[inferencer] Fix-2 GT velocity injection active (steer.py privileged cmd)")

        # ---- State for student loop ----
        prev_action  = teacher._target_dof.copy()  # last teacher target as initial
        _eff_proprio_dim = PROPRIO_DIM_PHASE if _use_phase else PROPRIO_DIM
        proprio_hist = collections.deque(
            [np.zeros(_eff_proprio_dim, dtype=np.float32)] * PROPRIO_K,
            maxlen=PROPRIO_K,
        )
        # Pre-fill proprio hist with settle-end state
        prop_now = _build_proprio(data_mj, prev_action)
        if _use_phase:
            q_lb_settle = data_mj.qpos[7:22].copy()
            ph_settle = _phase_tracker.update(q_lb_settle)
            prop_now = np.concatenate([prop_now, ph_settle])   # (57,)
        for _ in range(PROPRIO_K):
            proprio_hist.append(prop_now.copy())

        # Grounding cache (Arch A)
        # E6 fix: do NOT default to straight-ahead.  Instead, start in SCAN mode:
        # the robot turns in place (pure ωz) until the target is detected AND centered
        # (detected bearing < SCAN_ALIGNED_THR), or until SCAN_TIMEOUT steps elapsed.
        # This ensures the first committed goal is based on an unoccluded frontal detection.
        #
        # Design: scan RIGHT for first 60 steps, then LEFT for next 60 (120 total).
        # Acquired when:  target detected AND abs(bearing) < SCAN_ALIGNED_THR (target near-center).
        # Partial detections (target partially occluded, large bearing) keep scanning.
        cached_goal_vec    = np.array([2.0, 1.0, 0.0], dtype=np.float32)
        last_grounding_step = -999
        # Scan-and-acquire state
        _scan_active         = True      # True until target centered in frame
        _scan_yaw_delta      = 0.0       # cumulative yaw scanned (rad)
        # Scan budget: right 0→90° then left 90°→−90° then right again — full ±90° coverage.
        # 90°/0.6 rad/s / 0.02 dt = 75 steps per quarter. So 225 steps covers ±90°.
        # Keep it at 200 steps max (4s at 50Hz) to avoid wasting too much time.
        SCAN_TIMEOUT         = 200       # max steps in scan mode
        SCAN_RATE            = 0.6       # rad/s scan rate
        SCAN_DT              = SIM_DT * CONTROL_DECIMATION  # 0.02s per step
        # Exit scan when bearing < SCAN_ALIGNED_THR or first detection (whichever is looser).
        SCAN_ALIGNED_THR_DEG = 40.0     # target bearing < this → aligned, exit scan
        # Goal smoothing: exponential moving average of detected goals (E6)
        _goal_ema        = None      # set on first detection
        _GOAL_EMA_ALPHA  = 0.4       # blending factor: new=alpha*detected + (1-alpha)*ema
        # Last-known-good goal: hold when briefly lost (E6)
        _last_known_goal = None      # set on first detection; held through gaps
        _frames_since_detection = 0  # steps since last valid detection
        # V2: HOLD_GOAL_HORIZON extended to 100 steps (was 50).
        # Progressive re-detection: robot walks toward last-known goal for up to 100 steps,
        # then re-detects at closer range. At 4-9m, after walking 1-2m closer, the target
        # is 2-3m nearer → 40-80% larger in the image → much more reliable detection.
        # Key: 100 steps * 0.02s * 0.55m/s MAX_VX ≈ 1.1m forward progress during hold.
        HOLD_GOAL_HORIZON = 100      # V2: extended from 50 — progressive re-detection window

        # CAM-2 (Phase 1, docs/cam_opt2_multicam.md / docs/cam_p1.md): Schmitt-trigger
        # handoff between the GROUNDING camera (26° pitch, far/mid range) and the new
        # PROXIMITY camera (58° pitch, ~0.22-1.81m) on the EMA'd last-known distance.
        # Render ONLY the active camera each grounding cycle -> steady-state compute is
        # unchanged from pre-CAM-2 (still exactly one render per cycle in the common
        # case; the bounded fallback probe below adds a second render only on repeated
        # misses, a handful of times per episode at most).
        CAM_D_LO      = 1.2     # m — switch GROUNDING->PROXIMITY below this
        CAM_D_HI      = 1.6     # m — switch PROXIMITY->GROUNDING above this
        _active_cam   = 'GROUNDING'   # default at episode start (targets start 1.5-9m away)
        _cam_miss_count = 0            # consecutive misses on the active camera
        _video_frame_cache = None       # demo-viz: last labeled active-cam frame (video only)

        # Determine rendering and grounding behavior based on goal_source
        # 'gt':        zero ego_rgb, GT goal injected from sim state each step, no render
        # 'classical': render at GROUNDING_PERIOD cadence, classical HSV grounding
        # 'learned':   render at GROUNDING_PERIOD cadence, model grounding head predicts goal
        _need_classical_render = (self.arch == 'A' and self.goal_source == 'classical')
        _need_learned_render   = (self.arch == 'A' and self.goal_source == 'learned'
                                  and getattr(self, '_grounding_trained', False))
        _use_gt_goal = (self.arch == 'A' and self.goal_source == 'gt')
        _use_learned_goal = (self.arch != 'A') or (self.goal_source == 'learned')

        # Temporal ensembling buffer
        te_buffer: list = []   # list of (step_issued, weights(H,), actions(H,15))

        # Current student joint targets (start from teacher's settle targets)
        student_target_dof = teacher._target_dof.copy()

        # ---- Oscillation tracking ----
        _all_target_dofs: list = []      # collect commanded joint targets for osc check
        _start_xy = data_mj.qpos[0:2].copy()  # initial XY for forward displacement

        # ---- Main rollout loop ----
        step_times   = []
        hold_counter = 0
        fell         = False
        steps_done   = 0

        for step in range(maxsteps):
            t0 = time.perf_counter()

            # Height check
            height = float(data_mj.qpos[2])
            if height < FALL_HEIGHT:
                fell = True
                break

            # Current yaw
            yaw = _yaw_of(data_mj.qpos[3:7])

            # Rendering: needed for classical/learned grounding or video recording
            need_classical_grounding = (_need_classical_render and
                                        (step - last_grounding_step) >= GROUNDING_PERIOD)
            need_learned_grounding   = (_need_learned_render and
                                        (step - last_grounding_step) >= GROUNDING_PERIOD)
            need_render = render_video or need_classical_grounding or need_learned_grounding

            intr_active = None   # intrinsics of whichever camera was actually rendered below
            if need_render:
                if need_classical_grounding:
                    if CAMERA_MODE == 'widefov':
                        # CAM-1 (Phase 2, toggle): single wide-FOV camera — no proximity
                        # cam, no Schmitt handoff, no bounded fallback probe. This branch
                        # only executes when CAMERA_MODE=='widefov'; the 'cam2' default
                        # falls straight to the untouched elif/else below.
                        rgb, depth, intr_active = renderer.render_widefov(
                            data_mj, yaw, render_depth=True)
                    # CAM-2 (Phase 1): render ONLY the currently-active camera (Schmitt
                    # trigger state) — steady-state cost is still exactly one render/cycle.
                    elif _active_cam == 'PROXIMITY':
                        rgb, depth, intr_active = renderer.render_proximity(
                            data_mj, yaw, render_depth=True)
                    else:
                        # V2: use high-resolution grounding renderer (480x360) for classical HSV.
                        # This makes distant targets (4-9m) 2.25x larger in pixel area, dramatically
                        # improving HSV blob detection at demo distances.
                        # The grounding renderer reuses the same EGL context (no context exhaustion).
                        rgb, depth, intr_active = renderer.render_grounding(
                            data_mj, yaw, render_depth=True)
                    # CAM-2 demo viz: for video recording, show the ACTUAL active camera's
                    # frame (resized to a fixed EGO_W x EGO_H so video frame size stays
                    # consistent whether the grounding cam (480x360) or proximity cam
                    # (320x240) is active) + a label, rather than a separate neutral ego
                    # render — this is what's honestly driving detection this cycle, and
                    # is what makes the handoff visible in rendered clips. Cached below so
                    # in-between (non-grounding-cycle) steps reuse it instead of an extra
                    # render call.
                    if render_video:
                        _cam_label = 'WIDEFOV' if CAMERA_MODE == 'widefov' else _active_cam
                        _video_frame_cache = _label_active_cam(
                            rgb, _cam_label, float(cached_goal_vec[0]),
                            resize_to=(EGO_W, EGO_H))
                        rgb_video = _video_frame_cache
                    else:
                        rgb_video = rgb   # unused (render_video=False)
                else:
                    if render_video and _need_classical_render and _video_frame_cache is not None:
                        # In-between step (no new grounding-cycle render this step, but
                        # video is being recorded): reuse the cached active-camera frame
                        # instead of an extra neutral ego render.
                        rgb, depth = None, None
                        rgb_video  = _video_frame_cache
                    else:
                        rgb, depth, _intr = renderer.render_ego(data_mj, yaw,
                                                                 render_depth=_need_learned_render)
                        rgb_video = rgb
            else:
                rgb, depth = None, None
                rgb_video  = None

            # Grounding (Arch A, goal_source='classical', at ~5 Hz)
            # E6 fix: scan-and-acquire + temporal smoothing + hold-last-known-goal
            if need_classical_grounding and rgb is not None and depth is not None:
                gr = classical_ground(rgb, depth, target_color, target_shape, intr_active)
                last_grounding_step = step

                # CAM-2 (Phase 1): bounded fallback probe (docs/cam_opt2_multicam.md
                # handoff rule) — after 2 consecutive misses on the active camera, try
                # the OTHER camera once this cycle and adopt its result if it detects.
                # This is a transient second render only on repeated misses, not a
                # steady-state cost.
                if gr.not_visible:
                    _cam_miss_count += 1
                    # CAM-1 (Phase 2, toggle): no probe/handoff in widefov mode — there is
                    # no second camera to fall back to. Gate is a no-op for cam2 (default).
                    if CAMERA_MODE != 'widefov' and _cam_miss_count >= 2:
                        other_cam = 'GROUNDING' if _active_cam == 'PROXIMITY' else 'PROXIMITY'
                        # PLAUSIBILITY GATE (docs/cam_p1.md): only probe the PROXIMITY
                        # camera when the last-known EMA distance says the target could
                        # actually be inside its ~0.22-1.81m band. Without this gate, a
                        # far-range miss streak (e.g. blue/cyan wall-HSV collisions at
                        # 5-9m) probes the proximity cam, which stares at the blue-ish
                        # checkered floor (H~105, inside blue/cyan HSV bounds) and can
                        # adopt a floor false-positive at a bogus close distance --
                        # flipping active_cam into a self-reinforcing PROXIMITY trap.
                        # This exact failure regressed demo ep13 (blue ball, 4.96m) in
                        # the first CAM-2 gate run. Probing GROUNDING from PROXIMITY is
                        # always safe (its 1.14-21m band covers everything far).
                        _probe_ok = (other_cam == 'GROUNDING' or
                                     (_last_known_goal is not None and
                                      float(_last_known_goal[0]) <= CAM_D_HI))
                        if _probe_ok:
                            if other_cam == 'PROXIMITY':
                                rgb2, depth2, intr2 = renderer.render_proximity(
                                    data_mj, yaw, render_depth=True)
                            else:
                                rgb2, depth2, intr2 = renderer.render_grounding(
                                    data_mj, yaw, render_depth=True)
                            gr2 = classical_ground(rgb2, depth2, target_color, target_shape, intr2)
                            if not gr2.not_visible:
                                gr = gr2
                                _active_cam = other_cam
                                _cam_miss_count = 0
                else:
                    _cam_miss_count = 0

                if not gr.not_visible:
                    _frames_since_detection = 0
                    # Temporal smoothing (EMA) on detected goal — smooths out bearing jitter
                    raw_goal = gr.goal_vec.copy()
                    if _goal_ema is None:
                        _goal_ema = raw_goal.copy()
                        _last_known_goal = raw_goal.copy()
                    else:
                        _goal_ema = _GOAL_EMA_ALPHA * raw_goal + (1.0 - _GOAL_EMA_ALPHA) * _goal_ema
                        # Re-normalize cos/sin
                        th = math.atan2(_goal_ema[2], _goal_ema[1])
                        _goal_ema[1] = math.cos(th)
                        _goal_ema[2] = math.sin(th)
                        _last_known_goal = _goal_ema.copy()
                    cached_goal_vec = _goal_ema.copy()
                    # CAM-2 (Phase 1): Schmitt-trigger handoff on the EMA'd distance —
                    # D_LO/D_HI straddle the 0.92-1.81m dual-visible band (docs/cam_p1.md),
                    # so this only flips once per approach/retreat, not every cycle.
                    # CAM-1 (Phase 2, toggle): no handoff in widefov mode (single camera,
                    # _active_cam stays at its unused initial value) — gate is a no-op
                    # for cam2 (default).
                    if CAMERA_MODE != 'widefov':
                        _ema_dist = float(_goal_ema[0])
                        if _active_cam == 'GROUNDING' and _ema_dist < CAM_D_LO:
                            _active_cam = 'PROXIMITY'
                        elif _active_cam == 'PROXIMITY' and _ema_dist > CAM_D_HI:
                            _active_cam = 'GROUNDING'
                    # Exit scan mode when target is aligned (bearing < threshold).
                    # Partial detections (bearing still large) keep scanning so the robot
                    # continues rotating to better center the target in the image frame.
                    if _scan_active:
                        det_bearing_deg = abs(math.degrees(math.atan2(_goal_ema[2], _goal_ema[1])))
                        if det_bearing_deg < SCAN_ALIGNED_THR_DEG:
                            _scan_active = False
                            if self.verbose:
                                print(f"  [scan] ALIGNED at step={step}  "
                                      f"yaw_err={math.degrees(math.atan2(_goal_ema[2],_goal_ema[1])):+.1f}°",
                                      flush=True)
                        elif self.verbose:
                            print(f"  [scan] Partial det step={step}  "
                                  f"bearing={math.degrees(math.atan2(_goal_ema[2],_goal_ema[1])):+.1f}° "
                                  f"(still scanning, thr={SCAN_ALIGNED_THR_DEG}°)",
                                  flush=True)
                else:
                    _frames_since_detection += 1
                    # Hold last-known goal if recently seen, else keep cached (straight-ahead initially)
                    if _last_known_goal is not None and _frames_since_detection <= HOLD_GOAL_HORIZON:
                        cached_goal_vec = _last_known_goal.copy()
                    # else: keep whatever cached_goal_vec was (straight-ahead default or last ema)

            # Learned grounding (Arch A, goal_source='learned', trained grounding head)
            # Runs at GROUNDING_PERIOD cadence with EMA smoothing.
            if need_learned_grounding and rgb is not None:
                last_grounding_step = step
                img_t_gr = _rgb_to_tensor(rgb, self.device)
                with torch.no_grad():
                    out_gr = self.model(
                        ego_rgb   = img_t_gr,
                        lang_emb  = lang_t,
                        proprio_h = torch.zeros(1, 6, _eff_proprio_dim, device=self.device),
                        gt_goal   = None,
                        gt_vel    = None,
                    )
                raw_gr = out_gr['goal'].cpu().numpy().squeeze(0)   # (3,)
                # Normalize bearing component
                norm_gr = math.sqrt(raw_gr[1]**2 + raw_gr[2]**2 + 1e-6)
                raw_gr[1] /= norm_gr
                raw_gr[2] /= norm_gr
                if _goal_ema is None:
                    _goal_ema = raw_gr.copy()
                    _last_known_goal = raw_gr.copy()
                else:
                    _goal_ema = _GOAL_EMA_ALPHA * raw_gr + (1.0 - _GOAL_EMA_ALPHA) * _goal_ema
                    th = math.atan2(_goal_ema[2], _goal_ema[1])
                    _goal_ema[1] = math.cos(th)
                    _goal_ema[2] = math.sin(th)
                    _last_known_goal = _goal_ema.copy()
                cached_goal_vec = _goal_ema.copy()

            # GT goal (Arch A, goal_source='gt'): privileged sim-state goal, updated every step
            if _use_gt_goal:
                cached_goal_vec = _compute_gt_goal(data_mj, target_xy)

            # ---- H3: scan-and-acquire — STUDENT-DRIVEN (WBC-free) ----
            # If target not yet detected (scan_active=True, classical mode only),
            # inject a turning velocity command into the STUDENT action head so the
            # STUDENT produces the turning joint targets → PD → physics.
            # NO WBC teacher.step() called at runtime: 100% WBC-free deploy.
            # Pattern: right 75 steps (90°), left 150 steps (180°), right remainder —
            # sweeps ±90° arc covering targets at any initial bearing.
            # Timeout: after SCAN_TIMEOUT steps, exit scan and use default cached_goal_vec.
            if _scan_active and _need_classical_render:
                if step >= SCAN_TIMEOUT:
                    _scan_active = False
                    if self.verbose:
                        print(f"  [scan] TIMEOUT at step={step}, falling back to default goal", flush=True)
                else:
                    # Compute scan direction (same sweep pattern as before)
                    _quarter = int(SCAN_TIMEOUT * 75.0 / 200.0)
                    if step < _quarter:
                        scan_wz = SCAN_RATE          # right: 0→90°
                    elif step < _quarter * 3:
                        scan_wz = -SCAN_RATE         # left: 90°→−90°
                    else:
                        scan_wz = SCAN_RATE           # right: −90°→0°
                    _scan_yaw_delta += scan_wz * SCAN_DT
                    # --- STUDENT-driven scan: inject vel into student, not teacher ---
                    # Build proprio from current sim state
                    prop_now = _build_proprio(data_mj, prev_action)
                    if _use_phase:
                        q_lb_now = data_mj.qpos[7:22].copy()
                        ph_now   = _phase_tracker.update(q_lb_now)
                        prop_now = np.concatenate([prop_now, ph_now])
                    proprio_hist.append(prop_now)
                    prop_arr = np.stack(list(proprio_hist), axis=0)
                    prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(self.device)
                    # Zero image (no vision needed during scan)
                    img_t_scan = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE,
                                             dtype=torch.float32, device=self.device)
                    # Inject: goal = straight-ahead default (hold during scan)
                    scan_goal_t = torch.from_numpy(cached_goal_vec).unsqueeze(0).to(self.device)
                    # Inject: turning velocity command (vx=0, vy=0, wz=scan_wz)
                    scan_vel_cmd = np.array([0.0, 0.0, scan_wz], dtype=np.float32)
                    scan_vel_t   = torch.from_numpy(scan_vel_cmd).unsqueeze(0).to(self.device)
                    # Student forward pass with injected goal + vel
                    with torch.no_grad():
                        out_scan = self.model(
                            ego_rgb   = img_t_scan,
                            lang_emb  = lang_t,
                            proprio_h = prop_t,
                            gt_goal   = scan_goal_t,
                            gt_vel    = scan_vel_t,
                        )
                    # Extract action and convert to joint targets
                    actions_scan = out_scan['action'].cpu().numpy().squeeze(0)  # (H, 15)
                    raw_action_scan = actions_scan[0]   # take first action (no TE during scan)
                    if _use_residual:
                        scan_target_dof = _da_deflt + raw_action_scan * _da_std + _da_mean
                    else:
                        scan_target_dof = raw_action_scan
                    # Apply PD + physics (student drives physics — no teacher)
                    for _ in range(CONTROL_DECIMATION):
                        _apply_student_pd(data_mj, scan_target_dof, nj)
                        mujoco.mj_step(model_mj, data_mj)
                    prev_action = scan_target_dof.copy()
                    _all_target_dofs.append(prev_action.copy())
                    steps_done = step + 1
                    t1 = time.perf_counter()
                    step_times.append((t1 - t0) * 1000.0)
                    dist_to_target = float(np.linalg.norm(data_mj.qpos[0:2] - target_xy))
                    if render_video and rgb_video is not None:
                        # (rgb_video is already the labeled active-cam frame when
                        # _need_classical_render — see render-selection block above.)
                        frames_ego.append(rgb_video.copy())
                        if render_tp:
                            renderer.update_tp_cam(tp_cam, data_mj)
                            frames_tp.append(renderer.render_tp(data_mj, tp_cam).copy())
                    continue   # skip student forward pass (already done above)

            # ---- Student forward pass (normal mode) ----
            # Proprio (+ gait phase if Fix-4 active)
            prop_now = _build_proprio(data_mj, prev_action)
            if _use_phase:
                q_lb_now = data_mj.qpos[7:22].copy()
                ph_now   = _phase_tracker.update(q_lb_now)
                prop_now = np.concatenate([prop_now, ph_now])   # (57,)
            proprio_hist.append(prop_now)
            prop_arr = np.stack(list(proprio_hist), axis=0)     # (K, 55 or 57)
            prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(self.device)

            # Image tensor: zero for gt/learned (untrained vision backbone, skip render)
            if rgb is not None:
                img_t = _rgb_to_tensor(rgb, self.device)
            else:
                img_t = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32,
                                    device=self.device)

            # Build goal tensor for injection (Arch A)
            # For 'gt' and 'classical': inject cached_goal_vec, bypassing untrained grounding head.
            # For 'learned' + trained: inject EMA-smoothed cached_goal_vec (already computed above),
            #   bypassing per-step raw grounding head call for stability. The image was already
            #   processed at GROUNDING_PERIOD cadence in the learned-grounding block above.
            # For 'learned' + untrained (random-init): goal_inject_t=None, model runs its own head.
            _inject_cached = (self.arch == 'A' and
                              (not _use_learned_goal or
                               (self.goal_source == 'learned' and _need_learned_render)))
            if _inject_cached:
                goal_inject_t = torch.from_numpy(cached_goal_vec).unsqueeze(0).to(self.device)
            else:
                goal_inject_t = None

            # For 'learned' + trained, pass zero image to main forward (grounding already done).
            if _need_learned_render and rgb is not None:
                img_t = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32,
                                    device=self.device)

            # Fix 2: GT velocity injection — compute privileged steering vel from sim state
            gt_vel_inject_t = None
            if _inject_gt_vel:
                robot_xy  = data_mj.qpos[0:2].copy()
                robot_yaw = _yaw_of(data_mj.qpos[3:7])
                gt_vel_cmd, _, _ = _steer_cmd(robot_xy, robot_yaw, target_xy, stop_r)
                gt_vel_inject_t  = torch.from_numpy(gt_vel_cmd).unsqueeze(0).to(self.device)

            # Learned-grounding velocity injection: derive vel from cached_goal_vec so
            # the action head receives a consistent (goal_emb, vel_emb) pair.
            # Without this, the velocity head runs on zero-image grounding (garbage) and
            # produces incoherent velocity embeddings that corrupt the action head output.
            if _need_learned_render and _inject_cached and gt_vel_inject_t is None:
                d_gr  = float(cached_goal_vec[0])
                cs_gr = float(cached_goal_vec[1])
                sn_gr = float(cached_goal_vec[2])
                ye_gr = math.atan2(sn_gr, cs_gr)  # yaw_err from grounding
                if d_gr < stop_r:
                    vel_gr = np.zeros(3, dtype=np.float32)
                else:
                    from code.steer import MAX_VX, MAX_WZ, YAW_KP, FACE_THR_RAD, DECEL_DIST
                    wz_gr = float(np.clip(YAW_KP * ye_gr, -MAX_WZ, MAX_WZ))
                    if abs(ye_gr) > FACE_THR_RAD:
                        vx_gr = 0.0
                    else:
                        decel_gr = min(1.0, max(0.0, (d_gr - stop_r) / max(DECEL_DIST - stop_r, 0.1)))
                        vx_gr = float(np.clip(MAX_VX * max(0.0, cs_gr) * decel_gr, 0.0, MAX_VX))
                    vel_gr = np.array([vx_gr, 0.0, wz_gr], dtype=np.float32)
                gt_vel_inject_t = torch.from_numpy(vel_gr).unsqueeze(0).to(self.device)

            # Student forward pass
            with torch.no_grad():
                out = self.model(
                    ego_rgb   = img_t,
                    lang_emb  = lang_t,
                    proprio_h = prop_t,
                    gt_goal   = goal_inject_t,    # None → model predicts; tensor → injected
                    gt_vel    = gt_vel_inject_t,   # Fix 2: None → vel head predicts; tensor → injected
                )

            # Extract action chunk
            actions_raw = out['action'].cpu().numpy().squeeze(0)   # (H, 15)

            # Temporal ensembling (H > 1)
            if self.chunk_H > 1:
                H = self.chunk_H
                wt = np.exp(-0.1 * np.arange(H, dtype=np.float32))
                te_buffer.append((step, wt, actions_raw.copy()))
                te_buffer = [(s, w, a) for (s, w, a) in te_buffer if step - s < H]
                act_sum = np.zeros(15, dtype=np.float32)
                w_sum   = 0.0
                for (s, w, a) in te_buffer:
                    k = step - s
                    if 0 <= k < H:
                        act_sum += w[k] * a[k]
                        w_sum   += w[k]
                raw_action = (act_sum / w_sum) if w_sum > 1e-9 else actions_raw[0]
            else:
                raw_action = actions_raw[0]   # (15,)

            # Convert model output → absolute joint targets.
            #
            # Fix 1 (residual + standardized): if action_stats loaded from checkpoint,
            #   the model was trained to output standardized delta:
            #       normed_delta = (action - default - mean) / std
            #   De-normalize: target_dof = default + normed_delta * std + mean
            #
            # Fallback (old absolute mode, pre-gaitfix checkpoints):
            #   The model output IS the target_dof — no further scaling needed.
            if _use_residual:
                # De-normalize: default + pred*std + mean
                student_target_dof = _da_deflt + raw_action * _da_std + _da_mean
            else:
                student_target_dof = raw_action  # already absolute joint angles (old mode)

            # Track commanded targets for oscillation check
            _all_target_dofs.append(student_target_dof.copy())

            # Apply PD + physics substeps (student drives physics, no teacher here)
            for _ in range(CONTROL_DECIMATION):
                _apply_student_pd(data_mj, student_target_dof, nj)
                mujoco.mj_step(model_mj, data_mj)

            prev_action = student_target_dof.copy()
            steps_done  = step + 1

            # Record frames (always use ego-resolution rgb_video to keep size consistent)
            if render_video and rgb_video is not None:
                # (rgb_video is already the labeled active-cam frame when
                # _need_classical_render — see render-selection block above.)
                frames_ego.append(rgb_video.copy())
                if render_tp:
                    renderer.update_tp_cam(tp_cam, data_mj)
                    frames_tp.append(renderer.render_tp(data_mj, tp_cam).copy())

            t1 = time.perf_counter()
            step_times.append((t1 - t0) * 1000.0)

            # Distance to target
            dist_to_target = float(np.linalg.norm(data_mj.qpos[0:2] - target_xy))

            # Success check
            if dist_to_target < stop_r:
                hold_counter += 1
                if hold_counter >= HOLD_STEPS_REQUIRED:
                    break
            else:
                hold_counter = 0

            if self.verbose and step % 50 == 0:
                ms = (t1 - t0) * 1000.0
                print(f"  step={step:4d}  dist={dist_to_target:.2f}m  h={height:.3f}m  "
                      f"ms={ms:.1f}  hold={hold_counter}", flush=True)

        # ---- Done ----
        renderer.close()

        final_height = float(data_mj.qpos[2])
        upright      = final_height >= FALL_HEIGHT and not fell
        final_dist   = float(np.linalg.norm(data_mj.qpos[0:2] - target_xy))
        success      = (final_dist < stop_r) and upright

        if success:
            failure_tag = 'success'
        elif fell or not upright:
            failure_tag = 'fall'
        else:
            failure_tag = 'didnt-reach'

        ms_per_step  = float(np.mean(step_times)) if step_times else 0.0
        grounding_hz = (1.0 / (GROUNDING_PERIOD * SIM_DT * CONTROL_DECIMATION)
                        if self.arch == 'A' else 0.0)

        # ---- Gait oscillation check ----
        # Compute per-joint std of commanded targets over this rollout.
        # A near-static policy has std ≈ 0; a walking policy has notable std in
        # hip_pitch / knee joints.
        if _all_target_dofs:
            tdf_arr    = np.stack(_all_target_dofs, axis=0)   # (steps, 15)
            osc_std    = float(tdf_arr.std(axis=0).mean())    # mean per-joint std
        else:
            osc_std    = 0.0

        # Forward displacement from start
        fwd_xy      = data_mj.qpos[0:2].copy()
        forward_disp = float(np.linalg.norm(fwd_xy - _start_xy))

        # Write video
        if render_video and video_path and frames_ego:
            _write_video(frames_ego, frames_tp, video_path)

        return RolloutResult(
            success        = success,
            failure_tag    = failure_tag,
            steps          = steps_done,
            final_dist     = final_dist,
            fell           = fell,
            upright        = upright,
            ms_per_step    = ms_per_step,
            grounding_hz   = grounding_hz,
            goal_source    = self.goal_source,
            vel_source     = self.vel_source,
            residual_action = _use_residual,
            action_osc_std = osc_std,
            forward_disp   = forward_disp,
            scene_cfg      = scene_cfg,
            video_path     = video_path if (render_video and frames_ego) else None,
        )


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

def _write_video(frames_ego: list, frames_tp: list, out_path: str, fps: int = 50):
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if frames_tp and len(frames_tp) == len(frames_ego):
        import cv2
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
    print(f"[inferencer] Video written: {out_path} ({len(frames_out)} frames)", flush=True)


# ---------------------------------------------------------------------------
# Smoke test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("inferencer.py smoke test (random-init model, 30 steps)")
    print("=" * 60)

    from code.scene import sample_scene, derive_rng

    rng   = derive_rng(999, 0)
    scene = sample_scene(rng, difficulty='easy')
    tgt   = scene['objects'][scene['target_index']]
    print(f"Scene: {scene['instruction']}")
    print(f"Target: {tgt['color_name']} {tgt['shape_name']} at {tgt['dist_from_robot']:.2f}m")

    # Tiny maxsteps cap for smoke test
    SMOKE_MAXSTEPS = 30

    for arch in ('A', 'C'):
        print(f"\n--- Arch {arch} ---")
        inf = Inferencer(checkpoint_path=None, arch=arch, device='cpu', verbose=True)
        t0  = time.perf_counter()
        res = inf.rollout(scene_cfg=scene, instruction=scene['instruction'],
                          maxsteps=SMOKE_MAXSTEPS, render_video=False)
        dt  = time.perf_counter() - t0
        print(f"  steps={res.steps}  dist={res.final_dist:.2f}m  "
              f"fell={res.fell}  upright={res.upright}  "
              f"tag={res.failure_tag}  ms/step={res.ms_per_step:.1f}  wall={dt:.2f}s")
        assert res.steps > 0,     "No steps executed"
        assert not res.fell or res.steps >= 1, "Fell immediately during student phase"
        assert res.failure_tag in ('success', 'fall', 'didnt-reach',
                                   'lost-target', 'wrong-object')
        print(f"  Grounding Hz (arch A): {res.grounding_hz:.1f} Hz")

    print("\nSmoke PASS: student->PD->physics loop runs cleanly for both Arch A and C")
    sys.exit(0)
