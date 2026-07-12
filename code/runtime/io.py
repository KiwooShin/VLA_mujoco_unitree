"""
code.runtime.io — checkpoint/model loading + rollout outputs (RolloutResult,
video writer) for the closed-loop Inferencer.

RF-1 split of code/inferencer.py (docs/refactor_plan.md): the `Inferencer.
__init__` checkpoint-loading body, the `RolloutResult` dataclass, and the
`_write_video` helper, moved with control flow and numeric logic unchanged —
only *where* the code lives. `code/runtime/inferencer.py`'s constructor calls
`load_keyframe` / `build_model` below and assigns the results onto `self`,
in the same order the original monolithic `__init__` did (including the
`goal_source` vs. checkpoint-overridden-`arch` ordering subtlety noted in
`build_model`'s docstring).

`RolloutResult` and `_write_video` are kept import-visible at the old
`code.inferencer` path (`from code.inferencer import RolloutResult`:
code/eval/closedloop.py, code/deploy_eval.py, code/verify_egl_repro.py;
`from code.inferencer import _write_video`: code/eval/search_rollout.py,
code/apps/repl/maneuver_inferencer.py, code/render_showcase_videos.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from code.policy.small_vla import GroundedNav, DEFAULTS
from code.runtime.constants import PROPRIO_DIM, PROPRIO_DIM_PHASE, KEYFRAME_PATH


# ---------------------------------------------------------------------------
# Rollout result
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    """Outcome of one closed-loop `Inferencer.rollout()` episode.

    Attributes:
        success: True if the episode reached the target and stayed upright.
        failure_tag: One of 'success'|'fall'|'didnt-reach'|'lost-target'|'wrong-object'.
        steps: Number of student control steps executed.
        final_dist: Final distance to target (m).
        fell: True if the robot fell during the rollout.
        upright: True if the robot ended the episode upright.
        ms_per_step: Mean wall-clock time per control step (ms).
        grounding_hz: Effective grounding update rate (Hz); 0.0 for arch 'C'.
        goal_source: 'learned'|'classical'|'gt'.
        vel_source: 'predicted'|'gt' — Fix 2 flag.
        residual_action: True if checkpoint uses residual+standardized Fix 1.
        action_osc_std: Per-step std of commanded joint motion (gait oscillation).
        forward_disp: Forward displacement from start (m).
        scene_cfg: Scene configuration this episode was rolled out on.
        video_path: Path the rendered video was written to, if any.
        stall_break_triggers: NX-8 STALL_BREAK trigger count this episode (0 when off).
        avoid_bias_active_frac: NX-9 fraction of grounding cycles with |bias|>0 (0 when off).
    """
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
    video_path:    str | None = None
    stall_break_triggers: int = 0    # NX-8: STALL_BREAK trigger count this episode (0 when off)
    avoid_bias_active_frac: float = 0.0  # NX-9: fraction of grounding cycles with |bias|>0 (0 when off)


# ---------------------------------------------------------------------------
# Keyframe (WBC-free settle) loading
# ---------------------------------------------------------------------------

def load_keyframe(use_keyframe: bool) -> dict | None:
    """Loads the offline WBC-settle keyframe, if requested and present.

    When `use_keyframe` is True and checkpoint/stand_keyframe.npz exists, skip
    the 80-step WBC ONNX settle at episode init and instead restore physics
    from the saved standing keyframe. The keyframe was generated offline by
    running WBC settle once.

    Legality: WBC used only offline to make the keyframe (like a physics
    config step), not called during any episode rollout.

    Args:
        use_keyframe: True → attempt to load `KEYFRAME_PATH`.

    Returns:
        dict with 'qpos_local', 'qvel_local', 'target_dof', 'height', or None
        if `use_keyframe` is False or the file isn't present.
    """
    if use_keyframe and os.path.isfile(KEYFRAME_PATH):
        _kf = np.load(KEYFRAME_PATH)
        keyframe = {
            'qpos_local':  _kf['qpos_local'].copy(),    # (nq,) robot-local frame, xy=0
            'qvel_local':  _kf['qvel_local'].copy(),    # (nv,) near-zero at settle end
            'target_dof':  _kf['target_dof'].copy(),    # (15,) last WBC joint targets
            'height':      float(_kf['height']),
        }
        print(f"[inferencer] Keyframe init: loaded {KEYFRAME_PATH} "
              f"(height={keyframe['height']:.4f}m) — WBC-free settle active")
        return keyframe
    elif use_keyframe:
        print(f"[inferencer] Keyframe init: {KEYFRAME_PATH} not found, "
              f"falling back to WBC settle")
    return None


# ---------------------------------------------------------------------------
# Checkpoint parsing + model construction
# ---------------------------------------------------------------------------

@dataclass
class _LoadResult:
    """Bundle of everything `Inferencer.__init__` assigns onto `self` after
    parsing the checkpoint and constructing the model."""
    model: object
    checkpoint_loaded: bool
    arch: str
    chunk_H: int
    action_stats: dict | None
    use_phase: bool
    grounding_trained: bool
    vel_proprio: bool


def build_model(
    checkpoint_path: str | None,
    arch: str,
    chunk_H: int,
    device: torch.device,
    goal_source: str,
    vel_source: str,
) -> _LoadResult:
    """Parses the checkpoint (if any) and constructs the GroundedNav model.

    Mirrors the original (pre-RF-1) monolithic `Inferencer.__init__` body
    verbatim, including one ordering subtlety: `goal_source`/`vel_source`
    here must be the values already resolved by the caller from the
    *original* (pre-checkpoint-override) `arch` argument — exactly as the
    original `__init__` computed `self.goal_source`/`self.vel_source` at the
    very top of the function, BEFORE `arch` (a plain local variable, not
    `self.arch`) gets reassigned below from `ckpt['arch']` when present. The
    `_inject_goal`/`_inject_vel`/`teacher_forcing` computation further down
    then uses the POST-override `arch`, matching the original's own
    (intentional or not) inconsistency exactly.

    Args:
        checkpoint_path: Path to a GroundedNav .pt (None = random-init).
        arch: 'A' or 'C' (may be overridden by `ckpt['arch']` if present).
        chunk_H: Action-chunking horizon (may be overridden by ckpt).
        device: Torch device to build the model on.
        goal_source: Inferencer's already-resolved `self.goal_source`.
        vel_source: Inferencer's already-resolved `self.vel_source`.

    Returns:
        `_LoadResult` with the constructed (and possibly checkpoint-loaded)
        model plus every derived flag `__init__` needs to assign onto `self`.
    """
    model_state = None
    cfg = {}
    checkpoint_loaded = False

    action_stats: dict | None = None
    use_phase = False
    vel_proprio = False
    grounding_trained = False

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            if 'arch'    in ckpt: arch    = ckpt['arch']
            if 'chunk_H' in ckpt: chunk_H = ckpt['chunk_H']
            cfg = ckpt.get('cfg', {})

            # Fix 1: load action stats if embedded (train_gaitfix.py checkpoints)
            if 'action_stats' in ckpt:
                _as = ckpt['action_stats']
                action_stats = {
                    'mean':           np.array(_as['mean'],           dtype=np.float32),
                    'std':            np.array(_as['std'],            dtype=np.float32),
                    'default_angles': np.array(_as['default_angles'], dtype=np.float32),
                }
                print(f"[inferencer] Fix-1 residual action mode: loaded action_stats "
                      f"(n_frames={_as.get('n_frames', '?')})")

            # Fix 4: detect gait-phase checkpoints (proprio_dim=57)
            ckpt_proprio_dim = ckpt.get('proprio_dim', PROPRIO_DIM)
            if ckpt_proprio_dim == PROPRIO_DIM_PHASE or ckpt.get('dart_phase', False):
                use_phase = True
                print(f"[inferencer] Fix-4 gait-phase mode active: proprio_dim=57")

            # Grounding-trained flag: set if checkpoint was saved with grounding_trained=True
            if ckpt.get('grounding_trained', False):
                grounding_trained = True
                print(f"[inferencer] Grounding head trained: vision rendering enabled for learned grounding")

            # V6: vel_proprio flag
            if ckpt.get('vel_proprio', False):
                vel_proprio = True
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
            checkpoint_loaded = True
            print(f"[inferencer] Loaded checkpoint: {checkpoint_path}  arch={arch}  chunk_H={chunk_H}")
        else:
            print(f"[inferencer] WARN: unrecognized ckpt format in {checkpoint_path}; using random-init")
    elif checkpoint_path is not None:
        print(f"[inferencer] WARN: checkpoint not found: {checkpoint_path}; using random-init")

    # Use correct proprio_dim (57 for phase-conditioned checkpoints, 55 otherwise)
    _ckpt_proprio_dim = (PROPRIO_DIM_PHASE if use_phase else PROPRIO_DIM)
    model_cfg = {**DEFAULTS, **cfg, 'chunk_H': chunk_H,
                 'proprio_dim': _ckpt_proprio_dim,
                 'vel_proprio': vel_proprio}   # V6

    # teacher_forcing=True when we will inject an external goal OR an external velocity.
    # When True, forward() uses gt_goal (if not None) and gt_vel (if not None) in place
    # of the predicted values from the grounding/velocity heads.
    # For 'learned' goal and 'predicted' vel, keep teacher_forcing=False so both heads
    # run freely.
    _inject_goal = (arch == 'A' and
                    (goal_source in ('gt', 'classical') or
                     (goal_source == 'learned' and grounding_trained)))
    _inject_vel  = (arch == 'A' and vel_source == 'gt')
    _need_teacher_forcing = _inject_goal or _inject_vel
    model = GroundedNav(
        arch=arch,
        teacher_forcing=_need_teacher_forcing,   # True → gt injection active in forward()
        **{k: v for k, v in model_cfg.items() if k in DEFAULTS},
    ).to(device)
    model.eval()

    if model_state is not None:
        miss, unexp = model.load_state_dict(model_state, strict=False)
        if miss:  print(f"[inferencer]   {len(miss)} missing keys")
        if unexp: print(f"[inferencer]   {len(unexp)} unexpected keys")
    else:
        print(f"[inferencer] Random-init GroundedNav arch={arch} chunk_H={chunk_H}")

    return _LoadResult(
        model=model,
        checkpoint_loaded=checkpoint_loaded,
        arch=arch,
        chunk_H=chunk_H,
        action_stats=action_stats,
        use_phase=use_phase,
        grounding_trained=grounding_trained,
        vel_proprio=vel_proprio,
    )


# ---------------------------------------------------------------------------
# Video writer
# ---------------------------------------------------------------------------

def _write_video(
    frames_ego: list[np.ndarray],
    frames_tp: list[np.ndarray],
    out_path: str,
    fps: int = 50,
) -> None:
    """Writes recorded ego (and optional third-person) frames to a video file.

    If `frames_tp` is non-empty and matches `frames_ego` in length, each output
    frame is the ego frame with the (height-matched) third-person frame
    concatenated alongside it; otherwise only the ego frames are written.

    Args:
        frames_ego: List of (H,W,3) uint8 ego-camera frames.
        frames_tp: List of (H,W,3) uint8 third-person-camera frames (may be
            empty).
        out_path: Output video file path; parent directories are created if
            missing.
        fps: Output frame rate.
    """
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
