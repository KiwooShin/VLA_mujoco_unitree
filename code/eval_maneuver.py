"""
eval_maneuver.py — Closed-loop evaluation for the maneuver skill.

Task: "go straight, turn {left/right} after passing the {color}{shape}"

SUCCESS criteria (ALL must pass):
  1. Robot passes the landmark: robot_x > landmark_x + pass_margin
  2. Robot turns correct direction after passing: |final_heading_err| < HEADING_SUCCESS_THR
  3. Robot remains upright (height > 0.50m) throughout
  4. No fall

Eval protocol:
  - seed=999 (held-out)
  - n=15 episodes
  - MAXSTEPS=1400 (same as demo)
  - No render (fast eval)
  - Render first 3 success episodes for video

Uses the ManeuverInferencer (below) which:
  - Builds proprio as 62-d: 55 (base) + 2 (phase) + 5 (maneuver priv)
  - Sources privileged maneuver state from sim (subgoal_index, target_heading, etc.)
  - Injects GT maneuver conditioning into the model (teacher-forcing at deploy time)

Usage
-----
# Smoke test (1 episode, 150 steps):
MUJOCO_GL=egl python code/eval_maneuver.py \\
    --checkpoint runs/maneuver_A/epoch_0010.pt \\
    --smoke

# Full eval (15 episodes):
MUJOCO_GL=egl python code/eval_maneuver.py \\
    --checkpoint runs/maneuver_A/epoch_0010.pt \\
    --n 15 --device cuda \\
    --out eval/maneuver_A/ep10
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# GPU-rendering fix (2026-07-11): steer glvnd to the NVIDIA EGL ICD when
# present, BEFORE mujoco initializes EGL — otherwise Mesa can win the vendor
# race and MuJoCo silently renders on llvmpipe (CPU) at ~400 ms/frame vs
# ~1.3 ms on the GPU. Idempotent; no-op when the ICD file is absent or the
# user already chose a vendor. See code/arena.py for the measured numbers.
import os as _os
_NVIDIA_EGL_ICD = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if _os.path.exists(_NVIDIA_EGL_ICD):
    _os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NVIDIA_EGL_ICD)
import mujoco
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from code.small_vla import GroundedNav, DEFAULTS
from code.arena import build_arena, ArenaRenderer
from code.teacher import (WBCTeacher, _yaw_of, DEFAULT_ANGLES, KPS, KDS,
                           NUM_ACTIONS, SIM_DT, CONTROL_DECIMATION, RESET_HEIGHT)
from code.gen_dart_dataset import GaitPhaseTracker
from code.maneuver_scene import sample_maneuver_scene, derive_rng, SETTLE_STEPS, HORIZON
from code.maneuver_expert import ManeuverExpert, State
from code.dataset_maneuver import (PROPRIO_DIM_MANEUVER, PROPRIO_DIM_PHASE,
                                    PROPRIO_DIM_BASE, _build_maneuver_features)
from code.train_maneuver import load_loco_checkpoint

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


def run_maneuver_rollout(
    model:       GroundedNav,
    action_stats: dict,
    device:      torch.device,
    scene_cfg:   dict,
    maxsteps:    int  = 1400,
    render_video: bool = False,
    video_path:  str | None = None,
    vis_pooled_cache: torch.Tensor | None = None,  # pre-computed TinyViT pooled (B=1, vit_dim)
    free_vel:    bool = False,  # if True, use model's predicted vel (no teacher forcing)
    hybrid_vel:  bool = False,  # if True, TF only during TURN_PHASE (subgoal=1), free otherwise
) -> ManeuverResult:
    """Run one maneuver episode in closed-loop.

    Args:
        model: GroundedNav student model to roll out.
        action_stats: Dict with 'mean'/'std'/'default_angles' for de-normalizing
            residual actions, or None to use raw model output.
        device: Torch device to run inference on.
        scene_cfg: Scene configuration (robot pose, landmark, objects, etc.)
            from sample_maneuver_scene.
        maxsteps: Maximum number of control steps before terminating.
        render_video: Whether to render ego/third-person frames for video.
        video_path: Output path for the rendered video, or None to skip
            writing (still requires render_video=True to collect frames).
        vis_pooled_cache: Precomputed TinyViT output for a zero image (speeds
            up CPU eval by ~2x by bypassing the vision tower).
        free_vel: If True, always use the model's predicted velocity (no
            teacher forcing).
        hybrid_vel: If True, teacher-force velocity only during TURN_PHASE
            (subgoal_index == 1) and use the model's prediction otherwise.

    Returns:
        ManeuverResult summarizing success, failure tag, steps, and final
        state for the episode.
    """

    # Build arena
    arena_model = build_arena(scene_cfg)
    arena_model.opt.timestep = SIM_DT

    teacher = WBCTeacher(use_gpu=False)
    teacher.model = arena_model
    teacher.data  = mujoco.MjData(arena_model)
    teacher._nj   = arena_model.nq - 7
    teacher._pelvis_id = mujoco.mj_name2id(
        arena_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )

    rx, ry    = scene_cfg["robot_xy"]
    robot_yaw = float(scene_cfg.get("robot_yaw", 0.0))
    teacher.reset(pos_xy=(rx, ry), yaw=robot_yaw)

    data_mj  = teacher.data
    model_mj = teacher.model
    nj       = teacher._nj

    # Renderer (only needed for video)
    renderer = ArenaRenderer(model_mj) if render_video else None
    tp_cam   = renderer.make_tp_cam() if renderer else None
    frames_ego: list = []
    frames_tp:  list = []

    # Settle
    for _ in range(SETTLE_STEPS):
        teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        if teacher.base_height < FALL_HEIGHT:
            if renderer:
                renderer.close()
            return ManeuverResult(
                success=False, failure_tag='fall', steps=0,
                fell=True, upright=False,
                landmark_passed=False, final_heading_err=99.0,
                final_state=0, ms_per_step=0.0, scene_cfg=scene_cfg,
            )

    # Setup maneuver expert (privileged sim access for FSM state)
    expert = ManeuverExpert(scene_cfg)
    expert.reset()

    # Setup phase tracker
    phase_tracker = GaitPhaseTracker()

    # Action stats for de-normalization
    _use_residual = (action_stats is not None)
    if _use_residual:
        _da_mean  = action_stats['mean']
        _da_std   = action_stats['std']
        _da_deflt = action_stats['default_angles']

    # Language embedding (zeros — vision off, no lang cache)
    lang_t = torch.zeros(1, 2048, device=device)

    # Proprio history
    prev_action = teacher._target_dof.copy()
    # Prime priv state
    bxy_0  = data_mj.qpos[0:2].copy()
    byaw_0 = _yaw_of(data_mj.qpos[3:7])
    _, priv0 = expert.step(bxy_0, byaw_0)
    prop0 = _build_proprio_maneuver(data_mj, prev_action, phase_tracker, priv0)
    proprio_hist = collections.deque(
        [prop0.copy()] * PROPRIO_K, maxlen=PROPRIO_K
    )

    student_target_dof = teacher._target_dof.copy()
    step_times = []
    fell = False
    steps_done = 0

    for step in range(maxsteps):
        t0 = time.time()

        height = float(data_mj.qpos[2])
        if height < FALL_HEIGHT:
            fell = True
            break

        byaw = _yaw_of(data_mj.qpos[3:7])
        bxy  = data_mj.qpos[0:2].copy()

        # FSM expert for privileged state
        expert_vel_cmd, priv = expert.step(bxy, byaw)

        # Build proprio (62-d) with privileged maneuver state
        prop_now = _build_proprio_maneuver(data_mj, prev_action, phase_tracker, priv)
        proprio_hist.append(prop_now)
        prop_arr = np.stack(list(proprio_hist), axis=0)  # (K, 62)
        prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(device)

        # GT goal vector: egocentric to landmark
        lx, ly = scene_cfg["landmark_xy"]
        dx, dy = lx - float(bxy[0]), ly - float(bxy[1])
        dist_lm = math.hypot(dx, dy)
        bearing_lm = math.atan2(dy, dx)
        yaw_err_lm = math.atan2(math.sin(bearing_lm - byaw),
                                 math.cos(bearing_lm - byaw))
        gt_goal_np = np.array([dist_lm, math.cos(yaw_err_lm), math.sin(yaw_err_lm)],
                               dtype=np.float32)
        goal_inject_t = torch.from_numpy(gt_goal_np).unsqueeze(0).to(device)

        # Expert vel_cmd teacher-forcing (privileged, same as goal injection)
        expert_vel_t = torch.tensor(expert_vel_cmd, dtype=torch.float32, device=device).unsqueeze(0)
        # Determine vel injection mode:
        #   free_vel: always use model's vel_pred
        #   hybrid_vel: TF only during TURN_PHASE (subgoal_index=1), free otherwise
        #   default: always TF with expert vel_cmd
        fsm_state = int(priv["subgoal_index"])  # 0=STRAIGHT, 1=TURN_PHASE, 2=STRAIGHT2
        use_expert_vel = (not free_vel) and (not hybrid_vel or fsm_state == 1)

        with torch.no_grad():
            if vis_pooled_cache is not None:
                # Fast path: bypass TinyViT using precomputed pooled embedding
                lang   = model.lang_proj(lang_t)
                prop   = model.proprio_enc(prop_t)
                goal_pred = model.grounding(vis_pooled_cache, lang)
                vel_pred  = model.velocity(goal_pred, vis_pooled_cache, lang)
                goal_in   = goal_inject_t                                    # teacher-forced GT goal
                vel_in    = expert_vel_t if use_expert_vel else vel_pred     # TF or free vel
                goal_emb  = model.goal_proj(goal_in)
                vel_emb   = model.vel_proj(vel_in)
                feat_raw  = torch.cat([vis_pooled_cache, lang, prop, goal_emb, vel_emb], dim=-1)
                feat      = model.action_feat_proj(feat_raw)
                actions   = model.action_head(feat)
                out = {'action': actions}
            else:
                # Slow path: full model forward with zero image
                img_t = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32, device=device)
                gt_vel_arg = expert_vel_t if use_expert_vel else None
                out = model(
                    ego_rgb   = img_t,
                    lang_emb  = lang_t,
                    proprio_h = prop_t,
                    gt_goal   = goal_inject_t,
                    gt_vel    = gt_vel_arg,
                )

        raw_action = out['action'].cpu().numpy().squeeze(0)[0]  # (15,)

        # De-normalize (Fix 1 residual)
        if _use_residual:
            student_target_dof = _da_deflt + raw_action * _da_std + _da_mean
        else:
            student_target_dof = raw_action

        # Apply PD + physics
        for _ in range(CONTROL_DECIMATION):
            _apply_student_pd(data_mj, student_target_dof, nj)
            mujoco.mj_step(model_mj, data_mj)

        prev_action = student_target_dof.copy()
        steps_done  = step + 1

        # Optionally render
        if render_video and renderer:
            rgb, _, _ = renderer.render_ego(data_mj, byaw, render_depth=False)
            frames_ego.append(rgb.copy())
            if tp_cam:
                renderer.update_tp_cam(tp_cam, data_mj)
                frames_tp.append(renderer.render_tp(data_mj, tp_cam).copy())

        t1 = time.time()
        step_times.append((t1 - t0) * 1000.0)

    if renderer:
        renderer.close()

    # Final state evaluation
    final_height = float(data_mj.qpos[2])
    upright = final_height >= FALL_HEIGHT and not fell

    bxy_final  = data_mj.qpos[0:2].copy()
    byaw_final = _yaw_of(data_mj.qpos[3:7])
    _, priv_final = expert.step(bxy_final, byaw_final)

    landmark_passed  = expert.landmark_passed
    final_heading_err = float(priv_final["heading_err"])
    final_state       = int(expert.state)

    # Success logic
    if fell or not upright:
        success = False
        failure_tag = 'fall'
    elif not landmark_passed:
        success = False
        failure_tag = 'no_landmark'
    elif abs(final_heading_err) > HEADING_SUCCESS_THR:
        # Didn't turn enough OR turned wrong way
        success = False
        failure_tag = 'wrong_heading'
    else:
        success = True
        failure_tag = 'success'

    ms_per_step = float(np.mean(step_times)) if step_times else 0.0

    # Write video
    if render_video and video_path and frames_ego:
        _write_video(frames_ego, frames_tp, video_path)

    return ManeuverResult(
        success           = success,
        failure_tag       = failure_tag,
        steps             = steps_done,
        fell              = fell,
        upright           = upright,
        landmark_passed   = landmark_passed,
        final_heading_err = final_heading_err,
        final_state       = final_state,
        ms_per_step       = ms_per_step,
        scene_cfg         = scene_cfg,
        video_path        = video_path if (render_video and frames_ego) else None,
    )


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


def evaluate_maneuver(
    checkpoint_path: str,
    n_scenes: int = 15,
    seed: int = 999,
    device_str: str = 'cpu',
    out_dir: str = 'eval/maneuver',
    render_n: int = 3,       # render first N successes
    smoke: bool = False,
    smoke_steps: int = 150,
    free_vel: bool = False,   # if True, use model's predicted vel (no teacher forcing)
    hybrid_vel: bool = False, # if True, TF only during TURN_PHASE, free otherwise
) -> dict:
    """Run maneuver evaluation.

    Args:
        checkpoint_path: Path to the .pt checkpoint to evaluate.
        n_scenes: Number of held-out scenes to evaluate.
        seed: Eval seed (default=999, held-out).
        device_str: Torch device string ('cpu' or 'cuda').
        out_dir: Output directory for videos, incremental results, and the
            summary JSON.
        render_n: Number of successful episodes to re-render as video.
        smoke: If True, run a single episode with smoke_steps as the cap.
        smoke_steps: Step cap used when smoke=True.
        free_vel: If True, use the model's predicted velocity (no teacher
            forcing).
        hybrid_vel: If True, teacher-force velocity only during TURN_PHASE,
            free otherwise.

    Returns:
        Summary dict with success_rate, failure tag breakdown, mean steps,
        etc. (also written to out_dir/summary.json).

    Raises:
        FileNotFoundError: If checkpoint_path does not point to an existing
            file.
    """
    device = torch.device(device_str)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[eval_maneuver] Checkpoint: {checkpoint_path}")
    print(f"[eval_maneuver] seed={seed}  n={n_scenes}  device={device_str}")
    print(f"[eval_maneuver] out_dir={out_dir}", flush=True)

    # Load checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    ckpt_proprio_dim = ckpt.get('proprio_dim', PROPRIO_DIM_MANEUVER)
    is_maneuver = ckpt.get('maneuver', False)
    arch = ckpt.get('arch', 'A')

    print(f"[eval_maneuver] ckpt proprio_dim={ckpt_proprio_dim}  maneuver={is_maneuver}")

    # Build model at the checkpoint's proprio_dim, then expand if needed
    if ckpt_proprio_dim == PROPRIO_DIM_MANEUVER:
        model = GroundedNav(
            arch=arch, teacher_forcing=True, chunk_H=1,
            proprio_dim=PROPRIO_DIM_MANEUVER,
        ).to(device)
        model_state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
        model.load_state_dict(model_state, strict=False)
    else:
        # Locomotion checkpoint (57-d), expand to 62-d
        model, _ = load_loco_checkpoint(checkpoint_path, device)

    model.eval()

    # Precompute TinyViT output for zero image (cached for all steps — 2x speedup on CPU)
    with torch.no_grad():
        _img_zero = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32, device=device)
        _, vis_pooled_cache = model.vision(_img_zero)
    print(f"[eval_maneuver] Precomputed TinyViT embedding (shape={vis_pooled_cache.shape})", flush=True)

    # Action stats
    _as = ckpt.get('action_stats', None)
    if _as:
        action_stats = {
            'mean':           np.array(_as['mean'],           dtype=np.float32),
            'std':            np.array(_as['std'],            dtype=np.float32),
            'default_angles': np.array(_as['default_angles'], dtype=np.float32),
            'n_frames':       _as.get('n_frames', 0),
        }
        print(f"[eval_maneuver] Loaded action_stats from checkpoint", flush=True)
    else:
        action_stats = None
        print(f"[eval_maneuver] WARNING: No action_stats in checkpoint — using raw action output")

    # Eval loop
    results = []
    n_success = 0
    n_fall = 0
    n_no_lm = 0
    n_wrong_heading = 0
    videos_rendered = 0

    maxsteps = smoke_steps if smoke else HORIZON

    for ep_i in range(n_scenes if not smoke else 1):
        rng       = derive_rng(seed, ep_i)
        scene_cfg = sample_maneuver_scene(rng)

        turn_dir  = scene_cfg["turn_direction"]
        lm_obj    = scene_cfg["objects"][scene_cfg["landmark_index"]]
        print(f"\n[ep {ep_i:02d}/{n_scenes}]  '{scene_cfg['instruction'][:60]}'  "
              f"turn={turn_dir}  lm_dist={lm_obj['dist_from_robot']:.2f}m", flush=True)

        # Render first `render_n` successes
        do_render = render_video = False   # set after run to not slow things down

        ep_t0 = time.time()
        result = run_maneuver_rollout(
            model             = model,
            action_stats      = action_stats,
            device            = device,
            scene_cfg         = scene_cfg,
            maxsteps          = maxsteps,
            render_video      = False,
            video_path        = None,
            vis_pooled_cache  = vis_pooled_cache,
            free_vel          = free_vel,
            hybrid_vel        = hybrid_vel,
        )
        ep_elapsed = time.time() - ep_t0

        ep_sps = result.steps / max(ep_elapsed, 1e-6)
        print(f"  -> steps={result.steps}  lm_passed={result.landmark_passed}  "
              f"heading_err={math.degrees(result.final_heading_err):.1f}°  "
              f"fell={result.fell}  tag={result.failure_tag}  {ep_sps:.1f}stp/s",
              flush=True)

        if result.success:
            n_success += 1
            # Render video if we haven't done render_n yet
            if videos_rendered < render_n:
                vid_path = os.path.join(out_dir, f"ep{ep_i:03d}_success.mp4")
                print(f"  Rendering success video → {vid_path}")
                result2 = run_maneuver_rollout(
                    model             = model,
                    action_stats      = action_stats,
                    device            = device,
                    scene_cfg         = scene_cfg,
                    maxsteps          = maxsteps,
                    render_video      = True,
                    video_path        = vid_path,
                    vis_pooled_cache  = vis_pooled_cache,
                    free_vel          = free_vel,
                    hybrid_vel        = hybrid_vel,
                )
                result.video_path = result2.video_path
                videos_rendered += 1
        elif result.failure_tag == 'fall':
            n_fall += 1
        elif result.failure_tag == 'no_landmark':
            n_no_lm += 1
        elif result.failure_tag == 'wrong_heading':
            n_wrong_heading += 1

        results.append({
            "ep":              ep_i,
            "turn_direction":  turn_dir,
            "lm_dist":         round(lm_obj['dist_from_robot'], 2),
            "steps":           result.steps,
            "landmark_passed": result.landmark_passed,
            "final_heading_err_deg": round(math.degrees(result.final_heading_err), 1),
            "final_state":     result.final_state,
            "fell":            result.fell,
            "success":         result.success,
            "failure_tag":     result.failure_tag,
            "video_path":      result.video_path,
        })

        # Write incremental results
        with open(os.path.join(out_dir, 'results.jsonl'), 'a') as f:
            f.write(json.dumps(results[-1]) + "\n")

        # EGL non-determinism fix: force GC between episodes.
        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
        except Exception:
            pass

    n = len(results)
    success_rate = n_success / max(1, n)
    summary = {
        "checkpoint":          checkpoint_path,
        "seed":                seed,
        "n_episodes":          n,
        "n_success":           n_success,
        "success_rate":        round(success_rate, 3),
        "n_fall":              n_fall,
        "n_no_landmark":       n_no_lm,
        "n_wrong_heading":     n_wrong_heading,
        "pct_fall":            round(n_fall / max(1, n), 3),
        "pct_no_landmark":     round(n_no_lm / max(1, n), 3),
        "pct_wrong_heading":   round(n_wrong_heading / max(1, n), 3),
        "mean_steps":          round(float(np.mean([r["steps"] for r in results])), 1),
        "heading_thr_deg":     round(math.degrees(HEADING_SUCCESS_THR), 1),
    }

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  MANEUVER EVAL RESULTS  (seed={seed}, n={n})")
    print(f"{'='*70}")
    print(f"  {'ep':>4}  {'turn':>6}  {'lm_d':>5}  {'steps':>6}  "
          f"{'lm_pass':>8}  {'head_err':>9}  {'tag'}")
    print("  " + "-"*65)
    for r in results:
        print(f"  {r['ep']:>4}  {r['turn_direction']:>6}  {r['lm_dist']:>5.1f}  "
              f"{r['steps']:>6}  {str(r['landmark_passed']):>8}  "
              f"{r['final_heading_err_deg']:>+8.1f}°  {r['failure_tag']}")
    print()
    print(f"  SUCCESS: {n_success}/{n} = {success_rate:.1%}")
    print(f"  Falls:         {n_fall}/{n}")
    print(f"  No landmark:   {n_no_lm}/{n}")
    print(f"  Wrong heading: {n_wrong_heading}/{n}")
    print(f"  Mean steps:    {summary['mean_steps']:.1f}")
    print(f"{'='*70}", flush=True)

    return summary


def main() -> None:
    """Parse CLI arguments and run the maneuver evaluation."""
    ap = argparse.ArgumentParser(description="Maneuver skill evaluator")
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n',          type=int,   default=15)
    ap.add_argument('--seed',       type=int,   default=999)
    ap.add_argument('--device',     default='cpu')
    ap.add_argument('--out',        default='eval/maneuver')
    ap.add_argument('--render-n',   type=int,   default=3,
                    help='Render first N success episodes as video')
    ap.add_argument('--smoke',      action='store_true',
                    help='Quick smoke test (1 episode, 150 steps)')
    ap.add_argument('--smoke-steps', type=int, default=150)
    ap.add_argument('--free-vel',   action='store_true',
                    help='Ablation: use model predicted vel (no expert teacher-forcing)')
    ap.add_argument('--no-hybrid-vel', action='store_true',
                    help='Disable hybrid-vel (default: ON — TF vel only during TURN_PHASE)')
    ap.add_argument('--hybrid-vel', action='store_true',
                    help='[deprecated flag, now on by default] TF vel during TURN_PHASE only')
    args = ap.parse_args()

    # hybrid_vel is ON by default; disable with --no-hybrid-vel or --free-vel
    use_hybrid = not args.free_vel and not args.no_hybrid_vel

    summary = evaluate_maneuver(
        checkpoint_path = args.checkpoint,
        n_scenes        = args.n,
        seed            = args.seed,
        device_str      = args.device,
        out_dir         = args.out,
        render_n        = args.render_n,
        smoke           = args.smoke,
        smoke_steps     = args.smoke_steps,
        free_vel        = args.free_vel,
        hybrid_vel      = use_hybrid,
    )
    print(json.dumps(summary, indent=2))
    sys.exit(0)


if __name__ == '__main__':
    main()
