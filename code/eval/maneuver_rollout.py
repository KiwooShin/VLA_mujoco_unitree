"""code.eval.maneuver_rollout — standalone maneuver-skill rollout loop.

Split out of the original ``eval_maneuver.py`` (RF-1): drives one closed-loop
maneuver episode ("go straight, turn {left/right} after passing the
{color}{shape}") using the privileged ``ManeuverExpert`` FSM for teacher-forced
conditioning, exactly as the pre-RF-1 monolithic file did.

NOTE (no-consolidation invariant): this rollout loop is its own, independent
copy — not shared with eval_search.py's or demo.py's rollout loops — per
docs/refactor_plan.md.
"""

from __future__ import annotations

import collections
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from code.small_vla import GroundedNav
from code.arena import build_arena, ArenaRenderer
from code.teacher import WBCTeacher, _yaw_of, SIM_DT, CONTROL_DECIMATION
from code.gen_dart_dataset import GaitPhaseTracker
from code.maneuver_scene import SETTLE_STEPS
from code.maneuver_expert import ManeuverExpert

from code.eval.maneuver_types import (
    ManeuverResult, FALL_HEIGHT, PROPRIO_K, IMG_SIZE, HEADING_SUCCESS_THR,
    _build_proprio_maneuver, _apply_student_pd, _write_video,
)


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
