"""Closed-loop maneuver-skill executor for the REPL demo (code/demo.py, RF-1
split). Wraps eval_maneuver.py's logic directly, with scene_cfg supplied by
SceneManager, using hybrid_vel: GT vel injection during TURN_PHASE only.
"""

from __future__ import annotations

import collections
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from code.apps.repl.constants import MAXSTEPS_MANEUVER, _REPO


class ManeuverInferencer:
    """
    Closed-loop maneuver skill executor.
    Reuses logic from eval_maneuver.py but with scene_cfg from SceneManager.
    Uses hybrid_vel: GT vel injection during TURN_PHASE only.
    """

    def __init__(
        self, checkpoint_path: str, device: str = "cpu", use_keyframe: bool = True
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device_str      = device
        self._model          = None
        self._action_stats   = None
        self._loaded         = False

        # H4: WBC-free settle — load offline stand keyframe (same as Inferencer in inferencer.py)
        # When use_keyframe=True and checkpoint/stand_keyframe.npz exists, skip the WBC settle
        # and instead restore physics from the offline keyframe. No WBC ONNX called at runtime.
        self._keyframe: dict | None = None
        _kf_path = str(_REPO / "checkpoint" / "stand_keyframe.npz")
        if use_keyframe and os.path.isfile(_kf_path):
            _kf = np.load(_kf_path)
            self._keyframe = {
                'qpos_local': _kf['qpos_local'].copy(),
                'qvel_local': _kf['qvel_local'].copy(),
                'target_dof': _kf['target_dof'].copy(),
                'height':     float(_kf['height']),
            }
            print(f"[maneuver_inferencer] Keyframe init: loaded {_kf_path} "
                  f"(height={self._keyframe['height']:.4f}m) — WBC-free settle active")
        elif use_keyframe:
            print(f"[maneuver_inferencer] Keyframe init: {_kf_path} not found, "
                  f"falling back to WBC settle")

    def _load_model(self) -> None:
        """Lazy-load model on first use."""
        if self._loaded:
            return
        import torch
        from code.train_maneuver import load_loco_checkpoint

        device = torch.device(self.device_str)
        ckpt_path = self.checkpoint_path

        if os.path.isfile(ckpt_path):
            # load_loco_checkpoint handles GRU expansion from 57→62 and returns (model, ckpt)
            model, ckpt = load_loco_checkpoint(ckpt_path, device)
            if ckpt.get('action_stats'):
                _as = ckpt['action_stats']
                self._action_stats = {
                    'mean': np.array(_as['mean'], dtype=np.float32),
                    'std':  np.array(_as['std'],  dtype=np.float32),
                    'default_angles': np.array(_as['default_angles'], dtype=np.float32),
                }
            elif os.path.isfile(str(Path(ckpt_path).parent / 'action_stats.json')):
                # Load from companion file
                import json as _json
                with open(str(Path(ckpt_path).parent / 'action_stats.json')) as f:
                    _as = _json.load(f)
                self._action_stats = {
                    'mean': np.array(_as['mean'], dtype=np.float32),
                    'std':  np.array(_as['std'],  dtype=np.float32),
                    'default_angles': np.array(_as['default_angles'], dtype=np.float32),
                }
        else:
            print(f"[maneuver_inferencer] WARN: ckpt not found {ckpt_path}, random init", flush=True)
            from code.small_vla import GroundedNav
            from code.dataset_maneuver import PROPRIO_DIM_MANEUVER
            model = GroundedNav(
                arch='A', teacher_forcing=True, proprio_dim=PROPRIO_DIM_MANEUVER,
            ).to(device)

        model.eval()
        self._model  = model
        self._device = device
        self._loaded = True
        print(f"[maneuver_inferencer] Model ready (ckpt: {ckpt_path})", flush=True)

    def rollout(
        self,
        scene_cfg: dict,
        instruction: str,
        maxsteps: int = MAXSTEPS_MANEUVER,
        render_video: bool = False,
        video_path: str | None = None,
        progress_cb: Callable[[dict], None] | None = None,
    ) -> dict[str, Any]:
        """
        Run one closed-loop maneuver episode.

        Returns result dict with keys:
          success, failure_tag, steps, heading_err, landmark_passed, video_path
        """
        self._load_model()

        import torch
        import mujoco
        from code.arena import build_arena, ArenaRenderer
        from code.teacher import (WBCTeacher, _yaw_of, DEFAULT_ANGLES, KPS, KDS,
                                   NUM_ACTIONS, SIM_DT, CONTROL_DECIMATION)
        from code.gen_dart_dataset import GaitPhaseTracker
        from code.maneuver_scene import sample_maneuver_scene, derive_rng, SETTLE_STEPS, HORIZON, PASS_MARGIN
        from code.maneuver_expert import ManeuverExpert, State, FORWARD_VX, MAX_WZ, TURN_KP
        from code.dataset_maneuver import (PROPRIO_DIM_MANEUVER, PROPRIO_DIM_PHASE,
                                            PROPRIO_DIM_BASE, _build_maneuver_features)
        from code.inferencer import _GaitPhaseTracker, FALL_HEIGHT, PROPRIO_K

        # Build a maneuver scene_cfg from the provided scene_cfg's landmark info
        # The scene_cfg must have 'task'=='maneuver' keys
        if scene_cfg.get('task') != 'maneuver':
            # Build a fresh maneuver scene
            rng = derive_rng(999, scene_cfg.get('_maneuver_ep', 0))
            scene_cfg = sample_maneuver_scene(rng)

        arena_model = build_arena(scene_cfg)
        arena_model.opt.timestep = SIM_DT

        teacher = WBCTeacher(use_gpu=False)
        teacher.model = arena_model
        teacher.data  = mujoco.MjData(arena_model)
        teacher._nj   = arena_model.nq - 7
        teacher._pelvis_id = mujoco.mj_name2id(
            arena_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

        rx, ry    = scene_cfg['robot_xy']
        robot_yaw = float(scene_cfg.get('robot_yaw', 0.0))
        teacher.reset(pos_xy=(rx, ry), yaw=robot_yaw)
        data_mj  = teacher.data
        model_mj = teacher.model
        nj       = teacher._nj

        # H4: Settle — keyframe restore (WBC-free) OR WBC settle (fallback)
        if self._keyframe is not None:
            # Keyframe path: restore saved physics state — no WBC ONNX called at runtime.
            kf = self._keyframe
            kf_qpos = kf['qpos_local'].copy()
            # Apply scene xy/yaw: overwrite x, y, and orientation quaternion
            kf_qpos[0] = rx
            kf_qpos[1] = ry
            kf_qpos[3] = math.cos(robot_yaw / 2)  # w
            kf_qpos[4] = 0.0                        # x
            kf_qpos[5] = 0.0                        # y
            kf_qpos[6] = math.sin(robot_yaw / 2)   # z
            data_mj.qpos[:len(kf_qpos)] = kf_qpos
            data_mj.qvel[:len(kf['qvel_local'])] = kf['qvel_local']
            mujoco.mj_forward(model_mj, data_mj)
            teacher._target_dof = kf['target_dof'].copy()
            if teacher.base_height < FALL_HEIGHT:
                return dict(success=False, failure_tag='fall', steps=0,
                            heading_err=math.pi, landmark_passed=False, video_path=None)
        else:
            # WBC ONNX settle (fallback — only reached if stand_keyframe.npz missing)
            for _ in range(SETTLE_STEPS):
                teacher.step(vel_cmd=(0.0, 0.0, 0.0))
                if teacher.base_height < FALL_HEIGHT:
                    return dict(success=False, failure_tag='fall', steps=0,
                                heading_err=math.pi, landmark_passed=False, video_path=None)

        # Renderer
        renderer = ArenaRenderer(model_mj)
        tp_cam   = renderer.make_tp_cam()
        frames_ego, frames_tp = [], []

        # Expert FSM (privileged)
        expert = ManeuverExpert(scene_cfg)
        expert.reset()

        # Phase tracker
        phase_tracker = _GaitPhaseTracker()

        # Proprio history (62-d)
        prop_dim  = PROPRIO_DIM_MANEUVER  # 62
        prop_hist = collections.deque(
            [np.zeros(prop_dim, dtype=np.float32)] * PROPRIO_K,
            maxlen=PROPRIO_K,
        )

        # Zero lang embedding
        lang_t = torch.zeros(1, 2048, device=self._device)

        prev_action = teacher._target_dof.copy()
        ACTION_SCALE = 0.25

        fell    = False
        steps   = 0
        success = False
        failure_tag = 'didnt-reach'
        heading_err_final = math.pi
        lm_passed = False

        landmark_xy = np.array(scene_cfg.get('landmark_xy', [0.0, 0.0]))
        target_heading = float(scene_cfg.get('target_heading', math.pi / 2))
        pass_margin = float(scene_cfg.get('pass_margin', 0.6))

        HEADING_THR = math.radians(25.0)

        for step in range(maxsteps):
            height = float(data_mj.qpos[2])
            if height < FALL_HEIGHT:
                fell = True
                failure_tag = 'fall'
                break

            robot_xy  = data_mj.qpos[0:2].copy()
            robot_yaw = _yaw_of(data_mj.qpos[3:7])

            # Expert FSM step (privileged)
            vel_cmd, priv = expert.step(robot_xy, robot_yaw)
            subgoal_idx  = priv['subgoal_index']
            heading_err  = priv['heading_err']
            lm_passed    = priv['landmark_passed']
            heading_err_final = heading_err

            # Hybrid vel: inject expert vel only during TURN_PHASE
            _in_turn = (subgoal_idx == State.TURN_PHASE)

            # Build 62-d proprio
            from code.inferencer import _build_proprio
            prop_base = _build_proprio(data_mj, prev_action)  # 55-d
            q_lb      = data_mj.qpos[7:22].copy()
            ph        = phase_tracker.update(q_lb)              # 2-d
            mv_feat   = _build_maneuver_features(priv)          # 5-d
            prop_now  = np.concatenate([prop_base, ph, mv_feat]).astype(np.float32)  # 62-d
            prop_hist.append(prop_now)

            prop_arr = np.stack(list(prop_hist), axis=0)
            prop_t   = torch.from_numpy(prop_arr).unsqueeze(0).to(self._device)

            # Goal: egocentric vector to landmark
            delta   = landmark_xy - robot_xy
            dist_lm = float(np.linalg.norm(delta))
            cx_l    = math.cos(robot_yaw) * delta[0] + math.sin(robot_yaw) * delta[1]
            cy_l    = -math.sin(robot_yaw) * delta[0] + math.cos(robot_yaw) * delta[1]
            ye_lm   = math.atan2(cy_l, cx_l)
            goal_vec = np.array([dist_lm, math.cos(ye_lm), math.sin(ye_lm)], dtype=np.float32)
            goal_t  = torch.from_numpy(goal_vec).unsqueeze(0).to(self._device)

            # Vel injection (hybrid: only during TURN_PHASE)
            if _in_turn:
                vel_t = torch.tensor([[float(vel_cmd[0]), float(vel_cmd[1]), float(vel_cmd[2])]],
                                     dtype=torch.float32, device=self._device)
            else:
                vel_t = None

            img_t = torch.zeros(1, 3, 128, 128, device=self._device)

            with torch.no_grad():
                out = self._model(
                    ego_rgb   = img_t,
                    lang_emb  = lang_t,
                    proprio_h = prop_t,
                    gt_goal   = goal_t,
                    gt_vel    = vel_t,
                )

            raw_action = out['action'].cpu().numpy().squeeze(0)[0]  # (15,)

            # De-normalize (residual action)
            if self._action_stats is not None:
                _as = self._action_stats
                target_dof = _as['default_angles'] + raw_action * _as['std'] + _as['mean']
            else:
                target_dof = raw_action

            # PD + physics (reapply PD torques every substep, matching eval_maneuver.py)
            from code.inferencer import _apply_student_pd
            for _ in range(CONTROL_DECIMATION):
                _apply_student_pd(data_mj, target_dof, nj)
                mujoco.mj_step(model_mj, data_mj)

            prev_action = target_dof.copy()
            steps = step + 1

            # Render frames if requested
            if render_video:
                yaw_now = _yaw_of(data_mj.qpos[3:7])
                rgb, _, _ = renderer.render_ego(data_mj, yaw_now, render_depth=False)
                frames_ego.append(rgb.copy())
                renderer.update_tp_cam(tp_cam, data_mj)
                frames_tp.append(renderer.render_tp(data_mj, tp_cam).copy())

            # Progress callback
            if progress_cb and step % 50 == 0:
                pct = int(step / maxsteps * 100)
                progress_cb({
                    "step": step, "steps": maxsteps,
                    "pct": pct,
                    "heading_err_deg": math.degrees(abs(heading_err_final)),
                    "lm_passed": lm_passed,
                    "phase": ["STRAIGHT", "TURN", "STRAIGHT2"][subgoal_idx],
                })

            # Check success: landmark passed + heading aligned
            if lm_passed and abs(heading_err_final) < HEADING_THR:
                success = True
                failure_tag = 'success'
                break

        renderer.close()

        if not fell and not success:
            failure_tag = 'didnt-reach'

        # Write video
        out_vid = None
        if render_video and video_path and frames_ego:
            from code.inferencer import _write_video
            _write_video(frames_ego, frames_tp, video_path)
            out_vid = video_path

        return dict(
            success=success,
            failure_tag=failure_tag,
            steps=steps,
            heading_err_deg=math.degrees(abs(heading_err_final)),
            landmark_passed=lm_passed,
            video_path=out_vid,
        )
