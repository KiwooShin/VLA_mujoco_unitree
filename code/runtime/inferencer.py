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
MAXSTEPS hard cap: easy=600, demo=1700 (NX-10: was 1400; bumped alongside the H3 scan's
widened realized coverage -- see docs/nx10_scan_fix.md). `maxsteps` is a caller-supplied
`rollout()` argument, not hardcoded here -- see code/eval_closedloop.py's `MAXSTEPS` dict
and code/demo.py's `MAXSTEPS_GOTO` for the two callers this file's docstring tracks.

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

RF-1 (docs/refactor_plan.md): this file is now the thin facade re-assembling
the pieces split out of the original 1789-line monolith:
  - code/runtime/constants.py    — module-level constants + env toggles
  - code/runtime/gait_phase.py   — `_GaitPhaseTracker`
  - code/runtime/helpers.py      — `_build_proprio`/`_apply_student_pd`/
                                   `_rgb_to_tensor`/`_label_active_cam`
  - code/runtime/gt_goal.py      — `_compute_gt_goal`
  - code/runtime/io.py           — checkpoint/model loading, `RolloutResult`,
                                   `_write_video`
  - code/runtime/goal_pipeline.py + goal_config.py — grounding-cycle/goal/EMA/
                                   hold/handoff/scan state (`GoalPipeline`)
  - code/runtime/rollout_state.py — per-episode setup + mutable state bundle
  - code/runtime/rollout_step.py  — the per-step control loop body
All of these are re-imported here so `code.inferencer.<name>` (the old flat
path, via the sys.modules alias at code/inferencer.py) keeps resolving every
name external callers rely on, unchanged. `classical_ground` is imported
directly into THIS module (not re-exported from a sub-module) and called only
via the `Inferencer._ground` pass-through below — see that method's
docstring for why (monkeypatch preservation for diagnostic scripts).
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Re-exports for old-path compatibility (`code.inferencer.<name>`, via the
# sys.modules alias at code/inferencer.py) — every name below was a
# module-level import or definition in the original flat file.
# ---------------------------------------------------------------------------
from code.policy.small_vla import GroundedNav, DEFAULTS
from code.sim.arena import (build_arena, ArenaRenderer, EGO_W, EGO_H, EGO_FOVY, get_ego_intrinsics,
                            GROUNDING_W, GROUNDING_H, CAMERA_MODE, CAM_HEAD_Z)
from code.sim.scene import DIFFICULTY_PRESETS
from code.perception.grounding import ground as classical_ground, _parse_instruction, get_ego_intrinsics_rendered
from code.sim.teacher import (WBCTeacher, _yaw_of, DEFAULT_ANGLES, KPS, KDS,
                              NUM_ACTIONS, SIM_DT, CONTROL_DECIMATION, RESET_HEIGHT)
from code.control.steer import steer as _steer_cmd
from code.perception.lock_mgmt import LockGate, ReacquisitionScan
from code.control import avoid as _avoid
from code.control.scan_sched import BidirectionalScanSchedule, SCAN_DWELL_STEPS as _H3_DWELL_STEPS

from code.runtime.constants import (
    FALL_HEIGHT, GROUNDING_PERIOD, KEYFRAME_PATH, PROPRIO_K, PROPRIO_DIM,
    PROPRIO_DIM_PHASE, IMG_SIZE, HOLD_STEPS_REQUIRED, ACTION_SCALE, SETTLE_STEPS,
    _env_flag, STALL_BREAK, STALL_VX_THR_MPS, STALL_WINDOW_STEPS, STALL_DISP_THR_M,
    STALL_MIN_GOAL_DIST_M, STALL_RECOVERY_STEPS, STALL_COOLDOWN_STEPS, AVOID,
    _DEFAULT_ANGLES_NP, _LEFT_ANKLE_PITCH_IDX, _LEFT_ANKLE_DEFAULT,
)
from code.runtime.gait_phase import _GaitPhaseTracker
from code.runtime.helpers import _build_proprio, _apply_student_pd, _rgb_to_tensor, _label_active_cam
from code.runtime.gt_goal import _compute_gt_goal
from code.runtime.io import RolloutResult, _write_video, load_keyframe, build_model
from code.runtime.rollout_state import _setup_rollout
from code.runtime.rollout_step import _rollout_step


class Inferencer:
    """Closed-loop rollout harness for GroundedNav student.

    Args:
        checkpoint_path: Path to a GroundedNav .pt (None = random-init for
            harness test).
        arch: 'A' or 'C'.
        device: 'cpu' | 'cuda'.
        chunk_H: Action chunking horizon (1 = no chunking).
        goal_source: 'learned' | 'classical' | 'gt' (Arch A only).
            'learned'   — model's own grounding head (default).
            'classical' — HSV+depth classical grounding.
            'gt'        — privileged goal from sim state (upper bound).
        vel_source: 'predicted' | 'gt' (Fix 2 upper bound).
        verbose: Per-step print.
        use_keyframe: True → load stand_keyframe.npz (WBC-free settle).
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        arch:        str  = 'A',
        device:      str  = 'cpu',
        chunk_H:     int  = 1,
        goal_source: str  = 'classical',   # 'learned' | 'classical' | 'gt'
        vel_source:  str  = 'predicted',   # 'predicted' | 'gt' (Fix 2 upper bound)
        verbose:     bool = False,
        use_keyframe: bool = True,          # True → load stand_keyframe.npz (WBC-free settle)
    ) -> None:
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
        self._keyframe = load_keyframe(use_keyframe)

        # ---- Load / random-init model (checkpoint parsing lives in code/runtime/io.py) ----
        load_result = build_model(checkpoint_path, arch, chunk_H, self.device,
                                  self.goal_source, self.vel_source)
        self.model             = load_result.model
        self._checkpoint_loaded = load_result.checkpoint_loaded
        self.arch               = load_result.arch
        self.chunk_H             = load_result.chunk_H
        self._action_stats       = load_result.action_stats       # Fix 1: residual action stats
        self._use_phase          = load_result.use_phase          # Fix 4: gait phase input
        self._grounding_trained  = load_result.grounding_trained  # learned-grounding vision path
        self._vel_proprio        = load_result.vel_proprio        # V6: vel head proprio input

    # ------------------------------------------------------------------
    def _ground(self, rgb, depth, target_color, target_shape, intr):
        """Calls `classical_ground` — kept as a call site in THIS module (not
        in code/runtime/goal_pipeline.py or rollout_step.py) so diagnostic
        scripts that monkeypatch `code.inferencer.classical_ground`
        (code/gen_det_failcases.py, eval/nx7_ep1_diag/*, eval/nx8_stall/*)
        keep observing every grounding call this rollout makes, exactly as
        when this call site lived inline in this same file (RF-1 split,
        docs/refactor_plan.md)."""
        return classical_ground(rgb, depth, target_color, target_shape, intr)

    # ------------------------------------------------------------------
    def rollout(
        self,
        scene_cfg:    dict,
        instruction:  str,
        lang_emb:     np.ndarray | None = None,
        maxsteps:     int   = 600,
        render_video: bool  = False,
        video_path:   str | None = None,
        render_tp:    bool  = True,
        stop_r:       float | None = None,
    ) -> RolloutResult:
        """Runs one closed-loop episode.

        The WBC teacher is used ONLY for the settle phase (SETTLE_STEPS with zero
        velocity command) to bring the G1 to a stable standing pose.
        After settle, the STUDENT drives the robot: student output → PD → physics.

        Args:
            scene_cfg: Scene configuration dict (objects, target_index, robot_xy,
                robot_yaw, stop_r, difficulty, ...) as produced by code/scene.py.
            instruction: Natural-language instruction for this episode (unused
                directly here beyond bookkeeping; language conditioning comes from
                `lang_emb`).
            lang_emb: Optional (2048,) language embedding. If None, a zero vector
                is used, or (for a trained grounding head with 'learned' goal
                source) a one-hot color+shape encoding is built instead.
            maxsteps: Hard cap on the number of student control steps.
            render_video: If True, record ego (and optionally third-person) frames
                for video output.
            video_path: Output path for the rendered video (required if
                `render_video` is True and any frames were recorded).
            render_tp: If True (and `render_video` is True), also record
                third-person frames.
            stop_r: Success radius (m). Defaults to `scene_cfg['stop_r']` (or 0.6)
                when None.

        Returns:
            RolloutResult summarizing the episode outcome.
        """
        s = _setup_rollout(self, scene_cfg, lang_emb, stop_r)
        if s.early_result is not None:
            return s.early_result

        for step in range(maxsteps):
            done = _rollout_step(self, s, step, render_video, render_tp)
            if done:
                break

        # ---- Done ----
        s.renderer.close()

        final_height = float(s.data_mj.qpos[2])
        upright      = final_height >= FALL_HEIGHT and not s.fell
        final_dist   = float(np.linalg.norm(s.data_mj.qpos[0:2] - s.target_xy))
        success      = (final_dist < s.stop_r) and upright

        if success:
            failure_tag = 'success'
        elif s.fell or not upright:
            failure_tag = 'fall'
        else:
            failure_tag = 'didnt-reach'

        ms_per_step  = float(np.mean(s.step_times)) if s.step_times else 0.0
        grounding_hz = (1.0 / (GROUNDING_PERIOD * SIM_DT * CONTROL_DECIMATION)
                        if self.arch == 'A' else 0.0)

        # ---- Gait oscillation check ----
        if s.all_target_dofs:
            tdf_arr = np.stack(s.all_target_dofs, axis=0)   # (steps, 15)
            osc_std = float(tdf_arr.std(axis=0).mean())    # mean per-joint std
        else:
            osc_std = 0.0

        # Forward displacement from start
        fwd_xy       = s.data_mj.qpos[0:2].copy()
        forward_disp = float(np.linalg.norm(fwd_xy - s.start_xy))

        # Write video
        if render_video and video_path and s.frames_ego:
            _write_video(s.frames_ego, s.frames_tp, video_path)

        gp = s.goal_pipeline
        return RolloutResult(
            success        = success,
            failure_tag    = failure_tag,
            steps          = s.steps_done,
            final_dist     = final_dist,
            fell           = s.fell,
            upright        = upright,
            ms_per_step    = ms_per_step,
            grounding_hz   = grounding_hz,
            goal_source    = self.goal_source,
            vel_source     = self.vel_source,
            residual_action = s.use_residual,
            action_osc_std = osc_std,
            forward_disp   = forward_disp,
            scene_cfg      = scene_cfg,
            video_path     = video_path if (render_video and s.frames_ego) else None,
            stall_break_triggers = s.stall_trigger_count,
            avoid_bias_active_frac = (gp.avoid_cycles_active / gp.avoid_cycles_total
                                       if gp.avoid_cycles_total > 0 else 0.0),
        )


# ---------------------------------------------------------------------------
# Smoke test (run this file directly)
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    """Random-init smoke test for both Arch A and C (30 steps each)."""
    print("=" * 60)
    print("inferencer.py smoke test (random-init model, 30 steps)")
    print("=" * 60)

    from code.sim.scene import sample_scene, derive_rng

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


if __name__ == "__main__":
    _smoke_test()
    sys.exit(0)
