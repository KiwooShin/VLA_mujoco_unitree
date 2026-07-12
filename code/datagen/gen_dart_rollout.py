"""
code/datagen/gen_dart_rollout.py — Single DART episode rollout.

Role: split out of gen_dart_dataset.py (RF-1) — the DART episode-stepping
logic (teacher computes clean action; noisy action is executed; clean
action is stored as the supervision label). See the module docstring of
code/datagen/gen_dart_dataset.py for the DART algorithm description.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from code.arena import build_arena
from code.datagen.gen_dart_phase import GaitPhaseTracker, build_proprio
from code.steer import goal_vec, steer as steer_cmd
from code.teacher import (
    CONTROL_DECIMATION, DEFAULT_ANGLES, KDS, KPS, NUM_ACTIONS, SIM_DT, WBCTeacher, _yaw_of,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPS: int          = int(round(1.0 / (SIM_DT * CONTROL_DECIMATION)))   # 50 Hz
SETTLE_STEPS: int = 80
FALL_HEIGHT: float = 0.50
HOLD_STEPS: int   = 5
PROPRIO_DIM: int  = 55


# ---------------------------------------------------------------------------
# Single DART episode — uses teacher.step() then replaces physics substeps
# ---------------------------------------------------------------------------
def run_dart_episode(
    teacher: WBCTeacher,
    scene_cfg: dict,
    episode_idx: int,
    global_frame_offset: int,
    noise_sigma: float = 0.07,
    hard_maxsteps: int = 300,
    rng_noise: np.random.Generator | None = None,
    verbose: bool = False,
) -> dict | None:
    """Runs one DART episode: teacher determines clean action; noisy action is executed.

    Strategy:
      1. Save physics state before each step.
      2. Call teacher.step() normally → clean_targets + physics is advanced with clean action.
      3. Restore physics state to the saved snapshot.
      4. Apply noisy_targets = clean_targets + noise via PD for CONTROL_DECIMATION substeps.
      5. Teacher's internal state (obs history) now lags one step — we correct it by
         injecting the actual physical state into the next step's obs. This is handled
         naturally since teacher._build_single_obs() reads from teacher.data which is
         the actual current physics state.

    Supervision: proprio is built from the NOISY-executed physics state;
                 action label = clean_targets.

    Args:
        teacher: WBCTeacher instance to step; its model/data are replaced
            with the arena built from `scene_cfg`.
        scene_cfg: Scene configuration dict from `sample_scene`.
        episode_idx: Episode index recorded in the `episode_index` column.
        global_frame_offset: Running frame count offset added to the
            per-frame `index` column.
        noise_sigma: Standard deviation of the joint-target noise applied
            for DART.
        hard_maxsteps: Maximum number of control steps before the episode
            ends (if not already reached or fallen).
        rng_noise: Random generator used for the DART noise. If None, a
            fresh default generator is created.
        verbose: If True, prints periodic progress lines every 50 steps.

    Returns:
        A dict with keys `rows` (including the `phase` column), `reached`,
        and `n_steps`, or None if the robot fell during the episode.
    """
    if rng_noise is None:
        rng_noise = np.random.default_rng()

    difficulty = scene_cfg.get("difficulty", "easy")
    episode_maxsteps = hard_maxsteps

    model = build_arena(scene_cfg)
    data  = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    # Inject compiled model into teacher
    teacher.model = model
    teacher.data  = data
    teacher._nj   = model.nq - 7
    teacher._pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    rx, ry = scene_cfg["robot_xy"]
    ryaw   = scene_cfg["robot_yaw"]
    teacher.reset(pos_xy=(rx, ry), yaw=ryaw)

    nj        = teacher._nj
    stop_r    = scene_cfg["stop_r"]
    tgt       = scene_cfg["objects"][scene_cfg["target_index"]]
    txy       = (tgt["x"], tgt["y"])
    task_desc = scene_cfg["instruction"]

    prev_action   = DEFAULT_ANGLES.copy()
    phase_tracker = GaitPhaseTracker()

    rows: list = []
    reached    = False
    hold_count = 0
    fallen     = False

    # Settle (zero vel, not logged) — use teacher normally
    for _ in range(SETTLE_STEPS):
        teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        if teacher.base_height < FALL_HEIGHT:
            return None

    step_t0 = time.time()

    for t in range(episode_maxsteps):
        bpos = teacher.base_pos
        bxy  = bpos[:2]
        byaw = teacher.base_yaw

        # Steering
        vel, dist, yaw_err = steer_cmd(bxy, byaw, txy, stop_r)

        # ---- DART step ----
        # Save physics state snapshot (qpos + qvel)
        qpos_save = data.qpos.copy()
        qvel_save = data.qvel.copy()
        ctrl_save = data.ctrl.copy()

        # Call teacher.step() with clean vel_cmd → get clean_targets + advance physics
        clean_targets = teacher.step(vel_cmd=tuple(vel))

        # Restore physics to pre-step state
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        data.ctrl[:] = ctrl_save
        mujoco.mj_forward(model, data)   # recompute derived quantities

        # Now apply NOISY targets for CONTROL_DECIMATION substeps
        noise = rng_noise.normal(0.0, noise_sigma, size=NUM_ACTIONS).astype(np.float32)
        noisy_targets = clean_targets + noise

        for _ in range(CONTROL_DECIMATION):
            # Lower-body PD
            leg_tau = (
                (noisy_targets - data.qpos[7:7 + NUM_ACTIONS]) * KPS
                + (0.0 - data.qvel[6:6 + NUM_ACTIONS]) * KDS
            )
            data.ctrl[:NUM_ACTIONS] = leg_tau
            # Upper-body hold
            if nj > NUM_ACTIONS:
                n_up = nj - NUM_ACTIONS
                arm_tau = (
                    (0.0 - data.qpos[7 + NUM_ACTIONS:7 + nj]) * 100.0
                    + (0.0 - data.qvel[6 + NUM_ACTIONS:6 + nj]) * 0.5
                )
                data.ctrl[NUM_ACTIONS:nj] = arm_tau
            mujoco.mj_step(model, data)

        # Check fall from the NOISY-executed state
        if teacher.base_height < FALL_HEIGHT:
            fallen = True
            break

        # Gait phase from current joint positions
        q_lb = data.qpos[7:22].copy()
        sin_phi, cos_phi = phase_tracker.update(q_lb)

        # Build proprio from post-NOISY-step physics state
        proprio = build_proprio(data, noisy_targets)

        # Goal vector
        gv = goal_vec(dist, yaw_err)

        global_idx = global_frame_offset + len(rows)
        done_flag  = int(dist < stop_r)

        rows.append({
            "frame_index":      t,
            "episode_index":    episode_idx,
            "index":            global_idx,
            "task_index":       0,
            "timestamp":        float(t) / FPS,
            "proprio":          proprio.tolist(),
            "action":           clean_targets.tolist(),   # CLEAN supervision label
            "goal":             gv.tolist(),
            "vel_cmd":          vel.tolist(),
            "done":             done_flag,
            "task_description": task_desc,
            "phase":            [float(sin_phi), float(cos_phi)],
        })

        prev_action = noisy_targets.copy()

        if verbose and t % 50 == 0:
            elapsed = time.time() - step_t0
            sps = (t + 1) / max(elapsed, 1e-6)
            print(f"    [dart ep{episode_idx}] t={t}/{episode_maxsteps} "
                  f"dist={dist:.2f} h={teacher.base_height:.3f} {sps:.1f}stp/s", flush=True)

        if dist < stop_r:
            reached = True
            hold_count += 1
        if reached and hold_count >= HOLD_STEPS:
            break

    if fallen:
        return None

    return {
        "rows":    rows,
        "reached": reached,
        "n_steps": len(rows),
    }
