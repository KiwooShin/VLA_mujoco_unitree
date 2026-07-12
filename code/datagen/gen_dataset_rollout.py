"""
code/datagen/gen_dataset_rollout.py — Rollout/physics logic for gen_dataset.py.

Role: the pure-ish physics/rollout half of the clean (non-DART) teacher
dataset generator — split out of gen_dataset.py (RF-1) so the CLI/IO half
(code/datagen/gen_dataset.py) stays under the file-size budget.

Contents:
  - build_proprio        — 55-d proprio vector builder (shared row schema).
  - run_episode           — single-episode WBC teacher rollout + logging.
  - check_determinism     — two-pass determinism check (importable by tests).

See code/datagen/gen_dataset.py for the CLI, output layout, and parquet
schema documentation.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np

from code.arena import ArenaRenderer, EGO_H, EGO_W, build_arena
from code.scene import derive_rng, sample_scene
from code.steer import goal_vec, steer as steer_cmd
from code.teacher import CONTROL_DECIMATION, DEFAULT_ANGLES, SIM_DT, WBCTeacher, _yaw_of

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPS: int          = int(round(1.0 / (SIM_DT * CONTROL_DECIMATION)))  # 50
SETTLE_STEPS: int = 80       # steps to settle before logging begins (zero vel cmd)
FALL_HEIGHT: float = 0.50     # pelvis z below this → fallen → discard
HOLD_STEPS: int   = 5        # consecutive steps within stop_r → success (short: 5 steps = 0.1s)
MAXSTEPS: dict[str, int] = {"easy": 600, "demo": 1400}  # HARD per-episode step cap (never exceeded)

PROPRIO_DIM: int  = 55       # full proprio vector length


# ---------------------------------------------------------------------------
# Proprio builder
# ---------------------------------------------------------------------------
def build_proprio(data: mujoco.MjData, prev_action: np.ndarray) -> np.ndarray:
    """Builds the 55-d proprio vector from MuJoCo data.

    Layout:
      [0:15]   lower-body joint positions
      [15:30]  lower-body joint velocities
      [30:34]  base IMU quaternion [w,x,y,z]
      [34:37]  base angular velocity (rad/s)
      [37:40]  base linear velocity (proxy for accel) [vx,vy,vz] in world frame
      [40:55]  prev_action (15 joint targets)

    Args:
        data: MuJoCo data holding the current physics state.
        prev_action: Previous joint-target action (15-d), appended as part
            of the observation.

    Returns:
        A (55,) float32 array with the layout described above.
    """
    q_lb   = data.qpos[7:22].copy()      # 15 lower-body joint positions
    dq_lb  = data.qvel[6:21].copy()      # 15 lower-body joint velocities
    quat   = data.qpos[3:7].copy()       # [w, x, y, z]
    ang_v  = data.qvel[3:6].copy()       # angular velocity
    lin_v  = data.qvel[0:3].copy()       # linear velocity (proxy for accel)
    return np.concatenate([
        q_lb.astype(np.float32),
        dq_lb.astype(np.float32),
        quat.astype(np.float32),
        ang_v.astype(np.float32),
        lin_v.astype(np.float32),
        prev_action.astype(np.float32),
    ])  # shape (55,)


# ---------------------------------------------------------------------------
# Single episode rollout
# ---------------------------------------------------------------------------
def run_episode(
    teacher: WBCTeacher,
    scene_cfg: dict,
    episode_idx: int,
    global_frame_offset: int,
    noise_sigma: float = 0.0,
    render_tp: bool = False,
    render_depth: bool = True,
    verbose: bool = False,
) -> dict | None:
    """Drives the WBC teacher to the target object and collects data.

    Speed notes:
      - render_tp=False (default): skips third-person rendering (~530ms/step
        saved). Set True only for the demo side-by-side video.
      - render_depth=True (default): renders depth for grounding. Set False
        to further speed up (pure-RGB collection).

    Args:
        teacher: WBCTeacher instance to step; its model/data are replaced
            with the arena built from `scene_cfg`.
        scene_cfg: Scene configuration dict from `sample_scene`.
        episode_idx: Episode index recorded in the `episode_index` column.
        global_frame_offset: Running frame count offset added to the
            per-frame `index` column.
        noise_sigma: DART-style exec-noise std (rad) added to the velocity
            command. 0.0 disables noise.
        render_tp: If True, also renders third-person frames.
        render_depth: If True, also renders depth for grounding.
        verbose: If True, prints periodic progress lines every 50 steps.

    Returns:
        A dict with keys:
            rows: list of row dicts (one per control step).
            ego_rgb_seq: list of np.uint8 (H,W,3) ego RGB frames.
            ego_dep_seq: list of np.float32 (H,W) depth frames, or an empty
                list if `render_depth` is False.
            tp_rgb_seq: list of np.uint8 (H,W,3) third-person frames, or an
                empty list if `render_tp` is False.
            reached: bool, whether the episode reached the goal.
            n_steps: int, number of logged steps.
        Returns None if the episode fell (discard) or exceeded the
        MAXSTEPS cap (discard).
    """
    difficulty   = scene_cfg.get("difficulty", "easy")
    hard_maxsteps = MAXSTEPS.get(difficulty, 600)

    model = build_arena(scene_cfg)
    data  = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    renderer = ArenaRenderer(model, EGO_W, EGO_H)
    tp_cam   = renderer.make_tp_cam() if render_tp else None

    # Inject the compiled model into the teacher (replace model/data in-place)
    teacher.model = model
    teacher.data  = data
    teacher._nj   = model.nq - 7
    teacher._pelvis_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )

    # Reset to scene's robot start pose
    rx, ry   = scene_cfg["robot_xy"]
    ryaw     = scene_cfg["robot_yaw"]
    teacher.reset(pos_xy=(rx, ry), yaw=ryaw)

    stop_r  = scene_cfg["stop_r"]
    horizon = min(scene_cfg["horizon"], hard_maxsteps)  # HARD CAP — never infinite
    tgt     = scene_cfg["objects"][scene_cfg["target_index"]]
    txy     = (tgt["x"], tgt["y"])
    task_description = scene_cfg["instruction"]

    prev_action = DEFAULT_ANGLES.copy()

    ego_rgb_seq: list = []
    ego_dep_seq: list = []
    tp_rgb_seq: list  = []
    rows: list        = []

    reached     = False
    hold_count  = 0
    fallen      = False

    # ---- Settle (zero vel cmd, not logged) ----
    for settle_i in range(SETTLE_STEPS):
        teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        if teacher.base_height < FALL_HEIGHT:
            renderer.close()
            return None   # fell during settle → discard

    # ---- Main rollout ----
    step_t0 = time.time()
    for t in range(horizon):
        bpos = teacher.base_pos         # [x, y, z]
        bxy  = bpos[:2]
        byaw = teacher.base_yaw

        # Steering
        vel, dist, yaw_err = steer_cmd(bxy, byaw, txy, stop_r)

        # Teacher step (DART: perturb execution, log clean target)
        if noise_sigma > 0.0:
            # Perturb the vel_cmd slightly to induce state diversity (DART-style)
            noisy_vel = vel + noise_sigma * np.random.randn(3).astype(np.float32)
            clean_targets = teacher.step(vel_cmd=tuple(noisy_vel))
        else:
            clean_targets = teacher.step(vel_cmd=tuple(vel))

        # Check fall
        if teacher.base_height < FALL_HEIGHT:
            fallen = True
            break

        # Render ego RGB (always)
        rgb, depth, intr = renderer.render_ego(data, byaw,
                                               render_depth=render_depth)

        # Render third-person (only if requested)
        tp_rgb = None
        if render_tp and tp_cam is not None:
            renderer.update_tp_cam(tp_cam, data)
            tp_rgb = renderer.render_tp(data, tp_cam)

        # Build proprio
        proprio = build_proprio(data, prev_action)

        # Goal vector
        gv = goal_vec(dist, yaw_err)

        # Log row
        global_idx = global_frame_offset + len(rows)
        done_flag  = int(dist < stop_r)

        rows.append({
            "frame_index":      t,
            "episode_index":    episode_idx,
            "index":            global_idx,
            "task_index":       0,  # filled by caller
            "timestamp":        float(t) / FPS,
            "proprio":          proprio.tolist(),
            "action":           clean_targets.tolist(),
            "goal":             gv.tolist(),
            "vel_cmd":          vel.tolist(),
            "done":             done_flag,
            "task_description": task_description,
        })

        ego_rgb_seq.append(rgb)
        if render_depth:
            ego_dep_seq.append(depth)
        if render_tp and tp_rgb is not None:
            tp_rgb_seq.append(tp_rgb)

        prev_action = clean_targets.copy()

        # Verbose per-step progress
        if verbose and (t % 50 == 0 or dist < stop_r):
            elapsed = time.time() - step_t0
            sps = (t + 1) / max(elapsed, 0.001)
            print(f"    [ep{episode_idx}] t={t:4d}/{horizon}  dist={dist:.2f}m  "
                  f"h={teacher.base_height:.3f}  reached={reached}  "
                  f"{sps:.1f}stp/s", flush=True)

        # Termination: reached goal
        if dist < stop_r:
            reached    = True
            hold_count += 1
        else:
            # Don't reset hold_count — allow non-consecutive approach
            pass
        if reached and hold_count >= HOLD_STEPS:
            break

    renderer.close()

    if fallen:
        return None

    return {
        "rows":        rows,
        "ego_rgb_seq": ego_rgb_seq,
        "ego_dep_seq": ego_dep_seq,
        "tp_rgb_seq":  tp_rgb_seq,
        "reached":     reached,
        "n_steps":     len(rows),
    }


# ---------------------------------------------------------------------------
# Determinism check helper (importable by test scripts)
# ---------------------------------------------------------------------------
def check_determinism(difficulty: str, seed: int, n_check: int = 3) -> bool:
    """Runs two independent passes and verifies identical parquet outputs.

    Args:
        difficulty: Scene difficulty, forwarded to `sample_scene`.
        seed: Base seed forwarded to `derive_rng`.
        n_check: Number of episodes to check per pass.

    Returns:
        True if deterministic (identical actions across both passes),
        False otherwise.
    """
    import shutil
    import tempfile

    results = []
    for run in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            teacher = WBCTeacher()
            rows_all = []
            for ep_i in range(n_check):
                rng       = derive_rng(seed, ep_i)
                scene_cfg = sample_scene(rng, difficulty)
                result    = run_episode(
                    teacher             = teacher,
                    scene_cfg           = scene_cfg,
                    episode_idx         = ep_i,
                    global_frame_offset = 0,
                    noise_sigma         = 0.0,
                    render_tp           = False,  # skip tp for speed
                    render_depth        = False,  # skip depth for speed
                )
                if result is not None:
                    rows_all.append(result["rows"])
            results.append(rows_all)

    # Compare
    if len(results[0]) != len(results[1]):
        print(f"[determinism] FAIL: different episode counts {len(results[0])} vs {len(results[1])}")
        return False
    for ep_i, (r0, r1) in enumerate(zip(results[0], results[1])):
        if len(r0) != len(r1):
            print(f"[determinism] FAIL ep{ep_i}: different step counts {len(r0)} vs {len(r1)}")
            return False
        for step_i, (s0, s1) in enumerate(zip(r0, r1)):
            a0 = np.array(s0["action"])
            a1 = np.array(s1["action"])
            if not np.allclose(a0, a1, atol=1e-5):
                print(f"[determinism] FAIL ep{ep_i} step{step_i}: action mismatch")
                return False
    print(f"[determinism] PASS: {n_check} episodes identical across 2 runs")
    return True
