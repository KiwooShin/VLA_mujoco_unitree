"""
gen_maneuver_dataset.py — DART dataset generator for maneuver skill.

Task: "go straight, turn {left/right} after passing the {color}{shape}"

Same DART pattern as gen_dart_dataset.py:
  - Teacher computes CLEAN velocity command (via ManeuverExpert FSM).
  - DART: apply NOISY joint targets (clean + N(0, sigma^2)) to physics.
  - Clean teacher action saved as supervision label.
  - Gait phase [sin, cos] stored per frame.

Extended per-frame columns (beyond standard schema):
  - 'phase':            [sin_phi, cos_phi]  (2-d, same as gen_dart_dataset.py)
  - 'subgoal_index':    int  (FSM state: 0=STRAIGHT, 1=TURN_PHASE, 2=STRAIGHT2)
  - 'target_heading':   float (rad)
  - 'heading_err':      float (rad)
  - 'cos_target':       float
  - 'sin_target':       float
  - 'landmark_passed':  int (0/1)
  - 'vel_cmd':          [vx, vy, wz] (same as gen_dart_dataset.py)
  - 'goal':             [dist_to_landmark, cos_yaw_err, sin_yaw_err] while STRAIGHT;
                        [heading_err, cos_target, sin_target] during TURN_PHASE/STRAIGHT2

Usage
-----
# Smoke test (2 episodes):
MUJOCO_GL=egl python code/gen_maneuver_dataset.py generate \\
    --seed 0 --num-episodes 2 --noise 0.07 --maxsteps 200 \\
    --out dataset/maneuver_smoke

# Full dataset (150-200 episodes):
MUJOCO_GL=egl python code/gen_maneuver_dataset.py generate \\
    --seed 100 --num-episodes 200 --noise 0.07 --maxsteps 1400 \\
    --out dataset/maneuver
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np
import pandas as pd

_HERE: Path = Path(__file__).resolve().parent
_REPO: Path = _HERE.parent
sys.path.insert(0, str(_REPO))

from code.arena import build_arena
from code.gen_dart_dataset import GaitPhaseTracker, build_proprio
from code.maneuver_expert import ManeuverExpert, State
from code.maneuver_scene import HORIZON, SETTLE_STEPS, derive_rng, sample_maneuver_scene
from code.steer import goal_vec
from code.teacher import (
    CONTROL_DECIMATION, DEFAULT_ANGLES, KDS, KPS, NUM_ACTIONS, SIM_DT, WBCTeacher, _yaw_of,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPS: int         = int(round(1.0 / (SIM_DT * CONTROL_DECIMATION)))   # 50 Hz
FALL_HEIGHT: float = 0.50
HOLD_STEPS: int  = 5
PROPRIO_DIM: int = 55


def run_maneuver_episode(
    teacher:          WBCTeacher,
    scene_cfg:        dict,
    episode_idx:      int,
    global_frame_offset: int,
    noise_sigma:      float = 0.07,
    hard_maxsteps:    int = 1400,
    rng_noise:        np.random.Generator | None = None,
    verbose:          bool = False,
) -> dict | None:
    """Runs one DART episode for the maneuver task.

    The teacher computes a clean velocity command via the maneuver expert
    FSM; the resulting clean joint targets are perturbed with Gaussian noise
    before being applied to physics (DART), while the clean targets are
    saved as the supervision label.

    Args:
        teacher: WBCTeacher instance to step; its model/data are replaced
            with the arena built from `scene_cfg`.
        scene_cfg: Scene configuration dict from `sample_maneuver_scene`.
        episode_idx: Episode index recorded in the `episode_index` column
            and used in progress logs.
        global_frame_offset: Running frame count offset added to the
            per-frame `index` column.
        noise_sigma: Standard deviation of the joint-target noise applied
            for DART.
        hard_maxsteps: Maximum number of control steps before the episode
            ends (if the robot has not already fallen).
        rng_noise: Random generator used for the DART noise. If None, a
            fresh default generator is created.
        verbose: If True, prints periodic progress lines every 100 steps.

    Returns:
        A dict with keys `rows`, `success`, `n_steps`, `landmark_passed`,
        and `final_state`, or None if the robot fell during the episode.
    """
    if rng_noise is None:
        rng_noise = np.random.default_rng()

    # Build arena
    model = build_arena(scene_cfg)
    data  = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    # Inject compiled model into teacher
    teacher.model = model
    teacher.data  = data
    teacher._nj   = model.nq - 7
    teacher._pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    rx, ry   = scene_cfg["robot_xy"]
    ryaw     = scene_cfg["robot_yaw"]
    teacher.reset(pos_xy=(rx, ry), yaw=ryaw)

    nj        = teacher._nj
    task_desc = scene_cfg["instruction"]
    landmark_xy = scene_cfg["landmark_xy"]
    target_heading = scene_cfg["target_heading"]

    # Create maneuver expert
    expert = ManeuverExpert(scene_cfg)
    expert.reset()

    prev_action   = DEFAULT_ANGLES.copy()
    phase_tracker = GaitPhaseTracker()

    rows: list = []
    fallen    = False
    ep_success = False

    # Settle (zero vel, not logged)
    for _ in range(SETTLE_STEPS):
        teacher.step(vel_cmd=(0.0, 0.0, 0.0))
        if teacher.base_height < FALL_HEIGHT:
            return None

    step_t0 = time.time()

    for t in range(hard_maxsteps):
        bpos = data.qpos[0:3].copy()
        bxy  = bpos[:2]
        byaw = _yaw_of(data.qpos[3:7])

        # Expert FSM: compute clean velocity command and privileged state
        vel, priv = expert.step(bxy, byaw)

        # ---- DART step ----
        qpos_save = data.qpos.copy()
        qvel_save = data.qvel.copy()
        ctrl_save = data.ctrl.copy()

        # Teacher step with clean vel_cmd → get clean_targets + advance physics
        clean_targets = teacher.step(vel_cmd=tuple(vel))

        # Restore physics to pre-step state
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        data.ctrl[:] = ctrl_save
        mujoco.mj_forward(model, data)

        # Apply NOISY targets for CONTROL_DECIMATION substeps
        noise = rng_noise.normal(0.0, noise_sigma, size=NUM_ACTIONS).astype(np.float32)
        noisy_targets = clean_targets + noise

        for _ in range(CONTROL_DECIMATION):
            leg_tau = (
                (noisy_targets - data.qpos[7:7 + NUM_ACTIONS]) * KPS
                + (0.0 - data.qvel[6:6 + NUM_ACTIONS]) * KDS
            )
            data.ctrl[:NUM_ACTIONS] = leg_tau
            if nj > NUM_ACTIONS:
                n_up = nj - NUM_ACTIONS
                arm_tau = (
                    (0.0 - data.qpos[7 + NUM_ACTIONS:7 + nj]) * 100.0
                    + (0.0 - data.qvel[6 + NUM_ACTIONS:6 + nj]) * 0.5
                )
                data.ctrl[NUM_ACTIONS:nj] = arm_tau
            mujoco.mj_step(model, data)

        # Check fall
        if teacher.base_height < FALL_HEIGHT:
            fallen = True
            break

        # Gait phase
        q_lb = data.qpos[7:22].copy()
        sin_phi, cos_phi = phase_tracker.update(q_lb)

        # Build proprio from post-NOISY-step state
        proprio = build_proprio(data, noisy_targets)

        # Goal vector: egocentric to landmark (dist, cos_yaw_err, sin_yaw_err)
        lx, ly = landmark_xy
        robot_x, robot_y = float(data.qpos[0]), float(data.qpos[1])
        dx, dy = lx - robot_x, ly - robot_y
        dist_to_lm = math.hypot(dx, dy)
        bearing_lm = math.atan2(dy, dx)
        yaw_err_lm = math.atan2(math.sin(bearing_lm - byaw),
                                 math.cos(bearing_lm - byaw))
        gv = goal_vec(dist_to_lm, yaw_err_lm)

        global_idx = global_frame_offset + len(rows)

        rows.append({
            "frame_index":      t,
            "episode_index":    episode_idx,
            "index":            global_idx,
            "task_index":       0,
            "timestamp":        float(t) / FPS,
            "proprio":          proprio.tolist(),
            "action":           clean_targets.tolist(),
            "goal":             gv.tolist(),
            "vel_cmd":          vel.tolist(),
            "done":             0,
            "task_description": task_desc,
            "phase":            [float(sin_phi), float(cos_phi)],
            # Maneuver-specific privileged state
            "subgoal_index":    priv["subgoal_index"],
            "target_heading":   priv["target_heading"],
            "heading_err":      priv["heading_err"],
            "cos_target":       priv["cos_target"],
            "sin_target":       priv["sin_target"],
            "landmark_passed":  int(priv["landmark_passed"]),
        })

        prev_action = noisy_targets.copy()

        if verbose and t % 100 == 0:
            elapsed = time.time() - step_t0
            sps = (t + 1) / max(elapsed, 1e-6)
            state_name = State(priv["subgoal_index"]).name
            print(f"  [ep{episode_idx}] t={t} state={state_name:12s} "
                  f"dist_lm={dist_to_lm:.2f} heading_err={math.degrees(priv['heading_err']):.1f}° "
                  f"h={teacher.base_height:.3f} {sps:.1f}stp/s", flush=True)

    # Episode success: robot passed landmark AND completed the turn
    ep_success = expert.landmark_passed and (expert.state == State.STRAIGHT2)

    if fallen:
        return None

    return {
        "rows":          rows,
        "success":       ep_success,
        "n_steps":       len(rows),
        "landmark_passed": expert.landmark_passed,
        "final_state":   int(expert.state),
    }


def main_generate(args: argparse.Namespace) -> None:
    """Generates a maneuver DART dataset and writes it to `args.out`.

    Args:
        args: Parsed command-line arguments (see `main`), with fields
            `seed`, `num_episodes`, `noise`, `maxsteps`, `out`, and
            `verbose`.
    """
    t0 = time.time()

    out_path = Path(args.out)
    data_out = out_path / "data" / "chunk-000"
    meta_out = out_path / "meta"
    for d in [data_out, meta_out]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[maneuver_gen] episodes={args.num_episodes}  noise_sigma={args.noise}")
    print(f"[maneuver_gen] maxsteps={args.maxsteps}  seed={args.seed}")
    print(f"[maneuver_gen] NO rendering. Fast mode.")
    print(f"[maneuver_gen] Loading WBCTeacher...", flush=True)

    teacher = WBCTeacher()
    print(f"[maneuver_gen] Teacher loaded.", flush=True)

    noise_rng = np.random.default_rng(args.seed + 99999)

    tasks_map: dict[str, int] = {}
    episodes_meta: list = []
    all_proprio: list = []
    all_actions: list = []
    global_frame_idx = 0
    n_success  = 0
    n_fallen   = 0
    n_lm_passed = 0
    ep_written = 0

    for ep_i in range(args.num_episodes):
        rng       = derive_rng(args.seed, ep_i)
        scene_cfg = sample_maneuver_scene(rng)
        instr     = scene_cfg["instruction"]

        print(f"\n[maneuver ep {ep_i:03d}/{args.num_episodes}]  '{instr}'  "
              f"turn={scene_cfg['turn_direction']}  lm_dist={scene_cfg['objects'][0]['dist_from_robot']:.2f}m",
              flush=True)

        if instr not in tasks_map:
            tasks_map[instr] = len(tasks_map)

        ep_t0 = time.time()
        result = run_maneuver_episode(
            teacher             = teacher,
            scene_cfg           = scene_cfg,
            episode_idx         = ep_written,
            global_frame_offset = global_frame_idx,
            noise_sigma         = args.noise,
            hard_maxsteps       = args.maxsteps,
            rng_noise           = noise_rng,
            verbose             = args.verbose,
        )
        ep_elapsed = time.time() - ep_t0

        if result is None:
            n_fallen += 1
            print(f"  FALLEN (discarded)  t={ep_elapsed:.0f}s", flush=True)
            continue

        rows       = result["rows"]
        success    = result["success"]
        lm_passed  = result["landmark_passed"]
        task_idx   = tasks_map[instr]
        for r in rows:
            r["task_index"] = task_idx

        ep_name = f"episode_{ep_written:06d}"
        df      = pd.DataFrame(rows)
        df.to_parquet(data_out / f"{ep_name}.parquet", index=False)

        final_heading_err = float(rows[-1]["heading_err"]) if rows else 99.0
        episodes_meta.append({
            "episode_index":      ep_written,
            "task_index":         task_idx,
            "length":             len(rows),
            "success":            bool(success),
            "landmark_passed":    bool(lm_passed),
            "final_heading_err":  round(final_heading_err, 3),
            "turn_direction":     scene_cfg["turn_direction"],
            "seed":               args.seed,
            "ep_seed_index":      ep_i,
            "is_dart":            True,
            "tasks":              [instr],
        })

        all_proprio.extend([r["proprio"] for r in rows])
        all_actions.extend([r["action"]  for r in rows])
        global_frame_idx += len(rows)

        if success:
            n_success += 1
        if lm_passed:
            n_lm_passed += 1

        ep_sps = len(rows) / max(ep_elapsed, 1e-6)
        print(f"  ep {ep_i:03d} -> {ep_name}  steps={len(rows):4d}  "
              f"lm_passed={lm_passed}  success={success}  "
              f"heading_err={math.degrees(final_heading_err):.1f}°  "
              f"succ={n_success}/{ep_written+1}  "
              f"t={ep_elapsed:.1f}s ({ep_sps:.1f}stp/s)",
              flush=True)

        ep_written += 1

    # Stats
    arr_p = np.array(all_proprio, dtype=np.float32) if all_proprio else np.zeros((1, 55))
    arr_a = np.array(all_actions, dtype=np.float32) if all_actions else np.zeros((1, 15))

    def _stat(a: np.ndarray) -> dict[str, list[float]]:
        """Computes per-dimension mean/std/min/max for stats.json."""
        return {"mean": a.mean(0).tolist(), "std": (a.std(0)+1e-6).tolist(),
                "min": a.min(0).tolist(), "max": a.max(0).tolist()}

    stats = {"proprio": _stat(arr_p), "action": _stat(arr_a)}

    tasks_list = [{"task_index": v, "task": k}
                  for k, v in sorted(tasks_map.items(), key=lambda x: x[1])]

    elapsed = time.time() - t0
    success_rate = n_success / max(1, ep_written)
    lm_pass_rate = n_lm_passed / max(1, ep_written)

    info = {
        "codebase_version":  "maneuver_gen_v1",
        "fps":               FPS,
        "robot":             "unitree_g1_lowerbody",
        "task":              "maneuver",
        "seed":              args.seed,
        "noise_sigma":       args.noise,
        "maxsteps_per_ep":   args.maxsteps,
        "proprio_dim":       PROPRIO_DIM,
        "phase_dim":         2,
        "action_dim":        15,
        "total_episodes":    ep_written,
        "total_frames":      global_frame_idx,
        "n_fallen":          n_fallen,
        "n_attempted":       args.num_episodes,
        "success_rate":      round(success_rate, 3),
        "landmark_pass_rate": round(lm_pass_rate, 3),
        "total_time_s":      round(elapsed, 1),
        "eps_per_min":       round(ep_written / max(elapsed/60, 0.01), 2),
        "no_render":         True,
    }

    json.dump(info,  open(meta_out / "info.json",   "w"), indent=2)
    json.dump(stats, open(meta_out / "stats.json",  "w"), indent=2)
    with open(meta_out / "episodes.jsonl", "w") as f:
        for em in episodes_meta:
            f.write(json.dumps(em) + "\n")
    with open(meta_out / "tasks.jsonl", "w") as f:
        for tm in tasks_list:
            f.write(json.dumps(tm) + "\n")
    with open(meta_out / "manifest.jsonl", "w") as f:
        for i in range(ep_written):
            f.write(json.dumps({"path": f"data/chunk-000/episode_{i:06d}.parquet"}) + "\n")

    print(f"\n{'='*60}")
    print(f"MANEUVER GEN DONE: {ep_written} episodes, {global_frame_idx} frames")
    print(f"  success_rate (lm+turn): {success_rate:.3f}  ({n_success}/{ep_written})")
    print(f"  landmark_pass_rate:     {lm_pass_rate:.3f}  ({n_lm_passed}/{ep_written})")
    print(f"  fallen/discarded:       {n_fallen}")
    print(f"  total time: {elapsed:.0f}s  ({ep_written/max(elapsed/60,0.01):.1f} eps/min)")
    print(f"  output: {args.out}/")
    print(f"{'='*60}", flush=True)


def main() -> None:
    """Parses CLI arguments and dispatches to the `generate` subcommand.

    Raises:
        SystemExit: If no subcommand is given (prints help and exits 1).
    """
    ap = argparse.ArgumentParser(
        description="Maneuver DART dataset generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd")

    gen = sub.add_parser("generate", help="Generate maneuver DART episodes")
    gen.add_argument("--seed",         type=int, required=True)
    gen.add_argument("--num-episodes", type=int, required=True)
    gen.add_argument("--noise",        type=float, default=0.07)
    gen.add_argument("--maxsteps",     type=int, default=1400)
    gen.add_argument("--out",          required=True)
    gen.add_argument("--verbose",      action="store_true")

    args = ap.parse_args()

    if args.cmd == "generate":
        main_generate(args)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
