"""
gen_dart_dataset.py — Fast DART dataset generator (no-render, gait-phase stored).

DART (Laskey'17): on each step, the teacher computes a CLEAN action, we record it
as the supervision label, but we execute a NOISY action (clean + Gaussian noise)
to move the physics. This diversifies the state distribution the model sees during
training, covering recovery states that clean rollouts never visit.

Implementation detail:
  teacher.step() runs ONNX + physics substeps with clean targets internally.
  We intercept by calling step() to get clean_targets, then immediately undo
  the physics advance (restore qpos/qvel) and re-run substeps with noisy_targets.
  This gives: diverse physics states (noisy) + correct supervision (clean).

Key differences from gen_dataset.py:
  - --no-render by default (zero placeholder, no video writes)
  - Action noise applied to EXECUTED targets (noisy); clean target stored as label
  - Gait phase [sin(phi), cos(phi)] computed from foot-contact/ankle-pitch oscillation
    and stored in each row as 'phase' column (2-d)

Phase extraction:
  Use left ankle pitch zero-crossing counter as a simple gait phase oscillator.
  Phase convention: phi=0 at positive zero crossing of (ankle_pitch - default).

Subcommands:
  generate   — run DART episodes
  add-phase  — retroactively add phase column to an existing clean dataset parquet
  combine    — merge clean (with phase) + DART into one combined dataset

Usage
-----
# 1. Generate 200 DART episodes (fast: no render)
MUJOCO_GL=egl python code/gen_dart_dataset.py generate \\
    --difficulty easy --seed 42 --num-episodes 200 --noise 0.07 \\
    --maxsteps 300 --out dataset/dart_easy

# 2. Add phase to existing clean dataset
python code/gen_dart_dataset.py add-phase \\
    --in-dir dataset/easy_train80 --out-dir dataset/clean_with_phase

# 3. Combine into one training dataset
python code/gen_dart_dataset.py combine \\
    --clean-dir dataset/clean_with_phase \\
    --dart-dir dataset/dart_easy \\
    --out dataset/dart_combined
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

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from code.teacher import (
    WBCTeacher, _yaw_of, DEFAULT_ANGLES, KPS, KDS,
    NUM_ACTIONS, SIM_DT, CONTROL_DECIMATION,
)
from code.arena import build_arena
from code.scene import sample_scene, derive_rng
from code.steer import steer as steer_cmd, goal_vec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPS           = int(round(1.0 / (SIM_DT * CONTROL_DECIMATION)))   # 50 Hz
SETTLE_STEPS  = 80
FALL_HEIGHT   = 0.50
HOLD_STEPS    = 5
PROPRIO_DIM   = 55

# Left ankle pitch index in lower-body joint positions (index 4, per dataset.md)
LEFT_ANKLE_PITCH_IDX  = 4    # in qpos[7:22]
LEFT_ANKLE_DEFAULT    = -0.2  # from teacher.py default angles


# ---------------------------------------------------------------------------
# Phase extractor — running zero-crossing counter on ankle pitch oscillation
# ---------------------------------------------------------------------------
class GaitPhaseTracker:
    """
    Tracks gait phase phi in [0, 2pi] using left ankle pitch zero-crossings.

    The ankle pitch oscillates sinusoidally during walking. Positive-going
    zero-crossings of (ankle_pitch - default) mark the start of each cycle.
    Phase advances at a fixed estimated frequency between crossings.

    Output: (sin(phi), cos(phi)) — 2-d unit-circle encoding.
    """

    def __init__(self, freq_hz: float = 1.8):
        self._phi: float = 0.0
        self._prev_q: float = 0.0
        self._initialized: bool = False
        self._freq_hz: float = freq_hz   # typical walking gait frequency
        self._dt: float = SIM_DT * CONTROL_DECIMATION  # 0.02 s

    def update(self, q_lb: np.ndarray) -> tuple[float, float]:
        """
        Update phase from lower-body joint positions.

        q_lb : (15,) joint positions (same order as dataset).
        Returns: (sin_phi, cos_phi)
        """
        q_ankle = float(q_lb[LEFT_ANKLE_PITCH_IDX]) - LEFT_ANKLE_DEFAULT

        if not self._initialized:
            self._prev_q = q_ankle
            self._initialized = True
            return (0.0, 1.0)

        # Advance phase by estimated frequency
        self._phi += 2.0 * math.pi * self._freq_hz * self._dt

        # On positive zero-crossing: reset to 0 (start of new cycle)
        if self._prev_q < 0.0 and q_ankle >= 0.0:
            self._phi = 0.0

        self._prev_q = q_ankle
        self._phi = self._phi % (2.0 * math.pi)

        return (math.sin(self._phi), math.cos(self._phi))


# ---------------------------------------------------------------------------
# Proprio builder (identical to gen_dataset.py)
# ---------------------------------------------------------------------------
def build_proprio(data: mujoco.MjData, prev_action: np.ndarray) -> np.ndarray:
    q_lb   = data.qpos[7:22].copy()
    dq_lb  = data.qvel[6:21].copy()
    quat   = data.qpos[3:7].copy()
    ang_v  = data.qvel[3:6].copy()
    lin_v  = data.qvel[0:3].copy()
    return np.concatenate([
        q_lb.astype(np.float32),
        dq_lb.astype(np.float32),
        quat.astype(np.float32),
        ang_v.astype(np.float32),
        lin_v.astype(np.float32),
        prev_action.astype(np.float32),
    ])   # shape (55,)


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
    rng_noise: np.random.Generator = None,
    verbose: bool = False,
) -> dict | None:
    """
    DART episode: teacher determines clean action; noisy action is executed.

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

    Returns dict with rows (including 'phase' column), or None if fallen.
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


# ---------------------------------------------------------------------------
# Add gait phase to clean dataset (post-processing existing parquet)
# ---------------------------------------------------------------------------
def add_phase_to_clean_dataset(in_dir: str, out_dir: str) -> int:
    """
    Read each episode parquet from in_dir, compute gait phase from proprio,
    write new parquet with 'phase' column to out_dir.

    Returns total frames processed.
    """
    import shutil

    in_path  = Path(in_dir)
    out_path = Path(out_dir)
    data_out = out_path / "data" / "chunk-000"
    meta_out = out_path / "meta"
    data_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    # Copy meta files
    for meta_f in (in_path / "meta").glob("*"):
        shutil.copy2(meta_f, meta_out / meta_f.name)
    print(f"[phase] Copied meta from {in_path}/meta → {meta_out}")

    total_frames = 0
    chunk_dir = in_path / "data" / "chunk-000"
    for parq_f in sorted(chunk_dir.glob("episode_*.parquet")):
        df = pd.read_parquet(parq_f)

        tracker = GaitPhaseTracker()
        phases = []
        for _, row in df.iterrows():
            q_lb = np.array(row["proprio"][:15], dtype=np.float32)
            sin_phi, cos_phi = tracker.update(q_lb)
            phases.append([float(sin_phi), float(cos_phi)])

        df["phase"] = phases
        out_parq = data_out / parq_f.name
        df.to_parquet(out_parq, index=False)
        total_frames += len(df)
        print(f"  [phase] {parq_f.name}: {len(df)} frames", flush=True)

    return total_frames


# ---------------------------------------------------------------------------
# Combine DART + clean datasets into a single merged parquet dataset
# ---------------------------------------------------------------------------
def combine_datasets(clean_dir: str, dart_dir: str, out_dir: str) -> dict:
    """
    Merge clean (with phase) + DART datasets into one combined dataset.
    Re-indexes episode_index and global index.
    """
    out_path  = Path(out_dir)
    data_out  = out_path / "data" / "chunk-000"
    meta_out  = out_path / "meta"
    data_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    clean_chunk = Path(clean_dir) / "data" / "chunk-000"
    dart_chunk  = Path(dart_dir)  / "data" / "chunk-000"

    all_dfs = []

    # Load clean episodes
    for f in sorted(clean_chunk.glob("episode_*.parquet")):
        df = pd.read_parquet(f)
        # Ensure 'phase' column exists
        if "phase" not in df.columns:
            tracker = GaitPhaseTracker()
            phases = []
            for _, row in df.iterrows():
                q_lb = np.array(row["proprio"][:15], dtype=np.float32)
                sin_phi, cos_phi = tracker.update(q_lb)
                phases.append([float(sin_phi), float(cos_phi)])
            df["phase"] = phases
        all_dfs.append(df)

    n_clean = len(all_dfs)
    print(f"[combine] Loaded {n_clean} clean episodes from {clean_dir}")

    # Load DART episodes
    dart_count = 0
    for f in sorted(dart_chunk.glob("episode_*.parquet")):
        df = pd.read_parquet(f)
        all_dfs.append(df)
        dart_count += 1

    print(f"[combine] Loaded {dart_count} DART episodes from {dart_dir}")

    # Re-index episodes
    global_idx  = 0
    out_files   = []
    ep_meta_list = []
    tasks_map: dict[str, int] = {}
    all_actions = []
    all_proprio = []

    for ep_i, df in enumerate(all_dfs):
        df = df.copy()
        df["episode_index"] = ep_i
        df["index"]         = range(global_idx, global_idx + len(df))
        df["frame_index"]   = range(len(df))

        task_desc = str(df["task_description"].iloc[0])
        if task_desc not in tasks_map:
            tasks_map[task_desc] = len(tasks_map)
        df["task_index"] = tasks_map[task_desc]

        out_parq = data_out / f"episode_{ep_i:06d}.parquet"
        df.to_parquet(out_parq, index=False)
        out_files.append(f"data/chunk-000/episode_{ep_i:06d}.parquet")

        final_dist = float(df["goal"].iloc[-1][0]) if len(df) > 0 else 99.0
        reached    = bool(df["done"].iloc[-1] == 1) if len(df) > 0 else False
        ep_meta_list.append({
            "episode_index":   ep_i,
            "task_index":      tasks_map[task_desc],
            "length":          len(df),
            "success":         reached,
            "final_goal_dist": round(final_dist, 3),
            "is_dart":         ep_i >= n_clean,
            "tasks":           [task_desc],
        })

        all_actions.extend(df["action"].tolist())
        all_proprio.extend(df["proprio"].tolist())
        global_idx += len(df)

    # Stats
    arr_a = np.array(all_actions, dtype=np.float32) if all_actions else np.zeros((1, 15))
    arr_p = np.array(all_proprio, dtype=np.float32) if all_proprio else np.zeros((1, 55))

    def _stat(a):
        return {"mean": a.mean(0).tolist(), "std": (a.std(0) + 1e-6).tolist(),
                "min": a.min(0).tolist(), "max": a.max(0).tolist()}

    stats = {"proprio": _stat(arr_p), "action": _stat(arr_a)}

    info = {
        "codebase_version":  "dart+phase",
        "fps":               FPS,
        "robot":             "unitree_g1_lowerbody",
        "total_episodes":    len(all_dfs),
        "n_clean_episodes":  n_clean,
        "n_dart_episodes":   dart_count,
        "total_frames":      global_idx,
        "proprio_dim":       PROPRIO_DIM,
        "phase_dim":         2,
        "action_dim":        15,
    }

    tasks_list = [{"task_index": v, "task": k}
                  for k, v in sorted(tasks_map.items(), key=lambda x: x[1])]

    json.dump(info,  open(meta_out / "info.json",   "w"), indent=2)
    json.dump(stats, open(meta_out / "stats.json",  "w"), indent=2)
    with open(meta_out / "episodes.jsonl", "w") as f:
        for em in ep_meta_list:
            f.write(json.dumps(em) + "\n")
    with open(meta_out / "tasks.jsonl", "w") as f:
        for tm in tasks_list:
            f.write(json.dumps(tm) + "\n")
    with open(meta_out / "manifest.jsonl", "w") as f:
        for fp in out_files:
            f.write(json.dumps({"path": fp}) + "\n")

    print(f"[combine] Done: {len(all_dfs)} episodes, {global_idx} frames → {out_dir}")
    return info


# ---------------------------------------------------------------------------
# Main DART generation
# ---------------------------------------------------------------------------
def main_generate(args):
    t0 = time.time()

    out_path = Path(args.out)
    data_out = out_path / "data" / "chunk-000"
    meta_out = out_path / "meta"
    for d in [data_out, meta_out]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[dart_gen] difficulty={args.difficulty}  episodes={args.num_episodes}")
    print(f"[dart_gen] noise_sigma={args.noise}  maxsteps={args.maxsteps}")
    print(f"[dart_gen] NO rendering. Storing zero image placeholders in meta only.")
    print(f"[dart_gen] Loading WBCTeacher...", flush=True)

    teacher = WBCTeacher()
    print(f"[dart_gen] Teacher loaded.", flush=True)

    noise_rng = np.random.default_rng(args.seed + 10000)

    tasks_map: dict[str, int] = {}
    episodes_meta: list = []
    all_proprio: list = []
    all_actions: list = []
    global_frame_idx = 0
    n_success  = 0
    n_fallen   = 0
    ep_written = 0

    for ep_i in range(args.num_episodes):
        rng       = derive_rng(args.seed, ep_i)
        scene_cfg = sample_scene(rng, args.difficulty)
        instr     = scene_cfg["instruction"]
        tgt_obj   = scene_cfg["objects"][scene_cfg["target_index"]]

        print(f"\n[dart ep {ep_i:03d}/{args.num_episodes}]  '{instr}'  "
              f"dist={tgt_obj['dist_from_robot']:.2f}m", flush=True)

        if instr not in tasks_map:
            tasks_map[instr] = len(tasks_map)

        ep_t0 = time.time()
        result = run_dart_episode(
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

        rows     = result["rows"]
        reached  = result["reached"]
        task_idx = tasks_map[instr]
        for r in rows:
            r["task_index"] = task_idx

        ep_name = f"episode_{ep_written:06d}"
        df      = pd.DataFrame(rows)
        df.to_parquet(data_out / f"{ep_name}.parquet", index=False)

        final_dist = float(rows[-1]["goal"][0]) if rows else 99.0
        episodes_meta.append({
            "episode_index":   ep_written,
            "task_index":      task_idx,
            "length":          len(rows),
            "success":         bool(reached),
            "final_goal_dist": round(final_dist, 3),
            "difficulty":      args.difficulty,
            "seed":            args.seed,
            "ep_seed_index":   ep_i,
            "is_dart":         True,
            "tasks":           [instr],
        })

        all_proprio.extend([r["proprio"] for r in rows])
        all_actions.extend([r["action"]  for r in rows])
        global_frame_idx += len(rows)

        if reached:
            n_success += 1

        ep_sps = len(rows) / max(ep_elapsed, 1e-6)
        print(f"  dart {ep_i:03d} -> {ep_name}  steps={len(rows):4d}  "
              f"reached={reached}  dist={final_dist:.2f}m  "
              f"succ={n_success}/{ep_written+1}  "
              f"t={ep_elapsed:.1f}s ({ep_sps:.1f}stp/s)",
              flush=True)

        ep_written += 1

    # Stats
    arr_p = np.array(all_proprio, dtype=np.float32) if all_proprio else np.zeros((1, 55))
    arr_a = np.array(all_actions, dtype=np.float32) if all_actions else np.zeros((1, 15))

    def _stat(a):
        return {"mean": a.mean(0).tolist(), "std": (a.std(0)+1e-6).tolist(),
                "min": a.min(0).tolist(), "max": a.max(0).tolist()}

    stats = {"proprio": _stat(arr_p), "action": _stat(arr_a)}

    tasks_list = [{"task_index": v, "task": k}
                  for k, v in sorted(tasks_map.items(), key=lambda x: x[1])]

    elapsed = time.time() - t0
    success_rate = n_success / max(1, ep_written)
    info = {
        "codebase_version": "dart_gen_v1",
        "fps":              FPS,
        "robot":            "unitree_g1_lowerbody",
        "difficulty":       args.difficulty,
        "seed":             args.seed,
        "noise_sigma":      args.noise,
        "maxsteps_per_ep":  args.maxsteps,
        "proprio_dim":      PROPRIO_DIM,
        "phase_dim":        2,
        "action_dim":       15,
        "total_episodes":   ep_written,
        "total_frames":     global_frame_idx,
        "n_fallen":         n_fallen,
        "n_attempted":      args.num_episodes,
        "success_rate":     round(success_rate, 3),
        "total_time_s":     round(elapsed, 1),
        "eps_per_min":      round(ep_written / max(elapsed/60, 0.01), 2),
        "no_render":        True,
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
    print(f"DART GEN DONE: {ep_written} episodes, {global_frame_idx} frames")
    print(f"  success_rate: {success_rate:.3f}  ({n_success}/{ep_written})")
    print(f"  fallen/discarded: {n_fallen}")
    print(f"  total time: {elapsed:.0f}s  ({ep_written/max(elapsed/60,0.01):.1f} eps/min)")
    print(f"  output: {args.out}/")
    print(f"{'='*60}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="DART dataset generator (no-render, gait-phase aware)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd")

    # Subcommand: generate
    gen = sub.add_parser("generate", help="Generate DART episodes")
    gen.add_argument("--difficulty", choices=["easy", "demo"], required=True)
    gen.add_argument("--seed",         type=int, required=True)
    gen.add_argument("--num-episodes", type=int, required=True)
    gen.add_argument("--noise",        type=float, default=0.07,
                     help="Action noise std (rad). Applied to executed joint targets.")
    gen.add_argument("--maxsteps",     type=int, default=300,
                     help="Hard step cap per episode (shorter=faster; 300 gives ~120s/ep)")
    gen.add_argument("--out",          required=True, help="Output dataset dir")
    gen.add_argument("--verbose",      action="store_true")

    # Subcommand: add-phase
    addph = sub.add_parser("add-phase",
                           help="Add gait phase column to existing clean dataset")
    addph.add_argument("--in-dir",  required=True)
    addph.add_argument("--out-dir", required=True)

    # Subcommand: combine
    cmb = sub.add_parser("combine",
                         help="Combine clean+DART into one dataset")
    cmb.add_argument("--clean-dir", required=True)
    cmb.add_argument("--dart-dir",  required=True)
    cmb.add_argument("--out",       required=True)

    args = ap.parse_args()

    if args.cmd == "generate":
        main_generate(args)
    elif args.cmd == "add-phase":
        n = add_phase_to_clean_dataset(args.in_dir, args.out_dir)
        print(f"[add-phase] Done: {n} frames processed → {args.out_dir}")
    elif args.cmd == "combine":
        info = combine_datasets(args.clean_dir, args.dart_dir, args.out)
        print(json.dumps(info, indent=2))
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
