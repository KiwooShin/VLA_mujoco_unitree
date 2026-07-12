"""
code/datagen/gen_dart_dataset.py — Fast DART dataset generator (no-render, gait-phase stored).

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

Role: CLI entry point (argparse + subcommand dispatch + the `generate`
orchestration loop). Split out (RF-1) from the physics/data pieces, which
live in sibling modules:
  - code/datagen/gen_dart_phase.py    — GaitPhaseTracker, build_proprio
  - code/datagen/gen_dart_rollout.py  — run_dart_episode
  - code/datagen/gen_dart_combine.py  — add_phase_to_clean_dataset, combine_datasets
GaitPhaseTracker and build_proprio are re-exported here for old-path
compat (code/gen_maneuver_dataset.py imports them from `code.gen_dart_dataset`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pandas as pd

_HERE: Path = Path(__file__).resolve().parent
_REPO: Path = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from code.datagen.gen_dart_combine import add_phase_to_clean_dataset, combine_datasets
from code.datagen.gen_dart_phase import GaitPhaseTracker, build_proprio
from code.datagen.gen_dart_rollout import FPS, PROPRIO_DIM, run_dart_episode
from code.scene import derive_rng, sample_scene
from code.teacher import WBCTeacher

__all__ = [
    "GaitPhaseTracker", "build_proprio", "run_dart_episode",
    "add_phase_to_clean_dataset", "combine_datasets", "main_generate", "main",
]


# ---------------------------------------------------------------------------
# Main DART generation
# ---------------------------------------------------------------------------
def main_generate(args: argparse.Namespace) -> None:
    """Generates a DART dataset and writes it to `args.out`.

    Args:
        args: Parsed command-line arguments (see `main`), with fields
            `difficulty`, `seed`, `num_episodes`, `noise`, `maxsteps`,
            `out`, and `verbose`.
    """
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

    def _stat(a: np.ndarray) -> dict[str, list[float]]:
        """Computes per-dimension mean/std/min/max for stats.json."""
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
def main() -> None:
    """Parses CLI arguments and dispatches to the requested subcommand.

    Subcommands: `generate`, `add-phase`, `combine` (see module docstring).

    Raises:
        SystemExit: If no subcommand is given (prints help and exits 1).
    """
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
